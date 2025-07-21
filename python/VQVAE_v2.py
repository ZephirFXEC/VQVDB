"""
 Copyright (c) 2025, Enzo Crema

 SPDX-License-Identifier: BSD-3-Clause

 See the LICENSE file in the project root for full license text.
"""

from pathlib import Path
from typing import List, Optional, Callable, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

LEAF_DIM = 8


class VDBLeafDataset(Dataset):
    def __init__(
            self,
            npy_files: Sequence[str | Path],
            transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
            *,
            in_channels: int = 1,
            include_origins: bool = False,
            origins_root: str | Path | None = None,
            origins_suffix: str = "._origins.npy",
    ) -> None:
        super().__init__()
        self.transform = transform
        self.include_origins = include_origins
        self.buffer = torch.empty(0)  # Pre-allocate later
        self.in_channels = in_channels

        # Precompute offsets and mmap files
        self.arrays = []
        self.origin_arrays = [] if include_origins else None
        lengths = []

        expected_shape_suffix = (LEAF_DIM, LEAF_DIM, LEAF_DIM)
        if self.in_channels > 1:
            # For multi-channel, expect channels-last format from NumPy
            expected_shape_suffix = (LEAF_DIM, LEAF_DIM, LEAF_DIM, self.in_channels)

        for f in npy_files:
            arr = np.load(f, mmap_mode="r")
            if arr.shape[1:] != expected_shape_suffix:
                raise ValueError(f"File {f}: invalid shape {arr.shape}. Expected suffix {expected_shape_suffix}")
            self.arrays.append(arr)
            lengths.append(arr.shape[0])

            if include_origins:
                origin_path = Path(origins_root or f).with_suffix(origins_suffix)
                if not origin_path.exists():
                    raise FileNotFoundError(origin_path)
                self.origin_arrays.append(np.load(origin_path, mmap_mode="r"))

        self.file_offsets = np.cumsum([0] + lengths)
        self.total_leaves = int(self.file_offsets[-1])

    def __len__(self) -> int:
        return self.total_leaves

    def __getitem__(self, idx: int):
        file_idx = np.searchsorted(self.file_offsets, idx, side="right") - 1
        local_idx = idx - self.file_offsets[file_idx]

        leaf_np = self.arrays[file_idx][local_idx].astype(np.float32, copy=True)
        leaf = torch.from_numpy(leaf_np)

        if self.in_channels == 1:
            leaf = leaf.unsqueeze(0)  # (1, 8, 8, 8)
        else:
            leaf = leaf.permute(3, 0, 1, 2)  # (C, 8, 8, 8)

        if self.transform:
            leaf_norm = self.transform(leaf_norm)

        if self.include_origins:
            origin = torch.from_numpy(
                self.origin_arrays[file_idx][local_idx].astype(np.int32, copy=False))  # type: ignore[index]
            return leaf_norm, origin
        return leaf


class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int,
                 commitment_cost: float, decay: float = 0.95, eps: float = 1e-4):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.eps = eps

        # Initialize embeddings (small variance + normalize)
        embed = torch.randn(num_embeddings, embedding_dim)
        embed = F.normalize(embed, dim=1)

        self.register_buffer('embedding', embed)
        self.register_buffer('cluster_size', torch.ones(num_embeddings))
        self.register_buffer('embed_avg', embed.clone().detach())

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        D = self.embedding_dim

        # Build the permutation list explicitly: [0, 2, 3, …, n, 1]
        permute_fwd: List[int] = [0] + list(range(2, x.dim())) + [1]

        permuted_x = torch.permute(x, permute_fwd).contiguous()
        flat = permuted_x.view(-1, D)

        # Compute distances
        distances = (
                torch.sum(flat ** 2, dim=1, keepdim=True)
                + torch.sum(self.embedding ** 2, dim=1)
                - 2 * torch.mm(flat, self.embedding.t())
        )

        # Get nearest codes
        encoding_indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).type(flat.dtype)

        # Quantize
        quantized = encodings @ self.embedding
        quantized = quantized.view(permuted_x.shape)

        permute_back: List[int] = [0, x.dim() - 1] + list(range(1, x.dim() - 1))
        quantized = torch.permute(quantized, permute_back)

        # EMA updates
        if self.training:
            with torch.no_grad():
                encodings_sum = encodings.sum(0)
                self.cluster_size.mul_(self.decay).add_(encodings_sum, alpha=1 - self.decay)

                dw = encodings.t() @ flat.detach()
                self.embed_avg.mul_(self.decay).add_(dw, alpha=1 - self.decay)

                n = self.cluster_size.clamp(min=self.eps)
                self.embedding.copy_(self.embed_avg / n.unsqueeze(1))

        commitment_loss = self.commitment_cost * F.mse_loss(x, quantized.detach())
        loss = commitment_loss

        # Straight-through estimator
        quantized = x + (quantized - x).detach()

        # Perplexity
        avg_probs = encodings.mean(0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized, loss, perplexity


def icnr_(tensor, upscale_factor=2, initializer=nn.init.kaiming_normal_):
    """ICNR initialisation for 3-D conv/transpose-conv weights."""
    out_channels, in_channels, k, k2, k3 = tensor.shape
    sub = out_channels // (upscale_factor ** 3)
    if sub == 0:
        raise ValueError("ICNR: out_channels too small.")
    temp = torch.zeros([sub, in_channels, k, k2, k3])
    initializer(temp)
    temp = temp.repeat_interleave(upscale_factor ** 3, 0)
    with torch.no_grad():
        tensor.copy_(temp)


class PixelShuffle3D(nn.Module):
    def __init__(self, upscale_factor: int = 2):
        super().__init__()
        self.r = upscale_factor

    def forward(self, x):
        b, c, d, h, w = x.size()
        r = self.r
        oc = c // (r * r * r)

        if c % (r * r * r) != 0:
            raise RuntimeError("Channels not divisible by r^3.")

        x = x.view(b, oc, r, r, r, d, h, w)
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4)
        return x.contiguous().view(b, oc, d * r, h * r, w * r)


class ResidualBlock(nn.Module):
    """Pre-activation, GN-only residual block with residual scaling."""

    def __init__(self, channels, groups=8, scale=0.1):
        super().__init__()
        self.scale = scale
        self.gn1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv3d(channels, channels, 3, 1, 1, bias=True)
        self.gn2 = nn.GroupNorm(groups, channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(channels, channels, 3, 1, 1, bias=True)
        nn.init.normal_(self.conv2.weight, mean=0, std=1e-3)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x):
        residual = x
        x = self.relu(self.gn1(x))
        x = self.conv1(x)
        x = self.relu(self.gn2(x))
        x = self.conv2(x)
        return residual + self.scale * x


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)


class EncoderFloat(nn.Module):
    def __init__(self, in_channels: int, embedding_dim: int):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Conv3d(in_channels, 16, 3, 1, 1, bias=True),
            nn.GroupNorm(4, 16),
            nn.ReLU(inplace=True),
            ResidualBlock(16),
        )
        self.down = nn.Conv3d(16, 32, 4, 2, 1, bias=True)
        self.res_stack = nn.Sequential(ResidualBlock(32))
        self.attn = ChannelAttention(32)
        self.proj = nn.Conv3d(32, embedding_dim, 1)

    def forward(self, x):
        x = self.pre(x)
        x = self.down(x)
        x = self.res_stack(x)
        x = self.attn(x)
        return self.proj(x)


class DecoderFloat(nn.Module):
    def __init__(self, embedding_dim: int, out_channels: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(embedding_dim, 64, 3, 1, 1, bias=True),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
        )
        self.res_stack = nn.Sequential(ResidualBlock(64))
        self.attn = ChannelAttention(64)

        # upsample 4³ → 8³
        self.up_conv = nn.Conv3d(64, 32 * 8, 3, 1, 1, bias=True)
        icnr_(self.up_conv.weight)
        self.pixshuf = PixelShuffle3D(2)
        self.final = nn.Conv3d(32, out_channels, 3, 1, 1, bias=True)

    def forward(self, x):
        x = self.stem(x)
        x = self.res_stack(x)
        x = self.attn(x)
        x = self.pixshuf(self.up_conv(x))
        return torch.sigmoid(self.final(x))


class EncoderVec3(nn.Module):
    def __init__(self, in_channels: int, embedding_dim: int = 64):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Conv3d(in_channels, 64, 3, 1, 1, bias=True),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            ResidualBlock(64),
        )
        self.down1 = nn.Conv3d(64, 128, 3, 2, 1, bias=True)
        self.res_stack = nn.Sequential(
            ResidualBlock(128), ResidualBlock(128)
        )
        self.attn = ChannelAttention(128)
        self.proj = nn.Conv3d(128, embedding_dim, 1)

    def forward(self, x):
        x = self.pre(x)
        x = self.down1(x)
        x = self.res_stack(x)
        x = self.attn(x)
        return self.proj(x)


class DecoderVec3(nn.Module):
    def __init__(self, embedding_dim, out_channels):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(embedding_dim, 128, 3, 1, 1, bias=True),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True)
        )
        self.res_stack = nn.Sequential(
            ResidualBlock(128), ResidualBlock(128)
        )
        self.attn = ChannelAttention(128)

        self.up_conv = nn.Conv3d(128, 32 * 8, 3, 1, 1, bias=True)
        icnr_(self.up_conv.weight)
        self.pixshuf = PixelShuffle3D(2)
        self.final = nn.Conv3d(32, out_channels, 3, 1, 1, bias=True)

    def forward(self, x):
        x = self.stem(x)
        x = self.res_stack(x)
        x = self.attn(x)
        x = self.pixshuf(self.up_conv(x))
        return torch.tanh(self.final(x))


class VQVAE(nn.Module):
    def __init__(self, in_channels, embedding_dim, num_embeddings, commitment_cost):
        super(VQVAE, self).__init__()
        if in_channels == 1:  # float
            self.encoder = EncoderFloat(in_channels, embedding_dim)
            self.decoder = DecoderFloat(embedding_dim, in_channels)
        else:  # vec3f
            self.encoder = EncoderVec3(in_channels, embedding_dim)
            self.decoder = DecoderVec3(embedding_dim, in_channels)

        self.quantizer = VectorQuantizerEMA(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            commitment_cost=commitment_cost,
        )

    def forward(self, x):
        z = self.encoder(x)
        quantized, vq_loss, perplexity = self.quantizer(z)
        x_recon = self.decoder(quantized)
        return z, x_recon, vq_loss, perplexity

    @torch.jit.export
    def encode(self, x) -> torch.Tensor:
        z = self.encoder(x)

        shape = list(z.shape)  # List[int]
        B, D = shape[0], shape[1]
        spatial = shape[2:]

        permute_fwd = [0] + list(range(2, z.dim())) + [1]
        flat_z = (torch.permute(z, permute_fwd)
                  .contiguous()
                  .view(-1, D)
                  )

        distances = (torch.sum(flat_z ** 2, dim=1, keepdim=True)
                     + torch.sum(self.quantizer.embedding ** 2, dim=1)
                     - 2 * torch.matmul(flat_z, self.quantizer.embedding.t()))
        indices = torch.argmin(distances, dim=1)

        return indices.view([B] + spatial)

    @torch.jit.export
    def decode(self, indices):
        quantized_vectors = F.embedding(indices, self.quantizer.embedding)
        permute_back = [0, quantized_vectors.dim() - 1] + list(range(1, quantized_vectors.dim() - 1))
        quantized_for_decoder = torch.permute(quantized_vectors, permute_back)
        x_recon = self.decoder(quantized_for_decoder)
        return x_recon

    def get_codebook(self) -> torch.Tensor:
        return self.quantizer.embedding

    def check_and_reset_dead_codes(self, encoder_outputs):
        """
        Checks for and resets dead codes in the quantizer's codebook by
        resampling them from the given batch of encoder outputs.
        """
        # Ensure you are on the correct device
        device = self.quantizer.embedding.device

        # Use .detach() to ensure this operation is not part of the autograd graph
        flat_z = self.encoder_outputs_to_flat(encoder_outputs.detach())

        with torch.no_grad():
            # 1. Identify dead codes
            dead_code_threshold = 1.0  # A reasonable threshold
            dead_indices = torch.where(self.quantizer.cluster_size < dead_code_threshold)[0]

            if len(dead_indices) == 0:
                return  # Nothing to do

            print(f"INFO: Resetting {len(dead_indices)} dead codes.")

            # 2. Resample from the current batch of encoder outputs
            num_dead = len(dead_indices)
            num_active_vectors = flat_z.shape[0]

            if num_active_vectors == 0:
                print("WARNING: Cannot reset dead codes, encoder output batch is empty.")
                return

            sample_indices = torch.randint(0, num_active_vectors, (num_dead,), device=device)
            new_embeddings = flat_z[sample_indices]

            # 3. Assign the new embeddings and reset their stats
            self.quantizer.embedding.data[dead_indices] = new_embeddings
            self.quantizer.embed_avg.data[dead_indices] = new_embeddings
            self.quantizer.cluster_size.data[dead_indices] = 1.0  # Reset usage count to 1

    def encoder_outputs_to_flat(self, z):
        """Helper to flatten encoder outputs for processing."""
        D = self.quantizer.embedding_dim
        permute_fwd: List[int] = [0] + list(range(2, z.dim())) + [1]
        permuted_x = torch.permute(z, permute_fwd).contiguous()
        return permuted_x.view(-1, D)
