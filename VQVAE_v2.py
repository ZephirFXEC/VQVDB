import argparse
import os
import struct
import time
from pathlib import Path
from typing import List, Optional, Callable, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

LEAF_DIM = 8


class VDBLeafDataset(Dataset):
    def __init__(
            self,
            npy_files: Sequence[str | Path],
            transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
            *,
            include_origins: bool = False,
            origins_root: str | Path | None = None,
            origins_suffix: str = "._origins.npy",
    ) -> None:
        super().__init__()
        self.transform = transform
        self.include_origins = include_origins
        self.buffer = torch.empty(0)  # Pre-allocate later

        # Precompute offsets and mmap files
        self.arrays = []
        self.origin_arrays = [] if include_origins else None
        lengths = []

        for f in npy_files:
            arr = np.load(f, mmap_mode="r")
            if arr.shape[1:] != (LEAF_DIM, LEAF_DIM, LEAF_DIM):
                raise ValueError(f"{f}: invalid shape {arr.shape}")
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

        # Use buffer to avoid allocation
        leaf = torch.from_numpy(
            self.arrays[file_idx][local_idx].astype(np.float32, copy=False)
        ).unsqueeze(0)  # Add channel dim

        if self.transform:
            leaf = self.transform(leaf)

        if self.include_origins:
            origin = torch.from_numpy(
                self.origin_arrays[file_idx][local_idx].astype(np.int32, copy=False)
            )
            return leaf, origin
        return leaf



class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int,
                 commitment_cost: float, decay: float = 0.99, eps: float = 1e-5):
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


class Encoder(nn.Module):
    def __init__(self, in_channels, embedding_dim):
        super(Encoder, self).__init__()
        self.net = nn.Sequential(
            # 8³ → 4³
            nn.Conv3d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),

            # Refine at 4³
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),

            # Final projection
            nn.Conv3d(64, embedding_dim, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, embedding_dim, out_channels):
        super(Decoder, self).__init__()
        self.net = nn.Sequential(
            # Expand from embedding_dim
            nn.Conv3d(embedding_dim, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),

            # 4³ → 8³
            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),

            # Final reconstruction
            nn.Conv3d(32, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class VQVAE(nn.Module):
    def __init__(self, in_channels, embedding_dim, num_embeddings, commitment_cost):
        super(VQVAE, self).__init__()
        self.encoder = Encoder(in_channels, embedding_dim)
        self.quantizer = VectorQuantizerEMA(num_embeddings, embedding_dim, commitment_cost)
        self.decoder = Decoder(embedding_dim, in_channels)

    def forward(self, x):
        z = self.encoder(x)
        quantized, vq_loss, perplexity = self.quantizer(z)
        x_recon = self.decoder(quantized)
        return z, x_recon, vq_loss, perplexity

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

        return indices.view(B, *spatial)

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
            dead_code_threshold = 1.0 # A reasonable threshold
            dead_indices = torch.where(self.quantizer.cluster_size < dead_code_threshold)[0]

            if len(dead_indices) == 0:
                return # Nothing to do

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
            self.quantizer.cluster_size.data[dead_indices] = 1.0 # Reset usage count to 1

    def encoder_outputs_to_flat(self, z):
        """Helper to flatten encoder outputs for processing."""
        D = self.quantizer.embedding_dim
        permute_fwd: List[int] = [0] + list(range(2, z.dim())) + [1]
        permuted_x = torch.permute(z, permute_fwd).contiguous()
        return permuted_x.view(-1, D)


def train(args):
    # Hyperparameters
    BATCH_SIZE = 8192*2
    EPOCHS = 500
    LR = 5e-4
    IN_CHANNELS = 1
    EMBEDDING_DIM = 128  # The dimensionality of the embeddings
    NUM_EMBEDDINGS = 256  # The size of the codebook (the "dictionary")
    COMMITMENT_COST = 0.25
    TRAINING_STEPS_WARMUP = 100  # Steps before dead code reset
    RESET_DEAD_CODES_EVERY_N_STEPS = 50  # Frequency of dead
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    npy_files = list(Path(args.data_dir).glob("*.npy"))
    if not npy_files:
        raise ValueError(f"No .npy files found in /data/npy")

    print(f"Found {len(npy_files)} .npy files")

    vdb_dataset = VDBLeafDataset(npy_files=npy_files, include_origins=False)
    print(f"Dataset created with {len(vdb_dataset)} total blocks.")

    # keep 10% of the dataset for validation
    split_idx = int(len(vdb_dataset) * 0.9)

    vdb_dataset_train = torch.utils.data.Subset(vdb_dataset, range(split_idx))
    vdb_dataset_val = torch.utils.data.Subset(vdb_dataset, range(split_idx, len(vdb_dataset)))
    print(f"Training dataset size: {len(vdb_dataset_train)}")
    print(f"Validation dataset size: {len(vdb_dataset_val)}")

    if not os.path.exists(os.path.dirname(args.model_path)):
        os.makedirs(os.path.dirname(args.model_path))

    train_loader = DataLoader(
        vdb_dataset_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        vdb_dataset_val,
        batch_size=BATCH_SIZE,
        shuffle=False)

    model = VQVAE(IN_CHANNELS, EMBEDDING_DIM, NUM_EMBEDDINGS, COMMITMENT_COST).to(device)
    optimizer = Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    print("Starting training with data from DataLoader...")
    best_val_loss = float('inf')
    global_step = 0  # NEW: Initialize a global step counter
    

    for epoch in range(EPOCHS):
        model.train()
        total_recon_loss = 0.0
        total_vq_loss = 0.0
        last_perplexity = 0.0 # To store the perplexity for logging

        # Use a progress bar for the training loop
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}", leave=False)
        for leaves in pbar:
            leaves = leaves.to(device, non_blocking=True)

            optimizer.zero_grad()

            # MODIFIED: Unpack the new return value 'z'
            z, x_recon, vq_loss, perplexity = model(leaves)
            
            recon_error = F.mse_loss(x_recon, leaves)
            loss = recon_error + vq_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Update running losses and perplexity
            total_recon_loss += recon_error.item()
            total_vq_loss += vq_loss.item()
            last_perplexity = perplexity.item()

            # Update progress bar description
            pbar.set_postfix(
                recon_loss=recon_error.item(),
                vq_loss=vq_loss.item(),
                ppl=last_perplexity
            )
            
            # --- NEW: Periodic Dead Code Reset Logic ---
            if global_step > TRAINING_STEPS_WARMUP and global_step % RESET_DEAD_CODES_EVERY_N_STEPS == 0:
                # We pass `z` to the function. Use .detach() as this is a manual, non-gradient operation.
                model.check_and_reset_dead_codes(z)
            
            global_step += 1 # NEW: Increment the step counter

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for leaves in val_loader:
                leaves = leaves.to(device)
                # We don't need 'z' here, so we can ignore it
                _, x_recon, vq_loss, _ = model(leaves)
                recon_error = F.mse_loss(x_recon, leaves)
                val_loss += recon_error.item() + vq_loss.item()

        avg_train_recon_loss = total_recon_loss / len(train_loader)
        avg_train_vq_loss = total_vq_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), args.model_path)
            print(f"\nBest model saved at epoch {epoch + 1} with validation loss {best_val_loss:.4f}")
            
        print(f"\nEpoch {epoch + 1}/{EPOCHS}, "
            f"Train Recon Loss: {avg_train_recon_loss:.4f}, "
            f"Train VQ Loss: {avg_train_vq_loss:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}, "
            f"Perplexity: {last_perplexity:.2f}")

    print("Training finished.")
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    torch.save(model.state_dict(), args.model_path)
    print("Final Model saved successfully.")

    # Save JIT script for inference
    scripted_model = torch.jit.script(model)
    scripted_model.save(args.model_path.replace('.pth', '_scripted.pt'))
    print(f"Final Scripted model saved to {args.model_path.replace('.pth', '_scripted.pt')}")


# --- 5. Compression Function ---
def compress_vdb(args):
    # --- Validate inputs ---
    for path_attr in ("model_path", "input_vdb", "origins_path"):
        path = getattr(args, path_attr)
        if not os.path.exists(path):
            print(f"Error: '{path_attr}' not found at {path}")
            return

    # --- Load trained VQ-VAE model ---
    model = VQVAE(
        embedding_dim=args.embedding_dim,
        num_embeddings=args.num_embeddings,
        in_channels=1,
        commitment_cost=0.25
    )
    state = torch.load(args.model_path, map_location=args.device)
    model.load_state_dict(state)
    model.to(args.device)
    model.eval()

    # --- Memory-map input arrays ---
    leaves_mmap = np.load(args.input_vdb, mmap_mode='r')  # shape: (N, H, W, D)
    origins_mmap = np.load(args.origins_path, mmap_mode='r')  # shape: (N, ...)
    num_leaves = leaves_mmap.shape[0]
    num_origins = origins_mmap.shape[0]

    # --- Determine index-grid shape via a single forward pass ---
    sample_leaf = leaves_mmap[0:1].astype(np.float32)
    sample_t = torch.from_numpy(sample_leaf).unsqueeze(1).to(args.device)
    with torch.no_grad():
        sample_idx = model.encode(sample_t)
    idx_shape = tuple(sample_idx.shape[1:])  # e.g., (4, 4, 4)

    # --- Prepare chunking parameters ---
    CHUNK = 8192  # Tune to your available memory

    # --- Start writing compressed file ---
    start_time = time.time()
    with open(args.output_cvdb, 'wb') as f:
        # Header
        f.write(b'VQVDB')  # Magic
        f.write(struct.pack('<B', 2))  # Version w/ origins
        f.write(struct.pack('<I', args.num_embeddings))  # Codebook size
        f.write(struct.pack('<B', len(idx_shape)))  # # dims in index grid
        for dim in idx_shape:
            f.write(struct.pack('<H', dim))  # Each dim size

        # Origins block
        f.write(struct.pack('<I', num_origins))  # Count of origins
        # Write origins in chunks to avoid full-array load
        for i in range(0, num_origins, CHUNK):
            end = min(i + CHUNK, num_origins)
            block = origins_mmap[i:end].astype(np.int32, copy=False)
            f.write(block.tobytes())

        # Write indices as uint8
        f.write(struct.pack('<I', num_leaves))
        with torch.no_grad():
            for i in tqdm(range(0, num_leaves, CHUNK)):
                end = min(i + CHUNK, num_leaves)
                # Direct memory mapping without copy
                batch = np.copy(leaves_mmap[i:end])  # Required copy for torch conversion
                tensor = torch.from_numpy(batch).unsqueeze(1).to(args.device)

                indices = model.encode(tensor)
                indices_np = indices.cpu().numpy().astype(np.uint8)  # UINT8 here

                f.write(indices_np.tobytes())

                # Cleanup GPU/CPU memory
                del tensor, indices, indices_np
                if args.device.startswith('cuda'):
                    torch.cuda.empty_cache()

    elapsed = time.time() - start_time
    # --- Final Report ---
    original_size = os.path.getsize(args.input_vdb) + os.path.getsize(args.origins_path)
    compressed_size = os.path.getsize(args.output_cvdb)
    ratio = original_size / compressed_size if compressed_size > 0 else 0
    print(f"\n--- Compression Complete in {elapsed} ---")
    print(f"Original data size: {original_size / (1024 * 1024):.2f} MB")
    print(f"Compressed file size: {compressed_size / (1024 * 1024):.2f} MB")
    print(f"Compression Ratio: {ratio:.2f}x")
    print(f"Saved to {args.output_cvdb}")


# --- 6. Main Argument Parser ---

if __name__ == "__main__":
    
    model = VQVAE(1, 128, 256, 0.25)  # Replace with your model class
    model.load_state_dict(torch.load("models/vqvae.pth", map_location="cuda"))
    
    scripted_model = torch.jit.script(model)
    scripted_model.save("models/vqvae_scripted.pt")
    
    parser = argparse.ArgumentParser(description="VQ-VAE Compressor for OpenVDB files.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- Training Arguments ---
    parser_train = subparsers.add_parser("train", help="Train the VQ-VAE model.")
    parser_train.add_argument("--data_dir", type=str, default="data", help="Directory with .vdb files.")
    parser_train.add_argument("--grid_name", type=str, default="density", help="Name of the grid to extract.")
    parser_train.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser_train.add_argument("--batch_size", type=int, default=4096, help="Training batch size.")
    parser_train.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser_train.add_argument("--num_embeddings", type=int, default=256, help="Size of the codebook.")
    parser_train.add_argument("--embedding_dim", type=int, default=128, help="Dimension of the latent vectors.")
    parser_train.add_argument("--model_path", type=str, default="models/vqvae.pth",
                              help="Path to save the trained model.")
    parser_train.set_defaults(func=train)

    # --- Compression Arguments ---
    parser_compress = subparsers.add_parser("compress", help="Compress a VDB file.")
    parser_compress.add_argument("input_vdb", type=str, help="Path to the input .vdb file.")
    parser_compress.add_argument("origins_path", type=str, help="Path to the origins file.")
    parser_compress.add_argument("output_cvdb", type=str, help="Path for the compressed .cvdb file.")
    parser_compress.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                                 help="Device to run the model on (e.g., 'cuda' or 'cpu').")
    parser_compress.add_argument("--grid_name", type=str, default="density", help="Name of the grid to compress.")
    parser_compress.add_argument("--model_path", type=str, default="models/vqvae.pth",
                                 help="Path to the trained model.")
    parser_compress.add_argument("--num_embeddings", type=int, default=256,
                                 help="Size of the codebook (must match trained model).")
    parser_compress.add_argument("--embedding_dim", type=int, default=128,
                                 help="Dimension of the latent vectors (must match trained model).")
    parser_compress.set_defaults(func=compress_vdb)

    args = parser.parse_args()
    args.func(args)
