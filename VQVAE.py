import argparse
import os
import struct
import time
from pathlib import Path
from typing import List, Optional, Callable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class VDBLeafDataset(Dataset):
    """
    Efficient dataset for OpenVDB-leaf tensors stored one file per grid.

    *   Each ``.npy`` file is **memory-mapped once** and kept for the
        lifetime of the process – no repeated `np.load` in ``__getitem__``.
    *   Global-index → (file, local-index) lookup is done in vectorised C
        via `np.searchsorted`.
    *   Supports an optional companion ``_origins.npy`` file **per leaf file**
        that stores the integer (x, y, z) origins for later embedding.
    """

    def __init__(
            self,
            npy_files: Sequence[str | Path],
            transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
            *,
            include_origins: bool = False,
            origins_root: str | Path | None = None,  # <<< NEW
            origins_suffix: str = "._origins.npy",  # <<< configurable
    ) -> None:
        super().__init__()

        self.transform = transform
        self.include_origins = include_origins

        # --- mmap all data files -------------------------------------------------
        self.arrays: List[np.memmap] = []
        self.origin_arrays: List[np.memmap] | None = [] if include_origins else None

        for f in npy_files:
            arr = np.load(f, mmap_mode="r")
            if arr.shape[1:] != (32, 32, 32):
                raise ValueError(f"{f}: expected (N, 32, 32, 32), got {arr.shape}")
            self.arrays.append(arr)

            if include_origins:
                # Let the caller override where origins live -----------------
                if origins_root is not None:
                    origin_path = Path(origins_root) / (Path(f).stem + origins_suffix)
                else:
                    origin_path = Path(f).with_suffix(origins_suffix)
                if not origin_path.exists():
                    raise FileNotFoundError(origin_path)

                self.origin_arrays.append(np.load(origin_path, mmap_mode="r"))

        # --- pre-compute global index mapping ------------------------------------
        lengths = np.fromiter((a.shape[0] for a in self.arrays), dtype=np.int64)
        self.file_offsets = np.concatenate(([0], np.cumsum(lengths)))
        self.total_leaves: int = int(self.file_offsets[-1])

    # ---------------------------------------------------------------------------

    def __len__(self) -> int:
        return self.total_leaves

    def __getitem__(self, idx: int):
        if not (0 <= idx < self.total_leaves):
            raise IndexError(idx)

        # locate (file, local) in O(log n) inside highly-optimised C code
        file_idx = int(np.searchsorted(self.file_offsets, idx, side="right") - 1)
        local_idx = idx - int(self.file_offsets[file_idx])

        # zero-copy view from the mmap’d array
        leaf_np = self.arrays[file_idx][local_idx].astype(np.float32, copy=True)
        leaf = torch.from_numpy(leaf_np).to(torch.float32).unsqueeze(0)

        if self.transform is not None:
            leaf = self.transform(leaf)

        if self.include_origins:
            origin_np = self.origin_arrays[file_idx][local_idx].astype(np.int32, copy=False)
            origin = torch.from_numpy(origin_np)
            return leaf, origin

        return leaf


class ResidualBlock(nn.Module):
    def __init__(self, channels): 
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(channels)
        self.bn2 = nn.BatchNorm3d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        # Initialize embeddings
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, inputs):
        # Convert inputs from BCHWD -> BHWDC
        inputs = inputs.permute(0, 2, 3, 4, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self.embedding_dim)

        # Calculate L2 distance between inputs and embedding vectors
        distances = torch.sum(flat_input ** 2, dim=1, keepdim=True) + \
                    torch.sum(self.embedding.weight ** 2, dim=1) - \
                    2 * torch.matmul(flat_input, self.embedding.weight.t())

        # Get encoding indices
        encoding_indices = torch.argmin(distances, dim=1)

        # Quantize and unflatten
        quantized = self.embedding(encoding_indices).view(input_shape)

        # Compute loss for embedding
        q_latent_loss = F.mse_loss(quantized.detach(), inputs)
        e_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight-through estimator
        quantized = inputs + (quantized - inputs).detach()

        # Convert back to BCHWD
        quantized = quantized.permute(0, 4, 1, 2, 3).contiguous()

        return quantized, loss, encoding_indices.view(input_shape[:-1])


class VQVAE(nn.Module):
    def __init__(self, num_embeddings=512, embedding_dim=64, hidden_dims=128, commitment_cost=0.25):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(1, hidden_dims // 4, kernel_size=4, stride=2, padding=1),  # 32x32x32 -> 16x16x16
            nn.ReLU(),
            nn.Conv3d(hidden_dims // 4, hidden_dims // 2, kernel_size=4, stride=2, padding=1),  # 16x16x16 -> 8x8x8
            nn.ReLU(),
            nn.Conv3d(hidden_dims // 2, hidden_dims, kernel_size=4, stride=2, padding=1),  # 8x8x8 -> 4x4x4
            nn.ReLU(),
            ResidualBlock(hidden_dims),
            ResidualBlock(hidden_dims),
            nn.Conv3d(hidden_dims, embedding_dim, kernel_size=1)  # Projection to embedding dimension
        )

        # Vector Quantizer
        self.vq = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)

        # Decoder
        # Decoder for reconstructing 32x32x32 blocks from 4x4x4 latent grid
        self.decoder = nn.Sequential(
            nn.Conv3d(embedding_dim, hidden_dims, kernel_size=1),
            ResidualBlock(hidden_dims),
            ResidualBlock(hidden_dims),
            nn.ConvTranspose3d(hidden_dims, hidden_dims // 2, kernel_size=4, stride=2, padding=1),  # 4x4x4 -> 8x8x8
            nn.ReLU(),
            nn.ConvTranspose3d(hidden_dims // 2, hidden_dims // 4, kernel_size=4, stride=2, padding=1),
            # 8x8x8 -> 16x16x16
            nn.ReLU(),
            nn.ConvTranspose3d(hidden_dims // 4, 1, kernel_size=4, stride=2, padding=1),  # 16x16x16 -> 32x32x32
        )

    def encode(self, x):
        z = self.encoder(x)
        z_q, _, indices = self.vq(z)
        return z_q, indices

    @torch.jit.export
    def decode(self, z_q):
        return self.decoder(z_q)

    def forward(self, x):
        z = self.encoder(x)
        z_q, latent_loss, indices = self.vq(z)
        x_recon = self.decoder(z_q)

        return x_recon, latent_loss, indices

    def encode_indices(self, x):
        z = self.encoder(x)
        _, _, indices = self.vq(z)
        return indices


def train_model(model, train_loader, val_loader, args):
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    device = torch.device(args.device)
    model = model.to(device)

    if args.device == 'cuda':
        torch.backends.cudnn.benchmark = True

    scaler = torch.amp.GradScaler(enabled=(args.device == 'cuda'))

    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        # Training
        model.train()
        train_loss = 0
        recon_loss = 0
        latent_loss = 0

        start_time = time.time()
        for batch in train_loader:
            optimizer.zero_grad()

            x = batch.to(device, non_blocking=True)

            with torch.amp.autocast(enabled=(args.device == 'cuda'), device_type='cuda'):
                x_recon, vq_loss, _ = model(x)
                recon_error = F.mse_loss(x_recon, x)
                loss = recon_error + vq_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            recon_loss += recon_error.item()
            latent_loss += vq_loss.item()

        train_loss /= len(train_loader)
        recon_loss /= len(train_loader)
        latent_loss /= len(train_loader)
        epoch_time = time.time() - start_time

        # Validation
        model.eval()
        val_loss = 0
        val_recon_loss = 0
        val_latent_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                
                x = batch.to(device, non_blocking=True)
                with torch.amp.autocast(enabled=(args.device == 'cuda'), device_type='cuda'):
                    x_recon, vq_loss, _ = model(x)
                    recon_error = F.mse_loss(x_recon, x)
                    loss = recon_error + vq_loss
                    
                val_loss += loss.item()
                val_recon_loss += recon_error.item()
                val_latent_loss += vq_loss.item()

        val_loss /= len(val_loader)
        val_recon_loss /= len(val_loader)
        val_latent_loss /= len(val_loader)

        # Update learning rate
        scheduler.step(val_loss)

        print(f"Epoch {epoch + 1}/{args.epochs}: "
              f"Train Loss: {train_loss:.6f} (Recon: {recon_loss:.6f}, VQ: {latent_loss:.6f}) | "
              f"Val Loss: {val_loss:.6f} (Recon: {val_recon_loss:.6f}, VQ: {val_latent_loss:.6f}) | "
              f"Time: {epoch_time:.2f}s")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pt"))

        # Save checkpoint every few epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': train_loss,
            }, os.path.join(args.output_dir, f"checkpoint_epoch_{epoch + 1}.pt"))

    # Save final model
    torch.save(model.state_dict(), os.path.join(args.output_dir, "final_model.pt"))

    # Export TorchScript model
    model.eval()
    scripted_model = torch.jit.script(model)
    scripted_model.save(os.path.join(args.output_dir, "vqvae_model.pt"))

    return model


def evaluate_model(model, test_loader, args):
    device = torch.device(args.device)
    model = model.to(device)
    model.eval()

    total_mse = 0
    total_psnr = 0
    count = 0

    with torch.no_grad():
        for batch in test_loader:
            x = (batch[0] if isinstance(batch, (list, tuple)) else batch).to(device, non_blocking=True)
            x_recon, _, _ = model(x)

            mse = F.mse_loss(x_recon, x, reduction='none').mean((1, 2, 3, 4))
            psnr = 20 * torch.log10(torch.tensor(1.0, device=device)) - 10 * torch.log10(mse)

            total_mse += mse.sum().item()
            total_psnr += psnr.sum().item()
            count += x.size(0)

    avg_mse = total_mse / count
    avg_psnr = total_psnr / count

    print(f"Test MSE: {avg_mse:.6f}")
    print(f"Test PSNR: {avg_psnr:.2f} dB")

    return avg_mse, avg_psnr


def compress_dataset(model, data_loader, output_file, args, origins=None):
    """Compress a dataset and save indices to binary file.
    If ``origins`` is provided, it should be a NumPy array of shape ``(N,3)``
    with the leaf coordinates matching the order of the dataset. These
    positions will be embedded in the resulting ``.vqvdb`` file.
    """
    device = torch.device(args.device)
    model = model.to(device)
    model.eval()

    # Get expected index shape from a batch
    sample_batch = next(iter(data_loader))
    if isinstance(sample_batch, (list, tuple)):
        sample_input = sample_batch[0]
    else:
        sample_input = sample_batch
    with torch.no_grad():
        _, _, indices = model(sample_input.to(device))
    index_shape = indices.shape[1:]  # Shape without batch dimension

    # Calculate bits needed to represent each index
    bits_per_index = (args.num_embeddings - 1).bit_length()
    bytes_per_index = (bits_per_index + 7) // 8  # Round up to nearest byte

    with open(output_file, 'wb') as f:
        # Write header: magic number, version, num_embeddings, shape dimensions
        version = 2 if origins is not None else 1
        f.write(b'VQVDB')
        f.write(struct.pack('<B', version))
        f.write(struct.pack('<I', args.num_embeddings))
        f.write(struct.pack('<B', len(index_shape)))
        for dim in index_shape:
            f.write(struct.pack('<H', dim))

        # If we have origins, store them before the index batches
        if origins is not None:
            f.write(struct.pack('<I', origins.shape[0]))
            f.write(origins.astype(np.int32).tobytes())

        # Process all batches
        with torch.no_grad():
            for batch in data_loader:
                if isinstance(batch, (list, tuple)):
                    x = batch[0].to(device)
                else:
                    x = batch.to(device)
                _, _, indices = model(x)

                # Save indices (simple method - could be improved with entropy coding)
                flat_indices = indices.cpu().numpy().reshape(-1)

                # Write batch size
                f.write(struct.pack('<I', indices.shape[0]))

                # Naive packing - one index per byte/short
                if bytes_per_index == 1:
                    f.write(flat_indices.astype(np.uint8).tobytes())
                else:
                    f.write(flat_indices.astype(np.uint16).tobytes())

    print(f"Compressed indices saved to {output_file}")

    # Calculate and print compression ratio
    num_voxels = np.prod([32, 32, 32])  # Each leaf is 32x32x32
    original_bytes = num_voxels * 4  # 4 bytes per float32 voxel
    compressed_bytes = np.prod(index_shape) * bytes_per_index
    compression_ratio = original_bytes / compressed_bytes

    print(f"Compression ratio: {compression_ratio:.2f}x")
    print(f"Original: {original_bytes} bytes per leaf, Compressed: {compressed_bytes} bytes per leaf")

    return compression_ratio


def concatenate_origins(file_list, origins_dir):
    """Concatenate per-grid origins so order matches the dataset."""
    # --- FIX ---
    # If the input list is empty, return an empty numpy array of the correct shape.
    if not file_list:
        # Origins are (N, 3) for (x, y, z), so the shape is (0, 3).
        return np.empty((0, 3), dtype=np.int32)
    # -----------

    arrays = []
    for f in file_list:
        path = Path(origins_dir) / (Path(f).stem + "._origins.npy")
        if not path.exists():
            # Make it more robust by warning the user if a file is missing
            print(f"Warning: Origin file not found and will be skipped: {path}")
            continue
        arrays.append(np.load(path))

    # Also handle the case where files existed but couldn't be loaded
    if not arrays:
        return np.empty((0, 3), dtype=np.int32)

    return np.concatenate(arrays, axis=0).astype(np.int32)

def main():
    parser = argparse.ArgumentParser(description="Train VQ-VAE for OpenVDB leaf compression")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing .npy files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--num_embeddings", type=int, default=512, help="Size of codebook")
    parser.add_argument("--embedding_dim", type=int, default=64, help="Dimension of codebook vectors")
    parser.add_argument("--hidden_dims", type=int, default=128, help="Hidden dimensions in model")
    parser.add_argument("--commitment_cost", type=float, default=0.25, help="Commitment cost for VQ")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--test_split", type=float, default=0.1, help="Test split ratio")
    parser.add_argument("--origins_dir", type=str, default=None,
                        help="Optional NPY file containing leaf origins to embed in the .vqvdb output")
    parser.add_argument("--compress_file", type=str, default=None,
                        help="Path to a single .npy grid you want to compress "
                             "(overrides the normal test-set compression)")
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Get all .npy files
    npy_files = list(Path(args.input_dir).glob("*.npy"))
    if not npy_files:
        raise ValueError(f"No .npy files found in {args.input_dir}")

    print(f"Found {len(npy_files)} .npy files")

    # Split into train/val/test
    np.random.shuffle(npy_files)
    test_size = int(len(npy_files) * args.test_split)
    val_size = int(len(npy_files) * args.val_split)
    train_size = len(npy_files) - test_size - val_size

    train_files = npy_files[:train_size]
    val_files = npy_files[train_size:train_size + val_size]
    test_files = npy_files[train_size + val_size:]

    print(f"Train files: {len(train_files)}, Val files: {len(val_files)}, Test files: {len(test_files)}")

    train_dataset = VDBLeafDataset(
        train_files,
        include_origins=False,  # <<<
    )

    val_dataset = VDBLeafDataset(
        val_files,
        include_origins=False,  # <<<
    )

    test_dataset = VDBLeafDataset(
        test_files,
        include_origins=False,  # keep origins for later embedding
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=6, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    if args.origins_dir:
        origins = concatenate_origins(test_files, args.origins_dir)
    else:
        origins = None

    # Create model
    model = VQVAE(
        num_embeddings=args.num_embeddings,
        embedding_dim=args.embedding_dim,
        hidden_dims=args.hidden_dims,
        commitment_cost=args.commitment_cost
    )

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {total_params:,} parameters")
    print(
        f"Codebook size: {args.num_embeddings} x {args.embedding_dim} = {args.num_embeddings * args.embedding_dim:,} parameters")

    # Train model
    # print("Starting training...")
    # model = train_model(model, train_loader, val_loader, args)

    # Load best model if available
    #
    best_model_path = os.path.join(args.output_dir, "best_model.pt")
    if os.path.exists(best_model_path):
        print(f"Loading best model from {best_model_path}")
        model.load_state_dict(torch.load(best_model_path))

    scripted_model = torch.jit.script(model)
    scripted_model.save(os.path.join(args.output_dir, "vqvae_model.pt"))

    # Evaluate model
    # print("Evaluating model...")
    #mse, psnr = evaluate_model(model, test_loader, args)

    # Compress test dataset
    # Compress test dataset
    print("Compressing test dataset...")

    compressed_file = os.path.join(args.output_dir, "compressed_indices.vqvdb")

    if args.compress_file is not None:
        # ──►  ONE-FILE MODE  ◄────────────────────────────────────
        grid_path = Path(args.compress_file).resolve()
        if not grid_path.exists():
            raise FileNotFoundError(grid_path)

        # Build a tiny dataset / loader that yields just this file
        single_dataset = VDBLeafDataset(
            [grid_path],
            include_origins=False,
        )
        single_loader = DataLoader(single_dataset, batch_size=args.batch_size,
                                   shuffle=False, num_workers=0)

        # Find and load the matching origin file, if any
        if args.origins_dir:
            origin_path = Path(args.origins_dir) / (grid_path.stem + "._origins.npy")
            if origin_path.exists():
                single_origin = np.load(origin_path).astype(np.int32)

                # Compress just this grid
                compression_ratio = compress_dataset(model, single_loader,
                                                     compressed_file, args,
                                                     origins=single_origin)
    else:
        # ──►  NORMAL TEST-SET MODE  ◄─────────────────────────────
        compression_ratio = compress_dataset(model, test_loader,
                                             compressed_file, args,
                                             origins=origins)  # built earlier

    # Save metrics
    with open(os.path.join(args.output_dir, "metrics.txt"), "w") as f:
        f.write(f"MSE: {0:.6f}\n")
        f.write(f"PSNR: {0:.2f} dB\n")
        f.write(f"Compression ratio: {compression_ratio:.2f}x\n")
        f.write(f"Codebook size: {args.num_embeddings} entries, {args.embedding_dim} dimensions\n")

    print("Done!")


if __name__ == "__main__":
    main()
