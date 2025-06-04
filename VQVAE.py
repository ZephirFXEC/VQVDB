import argparse
import os
import struct
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class VDBLeafDataset(Dataset):
    def __init__(self, npy_files, transform=None, include_origins=False):
        """
        Dataset for loading multiple .npy files containing VDB leaf data

        Args:
            npy_files: List of .npy file paths
            transform: Optional transform to apply to samples
        """
        self.npy_files = npy_files
        self.transform = transform
        self.include_origins = include_origins
        if include_origins:
            self.origin_files = [str(Path(f).with_suffix("._origins.npy")) for f in npy_files]

        # Load and process the shape information from all files
        self.file_offsets = [0]
        self.total_leaves = 0

        for npy_file in npy_files:
            data = np.load(npy_file)
            if len(data.shape) != 4 or data.shape[1:] != (8, 8, 8):
                raise ValueError(f"File {npy_file} has incorrect shape. Expected (N, 8, 8, 8), got {data.shape}")
            self.total_leaves += data.shape[0]
            self.file_offsets.append(self.total_leaves)

    def __len__(self):
        return self.total_leaves

    def __getitem__(self, idx):
        # Find which file this index belongs to
        file_idx = next(i for i, offset in enumerate(self.file_offsets) if offset > idx) - 1
        local_idx = idx - self.file_offsets[file_idx]

        # Load data (lazy loading)
        data = np.load(self.npy_files[file_idx])
        leaf_data = data[local_idx].astype(np.float32)

        origin = None
        if self.include_origins:
            origin_data = np.load(self.origin_files[file_idx])
            origin = origin_data[local_idx].astype(np.int32)

        # Convert to tensor
        leaf_tensor = torch.from_numpy(leaf_data)

        # Apply any transforms
        if self.transform:
            leaf_tensor = self.transform(leaf_tensor)

        if self.include_origins:
            return leaf_tensor.unsqueeze(0), origin
        return leaf_tensor.unsqueeze(0)  # Add channel dimension: [1, 8, 8, 8]


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
            nn.Conv3d(1, hidden_dims // 2, kernel_size=4, stride=2, padding=1),  # 8x8x8 -> 4x4x4
            nn.ReLU(),
            nn.Conv3d(hidden_dims // 2, hidden_dims, kernel_size=4, stride=2, padding=1),  # 4x4x4 -> 2x2x2
            nn.ReLU(),
            ResidualBlock(hidden_dims),
            nn.Conv3d(hidden_dims, embedding_dim, kernel_size=1)  # Projection to embedding dimension
        )

        # Vector Quantizer
        self.vq = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv3d(embedding_dim, hidden_dims, kernel_size=1),
            ResidualBlock(hidden_dims),
            nn.ConvTranspose3d(hidden_dims, hidden_dims // 2, kernel_size=4, stride=2, padding=1),  # 2x2x2 -> 4x4x4
            nn.ReLU(),
            nn.ConvTranspose3d(hidden_dims // 2, 1, kernel_size=4, stride=2, padding=1),  # 4x4x4 -> 8x8x8
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

            x = batch.to(device)
            x_recon, vq_loss, _ = model(x)

            recon_error = F.mse_loss(x_recon, x)
            loss = recon_error + vq_loss

            loss.backward()
            optimizer.step()

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
                x = batch.to(device)
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
            x = batch.to(device)
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
    num_voxels = np.prod([8, 8, 8])  # Each leaf is 8x8x8
    original_bytes = num_voxels * 4  # 4 bytes per float32 voxel
    compressed_bytes = np.prod(index_shape) * bytes_per_index
    compression_ratio = original_bytes / compressed_bytes

    print(f"Compression ratio: {compression_ratio:.2f}x")
    print(f"Original: {original_bytes} bytes per leaf, Compressed: {compressed_bytes} bytes per leaf")

    return compression_ratio


def main():
    parser = argparse.ArgumentParser(description="Train VQ-VAE for OpenVDB leaf compression")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing .npy files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--num_embeddings", type=int, default=256, help="Size of codebook")
    parser.add_argument("--embedding_dim", type=int, default=32, help="Dimension of codebook vectors")
    parser.add_argument("--hidden_dims", type=int, default=128, help="Hidden dimensions in model")
    parser.add_argument("--commitment_cost", type=float, default=0.25, help="Commitment cost for VQ")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--test_split", type=float, default=0.1, help="Test split ratio")
    parser.add_argument("--origins_file", type=str, default=None,
                        help="Optional NPY file containing leaf origins to embed in the .vqvdb output")
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

    # Create datasets and dataloaders
    train_dataset = VDBLeafDataset(train_files)
    val_dataset = VDBLeafDataset(val_files)
    test_dataset = VDBLeafDataset(test_files, include_origins=args.origins_file is not None)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

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
    print("Starting training...")
    model = train_model(model, train_loader, val_loader, args)

    # Evaluate model
    print("Evaluating model...")
    mse, psnr = evaluate_model(model, test_loader, args)

    # Compress test dataset
    print("Compressing test dataset...")
    compressed_file = os.path.join(args.output_dir, "compressed_indices.vqvdb")
    origins = np.load(args.origins_file) if args.origins_file else None
    compression_ratio = compress_dataset(model, test_loader, compressed_file, args, origins)

    # Save metrics
    with open(os.path.join(args.output_dir, "metrics.txt"), "w") as f:
        f.write(f"MSE: {mse:.6f}\n")
        f.write(f"PSNR: {psnr:.2f} dB\n")
        f.write(f"Compression ratio: {compression_ratio:.2f}x\n")
        f.write(f"Codebook size: {args.num_embeddings} entries, {args.embedding_dim} dimensions\n")

    print("Done!")


if __name__ == "__main__":
    main()
