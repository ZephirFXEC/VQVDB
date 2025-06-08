import argparse
import os
import struct
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

try:
    # We now import the module from WITHIN the pyVDB package
    from pyVDB import vdb_leaf_extractor
except ImportError as e:
    print("Error: Could not import 'vdb_leaf_extractor' from the 'pyVDB' package.")
    print(f"Original error: {e}")
    print("Please ensure the compiled .so/.pyd file is inside the 'pyVDB' directory.")
    sys.exit(1)

LEAF_DIM = 8


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1 / self.num_embeddings, 1 / self.num_embeddings)

    def forward(self, inputs):
        # Flatten input: [B, C, D, H, W] -> [B*D*H*W, C]
        inputs_flat = inputs.permute(0, 2, 3, 4, 1).contiguous().view(-1, self.embedding_dim)

        # Calculate distances
        distances = (torch.sum(inputs_flat ** 2, dim=1, keepdim=True)
                     + torch.sum(self.embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(inputs_flat, self.embedding.weight.t()))

        # Find closest encodings
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)

        # Get the quantized vectors
        quantized = self.embedding(encoding_indices.view(-1))

        # Reshape back to original input shape
        quantized = quantized.view(inputs.shape)

        # Loss calculation
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight-through estimator
        quantized = inputs + (quantized - inputs).detach()

        return quantized, loss, encoding_indices.squeeze()


class VQVAE(nn.Module):
    """
    The full VQ-VAE model.
    """

    def __init__(self, embedding_dim=64, num_embeddings=512, commitment_cost=0.25):
        super(VQVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, embedding_dim, kernel_size=3, stride=1, padding=1)
        )

        self.vq_layer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)

        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(embedding_dim, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 1, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        z = self.encoder(x)
        quantized, vq_loss, indices = self.vq_layer(z)
        x_recon = self.decoder(quantized)
        return x_recon, vq_loss, indices

    def encode(self, x):
        z = self.encoder(x)
        _, _, indices = self.vq_layer(z)
        return indices

    def get_codebook(self):
        return self.vq_layer.embedding.weight.data


# --- 3. Data Handling ---

class VDBLeafDataset(Dataset):
    def __init__(self, vdb_files, grid_name, cache_file="leaf_cache.npy"):
        self.vdb_files = vdb_files
        self.grid_name = grid_name

        if os.path.exists(cache_file):
            print(f"Loading cached leaves from {cache_file}...")
            self.leaves = np.load(cache_file)
        else:
            print("Extracting leaves from VDB files...")
            all_leaves = []
            for vdb_file in tqdm(vdb_files):
                try:
                    leaves = vdb_leaf_extractor.extract_leaves(str(vdb_file), self.grid_name)
                    all_leaves.extend(leaves)
                except Exception as e:
                    print(f"Warning: Could not process {vdb_file}: {e}")
            self.leaves = np.array(all_leaves, dtype=np.float32)
            print(f"Saving {len(self.leaves)} leaves to cache file {cache_file}")
            np.save(cache_file, self.leaves)

    def __len__(self):
        return len(self.leaves)

    def __getitem__(self, idx):
        # Add a channel dimension for PyTorch Conv3d
        return torch.from_numpy(self.leaves[idx]).unsqueeze(0)


# --- 4. Training Function ---

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    vdb_files = list(Path(args.data_dir).glob("*.vdb"))
    if not vdb_files:
        print(f"No .vdb files found in {args.data_dir}. Please generate some.")
        return

    dataset = VDBLeafDataset(vdb_files, args.grid_name)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = VQVAE(
        embedding_dim=args.embedding_dim,
        num_embeddings=args.num_embeddings
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print("Starting training...")
    for epoch in range(args.epochs):
        total_recon_loss = 0
        total_vq_loss = 0
        for i, data in enumerate(tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}")):
            data = data.to(device)
            optimizer.zero_grad()

            x_recon, vq_loss, _ = model(data)
            recon_loss = F.mse_loss(x_recon, data)
            loss = recon_loss + vq_loss

            loss.backward()
            optimizer.step()

            total_recon_loss += recon_loss.item()
            total_vq_loss += vq_loss.item()

        avg_recon_loss = total_recon_loss / len(dataloader)
        avg_vq_loss = total_vq_loss / len(dataloader)
        print(f"Epoch {epoch + 1}: Recon Loss: {avg_recon_loss:.6f}, VQ Loss: {avg_vq_loss:.6f}")

    print("Training finished.")
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    torch.save(model.state_dict(), args.model_path)
    print(f"Model saved to {args.model_path}")


# --- 5. Compression Function ---

def compress_vdb(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(args.model_path):
        print(f"Model file not found at {args.model_path}. Please train first.")
        return

    model = VQVAE(
        embedding_dim=args.embedding_dim,
        num_embeddings=args.num_embeddings
    )
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    print(f"Loading leaves from {args.input_vdb}...")
    try:
        leaves_np = vdb_leaf_extractor.extract_leaves(args.input_vdb, args.grid_name)
    except Exception as e:
        print(f"Error extracting leaves: {e}")
        return

    leaves_tensor = torch.from_numpy(np.array(leaves_np, dtype=np.float32)).unsqueeze(1).to(device)

    print(f"Found {len(leaves_tensor)} leaves. Compressing...")

    with torch.no_grad():
        indices = model.encode(leaves_tensor).cpu().numpy()

    codebook = model.get_codebook().cpu().numpy()

    # --- .cvdb File Format ---
    # Header:
    #   - Magic Bytes (4 bytes: "CVDB")
    #   - Version (1 byte)
    #   - Num Embeddings (4 bytes, uint32)
    #   - Embedding Dim (4 bytes, uint32)
    #   - Num Indices (4 bytes, uint32)
    # Codebook:
    #   - [Num Embeddings * Embedding Dim] floats (4 bytes each)
    # Indices:
    #   - [Num Indices] shorts (2 bytes each, assuming num_embeddings < 65536)

    with open(args.output_cvdb, 'wb') as f:
        # Header
        f.write(b'CVDB')
        f.write(struct.pack('B', 1))  # Version 1
        f.write(struct.pack('I', codebook.shape[0]))  # num_embeddings
        f.write(struct.pack('I', codebook.shape[1]))  # embedding_dim
        f.write(struct.pack('I', len(indices)))  # num_indices

        # Codebook
        f.write(codebook.tobytes())

        # Indices
        # For simplicity, we use uint16. A more advanced format would use
        # variable bit-length encoding or entropy coding (ANS/Huffman).
        if args.num_embeddings > 65535:
            raise ValueError("Number of embeddings > 65535, change index type from short")
        f.write(indices.astype(np.uint16).tobytes())

    original_size = os.path.getsize(args.input_vdb)
    compressed_size = os.path.getsize(args.output_cvdb)
    ratio = original_size / compressed_size
    print("\n--- Compression Complete ---")
    print(f"Original VDB size: {original_size / 1024:.2f} KB")
    print(f"Compressed CVDB size: {compressed_size / 1024:.2f} KB")
    print(f"Compression Ratio: {ratio:.2f}x")
    print(f"Saved to {args.output_cvdb}")


# --- 6. Main Argument Parser ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VQ-VAE Compressor for OpenVDB files.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- Training Arguments ---
    parser_train = subparsers.add_parser("train", help="Train the VQ-VAE model.")
    parser_train.add_argument("--data_dir", type=str, default="data", help="Directory with .vdb files.")
    parser_train.add_argument("--grid_name", type=str, default="density", help="Name of the grid to extract.")
    parser_train.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser_train.add_argument("--batch_size", type=int, default=4096, help="Training batch size.")
    parser_train.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser_train.add_argument("--num_embeddings", type=int, default=512, help="Size of the codebook.")
    parser_train.add_argument("--embedding_dim", type=int, default=64, help="Dimension of the latent vectors.")
    parser_train.add_argument("--model_path", type=str, default="models/vqvae.pth",
                              help="Path to save the trained model.")
    parser_train.set_defaults(func=train)

    # --- Compression Arguments ---
    parser_compress = subparsers.add_parser("compress", help="Compress a VDB file.")
    parser_compress.add_argument("input_vdb", type=str, help="Path to the input .vdb file.")
    parser_compress.add_argument("output_cvdb", type=str, help="Path for the compressed .cvdb file.")
    parser_compress.add_argument("--grid_name", type=str, default="density", help="Name of the grid to compress.")
    parser_compress.add_argument("--model_path", type=str, default="models/vqvae.pth",
                                 help="Path to the trained model.")
    parser_compress.add_argument("--num_embeddings", type=int, default=1024,
                                 help="Size of the codebook (must match trained model).")
    parser_compress.add_argument("--embedding_dim", type=int, default=64,
                                 help="Dimension of the latent vectors (must match trained model).")
    parser_compress.set_defaults(func=compress_vdb)

    args = parser.parse_args()
    args.func(args)
