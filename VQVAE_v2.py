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
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from tqdm import tqdm

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

        # --- mmap all data files -------------------------------------------------
        self.arrays: List[np.memmap] = []
        self.origin_arrays: List[np.memmap] | None = [] if include_origins else None

        for f in npy_files:
            arr = np.load(f, mmap_mode="r")
            if arr.shape[1:] != (LEAF_DIM, LEAF_DIM, LEAF_DIM):
                raise ValueError(f"{f}: expected (N, {LEAF_DIM}, {LEAF_DIM}, {LEAF_DIM}), got {arr.shape}")
            self.arrays.append(arr)

            if include_origins:
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




class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int,
                 commitment_cost: float, decay: float = 0.99, eps: float = 1e-5):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.eps = eps

        # Initialize embeddings
        embed = torch.randn(num_embeddings, embedding_dim) * 0.1  # Small variance
        embed = F.normalize(embed, dim=1)  # Normalize initial embeddings
        
        self.register_buffer('embedding', embed)
        self.register_buffer('cluster_size', torch.ones(num_embeddings))
        self.register_buffer('embed_avg', embed.clone().detach())

    def forward(self, x)  -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        D = self.embedding_dim

        # Build the permutation list explicitly: [0, 2, 3, …, n, 1]
        permute_fwd: List[int] = [0] + list(range(2, x.dim())) + [1]
        
        # --- CHANGE 1: Store the permuted tensor ---
        # This tensor has the shape [B, ...spatial, D] which we'll need later.
        permuted_x = torch.permute(x, permute_fwd).contiguous()
        flat = permuted_x.view(-1, D)

        # Compute distances
        distances = torch.cdist(flat, self.embedding, p=2.0)

        # Get nearest codes
        encoding_indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).type(flat.dtype)

        # Quantize
        quantized = encodings @ self.embedding
        
        # --- CHANGE 2: Reshape using the saved tensor's shape ---
        # This is the key fix. It's JIT-friendly because the rank is known.
        quantized = quantized.view(permuted_x.shape)
        
        # --- CHANGE 3: Fix the back permutation using x.dim() ---
        # The rank of x determines the length of the permutation list.
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
        self.quantizer = VectorQuantizerEMA(num_embeddings, embedding_dim, commitment_cost=0.25)
        self.decoder = Decoder(embedding_dim, in_channels)

    def forward(self, x):
        z = self.encoder(x)
        quantized, vq_loss, perplexity = self.quantizer(z)
        x_recon = self.decoder(quantized)
        return x_recon, vq_loss, perplexity

    def encode(self, x) -> torch.Tensor:
        z = self.encoder(x)
        
        shape = list(z.shape)         # List[int]
        B, D = shape[0], shape[1]
        spatial = shape[2:]

        permute_fwd = [0] + list(range(2, z.dim())) + [1]
        flat_z = (torch.permute(z, permute_fwd)
                  .contiguous()
                  .view(-1, D)
                  )
    
        distances = (torch.sum(flat_z**2, dim=1, keepdim=True) 
                     + torch.sum(self.quantizer.embedding**2, dim=1)
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
        

def train(args):
    
        # Hyperparameters
    BATCH_SIZE = 8192
    EPOCHS = 100
    LR = 5e-4
    IN_CHANNELS = 1
    EMBEDDING_DIM = 128 # The dimensionality of the embeddings
    NUM_EMBEDDINGS = 256 # The size of the codebook (the "dictionary")
    COMMITMENT_COST = 0.25

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    npy_files = list(Path(args.data_dir).glob("*.npy"))
    if not npy_files:
        raise ValueError(f"No .npy files found in /data/npy")

    print(f"Found {len(npy_files)} .npy files")

    vdb_dataset = VDBLeafDataset(npy_files=npy_files, include_origins=False)
    print(f"Dataset created with {len(vdb_dataset)} total blocks.")

    # keep 10% of the dataset for validation
    split_idx = int(len(vdb_dataset) * 0.5)
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
        num_workers=10,
        pin_memory=True,
        persistent_workers=True,
    )
    
    
    model = VQVAE(IN_CHANNELS, EMBEDDING_DIM, NUM_EMBEDDINGS, COMMITMENT_COST).to(device)
    optimizer = Adam(model.parameters(), lr=LR)

    print("Starting training with data from DataLoader...")
    for epoch in range(EPOCHS):
        
        total_recon_loss = 0.0
        total_vq_loss = 0
        
        for i, data_batch in enumerate(train_loader):
            leaves = data_batch.to(device, non_blocking=True)
            
            x_recon, vq_loss, perplexity = model(leaves)
            recon_error = F.mse_loss(x_recon, leaves)
            loss = recon_error + vq_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_recon_loss += recon_error.item()
            total_vq_loss += vq_loss.item()

        # Log progress at the end of each epoch
        avg_recon_loss = total_recon_loss / len(train_loader)
        avg_vq_loss = total_vq_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] | "
                f"Avg Recon Loss: {avg_recon_loss:.5f} | "
                f"Avg VQ Loss: {avg_vq_loss:.5f} | "
                f"Last Perplexity: {perplexity.item():.4f}") # Perplexity from last batch


    print("Training finished.")
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    torch.save(model.state_dict(), args.model_path)
    print("Model saved successfully.")

    # Save JIT script for inference
    scripted_model = torch.jit.script(model)
    scripted_model.save(args.model_path.replace('.pth', '_scripted.pt'))
    print(f"Scripted model saved to {args.model_path.replace('.pth', '_scripted.pt')}")


# --- 5. Compression Function ---

def compress_vdb(args):
    """
    Compresses a single pair of leaf/origin .npy files into the .vqvdb 
    format compatible with the C++ batch-based loader.
    """
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        return
    if not os.path.exists(args.input_vdb):
        print(f"Error: Input VDB file not found at {args.input_vdb}")
        return
    if not os.path.exists(args.origins_path):
        print(f"Error: Origins file not found at {args.origins_path}")
        return

    model = VQVAE(
        embedding_dim=args.embedding_dim,
        num_embeddings=args.num_embeddings, 
        in_channels=1,
        commitment_cost=0.25
    )
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    model.to(args.device)
    model.eval()

    print(f"Loading leaves from {args.input_vdb}...")
    try:
        # Use mmap_mode for memory efficiency with large files
        leaves_np = np.load(args.input_vdb, mmap_mode='r')
        if leaves_np.ndim != 4 or leaves_np.shape[1:] != (LEAF_DIM, LEAF_DIM, LEAF_DIM):
            raise ValueError(f"Expected leaf shape (N, {LEAF_DIM}, {LEAF_DIM}, {LEAF_DIM}), got {leaves_np.shape}")
    except Exception as e:
        print(f"Error loading leaves: {e}")
        return

    print(f"Loading origins from {args.origins_path}...")
    try:
        origins_np = np.load(args.origins_path, mmap_mode='r')
        if origins_np.ndim != 2 or origins_np.shape[1] != 3:
            raise ValueError(f"Expected origins shape (N, 3), got {origins_np.shape}")
        origins_np = origins_np.astype(np.int32)
    except Exception as e:
        print(f"Error loading origins: {e}")
        return

    leaves_tensor = torch.from_numpy(np.array(leaves_np)).unsqueeze(1).to(args.device)
    num_leaves = len(leaves_tensor)
    
    print(f"Found {num_leaves} leaves. Compressing all at once...")
    start_time = time.time()

    with torch.no_grad():
        indices_tensor = model.encode(leaves_tensor)

    indices_np = indices_tensor.cpu().numpy()
    index_shape = indices_np.shape[1:]  # Shape without the batch dimension, e.g., (4, 4, 4)
    
    print(f"Compression completed in {time.time() - start_time:.2f} seconds.")
    print(f"Writing to {args.output_cvdb} in a compatible format...")

    with open(args.output_cvdb, 'wb') as f:
        # --- Write Header (matching C++ loader) ---
        f.write(b'VQVDB')
        f.write(struct.pack('<B', 2))  # Version 2, because we are including origins
        f.write(struct.pack('<I', args.num_embeddings))
        f.write(struct.pack('<B', len(index_shape)))  # Number of dimensions in the index grid
        for dim in index_shape:
            f.write(struct.pack('<H', dim))  # Write each dimension's size as a short

        # --- Write Origins Block (for Version 2) ---
        f.write(struct.pack('<I', len(origins_np)))  # Total number of origins
        f.write(origins_np.tobytes())

        # --- Write Data as a SINGLE BATCH ---
        # 1. Write the "batch size", which is the total number of leaves.
        # The C++ loader will read this and expect this many items.
        f.write(struct.pack('<I', num_leaves))

        # 2. Write all the index data that corresponds to this "batch".
        # The C++ loader expects uint16, so we must use that type.
        if args.num_embeddings > 65535:
           raise ValueError("num_embeddings > 65535 is not supported by the uint16 C++ loader.")
        f.write(indices_np.astype(np.uint16).tobytes())

    # --- Final Report ---
    original_size = os.path.getsize(args.input_vdb) + os.path.getsize(args.origins_path)
    compressed_size = os.path.getsize(args.output_cvdb)
    ratio = original_size / compressed_size if compressed_size > 0 else 0
    print("\n--- Compression Complete ---")
    print(f"Original data size: {original_size / (1024*1024):.2f} MB")
    print(f"Compressed file size: {compressed_size / (1024*1024):.2f} MB")
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
    parser_train.add_argument("--epochs", type=int, default=40, help="Number of training epochs.")
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
