"""
 Copyright (c) 2025, Enzo Crema

 SPDX-License-Identifier: BSD-3-Clause

 See the LICENSE file in the project root for full license text.
"""

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from VQVAE_v2 import *


def create_3d_sobel_kernels(device):
    # Sobel kernels for x, y, z directions
    sobel_x = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                            [[-2, 0, 2], [-4, 0, 4], [-2, 0, 2]],
                            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]).float().to(device).unsqueeze(0).unsqueeze(0)

    sobel_y = torch.tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                            [[-2, -4, -2], [0, 0, 0], [2, 4, 2]],
                            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]).float().to(device).unsqueeze(0).unsqueeze(0)

    sobel_z = torch.tensor([[[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]],
                            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                            [[1, 2, 1], [2, 4, 2], [1, 2, 1]]]).float().to(device).unsqueeze(0).unsqueeze(0)

    return sobel_x, sobel_y, sobel_z


# Then in loss computation:
def compute_gradient_loss(recon, target, sobel_x, sobel_y, sobel_z):
    grad_true_x = F.conv3d(target, sobel_x, padding=1)
    grad_true_y = F.conv3d(target, sobel_y, padding=1)
    grad_true_z = F.conv3d(target, sobel_z, padding=1)

    grad_recon_x = F.conv3d(recon, sobel_x, padding=1)
    grad_recon_y = F.conv3d(recon, sobel_y, padding=1)
    grad_recon_z = F.conv3d(recon, sobel_z, padding=1)

    return (F.mse_loss(grad_recon_x, grad_true_x) +
            F.mse_loss(grad_recon_y, grad_true_y) +
            F.mse_loss(grad_recon_z, grad_true_z)) / 3


def train(args):
    # Hyperparameters
    BATCH_SIZE = 2048
    EPOCHS = 30
    LR = 1e-4
    IN_CHANNELS = 1
    EMBEDDING_DIM = 128  # The dimensionality of the embeddings
    NUM_EMBEDDINGS = 256  # The size of the codebook (the "dictionary")
    COMMITMENT_COST = 0.25

    device = torch.device("cuda")
    print(f"Using device: {device}")

    npy_files = list(Path(args.data_dir).glob("*.npy"))
    if not npy_files:
        raise ValueError(f"No .npy files found in /data/npy")

    print(f"Found {len(npy_files)} .npy files")

    vdb_dataset = VDBLeafDataset(npy_files=npy_files, include_origins=False, in_channels=IN_CHANNELS)
    vdb_dataset = torch.utils.data.Subset(vdb_dataset,
                                          range(0, len(vdb_dataset), 6))  # Subsample to reduce dataset size
    print(f"Dataset created with {len(vdb_dataset)} total blocks.")

    # Split dataset randomly with 90% for training and 10% for validation
    train_size = int(0.8 * len(vdb_dataset))
    val_size = len(vdb_dataset) - train_size
    vdb_dataset_train, vdb_dataset_val = torch.utils.data.random_split(
        vdb_dataset, [train_size, val_size]
    )
    print(f"Training dataset size: {len(vdb_dataset_train)}")
    print(f"Validation dataset size: {len(vdb_dataset_val)}")

    if not os.path.exists(os.path.dirname(args.model_path)):
        os.makedirs(os.path.dirname(args.model_path))

    train_loader = DataLoader(
        vdb_dataset_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        vdb_dataset_val,
        batch_size=BATCH_SIZE,
        shuffle=False)

    model = VQVAE(IN_CHANNELS, EMBEDDING_DIM, NUM_EMBEDDINGS, COMMITMENT_COST).to(device)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4, betas=(0.9, 0.999))

    # Better scheduler with warmup
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS * len(train_loader))

    # Mixed precision training
    scaler = torch.GradScaler()

    print("Starting training with data from DataLoader...")
    best_val_loss = float('inf')

    # Better 3D Sobel kernels
    sobel_x, sobel_y, sobel_z = create_3d_sobel_kernels(device)

    # Actually use these lists
    recon_loss_l = []
    vq_loss_l = []
    perplexity_l = []

    # Dead code reset counter
    dead_code_reset_interval = 5  # Reset every 5 epochs instead of every epoch

    for epoch in range(EPOCHS):
        model.train()
        total_recon_loss = 0.0
        total_vq_loss = 0.0
        last_perplexity = 0.0

        # Store encoder outputs for dead code reset (outside autocast)
        encoder_outputs_for_reset = None

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}", leave=False)

        for batch_idx, batch in enumerate(pbar):
            leaves_norm = batch.to(device, non_blocking=True)

            optimizer.zero_grad()

            # Mixed precision forward pass
            with torch.amp.autocast("cuda"):
                z, recon_norm, vq_loss, perplexity = model(leaves_norm)

                # Enhanced loss computation
                recon_mse = F.mse_loss(recon_norm, leaves_norm)
                recon_l1 = F.l1_loss(recon_norm, leaves_norm)

                # Adaptive loss weighting
                mse_weight = 0.8
                l1_weight = 0.2

                recon_error = mse_weight * recon_mse + l1_weight * recon_l1
                loss = recon_error + vq_loss

            # Store encoder outputs for dead code reset (outside autocast to avoid dtype issues)
            if batch_idx == 0:  # Store from first batch for dead code reset
                encoder_outputs_for_reset = z.detach().float()  # Convert to float32

            # Mixed precision backward pass
            scaler.scale(loss).backward()

            # Gradient clipping
            scaler.unscale_(optimizer)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # Update running losses
            total_recon_loss += recon_error.item()
            total_vq_loss += vq_loss.item()
            last_perplexity = perplexity.item()

            # Update progress bar
            pbar.set_postfix(
                recon_loss=recon_error.item(),
                vq_loss=vq_loss.item(),
                ppl=last_perplexity,
                lr=scheduler.get_last_lr()[0]
            )

        # Dead code reset less frequently and with proper dtype handling
        if (epoch + 1) % dead_code_reset_interval == 0 and encoder_outputs_for_reset is not None:
            model.check_and_reset_dead_codes(encoder_outputs_for_reset)

        # Validation phase
        model.eval()
        val_recon_loss = 0.0
        val_vq_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                leaves_norm = batch.to(device)

                with torch.amp.autocast("cuda"):
                    _, recon_norm, vq_loss, _ = model(leaves_norm)

                    # Same loss computation for validation
                    recon_mse = F.mse_loss(recon_norm, leaves_norm)
                    recon_l1 = F.l1_loss(recon_norm, leaves_norm)

                    val_recon_loss += recon_error.item()
                    val_vq_loss += vq_loss.item()

        # Calculate averages
        avg_train_recon = total_recon_loss / len(train_loader)
        avg_train_vq = total_vq_loss / len(train_loader)

        avg_val_recon = val_recon_loss / len(val_loader)
        avg_val_vq = val_vq_loss / len(val_loader)
        avg_val_loss = avg_val_recon + avg_val_vq

        # Store in tracking lists
        recon_loss_l.append(avg_train_recon)
        vq_loss_l.append(avg_train_vq)
        perplexity_l.append(last_perplexity)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

            checkpoint = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "recon_loss_l": recon_loss_l,
                "vq_loss_l": vq_loss_l,
                "perplexity_l": perplexity_l,
                "best_val_loss": best_val_loss,
            }

            os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
            torch.save(checkpoint, args.model_path)
            print(f"New best validation loss: {avg_val_loss:.6f} – model saved.")

        # Enhanced epoch summary
        print(
            f"\nEpoch {epoch + 1:02d}/{EPOCHS} | "
            f"Train Recon: {avg_train_recon:.6f} | "
            f"Train VQ: {avg_train_vq:.6f} | "
            f"Val Loss: {avg_val_loss:.6f} | "
            f"Perplexity: {last_perplexity:.2f} | "
            f"LR: {scheduler.get_last_lr()[0]:.2e}"
        )

        # Early stopping (optional)
        if epoch > 10 and last_perplexity < 5.0:
            print("Warning: Very low perplexity detected. Consider increasing commitment cost.")

    print("Training completed!")

    # Save final model
    torch.save(model.state_dict(), args.model_path.replace('.pth', '_final.pth'))

    # Save JIT script for inference
    model.eval()
    scripted_model = torch.jit.script(model)
    scripted_model.save(args.model_path.replace('.pth', '_scripted.pt'))
    print(f"Scripted model saved to {args.model_path.replace('.pth', '_scripted.pt')}")


if __name__ == "__main__":
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

    args = parser.parse_args()
    args.func(args)
