from VQVAE_v2 import *

from torch.optim.lr_scheduler import CosineAnnealingLR

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
    vdb_dataset = torch.utils.data.Subset(vdb_dataset, range(0, len(vdb_dataset), 4))  # Subsample to reduce dataset size
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
        
    optimizer = Adam(model.parameters(), lr=LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS * len(train_loader))

    print("Starting training with data from DataLoader...")
    best_val_loss = float('inf')


    recon_loss_l = []  # List to store reconstruction losses
    vq_loss_l = []  # List to store VQ losses
    perplexity_l = []  # List to store perplexity values

    for epoch in range(EPOCHS):
        model.train()
        total_recon_loss = 0.0
        total_vq_loss = 0.0
        last_perplexity = 0.0 # To store the perplexity for logging

        # Use a progress bar for the training loop
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}", leave=False)
        for batch in pbar:
            leaves_norm = batch
            leaves_norm = leaves_norm.to(device, non_blocking=True)

            optimizer.zero_grad()

            z, recon_norm, vq_loss, perplexity = model(leaves_norm)
            recon_error = F.mse_loss(recon_norm, leaves_norm)
            loss = recon_error + vq_loss

            loss.backward()
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
            
        scheduler.step()
            
        if epoch == 5 or epoch == 10 or epoch == 15 or epoch == 20:
            model.check_and_reset_dead_codes(z)


        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                leaves_norm = batch
                leaves_norm = leaves_norm.to(device)
                _, recon_norm, vq_loss, _ = model(leaves_norm)
                recon_error = F.mse_loss(recon_norm, leaves_norm)
                val_loss += recon_error.item() + vq_loss.item()
                
        avg_train_recon = total_recon_loss / len(train_loader)
        avg_train_vq = total_vq_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            state = {'epoch': epoch + 1, 'state_dict': model.state_dict(),
                     'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), "recon_loss_l": recon_loss_l,
                     "vq_loss_l": vq_loss_l, "perplexity_l": perplexity_l}
            print(f"New best validation loss: {avg_val_loss:.6f}, saving model...")
            os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
            torch.save(state, args.model_path)

        print(f"\nEpoch {epoch + 1}/{EPOCHS}, "
              f"Train Recon Loss: {avg_train_recon:.6f}, "
              f"Train VQ Loss: {avg_train_vq:.6f}, "
              f"Val Loss: {avg_val_loss:.6f}, "
              f"Perplexity: {last_perplexity:.6f}")

        recon_loss_l.append(avg_train_recon)
        vq_loss_l.append(avg_train_vq)
        perplexity_l.append(last_perplexity)

    state = {'epoch': epoch + 1, 'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), "recon_loss_l": recon_loss_l,
             "vq_loss_l": vq_loss_l, "perplexity_l": perplexity_l}
    
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    torch.save(state, args.model_path)


    print("Final Model saved successfully.")
    # Save JIT script for inference
    scripted_model = torch.jit.script(model)
    scripted_model.save(args.model_path.replace('.pth', '_scripted.pt'))
    print(f"Final Scripted model saved to {args.model_path.replace('.pth', '_scripted.pt')}")

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
