import torch
import torch.nn as nn
import torch.nn.functional as F

from VQVAE_v2 import EncoderFloat, DecoderFloat


class InferenceEncoder(nn.Module):
    """A lean encoder for inference. Structurally identical to the training Encoder."""

    def __init__(self, in_channels: int, embedding_dim: int):
        super().__init__()
        # The structure MUST be identical to the training Encoder to load weights.
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, embedding_dim, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class InferenceDecoder(nn.Module):
    """A lean decoder for inference."""

    def __init__(self, embedding_dim: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(embedding_dim, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class InferenceVectorQuantizer(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        # The `embedding` buffer is the only state we need from the trained quantizer.
        # It's initialized empty and will be loaded from the checkpoint.
        self.register_buffer('embedding', torch.empty(num_embeddings, embedding_dim))

    def get_indices(self, flat_x: torch.Tensor) -> torch.Tensor:
        distances = (
                torch.sum(flat_x ** 2, dim=1, keepdim=True)
                + torch.sum(self.embedding ** 2, dim=1)
                - 2 * torch.matmul(flat_x, self.embedding.t())
        )
        return torch.argmin(distances, dim=1)

    def get_quantized(self, indices: torch.Tensor) -> torch.Tensor:
        return F.embedding(indices, self.embedding)


class InferenceVQVAE(nn.Module):
    def __init__(self, in_channels: int, embedding_dim: int, num_embeddings: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.encoder = EncoderFloat(in_channels, embedding_dim)
        self.quantizer = InferenceVectorQuantizer(num_embeddings, embedding_dim)
        self.decoder = DecoderFloat(embedding_dim, in_channels)

    @torch.jit.export
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)  # Shape: [B, embedding_dim, 4, 4, 4]

        # Permute channels to be the last dimension: [B, 4, 4, 4, embedding_dim]
        z_permuted = z.permute(0, 2, 3, 4, 1)

        # Flatten to [B*4*4*4, embedding_dim]
        flat_z = z_permuted.contiguous().view(-1, self.embedding_dim)

        indices = self.quantizer.get_indices(flat_z)

        # Reshape indices back to the spatial shape [B, 4, 4, 4]
        return indices.view(z.shape[0], 4, 4, 4)

    @torch.jit.export
    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        # Flatten indices to [B*4*4*4]
        flat_indices = indices.view(-1)
        quantized_vectors = self.quantizer.get_quantized(flat_indices)

        # Reshape to [B, 4, 4, 4, C]
        batch_size = indices.shape[0]
        unflattened = quantized_vectors.view(batch_size, 4, 4, 4, self.embedding_dim)

        # Permute back to the shape the decoder expects: [B, C, 4, 4, 4]
        quantized_for_decoder = unflattened.permute(0, 4, 1, 2, 3)

        x_recon = self.decoder(quantized_for_decoder)
        return x_recon


IN_CHANNELS = 1
EMBEDDING_DIM = 128
NUM_EMBEDDINGS = 256
COMMITMENT_COST = 0.25
torch.backends.cudnn.benchmark = True


def main():
    # Load the state dict from your trained model.
    trained_state_dict = torch.load('models/scalar/vqvae_128_256_singlechannel_residual.pth', map_location='cpu')

    # Instantiate the NEW, inference-only model.
    print("Instantiating the lean InferenceVQVAE model...")
    inference_model = InferenceVQVAE(IN_CHANNELS, EMBEDDING_DIM, NUM_EMBEDDINGS)

    # Load the weights. This works because the module/parameter names are compatible.
    # - encoder.net.* matches InferenceEncoder.net.*
    # - quantizer.embedding matches InferenceVectorQuantizer.embedding
    inference_model.load_state_dict(trained_state_dict, strict=False)
    print("Successfully loaded weights into the inference model.")

    # CRITICAL: Prepare the model for inference and JIT scripting.
    inference_model.to("cuda")
    inference_model.eval()

    # Script the fully configured instance.
    print("\nScripting the model for maximum performance...")
    scripted_inference_model = torch.jit.script(inference_model)
    print("Scripting complete!")

    # Save the final, optimized artifact for C++ deployment.
    output_path = "inference_vqvae_optimized.pt"
    scripted_inference_model.save(output_path)
    print(f"Saved optimized inference model to '{output_path}'")

    # Test the scripted model
    print("\nTesting the scripted model...")
    with torch.no_grad():
        dummy_input = torch.randn(4, IN_CHANNELS, 8, 8, 8).to("cuda")

        indices = scripted_inference_model.encode(dummy_input)
        reconstruction = scripted_inference_model.decode(indices)

        print(f"Input shape: {dummy_input.shape}")
        print(f"Encoded indices shape: {indices.shape}")
        print(f"Reconstructed output shape: {reconstruction.shape}")
        print("Inference test successful!")


if __name__ == '__main__':
    main()
