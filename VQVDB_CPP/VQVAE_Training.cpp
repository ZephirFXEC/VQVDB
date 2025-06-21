//
// Created by zphrfx on 10/06/2025.
//

#include <torch/script.h>  // One-stop header for TorchScript

#include <iostream>

#include "VQVAE_v2.hpp"

int main() {
	// --- Model Configuration ---
	int64_t in_channels = 1;
	int64_t embedding_dim = 128;
	int64_t num_embeddings = 256;
	float commitment_cost = 0.25;

	// --- Device ---
	torch::Device device = torch::kCPU;
	if (torch::cuda::is_available()) {
		std::cout << "CUDA is available! Using GPU." << std::endl;
		device = torch::kCUDA;
	}

	// --- Create the VQ-VAE model ---
	VQVAE model(in_channels, embedding_dim, num_embeddings, commitment_cost);
	model->to(device);
	std::cout << "Model created successfully." << std::endl;

	// --- Create a dummy input tensor ---
	// Shape: [Batch, Channels, Depth, Height, Width]
	const at::Tensor input = torch::randn({2, in_channels, 8, 8, 8}, device);

	// --- Forward Pass ---
	std::cout << "\n--- Testing Forward Pass ---" << std::endl;
	model->train();  // Set model to training mode to test EMA updates
	auto [z, recon, loss, perplexity] = model->forward(input);

	std::cout << "Input shape: " << input.sizes() << std::endl;
	std::cout << "Latent z shape: " << z.sizes() << std::endl;
	std::cout << "Reconstruction shape: " << recon.sizes() << std::endl;
	std::cout << "VQ Loss: " << loss.item<float>() << std::endl;
	std::cout << "Perplexity: " << perplexity.item<float>() << std::endl;

	model->eval();
	// --- Encode/Decode ---
	std::cout << "\n--- Testing Encode/Decode ---" << std::endl;

	// Encode the input into discrete indices
	const torch::Tensor indices = model->encode(input);
	std::cout << "Encoded indices shape: " << indices.sizes() << std::endl;

	// Decode the indices back into the original space
	const torch::Tensor decoded_from_indices = model->decode(indices);
	std::cout << "Decoded from indices shape: " << decoded_from_indices.sizes() << std::endl;

	// --- Get Codebook ---
	const torch::Tensor codebook = model->get_codebook();
	std::cout << "\nCodebook shape: " << codebook.sizes() << std::endl;


	// --- JIT Scripting and Saving ---
	try {
		std::cout << "\n--- Scripting and Saving Model ---" << std::endl;

		// ---------------------- THE FIX ----------------------
		// 1. Explicitly script your C++ module into a torch::jit::Module.
		//    This traces the `forward` method and converts its logic into TorchScript.
		auto scripted_model = torch::jit::script(model);

		// 2. Save the *scripted* module. This saves both the code and the weights.
		scripted_model.save("vq_vae_scripted.pt");
		// -----------------------------------------------------

		std::cout << "Model scripted and saved to vq_vae_scripted.pt" << std::endl;

		// --- Loading the Scripted Model ---
		// This part will now work correctly because the file contains the 'forward' method.
		torch::jit::Module loaded_model = torch::jit::load("vq_vae_scripted.pt");
		std::vector<torch::jit::IValue> inputs;
		inputs.emplace_back(input);

		// The forward method now exists on the loaded module
		const auto output_tuple = loaded_model.forward(inputs).toTuple();
		const torch::Tensor loaded_recon = output_tuple->elements()[1].toTensor();

		std::cout << "Loaded model reconstruction shape: " << loaded_recon.sizes() << std::endl;
		std::cout << "Test successful." << std::endl;

	} catch (const c10::Error& e) {
		std::cerr << "Error during JIT scripting/loading: " << e.what() << std::endl;
		return -1;
	}

	return 0;
}