//
// Created by zphrfx on 10/06/2025.
//

#pragma once

#include <torch/torch.h>

#include <tuple>
#include <vector>

// --- VectorQuantizerEMA ---
struct VectorQuantizerEMAImpl final : torch::nn::Module {
	float commitment_cost;
	float decay;
	float eps;

	int64_t num_embeddings;
	int64_t embedding_dim;

	torch::Tensor embedding;     // This is a parameter
	torch::Tensor cluster_size;  // These are buffers
	torch::Tensor embed_avg;

	VectorQuantizerEMAImpl(int64_t num_embeddings, int64_t embedding_dim, float commitment_cost, float decay = 0.99, float eps = 1e-5);

	std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(const torch::Tensor& x);
};
TORCH_MODULE(VectorQuantizerEMA);

struct EncoderImpl final : torch::nn::Module {
	torch::nn::Sequential net = nullptr;

	EncoderImpl(int64_t in_channels, int64_t embedding_dim);

	torch::Tensor forward(const torch::Tensor& x) { return net->forward(x); }
};

TORCH_MODULE(Encoder);


struct DecoderImpl final : torch::nn::Module {
	torch::nn::Sequential net = nullptr;

	DecoderImpl(int64_t embedding_dim, int64_t out_channels);

	torch::Tensor forward(const torch::Tensor& x) { return net->forward(x); }
};
TORCH_MODULE(Decoder);


struct VQVAEImpl final : torch::nn::Module {
	Encoder encoder = nullptr;
	VectorQuantizerEMA quantizer = nullptr;
	Decoder decoder = nullptr;

	VQVAEImpl(int64_t in_channels, int64_t embedding_dim, int64_t num_embeddings, float commitment_cost = 0.25) {
		encoder = register_module("encoder", Encoder(in_channels, embedding_dim));
		quantizer = register_module("quantizer", VectorQuantizerEMA(num_embeddings, embedding_dim, commitment_cost));
		decoder = register_module("decoder", Decoder(embedding_dim, in_channels));
	}

	std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> forward(const torch::Tensor& x);
	torch::Tensor encode(const torch::Tensor& x);
	torch::Tensor decode(const torch::Tensor& indices);

	const torch::Tensor& get_codebook() { return quantizer->embedding; }

	// --- Extras to mirror Python API -----------------------------------------
	const torch::Tensor& encoder_outputs_to_flat(const torch::Tensor& z);
	void check_and_reset_dead_codes(const torch::Tensor& encoder_outputs);
};
TORCH_MODULE(VQVAE);