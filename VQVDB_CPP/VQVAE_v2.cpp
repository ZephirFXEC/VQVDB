//
// Created by zphrfx on 10/06/2025.
//

#include "VQVAE_v2.hpp"

VectorQuantizerEMAImpl::VectorQuantizerEMAImpl(const int64_t num_embeddings, const int64_t embedding_dim, const float commitment_cost,
                                               const float decay, const float eps)
    : commitment_cost(commitment_cost), decay(decay), eps(eps), num_embeddings(num_embeddings), embedding_dim(embedding_dim) {
	embedding = register_parameter("embedding", torch::empty({num_embeddings, embedding_dim}));
	torch::nn::init::uniform_(embedding, -1.0f / num_embeddings, 1.0f / num_embeddings);
	cluster_size = register_buffer("cluster_size", torch::zeros({num_embeddings}));
	embed_avg = register_buffer("embed_avg", embedding.clone());
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> VectorQuantizerEMAImpl::forward(const torch::Tensor& x) {
	torch::IntArrayRef x_shape = x.sizes();
	const int64_t dim = x.dim();

	// Permute and flatten the input tensor
	std::vector<int64_t> permute_fwd;
	permute_fwd.push_back(0);                                    // Batch dim
	for (int64_t i = 2; i < dim; ++i) permute_fwd.push_back(i);  // Spatial dims
	permute_fwd.push_back(1);                                    // Channel dim

	const torch::Tensor permuted_x = x.permute(permute_fwd).contiguous();
	const torch::Tensor flat = permuted_x.view({-1, this->embedding_dim});

	// Calculate L2 distances between input vectors and embedding vectors
	const at::Tensor distances = torch::sum(flat.pow(2), /*dim=*/1, /*keepdim=*/true) + torch::sum(embedding.pow(2), /*dim=*/1) -
	                             2 * torch::matmul(flat, embedding.t());

	// Get nearest codes
	const at::Tensor encoding_indices = torch::argmin(distances, /*dim=*/1);
	const at::Tensor encodings = torch::one_hot(encoding_indices, this->num_embeddings).to(flat.dtype());

	// Quantize
	at::Tensor quantized = torch::matmul(encodings, embedding);

	// Reshape using the saved tensor's shape
	quantized = quantized.view(permuted_x.sizes());

	// Fix the back permutation using x.dim()
	std::vector<int64_t> permute_back;
	permute_back.push_back(0);                                        // Batch dim
	permute_back.push_back(dim - 1);                                  // Channel dim
	for (int64_t i = 1; i < dim - 1; ++i) permute_back.push_back(i);  // Spatial dims
	quantized = quantized.permute(permute_back).contiguous();

	// EMA updates
	if (this->is_training()) {
		torch::NoGradGuard guard;  // Equivalent to `with torch.no_grad():`

		const at::Tensor encodings_sum = encodings.sum(0);
		cluster_size.mul_(this->decay).add_(encodings_sum, 1 - this->decay);

		const at::Tensor dw = torch::matmul(encodings.t(), flat.detach());
		embed_avg.mul_(this->decay).add_(dw, 1 - this->decay);

		const at::Tensor n = cluster_size.clamp_min(this->eps);
		// Use .data() to get the underlying tensor for in-place copy
		embedding.data().copy_(embed_avg / n.unsqueeze(1));
	}

	// Loss calculation
	const at::Tensor commitment_loss = this->commitment_cost * torch::nn::functional::mse_loss(x, quantized.detach());
	at::Tensor loss = commitment_loss;

	// Straight-through estimator
	quantized = x + (quantized - x).detach();

	// Perplexity
	const at::Tensor avg_probs = encodings.mean(0);
	at::Tensor perplexity = torch::exp(-torch::sum(avg_probs * torch::log(avg_probs + 1e-10)));

	return std::make_tuple(quantized, loss, perplexity);
}

DecoderImpl::DecoderImpl(const int64_t embedding_dim, const int64_t out_channels) {
	net = torch::nn::Sequential(
	    // Expand from embedding_dim
	    torch::nn::Conv3d(torch::nn::Conv3dOptions(embedding_dim, 64, 3).stride(1).padding(1)), torch::nn::BatchNorm3d(64),
	    torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),

	    // 4³ → 8³
	    torch::nn::ConvTranspose3d(torch::nn::ConvTranspose3dOptions(64, 32, 4).stride(2).padding(1)), torch::nn::BatchNorm3d(32),
	    torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),

	    // Final reconstruction
	    torch::nn::Conv3d(torch::nn::Conv3dOptions(32, out_channels, 3).stride(1).padding(1)), torch::nn::Sigmoid());
	register_module("net", net);
}


EncoderImpl::EncoderImpl(const int64_t in_channels, const int64_t embedding_dim) {
	net = torch::nn::Sequential(
	    // 8³ → 4³
	    torch::nn::Conv3d(torch::nn::Conv3dOptions(in_channels, 32, 4).stride(2).padding(1)), torch::nn::BatchNorm3d(32),
	    torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),

	    // Refine at 4³
	    torch::nn::Conv3d(torch::nn::Conv3dOptions(32, 64, 3).stride(1).padding(1)), torch::nn::BatchNorm3d(64),
	    torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),

	    // Final projection
	    torch::nn::Conv3d(torch::nn::Conv3dOptions(64, embedding_dim, 3).stride(1).padding(1)));
	register_module("net", net);
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> VQVAEImpl::forward(const torch::Tensor& x) {
	const torch::Tensor z = encoder->forward(x);
	auto [quantized, vq_loss, perplexity] = quantizer->forward(z);
	torch::Tensor x_recon = decoder->forward(quantized);
	return std::make_tuple(x_recon, vq_loss, perplexity);
}


torch::Tensor VQVAEImpl::encode(const torch::Tensor& x) {
	const torch::Tensor z = encoder->forward(x);
	const std::vector<long long> z_shape = z.sizes().vec();
	const int64_t B = z_shape[0];
	int64_t D = z_shape[1];

	std::vector<int64_t> permute_fwd;
	permute_fwd.push_back(0);
	for (int64_t i = 2; i < z.dim(); ++i) permute_fwd.push_back(i);
	permute_fwd.push_back(1);

	const at::Tensor flat_z = z.permute(permute_fwd).contiguous().view({-1, D});

	const torch::Tensor q_embedding = quantizer->embedding;
	const at::Tensor distances = (torch::sum(flat_z.pow(2), /*dim=*/1, /*keepdim=*/true) + torch::sum(q_embedding.pow(2), /*dim=*/1) -
	                              2 * torch::matmul(flat_z, q_embedding.t()));

	const at::Tensor indices = torch::argmin(distances, /*dim=*/1);

	std::vector<int64_t> final_shape = {B};
	for (size_t i = 2; i < z_shape.size(); ++i) {
		final_shape.push_back(z_shape[i]);
	}

	return indices.view(final_shape);
}


torch::Tensor VQVAEImpl::decode(const torch::Tensor& indices) {

    auto q_embedding = quantizer->embedding;
	auto original_indices_shape = indices.sizes().vec();
	auto flat_indices = indices.view({-1});

	// 2. Use index_select to gather the vectors.
	// This selects rows (dim=0) from q_embedding based on the values in flat_indices.
	auto selected_vectors = torch::index_select(q_embedding, 0, flat_indices);

	// 3. Reshape the result back to the expected output shape.
	// The target shape is the original index shape with the embedding_dim appended.
	std::vector<int64_t> target_shape = original_indices_shape;
	target_shape.push_back(q_embedding.size(1)); // Append the embedding dimension

	auto quantized_vectors = selected_vectors.view(target_shape);
	const int64_t dim = quantized_vectors.dim();
	std::vector<int64_t> permute_back;
	permute_back.push_back(0);
	permute_back.push_back(dim - 1);
	for (int64_t i = 1; i < dim - 1; ++i) permute_back.push_back(i);

	const at::Tensor quantized_for_decoder = quantized_vectors.permute(permute_back).contiguous();
	auto x_recon = decoder->forward(quantized_for_decoder);
	return x_recon;
}
