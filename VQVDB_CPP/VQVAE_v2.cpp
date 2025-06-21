//
// Created by zphrfx on 10/06/2025.
//

#include "VQVAE_v2.hpp"


VectorQuantizerEMAImpl::VectorQuantizerEMAImpl(const int64_t num_embeddings, const int64_t embedding_dim, const float commitment_cost,
                                               const float decay, const float eps)
    : commitment_cost(commitment_cost), decay(decay), eps(eps), num_embeddings(num_embeddings), embedding_dim(embedding_dim) {
	auto embed = torch::randn({num_embeddings, embedding_dim});
	embed = torch::nn::functional::normalize(embed,
	                                         torch::nn::functional::NormalizeFuncOptions().p(2).dim(1));  // L2, dim=1

	embedding = register_buffer("embedding", embed.clone());
	cluster_size = register_buffer("cluster_size", torch::ones({num_embeddings}));
	embed_avg = register_buffer("embed_avg", embed.clone());
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> VectorQuantizerEMAImpl::forward(const torch::Tensor& x) {
	const int64_t D = embedding_dim;
	const int64_t dim = x.dim();

	// Build permutation: [0, 2, 3, …, n, 1]
	std::vector<int64_t> perm_fwd{0};
	for (int64_t i = 2; i < dim; ++i) perm_fwd.push_back(i);
	perm_fwd.push_back(1);

	auto perm_x = x.permute(perm_fwd).contiguous();
	auto flat = perm_x.view({-1, D});

	// L2 distance
	auto distances = torch::sum(flat.pow(2), 1, true) + torch::sum(embedding.pow(2), 1) - 2 * torch::matmul(flat, embedding.t());

	auto encoding_indices = torch::argmin(distances, 1);
	auto encodings = torch::one_hot(encoding_indices, num_embeddings).to(flat.dtype());

	auto quantized = torch::matmul(encodings, embedding).view(perm_x.sizes());  // reshape

	// Permute back: [0, C, D1, D2, …]
	std::vector<int64_t> perm_back{0, dim - 1};
	for (int64_t i = 1; i < dim - 1; ++i) perm_back.push_back(i);
	quantized = quantized.permute(perm_back).contiguous();

	// ── EMA updates ───────────────────────────────────────────────────────────
	if (is_training()) {
		torch::NoGradGuard guard;

		auto enc_sum = encodings.sum(0);
		cluster_size.mul_(decay).add_(enc_sum, 1 - decay);

		auto dw = torch::matmul(encodings.t(), flat.detach());
		embed_avg.mul_(decay).add_(dw, 1 - decay);

		auto n = cluster_size.clamp_min(eps);
		embedding.copy_(embed_avg / n.unsqueeze(1));
	}

	// Loss + straight-through
	auto commitment = commitment_cost * torch::nn::functional::mse_loss(
	                                        x, quantized.detach(), torch::nn::functional::MSELossFuncOptions().reduction(torch::kMean));

	quantized = x + (quantized - x).detach();

	// Perplexity
	auto avg_probs = encodings.mean(0);
	auto perplex = torch::exp(-torch::sum(avg_probs * torch::log(avg_probs + 1e-10)));

	return {quantized, commitment, perplex};
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


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> VQVAEImpl::forward(const torch::Tensor& x) {
	const torch::Tensor z = encoder->forward(x);
	auto [quantized, vq_loss, perplexity] = quantizer->forward(z);
	torch::Tensor x_recon = decoder->forward(quantized);
	return std::make_tuple(z, x_recon, vq_loss, perplexity);
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
	const torch::Tensor q_embedding = quantizer->embedding;
	const std::vector<long long> original_indices_shape = indices.sizes().vec();
	const at::Tensor flat_indices = indices.view({-1});

	const at::Tensor selected_vectors = torch::index_select(q_embedding, 0, flat_indices);

	std::vector<int64_t> target_shape = original_indices_shape;
	target_shape.push_back(q_embedding.size(1));  // Append the embedding dimension

	const at::Tensor quantized_vectors = selected_vectors.view(target_shape);
	const int64_t dim = quantized_vectors.dim();
	std::vector<int64_t> permute_back;
	permute_back.push_back(0);
	permute_back.push_back(dim - 1);
	for (int64_t i = 1; i < dim - 1; ++i) permute_back.push_back(i);

	const at::Tensor quantized_for_decoder = quantized_vectors.permute(permute_back).contiguous();
	auto x_recon = decoder->forward(quantized_for_decoder);
	return x_recon;
}


// ── Helpers matching Python extras ─────────────────────────────────────────────
const torch::Tensor& VQVAEImpl::encoder_outputs_to_flat(const torch::Tensor& z)
{
	int64_t D = quantizer->embedding_dim;

	std::vector<int64_t> perm_fwd{0};
	for (int64_t i = 2; i < z.dim(); ++i) perm_fwd.push_back(i);
	perm_fwd.push_back(1);

	return z.permute(perm_fwd).contiguous().view({-1, D});
}

void VQVAEImpl::check_and_reset_dead_codes(const torch::Tensor& encoder_outputs)
{
	const float dead_thr = 1.0;
	auto      device     = quantizer->embedding.device();
	auto      flat_z     = encoder_outputs_to_flat(encoder_outputs.detach());

	auto dead_idx = torch::where(quantizer->cluster_size < dead_thr)[0];

	if (dead_idx.numel() == 0) return;

	std::cout << "INFO: Resetting " << dead_idx.numel() << " dead codes.\n";

	auto num_active = flat_z.size(0);
	if (num_active == 0) {
		std::cerr << "WARNING: Cannot reset dead codes, encoder batch empty.\n";
		return;
	}

	auto sample_idx = torch::randint(0, num_active,
									 {dead_idx.numel()},
									 torch::TensorOptions().device(device)
												   .dtype(torch::kLong));
	auto new_embeds = flat_z.index_select(0, sample_idx);

	quantizer->embedding.index_copy_(0, dead_idx, new_embeds);
	quantizer->embed_avg.index_copy_(0, dead_idx, new_embeds);
	quantizer->cluster_size.index_fill_(0, dead_idx, 1.0);
}