//
// Created by zphrfx on 05/07/2025.
//

#pragma once

#include <torch/types.h>

struct VQConfig {  // runtime-selectable
	enum class Device { CPU, CUDA, Metal, Vulkan };
	Device device;
};

class IVQVAECodec {
   public:
	virtual ~IVQVAECodec() = default;

	/**
	 * @brief Encodes a batch of VDB leaf data.
	 * @param leafBatch A CPU tensor of shape [B, 1, 8, 8, 8] with float32 data.
	 * @return A CPU tensor with uint8 quantized indices.
	 */
	virtual torch::Tensor encode(const torch::Tensor& leafBatch) const = 0;

	/**
	 * @brief Decodes a batch of quantized indices.
	 * @param indices A CPU tensor with uint8 quantized indices.
	 * @return A CPU tensor of shape [B, 1, 8, 8, 8] with float32 reconstructed data.
	 */
	virtual torch::Tensor decode(const torch::Tensor& indices) const = 0;

	/**
	 * @brief Gets the shape of the latent tensor produced by the encoder.
	 * @return A vector of dimensions (e.g., {4, 4, 4}).
	 */
	virtual const std::vector<int64_t>& getLatentShape() const = 0;
};