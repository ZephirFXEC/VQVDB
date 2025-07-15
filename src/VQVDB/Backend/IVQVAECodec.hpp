//
// Created by zphrfx on 05/07/2025.
//

#pragma once

#include <torch/types.h>

#include <filesystem>
#include <memory>
#include <string>
#include <variant>
#include <vector>


/**
 * @brief Specifies the source for loading the VQ-VAE model.
 */
struct EmbeddedModel {};  // Tag for using the embedded model
using ModelSource = std::variant<EmbeddedModel, std::filesystem::path>;

/**
 * @brief Configuration for creating a VQ-VAE codec.
 */
struct CodecConfig {
	enum class Device { CPU, CUDA };
	Device device = Device::CPU;
	ModelSource source = EmbeddedModel{};
};

/**
 * @brief Interface for a Vector-Quantized Variational Autoencoder Codec.
 *
 * This class provides an abstract interface for encoding and decoding
 * 3D tensor data (e.g., VDB leaf nodes).
 *
 * Instances should be created via the static `create` factory method.
 */
class IVQVAECodec {
   public:
	virtual ~IVQVAECodec() = default;

	/**
	 * @brief Creates a new instance of a VQ-VAE Codec.
	 * @param config The configuration specifying device and model source.
	 * @return A unique_ptr to the created codec, or nullptr on failure.
	 */
	static std::unique_ptr<IVQVAECodec> create(const CodecConfig& config);

	/**
	 * @brief Encodes a batch of 3D data blocks.
	 * @param leafBatch A CPU tensor of shape [B, 1, 8, 8, 8] with float32 data.
	 * @return A CPU tensor of shape [B, L_d, L_h, L_w] with uint8 quantized indices.
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