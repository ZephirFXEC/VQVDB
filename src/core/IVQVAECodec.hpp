//
// Created by zphrfx on 05/07/2025.
//

#pragma once

#include <cstddef>  // For std::byte
#include <cstdint>
#include <filesystem>
#include <memory>
#include <variant>
#include <vector>


/**
 * @brief An enumeration to select the desired backend at runtime.
 */
enum class BackendType { LibTorch, ONNX };


/**
 * @brief Specifies the source for loading the VQ-VAE model.
 */
struct EmbeddedModel {};  // Tag for using the embedded model
using ModelSource = std::variant<EmbeddedModel, std::filesystem::path>;

/**
 * @brief Generic data types for tensor elements.
 */
enum class DataType {
	FLOAT32,
	UINT8,
};

/**
 * @brief A non-owning, read-only view of tensor data.
 *
 * Used for passing data into the codec without copying. The caller retains
 * ownership of the data buffer.
 */
struct TensorView {
	const void* data = nullptr;
	std::vector<int64_t> shape;
	DataType dtype;
};

/**
 * @brief An owning container for tensor data.
 *
 * Used for returning data from the codec. It owns the data buffer and handles
 * its lifecycle, preventing memory leaks.
 */
struct Tensor {
	std::vector<std::byte> buffer;
	std::vector<int64_t> shape;
	DataType dtype;

	/**
	 * @brief Provides a typed pointer to the tensor's data.
	 * @tparam T The data type (e.g., float, uint8_t).
	 * @return A typed pointer to the beginning of the data buffer.
	 */
	template <typename T>
	const T* getData() const {
		return reinterpret_cast<const T*>(buffer.data());
	}

	template <typename T>
	T* getData() {
		return reinterpret_cast<T*>(buffer.data());
	}
};

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
 * 3D tensor data using a backend-agnostic representation.
 *
 * Instances should be created via the static `create` factory method.
 */
class IVQVAECodec {
   public:
	virtual ~IVQVAECodec() = default;

	/**
	 * @brief Creates a new instance of a VQ-VAE Codec.
	 *
	 * REFACTORED: Now takes a `BackendType` to explicitly choose the implementation.
	 *
	 * @param config The configuration specifying device and model source.
	 * @param type The backend implementation to create (e.g., LibTorch or ONNX).
	 * @return A unique_ptr to the created codec, or throws on failure.
	 */
	static std::unique_ptr<IVQVAECodec> create(const CodecConfig& config, BackendType type);

	/**
	 * @brief Encodes a batch of 3D data blocks.
	 * @param leafBatch A non-owning view of the input tensor, expecting
	 *                  shape [B, 1, 8, 8, 8] and FLOAT32 data.
	 * @return An owning Tensor containing the quantized indices with
	 *         shape [B, L_d, L_h, L_w] and UINT8 data.
	 */
	virtual Tensor encode(const TensorView& leafBatch) const = 0;

	/**
	 * @brief Decodes a batch of quantized indices.
	 * @param indices A non-owning view of the quantized indices tensor,
	 *                expecting UINT8 data.
	 * @return An owning Tensor containing the reconstructed data with
	 *         shape [B, 1, 8, 8, 8] and FLOAT32 data.
	 */
	virtual Tensor decode(const TensorView& indices) const = 0;

	/**
	 * @brief Gets the shape of the latent tensor produced by the encoder.
	 * @return A vector of dimensions (e.g., {4, 4, 4}).
	 */
	virtual const std::vector<int64_t>& getLatentShape() const = 0;
};