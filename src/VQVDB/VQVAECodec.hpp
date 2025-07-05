// VQVAECodec.hpp
//
// Created by zphrfx on 23/06/2025.
//

#pragma once

#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <openvdb/openvdb.h>
#include <torch/script.h>

#include <filesystem>
#include <string>

/**
 * @class VQVAECodec
 * @brief A class for compressing and decompressing OpenVDB FloatGrids using a VQ-VAE model.
 *
 * This class provides a high-level interface to stream VDB leaf nodes to a neural network
 * for compression, and to stream encoded data from a file for decompression. It handles
 * GPU acceleration via PyTorch/libtorch if a CUDA-enabled device is available.
 */
class VQVAECodec {
   public:
	/**
	 * @brief Constructs the VQVAECodec.
	 */
	explicit VQVAECodec();

	~VQVAECodec() = default;

	/**
	 * @brief Compresses an OpenVDB FloatGrid into a .vqvdb file.
	 *
	 * This method streams leaf blocks from the grid, encodes them in batches,
	 * and writes the result to a file. The file format stores all block origins
	 * first, followed by the concatenated stream of encoded data.
	 *
	 * @param grid The input grid to compress.
	 * @param outPath The path for the output .vqvdb file.
	 * @param batchSize The number of leaf blocks to process in a single GPU batch.
	 */
	void compress(const openvdb::FloatGrid::Ptr& grid, const std::string& outPath, size_t batchSize) const;

	/**
	 * @brief Decompresses a .vqvdb file into an OpenVDB FloatGrid.
	 *
	 * This method streams encoded data from the file, decodes it in batches,
	 * and reconstructs the grid. It's memory-efficient as it doesn't load all
	 * encoded data at once.
	 *
	 * @param inPath The path to the input .vqvdb file.
	 * @param grid A pointer to a FloatGrid which will be created and populated.
	 * @param batchSize The number of leaf blocks to process in a single GPU batch.
	 */
	void decompress(const std::string& inPath, openvdb::FloatGrid::Ptr& grid, size_t batchSize) const;

   private:
	/**
	 * @brief Encodes a batch of VDB leaf data.
	 * @param cpuBatch A pinned CPU tensor of shape [B, 1, D, D, D] with float32 data.
	 * @return A CPU tensor of shape [B, H, W] (or similar) with uint8 quantized indices.
	 */
	torch::Tensor encodeBatch(const torch::Tensor& cpuBatch) const;

	/**
	 * @brief Decodes a batch of quantized indices.
	 * @param cpuBatch A CPU tensor of shape [B, H, W] (or similar) with uint8 quantized indices.
	 * @return A CPU tensor of shape [B, 1, D, D, D] with float32 reconstructed data.
	 */
	torch::Tensor decodeBatch(const torch::Tensor& cpuBatch) const;

	// Helper to convert torch::Tensor to Ort::Value
	Ort::Value tensorToOrtValue(const torch::Tensor& tensor, const Ort::MemoryInfo& memory_info) const;

	// --- ONNX Runtime Members ---
	Ort::Env env_;
	Ort::SessionOptions session_options_;
	std::unique_ptr<Ort::Session> encoder_session_;
	std::unique_ptr<Ort::Session> decoder_session_;

	// Store input/output names for convenience
	std::vector<const char*> encoder_input_names_;
	std::vector<const char*> encoder_output_names_;
	std::vector<const char*> decoder_input_names_;
	std::vector<const char*> decoder_output_names_;

	// --- Other Members ---
	torch::Device device_;
	Ort::MemoryInfo memory_info_cuda_{nullptr};  // To be initialized
};