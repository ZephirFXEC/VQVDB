/*
 * Copyright (c) 2025, Enzo Crema
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See the LICENSE file in the project root for full license text.
 */

#pragma once

#include <openvdb/openvdb.h>

#include <filesystem>

#include "core/IVQVAECodec.hpp"

/**
 * @class VQVAECodec
 * @brief A class for compressing and decompressing OpenVDB FloatGrids using a VQ-VAE model.
 *
 * This class provides a high-level interface to stream VDB leaf nodes to a neural network
 * for compression, and to stream encoded data from a file for decompression.
 */
class VQVAECodec {
   public:
	/**
	 * @brief Constructs the VQVAECodec, taking ownership of the backend.
	 * @param backend A unique_ptr to a concrete implementation of IVQVAECodec.
	 */
	explicit VQVAECodec(std::unique_ptr<IVQVAECodec> backend);


	/**
	 * @brief Compresses OpenVDB FloatGrids into a .vqvdb file.
	 * @param grids Vector of OpenVDB FloatGrids to compress.
	 * @param outPath The path for the output .vqvdb file.
	 * @param batchSize The number of leaf blocks to process in a single batch.
	 * @param boss A pointer to Houdini's interrupt handler for progress and cancellation. Can be nullptr.
	 */
	void compress(const std::vector<openvdb::FloatGrid::Ptr>& grids, const std::filesystem::path& outPath, size_t batchSize) const;

	/**
	 * @brief Decompresses a .vqvdb file into OpenVDB FloatGrids.
	 * @param inPath The path to the input .vqvdb file.
	 * @param grids An empty vector that will be populated with the created grids.
	 * @param batchSize The number of leaf blocks to process in a single batch.
	 * @param boss A pointer to Houdini's interrupt handler for progress and cancellation. Can be nullptr.
	 */
	void decompress(const std::filesystem::path& inPath, std::vector<openvdb::FloatGrid::Ptr>& grids, size_t batchSize) const;

   private:
	/**
	 * @brief Encodes a batch of VDB leaf data.
	 * @param cpuBatch A pinned CPU tensor of shape [B, 1, D, D, D] with float32 data.
	 * @return A CPU tensor of shape [B, H, W] (or similar) with uint8 quantized indices.
	 */
	[[nodiscard]] Tensor encodeBatch(const TensorView& cpuBatch) const;

	/**
	 * @brief Decodes a batch of quantized indices.
	 * @param cpuBatch A CPU tensor of shape [B, H, W] (or similar) with uint8 quantized indices.
	 * @return A CPU tensor of shape [B, 1, D, D, D] with float32 reconstructed data.
	 */
	[[nodiscard]] Tensor decodeBatch(const TensorView& cpuBatch) const;

	std::unique_ptr<IVQVAECodec> backend_;
};