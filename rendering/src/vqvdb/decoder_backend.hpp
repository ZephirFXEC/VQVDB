/*
 * Copyright (c) 2025, Enzo Crema
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>

#include <glm/glm.hpp>

#include "vqvdb/vqvdb_types.hpp"

namespace vqvdb {

/// TensorRT decoder backend used by the rendering pipeline.
class DecoderBackend {
   public:
	DecoderBackend();
	~DecoderBackend();

	[[nodiscard]] DecoderResult<void> init(const std::filesystem::path& onnxModelPath, uint32_t maxBatchSize);

	/// Decode compressed latent blocks into atlas slots.
	/// `indices` layout: [batchSize, 4, 4, 4] packed uint8.
	[[nodiscard]] DecoderResult<void> decodeBatch(std::span<const uint8_t> indices, uint32_t batchSize, uint32_t atlasTexture,
	                                              std::span<const glm::ivec3> slotOffsets);

	[[nodiscard]] bool isReady() const noexcept;
	[[nodiscard]] const char* name() const noexcept { return "TensorRT"; }
	[[nodiscard]] const std::string& lastErrorMessage() const noexcept;

   private:
	struct Impl;
	std::unique_ptr<Impl> impl_;

	[[nodiscard]] DecoderResult<void> buildEngine(const std::filesystem::path& onnxPath, uint32_t maxBatchSize);
	[[nodiscard]] DecoderResult<void> loadCachedEngine(const std::filesystem::path& enginePath);
	[[nodiscard]] std::filesystem::path engineCachePath(const std::filesystem::path& onnxPath) const;
};

}  // namespace vqvdb
