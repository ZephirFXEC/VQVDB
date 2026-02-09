/*
 * Copyright (c) 2025, Enzo Crema
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include <cstdint>

#include <glm/glm.hpp>

#include "vqvdb/vqvdb_types.hpp"

namespace vqvdb {

/// Manages CUDA-GL interop for writing decoded bricks to an atlas texture.
struct CudaGLInterop {
	void* resource{nullptr};  ///< Opaque cudaGraphicsResource_t
	uint32_t registeredTexture{0};
	bool mapped{false};

	[[nodiscard]] DecoderResult<void> registerTexture(uint32_t texture);
	void unregisterTexture() noexcept;

	[[nodiscard]] DecoderResult<void> mapForCuda();
	void unmapFromCuda() noexcept;

	[[nodiscard]] DecoderResult<void> copyBrickToAtlas(const float* d_brickData, glm::ivec3 atlasOffset, glm::ivec3 atlasDims);
};

}  // namespace vqvdb
