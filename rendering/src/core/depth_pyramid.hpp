/*
 * Copyright (c) 2025, Enzo Crema
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include <cstdint>
#include <vector>

#include <glm/glm.hpp>

#include "vqvdb/vqvdb_types.hpp"

namespace depth_pyramid {

struct Level {
	int width{0};
	int height{0};
	std::vector<float> maxDepth;  // Farther depth = larger value in [0,1]
};

struct DepthPyramid {
	int baseWidth{0};
	int baseHeight{0};
	std::vector<Level> levels;

	[[nodiscard]] bool valid() const noexcept { return !levels.empty() && baseWidth > 0 && baseHeight > 0; }
	void clear() noexcept {
		baseWidth = 0;
		baseHeight = 0;
		levels.clear();
	}
};

/// Double-buffered PBO state for async depth readback.
/// Eliminates the synchronous glReadPixels stall (~12ms) by reading into a PBO
/// on one frame and mapping the previous frame's PBO on the next frame.
struct AsyncDepthReadback {
	uint32_t pbos[2]{0, 0};       ///< Double-buffered Pixel Buffer Objects
	int allocatedWidth{0};        ///< Width the PBOs were allocated for
	int allocatedHeight{0};       ///< Height the PBOs were allocated for
	int frameIndex{0};            ///< Alternates 0/1 each frame
	bool hasValidData{false};     ///< True after the first full round-trip (2 frames)

	[[nodiscard]] bool isAllocated() const noexcept { return pbos[0] != 0 && pbos[1] != 0; }
};

/// Initialize or resize PBOs for async depth readback.
void initAsyncReadback(AsyncDepthReadback& state, int width, int height) noexcept;

/// Release PBO resources.
void shutdownAsyncReadback(AsyncDepthReadback& state) noexcept;

/// Initiate an async depth read from the current framebuffer into a PBO (non-blocking).
/// Call this after the depth prepass draw.
void initiateAsyncCapture(AsyncDepthReadback& state, int x, int y, int width, int height) noexcept;

/// Map the previous frame's PBO and build the depth pyramid from it.
/// Returns true if a valid pyramid was built (requires at least 2 frames of readback).
[[nodiscard]] bool buildPyramidFromPBO(AsyncDepthReadback& state, DepthPyramid& outPyramid) noexcept;

// Conservative occlusion test against the max-depth pyramid.
[[nodiscard]] bool isAABBOccluded(const DepthPyramid& pyramid, const vqvdb::AABB& worldBounds, const glm::mat4& viewProjection,
                                  float depthBias = 1e-4f) noexcept;

}  // namespace depth_pyramid
