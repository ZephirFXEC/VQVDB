/*
 * Copyright (c) 2025, Enzo Crema
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "core/depth_pyramid.hpp"

#include <glad/glad.h>

#include <algorithm>
#include <cmath>
#include <cstring>

namespace depth_pyramid {

namespace {

[[nodiscard]] size_t idx(int x, int y, int w) {
	return static_cast<size_t>(y) * static_cast<size_t>(w) + static_cast<size_t>(x);
}

[[nodiscard]] bool projectAABBToScreen(const vqvdb::AABB& bounds, const glm::mat4& vp, float& outMinU, float& outMinV, float& outMaxU,
                                       float& outMaxV, float& outNearDepth) noexcept {
	if (!bounds.isValid()) {
		return false;
	}

	float minU = 1.0f;
	float minV = 1.0f;
	float maxU = 0.0f;
	float maxV = 0.0f;
	float nearDepth = 1.0f;

	for (int corner = 0; corner < 8; ++corner) {
		const glm::vec3 p((corner & 1) ? bounds.max.x : bounds.min.x, (corner & 2) ? bounds.max.y : bounds.min.y,
		                  (corner & 4) ? bounds.max.z : bounds.min.z);

		const glm::vec4 clip = vp * glm::vec4(p, 1.0f);
		if (clip.w <= 0.0f) {
			// Conservative: avoid occlusion cull for boxes crossing camera plane.
			return false;
		}

		const glm::vec3 ndc = glm::vec3(clip) / clip.w;
		const float u = ndc.x * 0.5f + 0.5f;
		const float v = ndc.y * 0.5f + 0.5f;
		const float d = ndc.z * 0.5f + 0.5f;

		minU = std::min(minU, u);
		minV = std::min(minV, v);
		maxU = std::max(maxU, u);
		maxV = std::max(maxV, v);
		nearDepth = std::min(nearDepth, d);
	}

	if (maxU < 0.0f || maxV < 0.0f || minU > 1.0f || minV > 1.0f) {
		return false;
	}

	outMinU = std::clamp(minU, 0.0f, 1.0f);
	outMinV = std::clamp(minV, 0.0f, 1.0f);
	outMaxU = std::clamp(maxU, 0.0f, 1.0f);
	outMaxV = std::clamp(maxV, 0.0f, 1.0f);
	outNearDepth = std::clamp(nearDepth, 0.0f, 1.0f);

	return outMaxU > outMinU && outMaxV > outMinV;
}

[[nodiscard]] int chooseLevel(const DepthPyramid& pyramid, int rectW, int rectH) noexcept {
	const int longest = std::max(1, std::max(rectW, rectH));
	int level = 0;
	int span = longest;
	while (span > 4 && (level + 1) < static_cast<int>(pyramid.levels.size())) {
		span >>= 1;
		++level;
	}
	return level;
}

/// Build the mip pyramid from an existing base-level depth buffer.
void buildMipChain(DepthPyramid& pyramid) {
	while (true) {
		const Level& prev = pyramid.levels.back();
		if (prev.width == 1 && prev.height == 1) {
			break;
		}

		Level next{};
		next.width = std::max(1, (prev.width + 1) / 2);
		next.height = std::max(1, (prev.height + 1) / 2);
		next.maxDepth.resize(static_cast<size_t>(next.width) * static_cast<size_t>(next.height), 1.0f);

		for (int py = 0; py < next.height; ++py) {
			for (int px = 0; px < next.width; ++px) {
				const int x0 = px * 2;
				const int y0 = py * 2;
				const int x1 = std::min(x0 + 1, prev.width - 1);
				const int y1 = std::min(y0 + 1, prev.height - 1);

				float m = prev.maxDepth[idx(x0, y0, prev.width)];
				m = std::max(m, prev.maxDepth[idx(x1, y0, prev.width)]);
				m = std::max(m, prev.maxDepth[idx(x0, y1, prev.width)]);
				m = std::max(m, prev.maxDepth[idx(x1, y1, prev.width)]);
				next.maxDepth[idx(px, py, next.width)] = m;
			}
		}

		pyramid.levels.push_back(std::move(next));
	}
}

}  // namespace

// ============================================================================
// Async PBO double-buffered depth readback
// ============================================================================

void initAsyncReadback(AsyncDepthReadback& state, int width, int height) noexcept {
	if (width <= 0 || height <= 0) {
		return;
	}

	// If already allocated at the right size, nothing to do.
	if (state.isAllocated() && state.allocatedWidth == width && state.allocatedHeight == height) {
		return;
	}

	// Tear down old PBOs if size changed.
	shutdownAsyncReadback(state);

	const auto bufferSize = static_cast<GLsizeiptr>(width) * height * sizeof(float);

	glCreateBuffers(2, state.pbos);
	for (int i = 0; i < 2; ++i) {
		if (state.pbos[i] == 0) {
			shutdownAsyncReadback(state);
			return;
		}
		// GL_MAP_READ_BIT: we will map this buffer for reading.
		// GL_CLIENT_STORAGE_BIT: hint to the driver to keep a CPU-accessible copy.
		glNamedBufferStorage(state.pbos[i], bufferSize, nullptr, GL_MAP_READ_BIT | GL_CLIENT_STORAGE_BIT);
	}

	state.allocatedWidth = width;
	state.allocatedHeight = height;
	state.frameIndex = 0;
	state.hasValidData = false;
}

void shutdownAsyncReadback(AsyncDepthReadback& state) noexcept {
	for (int i = 0; i < 2; ++i) {
		if (state.pbos[i] != 0) {
			glDeleteBuffers(1, &state.pbos[i]);
			state.pbos[i] = 0;
		}
	}
	state.allocatedWidth = 0;
	state.allocatedHeight = 0;
	state.frameIndex = 0;
	state.hasValidData = false;
}

void initiateAsyncCapture(AsyncDepthReadback& state, int x, int y, int width, int height) noexcept {
	initAsyncReadback(state, width, height);
	if (!state.isAllocated()) {
		return;
	}

	// Bind the "write" PBO and issue a non-blocking glReadPixels into it.
	const int writeIdx = state.frameIndex & 1;
	glBindBuffer(GL_PIXEL_PACK_BUFFER, state.pbos[writeIdx]);
	glPixelStorei(GL_PACK_ALIGNMENT, 1);
	glReadPixels(x, y, width, height, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
	glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
}

bool buildPyramidFromPBO(AsyncDepthReadback& state, DepthPyramid& outPyramid) noexcept {
	outPyramid.clear();

	if (!state.isAllocated()) {
		return false;
	}

	// On the very first frame, we have no previous-frame data to map.
	// After the first initiateAsyncCapture, flip parity and mark data as valid for next frame.
	const int readIdx = (state.frameIndex ^ 1) & 1;

	// Advance frame parity for next call.
	state.frameIndex++;

	if (!state.hasValidData) {
		// First frame: we just issued the first readback; no data to map yet.
		state.hasValidData = true;
		return false;
	}

	// Map the "read" PBO (contains PREVIOUS frame's depth data, already DMA'd).
	const auto* mapped = static_cast<const float*>(
	    glMapNamedBufferRange(state.pbos[readIdx], 0,
	                          static_cast<GLsizeiptr>(state.allocatedWidth) * state.allocatedHeight * sizeof(float),
	                          GL_MAP_READ_BIT));
	if (mapped == nullptr) {
		return false;
	}

	const int width = state.allocatedWidth;
	const int height = state.allocatedHeight;

	Level base{};
	base.width = width;
	base.height = height;
	base.maxDepth.resize(static_cast<size_t>(width) * static_cast<size_t>(height));
	std::memcpy(base.maxDepth.data(), mapped, base.maxDepth.size() * sizeof(float));

	glUnmapNamedBuffer(state.pbos[readIdx]);

	outPyramid.baseWidth = width;
	outPyramid.baseHeight = height;
	outPyramid.levels.push_back(std::move(base));

	buildMipChain(outPyramid);

	return outPyramid.valid();
}

// ============================================================================
// Occlusion test (unchanged logic)
// ============================================================================

bool isAABBOccluded(const DepthPyramid& pyramid, const vqvdb::AABB& worldBounds, const glm::mat4& viewProjection, float depthBias) noexcept {
	if (!pyramid.valid()) {
		return false;
	}

	float minU = 0.0f;
	float minV = 0.0f;
	float maxU = 0.0f;
	float maxV = 0.0f;
	float nearDepth = 0.0f;
	if (!projectAABBToScreen(worldBounds, viewProjection, minU, minV, maxU, maxV, nearDepth)) {
		return false;
	}

	const int x0 = std::clamp(static_cast<int>(std::floor(minU * static_cast<float>(pyramid.baseWidth))), 0, pyramid.baseWidth - 1);
	const int y0 = std::clamp(static_cast<int>(std::floor(minV * static_cast<float>(pyramid.baseHeight))), 0, pyramid.baseHeight - 1);
	const int x1 = std::clamp(static_cast<int>(std::ceil(maxU * static_cast<float>(pyramid.baseWidth))) - 1, 0, pyramid.baseWidth - 1);
	const int y1 = std::clamp(static_cast<int>(std::ceil(maxV * static_cast<float>(pyramid.baseHeight))) - 1, 0, pyramid.baseHeight - 1);

	if (x1 < x0 || y1 < y0) {
		return false;
	}

	const int rectW = x1 - x0 + 1;
	const int rectH = y1 - y0 + 1;
	const int levelIdx = chooseLevel(pyramid, rectW, rectH);
	const Level& level = pyramid.levels[static_cast<size_t>(levelIdx)];
	const int scale = 1 << levelIdx;

	const int lx0 = std::clamp(x0 / scale, 0, level.width - 1);
	const int ly0 = std::clamp(y0 / scale, 0, level.height - 1);
	const int lx1 = std::clamp(x1 / scale, 0, level.width - 1);
	const int ly1 = std::clamp(y1 / scale, 0, level.height - 1);

	// Fully occluded only if block near-depth is behind every covered max-depth cell.
	for (int y = ly0; y <= ly1; ++y) {
		for (int x = lx0; x <= lx1; ++x) {
			const float zMax = level.maxDepth[idx(x, y, level.width)];
			if (!(nearDepth > zMax + depthBias)) {
				return false;
			}
		}
	}

	return true;
}

}  // namespace depth_pyramid
