/*
 * Copyright (c) 2025, Enzo Crema
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include <array>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

#include <glm/glm.hpp>

#include "core/depth_pyramid.hpp"
#include "vqvdb/brick_cache.hpp"
#include "vqvdb/vqvdb_types.hpp"

namespace vqvdb {

enum class BlockDebugState : uint8_t {
	NotVisible = 0,
	VisibleMissing = 1,
	VisibleCached = 2
};

struct FrustumPlane {
	glm::vec3 normal{0.0f, 0.0f, 0.0f};
	float d{0.0f};
};

struct CameraFrustum {
	std::array<FrustumPlane, 6> planes{};
};

struct VisibleBlock {
	uint32_t blockIndex{0};
	BlockOrigin origin{};
	uint64_t mortonCode{0};
	AABB worldBounds{};
	float distanceToCamera{0.0f};
	bool cached{false};
};

struct DecodeRequest {
	uint32_t blockIndex{0};
	BlockOrigin origin{};
	uint64_t mortonCode{0};
	AABB worldBounds{};
	float distanceToCamera{0.0f};
};

struct SchedulerConfig {
	uint32_t maxDecodesPerFrame{64};
	float maxDecodeDistance{std::numeric_limits<float>::infinity()};
	bool frustumCullingEnabled{true};

	// Occlusion culling (integrated into scheduling to avoid a second pass)
	const depth_pyramid::DepthPyramid* depthPyramid{nullptr};
	float occlusionDepthBias{1e-4f};
};

struct ScheduleResult {
	std::vector<VisibleBlock> visibleBlocks;
	std::vector<DecodeRequest> decodeRequests;
	uint32_t occludedCount{0};  ///< Decode candidates rejected by occlusion
};

struct CacheUpdateStats {
	uint32_t touchedVisibleCached{0};
	uint32_t inserted{0};
	uint32_t evicted{0};
};

[[nodiscard]] AABB computeBlockWorldAABB(const BlockOrigin& origin, const GridTransform& transform,
                                         uint32_t blockSize = static_cast<uint32_t>(kBlockSize)) noexcept;

[[nodiscard]] CameraFrustum extractFrustum(const glm::mat4& viewProjection) noexcept;

[[nodiscard]] bool aabbIntersectsFrustum(const AABB& bounds, const CameraFrustum& frustum) noexcept;

[[nodiscard]] ScheduleResult scheduleDecodes(const BlockData& blocks, const GridTransform& transform, const glm::mat4& viewProjection,
                                             const glm::vec3& cameraPosition, const BrickCache* cache, const SchedulerConfig& config = {});

[[nodiscard]] GPUResult<CacheUpdateStats> updateCacheFromVisibleSet(BrickCache& cache, std::span<const VisibleBlock> visibleBlocks,
                                                                     std::span<const DecodeRequest> decodeRequests);

}  // namespace vqvdb
