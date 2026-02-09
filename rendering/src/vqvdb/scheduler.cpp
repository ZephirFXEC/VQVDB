/*
 * Copyright (c) 2025, Enzo Crema
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "vqvdb/scheduler.hpp"

#include <algorithm>
#include <unordered_set>

#include "core/depth_pyramid.hpp"

namespace vqvdb {

namespace {

constexpr uint32_t kFrustumPlaneCount = 6;

[[nodiscard]] FrustumPlane normalizePlane(const glm::vec4& rawPlane) noexcept {
	FrustumPlane out{};
	out.normal = glm::vec3(rawPlane.x, rawPlane.y, rawPlane.z);
	const float len = glm::length(out.normal);
	if (len <= 0.0f) {
		return out;
	}
	const float invLen = 1.0f / len;
	out.normal *= invLen;
	out.d = rawPlane.w * invLen;
	return out;
}

[[nodiscard]] bool aabbOutsidePlane(const AABB& bounds, const FrustumPlane& plane) noexcept {
	glm::vec3 p = bounds.min;
	if (plane.normal.x >= 0.0f) p.x = bounds.max.x;
	if (plane.normal.y >= 0.0f) p.y = bounds.max.y;
	if (plane.normal.z >= 0.0f) p.z = bounds.max.z;
	return (glm::dot(plane.normal, p) + plane.d) < 0.0f;
}

[[nodiscard]] glm::vec4 matrixRow(const glm::mat4& m, int row) noexcept {
	return {m[0][row], m[1][row], m[2][row], m[3][row]};
}

}  // namespace

AABB computeBlockWorldAABB(const BlockOrigin& origin, const GridTransform& transform, uint32_t blockSize) noexcept {
	const glm::vec3 localMin = origin.toVec3();
	const glm::vec3 localMax = localMin + glm::vec3(static_cast<float>(blockSize));
	const glm::vec3 localCenter = (localMin + localMax) * 0.5f;
	const glm::vec3 localExtent = (localMax - localMin) * 0.5f;

	const glm::mat4 worldMat = transform.toMat4();
	const glm::vec3 worldCenter = glm::vec3(worldMat * glm::vec4(localCenter, 1.0f));

	const glm::mat3 linear(worldMat);
	const glm::mat3 absLinear(glm::abs(linear[0]), glm::abs(linear[1]), glm::abs(linear[2]));
	const glm::vec3 worldExtent = absLinear * localExtent;

	return {.min = worldCenter - worldExtent, .max = worldCenter + worldExtent};
}

CameraFrustum extractFrustum(const glm::mat4& viewProjection) noexcept {
	CameraFrustum frustum{};

	const glm::vec4 r0 = matrixRow(viewProjection, 0);
	const glm::vec4 r1 = matrixRow(viewProjection, 1);
	const glm::vec4 r2 = matrixRow(viewProjection, 2);
	const glm::vec4 r3 = matrixRow(viewProjection, 3);

	frustum.planes[0] = normalizePlane(r3 + r0);  // Left
	frustum.planes[1] = normalizePlane(r3 - r0);  // Right
	frustum.planes[2] = normalizePlane(r3 + r1);  // Bottom
	frustum.planes[3] = normalizePlane(r3 - r1);  // Top
	frustum.planes[4] = normalizePlane(r3 + r2);  // Near
	frustum.planes[5] = normalizePlane(r3 - r2);  // Far

	return frustum;
}

bool aabbIntersectsFrustum(const AABB& bounds, const CameraFrustum& frustum) noexcept {
	if (!bounds.isValid()) {
		return false;
	}

	for (uint32_t i = 0; i < kFrustumPlaneCount; ++i) {
		if (aabbOutsidePlane(bounds, frustum.planes[i])) {
			return false;
		}
	}
	return true;
}

ScheduleResult scheduleDecodes(const BlockData& blocks, const GridTransform& transform, const glm::mat4& viewProjection,
                               const glm::vec3& cameraPosition, const BrickCache* cache, const SchedulerConfig& config) {
	ScheduleResult result{};
	if (blocks.empty()) {
		return result;
	}

	const bool useCache = cache != nullptr && cache->isInitialized();
	const CameraFrustum frustum = extractFrustum(viewProjection);
	const float maxDecodeDistance = std::max(0.0f, config.maxDecodeDistance);
	const bool useOcclusion = config.depthPyramid != nullptr && config.depthPyramid->valid();

	result.visibleBlocks.reserve(blocks.count());
	result.decodeRequests.reserve(std::min<size_t>(blocks.count(), static_cast<size_t>(config.maxDecodesPerFrame)));

	std::unordered_set<uint64_t> decodeDedup;
	decodeDedup.reserve(blocks.count() / 4u + 1u);

	for (size_t i = 0; i < blocks.origins.size(); ++i) {
		const BlockOrigin& origin = blocks.origins[i];
		const AABB worldBounds = computeBlockWorldAABB(origin, transform);

		if (config.frustumCullingEnabled && !aabbIntersectsFrustum(worldBounds, frustum)) {
			continue;
		}

		const uint64_t morton = blocks.mortonCode(i);
		const bool isCached = useCache && cache->mortonToSlot.contains(morton);
		const float distance = glm::distance(cameraPosition, worldBounds.center());

		result.visibleBlocks.push_back(VisibleBlock{
		    .blockIndex = static_cast<uint32_t>(i),
		    .origin = origin,
		    .mortonCode = morton,
		    .worldBounds = worldBounds,
		    .distanceToCamera = distance,
		    .cached = isCached,
		});

		if (!isCached && distance <= maxDecodeDistance) {
			// Occlusion test: reject decode candidates hidden behind already-rendered geometry.
			if (useOcclusion &&
			    depth_pyramid::isAABBOccluded(*config.depthPyramid, worldBounds, viewProjection, config.occlusionDepthBias)) {
				++result.occludedCount;
				continue;
			}

			if (decodeDedup.insert(morton).second) {
				result.decodeRequests.push_back(DecodeRequest{
				    .blockIndex = static_cast<uint32_t>(i),
				    .origin = origin,
				    .mortonCode = morton,
				    .worldBounds = worldBounds,
				    .distanceToCamera = distance,
				});
			}
		}
	}

	std::sort(result.decodeRequests.begin(), result.decodeRequests.end(),
	          [](const DecodeRequest& a, const DecodeRequest& b) { return a.distanceToCamera < b.distanceToCamera; });

	const size_t budget = static_cast<size_t>(config.maxDecodesPerFrame);
	if (result.decodeRequests.size() > budget) {
		result.decodeRequests.resize(budget);
	}

	return result;
}

GPUResult<CacheUpdateStats> updateCacheFromVisibleSet(BrickCache& cache, std::span<const VisibleBlock> visibleBlocks,
                                                       std::span<const DecodeRequest> decodeRequests) {
	if (!cache.isInitialized()) {
		return std::unexpected(GPUError::NotInitialized);
	}

	CacheUpdateStats stats{};

	for (const VisibleBlock& visible : visibleBlocks) {
		if (!visible.cached) {
			continue;
		}

		const auto lookup = lookupBrick(cache, visible.mortonCode);
		if (!lookup.has_value()) {
			return std::unexpected(lookup.error());
		}
		if (lookup->has_value()) {
			++stats.touchedVisibleCached;
		}
	}

	for (const DecodeRequest& request : decodeRequests) {
		const auto alloc = allocateSlot(cache, request.mortonCode);
		if (!alloc.has_value()) {
			return std::unexpected(alloc.error());
		}
		if (!alloc->wasCached) {
			++stats.inserted;
		}
		if (alloc->evicted) {
			++stats.evicted;
		}
	}

	const auto upload = uploadCacheHashTable(cache);
	if (!upload.has_value()) {
		return std::unexpected(upload.error());
	}

	return stats;
}

}  // namespace vqvdb
