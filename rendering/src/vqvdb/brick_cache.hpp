/*
 * Copyright (c) 2025, Enzo Crema
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include <cstdint>
#include <optional>
#include <unordered_map>
#include <vector>

#include <glm/glm.hpp>

#include "vqvdb/gpu_resources.hpp"
#include "vqvdb/vqvdb_types.hpp"

namespace vqvdb {

struct BrickCacheConfig {
	uint32_t capacitySlots{2048};
	uint32_t brickSizeVoxels{kBlockSize};
	bool allocateAtlasTexture{true};
};

struct BrickCacheStats {
	uint32_t capacitySlots{0};
	uint32_t residentBricks{0};
	uint64_t lookupCount{0};
	uint64_t hitCount{0};
	uint64_t evictionCount{0};
	float hitRate{0.0f};
};

struct BrickAllocation {
	uint32_t slotIndex{0};
	bool wasCached{false};
	bool evicted{false};
	uint64_t evictedMorton{0};
};

struct CacheHashEntry {
	uint32_t mortonHigh{0};
	uint32_t mortonLow{0};
	uint32_t slotIndex{0xFFFFFFFFu};  // Empty marker
	uint32_t padding{0};
};
static_assert(sizeof(CacheHashEntry) == 16, "CacheHashEntry must be 16 bytes");

struct BrickSlotDebugInfo {
	bool occupied{false};
	uint64_t mortonCode{0};
	uint32_t touchCount{0};
	uint64_t lastAccessCounter{0};
	uint32_t lruRank{0xFFFFFFFFu};  // 0 = most recently used, UINT32_MAX = not resident
};

struct BrickCacheDebugSnapshot {
	glm::ivec3 slotGridDims{0, 0, 0};
	uint32_t capacitySlots{0};
	uint32_t residentBricks{0};
	uint64_t accessCounter{0};
	uint32_t maxTouchCount{0};
	uint64_t maxAge{0};
	std::vector<BrickSlotDebugInfo> slots;
};

/// GPU brick cache state (atlas + CPU lookup state + GPU lookup table).
struct BrickCache {
	// Atlas texture (single channel float volume with packed 8^3 slots)
	uint32_t atlasTexture{0};
	glm::ivec3 atlasDimsVoxels{0, 0, 0};
	glm::ivec3 slotGridDims{0, 0, 0};
	uint32_t brickSizeVoxels{kBlockSize};
	uint32_t capacitySlots{0};
	uint32_t residentCount{0};

	// CPU side mapping and LRU state
	std::unordered_map<uint64_t, uint32_t> mortonToSlot;
	std::vector<uint64_t> slotToMorton;
	std::vector<uint32_t> freeSlots;
	std::vector<int32_t> lruPrev;
	std::vector<int32_t> lruNext;
	int32_t lruHead{-1};  // Most recently used slot
	int32_t lruTail{-1};  // Least recently used slot

	// GPU hash table used by shaders for morton->slot lookup
	GPUBuffer hashTableBuffer;
	uint32_t hashTableCapacity{0};
	bool hashTableDirty{true};

	// Diagnostics
	uint64_t lookupCount{0};
	uint64_t hitCount{0};
	uint64_t evictionCount{0};
	uint64_t accessCounter{0};
	std::vector<uint32_t> slotTouchCount;
	std::vector<uint64_t> slotLastAccess;

	[[nodiscard]] bool isInitialized() const noexcept { return capacitySlots > 0; }
};

/// Initialize cache state and optionally allocate the atlas texture.
[[nodiscard]] GPUResult<void> initBrickCache(BrickCache& cache, const BrickCacheConfig& config = {});

/// Release atlas/hash-table GPU resources and clear CPU state.
void shutdownBrickCache(BrickCache& cache) noexcept;

/// Lookup a brick by morton code. On hit, updates LRU recency.
[[nodiscard]] GPUResult<std::optional<uint32_t>> lookupBrick(BrickCache& cache, uint64_t mortonCode) noexcept;

/// Lookup a brick by block origin. On hit, updates LRU recency.
[[nodiscard]] GPUResult<std::optional<uint32_t>> lookupBrick(BrickCache& cache, const BlockOrigin& origin) noexcept;

/// Allocate a cache slot for morton code, evicting LRU entry when full.
[[nodiscard]] GPUResult<BrickAllocation> allocateSlot(BrickCache& cache, uint64_t mortonCode) noexcept;

/// Allocate a cache slot for block origin, evicting LRU entry when full.
[[nodiscard]] GPUResult<BrickAllocation> allocateSlot(BrickCache& cache, const BlockOrigin& origin) noexcept;

/// Convert slot index to 3D atlas voxel offset.
[[nodiscard]] GPUResult<glm::ivec3> slotToAtlasOffset(const BrickCache& cache, uint32_t slotIndex) noexcept;

/// Return cache diagnostics (hit rate, occupancy, eviction count).
[[nodiscard]] BrickCacheStats getCacheStats(const BrickCache& cache) noexcept;

/// Build per-slot telemetry snapshot for debug/visualization UI.
[[nodiscard]] BrickCacheDebugSnapshot buildBrickCacheDebugSnapshot(const BrickCache& cache);

/// Build an open-addressing hash table from mortonToSlot mapping.
[[nodiscard]] std::vector<CacheHashEntry> buildCacheHashTable(const BrickCache& cache);

/// Build and upload hash table to GPU SSBO. Re-uploads only when dirty.
[[nodiscard]] GPUResult<void> uploadCacheHashTable(BrickCache& cache);

/// Delete uploaded hash table buffer.
void deleteCacheHashTable(BrickCache& cache) noexcept;

}  // namespace vqvdb
