/*
 * Copyright (c) 2025, Enzo Crema
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "vqvdb/brick_cache.hpp"

#include <glad/glad.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace vqvdb {

namespace {

constexpr uint64_t kInvalidMorton = std::numeric_limits<uint64_t>::max();
constexpr uint32_t kHashTableEmptySlot = 0xFFFFFFFFu;

[[nodiscard]] uint32_t nextPowerOfTwo(uint32_t v) {
	if (v <= 1) return 1;
	--v;
	v |= v >> 1;
	v |= v >> 2;
	v |= v >> 4;
	v |= v >> 8;
	v |= v >> 16;
	return v + 1;
}

[[nodiscard]] glm::ivec3 computeSlotGridDims(uint32_t capacitySlots) {
	const double cbrtCap = std::cbrt(static_cast<double>(capacitySlots));
	const uint32_t axis = nextPowerOfTwo(static_cast<uint32_t>(std::ceil(cbrtCap)));
	const uint32_t xy = axis * axis;

	uint32_t z = (capacitySlots + xy - 1u) / xy;
	z = nextPowerOfTwo(std::max(1u, z));
	while (z > 1u && (xy * (z / 2u)) >= capacitySlots) {
		z /= 2u;
	}

	return {static_cast<int32_t>(axis), static_cast<int32_t>(axis), static_cast<int32_t>(z)};
}

void detachLRUSlot(BrickCache& cache, int32_t slot) {
	const int32_t prev = cache.lruPrev[static_cast<size_t>(slot)];
	const int32_t next = cache.lruNext[static_cast<size_t>(slot)];

	if (prev != -1) cache.lruNext[static_cast<size_t>(prev)] = next;
	if (next != -1) cache.lruPrev[static_cast<size_t>(next)] = prev;

	if (cache.lruHead == slot) cache.lruHead = next;
	if (cache.lruTail == slot) cache.lruTail = prev;

	cache.lruPrev[static_cast<size_t>(slot)] = -1;
	cache.lruNext[static_cast<size_t>(slot)] = -1;
}

void pushLRUFront(BrickCache& cache, int32_t slot) {
	cache.lruPrev[static_cast<size_t>(slot)] = -1;
	cache.lruNext[static_cast<size_t>(slot)] = cache.lruHead;

	if (cache.lruHead != -1) {
		cache.lruPrev[static_cast<size_t>(cache.lruHead)] = slot;
	}
	cache.lruHead = slot;

	if (cache.lruTail == -1) {
		cache.lruTail = slot;
	}
}

void touchLRUSlot(BrickCache& cache, int32_t slot) {
	if (cache.lruHead == slot) {
		return;
	}
	detachLRUSlot(cache, slot);
	pushLRUFront(cache, slot);
}

void touchSlotTelemetry(BrickCache& cache, uint32_t slot) {
	if (slot >= cache.slotTouchCount.size() || slot >= cache.slotLastAccess.size()) {
		return;
	}
	++cache.accessCounter;
	cache.slotLastAccess[slot] = cache.accessCounter;
	++cache.slotTouchCount[slot];
}

[[nodiscard]] int32_t popLRUTail(BrickCache& cache) {
	const int32_t slot = cache.lruTail;
	if (slot != -1) {
		detachLRUSlot(cache, slot);
	}
	return slot;
}

[[nodiscard]] uint32_t mortonHash(uint64_t morton) {
	uint64_t x = morton;
	x ^= x >> 33;
	x *= 0xff51afd7ed558ccdULL;
	x ^= x >> 33;
	x *= 0xc4ceb9fe1a85ec53ULL;
	x ^= x >> 33;
	return static_cast<uint32_t>(x);
}

void resetCacheState(BrickCache& cache) {
	cache.mortonToSlot.clear();
	cache.slotToMorton.assign(cache.capacitySlots, kInvalidMorton);
	cache.freeSlots.clear();
	cache.freeSlots.reserve(cache.capacitySlots);
	for (uint32_t i = cache.capacitySlots; i > 0; --i) {
		cache.freeSlots.push_back(i - 1u);
	}

	cache.lruPrev.assign(cache.capacitySlots, -1);
	cache.lruNext.assign(cache.capacitySlots, -1);
	cache.lruHead = -1;
	cache.lruTail = -1;

	cache.lookupCount = 0;
	cache.hitCount = 0;
	cache.evictionCount = 0;
	cache.accessCounter = 0;
	cache.slotTouchCount.assign(cache.capacitySlots, 0);
	cache.slotLastAccess.assign(cache.capacitySlots, 0);
	cache.hashTableDirty = true;
}

[[nodiscard]] bool glOk() { return glGetError() == GL_NO_ERROR; }

}  // namespace

GPUResult<void> initBrickCache(BrickCache& cache, const BrickCacheConfig& config) {
	if (config.capacitySlots == 0 || config.brickSizeVoxels == 0) {
		return std::unexpected(GPUError::InvalidData);
	}

	shutdownBrickCache(cache);

	cache.capacitySlots = config.capacitySlots;
	cache.brickSizeVoxels = config.brickSizeVoxels;
	cache.slotGridDims = computeSlotGridDims(config.capacitySlots);
	cache.atlasDimsVoxels = cache.slotGridDims * static_cast<int32_t>(cache.brickSizeVoxels);

	resetCacheState(cache);

	if (!config.allocateAtlasTexture) {
		return {};
	}

	GLint max3DTextureSize = 0;
	glGetIntegerv(GL_MAX_3D_TEXTURE_SIZE, &max3DTextureSize);
	if (max3DTextureSize <= 0 || cache.atlasDimsVoxels.x > max3DTextureSize || cache.atlasDimsVoxels.y > max3DTextureSize ||
	    cache.atlasDimsVoxels.z > max3DTextureSize) {
		return std::unexpected(GPUError::AllocationFailed);
	}

	glCreateTextures(GL_TEXTURE_3D, 1, &cache.atlasTexture);
	if (cache.atlasTexture == 0) {
		return std::unexpected(GPUError::AllocationFailed);
	}

	glTextureParameteri(cache.atlasTexture, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTextureParameteri(cache.atlasTexture, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTextureParameteri(cache.atlasTexture, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTextureParameteri(cache.atlasTexture, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTextureParameteri(cache.atlasTexture, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

	glTextureStorage3D(cache.atlasTexture, 1, GL_R32F, cache.atlasDimsVoxels.x, cache.atlasDimsVoxels.y, cache.atlasDimsVoxels.z);
	if (!glOk()) {
		glDeleteTextures(1, &cache.atlasTexture);
		cache.atlasTexture = 0;
		return std::unexpected(GPUError::AllocationFailed);
	}

	const float zero = 0.0f;
	glClearTexImage(cache.atlasTexture, 0, GL_RED, GL_FLOAT, &zero);
	if (!glOk()) {
		glDeleteTextures(1, &cache.atlasTexture);
		cache.atlasTexture = 0;
		return std::unexpected(GPUError::UploadFailed);
	}

	return {};
}

void deleteCacheHashTable(BrickCache& cache) noexcept {
	if (cache.hashTableBuffer.id != 0) {
		glDeleteBuffers(1, &cache.hashTableBuffer.id);
		cache.hashTableBuffer.invalidate();
	}
	cache.hashTableCapacity = 0;
	cache.hashTableDirty = true;
}

void shutdownBrickCache(BrickCache& cache) noexcept {
	deleteCacheHashTable(cache);

	if (cache.atlasTexture != 0) {
		glDeleteTextures(1, &cache.atlasTexture);
	}

	cache = BrickCache{};
}

GPUResult<std::optional<uint32_t>> lookupBrick(BrickCache& cache, uint64_t mortonCode) noexcept {
	if (!cache.isInitialized()) {
		return std::unexpected(GPUError::NotInitialized);
	}

	++cache.lookupCount;
	const auto it = cache.mortonToSlot.find(mortonCode);
	if (it == cache.mortonToSlot.end()) {
		return std::optional<uint32_t>{};
	}

	++cache.hitCount;
	touchLRUSlot(cache, static_cast<int32_t>(it->second));
	touchSlotTelemetry(cache, it->second);
	return std::optional<uint32_t>{it->second};
}

GPUResult<BrickAllocation> allocateSlot(BrickCache& cache, uint64_t mortonCode) noexcept {
	if (!cache.isInitialized()) {
		return std::unexpected(GPUError::NotInitialized);
	}

	const auto existing = cache.mortonToSlot.find(mortonCode);
	if (existing != cache.mortonToSlot.end()) {
		touchLRUSlot(cache, static_cast<int32_t>(existing->second));
		touchSlotTelemetry(cache, existing->second);
		return BrickAllocation{.slotIndex = existing->second, .wasCached = true, .evicted = false, .evictedMorton = 0};
	}

	BrickAllocation allocation{};
	if (!cache.freeSlots.empty()) {
		allocation.slotIndex = cache.freeSlots.back();
		cache.freeSlots.pop_back();
	} else {
		const int32_t evictedSlot = popLRUTail(cache);
		if (evictedSlot < 0) {
			return std::unexpected(GPUError::InvalidData);
		}

		allocation.slotIndex = static_cast<uint32_t>(evictedSlot);
		allocation.evicted = true;
		allocation.evictedMorton = cache.slotToMorton[static_cast<size_t>(allocation.slotIndex)];
		cache.mortonToSlot.erase(allocation.evictedMorton);
		cache.slotToMorton[static_cast<size_t>(allocation.slotIndex)] = kInvalidMorton;
		cache.slotTouchCount[allocation.slotIndex] = 0;
		cache.slotLastAccess[allocation.slotIndex] = 0;
		++cache.evictionCount;
	}

	cache.mortonToSlot[mortonCode] = allocation.slotIndex;
	cache.slotToMorton[static_cast<size_t>(allocation.slotIndex)] = mortonCode;
	pushLRUFront(cache, static_cast<int32_t>(allocation.slotIndex));
	touchSlotTelemetry(cache, allocation.slotIndex);
	cache.hashTableDirty = true;
	return allocation;
}

GPUResult<glm::ivec3> slotToAtlasOffset(const BrickCache& cache, uint32_t slotIndex) noexcept {
	if (!cache.isInitialized()) {
		return std::unexpected(GPUError::NotInitialized);
	}
	if (slotIndex >= cache.capacitySlots) {
		return std::unexpected(GPUError::InvalidData);
	}

	const uint32_t sx = static_cast<uint32_t>(cache.slotGridDims.x);
	const uint32_t sy = static_cast<uint32_t>(cache.slotGridDims.y);
	const uint32_t slotsPerLayer = sx * sy;

	const uint32_t z = slotIndex / slotsPerLayer;
	const uint32_t rem = slotIndex % slotsPerLayer;
	const uint32_t y = rem / sx;
	const uint32_t x = rem % sx;

	const int32_t brick = static_cast<int32_t>(cache.brickSizeVoxels);
	return glm::ivec3(static_cast<int32_t>(x) * brick, static_cast<int32_t>(y) * brick, static_cast<int32_t>(z) * brick);
}

BrickCacheStats getCacheStats(const BrickCache& cache) noexcept {
	BrickCacheStats stats{};
	stats.capacitySlots = cache.capacitySlots;
	stats.residentBricks = cache.residentCount();
	stats.lookupCount = cache.lookupCount;
	stats.hitCount = cache.hitCount;
	stats.evictionCount = cache.evictionCount;
	stats.hitRate = (stats.lookupCount == 0) ? 0.0f : static_cast<float>(stats.hitCount) / static_cast<float>(stats.lookupCount);
	return stats;
}

BrickCacheDebugSnapshot buildBrickCacheDebugSnapshot(const BrickCache& cache) {
	BrickCacheDebugSnapshot snapshot{};
	snapshot.slotGridDims = cache.slotGridDims;
	snapshot.capacitySlots = cache.capacitySlots;
	snapshot.residentBricks = cache.residentCount();
	snapshot.accessCounter = cache.accessCounter;
	snapshot.slots.resize(cache.capacitySlots);

	for (uint32_t slot = 0; slot < cache.capacitySlots; ++slot) {
		BrickSlotDebugInfo info{};
		const uint64_t morton = (slot < cache.slotToMorton.size()) ? cache.slotToMorton[slot] : kInvalidMorton;
		info.occupied = (morton != kInvalidMorton);
		info.mortonCode = info.occupied ? morton : 0;
		info.touchCount = (slot < cache.slotTouchCount.size()) ? cache.slotTouchCount[slot] : 0;
		info.lastAccessCounter = (slot < cache.slotLastAccess.size()) ? cache.slotLastAccess[slot] : 0;
		snapshot.maxTouchCount = std::max(snapshot.maxTouchCount, info.touchCount);
		if (info.occupied && info.lastAccessCounter <= snapshot.accessCounter) {
			snapshot.maxAge = std::max(snapshot.maxAge, snapshot.accessCounter - info.lastAccessCounter);
		}
		snapshot.slots[slot] = info;
	}

	uint32_t rank = 0;
	for (int32_t cur = cache.lruHead; cur != -1; cur = cache.lruNext[static_cast<size_t>(cur)]) {
		snapshot.slots[static_cast<size_t>(cur)].lruRank = rank++;
	}

	return snapshot;
}

std::vector<CacheHashEntry> buildCacheHashTable(const BrickCache& cache) {
	const uint32_t minSize = std::max(1u, cache.capacitySlots * 2u);
	const uint32_t tableSize = nextPowerOfTwo(minSize);
	std::vector<CacheHashEntry> table(static_cast<size_t>(tableSize));

	for (auto& e : table) {
		e.mortonHigh = 0;
		e.mortonLow = 0;
		e.slotIndex = kHashTableEmptySlot;
		e.padding = 0;
	}

	if (cache.mortonToSlot.empty()) {
		return table;
	}

	const uint32_t mask = tableSize - 1u;
	for (const auto& [morton, slot] : cache.mortonToSlot) {
		uint32_t idx = mortonHash(morton) & mask;
		for (uint32_t probe = 0; probe < tableSize; ++probe) {
			CacheHashEntry& entry = table[static_cast<size_t>(idx)];
			if (entry.slotIndex == kHashTableEmptySlot) {
				entry.mortonHigh = static_cast<uint32_t>(morton >> 32u);
				entry.mortonLow = static_cast<uint32_t>(morton & 0xFFFFFFFFu);
				entry.slotIndex = slot;
				break;
			}
			idx = (idx + 1u) & mask;
		}
	}

	return table;
}

GPUResult<void> uploadCacheHashTable(BrickCache& cache) {
	if (!cache.isInitialized()) {
		return std::unexpected(GPUError::NotInitialized);
	}
	if (!cache.hashTableDirty && cache.hashTableBuffer.isValid()) {
		return {};
	}

	const std::vector<CacheHashEntry> table = buildCacheHashTable(cache);
	const size_t bytes = table.size() * sizeof(CacheHashEntry);

	if (!cache.hashTableBuffer.isValid() || cache.hashTableBuffer.sizeBytes != bytes) {
		if (cache.hashTableBuffer.isValid()) {
			glDeleteBuffers(1, &cache.hashTableBuffer.id);
			cache.hashTableBuffer.invalidate();
		}

		glCreateBuffers(1, &cache.hashTableBuffer.id);
		if (cache.hashTableBuffer.id == 0) {
			return std::unexpected(GPUError::AllocationFailed);
		}

		glNamedBufferStorage(cache.hashTableBuffer.id, static_cast<GLsizeiptr>(bytes), table.data(), GL_DYNAMIC_STORAGE_BIT);
		if (!glOk()) {
			glDeleteBuffers(1, &cache.hashTableBuffer.id);
			cache.hashTableBuffer.invalidate();
			return std::unexpected(GPUError::AllocationFailed);
		}
		cache.hashTableBuffer.sizeBytes = bytes;
	} else {
		glNamedBufferSubData(cache.hashTableBuffer.id, 0, static_cast<GLsizeiptr>(bytes), table.data());
		if (!glOk()) {
			return std::unexpected(GPUError::UploadFailed);
		}
	}

	cache.hashTableCapacity = static_cast<uint32_t>(table.size());
	cache.hashTableDirty = false;
	return {};
}

}  // namespace vqvdb
