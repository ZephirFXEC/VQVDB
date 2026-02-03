/*
 * Copyright (c) 2025, Enzo Crema
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Data-oriented types for VQVDB GPU rendering.
 * These types are designed for cache efficiency and GPU upload.
 */

#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include <glm/glm.hpp>

namespace vqvdb {

// ============================================================================
// Constants
// ============================================================================

/// Block dimensions in voxels (8x8x8 = 512 voxels per decoded block)
inline constexpr int32_t kBlockSize = 8;

/// Latent dimensions per block (4x4x4 = 64 indices per block)
inline constexpr int32_t kLatentSize = 4;

/// Number of indices per block (4³)
inline constexpr size_t kIndicesPerBlock = kLatentSize * kLatentSize * kLatentSize;

/// Default codebook dimensions
inline constexpr int32_t kDefaultNumEmbeddings = 256;
inline constexpr int32_t kDefaultEmbeddingDim = 128;

// ============================================================================
// Block Origin (replaces openvdb::Coord)
// ============================================================================

/// 3D integer coordinate for block origin in voxel space
struct BlockOrigin {
	int32_t x{0};
	int32_t y{0};
	int32_t z{0};

	[[nodiscard]] constexpr bool operator==(const BlockOrigin& other) const noexcept {
		return x == other.x && y == other.y && z == other.z;
	}

	/// Convert to glm::ivec3 for convenience
	[[nodiscard]] constexpr glm::ivec3 toIVec3() const noexcept { return {x, y, z}; }

	/// Convert to glm::vec3 for world-space calculations
	[[nodiscard]] constexpr glm::vec3 toVec3() const noexcept {
		return {static_cast<float>(x), static_cast<float>(y), static_cast<float>(z)};
	}
};

static_assert(sizeof(BlockOrigin) == 12, "BlockOrigin must be 12 bytes for file compatibility");

// ============================================================================
// Grid Transform
// ============================================================================

/// 4x4 transformation matrix (index space to world space)
/// Stored in column-major order (OpenGL/GLM convention)
struct GridTransform {
	std::array<float, 16> data{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};

	/// Convert to glm::mat4 for rendering
	[[nodiscard]] glm::mat4 toMat4() const noexcept;

	/// Get voxel size (scale component)
	[[nodiscard]] float voxelSize() const noexcept;

	/// Transform point from index space to world space
	[[nodiscard]] glm::vec3 indexToWorld(const glm::vec3& indexPos) const noexcept;
};

// ============================================================================
// Axis-Aligned Bounding Box
// ============================================================================

/// World-space bounding box
struct AABB {
	glm::vec3 min{0.0f};
	glm::vec3 max{0.0f};

	[[nodiscard]] constexpr glm::vec3 center() const noexcept { return (min + max) * 0.5f; }

	[[nodiscard]] constexpr glm::vec3 size() const noexcept { return max - min; }

	[[nodiscard]] constexpr glm::vec3 extents() const noexcept { return size() * 0.5f; }

	[[nodiscard]] constexpr bool isValid() const noexcept {
		return min.x <= max.x && min.y <= max.y && min.z <= max.z;
	}

	/// Expand to include a point
	void expand(const glm::vec3& point) noexcept;

	/// Expand to include another AABB
	void expand(const AABB& other) noexcept;
};

// ============================================================================
// Grid Metadata (per-grid header information)
// ============================================================================

/// Metadata for a single grid within a VQVDB file
struct GridMetadata {
	std::string name;
	GridTransform transform;
	AABB worldBounds;

	/// Total number of 8³ blocks in this grid
	uint32_t totalBlocks{0};

	/// Latent shape (typically {4, 4, 4})
	std::array<int32_t, 3> latentShape{kLatentSize, kLatentSize, kLatentSize};
};

// ============================================================================
// Block Data - SoA (Structure of Arrays) layout for GPU efficiency
// ============================================================================

/// Container for all block data in Structure-of-Arrays layout.
/// This layout is optimized for:
/// - GPU buffer uploads (contiguous memory)
/// - Cache-friendly iteration
/// - Easy SIMD processing
struct BlockData {
	/// Block origins in voxel space (N blocks)
	std::vector<BlockOrigin> origins;

	/// Packed block indices (N * 64 uint8 values)
	/// Layout: [block0_indices(64)] [block1_indices(64)] ...
	std::vector<uint8_t> indices;

	/// Number of blocks
	[[nodiscard]] size_t count() const noexcept { return origins.size(); }

	/// Check if data is empty
	[[nodiscard]] bool empty() const noexcept { return origins.empty(); }

	/// Get indices for a specific block
	[[nodiscard]] std::span<const uint8_t> blockIndices(size_t blockIdx) const noexcept {
		const size_t offset = blockIdx * kIndicesPerBlock;
		return {indices.data() + offset, kIndicesPerBlock};
	}

	/// Reserve memory for expected number of blocks
	void reserve(size_t numBlocks) {
		origins.reserve(numBlocks);
		indices.reserve(numBlocks * kIndicesPerBlock);
	}

	/// Clear all data
	void clear() noexcept {
		origins.clear();
		indices.clear();
	}

	/// Compute the axis-aligned bounding box in index space
	[[nodiscard]] AABB computeIndexBounds() const noexcept;

	/// Compute the axis-aligned bounding box in world space
	[[nodiscard]] AABB computeWorldBounds(const GridTransform& transform) const noexcept;
};

// ============================================================================
// Codebook (VQ Embedding Table)
// ============================================================================

/// VQ Codebook: maps indices → embedding vectors
/// Shape: [numEmbeddings, embeddingDim] (e.g., 256 × 128)
struct Codebook {
	std::vector<float> data;
	int32_t numEmbeddings{0};
	int32_t embeddingDim{0};

	[[nodiscard]] bool empty() const noexcept { return data.empty(); }

	[[nodiscard]] size_t sizeBytes() const noexcept { return data.size() * sizeof(float); }

	/// Get embedding vector for a given index
	[[nodiscard]] std::span<const float> embedding(uint8_t index) const noexcept {
		const size_t offset = static_cast<size_t>(index) * embeddingDim;
		return {data.data() + offset, static_cast<size_t>(embeddingDim)};
	}
};

// ============================================================================
// VQVDBGrid - Complete grid data ready for GPU
// ============================================================================

/// A single decoded grid from a VQVDB file.
/// Contains all data needed for GPU-based rendering.
struct VQVDBGrid {
	GridMetadata metadata;
	BlockData blocks;
};

// ============================================================================
// VQVDBFile - Complete file data
// ============================================================================

/// Complete VQVDB file contents.
/// Supports multiple grids sharing a common codebook.
struct VQVDBFile {
	/// File format version
	uint8_t version{0};

	/// Shared codebook for all grids (embedded in file or loaded separately)
	Codebook codebook;

	/// All grids in the file
	std::vector<VQVDBGrid> grids;

	[[nodiscard]] bool empty() const noexcept { return grids.empty(); }

	[[nodiscard]] size_t gridCount() const noexcept { return grids.size(); }

	/// Total number of blocks across all grids
	[[nodiscard]] size_t totalBlockCount() const noexcept;

	/// Get combined world bounds of all grids
	[[nodiscard]] AABB combinedWorldBounds() const noexcept;

	/// Clear all data
	void clear() noexcept {
		version = 0;
		codebook = {};
		grids.clear();
	}
};

// ============================================================================
// Statistics for debugging/profiling
// ============================================================================

/// Statistics about a loaded VQVDB file
struct VQVDBStats {
	size_t totalBlocks{0};
	size_t totalGrids{0};
	size_t indexDataBytes{0};
	size_t codebookBytes{0};
	AABB worldBounds;
	float voxelSize{1.0f};

	/// Print statistics to stdout
	void print() const;
};

/// Compute statistics for a VQVDBFile
[[nodiscard]] VQVDBStats computeStats(const VQVDBFile& file);

}  // namespace vqvdb
