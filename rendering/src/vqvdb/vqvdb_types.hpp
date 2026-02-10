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
#include <expected>
#include <glm/glm.hpp>
#include <span>
#include <string>
#include <vector>

namespace vqvdb {

// ============================================================================
// Error Handling Infrastructure
// ============================================================================

/// Concept for error enum types that support errorToString
template <typename E>
concept ErrorEnum = std::is_enum_v<E>;

/// Generic result type alias for any error enum
template <typename T, ErrorEnum E>
using Result = std::expected<T, E>;

// ============================================================================
// Error Enums
// ============================================================================

/// Error codes for GPU operations
enum class GPUError { NotInitialized, AllocationFailed, UploadFailed, ReadbackFailed, VerificationFailed, InvalidData };

/// Error codes for file loading operations
enum class LoadError {
	FileNotFound,
	FileOpenFailed,
	InvalidMagic,
	UnsupportedVersion,
	HeaderReadFailed,
	GridMetadataReadFailed,
	BlockDataReadFailed,
	FileTruncated,
	AllocationFailed,
	InvalidData
};

/// Error codes for codebook operations
enum class CodebookError { FileNotFound, FileOpenFailed, InvalidMagic, HeaderReadFailed, DataReadFailed, InvalidDimensions, FileTruncated };

/// Error codes for decoder backend operations
enum class DecoderError {
	EngineNotLoaded,
	OnnxModelNotFound,
	EngineBuildFailed,
	EngineDeserializeFailed,
	InferenceFailed,
	CudaError,
	InteropError,
	InvalidInput
};

// ============================================================================
// Error String Conversion (constexpr for compile-time evaluation)
// ============================================================================

[[nodiscard]] constexpr const char* errorToString(GPUError error) noexcept {
	switch (error) {
		case GPUError::NotInitialized:
			return "GPU resources not initialized";
		case GPUError::AllocationFailed:
			return "GPU buffer allocation failed";
		case GPUError::UploadFailed:
			return "GPU data upload failed";
		case GPUError::ReadbackFailed:
			return "GPU data readback failed";
		case GPUError::VerificationFailed:
			return "GPU data verification failed";
		case GPUError::InvalidData:
			return "Invalid input data";
	}
	return "Unknown GPU error";
}

[[nodiscard]] constexpr const char* errorToString(LoadError error) noexcept {
	switch (error) {
		case LoadError::FileNotFound:
			return "File not found";
		case LoadError::FileOpenFailed:
			return "Failed to open file";
		case LoadError::InvalidMagic:
			return "Invalid VQVDB magic number";
		case LoadError::UnsupportedVersion:
			return "Unsupported file version";
		case LoadError::HeaderReadFailed:
			return "Failed to read file header";
		case LoadError::GridMetadataReadFailed:
			return "Failed to read grid metadata";
		case LoadError::BlockDataReadFailed:
			return "Failed to read block data";
		case LoadError::FileTruncated:
			return "File appears to be truncated";
		case LoadError::AllocationFailed:
			return "Memory allocation failed";
		case LoadError::InvalidData:
			return "Invalid data in file";
	}
	return "Unknown load error";
}

[[nodiscard]] constexpr const char* errorToString(CodebookError error) noexcept {
	switch (error) {
		case CodebookError::FileNotFound:
			return "Codebook file not found";
		case CodebookError::FileOpenFailed:
			return "Failed to open codebook file";
		case CodebookError::InvalidMagic:
			return "Invalid codebook magic number";
		case CodebookError::HeaderReadFailed:
			return "Failed to read codebook header";
		case CodebookError::DataReadFailed:
			return "Failed to read codebook data";
		case CodebookError::InvalidDimensions:
			return "Invalid codebook dimensions";
		case CodebookError::FileTruncated:
			return "Codebook file appears truncated";
	}
	return "Unknown codebook error";
}

[[nodiscard]] constexpr const char* errorToString(DecoderError error) noexcept {
	switch (error) {
		case DecoderError::EngineNotLoaded:
			return "Decoder engine not loaded";
		case DecoderError::OnnxModelNotFound:
			return "ONNX decoder model not found";
		case DecoderError::EngineBuildFailed:
			return "Failed to build decoder engine";
		case DecoderError::EngineDeserializeFailed:
			return "Failed to deserialize decoder engine";
		case DecoderError::InferenceFailed:
			return "Decoder inference failed";
		case DecoderError::CudaError:
			return "CUDA operation failed";
		case DecoderError::InteropError:
			return "CUDA/OpenGL interop failed";
		case DecoderError::InvalidInput:
			return "Invalid decoder input";
	}
	return "Unknown decoder error";
}

// ============================================================================
// Legacy Type Aliases (for backwards compatibility)
// ============================================================================

template <typename T>
using LoadResult = Result<T, LoadError>;
template <typename T>
using GPUResult = Result<T, GPUError>;
template <typename T>
using CodebookResult = Result<T, CodebookError>;
template <typename T>
using DecoderResult = Result<T, DecoderError>;


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
// Morton Code Encoding (for block spatial indexing)
// ============================================================================

/// Encode 3D integer coordinates into a 64-bit Morton code (Z-order curve)
/// This provides cache-efficient spatial ordering for GPU block lookup
[[nodiscard]] constexpr uint64_t encodeMorton64(int32_t x, int32_t y, int32_t z) noexcept {
	// Convert signed to unsigned with offset (handle negative coords)
	// Shift by 2^21 to handle coordinates in range [-2^21, 2^21)
	constexpr uint32_t offset = 1u << 21;
	const uint32_t ux = static_cast<uint32_t>(x + static_cast<int32_t>(offset));
	const uint32_t uy = static_cast<uint32_t>(y + static_cast<int32_t>(offset));
	const uint32_t uz = static_cast<uint32_t>(z + static_cast<int32_t>(offset));

	// Helper lambda to expand bits with 2-bit gaps: 0b...xyz -> 0b...x00y00z00
	auto expandBits = [](uint32_t v) -> uint64_t {
		uint64_t x = v & 0x1FFFFFu;  // Only use 21 bits
		x = (x | (x << 32)) & 0x1F00000000FFFFull;
		x = (x | (x << 16)) & 0x1F0000FF0000FFull;
		x = (x | (x << 8)) & 0x100F00F00F00F00Full;
		x = (x | (x << 4)) & 0x10C30C30C30C30C3ull;
		x = (x | (x << 2)) & 0x1249249249249249ull;
		return x;
	};

	return expandBits(ux) | (expandBits(uy) << 1) | (expandBits(uz) << 2);
}

/// Encode a BlockOrigin into a 64-bit Morton code
[[nodiscard]] constexpr uint64_t encodeMorton64(const BlockOrigin& origin) noexcept { return encodeMorton64(origin.x, origin.y, origin.z); }

/// Block metadata for GPU spatial lookup
/// Stores morton code and index into the block data buffer
struct BlockMetadata {
	uint64_t mortonCode{0};  ///< Morton code for spatial ordering/lookup
	uint32_t blockIndex{0};  ///< Index into block indices buffer
	uint32_t padding{0};     ///< Padding for 16-byte alignment (std430)
};

static_assert(sizeof(BlockMetadata) == 16, "BlockMetadata must be 16 bytes for GPU alignment");

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

	[[nodiscard]] constexpr bool isValid() const noexcept { return min.x <= max.x && min.y <= max.y && min.z <= max.z; }

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

	/// Pre-computed morton codes (N entries, one per block).
	/// Populated once at load time by computeMortonCodes() to avoid
	/// per-frame recomputation in the scheduler.
	std::vector<uint64_t> mortonCodes;

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
		mortonCodes.reserve(numBlocks);
	}

	/// Clear all data
	void clear() noexcept {
		origins.clear();
		indices.clear();
		mortonCodes.clear();
	}

	/// Compute and cache morton codes from origins. Call once after loading.
	void computeMortonCodes() {
		mortonCodes.resize(origins.size());
		for (size_t i = 0; i < origins.size(); ++i) {
			mortonCodes[i] = encodeMorton64(origins[i]);
		}
	}

	/// Get the morton code for a block (falls back to computing if not cached).
	[[nodiscard]] uint64_t mortonCode(size_t blockIdx) const noexcept {
		if (blockIdx < mortonCodes.size()) {
			return mortonCodes[blockIdx];
		}
		return encodeMorton64(origins[blockIdx]);
	}

	/// Compute the axis-aligned bounding box in index space
	[[nodiscard]] AABB computeIndexBounds() const noexcept;

	/// Compute the axis-aligned bounding box in world space
	[[nodiscard]] AABB computeWorldBounds(const GridTransform& transform) const noexcept;
};


/// VQ Codebook: maps indices → embedding vectors
/// Shape: [numEmbeddings, embeddingDim] (e.g., 256 × 128)
/// CPU-side codebook data loaded from file
struct Codebook {
	std::vector<float> data;  ///< Row-major: [numEmbeddings][embeddingDim]
	uint32_t numEmbeddings{0};
	uint32_t embeddingDim{0};

	/// Check if codebook contains valid data
	[[nodiscard]] bool isValid() const noexcept {
		return numEmbeddings > 0 && embeddingDim > 0 && data.size() == static_cast<size_t>(numEmbeddings) * embeddingDim;
	}

	/// Get a pointer to a specific embedding vector
	[[nodiscard]] const float* embedding(uint32_t index) const noexcept { return data.data() + static_cast<size_t>(index) * embeddingDim; }

	/// Get a span view of a specific embedding vector
	[[nodiscard]] std::span<const float> embeddingSpan(uint32_t index) const noexcept { return {embedding(index), embeddingDim}; }

	/// Total size in bytes
	[[nodiscard]] size_t sizeBytes() const noexcept { return data.size() * sizeof(float); }

	[[nodiscard]] bool empty() const noexcept { return numEmbeddings == 0 || embeddingDim == 0; }

	/// Clear all data
	void clear() noexcept {
		data.clear();
		numEmbeddings = 0;
		embeddingDim = 0;
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
};

/// Compute statistics for a VQVDBFile
[[nodiscard]] VQVDBStats computeStats(const VQVDBFile& file);

}  // namespace vqvdb
