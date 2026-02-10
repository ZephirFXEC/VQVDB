/*
 * Copyright (c) 2025, Enzo Crema
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * GPU resource management for VQVDB rendering.
 * Handles codebook SSBO, index buffers, and block cache.
 */

#pragma once

#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <vector>

#include "vqvdb/vqvdb_types.hpp"

namespace vqvdb {


// ============================================================================
// GPU Buffer Handle
// ============================================================================

/// Opaque handle to a GPU buffer (SSBO)
struct GPUBuffer {
	uint32_t id{0};       ///< OpenGL buffer ID (0 = invalid)
	size_t sizeBytes{0};  ///< Allocated size in bytes

	[[nodiscard]] bool isValid() const noexcept { return id != 0; }

	/// Invalidate handle (does not delete GPU resource)
	void invalidate() noexcept {
		id = 0;
		sizeBytes = 0;
	}
};

// ============================================================================
// GPU Resources State
// ============================================================================

/// Holds all GPU resources for VQVDB rendering
struct GPUResources {
	// Codebook (Milestone 1.2)
	GPUBuffer codebookBuffer;
	uint32_t codebookNumEmbeddings{0};
	uint32_t codebookEmbeddingDim{0};

	// Block indices (Milestone 1.3)
	GPUBuffer blockIndicesBuffer;
	size_t numBlocks{0};

	// Block origins/metadata (Milestone 1.3)
	GPUBuffer blockOriginsBuffer;

	// Block metadata with morton codes (Milestone 1.3)
	GPUBuffer blockMetadataBuffer;

	// Decoder weights (Phase 4)
	GPUBuffer decoderWeightsBuffer;

	/// Check if codebook is uploaded
	[[nodiscard]] bool hasCodebook() const noexcept { return codebookBuffer.isValid(); }

	/// Check if block data is uploaded
	[[nodiscard]] bool hasBlockData() const noexcept { return blockIndicesBuffer.isValid(); }

	/// Check if block metadata (with morton codes) is uploaded
	[[nodiscard]] bool hasBlockMetadata() const noexcept { return blockMetadataBuffer.isValid(); }
};

// ============================================================================
// Verification Result
// ============================================================================

/// Result of GPU data verification
struct VerificationResult {
	bool passed{false};
	size_t testedElements{0};
	size_t mismatchCount{0};
	float maxError{0.0f};
	std::string message;
};

// ============================================================================
// Public API
// ============================================================================

/// Initialize GPU resources (call once after GL context is ready)
void initGPUResources(GPUResources& resources) noexcept;

/// Shutdown and release all GPU resources
void shutdownGPUResources(GPUResources& resources) noexcept;

/// Upload codebook data to GPU as SSBO
/// @param resources GPU resources state
/// @param codebook Codebook data loaded from file
/// @return Success or error
[[nodiscard]] GPUResult<void> uploadCodebook(GPUResources& resources, const Codebook& codebook);

/// Verify codebook data on GPU matches CPU data via readback
/// @param resources GPU resources state
/// @param codebook Original CPU codebook data for comparison
/// @return Verification result with details
[[nodiscard]] VerificationResult verifyCodebook(const GPUResources& resources, const Codebook& codebook);

/// Delete codebook from GPU
void deleteCodebook(GPUResources& resources) noexcept;

/// Upload block index data to GPU as SSBO
/// @param resources GPU resources state
/// @param blocks Block data from loaded VQVDB file
/// @return Success or error
[[nodiscard]] GPUResult<void> uploadBlockIndices(GPUResources& resources, const BlockData& blocks);

/// Upload block origin data to GPU as SSBO
/// @param resources GPU resources state
/// @param blocks Block data from loaded VQVDB file
/// @return Success or error
[[nodiscard]] GPUResult<void> uploadBlockOrigins(GPUResources& resources, const BlockData& blocks);

/// Upload block metadata (morton codes + indices) to GPU as SSBO
/// @param resources GPU resources state
/// @param blocks Block data from loaded VQVDB file
/// @return Success or error
[[nodiscard]] GPUResult<void> uploadBlockMetadata(GPUResources& resources, const BlockData& blocks);

/// Verify block index data on GPU matches CPU data
/// @param resources GPU resources state
/// @param blocks Original CPU block data for comparison
/// @return Verification result with details
[[nodiscard]] VerificationResult verifyBlockIndices(const GPUResources& resources, const BlockData& blocks);

/// Verify block metadata (morton codes) on GPU matches CPU data
/// @param resources GPU resources state
/// @param blocks Original CPU block data for comparison
/// @return Verification result with details
[[nodiscard]] VerificationResult verifyBlockMetadata(const GPUResources& resources, const BlockData& blocks);

/// Delete block data from GPU
void deleteBlockData(GPUResources& resources) noexcept;

/// Get GPU memory usage statistics
struct GPUMemoryStats {
	size_t codebookBytes{0};
	size_t blockIndicesBytes{0};
	size_t blockOriginsBytes{0};
	size_t blockMetadataBytes{0};
	size_t decoderWeightsBytes{0};
	size_t totalBytes{0};
};

[[nodiscard]] GPUMemoryStats getGPUMemoryStats(const GPUResources& resources) noexcept;

}  // namespace vqvdb
