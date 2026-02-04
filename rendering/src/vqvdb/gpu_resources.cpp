/*
 * Copyright (c) 2025, Enzo Crema
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "vqvdb/gpu_resources.hpp"

#include <glad/glad.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <format>

namespace vqvdb {


// ============================================================================
// Internal Helpers
// ============================================================================

namespace {

/// Create an SSBO and upload data
[[nodiscard]] GPUBuffer createSSBO(const void* data, size_t sizeBytes, GLenum usage = GL_STATIC_DRAW) {
	GPUBuffer buffer;

	glCreateBuffers(1, &buffer.id);
	if (buffer.id == 0) {
		return buffer;  // Allocation failed
	}

	glNamedBufferStorage(buffer.id, static_cast<GLsizeiptr>(sizeBytes), data,
	                     GL_MAP_READ_BIT);  // Allow readback for verification

	// Check for errors
	GLenum err = glGetError();
	if (err != GL_NO_ERROR) {
		glDeleteBuffers(1, &buffer.id);
		buffer.id = 0;
		return buffer;
	}

	buffer.sizeBytes = sizeBytes;
	return buffer;
}

/// Delete an SSBO
void deleteSSBO(GPUBuffer& buffer) noexcept {
	if (buffer.id != 0) {
		glDeleteBuffers(1, &buffer.id);
		buffer.invalidate();
	}
}

/// Readback data from an SSBO
[[nodiscard]] bool readbackSSBO(uint32_t bufferId, size_t sizeBytes, void* outData) {
	if (bufferId == 0 || outData == nullptr) {
		return false;
	}

	void* mappedPtr = glMapNamedBufferRange(bufferId, 0, static_cast<GLsizeiptr>(sizeBytes), GL_MAP_READ_BIT);
	if (mappedPtr == nullptr) {
		return false;
	}

	std::memcpy(outData, mappedPtr, sizeBytes);
	glUnmapNamedBuffer(bufferId);

	return glGetError() == GL_NO_ERROR;
}

}  // namespace

// ============================================================================
// Public API Implementation
// ============================================================================

void initGPUResources(GPUResources& resources) noexcept {
	// Currently no global initialization needed
	// Resources are created on-demand
	resources = GPUResources{};
}

void shutdownGPUResources(GPUResources& resources) noexcept {
	deleteCodebook(resources);
	deleteBlockData(resources);

	if (resources.decoderWeightsBuffer.isValid()) {
		deleteSSBO(resources.decoderWeightsBuffer);
	}
}

// ============================================================================
// Codebook Operations
// ============================================================================

GPUResult<void> uploadCodebook(GPUResources& resources, const Codebook& codebook) {
	// Validate input
	if (!codebook.isValid()) {
		return std::unexpected(GPUError::InvalidData);
	}

	// Delete existing codebook if any
	deleteCodebook(resources);

	// Create SSBO and upload
	const size_t sizeBytes = codebook.sizeBytes();
	resources.codebookBuffer = createSSBO(codebook.data.data(), sizeBytes);

	if (!resources.codebookBuffer.isValid()) {
		return std::unexpected(GPUError::AllocationFailed);
	}

	resources.codebookNumEmbeddings = codebook.numEmbeddings;
	resources.codebookEmbeddingDim = codebook.embeddingDim;

	return {};
}

VerificationResult verifyCodebook(const GPUResources& resources, const Codebook& codebook) {
	VerificationResult result;

	// Check preconditions
	if (!resources.codebookBuffer.isValid()) {
		result.message = "No codebook uploaded to GPU";
		return result;
	}

	if (!codebook.isValid()) {
		result.message = "Invalid CPU codebook data";
		return result;
	}

	const size_t numFloats = codebook.data.size();
	const size_t sizeBytes = numFloats * sizeof(float);

	if (resources.codebookBuffer.sizeBytes != sizeBytes) {
		result.message = std::format("Size mismatch: GPU={} bytes, CPU={} bytes", resources.codebookBuffer.sizeBytes, sizeBytes);
		return result;
	}

	// Readback GPU data
	std::vector<float> gpuData(numFloats);
	if (!readbackSSBO(resources.codebookBuffer.id, sizeBytes, gpuData.data())) {
		result.message = "Failed to read back GPU data";
		return result;
	}

	// Compare data
	result.testedElements = numFloats;
	result.maxError = 0.0f;
	result.mismatchCount = 0;

	for (size_t i = 0; i < numFloats; ++i) {
		const float diff = std::abs(gpuData[i] - codebook.data[i]);
		result.maxError = std::max(result.maxError, diff);

		// Use a small tolerance for floating-point comparison
		constexpr float kEpsilon = 1e-6f;
		if (diff > kEpsilon) {
			++result.mismatchCount;
		}
	}

	result.passed = (result.mismatchCount == 0);

	if (result.passed) {
		result.message = std::format("Verification passed: {} elements, max error = {:.2e}", result.testedElements, result.maxError);
	} else {
		result.message = std::format("Verification FAILED: {}/{} mismatches, max error = {:.6f}", result.mismatchCount,
		                             result.testedElements, result.maxError);
	}

	return result;
}

void deleteCodebook(GPUResources& resources) noexcept {
	deleteSSBO(resources.codebookBuffer);
	resources.codebookNumEmbeddings = 0;
	resources.codebookEmbeddingDim = 0;
}

// ============================================================================
// Block Data Operations
// ============================================================================

GPUResult<void> uploadBlockIndices(GPUResources& resources, const BlockData& blocks) {
	if (blocks.empty()) {
		return std::unexpected(GPUError::InvalidData);
	}

	// Delete existing buffer
	if (resources.blockIndicesBuffer.isValid()) {
		deleteSSBO(resources.blockIndicesBuffer);
	}

	// Upload indices as contiguous buffer
	const size_t sizeBytes = blocks.indices.size() * sizeof(uint8_t);
	resources.blockIndicesBuffer = createSSBO(blocks.indices.data(), sizeBytes);

	if (!resources.blockIndicesBuffer.isValid()) {
		return std::unexpected(GPUError::AllocationFailed);
	}

	resources.numBlocks = blocks.count();
	return {};
}

GPUResult<void> uploadBlockOrigins(GPUResources& resources, const BlockData& blocks) {
	if (blocks.empty()) {
		return std::unexpected(GPUError::InvalidData);
	}

	// Delete existing buffer
	if (resources.blockOriginsBuffer.isValid()) {
		deleteSSBO(resources.blockOriginsBuffer);
	}

	// Convert BlockOrigin (12 bytes) to ivec4 (16 bytes) for shader compatibility
	// GLSL std430 requires vec4/ivec4 alignment for array elements
	struct PaddedOrigin {
		int32_t x, y, z, pad;
	};
	static_assert(sizeof(PaddedOrigin) == 16, "PaddedOrigin must be 16 bytes");

	std::vector<PaddedOrigin> paddedOrigins;
	paddedOrigins.reserve(blocks.origins.size());
	for (const auto& origin : blocks.origins) {
		paddedOrigins.push_back({origin.x, origin.y, origin.z, 0});
	}

	// Upload padded origins
	const size_t sizeBytes = paddedOrigins.size() * sizeof(PaddedOrigin);
	resources.blockOriginsBuffer = createSSBO(paddedOrigins.data(), sizeBytes);

	if (!resources.blockOriginsBuffer.isValid()) {
		return std::unexpected(GPUError::AllocationFailed);
	}

	return {};
}

GPUResult<void> uploadBlockMetadata(GPUResources& resources, const BlockData& blocks) {
	if (blocks.empty()) {
		return std::unexpected(GPUError::InvalidData);
	}

	// Delete existing buffer
	if (resources.blockMetadataBuffer.isValid()) {
		deleteSSBO(resources.blockMetadataBuffer);
	}

	// Build metadata array with morton codes and indices
	std::vector<BlockMetadata> metadata;
	metadata.reserve(blocks.origins.size());

	for (size_t i = 0; i < blocks.origins.size(); ++i) {
		BlockMetadata meta;
		meta.mortonCode = encodeMorton64(blocks.origins[i]);
		meta.blockIndex = static_cast<uint32_t>(i);
		meta.padding = 0;
		metadata.push_back(meta);
	}

	// Upload metadata
	const size_t sizeBytes = metadata.size() * sizeof(BlockMetadata);
	resources.blockMetadataBuffer = createSSBO(metadata.data(), sizeBytes);

	if (!resources.blockMetadataBuffer.isValid()) {
		return std::unexpected(GPUError::AllocationFailed);
	}

	return {};
}

VerificationResult verifyBlockIndices(const GPUResources& resources, const BlockData& blocks) {
	VerificationResult result;

	if (!resources.blockIndicesBuffer.isValid()) {
		result.message = "No block indices uploaded to GPU";
		return result;
	}

	if (blocks.empty()) {
		result.message = "Invalid CPU block data";
		return result;
	}

	const size_t numBytes = blocks.indices.size();

	if (resources.blockIndicesBuffer.sizeBytes != numBytes) {
		result.message = std::format("Size mismatch: GPU={} bytes, CPU={} bytes", resources.blockIndicesBuffer.sizeBytes, numBytes);
		return result;
	}

	// Readback GPU data
	std::vector<uint8_t> gpuData(numBytes);
	if (!readbackSSBO(resources.blockIndicesBuffer.id, numBytes, gpuData.data())) {
		result.message = "Failed to read back GPU data";
		return result;
	}

	// Compare data (exact match for indices)
	result.testedElements = numBytes;
	result.mismatchCount = 0;

	for (size_t i = 0; i < numBytes; ++i) {
		if (gpuData[i] != blocks.indices[i]) {
			++result.mismatchCount;
		}
	}

	result.passed = (result.mismatchCount == 0);

	if (result.passed) {
		result.message = std::format("Verification passed: {} bytes match exactly", result.testedElements);
	} else {
		result.message = std::format("Verification FAILED: {}/{} byte mismatches", result.mismatchCount, result.testedElements);
	}

	return result;
}

VerificationResult verifyBlockMetadata(const GPUResources& resources, const BlockData& blocks) {
	VerificationResult result;

	if (!resources.blockMetadataBuffer.isValid()) {
		result.message = "No block metadata uploaded to GPU";
		return result;
	}

	if (blocks.empty()) {
		result.message = "Invalid CPU block data";
		return result;
	}

	const size_t numBlocks = blocks.origins.size();
	const size_t expectedBytes = numBlocks * sizeof(BlockMetadata);

	if (resources.blockMetadataBuffer.sizeBytes != expectedBytes) {
		result.message = std::format("Size mismatch: GPU={} bytes, expected={} bytes", resources.blockMetadataBuffer.sizeBytes, expectedBytes);
		return result;
	}

	// Readback GPU data
	std::vector<BlockMetadata> gpuData(numBlocks);
	if (!readbackSSBO(resources.blockMetadataBuffer.id, expectedBytes, gpuData.data())) {
		result.message = "Failed to read back GPU data";
		return result;
	}

	// Compare data
	result.testedElements = numBlocks;
	result.mismatchCount = 0;

	for (size_t i = 0; i < numBlocks; ++i) {
		const uint64_t expectedMorton = encodeMorton64(blocks.origins[i]);
		const uint32_t expectedIndex = static_cast<uint32_t>(i);

		if (gpuData[i].mortonCode != expectedMorton || gpuData[i].blockIndex != expectedIndex) {
			++result.mismatchCount;
		}
	}

	result.passed = (result.mismatchCount == 0);

	if (result.passed) {
		result.message = std::format("Verification passed: {} blocks with correct morton codes", result.testedElements);
	} else {
		result.message = std::format("Verification FAILED: {}/{} block metadata mismatches", result.mismatchCount, result.testedElements);
	}

	return result;
}

void deleteBlockData(GPUResources& resources) noexcept {
	deleteSSBO(resources.blockIndicesBuffer);
	deleteSSBO(resources.blockOriginsBuffer);
	deleteSSBO(resources.blockMetadataBuffer);
	resources.numBlocks = 0;
}

// ============================================================================
// Memory Statistics
// ============================================================================

GPUMemoryStats getGPUMemoryStats(const GPUResources& resources) noexcept {
	GPUMemoryStats stats;

	stats.codebookBytes = resources.codebookBuffer.sizeBytes;
	stats.blockIndicesBytes = resources.blockIndicesBuffer.sizeBytes;
	stats.blockOriginsBytes = resources.blockOriginsBuffer.sizeBytes;
	stats.blockMetadataBytes = resources.blockMetadataBuffer.sizeBytes;
	stats.decoderWeightsBytes = resources.decoderWeightsBuffer.sizeBytes;

	stats.totalBytes = stats.codebookBytes + stats.blockIndicesBytes + stats.blockOriginsBytes + stats.blockMetadataBytes + stats.decoderWeightsBytes;

	return stats;
}

}  // namespace vqvdb
