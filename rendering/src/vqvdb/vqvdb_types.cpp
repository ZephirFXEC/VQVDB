/*
 * Copyright (c) 2025, Enzo Crema
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Implementation of VQVDB types.
 */

#include "vqvdb_types.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>

namespace vqvdb {

// ============================================================================
// GridTransform Implementation
// ============================================================================

glm::mat4 GridTransform::toMat4() const noexcept {
	// GLM uses column-major order, same as our storage
	glm::mat4 result;
	std::memcpy(&result[0][0], data.data(), 16 * sizeof(float));
	return result;
}

float GridTransform::voxelSize() const noexcept {
	// Extract scale from the first column (assuming uniform scale)
	const float sx = std::sqrt(data[0] * data[0] + data[1] * data[1] + data[2] * data[2]);
	return sx;
}

glm::vec3 GridTransform::indexToWorld(const glm::vec3& indexPos) const noexcept {
	const glm::mat4 mat = toMat4();
	const glm::vec4 homogeneous = mat * glm::vec4(indexPos, 1.0f);
	return glm::vec3(homogeneous);
}

// ============================================================================
// AABB Implementation
// ============================================================================

void AABB::expand(const glm::vec3& point) noexcept {
	min = glm::min(min, point);
	max = glm::max(max, point);
}

void AABB::expand(const AABB& other) noexcept {
	if (other.isValid()) {
		min = glm::min(min, other.min);
		max = glm::max(max, other.max);
	}
}

// ============================================================================
// BlockData Implementation
// ============================================================================

AABB BlockData::computeIndexBounds() const noexcept {
	if (origins.empty()) {
		return {};
	}

	constexpr float fMax = std::numeric_limits<float>::max();
	constexpr float fMin = std::numeric_limits<float>::lowest();

	AABB bounds;
	bounds.min = glm::vec3(fMax);
	bounds.max = glm::vec3(fMin);

	for (const auto& origin : origins) {
		const glm::vec3 blockMin = origin.toVec3();
		const glm::vec3 blockMax = blockMin + glm::vec3(static_cast<float>(kBlockSize));
		bounds.expand(blockMin);
		bounds.expand(blockMax);
	}

	return bounds;
}

AABB BlockData::computeWorldBounds(const GridTransform& transform) const noexcept {
	if (origins.empty()) {
		return {};
	}

	constexpr float fMax = std::numeric_limits<float>::max();
	constexpr float fMin = std::numeric_limits<float>::lowest();

	AABB bounds;
	bounds.min = glm::vec3(fMax);
	bounds.max = glm::vec3(fMin);

	// Transform all 8 corners of each block's bounding box
	for (const auto& origin : origins) {
		const glm::vec3 blockMin = origin.toVec3();
		const glm::vec3 blockMax = blockMin + glm::vec3(static_cast<float>(kBlockSize));

		// Check all 8 corners of the block
		for (int corner = 0; corner < 8; ++corner) {
			const glm::vec3 cornerPos((corner & 1) ? blockMax.x : blockMin.x, (corner & 2) ? blockMax.y : blockMin.y,
			                          (corner & 4) ? blockMax.z : blockMin.z);

			const glm::vec3 worldPos = transform.indexToWorld(cornerPos);
			bounds.expand(worldPos);
		}
	}

	return bounds;
}

// ============================================================================
// VQVDBFile Implementation
// ============================================================================

size_t VQVDBFile::totalBlockCount() const noexcept {
	size_t total = 0;
	for (const auto& grid : grids) {
		total += grid.blocks.count();
	}
	return total;
}

AABB VQVDBFile::combinedWorldBounds() const noexcept {
	if (grids.empty()) {
		return {};
	}

	constexpr float fMax = std::numeric_limits<float>::max();
	constexpr float fMin = std::numeric_limits<float>::lowest();

	AABB combined;
	combined.min = glm::vec3(fMax);
	combined.max = glm::vec3(fMin);

	for (const auto& grid : grids) {
		combined.expand(grid.metadata.worldBounds);
	}

	return combined;
}

// ============================================================================
// VQVDBStats Implementation
// ============================================================================

void VQVDBStats::print() const {
	std::cout << "\n";
	std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
	std::cout << "║                    VQVDB File Statistics                     ║\n";
	std::cout << "╠══════════════════════════════════════════════════════════════╣\n";

	std::cout << "║ Grids:           " << std::setw(10) << totalGrids << std::setw(35) << " ║\n";
	std::cout << "║ Total Blocks:    " << std::setw(10) << totalBlocks << std::setw(35) << " ║\n";

	const float indexDataKB = static_cast<float>(indexDataBytes) / 1024.0f;
	const float codebookKB = static_cast<float>(codebookBytes) / 1024.0f;

	std::cout << std::fixed << std::setprecision(2);
	std::cout << "║ Index Data:      " << std::setw(10) << indexDataKB << " KB" << std::setw(32) << " ║\n";
	std::cout << "║ Codebook:        " << std::setw(10) << codebookKB << " KB" << std::setw(32) << " ║\n";
	std::cout << "║ Voxel Size:      " << std::setw(10) << voxelSize << std::setw(35) << " ║\n";

	std::cout << "╠══════════════════════════════════════════════════════════════╣\n";
	std::cout << "║ World Bounds:                                                ║\n";

	std::cout << std::fixed << std::setprecision(3);
	std::cout << "║   Min: (" << std::setw(9) << worldBounds.min.x << ", " << std::setw(9) << worldBounds.min.y << ", " << std::setw(9)
	          << worldBounds.min.z << ")" << std::setw(17) << " ║\n";
	std::cout << "║   Max: (" << std::setw(9) << worldBounds.max.x << ", " << std::setw(9) << worldBounds.max.y << ", " << std::setw(9)
	          << worldBounds.max.z << ")" << std::setw(17) << " ║\n";

	const glm::vec3 size = worldBounds.size();
	std::cout << "║   Size: (" << std::setw(8) << size.x << ", " << std::setw(8) << size.y << ", " << std::setw(8) << size.z << ")"
	          << std::setw(17) << " ║\n";

	std::cout << "╚══════════════════════════════════════════════════════════════╝\n";
	std::cout << "\n";
}

VQVDBStats computeStats(const VQVDBFile& file) {
	VQVDBStats stats;

	stats.totalGrids = file.grids.size();
	stats.totalBlocks = file.totalBlockCount();
	stats.codebookBytes = file.codebook.sizeBytes();

	for (const auto& grid : file.grids) {
		stats.indexDataBytes += grid.blocks.indices.size();
	}

	stats.worldBounds = file.combinedWorldBounds();

	if (!file.grids.empty()) {
		stats.voxelSize = file.grids[0].metadata.transform.voxelSize();
	}

	return stats;
}

}  // namespace vqvdb
