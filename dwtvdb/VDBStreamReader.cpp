/*
 * Copyright (c) 2025, Enzo Crema
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See the LICENSE file in the project root for full license text.
 */

#include "VDBStreamReader.hpp"

#include <openvdb/tools/ValueTransformer.h>

#include "Logger.hpp"

std::vector<DenseBlock> VDBStreamReader::readFrame(const std::string& vdbPath, const std::string& gridName) const {
	logger::debug("Reading frame {}", vdbPath);

	openvdb::io::File file(vdbPath);
	file.open();

	auto baseGrid = file.readGrid(gridName);
	file.close();

	auto grid = openvdb::gridPtrCast<openvdb::FloatGrid>(baseGrid);
	if (!grid) throw std::runtime_error("Grid is not FloatGrid");

	std::vector<DenseBlock> blocks;
	const int B = m_blockSize;

	for (auto it = grid->tree().cbeginLeaf(); it; ++it) {
		// OpenVDB default leaf dimension is 8 (=B).
		static_assert(openvdb::FloatTree::LeafNodeType::DIM == 8, "Code assumes 8³ leaf nodes");
		const openvdb::Coord& leafOrigin = it->origin();

		DenseBlock blk;
		blk.origin = leafOrigin;
		blk.data = Eigen::Tensor<float, 3>(B, B, B);

		// Direct memory copy from leaf buffer to tensor
		const float* leafBuffer = it->buffer().data();
		std::memcpy(blk.data.data(), leafBuffer, B * B * B * sizeof(float));

		blocks.emplace_back(std::move(blk));
	}

	logger::info("  extracted {} leaf blocks", blocks.size());
	return blocks;
}

std::vector<std::vector<DenseBlock>> VDBStreamReader::readSequence(const std::vector<std::string>& paths,
                                                                   const std::string& gridName) const {
	std::vector<std::vector<DenseBlock>> seq;
	seq.reserve(paths.size());
	for (const auto& p : paths) seq.emplace_back(readFrame(p, gridName));
	return seq;
}