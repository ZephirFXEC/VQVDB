/*
 * Copyright (c) 2025, Enzo Crema
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See the LICENSE file in the project root for full license text.
 */

#pragma once

#include <openvdb/openvdb.h>

#include <unsupported/Eigen/CXX11/Tensor>
#include <string>
#include <vector>

struct DenseBlock {
	Eigen::Tensor<float, 3> data;  ///< blockSize³ values
	openvdb::Coord origin;        ///< index-space origin in the grid
};

class VDBStreamReader {
   public:
	explicit VDBStreamReader(int blockSize = 8) : m_blockSize(blockSize) { openvdb::initialize(); }

	/// Extract all leaf blocks of size `blockSize³` from grid `gridName`
	/// in frame `vdbPath`.
	std::vector<DenseBlock> readFrame(const std::string& vdbPath, const std::string& gridName = "density") const;

	/// Returns N frames worth of blocks + union‐of‐origins list.
	std::vector<std::vector<DenseBlock>> readSequence(const std::vector<std::string>& paths, const std::string& gridName = "density") const;

   private:
	int m_blockSize;
};