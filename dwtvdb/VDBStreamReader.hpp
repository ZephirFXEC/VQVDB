/*
 * Copyright (c) 2025, Enzo Crema
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See the LICENSE file in the project root for full license text.
 */

#pragma once

#include <openvdb/openvdb.h>

#include <string>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>

struct DenseBlock {
	openvdb::Coord origin;
	Eigen::Tensor<float, 3> data;
};

struct VDBFrame {
	std::vector<DenseBlock> blocks;
	openvdb::FloatGrid::ConstPtr grid;
};


class VDBSequence {
   public:
	explicit VDBSequence(std::vector<VDBFrame> frames) : m_frames(std::move(frames)) {}
	[[nodiscard]] const std::vector<VDBFrame>& frames() const noexcept { return m_frames; }
	[[nodiscard]] size_t size() const noexcept { return m_frames.size(); }
	[[nodiscard]] const VDBFrame& operator[](size_t i) const { return m_frames[i]; }

   private:
	std::vector<VDBFrame> m_frames;
};


class VDBLoader {
   public:
	explicit VDBLoader() = default;

	[[nodiscard]] VDBFrame loadFrame(const std::string& filePath, const std::string& gridName) const;

	[[nodiscard]] VDBSequence loadSequence(const std::vector<std::string>& filePaths, const std::string& gridName) const;
};