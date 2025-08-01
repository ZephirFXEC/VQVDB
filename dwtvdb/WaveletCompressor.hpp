/*
 * Copyright (c) 2025, Enzo Crema
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See the LICENSE file in the project root for full license text.
 */

#pragma once

#include <openvdb/openvdb.h>

#include <Eigen/Dense>
#include <string>

#include "VDBStreamReader.hpp"

class WaveletCompressor {
   public:
	struct Options {
		int batchSize = 256;
		int block = 8;
		int levels = 2;
		int rank = 32;
		int quantBits = 12;
		float background = 0.0f;
		std::array<float, 3> voxelSize = {1.0, 1.0, 1.0};
		std::string gridName{"density"};
	};

	explicit WaveletCompressor(const Options& opt = {}) : m_opt(opt) { openvdb::initialize(); }

	// compress ≤-- VDB paths …→ single binary .vdbr
	void compress(const std::vector<std::string>& vdbPaths, const std::string& outFile);

	// decompress ≤-- .vdbr …→ directory with per-frame .vdb
	void decompress(const std::string& inFile, const std::string& outDir);

   private:
	// ---------------------------------------- helpers
	Eigen::VectorXf waveletForward(const Eigen::Tensor<float, 3>& block) const;
	Eigen::Tensor<float, 3> waveletInverse(const Eigen::VectorXf& coeffs, const Options& meta) const;

	void quantise(Eigen::MatrixXf& coeffTensor, float& scale) const;

	void dequantise(Eigen::MatrixXf& qTensor, float scale) const;

	// Two-stage CP (PCA + SVD) and optional small ALS polish.
	void cpCompress(const Eigen::MatrixXf& flatTensor, int rank, int numBlocks, int coeffLen, Eigen::VectorXf& out_weights,
	                Eigen::MatrixXf& out_A, Eigen::MatrixXf& out_B, Eigen::MatrixXf& out_C) const;

	Eigen::MatrixXf cpReconstruct(const Eigen::VectorXf& w, const Eigen::MatrixXf& A, const Eigen::MatrixXf& B,
	                              const Eigen::MatrixXf& C) const;

	// I/O helpers for our very simple container
	void writeContainer(const std::string& path, float scale, int coeffLen, const std::vector<openvdb::Coord>& coords,
	                    const Eigen::VectorXf& w, const Eigen::MatrixXf& A, const Eigen::MatrixXf& B, const Eigen::MatrixXf& C) const;

	void readContainer(const std::string& path, float& scale, int& blk, int& levels, int& coeffLen, int& nFrames,
	                   std::vector<openvdb::Coord>& coords, Eigen::VectorXf& w, Eigen::MatrixXf& A, Eigen::MatrixXf& B, Eigen::MatrixXf& C,
	                   Options& meta) const;

	Options m_opt;
};