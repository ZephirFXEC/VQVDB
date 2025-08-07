/*
* Copyright (c) 2025, Enzo Crema
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See the LICENSE file in the project root for full license text.
 */

#pragma once

#include <string>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>

#include "LeafActivityMask.hpp" // For GOPLayout and VDBSequence
#include "VDBStreamReader.hpp"  // For VDBSequence
#include "pocketfft_hdronly.h"
#include "zlib.h"

// ====================================================================
// ZLIB HELPER FUNCTIONS
// ====================================================================

namespace {
std::string zerr(int code) {
	switch (code) {
		case Z_MEM_ERROR:
			return "Z_MEM_ERROR";
		case Z_BUF_ERROR:
			return "Z_BUF_ERROR";
		case Z_DATA_ERROR:
			return "Z_DATA_ERROR";
		default:
			return "Zlib error " + std::to_string(code);
	}
}

std::vector<char> zcompress(const std::vector<int16_t>& src) {
	if (src.empty()) return {};
	const uLong srcBytes = src.size() * sizeof(int16_t);
	uLongf bound = compressBound(srcBytes);
	std::vector<Bytef> tmp(bound);
	uLongf compBytes = bound;
	int res = compress2(tmp.data(), &compBytes, reinterpret_cast<const Bytef*>(src.data()), srcBytes, Z_BEST_COMPRESSION);
	if (res != Z_OK) throw std::runtime_error("zlib compress: " + zerr(res));
	std::vector<char> blob(sizeof(uint32_t) + compBytes);
	uint32_t szLE = static_cast<uint32_t>(srcBytes);
	memcpy(blob.data(), &szLE, sizeof(uint32_t));
	memcpy(blob.data() + sizeof(uint32_t), tmp.data(), compBytes);
	return blob;
}

std::vector<int16_t> zdecompress(const std::vector<char>& blob) {
	if (blob.size() < sizeof(uint32_t)) throw std::runtime_error("blob too small");
	uint32_t dstBytes;
	memcpy(&dstBytes, blob.data(), sizeof(uint32_t));
	if (dstBytes == 0) return {};
	std::vector<int16_t> dst(dstBytes / sizeof(int16_t));
	uLongf dstLen = dstBytes;
	const Bytef* src = reinterpret_cast<const Bytef*>(blob.data() + sizeof(uint32_t));
	const uLong srcLen = blob.size() - sizeof(uint32_t);
	int res = uncompress(reinterpret_cast<Bytef*>(dst.data()), &dstLen, src, srcLen);
	if (res != Z_OK || dstLen != dstBytes) throw std::runtime_error("zlib uncompress: " + zerr(res));
	return dst;
}
} // namespace

// ====================================================================
// FILE FORMAT STRUCTURES
// ====================================================================

#pragma pack(push, 1)
struct FileHeader {
	char magic[4] = {'V', '4', 'D', 'C'}; // VDB 4D-DCT Compressor
	float qstep{};
	uint32_t num_leaf_runs{};
};

struct RunMetadata {
	openvdb::Coord origin;
	int32_t start_frame{};
	int32_t num_frames{};
	uint32_t blob_size{};
	uint32_t mask_bytes{};
};
#pragma pack(pop)

// ====================================================================
// DCT COMPRESSOR CLASS
// ====================================================================

/**
 * @brief Compressor/decompressor for sparse VDB sequences using a 4D DCT scheme.
 *
 * The compression is data-driven, based on a `GOPLayout` which describes
 * the continuous temporal presence of each VDB leaf node. Each such "leaf run"
 * is treated as an independent 4D block (LeafDim x LeafDim x LeafDim x NumFrames).
 *
 * The compression pipeline for each leaf run is:
 * 1. Extract voxel data into a 4D tensor.
 * 2. Apply 3D DCT to each spatial slice (frame).
 * 3. Apply 1D DCT along the temporal axis for each spatial frequency.
 * 4. Quantize the resulting 4D DCT coefficients.
 * 5. Compress the quantized data using Zlib.
 */
class DCTCompressor {
public:
	struct Params {
		float qstep = 0.5f;
	};

	explicit DCTCompressor(Params p) : par(p) {
	}

	/**
	 * @brief Compresses a VDB sequence and saves it to a file.
	 *
	 * @param seq The VDB sequence to compress.
	 * @param layout The pre-computed layout of leaf runs.
	 * @param output_path The path to the output file.
	 */
	void compress(const VDBSequence& seq, const GOPLayout& layout, const std::string& output_path);

	/**
	 * @brief Decompresses a file into a VDB sequence.
	 *
	 * @param input_path The path to the compressed file.
	 * @return The reconstructed VDB sequence.
	 */
	[[nodiscard]] VDBSequence decompress(const std::string& input_path);

private:
	using LeafType = openvdb::FloatGrid::TreeType::LeafNodeType;
	static constexpr int LeafDim = LeafType::DIM;
	static constexpr int LeafVoxelCount = LeafType::SIZE;

	Params par;

	// Core transform logic for a single leaf run
	std::vector<int16_t> transformAndQuantize(Eigen::Tensor<float, 4>& block);
	Eigen::Tensor<float, 4> dequantizeAndInverseTransform(const std::vector<int16_t>& q_coeffs, int T);

	// DCT wrappers
	static void dct4d(Eigen::Tensor<float, 4>& block, bool inverse);
};