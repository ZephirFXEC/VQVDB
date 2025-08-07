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
#include <utility>
#include <vector>

#include "LeafActivityMask.hpp"
#include "VDBStreamReader.hpp"
#include "Wavelet.hpp"
#include "pocketfft_hdronly.h"
#include "zlib.h"

// Flattened sparse coefficient entry
struct SparseCoeff {
	uint32_t index; // row * cols + col
	int16_t value;
};

// Compressed data per leaf-GOP
#pragma pack(push, 1)
struct CompressedLeafGOP {
	openvdb::Coord origin;
	uint32_t gopIndex;
	uint64_t presenceMask;

	// shapes for reconstruction
	uint32_t approxRows, approxCols;
	uint32_t detailRows, detailCols;

	// quantization scales
	float quantScaleDCApprox;
	float quantScaleACApprox;
	float quantScaleDCDetail;
	float quantScaleACDetail;

	// zstd-compressed sparse blobs
	std::vector<char> compressedDCApprox;
	std::vector<char> compressedACApprox;
	std::vector<char> compressedDCDetail;
	std::vector<char> compressedACDetail;
};
#pragma pack(pop)

#pragma pack(push, 1)
struct Header {
	char magic[5] = {'V', 'D', 'B', 'G', '\0'}; // "VDBG"
	uint16_t block{};
	uint32_t gopSize{};
	uint32_t nFrames{};
	uint32_t nGOPs{};
	uint32_t nLeafGOPs{};
};
#pragma pack(pop)

class WaveletCompressor {
public:
	struct Options {
		int block = 8;
		int gopSize = 32;
		int quantBits = 16;
		Wavelet temporal_wavelet = wavelet::haar();
		float approx_threshold = 0.1f;
		float detail_threshold = 0.1f;
		std::string gridName = "density";
	};

	explicit WaveletCompressor(Options opt) : m_opt(std::move(opt)) {
	}

	// compress sequence to custom .vdbr format
	void compress(const VDBSequence& seq, const GOPLayout& layout, const std::string& outFile);

	// decompress .vdbr back to per-frame .vdb files
	void decompress(const std::string& inFile, const std::string& outDir);

private:
	Options m_opt;

	// 3D spatial DCT / IDCT
	static void dct3d(Eigen::Tensor<float, 3>& block);
	static void idct3d(Eigen::Tensor<float, 3>& block);

	// quantization helpers
	static void quantize(const Eigen::MatrixXf& coeffs, float threshold, float& out_scale, std::vector<SparseCoeff>& out_sparse);
	static void dequantize(const std::vector<SparseCoeff>& sparse_coeffs, float scale, Eigen::MatrixXf& out_matrix);
	static void dequantize(const std::vector<SparseCoeff>& sparse_coeffs, float scale, Eigen::VectorXf& out_matrix);

	// zstd compression helpers
	static std::vector<char> compressBlob(const std::vector<SparseCoeff>& data);
	static std::vector<SparseCoeff> decompressBlob(const std::vector<char>& compressed_data);

	// VDB grid insertion
	static void insertLeafBlock(openvdb::FloatGrid::Accessor& accessor, const openvdb::Coord& origin, const Eigen::Tensor<float, 3>& data);

	// I/O
	static void writeCompressedGOP(std::ostream& os, const CompressedLeafGOP& cg);
	static void readCompressedGOP(std::istream& is, CompressedLeafGOP& cg);
};


// Corrected and more efficient 3D DCT
inline void WaveletCompressor::dct3d(Eigen::Tensor<float, 3>& block) {
	// We will operate directly on the data of the input 'block'
	// Eigen::Tensor is column-major by default.
	float* data_ptr = block.data();

	const pocketfft::shape_t shape = {
		static_cast<size_t>(block.dimension(0)),
		static_cast<size_t>(block.dimension(1)),
		static_cast<size_t>(block.dimension(2))
	};

	// IMPORTANT: Calculate strides for a COLUMN-MAJOR layout
	const pocketfft::stride_t col_major_strides = {
		static_cast<long long>(sizeof(float) * 1),
		static_cast<long long>(sizeof(float) * shape[0]),
		static_cast<long long>(sizeof(float) * shape[0] * shape[1])
	};

	const pocketfft::shape_t axes = {0, 1, 2};

	// Perform an in-place, forward DCT (type 2), ortho-normalized
	// The Python `scipy.fft.dctn(..., norm='ortho')` is equivalent to this.
	pocketfft::dct<float>(shape, col_major_strides, col_major_strides, axes,
	                      /*type=*/2, data_ptr, data_ptr,
	                      /*fct=*/1.0f, /*ortho=*/true, /*nthreads=*/6);

	// No copy back is needed as the operation was in-place.
}

// Corrected and more efficient 3D IDCT
inline void WaveletCompressor::idct3d(Eigen::Tensor<float, 3>& block) {
	// We operate directly on the data of 'block'
	float* data_ptr = block.data();

	const pocketfft::shape_t shape = {
		static_cast<size_t>(block.dimension(0)),
		static_cast<size_t>(block.dimension(1)),
		static_cast<size_t>(block.dimension(2))
	};

	// Calculate strides for a COLUMN-MAJOR layout
	const pocketfft::stride_t col_major_strides = {
		static_cast<long long>(sizeof(float) * 1),
		static_cast<long long>(sizeof(float) * shape[0]),
		static_cast<long long>(sizeof(float) * shape[0] * shape[1])
	};

	const pocketfft::shape_t axes = {0, 1, 2};

	// Perform an in-place, inverse DCT (type 3), ortho-normalized
	// `ortho=true` handles the normalization automatically, matching scipy.
	pocketfft::dct<float>(shape, col_major_strides, col_major_strides, axes,
	                      /*type=*/3, data_ptr, data_ptr,
	                      /*fct=*/1.0f, /*ortho=*/true, /*nthreads=*/6);
}


inline void WaveletCompressor::quantize(const Eigen::MatrixXf& coeffs, float threshold, float& out_scale,
                                        std::vector<SparseCoeff>& out_sparse) {
	float max_abs = 0.f;
	for (int i = 0, N = coeffs.size(); i < N; ++i) {
		float val = std::abs(coeffs(i));
		if (val > threshold) max_abs = std::max(max_abs, val);
	}

	if (max_abs < 1e-9f) {
		out_scale = 0.0f; // Assign a float
		out_sparse.clear();
		return;
	}

	// Now correctly assigns the float scale without casting to int16_t
	out_scale = max_abs / 32767.f;
	float inv = 1.f / out_scale;

	out_sparse.clear();
	out_sparse.reserve(coeffs.size() / 10);
	for (int r = 0; r < coeffs.rows(); ++r) {
		for (int c = 0; c < coeffs.cols(); ++c) {
			float v = coeffs(r, c);
			if (std::abs(v) > threshold) {
				int16_t q = static_cast<int16_t>(std::round(v * inv));
				if (q) out_sparse.push_back({static_cast<uint32_t>(r * coeffs.cols() + c), q});
			}
		}
	}
}

inline void WaveletCompressor::dequantize(const std::vector<SparseCoeff>& sparse, float scale, Eigen::MatrixXf& out_matrix) {
	out_matrix.setZero();
	uint32_t cols = out_matrix.cols(); // Get the columns of the *target* matrix
	for (auto const& sc : sparse) {
		uint32_t r = sc.index / cols;
		uint32_t c = sc.index % cols;
		out_matrix(r, c) = static_cast<float>(sc.value) * scale;
	}
}

inline void WaveletCompressor::dequantize(const std::vector<SparseCoeff>& sparse, float scale, Eigen::VectorXf& out_vector) {
	out_vector.setZero();
	for (const auto& sc : sparse) {
		uint32_t index = sc.index;
		if (index < out_vector.size()) {
			out_vector[index] = static_cast<float>(sc.value) * scale;
		} else {
			throw std::out_of_range("SparseCoeff index out of bounds for dequantization.");
		}
	}
}

// Helper to convert zlib error codes to strings for better error messages
inline std::string zlibErrorToString(int errorCode) {
	switch (errorCode) {
		case Z_MEM_ERROR:
			return "Z_MEM_ERROR: Not enough memory.";
		case Z_BUF_ERROR:
			return "Z_BUF_ERROR: Output buffer not large enough.";
		case Z_DATA_ERROR:
			return "Z_DATA_ERROR: Input data corrupted or incomplete.";
		default:
			return "Unknown zlib error: " + std::to_string(errorCode);
	}
}

// Compresses data using zlib and prepends the original size to the blob.
inline std::vector<char> WaveletCompressor::compressBlob(const std::vector<SparseCoeff>& data) {
	if (data.empty()) {
		return {};
	}

	// 1. Calculate source size and the required buffer size for the compressed data.
	const uLong sourceSize = data.size() * sizeof(SparseCoeff);
	const uLongf destBound = compressBound(sourceSize);

	// 2. Create a temporary buffer for the compressed data.
	std::vector<Bytef> compressed_data(destBound);

	// 3. Compress the data. We use Z_BEST_COMPRESSION to match Python's level=9.
	uLongf destLen = destBound;
	int result = compress2(compressed_data.data(), &destLen,
	                       reinterpret_cast<const Bytef*>(data.data()), sourceSize,
	                       Z_BEST_COMPRESSION);

	if (result != Z_OK) {
		throw std::runtime_error("ZLIB compression failed: " + zlibErrorToString(result));
	}

	// 4. Create the final output blob. It will contain:
	//    [ 8 bytes for original size | compressed data... ]
	const size_t final_blob_size = sizeof(uLong) + destLen;
	std::vector<char> final_blob(final_blob_size);

	// 5. Prepend the original uncompressed size.
	uLong original_size_le = sourceSize; // Assuming little-endian, consistent with most systems.
	memcpy(final_blob.data(), &original_size_le, sizeof(uLong));

	// 6. Copy the compressed data after the size header.
	memcpy(final_blob.data() + sizeof(uLong), compressed_data.data(), destLen);

	return final_blob;
}

// Decompresses a blob that has the original size prepended to it.
inline std::vector<SparseCoeff> WaveletCompressor::decompressBlob(const std::vector<char>& comp) {
	if (comp.empty()) {
		return {};
	}

	const size_t size_header_len = sizeof(uLong);
	if (comp.size() < size_header_len) {
		throw std::runtime_error("ZLIB blob is too small to contain size header.");
	}

	// 1. Read the prepended original size from the start of the blob.
	uLongf uncompressedSize = 0;
	memcpy(&uncompressedSize, comp.data(), size_header_len);

	if (uncompressedSize == 0) {
		return {};
	}

	// 2. Create the output vector of the exact required size.
	std::vector<SparseCoeff> decompressed_data(uncompressedSize / sizeof(SparseCoeff));

	// 3. Get a pointer to the actual compressed data (which starts after the size header).
	const Bytef* source = reinterpret_cast<const Bytef*>(comp.data() + size_header_len);
	const uLong sourceLen = comp.size() - size_header_len;

	// 4. Decompress.
	uLongf destLen = uncompressedSize;
	int result = uncompress(reinterpret_cast<Bytef*>(decompressed_data.data()), &destLen,
	                        source, sourceLen);

	if (result != Z_OK) {
		throw std::runtime_error("ZLIB decompression failed: " + zlibErrorToString(result));
	}
	if (destLen != uncompressedSize) {
		throw std::runtime_error("ZLIB decompression failed: output size mismatch.");
	}

	return decompressed_data;
}