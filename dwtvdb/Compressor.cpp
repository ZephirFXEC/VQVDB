#include "Compressor.hpp"

#include <cassert>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include "CompressionBackend.hpp"
#include "FileFormat.hpp"
#include "Logger.hpp"
#include "pocketfft_hdronly.h"

using RowMajorMatXf =
Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

FileHeader DCTCompressor::makeHeader() const {
	FileHeader header;
	header.qstep = par_.qstep;
	header.codec = static_cast<uint32_t>(CompressionCodec::Zstd);
	return header;
}

DCTCompressor::EncodedRun DCTCompressor::encodeRun(
	const RunInput& in) const {
	if (in.frames.empty()) {
		throw std::runtime_error("encodeRun: empty frames");
	}
	const int T = static_cast<int>(in.frames.size());

	// Copy voxel data and masks
	Eigen::Tensor<float, 4> block(LeafDim, LeafDim, LeafDim, T);
	std::vector<uint8_t> value_masks(static_cast<size_t>(T) * 64);

	for (int t = 0; t < T; ++t) {
		const auto& grid = in.frames[t];
		const LeafType* leaf =
			grid->tree().probeConstLeaf(in.origin);
		assert(leaf && "Leaf must be present as indicated by mask");

		Eigen::TensorMap<Eigen::Tensor<const float, 3>> leaf_map(
			leaf->buffer().data(), LeafDim, LeafDim, LeafDim);
		block.chip(t, 3) = leaf_map;

		// 512-bit mask = 64 bytes
		const auto& mask = leaf->valueMask();
		std::memcpy(value_masks.data() + (t * 64), &mask, 64);
	}

	// Transform + quantize + compress
	std::vector<int16_t> q_coeffs = transformAndQuantize(block);
	std::vector<char> blob = zcompress(q_coeffs);

	EncodedRun out;
	out.meta.origin = in.origin;
	out.meta.start_frame = in.start_frame;
	out.meta.num_frames = T;
	out.meta.blob_size = static_cast<uint32_t>(blob.size());
	out.meta.mask_bytes =
		static_cast<uint32_t>(value_masks.size());
	out.blob = std::move(blob);
	out.valueMasks = std::move(value_masks);
	return out;
}

std::vector<int16_t> DCTCompressor::transformAndQuantize(
	Eigen::Tensor<float, 4>& block) const {
	dct4d(block, false);

	std::vector<int16_t> q_coeffs(block.size());
	for (long long i = 0; i < block.size(); ++i) {
		const float scaled = block.data()[i] / par_.qstep;
		q_coeffs[static_cast<size_t>(i)] =
			static_cast<int16_t>(std::round(scaled));
	}
	return q_coeffs;
}

Eigen::Tensor<float, 4> DCTCompressor::dequantizeAndInverseTransform(
	const std::vector<int16_t>& q_coeffs, int T) const {
	if (q_coeffs.size() !=
	    static_cast<size_t>(T * LeafVoxelCount)) {
		throw std::runtime_error(
			"Size mismatch during dequantization");
	}

	Eigen::Tensor<float, 4> block(LeafDim, LeafDim, LeafDim, T);

	for (size_t i = 0; i < q_coeffs.size(); ++i) {
		block.data()[i] = static_cast<float>(q_coeffs[i]) * par_.qstep;
	}

	dct4d(block, true);
	return block;
}

void DCTCompressor::dct4d(Eigen::Tensor<float, 4>& block,
                          bool inverse) {
	const size_t T = block.dimension(3);
	const pocketfft::shape_t s{8, 8, 8, T};

	const pocketfft::stride_t st = {
		static_cast<long long>(sizeof(float)),
		static_cast<long long>(sizeof(float) * s[0]),
		static_cast<long long>(sizeof(float) * s[0] * s[1]),
		static_cast<long long>(sizeof(float) * s[0] * s[1] * s[2])};

	const pocketfft::shape_t axes = {0, 1, 2, 3};
	const float norm_factor =
		1.0f / std::sqrt(static_cast<float>(8 * 8 * 8 * T));

	if (!inverse) {
		pocketfft::dct<float>(s, st, st, axes, 2, block.data(),
		                      block.data(), norm_factor, true, 6);
	} else {
		pocketfft::dct<float>(s, st, st, axes, 3, block.data(),
		                      block.data(), norm_factor, true, 6);
	}
}

VDBSequence DCTCompressor::decompress(
	const std::string& input_path) const {
	logger::info("Starting decompression from {}", input_path);

	std::ifstream file(input_path, std::ios::binary);
	if (!file) {
		throw std::runtime_error("Failed to open input file");
	}

	FileHeader header{};
	file.read(reinterpret_cast<char*>(&header), sizeof(header));
	if (std::string(header.magic, 4) != "V4DC") {
		throw std::runtime_error("Invalid file format");
	}
	if (header.version != 2) {
		throw std::runtime_error("Unsupported version");
	}
	if (header.codec !=
	    static_cast<uint32_t>(CompressionCodec::Zstd)) {
		throw std::runtime_error("Unsupported codec");
	}

	// Preserve qstep from the file
	DCTCompressor tmp({header.qstep, par_.zstdLevel});

	// First pass: scan to find max frame index
	int max_frame = 0;
	while (true) {
		RunMetadata meta{};
		file.read(reinterpret_cast<char*>(&meta), sizeof(meta));
		if (!file || file.gcount() != sizeof(meta)) break;

		// skip payload
		file.seekg(meta.blob_size + meta.mask_bytes, std::ios::cur);
		const int end_frame = meta.start_frame + meta.num_frames;
		max_frame = std::max(max_frame, end_frame);
	}

	if (max_frame <= 0) return VDBSequence(0);

	// Prepare target sequence
	VDBSequence seq(max_frame);
	for (int i = 0; i < max_frame; ++i) {
		seq[i].grid = openvdb::FloatGrid::create();
	}

	// Second pass: decode
	file.clear();
	file.seekg(sizeof(FileHeader), std::ios::beg);

	size_t run_idx = 0;
	while (true) {
		RunMetadata meta{};
		file.read(reinterpret_cast<char*>(&meta), sizeof(meta));
		if (!file || file.gcount() != sizeof(meta)) break;

		std::vector<char> blob(meta.blob_size);
		file.read(blob.data(), meta.blob_size);

		std::vector<uint8_t> value_masks(meta.mask_bytes);
		file.read(reinterpret_cast<char*>(value_masks.data()),
		          meta.mask_bytes);

		std::vector<int16_t> q_coeffs = zdecompress(blob);
		Eigen::Tensor<float, 4> block = tmp.dequantizeAndInverseTransform(
			q_coeffs, meta.num_frames);

		for (int t = 0; t < meta.num_frames; ++t) {
			const int frame_idx = meta.start_frame + t;
			auto& grid = seq[frame_idx].grid;
			using LeafType = openvdb::FloatGrid::TreeType::LeafNodeType;
			LeafType* leaf = grid->tree().touchLeaf(meta.origin);

			// restore mask
			std::memcpy((void*)&leaf->valueMask(),
			            value_masks.data() + t * 64, 64);

			Eigen::TensorMap<Eigen::Tensor<float, 3>> leaf_map(
				leaf->buffer().data(), LeafDim, LeafDim, LeafDim);

			const auto& mask = leaf->valueMask();
			if (mask.isOn()) {
				leaf_map = block.chip(t, 3);
			} else {
				for (int z = 0; z < LeafDim; ++z) {
					for (int y = 0; y < LeafDim; ++y) {
						for (int x = 0; x < LeafDim; ++x) {
							const int idx =
								(z * LeafDim + y) * LeafDim + x;
							if (mask.isOn(idx)) {
								leaf_map(x, y, z) = block(x, y, z, t);
							}
						}
					}
				}
			}
		}

		if (++run_idx % 100 == 0) {
			logger::debug("...reconstructed run {}", run_idx);
		}
	}

	logger::info("Decompression finished. Reconstructed {} frames",
	             max_frame);
	return seq;
}