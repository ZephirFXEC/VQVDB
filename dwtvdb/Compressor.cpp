/*
* Copyright (c) 2025, Enzo Crema
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See the LICENSE file in the project root for full license text.
 */

#include "Compressor.hpp"
#include "Logger.hpp"
#include <fstream>
#include <stdexcept>

void DCTCompressor::compress(
	const VDBSequence& seq,
	const GOPLayout& layout,
	const std::string& output_path
	) {
	if (layout.empty()) {
		logger::warn("GOP layout is empty, creating empty file.");
		std::ofstream(output_path, std::ios::binary).close();
		return;
	}

	logger::info("Starting compression to {} with qstep={}", output_path, par.qstep);

	std::vector<RunMetadata> metadata_table;
	std::vector<std::vector<char>> blobs;
	std::vector<std::vector<uint8_t>> masks;

	size_t total_runs = 0;
	for (const auto& [origin, timeline] : layout.timelines) {
		for (const auto& gop_mask : timeline) {
			const auto& mask = gop_mask.presenceMask;
			if (!mask.empty()) {
				bool in_run = false;
				for (bool present : mask) {
					if (present && !in_run) {
						total_runs++;
						in_run = true;
					} else if (!present) {
						in_run = false;
					}
				}
			}
		}
	}

	logger::info("Processing {} unique leaves with {} runs...",
	             layout.totalUniqueLeaves(), total_runs);

	size_t run_idx = 0;
	for (const auto& [origin, timeline] : layout.timelines) {
		for (size_t gop_idx = 0; gop_idx < timeline.size(); ++gop_idx) {
			const auto& gop_desc = layout.gops[gop_idx];
			const auto& presence_mask = timeline[gop_idx].presenceMask;
			if (presence_mask.empty()) continue;

			int frame_cursor = 0;
			while (frame_cursor < (int)presence_mask.size()) {
				while (frame_cursor < (int)presence_mask.size() && !presence_mask[frame_cursor])
					frame_cursor++;
				if (frame_cursor == (int)presence_mask.size()) break;

				int run_start_in_gop = frame_cursor;
				int T = 0;
				while (frame_cursor < (int)presence_mask.size() && presence_mask[frame_cursor]) {
					T++;
					frame_cursor++;
				}
				if (T == 0) continue;

				Eigen::Tensor<float, 4> block(LeafDim, LeafDim, LeafDim, T);
				std::vector<uint8_t> value_masks(T * 64); // 64 bytes per frame

				for (int t = 0; t < T; ++t) {
					int frame_idx = gop_desc.startFrame + run_start_in_gop + t;
					auto grid = seq[frame_idx].grid;
					const LeafType* leaf = grid->tree().probeConstLeaf(origin);
					assert(leaf);

					// Copy voxel data
					Eigen::TensorMap<Eigen::Tensor<const float, 3>> leaf_map(leaf->buffer().data(), LeafDim, LeafDim, LeafDim);
					block.chip(t, 3) = leaf_map;

					// Copy value mask (512 bits = 64 bytes)
					const auto& mask = leaf->valueMask();
					std::memcpy(value_masks.data() + t * 64, &mask, 64);
				}

				// Transform + quantize
				std::vector<int16_t> q_coeffs = transformAndQuantize(block);
				std::vector<char> blob = zcompress(q_coeffs);

				metadata_table.push_back({
					origin,
					gop_desc.startFrame + run_start_in_gop,
					T,
					static_cast<uint32_t>(blob.size()),
					static_cast<uint32_t>(value_masks.size())
				});

				blobs.push_back(std::move(blob));
				masks.push_back(std::move(value_masks));

				if (++run_idx % 100 == 0) {
					logger::debug("...compressed run {}/{}", run_idx, total_runs);
				}
			}
		}
	}

	// Write file
	std::ofstream file(output_path, std::ios::binary);
	if (!file) throw std::runtime_error("Failed to open output file");

	FileHeader header;
	header.qstep = par.qstep;
	header.num_leaf_runs = static_cast<uint32_t>(metadata_table.size());
	file.write(reinterpret_cast<const char*>(&header), sizeof(header));

	file.write(reinterpret_cast<const char*>(metadata_table.data()),
	           metadata_table.size() * sizeof(RunMetadata));

	for (size_t i = 0; i < blobs.size(); ++i) {
		file.write(blobs[i].data(), blobs[i].size());
		file.write(reinterpret_cast<const char*>(masks[i].data()), masks[i].size());
	}

	logger::info("Compression finished. Wrote {} runs to {}",
	             metadata_table.size(), output_path);
}

VDBSequence DCTCompressor::decompress(const std::string& input_path) {
	logger::info("Starting decompression from {}", input_path);
	std::ifstream file(input_path, std::ios::binary);
	if (!file) throw std::runtime_error("Failed to open input file");

	FileHeader header;
	file.read(reinterpret_cast<char*>(&header), sizeof(header));
	if (std::string(header.magic, 4) != "V4DC") {
		throw std::runtime_error("Invalid file format");
	}
	par.qstep = header.qstep;

	std::vector<RunMetadata> metadata_table(header.num_leaf_runs);
	file.read(reinterpret_cast<char*>(metadata_table.data()),
	          header.num_leaf_runs * sizeof(RunMetadata));

	int max_frame = 0;
	for (auto& meta : metadata_table) {
		max_frame = std::max(max_frame, meta.start_frame + meta.num_frames);
	}
	VDBSequence seq(max_frame);
	for (int i = 0; i < max_frame; ++i) {
		seq[i].grid = openvdb::FloatGrid::create();
	}

	for (size_t i = 0; i < metadata_table.size(); ++i) {
		const auto& meta = metadata_table[i];

		std::vector<char> blob(meta.blob_size);
		file.read(blob.data(), meta.blob_size);

		std::vector<uint8_t> value_masks(meta.mask_bytes);
		file.read(reinterpret_cast<char*>(value_masks.data()), meta.mask_bytes);

		std::vector<int16_t> q_coeffs = zdecompress(blob);
		Eigen::Tensor<float, 4> block = dequantizeAndInverseTransform(q_coeffs, meta.num_frames);

		for (int t = 0; t < meta.num_frames; ++t) {
			int frame_idx = meta.start_frame + t;
			auto grid = seq[frame_idx].grid;
			LeafType* leaf = grid->tree().touchLeaf(meta.origin);

			// Restore mask
			std::memcpy((void*)&leaf->valueMask(), value_masks.data() + t * 64, 64);

			Eigen::TensorMap<Eigen::Tensor<float, 3>> leaf_map(leaf->buffer().data(), LeafDim, LeafDim, LeafDim);

			if (const auto& mask = leaf->valueMask(); mask.isOn()) {
				leaf_map = block.chip(t, 3);
			} else {
				for (int z = 0; z < LeafDim; ++z) {
					for (int y = 0; y < LeafDim; ++y) {
						for (int x = 0; x < LeafDim; ++x) {
							int idx = (z * LeafDim + y) * LeafDim + x;
							if (mask.isOn(idx)) {
								leaf_map(x, y, z) = block(x, y, z, t);
							}
						}
					}
				}
			}
		}

		if ((i + 1) % 100 == 0) {
			logger::debug("...reconstructed run {}/{}", i + 1, metadata_table.size());
		}
	}

	return seq;
}

using RowMajorMatXf =
Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

std::vector<int16_t> DCTCompressor::transformAndQuantize(Eigen::Tensor<float, 4>& block) {
	// 1. Perform 4D DCT in a single pass
	dct4d(block, false);

	// 2. Quantize the resulting coefficients
	std::vector<int16_t> q_coeffs(block.size());
	for (long long i = 0; i < block.size(); ++i) {
		const float scaled = block.data()[i] / par.qstep;
		q_coeffs[i] = static_cast<int16_t>(std::round(scaled));
	}

	return q_coeffs;
}

// --- NEW, SIMPLIFIED VERSION ---
Eigen::Tensor<float, 4> DCTCompressor::dequantizeAndInverseTransform(const std::vector<int16_t>& q_coeffs, int T) {
	if (q_coeffs.size() != static_cast<size_t>(T * LeafVoxelCount)) {
		throw std::runtime_error("Size mismatch during dequantization");
	}

	// 1. Create the output tensor first
	Eigen::Tensor<float, 4> block(LeafDim, LeafDim, LeafDim, T);

	// 2. Dequantize coefficients directly into the tensor
	for (size_t i = 0; i < q_coeffs.size(); ++i) {
		block.data()[i] = static_cast<float>(q_coeffs[i]) * par.qstep;
	}

	// 3. Perform inverse 4D DCT in place
	dct4d(block, true);

	return block;
}

void DCTCompressor::dct4d(Eigen::Tensor<float, 4>& block, bool inverse) {
	const size_t T = block.dimension(3);
	const pocketfft::shape_t s{8, 8, 8, T};

	const pocketfft::stride_t st = {
		static_cast<long long>(sizeof(float)),
		static_cast<long long>(sizeof(float) * s[0]),
		static_cast<long long>(sizeof(float) * s[0] * s[1]),
		static_cast<long long>(sizeof(float) * s[0] * s[1] * s[2])
	};

	const pocketfft::shape_t axes = {0, 1, 2, 3};

	// Calculate the normalization factor for the 4D DCT
	const float norm_factor = 1.0f / std::sqrt(static_cast<float>(8 * 8 * 8 * T));

	if (!inverse) {
		// Forward DCT with normalization
		pocketfft::dct<float>(s, st, st, axes, 2, block.data(), block.data(), norm_factor, true, 6);
	} else {
		// Inverse DCT with normalization
		pocketfft::dct<float>(s, st, st, axes, 3, block.data(), block.data(), norm_factor, true, 6);
	}
}