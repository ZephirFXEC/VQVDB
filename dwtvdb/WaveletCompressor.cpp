/*
 * Copyright (c) 2025, Enzo Crema
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See the LICENSE file in the project root for full license text.
 */

#include "WaveletCompressor.hpp"

#include <filesystem>
#include <fstream>

#include "Logger.hpp"
#include <numeric>

void WaveletCompressor::insertLeafBlock(openvdb::FloatGrid::Accessor& accessor, const openvdb::Coord& origin,
                                        const Eigen::Tensor<float, 3>& data) {
	const int dimX = data.dimension(0);
	const int dimY = data.dimension(1);
	const int dimZ = data.dimension(2);

	constexpr float zero_tolerance = 1e-6f;

	for (int z = 0; z < dimZ; ++z) {
		for (int y = 0; y < dimY; ++y) {
			for (int x = 0; x < dimX; ++x) {
				float value = data(z, y, x);
				if (std::abs(value) > zero_tolerance) {
					openvdb::Coord worldCoord = origin + openvdb::Coord(x, y, z);
					accessor.setValue(worldCoord, value);
				}
			}
		}
	}
}


// NEW: I/O functions for the new struct
void WaveletCompressor::writeCompressedGOP(std::ostream& os, const CompressedLeafGOP& cg) {
	os.write(reinterpret_cast<const char*>(&cg.origin), sizeof(cg.origin));
	os.write(reinterpret_cast<const char*>(&cg.gopIndex), sizeof(cg.gopIndex));
	os.write(reinterpret_cast<const char*>(&cg.presenceMask), sizeof(cg.presenceMask));
	os.write(reinterpret_cast<const char*>(&cg.approxRows), sizeof(cg.approxRows));
	os.write(reinterpret_cast<const char*>(&cg.approxCols), sizeof(cg.approxCols));
	os.write(reinterpret_cast<const char*>(&cg.detailRows), sizeof(cg.detailRows));
	os.write(reinterpret_cast<const char*>(&cg.detailCols), sizeof(cg.detailCols));

	// Write new scales
	os.write(reinterpret_cast<const char*>(&cg.quantScaleDCApprox), sizeof(cg.quantScaleDCApprox));
	os.write(reinterpret_cast<const char*>(&cg.quantScaleACApprox), sizeof(cg.quantScaleACApprox));
	os.write(reinterpret_cast<const char*>(&cg.quantScaleDCDetail), sizeof(cg.quantScaleDCDetail));
	os.write(reinterpret_cast<const char*>(&cg.quantScaleACDetail), sizeof(cg.quantScaleACDetail));

	auto writeBlob = [&](auto const& vec) {
		uint64_t sz = vec.size();
		os.write(reinterpret_cast<const char*>(&sz), sizeof(sz));
		if (sz) os.write(vec.data(), sz);
	};

	// Write new blobs
	writeBlob(cg.compressedDCApprox);
	writeBlob(cg.compressedACApprox);
	writeBlob(cg.compressedDCDetail);
	writeBlob(cg.compressedACDetail);
}

// NEW: I/O function to read the DCT-DWT compressed GOP from a file stream.
void WaveletCompressor::readCompressedGOP(std::istream& is, CompressedLeafGOP& cg) {
	is.read(reinterpret_cast<char*>(&cg.origin), sizeof(cg.origin));
	is.read(reinterpret_cast<char*>(&cg.gopIndex), sizeof(cg.gopIndex));
	is.read(reinterpret_cast<char*>(&cg.presenceMask), sizeof(cg.presenceMask));
	is.read(reinterpret_cast<char*>(&cg.approxRows), sizeof(cg.approxRows));
	is.read(reinterpret_cast<char*>(&cg.approxCols), sizeof(cg.approxCols));
	is.read(reinterpret_cast<char*>(&cg.detailRows), sizeof(cg.detailRows));
	is.read(reinterpret_cast<char*>(&cg.detailCols), sizeof(cg.detailCols));

	// Read new scales
	is.read(reinterpret_cast<char*>(&cg.quantScaleDCApprox), sizeof(cg.quantScaleDCApprox));
	is.read(reinterpret_cast<char*>(&cg.quantScaleACApprox), sizeof(cg.quantScaleACApprox));
	is.read(reinterpret_cast<char*>(&cg.quantScaleDCDetail), sizeof(cg.quantScaleDCDetail));
	is.read(reinterpret_cast<char*>(&cg.quantScaleACDetail), sizeof(cg.quantScaleACDetail));

	auto readBlob = [&](std::vector<char>& vec) {
		uint64_t sz = 0;
		is.read(reinterpret_cast<char*>(&sz), sizeof(sz));
		vec.resize(sz);
		if (sz) {
			is.read(vec.data(), sz);
		}
	};

	// Read new blobs
	readBlob(cg.compressedDCApprox);
	readBlob(cg.compressedACApprox);
	readBlob(cg.compressedDCDetail);
	readBlob(cg.compressedACDetail);
}


void WaveletCompressor::compress(const VDBSequence& seq, const GOPLayout& layout, const std::string& outFile) {
	std::vector<CompressedLeafGOP> all_compressed_runs;
	const int coeffLen = m_opt.block * m_opt.block * m_opt.block;

	logger::info("Compressing {} unique leaves over {} total runs...", layout.totalUniqueLeaves(),
	             std::accumulate(layout.leaf_runs.begin(), layout.leaf_runs.end(), 0,
	                             [](int sum, const auto& p) { return sum + p.second.size(); }));


	// Process the sequence one continuous leaf run at a time
	for (const auto& [origin, runs] : layout.leaf_runs) {
		for (const auto& run : runs) {
			if (run.numFrames == 0) continue;

			// --- START: DCT-DWT LOGIC FOR A SINGLE LEAF RUN ---

			// Step 1: Extract leaf data for this specific run
			std::vector<const openvdb::FloatGrid::TreeType::LeafNodeType*> leaf_buffers;
			leaf_buffers.reserve(run.numFrames);
			for (int i = 0; i < run.numFrames; ++i) {
				const int frameIdx = run.startFrame + i;
				const auto& grid = seq[frameIdx].grid;
				if (const auto* leaf = grid->tree().probeConstLeaf(origin)) {
					leaf_buffers.push_back(leaf);
				} else {
					// This should not happen with the new GOPAnalyzer logic
					logger::error("Data extraction error: Leaf at [{},{},{}] not found at frame {}.", origin[0], origin[1], origin[2],
					              frameIdx);
				}
			}

			// Step 2: Apply 3D Spatial DCT to each block in the run
			Eigen::MatrixXf dct_coeff_matrix(run.numFrames, coeffLen);
			for (int i = 0; i < run.numFrames; ++i) {
				Eigen::TensorMap<const Eigen::Tensor<const float, 3>> data_tensor(leaf_buffers[i]->buffer().data(), m_opt.block,
				                                                                  m_opt.block, m_opt.block);
				Eigen::Tensor<float, 3> block = data_tensor; // Make a mutable copy
				dct3d(block); // Apply 3D DCT in-place
				dct_coeff_matrix.row(i) = Eigen::Map<Eigen::VectorXf>(block.data(), coeffLen).transpose();
			}

			// Step 3: Apply 1D Temporal DWT to each coefficient's timeline
			Eigen::VectorXf approx_out, detail_out;
			wavelet::dwt(dct_coeff_matrix.col(0), m_opt.temporal_wavelet, approx_out, detail_out, run.numFrames);

			Eigen::MatrixXf approx_coeffs(approx_out.size(), coeffLen);
			Eigen::MatrixXf detail_coeffs(detail_out.size(), coeffLen);

			for (int c = 0; c < coeffLen; ++c) {
				wavelet::dwt(dct_coeff_matrix.col(c), m_opt.temporal_wavelet, approx_out, detail_out, run.numFrames);
				approx_coeffs.col(c) = approx_out;
				detail_coeffs.col(c) = detail_out;
			}

			if (coeffLen <= 1) continue;

			// Step 4: Separate DC/AC components
			Eigen::VectorXf dc_approx = approx_coeffs.col(0);
			Eigen::MatrixXf ac_approx = approx_coeffs.rightCols(coeffLen - 1);
			Eigen::VectorXf dc_detail = detail_coeffs.col(0);
			Eigen::MatrixXf ac_detail = detail_coeffs.rightCols(coeffLen - 1);

			// Step 5: Quantize each of the four components
			CompressedLeafGOP cg;
			std::vector<SparseCoeff> sparse_dc_approx, sparse_ac_approx, sparse_dc_detail, sparse_ac_detail;
			quantize(dc_approx, m_opt.approx_threshold, cg.quantScaleDCApprox, sparse_dc_approx);
			quantize(ac_approx, m_opt.approx_threshold, cg.quantScaleACApprox, sparse_ac_approx);
			quantize(dc_detail, m_opt.detail_threshold, cg.quantScaleDCDetail, sparse_dc_detail);
			quantize(ac_detail, m_opt.detail_threshold, cg.quantScaleACDetail, sparse_ac_detail);

			// Step 6: Losslessly compress the four sparse data blobs
			cg.compressedDCApprox = compressBlob(sparse_dc_approx);
			cg.compressedACApprox = compressBlob(sparse_ac_approx);
			cg.compressedDCDetail = compressBlob(sparse_dc_detail);
			cg.compressedACDetail = compressBlob(sparse_ac_detail);

			// Populate the rest of the struct using the repurposed fields
			cg.origin = origin;
			cg.gopIndex = run.startFrame;
			cg.presenceMask = run.numFrames;
			cg.approxRows = approx_coeffs.rows();
			cg.approxCols = approx_coeffs.cols();
			cg.detailRows = detail_coeffs.rows();
			cg.detailCols = detail_coeffs.cols();

			all_compressed_runs.push_back(std::move(cg));
		}
	}

	logger::info("Compression of all leaf runs complete.");

	// Write all compressed data to the container format
	std::ofstream fp(outFile, std::ios::binary | std::ios::trunc);
	if (!fp) throw std::runtime_error("Cannot open " + outFile + " for writing");

	Header hdr{};
	hdr.block = static_cast<uint16_t>(m_opt.block);
	hdr.gopSize = static_cast<uint32_t>(m_opt.gopSize);
	hdr.nFrames = static_cast<uint32_t>(seq.size());
	hdr.nGOPs = 0; // This field is no longer meaningful in the new model
	hdr.nLeafGOPs = static_cast<uint32_t>(all_compressed_runs.size());
	fp.write(reinterpret_cast<char*>(&hdr), sizeof(hdr));

	for (const auto& cg : all_compressed_runs) {
		writeCompressedGOP(fp, cg);
	}

	fp.close();
	logger::info("Finished. Output size = {} kB", std::filesystem::file_size(outFile) / 1024);
}

void WaveletCompressor::decompress(const std::string& inFile, const std::string& outDir) {
	logger::info("Decompressing {} -> {}", inFile, outDir);

	std::ifstream fp(inFile, std::ios::binary);
	if (!fp) throw std::runtime_error("Cannot open " + inFile + " for reading");

	Header hdr{};
	fp.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
	if (std::string(hdr.magic) != "VDBG") throw std::runtime_error("Invalid file magic or format.");

	std::vector<CompressedLeafGOP> all_compressed_runs(hdr.nLeafGOPs);
	for (uint32_t i = 0; i < hdr.nLeafGOPs; ++i) {
		readCompressedGOP(fp, all_compressed_runs[i]);
	}
	fp.close();
	logger::debug("Container read: {} frames, {} compressed leaf-runs.", hdr.nFrames, hdr.nLeafGOPs);

	// Create all frame grids and accessors in memory upfront
	std::vector<openvdb::FloatGrid::Ptr> frame_grids;
	std::vector<openvdb::FloatGrid::Accessor> frame_accessors;
	for (uint32_t f = 0; f < hdr.nFrames; ++f) {
		auto grid = openvdb::FloatGrid::create();
		grid->setName(m_opt.gridName);
		frame_grids.push_back(grid);
		frame_accessors.push_back(grid->getAccessor());
	}
	std::filesystem::create_directories(outDir);


	// Decompress each run and insert its data into the appropriate frames
	logger::info("Reconstructing and populating {} frames...", hdr.nFrames);
	for (const auto& cg : all_compressed_runs) {
		const uint32_t startFrame = cg.gopIndex;
		const uint32_t numFramesInRun = cg.presenceMask;

		// Step 1: Decompress sparse data
		auto sparse_dc_approx = decompressBlob(cg.compressedDCApprox);
		auto sparse_ac_approx = decompressBlob(cg.compressedACApprox);
		auto sparse_dc_detail = decompressBlob(cg.compressedDCDetail);
		auto sparse_ac_detail = decompressBlob(cg.compressedACDetail);

		// Step 2: Dequantize to reconstruct coefficient components
		Eigen::VectorXf dc_approx_dequant(cg.approxRows);
		Eigen::MatrixXf ac_approx_dequant(cg.approxRows, cg.approxCols - 1);
		Eigen::VectorXf dc_detail_dequant(cg.detailRows);
		Eigen::MatrixXf ac_detail_dequant(cg.detailRows, cg.detailCols - 1);
		dequantize(sparse_dc_approx, cg.quantScaleDCApprox, dc_approx_dequant);
		dequantize(sparse_ac_approx, cg.quantScaleACApprox, ac_approx_dequant);
		dequantize(sparse_dc_detail, cg.quantScaleDCDetail, dc_detail_dequant);
		dequantize(sparse_ac_detail, cg.quantScaleACDetail, ac_detail_dequant);

		// Step 3: Reassemble full coefficient matrices
		Eigen::MatrixXf approx_coeffs(cg.approxRows, cg.approxCols);
		approx_coeffs.col(0) = dc_approx_dequant;
		if (cg.approxCols > 1) approx_coeffs.rightCols(cg.approxCols - 1) = ac_approx_dequant;

		Eigen::MatrixXf detail_coeffs(cg.detailRows, cg.detailCols);
		detail_coeffs.col(0) = dc_detail_dequant;
		if (cg.detailCols > 1) detail_coeffs.rightCols(cg.detailCols - 1) = ac_detail_dequant;


		// Step 4: Apply 1D Inverse Temporal DWT for each spatial frequency
		Eigen::MatrixXf dct_coeff_matrix(numFramesInRun, cg.approxCols);
		for (int c = 0; c < dct_coeff_matrix.cols(); ++c) {
			dct_coeff_matrix.col(c) = wavelet::idwt(approx_coeffs.col(c), detail_coeffs.col(c), m_opt.temporal_wavelet,
			                                        numFramesInRun);
		}

		// Step 5: Reconstruct each frame in the run
		for (uint32_t i = 0; i < numFramesInRun; ++i) {
			const int globalFrameIdx = startFrame + i;
			if (globalFrameIdx >= hdr.nFrames) continue;

			Eigen::VectorXf frame_dct_coeffs = dct_coeff_matrix.row(i);

			// Apply 3D Inverse Spatial DCT
			Eigen::Tensor<float, 3> blockData(hdr.block, hdr.block, hdr.block);
			memcpy(blockData.data(), frame_dct_coeffs.data(), frame_dct_coeffs.size() * sizeof(float));
			idct3d(blockData);

			// Insert the reconstructed block into the correct global frame's grid
			insertLeafBlock(frame_accessors[globalFrameIdx], cg.origin, blockData);
		}
	}

	// Write all completed frames to .vdb files.
	logger::info("Writing {} VDB files...", hdr.nFrames);
	for (uint32_t f = 0; f < hdr.nFrames; ++f) {
		std::stringstream ss;
		ss << outDir << "/frame_" << std::setw(4) << std::setfill('0') << f << ".vdb";
		openvdb::io::File outFile(ss.str());
		openvdb::GridPtrVec grids_to_write;
		grids_to_write.push_back(frame_grids[f]);
		outFile.write(grids_to_write);
		outFile.close();
	}

	logger::info("Decompression finished. {} frames written to {}.", hdr.nFrames, outDir);
}