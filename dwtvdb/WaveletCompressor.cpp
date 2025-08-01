/*
 * Copyright (c) 2025, Enzo Crema
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See the LICENSE file in the project root for full license text.
 */

#include "WaveletCompressor.hpp"

#include <Eigen/SVD>
#include <filesystem>
#include <fstream>
#include <unordered_set>
#include <unsupported/Eigen/CXX11/Tensor>

#include "IncrementalPCA.hpp"
#include "Logger.hpp"
#include "Wavelet.hpp"

static std::vector<CoeffSlice> makeSliceTable(int blockSize, int levels) {
	Eigen::Tensor<float, 3> dummy(blockSize, blockSize, blockSize);
	dummy.setZero();
	auto [arr, slices, unused] = wavedec3(dummy, "haar", levels);
	return slices;  // only 1-time cost
}

static bool rank1_svd(const Eigen::MatrixXf& matrix, float& weight, Eigen::VectorXf& u, Eigen::VectorXf& v) {
	// JacobiSVD is robust for small-to-medium matrices
	Eigen::JacobiSVD<Eigen::MatrixXf> svd(matrix, Eigen::ComputeThinU | Eigen::ComputeThinV);

	// Check if the computation was successful
	if (svd.info() != Eigen::Success) {
		return false;
	}

	weight = svd.singularValues()(0);
	u = svd.matrixU().col(0);
	v = svd.matrixV().col(0);  // In Eigen, V is returned, not Vh/V.T
	return true;
}


void insertLeafBlock(openvdb::FloatGrid::Accessor& accessor, const openvdb::Coord& origin, const Eigen::Tensor<float, 3>& data) {
	const int dimX = data.dimension(0);
	const int dimY = data.dimension(1);
	const int dimZ = data.dimension(2);

	// Epsilon for floating-point zero comparison.
	const float zero_tolerance = 1e-6f;

	// Iterate through the LOCAL coordinates of the data block
	for (int z = 0; z < dimZ; ++z) {
		for (int y = 0; y < dimY; ++y) {
			for (int x = 0; x < dimX; ++x) {
				// Get the value at the local coordinate (x, y, z)
				float value = data(x, y, z);

				// --- This is the equivalent of Python's np.nonzero() ---
				// Only write the value if it's meaningfully different from zero.
				if (std::abs(value) > zero_tolerance) {
					// Calculate the absolute world coordinate
					openvdb::Coord worldCoord = origin + openvdb::Coord(x, y, z);

					// Set the value at the world coordinate.
					// accessor.setValue() will automatically activate the leaf node and voxel.
					accessor.setValue(worldCoord, value);
				}
			}
		}
	}
}


// --------------------------- quantisation
void WaveletCompressor::quantise(Eigen::MatrixXf& M, float& scale) const {
	float maxval = M.array().abs().maxCoeff();
	float quant_range = static_cast<float>((1 << (m_opt.quantBits - 1)) - 1);

	if (maxval < 1e-9f) {
		scale = 1.0f;
	} else {
		scale = maxval / quant_range;
	}

	M = (M.array() / scale).round();
}

void WaveletCompressor::dequantise(Eigen::MatrixXf& M, float scale) const { M *= scale; }

// --------------------------- two-stage CP (very close to python)
void WaveletCompressor::cpCompress(const Eigen::MatrixXf& flatTensor, int rank, int numBlocks, int coeffLen, Eigen::VectorXf& out_weights,
                                   Eigen::MatrixXf& out_A, Eigen::MatrixXf& out_B, Eigen::MatrixXf& out_C) const {
	const int n_frames = flatTensor.rows();
	const int final_rank = rank + 1;

	// --- Stage 1: Incremental PCA ---
	logger::info("Starting Incremental PCA with rank={} and batch_size={}", rank, m_opt.batchSize);

	// Create and run the Incremental PCA simulator
	IncrementalPCA ipca(rank, m_opt.batchSize);
	ipca.fit(flatTensor);

	// Get the results from the fitted model. This part is memory-efficient.
	// The `transform` is also batched internally to produce the full `A` matrix.
	Eigen::MatrixXf A = ipca.transform(flatTensor);
	Eigen::VectorXf mean_flat = ipca.getMean();
	// The components from IPCA are V.T. We need V for the SVD stage.
	Eigen::MatrixXf combined_factors = ipca.getComponents().transpose();  // Shape: (n_blocks*coeff_len, rank)

	logger::info("Incremental PCA finished.");

	// --- Stage 2: Separate Spatial and Coefficient Factors (this part remains the same) ---
	logger::debug("Separating spatial and coefficient factors via SVD...");
	Eigen::VectorXf weights(rank);
	Eigen::MatrixXf B(numBlocks, rank);
	Eigen::MatrixXf C(coeffLen, rank);

	for (int r = 0; r < rank; ++r) {
		// BUG 1 FIX: Use a RowMajor map to ensure the matrix shape matches NumPy's reshape.
		// Eigen's default is Column-Major, which would effectively be a transpose of the Python matrix.
		Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> comp_matrix(combined_factors.col(r).data(),
		                                                                                                    numBlocks, coeffLen);

		float w_r;
		Eigen::VectorXf b_r, c_r;
		if (rank1_svd(comp_matrix, w_r, b_r, c_r)) {
			weights(r) = w_r;
			B.col(r) = b_r;
			C.col(r) = c_r;
		} else {
			logger::error("SVD failed on component {} of the PCA matrix; using zeros.", r);
			weights(r) = 0.0f;
			B.col(r).setZero();
			C.col(r).setZero();
		}
	}

	// --- Incorporate the PCA mean as an extra component (this part remains the same) ---
	logger::debug("Incorporating mean component into CP decomposition");

	// BUG 1 FIX: Also apply the RowMajor map to the mean matrix to match Python's behavior.
	Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> mean_matrix(mean_flat.data(), numBlocks,
	                                                                                                    coeffLen);

	float weight_mean;
	Eigen::VectorXf B_mean, C_mean;

	if (rank1_svd(mean_matrix, weight_mean, B_mean, C_mean)) {
		// Success
	} else {
		logger::error("SVD failed on the mean component; using zeros.");
		weight_mean = 0.0f;
		B_mean = Eigen::VectorXf::Zero(numBlocks);
		C_mean = Eigen::VectorXf::Zero(coeffLen);
	}

	Eigen::MatrixXf A_mean = Eigen::MatrixXf::Ones(n_frames, 1);

	// --- Final Assembly ---
	out_weights.resize(final_rank);
	out_A.resize(n_frames, final_rank);
	out_B.resize(numBlocks, final_rank);
	out_C.resize(coeffLen, final_rank);

	out_weights.head(rank) = weights;
	out_weights(rank) = weight_mean;

	out_A.leftCols(rank) = A;
	out_A.col(rank) = A_mean;

	out_B.leftCols(rank) = B;
	out_B.col(rank) = B_mean;

	out_C.leftCols(rank) = C;
	out_C.col(rank) = C_mean;

	logger::info("CP compression complete: rank = {}", final_rank);
}

// --------------------------- tiny CP ► tensor
Eigen::MatrixXf WaveletCompressor::cpReconstruct(const Eigen::VectorXf& w, const Eigen::MatrixXf& A, const Eigen::MatrixXf& B,
                                                 const Eigen::MatrixXf& C) const {
	const int n_frames = A.rows();
	const int n_blocks = B.rows();
	const int coeff_len = C.rows();
	const int rank = A.cols();

	Eigen::MatrixXf reconstructed_flat = Eigen::MatrixXf::Zero(n_frames, n_blocks * coeff_len);
	Eigen::RowVectorXf bc_flat(n_blocks * coeff_len);  // reuse

	for (int r = 0; r < rank; ++r) {
		const auto& b_r = B.col(r);
		const auto& c_r = C.col(r);

		// BUG 2 FIX: Correctly compute the flattened outer product of B's and C's factors.
		// The original logic was effectively flattening outer(C, B) instead of outer(B, C).
		// This loop builds the flattened row-major matrix corresponding to outer(b_r, c_r).
		// The structure is [b_r(0)*c_r^T, b_r(1)*c_r^T, ...].
		for (int i = 0; i < n_blocks; ++i) {
			bc_flat.segment(i * coeff_len, coeff_len) = c_r.transpose() * b_r(i);
		}

		reconstructed_flat.noalias() += w(r) * (A.col(r) * bc_flat);
	}

	return reconstructed_flat;
}

// --------------------------- container I/O
namespace {
#pragma pack(push, 1)
struct Header {
	char magic[5] = {'V', 'D', 'B', 'R', '\0'};
	uint16_t vmaj = 0, vmin = 2;
	uint16_t block;
	uint16_t levels;
	uint16_t rank;
	uint32_t nFrames;
};
#pragma pack(pop)
}  // namespace

void WaveletCompressor::writeContainer(const std::string& path, float scale, int coeffLen, const std::vector<openvdb::Coord>& coords,
                                       const Eigen::VectorXf& w, const Eigen::MatrixXf& A, const Eigen::MatrixXf& B,
                                       const Eigen::MatrixXf& C) const {
	std::ofstream fp(path, std::ios::binary | std::ios::trunc);
	if (!fp) throw std::runtime_error("Cannot open " + path + " for writing");

	Header hdr{};
	hdr.block = uint16_t(m_opt.block);
	hdr.levels = uint16_t(m_opt.levels);
	hdr.rank = uint16_t(w.size());
	hdr.nFrames = uint32_t(A.rows());
	fp.write(reinterpret_cast<char*>(&hdr), sizeof(hdr));

	fp.write(reinterpret_cast<char*>(&scale), sizeof(float));
	uint32_t coeffLen32 = coeffLen;
	uint32_t nBlocks = coords.size();
	fp.write(reinterpret_cast<char*>(&coeffLen32), sizeof(uint32_t));
	fp.write(reinterpret_cast<char*>(&nBlocks), sizeof(uint32_t));

	fp.write(reinterpret_cast<const char*>(coords.data()), coords.size() * sizeof(openvdb::Coord));

	fp.write(reinterpret_cast<const char*>(w.data()), w.size() * sizeof(float));
	fp.write(reinterpret_cast<const char*>(A.data()), A.size() * sizeof(float));
	fp.write(reinterpret_cast<const char*>(B.data()), B.size() * sizeof(float));
	fp.write(reinterpret_cast<const char*>(C.data()), C.size() * sizeof(float));

	fp.flush();  // be explicit
	fp.close();
}

// readContainer omitted for brevity (mirror of writeContainer)

// --------------------------- main compress() driver
void WaveletCompressor::compress(const std::vector<std::string>& vdbPaths, const std::string& outFile) {
	logger::info("Compressing {} frames → {}", vdbPaths.size(), outFile);

	VDBStreamReader reader(m_opt.block);
	auto seq = reader.readSequence(vdbPaths, m_opt.gridName);
	int F = seq.size();
	logger::debug("Sequence loaded: {} frames", F);

	// ---- Build master origin list
	std::vector<openvdb::Coord> allCoords;
	{
		std::unordered_set<openvdb::Coord, std::hash<openvdb::Coord>> set;
		for (auto& frame : seq)
			for (auto& blk : frame) set.insert(blk.origin);
		allCoords.assign(set.begin(), set.end());
		std::sort(allCoords.begin(), allCoords.end());
	}
	int B = allCoords.size();
	logger::debug("Unique blocks: {}", B);

	int coeffLen = m_opt.block * m_opt.block * m_opt.block;


	// ---- Dense tensor: (F , B*coeffLen)
	Eigen::MatrixXf coeffTensor = Eigen::MatrixXf::Zero(F, B * coeffLen);

	std::unordered_map<openvdb::Coord, int, std::hash<openvdb::Coord>> map;
	for (int i = 0; i < B; ++i) map[allCoords[i]] = i;

	for (int f = 0; f < F; ++f)
		for (auto& blk : seq[f]) {
			int bi = map[blk.origin];

			auto [arr, _, __] = wavedec3(blk.data, "haar", m_opt.levels);

			// 2) Store the flat coefficient vector
			coeffTensor.block(f, bi * coeffLen, 1, coeffLen) = arr.transpose();
		}

	// ---- Quantise
	float scale;
	quantise(coeffTensor, scale);
	logger::debug("Global quantisation scale: {}", scale);

	// ---- CP compression
	Eigen::VectorXf w;
	Eigen::MatrixXf A, Bmat, C;
	cpCompress(coeffTensor, m_opt.rank, B, coeffLen, w, A, Bmat, C);
	logger::info("Performing CP compression (rank = {})", m_opt.rank);

	writeContainer(outFile, scale, coeffLen, allCoords, w, A, Bmat, C);
	logger::info("Finished. Output size = {} kB", std::filesystem::file_size(outFile) / 1024);
}


// --------------------------- decompress() driver (summarised)
void WaveletCompressor::decompress(const std::string& inFile, const std::string& outDir) {
	logger::info("Decompressing {} -> {}", inFile, outDir);

	// 1) Read container using the new signature
	float scale;
	int blk, levels, coeffLen, nFrames;
	std::vector<openvdb::Coord> coords;
	Eigen::VectorXf w;
	Eigen::MatrixXf A, B, C;
	Options meta_from_file;  // This will be populated with file-specific metadata

	readContainer(inFile, scale, blk, levels, coeffLen, nFrames, coords, w, A, B, C, meta_from_file);
	logger::debug("Container read successfully. Frames: {}, Blocks: {}, Rank: {}", nFrames, coords.size(), w.size());
	logger::debug("Metadata from file: block_size={}, levels={}", meta_from_file.block, meta_from_file.levels);

	// 2) Reconstruct and dequantise the coefficient tensor
	logger::info("Reconstructing coefficient tensor from CP factors...");
	Eigen::MatrixXf coeffTensor = cpReconstruct(w, A, B, C);
	dequantise(coeffTensor, scale);
	logger::debug("Tensor reconstructed and dequantised.");

	std::filesystem::create_directories(outDir);
	const int nBlocks = coords.size();
	static const auto kSlices = makeSliceTable(meta_from_file.block, meta_from_file.levels);

	// 3 & 4) Inverse wavelet per block and write frames
	for (int f = 0; f < nFrames; ++f) {
		logger::info("Reconstructing frame {}/{}", f + 1, nFrames);

		openvdb::FloatGrid::Ptr grid = openvdb::FloatGrid::create();
		grid->setName(m_opt.gridName);  // Use the grid name from the compressor's initial options
		auto accessor = grid->getAccessor();

		for (int b = 0; b < nBlocks; ++b) {
			// BUG 3 FIX: Removed confusing and unused rowPtr variable. The .block() call below
			// is correct and sufficient, as it creates a contiguous copy.
			Eigen::VectorXf coeffs_vec = coeffTensor.block(f, b * coeffLen, 1, coeffLen).transpose();

			Eigen::Tensor<float, 3> blockData = waverec3(coeffs_vec, kSlices, "haar", meta_from_file.levels);

			insertLeafBlock(accessor, coords[b], blockData);
		}

		// 5) Write the reconstructed frame to a .vdb file
		std::stringstream ss;
		ss << outDir << "/frame_" << std::setw(4) << std::setfill('0') << f << ".vdb";
		openvdb::io::File outFile(ss.str());
		openvdb::GridPtrVec grids;
		grids.push_back(grid);
		outFile.write(grids);
		outFile.close();
	}

	logger::info("Decompression finished. {} frames written to {}.", nFrames, outDir);
}


void WaveletCompressor::readContainer(const std::string& path, float& scale, int& blk, int& levels, int& coeffLen, int& nFrames,
                                      std::vector<openvdb::Coord>& coords, Eigen::VectorXf& w, Eigen::MatrixXf& A, Eigen::MatrixXf& B,
                                      Eigen::MatrixXf& C, Options& meta) const {
	std::ifstream fp(path, std::ios::binary);
	if (!fp) throw std::runtime_error("Cannot open " + path + " for reading");

	Header hdr{};
	fp.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
	if (std::string(hdr.magic) != "VDBR") throw std::runtime_error("Invalid file magic");

	// Populate the output parameters from the header
	blk = hdr.block;
	levels = hdr.levels;
	nFrames = hdr.nFrames;

	// Populate the metadata struct for use in inverse transforms
	meta.block = hdr.block;
	meta.levels = hdr.levels;
	// Note: Other fields in 'meta' (like gridName) will retain their original values
	// from the WaveletCompressor instance, which is the desired behavior.

	fp.read(reinterpret_cast<char*>(&scale), sizeof(float));
	uint32_t coeffLen32, nBlocks;
	fp.read(reinterpret_cast<char*>(&coeffLen32), sizeof(uint32_t));
	fp.read(reinterpret_cast<char*>(&nBlocks), sizeof(uint32_t));
	coeffLen = coeffLen32;

	coords.resize(nBlocks);
	fp.read(reinterpret_cast<char*>(coords.data()), nBlocks * sizeof(openvdb::Coord));

	// Resize Eigen structures before reading data into them
	w.resize(hdr.rank);
	A.resize(hdr.nFrames, hdr.rank);
	B.resize(nBlocks, hdr.rank);
	C.resize(coeffLen, hdr.rank);

	fp.read(reinterpret_cast<char*>(w.data()), w.size() * sizeof(float));
	fp.read(reinterpret_cast<char*>(A.data()), A.size() * sizeof(float));
	fp.read(reinterpret_cast<char*>(B.data()), B.size() * sizeof(float));
	fp.read(reinterpret_cast<char*>(C.data()), C.size() * sizeof(float));

	fp.close();
}