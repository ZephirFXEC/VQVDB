// VQVAECodec.cpp
//
// Created by zphrfx on 23/06/2025.
//

#include "VQVAECodec.hpp"

#include <openvdb/tools/GridOperators.h>
#include <torch/cuda.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>

// Constants for VDB leaf nodes. A leaf is a dense 8x8x8 grid of voxels.
constexpr uint8_t LEAF_DIM = 8;
constexpr uint16_t LEAF_VOXELS = LEAF_DIM * LEAF_DIM * LEAF_DIM;  // 512

// Helper to write a simple type to a binary stream (assumes little-endian)
template <typename T>
void write_binary(std::ostream& stream, const T& value) {
	stream.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

// Helper to read a simple type from a binary stream
template <typename T>
void read_binary(std::istream& stream, T& value) {
	stream.read(reinterpret_cast<char*>(&value), sizeof(T));
}

// =========================================================================================
// Helper Streamer for Reading VDB Leaf Blocks (for Compression)
// =========================================================================================
class VDBInputBlockStreamer {
   public:
	explicit VDBInputBlockStreamer(const openvdb::tree::LeafManager<openvdb::FloatTree>& leafManager)
	    : leafManager_(leafManager), currentPos_(0), totalLeaves_(leafManager.leafCount()) {}

	[[nodiscard]] bool hasNext() const noexcept { return currentPos_ < totalLeaves_; }

	torch::Tensor nextBatch(size_t maxBatch) {
		if (!hasNext()) return torch::empty({0});

		const size_t start = currentPos_;
		const size_t end = std::min(start + maxBatch, totalLeaves_);
		currentPos_ = end;
		const size_t B = end - start;

		// Use pinned memory for asynchronous H2D copy later
		auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU).pinned_memory(true);
		torch::Tensor batch = torch::empty({static_cast<long>(B), LEAF_DIM, LEAF_DIM, LEAF_DIM}, opts);

		float* dstBase = batch.data_ptr<float>();
		for (size_t i = 0; i < B; ++i) {
			// The order is implicitly guaranteed by iterating sequentially from index 0
			const auto& leaf = leafManager_.leaf(start + i);
			const float* src = leaf.buffer().data();
			float* dst = dstBase + i * LEAF_VOXELS;
			std::memcpy(dst, src, LEAF_VOXELS * sizeof(float));
		}

		return batch.unsqueeze(1);  // Add channel dimension: [B, 1, D, D, D]
	}

   private:
	const openvdb::tree::LeafManager<openvdb::FloatTree>& leafManager_;
	size_t currentPos_;
	const size_t totalLeaves_;
};


class EncodedBlockStreamer {
   public:
	explicit EncodedBlockStreamer(const std::string& inPath) : fileStream_(inPath, std::ios::binary), currentBlockIndex_(0) {
		if (!fileStream_) {
			throw std::runtime_error("Cannot open input file " + inPath);
		}

		// --- Read Header ---
		char magic[6] = {0};
		fileStream_.read(magic, 5);
		if (std::string(magic) != "VQVDB") {
			throw std::runtime_error("Invalid VQVDB magic number.");
		}

		uint8_t version;
		read_binary(fileStream_, version);  // Version not used yet, but good practice

		read_binary(fileStream_, numEmbeddings_);

		uint8_t latentDimCount;
		read_binary(fileStream_, latentDimCount);
		latentShape_.resize(latentDimCount);
		blockDataSize_ = 1;
		for (uint8_t i = 0; i < latentDimCount; ++i) {
			uint16_t dim;
			read_binary(fileStream_, dim);
			latentShape_[i] = dim;
			blockDataSize_ *= dim;
		}

		uint32_t numOrigins;
		read_binary(fileStream_, numOrigins);
		origins_.resize(numOrigins);
		for (uint32_t i = 0; i < numOrigins; ++i) {
			read_binary(fileStream_, origins_[i].x());
			read_binary(fileStream_, origins_[i].y());
			read_binary(fileStream_, origins_[i].z());
		}

		uint32_t numBlocks;
		read_binary(fileStream_, numBlocks);
		if (numBlocks != numOrigins) {
			throw std::runtime_error("Mismatch between number of origins and blocks.");
		}
		totalBlocks_ = numBlocks;

		// The rest of the file is the encoded data stream
		dataStartOffset_ = fileStream_.tellg();
	}

	[[nodiscard]] bool hasNext() const noexcept { return currentBlockIndex_ < totalBlocks_; }

	EncodedBatch nextBatch(size_t maxBatch) {
		if (!hasNext()) return {torch::empty({0}), {}};

		const size_t start = currentBlockIndex_;
		const size_t end = std::min(start + maxBatch, totalBlocks_);
		currentBlockIndex_ = end;
		const size_t B = end - start;

		// Prepare tensor for encoded data
		std::vector<int64_t> tensorShape = {static_cast<int64_t>(B)};
		tensorShape.insert(tensorShape.end(), latentShape_.begin(), latentShape_.end());
		auto opts = torch::TensorOptions().dtype(torch::kU8).device(torch::kCPU);
		torch::Tensor data = torch::empty(tensorShape, opts);

		// Read data block from file
		fileStream_.seekg(dataStartOffset_ + static_cast<std::streamoff>(start * blockDataSize_));
		fileStream_.read(reinterpret_cast<char*>(data.data_ptr<uint8_t>()), data.numel());

		// Get the corresponding origins for this batch
		std::vector<openvdb::Coord> originsBatch(origins_.begin() + start, origins_.begin() + end);

		return {data, originsBatch};
	}

   private:
	std::ifstream fileStream_;
	std::vector<openvdb::Coord> origins_;
	std::vector<int64_t> latentShape_;
	uint32_t numEmbeddings_;
	size_t totalBlocks_;
	size_t currentBlockIndex_;
	size_t blockDataSize_;
	std::streampos dataStartOffset_;
};


// =========================================================================================
// VQVAECodec Method Implementations
// =========================================================================================

VQVAECodec::VQVAECodec(const std::string& modelPath)
    : device_(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU),
      model_(torch::jit::load(modelPath)),
      encodeMethod_(model_.get_method("encode")),
      decodeMethod_(model_.get_method("decode")) {
	std::cout << "Using device: " << device_ << '\n';
	model_.eval();
	model_.to(device_, torch::kFloat32, /*non_blocking=*/false);  // Wait for model to be ready
}

void VQVAECodec::compress(const openvdb::FloatGrid::Ptr& grid, const std::string& outPath, const size_t batchSize) const {
	const auto t0 = std::chrono::high_resolution_clock::now();
	const openvdb::tree::LeafManager<openvdb::FloatTree> leafMgr(grid->tree());
	const int64_t N = leafMgr.leafCount();

	if (N == 0) {
		std::cout << "Grid has no active voxels. Nothing to compress.\n";
		return;
	}

	// --- Step 1: Collect all leaf origins. ---
	// This is required by the file format, which stores all origins in the header.
	// The order of origins collected here defines the order of data blocks streamed later,
	// ensuring origins and data are perfectly synchronized.
	std::vector<openvdb::Coord> origins;
	origins.reserve(N);
	for (size_t i = 0; i < N; ++i) {
		origins.push_back(leafMgr.leaf(i).origin());
	}

	// --- Step 2: Open output file and write header. ---
	std::ofstream out(outPath, std::ios::binary);
	if (!out) throw std::runtime_error("Cannot open output file " + outPath);

	out.write("VQVDB", 5);          // Magic number
	write_binary<uint8_t>(out, 1);  // Version

	// Write latent shape by running a dummy tensor through the encoder
	{
		torch::NoGradGuard nograd;
		auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
		torch::Tensor dummyInput = torch::randn({1, 1, LEAF_DIM, LEAF_DIM, LEAF_DIM}, opts);
		auto idx = encodeBatch(dummyInput).sizes();  // e.g., [1, H, W]

		write_binary<uint32_t>(out, 256);                                  // numEmbeddings (hardcoded for now)
		write_binary<uint8_t>(out, static_cast<uint8_t>(idx.size() - 1));  // latent dim count
		for (size_t i = 1; i < idx.size(); ++i) {                          // Write each latent dimension size
			write_binary<uint16_t>(out, static_cast<uint16_t>(idx[i]));
		}
	}

	write_binary<uint32_t>(out, static_cast<uint32_t>(N));  // # origins
	for (const auto& o : origins) {
		write_binary<int32_t>(out, o.x());
		write_binary<int32_t>(out, o.y());
		write_binary<int32_t>(out, o.z());
	}
	write_binary<uint32_t>(out, static_cast<uint32_t>(N));  // # blocks

	// --- Step 3: Stream, encode, and write data blocks. ---
	VDBInputBlockStreamer streamer(leafMgr);
	int64_t done = 0;
	const auto t1 = std::chrono::high_resolution_clock::now();
	std::cout << "Starting compression of " << N << " blocks...\n";

	while (streamer.hasNext()) {
		torch::Tensor hostTensor = streamer.nextBatch(batchSize);
		if (hostTensor.numel() == 0) break;

		torch::Tensor encodedIndices = encodeBatch(hostTensor);  // Returns uint8 on CPU

		out.write(reinterpret_cast<const char*>(encodedIndices.data_ptr<uint8_t>()), encodedIndices.numel());
		done += hostTensor.size(0);
		std::cout << "\rProcessed " << done << " / " << N << " blocks..." << std::flush;
	}
	std::cout << "\nCompression finished.\n";
	out.close();

	const auto t2 = std::chrono::high_resolution_clock::now();
	const auto setup = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
	const auto comp = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
	printf("\n-- Setup  : %lld ms\n-- Encode : %lld ms\n", setup, comp);
}


void VQVAECodec::decompress(const std::string& inPath, openvdb::FloatGrid::Ptr& grid, const size_t batchSize) const {
	const auto t0 = std::chrono::high_resolution_clock::now();

	// --- Step 1: Open file and read header/metadata. ---
	// The streamer reads the header, including all origins, upon construction.
	EncodedBlockStreamer streamer(inPath);
	std::cout << "Starting decompression...\n";

	// --- Step 2: Prepare the output grid. ---
	grid = openvdb::FloatGrid::create();
	auto accessor = grid->getAccessor();

	// --- Step 3: Stream, decode, and write data to the new grid. ---
	const auto t1 = std::chrono::high_resolution_clock::now();
	int64_t done = 0;

	while (streamer.hasNext()) {
		EncodedBatch batch = streamer.nextBatch(batchSize);
		if (batch.data.numel() == 0) break;

		torch::Tensor decodedData = decodeBatch(batch.data);  // Returns float32 on CPU

		// The key synchronization step: Pair each decoded block with its pre-read origin.
		const float* dataPtr = decodedData.data_ptr<float>();
		for (size_t i = 0; i < batch.origins.size(); ++i) {
			const openvdb::Coord& origin = batch.origins[i];

			// This creates the leaf in the tree if it doesn't exist.
			if (auto* leaf = grid->tree().touchLeaf(origin)) {
				const float* src = dataPtr + i * LEAF_VOXELS;
				std::memcpy(leaf->buffer().data(), src, LEAF_VOXELS * sizeof(float));
				leaf->setValuesOn();
			}
		}
		done += batch.origins.size();
		std::cout << "\rProcessed " << done << " blocks..." << std::flush;
	}

	std::cout << "\nDecompression finished.\n";

	const auto t2 = std::chrono::high_resolution_clock::now();
	const auto setup = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
	const auto decomp = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
	printf("\n-- Setup  : %lld ms\n-- Decode : %lld ms\n", setup, decomp);
}

torch::Tensor VQVAECodec::encodeBatch(const torch::Tensor& cpuBatch) const {
	torch::NoGradGuard nograd;
	torch::Tensor gpuTensor = cpuBatch.to(device_, /*non_blocking=*/true);
	const torch::Tensor result = encodeMethod_({gpuTensor}).toTensor();
	return result.to(torch::kCPU, torch::kU8);
}

torch::Tensor VQVAECodec::decodeBatch(const torch::Tensor& cpuBatch) const {
	torch::NoGradGuard nograd;
	torch::Tensor gpuTensor = cpuBatch.to(device_, torch::kLong, /*non_blocking=*/true);
	const torch::Tensor result = decodeMethod_({gpuTensor}).toTensor();
	return result.to(torch::kCPU, torch::kFloat32);
}