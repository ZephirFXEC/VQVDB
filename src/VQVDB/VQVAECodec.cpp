// VQVAECodec.cpp
//
// Created by zphrfx on 23/06/2025.
//

#include "VQVAECodec.hpp"

#include <openvdb/tools/GridOperators.h>
#include <torch/cuda.h>

#include <chrono>
#include <iostream>
#include <vector>

#include "VQVDB_Reader.hpp"

// Constants for VDB leaf nodes. A leaf is a dense 8x8x8 grid of voxels.
constexpr uint8_t LEAF_DIM = 8;
constexpr uint16_t LEAF_VOXELS = LEAF_DIM * LEAF_DIM * LEAF_DIM;  // 512

#include "bin_model.h"

// =========================================================================================
// Helper Streamer for Reading VDB Leaf Blocks (for Compression)
// =========================================================================================
class VDBInputBlockStreamer {
   public:
	explicit VDBInputBlockStreamer(const openvdb::tree::LeafManager<openvdb::FloatTree>& leafManager)
	    : leafManager_(leafManager), currentPos_(0), totalLeaves_(leafManager.leafCount()) {}

	[[nodiscard]] bool hasNext() const noexcept { return currentPos_ < totalLeaves_; }

	std::pair<torch::Tensor, std::vector<openvdb::Coord>> nextBatch(size_t maxBatch) {
		if (!hasNext()) return {torch::empty({0}), {}};
		const size_t start = currentPos_;
		const size_t end = std::min(start + maxBatch, totalLeaves_);
		currentPos_ = end;
		const size_t B = end - start;

		// Use pinned memory for asynchronous H2D copy later
		auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU).pinned_memory(true);
		torch::Tensor batch = torch::empty({static_cast<long>(B), 1, LEAF_DIM, LEAF_DIM, LEAF_DIM}, opts);
		std::vector<openvdb::Coord> origins(B);

		float* dstBase = batch.data_ptr<float>();
		tbb::parallel_for(tbb::blocked_range<size_t>(0, B), [&](const tbb::blocked_range<size_t>& r) {
			for (size_t i = r.begin(); i != r.end(); ++i) {
				const auto& leaf = leafManager_.leaf(start + i);
				origins[i] = leaf.origin();
				const float* src = leaf.buffer().data();
				float* dst = dstBase + i * LEAF_VOXELS;
				std::memcpy(dst, src, LEAF_VOXELS * sizeof(float));
			}
		});

		return {batch, origins};  // Add channel dim: [B, 1, D, D, D]
	}

   private:
	const openvdb::tree::LeafManager<openvdb::FloatTree>& leafManager_;
	size_t currentPos_;
	const size_t totalLeaves_;
};


// =========================================================================================
// VQVAECodec Method Implementations
// =========================================================================================

// --- Implementation of the private static helper function ---
std::tuple<torch::jit::Module, torch::jit::Method, torch::jit::Method> VQVAECodec::load_embedded_model(const torch::Device& device) {
	// Create a string stream from the embedded byte array
	const std::string model_string(reinterpret_cast<const char*>(g_model_data), g_model_data_size);
	std::istringstream stream(model_string);

	torch::jit::Module module;
	try {
		// Load the model from the stream (onto CPU by default)
		module = torch::jit::load(stream);
	} catch (const c10::Error& e) {
		throw std::runtime_error("Failed to load TorchScript model from memory: " + std::string(e.what()));
	}

	// Move the loaded module to the target device
	module.to(device);
	module.eval();

	// Get the methods from the now-configured module
	torch::jit::Method encode_method = module.get_method("encode");
	torch::jit::Method decode_method = module.get_method("decode");

	std::cout << "VQVAECodec: Model successfully loaded from memory onto device: " << device << '\n';

	// Return all the constructed objects in a tuple
	return {std::move(module), std::move(encode_method), std::move(decode_method)};
}

VQVAECodec::VQVAECodec()
    : device_(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU),
      model_parts_(load_embedded_model(device_)),
      model_(std::get<0>(model_parts_)),
      encodeMethod_(std::get<1>(model_parts_)),
      decodeMethod_(std::get<2>(model_parts_)) {}

void VQVAECodec::compress(const openvdb::FloatGrid::Ptr& grid, const std::string& outPath, const size_t batchSize) const {
	const auto t0 = std::chrono::high_resolution_clock::now();
	const openvdb::tree::LeafManager<openvdb::FloatTree> leafMgr(grid->tree());
	const int64_t N = leafMgr.leafCount();

	if (N == 0) {
		std::cout << "Grid has no active voxels. Nothing to compress.\n";
		return;
	}

	// --- Step 1: Get latent shape from a dummy tensor ---
	// (This part remains the same, but is necessary for the header)
	std::vector<int64_t> latentShapeVec;
	{
		torch::NoGradGuard nograd;
		auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
		torch::Tensor dummyInput = torch::randn({1, 1, LEAF_DIM, LEAF_DIM, LEAF_DIM}, opts);
		auto idx = encodeBatch(dummyInput).sizes();  // e.g., [1, H, W]
		latentShapeVec.assign(idx.begin() + 1, idx.end());
	}

	// --- Step 2: Create the optimized writer ---
	// The writer handles the header and buffered I/O automatically.
	VDBStreamWriter writer(outPath, 256, latentShapeVec, N);  // 256 numEmbeddings hardcoded

	// --- Step 3: Stream, encode, and write data blocks ---
	VDBInputBlockStreamer streamer(leafMgr);
	int64_t done = 0;
	const auto t1 = std::chrono::high_resolution_clock::now();
	std::cout << "Starting compression of " << N << " blocks using optimized writer...\n";

	while (streamer.hasNext()) {
		auto [hostTensor, origins] = streamer.nextBatch(batchSize);
		if (hostTensor.numel() == 0) break;

		torch::Tensor encodedIndices = encodeBatch(hostTensor);  // Returns uint8 on CPU

		// The writer handles buffering and interleaving origins and data
		writer.writeBatch(encodedIndices, origins);

		done += hostTensor.size(0);
		std::cout << "\rProcessed " << done << " / " << N << " blocks..." << std::flush;
	}

	const auto t2 = std::chrono::high_resolution_clock::now();
	const auto setup = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
	const auto comp = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
	printf("\n-- Setup  : %lld ms\n-- Encode : %lld ms\n", setup, comp);
}


void VQVAECodec::decompress(const std::string& inPath, openvdb::FloatGrid::Ptr& grid, const size_t batchSize) const {
	const auto t0 = std::chrono::high_resolution_clock::now();

	// --- Step 1: Use the optimized stream reader ---
	// The reader handles the header and buffered I/O automatically.
	VDBStreamReader streamer(inPath);
	std::cout << "Starting decompression using optimized reader...\n";

	// --- Step 2: Prepare the output grid ---
	grid = openvdb::FloatGrid::create();
	auto accessor = grid->getAccessor();

	// --- Step 3: Stream, decode, and write data to the new grid ---
	const auto t1 = std::chrono::high_resolution_clock::now();
	int64_t done = 0;

	while (streamer.hasNext()) {
		EncodedBatch batch = streamer.nextBatch(batchSize);
		if (batch.data.numel() == 0) break;

		torch::Tensor decodedData = decodeBatch(batch.data);  // Returns float32 on CPU

		const float* dataPtr = decodedData.data_ptr<float>();

		for (size_t i = 0; i < batch.origins.size(); ++i) {
			const openvdb::Coord& origin = batch.origins[i];

			// This creates the leaf in the tree if it doesn't exist.
			if (auto* leaf = accessor.touchLeaf(origin)) {
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