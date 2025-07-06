// VQVAECodec.cpp
//
// Created by zphrfx on 23/06/2025.
//

#include "VQVAECodec.hpp"

#include <openvdb/tools/GridOperators.h>

#include <chrono>
#include <iostream>
#include <vector>

#include "VQVDB_Reader.hpp"

// Constants for VDB leaf nodes. A leaf is a dense 8x8x8 grid of voxels.
constexpr uint8_t LEAF_DIM = 8;
constexpr uint16_t LEAF_VOXELS = LEAF_DIM * LEAF_DIM * LEAF_DIM;  // 512

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
VQVAECodec::VQVAECodec(const std::shared_ptr<IVQVAECodec>& backend) : backend_(backend) {
	if (!backend_) {
		throw std::runtime_error("VQVAECodec: Backend is not initialized.");
	}
}

void VQVAECodec::compress(const openvdb::FloatGrid::Ptr& grid, const std::string& outPath, const size_t batchSize) const {
	const auto t0 = std::chrono::high_resolution_clock::now();
	const openvdb::tree::LeafManager<openvdb::FloatTree> leafMgr(grid->tree());
	const int64_t N = leafMgr.leafCount();

	if (N == 0) {
		std::cout << "Grid has no active voxels. Nothing to compress.\n";
		return;
	}

	const std::vector<int64_t>& latentShapeVec = backend_->getLatentShape();

	// --- Gather all metadata for the VQVDB file header ---
	VQVDBMetadata metadata;
	metadata.fileVersion = 2;
	metadata.numEmbeddings = 256;  // Assuming a fixed codebook size
	metadata.latentShape = backend_->getLatentShape();
	metadata.totalBlocks = N;
	metadata.voxelSize = grid->voxelSize();
	metadata.transform = grid->transform().baseMap()->getAffineMap()->getMat4();

	VDBStreamWriter writer(outPath, metadata);
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

	VDBStreamReader streamer(inPath);
	const VQVDBMetadata& metadata = streamer.getMetadata();
	std::cout << "Starting decompression ...\n";

	// Create grid and apply the metadata read from the file
	grid = openvdb::FloatGrid::create();
	auto transform = openvdb::math::Transform::createLinearTransform(metadata.transform);
	grid->setTransform(transform);

	auto accessor = grid->getAccessor();

	const auto t1 = std::chrono::high_resolution_clock::now();
	int64_t done = 0;

	while (streamer.hasNext()) {
		EncodedBatch batch = streamer.nextBatch(batchSize);
		if (batch.data.numel() == 0) break;

		torch::Tensor decodedData = decodeBatch(batch.data);

		const float* dataPtr = decodedData.data_ptr<float>();

		for (size_t i = 0; i < batch.origins.size(); ++i) {
			const openvdb::Coord& origin = batch.origins[i];

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

torch::Tensor VQVAECodec::encodeBatch(const torch::Tensor& cpuBatch) const { return backend_->encode(cpuBatch); }

torch::Tensor VQVAECodec::decodeBatch(const torch::Tensor& cpuBatch) const { return backend_->decode(cpuBatch); }