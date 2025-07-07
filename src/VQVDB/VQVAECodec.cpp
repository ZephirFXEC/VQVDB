// VQVAECodec.cpp
//
// Created by zphrfx on 23/06/2025.
//

#include "VQVAECodec.hpp"

#include <openvdb/tools/GridOperators.h>
#include <tbb/task_scheduler_init.h>

#include <chrono>
#include <iostream>
#include <vector>

#include "VQVDB_Reader.hpp"

namespace {
constexpr int LEAF_LOG2DIM = 3;
constexpr int LEAF_DIM = 1 << LEAF_LOG2DIM;                     // 8
constexpr size_t LEAF_VOXELS = LEAF_DIM * LEAF_DIM * LEAF_DIM;  // 512
// =========================================================================================
// Helper Streamer for Reading VDB Leaf Blocks (for Compression)
// =========================================================================================
class VDBInputBlockStreamer {
   public:
	using LeafManager = openvdb::tree::LeafManager<openvdb::FloatTree>;

	explicit VDBInputBlockStreamer(const LeafManager& leafManager)
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
				std::memcpy(dstBase + i * LEAF_VOXELS, leaf.buffer().data(), LEAF_VOXELS * sizeof(float));
			}
		});

		return {batch, origins};  // Add channel dim: [B, 1, D, D, D]
	}

   private:
	const LeafManager& leafManager_;
	size_t currentPos_;
	const size_t totalLeaves_;
};
}  // namespace

// =========================================================================================
// VQVAECodec Method Implementations
// =========================================================================================
VQVAECodec::VQVAECodec(const std::shared_ptr<IVQVAECodec>& backend) : backend_(backend) {
	if (!backend_) {
		throw std::runtime_error("VQVAECodec: Backend cannot be null.");
	}
}

void VQVAECodec::compress(const openvdb::FloatGrid::Ptr& grid, const std::string& outPath, const size_t batchSize) const {
	const auto t0 = std::chrono::high_resolution_clock::now();

	const openvdb::tree::LeafManager<openvdb::FloatTree> leafMgr(grid->tree());
	const int64_t totalBlocks = leafMgr.leafCount();

	if (totalBlocks == 0) {
		std::cout << "Grid has no active voxels. Nothing to compress.\n";
		return;
	}

	// --- Gather all metadata for the VQVDB file header ---
	VQVDBMetadata metadata;
	metadata.fileVersion = 2;
	metadata.numEmbeddings = 256;  // Assuming a fixed codebook size
	metadata.latentShape = backend_->getLatentShape();
	metadata.totalBlocks = totalBlocks;
	metadata.voxelSize = grid->voxelSize();
	metadata.transform = grid->transform().baseMap()->getAffineMap()->getMat4();

	VDBStreamWriter writer(outPath, metadata);
	VDBInputBlockStreamer streamer(leafMgr);


	const auto t1 = std::chrono::high_resolution_clock::now();
	std::cout << "Starting compression of " << totalBlocks << " blocks with batch size " << batchSize << "...\n";
	int64_t blocksProcessed = 0;

	while (streamer.hasNext()) {
		auto [hostTensor, origins] = streamer.nextBatch(batchSize);
		if (hostTensor.numel() == 0) break;

		torch::Tensor encodedIndices = this->encodeBatch(hostTensor);
		writer.writeBatch(encodedIndices, origins);

		blocksProcessed += hostTensor.size(0);
		fprintf(stderr, "\rProcessed %lld / %lld blocks...", (long long)blocksProcessed, totalBlocks);
	}

	fprintf(stderr, "\n");

	const auto t2 = std::chrono::high_resolution_clock::now();
	const auto setup_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
	const auto compress_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
	printf("\nCompression Complete.\n-- Setup  : %lld ms\n-- Encode : %lld ms\n", setup_ms, compress_ms);
}


void VQVAECodec::decompress(const std::string& inPath, openvdb::FloatGrid::Ptr& grid, const size_t batchSize) const {
	const auto t0 = std::chrono::high_resolution_clock::now();

	VDBStreamReader streamer(inPath);
	const VQVDBMetadata& metadata = streamer.getMetadata();
	std::cout << "Starting decompression ...\n";

	// Create the final grid and apply the metadata read from the file
	grid = openvdb::FloatGrid::create();
	auto transform = openvdb::math::Transform::createLinearTransform(metadata.transform);
	grid->setTransform(transform);
	const int64_t totalBlocks = streamer.totalBlocks();

	const auto t1 = std::chrono::high_resolution_clock::now();
	std::cout << "Starting decompression of " << totalBlocks << " blocks with batch size " << batchSize << "...\n";

	// Determine number of threads and create separate grids
	const size_t numThreads = tbb::task_scheduler_init::default_num_threads();
	std::vector<openvdb::FloatGrid::Ptr> threadGrids(numThreads);
	std::vector<openvdb::tree::ValueAccessor<openvdb::FloatTree>> threadAccessors;

	for (size_t i = 0; i < numThreads; ++i) {
		threadGrids[i] = openvdb::FloatGrid::create();
		threadGrids[i]->setTransform(transform);
		threadAccessors.emplace_back(threadGrids[i]->getAccessor());
	}

	std::atomic<int64_t> blocksProcessed = 0;

	while (streamer.hasNext()) {
		EncodedBatch batch = streamer.nextBatch(batchSize);
		if (batch.data.numel() == 0) break;

		const size_t numInBatch = batch.origins.size();
		torch::Tensor decodedData = this->decodeBatch(batch.data);
		const float* decodedDataPtr = decodedData.data_ptr<float>();

		tbb::parallel_for(tbb::blocked_range<size_t>(0, numInBatch), [&](const tbb::blocked_range<size_t>& r) {
			const size_t threadId = tbb::this_task_arena::current_thread_index() % numThreads;
			auto& accessor = threadAccessors[threadId];

			for (size_t i = r.begin(); i != r.end(); ++i) {
				if (auto* leaf = accessor.touchLeaf(batch.origins[i])) {
					const float* src = decodedDataPtr + i * LEAF_VOXELS;
					std::memcpy(leaf->buffer().data(), src, LEAF_VOXELS * sizeof(float));
					leaf->setValuesOn();
				}
			}
		});

		blocksProcessed += numInBatch;
		fprintf(stderr, "\rProcessed %lld / %lld blocks...", static_cast<long long>(blocksProcessed), totalBlocks);
	}
	fprintf(stderr, "\n");

	const auto t2 = std::chrono::high_resolution_clock::now();

	// Merge all thread grids into the final grid
	std::cout << "Merging grids from " << numThreads << " threads...\n";
	for (size_t i = 0; i < numThreads; ++i) {
		if (threadGrids[i]->activeVoxelCount() > 0) {
			grid->tree().merge(threadGrids[i]->tree());
		}
	}

	const auto t3 = std::chrono::high_resolution_clock::now();
	const auto setup_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
	const auto decompress_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
	const auto merge_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();
	printf("\nDecompression Complete.\n-- Setup  : %lld ms\n-- Decode : %lld ms\n-- Merge : %lld ms\n", setup_ms, decompress_ms, merge_ms);
}

torch::Tensor VQVAECodec::encodeBatch(const torch::Tensor& cpuBatch) const { return backend_->encode(cpuBatch); }

torch::Tensor VQVAECodec::decodeBatch(const torch::Tensor& cpuBatch) const { return backend_->decode(cpuBatch); }