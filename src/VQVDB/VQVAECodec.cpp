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

void VQVAECodec::compress(const std::vector<openvdb::FloatGrid::Ptr>& grids, const std::string& outPath, const size_t batchSize) const {
	const auto t0 = std::chrono::high_resolution_clock::now();

	VDBStreamWriter writer(outPath);

	for (size_t g = 0; g < grids.size(); ++g) {
		const auto& grid = grids[g];
		const std::string name = grid->getName();
		const openvdb::tree::LeafManager<openvdb::FloatTree> leafMgr(grid->tree());
		const int64_t totalBlocks = leafMgr.leafCount();

		if (totalBlocks == 0) {
			std::cout << "Grid '" << name << "' has no active voxels. Skipping.\n";
			continue;
		}

		// Gather metadata
		VQVDBMetadata metadata;
		metadata.name = name;
		metadata.fileVersion = 3;
		metadata.numEmbeddings = 256;  // Assuming fixed codebook size
		metadata.latentShape = backend_->getLatentShape();
		metadata.totalBlocks = totalBlocks;
		metadata.voxelSize = grid->voxelSize();
		metadata.transform = grid->transform().baseMap()->getAffineMap()->getMat4();

		writer.startGrid(metadata);

		VDBInputBlockStreamer streamer(leafMgr);
		int64_t blocksProcessed = 0;

		while (streamer.hasNext()) {
			auto [hostTensor, origins] = streamer.nextBatch(batchSize);
			if (hostTensor.numel() == 0) break;

			torch::Tensor encodedIndices = this->encodeBatch(hostTensor);
			writer.writeBatch(encodedIndices, origins);

			blocksProcessed += hostTensor.size(0);
			fprintf(stderr, "\rGrid '%s': Processed %lld / %lld blocks...", name.c_str(), (long long)blocksProcessed, totalBlocks);
		}
		fprintf(stderr, "\n");

		writer.endGrid();
	}

	const auto t2 = std::chrono::high_resolution_clock::now();
	const auto compress_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t0).count();
	printf("\nMulti-Grid Compression Complete in %lld ms.\n", compress_ms);
}


void VQVAECodec::decompress(const std::string& inPath, std::vector<openvdb::FloatGrid::Ptr>& grids, const size_t batchSize) const {
	const auto t0 = std::chrono::high_resolution_clock::now();

	VDBStreamReader reader(inPath);
	grids.clear();

	const size_t numThreads = tbb::task_scheduler_init::default_num_threads();

	while (reader.hasNextGrid()) {
		VQVDBMetadata metadata = reader.nextGridMetadata();
		const int64_t totalBlocks = metadata.totalBlocks;

		std::cout << "Decompressing grid '" << metadata.name << "' with " << totalBlocks << " blocks...\n";

		auto grid = openvdb::FloatGrid::create();
		auto transform = openvdb::math::Transform::createLinearTransform(metadata.transform);
		grid->setTransform(transform);
		grid->setName(metadata.name);

		// Thread-local grids for parallel insertion
		std::vector<openvdb::FloatGrid::Ptr> threadGrids(numThreads);
		std::vector<openvdb::tree::ValueAccessor<openvdb::FloatTree>> threadAccessors;
		for (size_t i = 0; i < numThreads; ++i) {
			threadGrids[i] = openvdb::FloatGrid::create();
			threadGrids[i]->setTransform(transform);
			threadAccessors.emplace_back(threadGrids[i]->getAccessor());
		}

		std::atomic<int64_t> blocksProcessed = 0;

		while (reader.hasNext()) {
			EncodedBatch batch = reader.nextBatch(batchSize);
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

		// Merge thread grids into main grid
		for (size_t i = 0; i < numThreads; ++i) {
			if (threadGrids[i]->activeVoxelCount() > 0) {
				grid->tree().merge(threadGrids[i]->tree());
			}
		}

		grids.push_back(grid);
	}

	const auto t3 = std::chrono::high_resolution_clock::now();
	const auto decompress_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t0).count();
	printf("\nMulti-Grid Decompression Complete in %lld ms.\n", decompress_ms);
}

torch::Tensor VQVAECodec::encodeBatch(const torch::Tensor& cpuBatch) const { return backend_->encode(cpuBatch); }

torch::Tensor VQVAECodec::decodeBatch(const torch::Tensor& cpuBatch) const { return backend_->decode(cpuBatch); }