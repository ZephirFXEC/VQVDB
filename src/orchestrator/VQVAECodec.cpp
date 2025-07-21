/*
 * Copyright (c) 2025, Enzo Crema
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See the LICENSE file in the project root for full license text.
 */

#include "VQVAECodec.hpp"

#include <openvdb/tools/GridOperators.h>

#include <chrono>
#include <iostream>
#include <vector>

#include "utils/VQVDB_Reader.hpp"

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

	// Returns a standard vector of floats, not a torch tensor.
	std::pair<std::vector<float>, std::vector<openvdb::Coord>> nextBatch(size_t maxBatch) {
		if (!hasNext()) return {{}, {}};

		const size_t start = currentPos_;
		const size_t end = std::min(start + maxBatch, totalLeaves_);
		currentPos_ = end;
		const size_t B = end - start;

		// Create a standard vector to hold the batch data.
		std::vector<float> batchData(B * LEAF_VOXELS);
		std::vector<openvdb::Coord> origins(B);

		float* dstBase = batchData.data();

		tbb::parallel_for(tbb::blocked_range<size_t>(0, B), [&](const tbb::blocked_range<size_t>& r) {
			for (size_t i = r.begin(); i != r.end(); ++i) {
				const auto& leaf = leafManager_.leaf(start + i);
				origins[i] = leaf.origin();
				std::memcpy(dstBase + i * LEAF_VOXELS, leaf.buffer().data(), LEAF_VOXELS * sizeof(float));
			}
		});

		return {batchData, origins};
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
VQVAECodec::VQVAECodec(std::unique_ptr<IVQVAECodec> backend) : backend_(std::move(backend)) {
	if (!backend_) {
		throw std::runtime_error("VQVAECodec: Backend cannot be null.");
	}
}


void VQVAECodec::compress(const std::vector<openvdb::FloatGrid::Ptr>& grids, const std::filesystem::path& outPath, size_t batchSize) const {
	const auto t0 = std::chrono::high_resolution_clock::now();

	VDBStreamWriter writer(outPath.string());

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
		metadata.transform = grid->transform().baseMap()->getAffineMap()->getMat4();

		writer.startGrid(metadata);

		VDBInputBlockStreamer streamer(leafMgr);
		int64_t blocksProcessed = 0;

		while (streamer.hasNext()) {
			// 1. Get raw float data from the streamer
			auto [hostData, origins] = streamer.nextBatch(batchSize);
			if (hostData.empty()) break;

			// 2. Create a non-owning TensorView to wrap the raw data
			TensorView batchView;
			batchView.data = hostData.data();
			batchView.shape = {static_cast<long>(origins.size()), 1, LEAF_DIM, LEAF_DIM, LEAF_DIM};
			batchView.dtype = DataType::FLOAT32;

			// 3. Encode the batch. The backend handles framework specifics.
			Tensor encodedTensor = this->encodeBatch(batchView);

			// 4. Write the resulting owning Tensor. (Assumes writer API is updated).
			writer.writeBatch(encodedTensor, origins);

			blocksProcessed += origins.size();
			printf("Processed %lld / %lld blocks for grid '%s'...\n", blocksProcessed, totalBlocks, name.c_str());
		}
		writer.endGrid();
	}

	const auto t2 = std::chrono::high_resolution_clock::now();
	const auto compress_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t0).count();
	printf("\nGrid Compression Complete in %lld ms.\n", compress_ms);
}


void VQVAECodec::decompress(const std::filesystem::path& inPath, std::vector<openvdb::FloatGrid::Ptr>& grids, size_t batchSize) const {
	const auto t0 = std::chrono::high_resolution_clock::now();

	VDBStreamReader reader(inPath.string());
	grids.clear();

	const size_t numThreads = tbb::this_task_arena::max_concurrency();

	while (reader.hasNextGrid()) {
		VQVDBMetadata metadata = reader.nextGridMetadata();
		const int64_t totalBlocks = metadata.totalBlocks;

		std::cout << "Decompressing grid '" << metadata.name << "' with " << totalBlocks << " blocks...\n";

		auto grid = openvdb::FloatGrid::create();
		const auto transform = openvdb::math::Transform::createLinearTransform(openvdb::Mat4R(metadata.transform));
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
			// 1. Reader provides a batch (Assumes EncodedBatch now contains a generic Tensor)
			EncodedBatch batch = reader.nextBatch(batchSize);
			if (batch.data.buffer.empty()) break;

			// 2. Create a non-owning view of the loaded data
			TensorView indicesView;
			indicesView.data = batch.data.buffer.data();
			indicesView.shape = batch.data.shape;
			indicesView.dtype = batch.data.dtype;

			// 3. Decode the batch
			Tensor decodedTensor = this->decodeBatch(indicesView);
			const float* decodedDataPtr = decodedTensor.getData<float>();  // Use the new getter

			// 4. Parallel reconstruction (this part remains the same)
			tbb::parallel_for(tbb::blocked_range<size_t>(0, batch.origins.size()), [&](const tbb::blocked_range<size_t>& r) {
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

			blocksProcessed += batch.origins.size();
			printf("Processed %lld / %lld blocks...\n", blocksProcessed.load(), totalBlocks);
		}

		for (size_t i = 0; i < numThreads; ++i) {
			grid->tree().merge(threadGrids[i]->tree());
		}

		grids.push_back(grid);
	}

	const auto t3 = std::chrono::high_resolution_clock::now();
	const auto decompress_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t0).count();
	printf("\nMulti-Grid Decompression Complete in %lld ms.\n", decompress_ms);
}

Tensor VQVAECodec::encodeBatch(const TensorView& cpuBatch) const { return backend_->encode(cpuBatch); }

Tensor VQVAECodec::decodeBatch(const TensorView& cpuBatch) const { return backend_->decode(cpuBatch); }