//
// Created by zphrfx on 22/06/2025.
//

#pragma once

#include <openvdb/openvdb.h>
#include <openvdb/tools/GridOperators.h>  // For LeafCIter
#include <torch/script.h>

#include <filesystem>
#include <vector>

constexpr uint8_t LEAF_DIM = 8;
constexpr uint16_t LEAF_VOXELS = LEAF_DIM * LEAF_DIM * LEAF_DIM;  // 512
// Struct to hold the extracted data from a VDB file
struct VDBData {
	torch::Tensor blocks;  // A single tensor of shape [N, 1, 8, 8, 8]
};


class VDBBlockStreamer {
   public:
	VDBBlockStreamer(const openvdb::tree::LeafManager<openvdb::FloatTree>& leafManager, const std::vector<size_t>& nonEmptyLeafIndices)
	    : leafManager_(leafManager), leafIndices_(nonEmptyLeafIndices), currentPos_(0) {}

	[[nodiscard]] bool hasNext() const noexcept { return currentPos_ < leafIndices_.size(); }

	torch::Tensor nextBatch(size_t maxBatch);

   private:
	const openvdb::tree::LeafManager<openvdb::FloatTree>& leafManager_;
	const std::vector<size_t>& leafIndices_;
	size_t currentPos_;
};

class VQVAEEncoder {
   public:
	explicit VQVAEEncoder(const std::string& modelPath);
	/**  Compress one grid into a .vqvdb file  */
	void compress(const openvdb::FloatGrid::Ptr& grid, const std::string& outPath) const;

   private:
	/** hostBatch: pinned FP32 on CPU.  Returns **uint8 on CPU**. */
	torch::Tensor encodeBatch(const torch::Tensor& hostBatch) const;
	torch::Device device_;
	torch::jit::Module model_;
	torch::jit::Method encodeMethod_;
};
