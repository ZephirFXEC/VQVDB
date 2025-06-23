//
// Created by zphrfx on 22/06/2025.
//

#pragma once

//
// Created by zphrfx on 04/06/2025.
//

#include <cuda_runtime.h>
#include <openvdb/openvdb.h>
#include <openvdb/tools/GridOperators.h>
#include <torch/script.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

#include "Utils/VQVDB_Reader.hpp"

constexpr int BLOCK_DIM = 8;
constexpr int BLOCK_VOXELS = BLOCK_DIM * BLOCK_DIM * BLOCK_DIM;

// Main decoder class
class VQVAEDecoder {
   public:
	explicit VQVAEDecoder(const std::string& model_path);

	// Core function: Decodes a batch of indices into a [B,1,D,H,W] voxel tensor
	torch::Tensor decodeIndices(const torch::Tensor& indices) const;


	template <typename GridType>
	void decodeToGrid(const std::string& compressed_file, const typename GridType::Ptr& output_grid);

template <typename GridType>
void writeVoxelsToGrid(const openvdb::tree::ValueAccessor<openvdb::FloatTree>& grid, const std::vector<openvdb::Coord>& origins,
                       const float* __restrict__ raw, int B);


   private:
	torch::Device device_;
	torch::jit::Module model_;
	torch::jit::Method decodeMethod_;
};
