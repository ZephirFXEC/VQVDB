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

#include "Utils/Kernel.cuh"
#include "Utils/VQVDB_Reader.hpp"

constexpr int BLOCK_DIM = 8;
constexpr int BLOCK_VOXELS = BLOCK_DIM * BLOCK_DIM * BLOCK_DIM;


// Host function to launch lookup kernel
void lookupCodebook(const torch::Tensor& codebook, const torch::Tensor& indices, torch::Tensor& output) {
	// Get tensor shapes and sizes
	const int batch_size = indices.size(0);
	const int depth = indices.size(1);
	const int height = indices.size(2);
	const int width = indices.size(3);
	const int embedding_dim = codebook.size(1);
	const int num_embeddings = codebook.size(0);

	// Get raw pointers to tensor data
	const float* codebook_ptr = codebook.data_ptr<float>();

	const uint16_t* indices_ptr = reinterpret_cast<const uint16_t*>(indices.data_ptr<int>());
	float* output_ptr = output.data_ptr<float>();

	// Launch kernel via CUDA wrapper
	lookupCodebook_launch(codebook_ptr, indices_ptr, output_ptr, batch_size, depth, height, width, embedding_dim, num_embeddings);

	// Check for errors
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
		throw std::runtime_error("CUDA kernel launch failed");
	}
}

template <typename GridType>
void writeVoxelsToGrid(const openvdb::tree::ValueAccessor<openvdb::FloatTree>& grid, const std::vector<openvdb::Coord>& origins,
                       const float* __restrict__ raw, int B) {
	using TreeType = typename GridType::TreeType;
	using ValueT = typename GridType::ValueType;

	const int D = BLOCK_DIM, H = BLOCK_DIM, W = BLOCK_DIM;
	const int WH = W * H;
	const int blockSize = D * WH;

	for (int b = 0; b < B; ++b) {
		const openvdb::Coord& org = origins[b];
		const float* blockPtr = raw + int64_t(b) * blockSize;

		// grab (or create) the leaf node covering this block
		const auto leaf = grid.tree().touchLeaf(org);
		ValueT* leafPtr = leaf->buffer().data();
		// leaf->origin() gives the base coords of this leaf; not actually needed here

		// flattened 1D loop
		for (int idx = 0; idx < blockSize; ++idx) {
			float v = blockPtr[idx];
			if (v == 0.0f) continue;  // skip empties

			int z = idx / WH;
			int rem = idx % WH;
			int y = rem / W;
			int x = rem % W;

			// compute local index in leaf’s backing array
			int localIdx = (z * H + y) * W + x;
			leafPtr[localIdx] = static_cast<ValueT>(v);
		}
	}
}

// Main decoder class
class VQVAEDecoder {
   public:
	explicit VQVAEDecoder(const std::string& model_path)
	    : device_(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU),
	      model_(torch::jit::load(model_path)),
	      decodeMethod_(model_.get_method("decode")) {
		std::cout << "Using device: " << device_ << std::endl;

		model_.eval();
		model_.to(device_, device_.is_cuda() ? torch::kHalf : torch::kFloat32);

		// Lookup method "decode" from the JIT module
		if (!model_.find_method("decode")) {
			throw std::runtime_error("The provided PyTorch JIT model has no 'decode' method.");
		}
	}

	// Core function: Decodes a batch of indices into a [B,1,D,H,W] voxel tensor
	torch::Tensor decodeIndices(const torch::Tensor& indices) const {
		torch::NoGradGuard no_grad;
		// Convert indices to device-friendly type
		auto final_indices = indices.to(device_, torch::kInt32, /*non_blocking=*/true);
		auto result = decodeMethod_({final_indices}).toTensor();  // shape [B,1,D,H,W]
		return result;                                            // Still on device; final reshape done on CPU side
	}


	template <typename GridType>
	void decodeToGrid(const std::string& compressed_file, const typename GridType::Ptr& output_grid) {
		CompressedIndexReader reader(compressed_file);
		int total_blocks = 0;
		auto start_time = std::chrono::high_resolution_clock::now();

		openvdb::tree::ValueAccessor<openvdb::FloatTree> accessor(output_grid->tree());

		while (auto opt = reader.readNextBatch(8192)) {
			torch::Tensor indices = *opt;                         // [B, …]
			torch::Tensor voxelsDevice = decodeIndices(indices);  // [B,1,D,H,W] on GPU/CPU

			int B = indices.size(0);
			// one-shot move to CPU-float
			auto voxels = voxelsDevice.to(torch::kCPU, torch::kFloat32, /*non_blocking=*/true).contiguous();
			const float* raw = voxels.data_ptr<float>();

			// slice out exactly B origins
			int start = total_blocks;
			int end = std::min(total_blocks + B, int(reader.header.origins.size()));
			std::vector<openvdb::Coord> origins_slice(reader.header.origins.begin() + start, reader.header.origins.begin() + end);

			writeVoxelsToGrid<GridType>(accessor, origins_slice, raw, B);
			total_blocks += B;
		}

		auto end_time = std::chrono::high_resolution_clock::now();
		auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
		std::cout << "Decoded " << total_blocks << " blocks in " << ms << " ms\n";
		if (total_blocks) {
			std::cout << "Average " << double(ms * 1000) / total_blocks << " us/block\n";
		}
	}

   private:
	torch::Device device_;
	torch::jit::Module model_;
	torch::jit::Method decodeMethod_;
};


int main(int argc, char** argv) {
	if (argc < 5) {
		std::cout << "Usage: " << argv[0] << " <model.pt> <compressed_indices.vqvdb> <output.vdb> [grid_name]" << std::endl;
		return 1;
	}

	const std::string model_path = argv[1];
	const std::string compressed_path = argv[2];
	const std::string output_path = argv[3];
	const std::string grid_name = (argc > 4) ? argv[4] : "";

	try {
		// Initialize OpenVDB
		openvdb::initialize();

		VQVAEDecoder decoder(model_path);
		auto output_grid = openvdb::FloatGrid::create();
		decoder.decodeToGrid<openvdb::FloatGrid>(compressed_path, output_grid);

		// Write output grid
		openvdb::io::File file(output_path);
		file.write({output_grid});
		file.close();


		std::cout << "Successfully decoded and saved to " << output_path << std::endl;

	} catch (const std::exception& e) {
		std::cerr << "Error: " << e.what() << std::endl;
		return 1;
	}

	return 0;
}
