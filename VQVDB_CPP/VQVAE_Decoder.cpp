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
void writeVoxelsToGrid(const typename GridType::Ptr& grid, const std::vector<openvdb::Coord>& origins, const torch::Tensor& voxelData) {
	// --- Type aliases for clarity ---
	using TreeType = typename GridType::TreeType;
	using LeafNodeType = typename TreeType::LeafNodeType;
	using ValueT = typename GridType::ValueType;

	// Move data to CPU float tensor once
	const auto cpu = voxelData.to(torch::kCPU, torch::kFloat32);
	const float* raw = cpu.data_ptr<float>();

	const int B = cpu.size(0);
	const int D = cpu.size(2);
	const int H = cpu.size(3);
	const int W = cpu.size(4);

	if (D != BLOCK_DIM || H != BLOCK_DIM || W != BLOCK_DIM) throw std::runtime_error("Decoded block is not 8³ – shape mismatch");

	// Precompute per‐block stride: B × C × D × H × W with C==1
	const int64_t blockSize = int64_t(D) * H * W;
	openvdb::tree::ValueAccessor<openvdb::FloatTree> acc(grid->tree());

	for (int b = 0; b < B && b < (int)origins.size(); ++b) {
		const auto& origin = origins[b];
		const float* blockPtr = raw + b * blockSize;

		// Each voxel
		for (int z = 0; z < D; ++z) {
			int64_t zOff = z * (H * W);
			for (int y = 0; y < H; ++y) {
				int64_t yzOff = zOff + y * W;
				for (int x = 0; x < W; ++x) {
					float v = blockPtr[yzOff + x];
					acc.setValue(openvdb::Coord(origin.x() + x, origin.y() + y, origin.z() + z), static_cast<ValueT>(v));
				}
			}
		}
	}
}

// Main decoder class
class VQVAEDecoder {
   public:
	explicit VQVAEDecoder(const std::string& model_path) : device_(torch::kCUDA) {
		try {
			model_ = torch::jit::load(model_path);
			model_.eval();

			model_.to(torch::kCUDA);
			device_ = torch::kCUDA;
			std::cout << "Using CUDA device for decoding" << std::endl;

			extractCodebook();

		} catch (const c10::Error& e) {
			std::cerr << "Error loading the model: " << e.what() << std::endl;
			throw;
		}
	}

	void extractCodebook() {
		for (const auto& [name, value] : model_.named_buffers()) {
			if (name == "quantizer.embedding") {
				codebook_ = value.clone().to(device_);
				std::cout << "Found codebook buffer with shape: [" << codebook_.size(0) << ", " << codebook_.size(1) << "]" << std::endl;
				return;
			}
		}

		// If we get here, neither worked.
		throw std::runtime_error("Could not find codebook in model's buffers ('vq.embedding') or parameters ('vq.embedding.weight')");
	}

	torch::Tensor decodeIndices(const torch::Tensor& indices) const {
		torch::NoGradGuard no_grad;

		// --- The ONLY thing you need to do ---
		// 1. Ensure the indices are on the correct device and are of type long.
		//    `torch::kInt64` (long) is the safest choice for F.embedding.
		torch::Tensor final_indices = indices.to(device_, torch::kInt64);

		// 2. Prepare the input for the JIT method call.
		std::vector<torch::jit::IValue> inputs;
		inputs.emplace_back(final_indices);

		// 3. Call the scripted 'decode' method and return the result.
		torch::Tensor output;
		try {
			// This single call will perform the embedding lookup, permutation,
			// and decoding, all within the optimized JIT graph.
			output = model_.get_method("decode")(inputs).toTensor();
		} catch (const c10::Error& e) {
			// It's good to keep this for debugging.
			std::cerr << "An error occurred while calling the 'decode' method: " << e.what() << std::endl;
			throw;  // Re-throw the exception so the program terminates.
		}

		return output;
	}

	template <typename GridType>
	void decodeToGrid(const std::string& compressed_file, const typename GridType::Ptr& output_grid) {
		// Note: The batch size is now determined by the file, not the function argument.

		// 1. Create the reader. It opens the file and reads the header.
		CompressedIndexReader reader(compressed_file);

		int total_blocks = 0;
		auto start_time = std::chrono::high_resolution_clock::now();

		// 2. Loop by reading one batch at a time until the file is exhausted.
		while (auto opt_indices_batch = reader.readNextBatch(512)) {
			torch::Tensor& indices = *opt_indices_batch;

			torch::Tensor voxels = decodeIndices(indices);

			int current_batch_size = indices.size(0);

			size_t end_offset = std::min<size_t>(total_blocks + current_batch_size, reader.header.origins.size());

			std::vector<openvdb::Coord> origins_slice(reader.header.origins.begin() + total_blocks,
			                                          reader.header.origins.begin() + end_offset);

			writeVoxelsToGrid<GridType>(output_grid, origins_slice, voxels);

			total_blocks += current_batch_size;
		}

		auto end_time = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

		std::cout << "Decoded " << total_blocks << " blocks in " << duration << " ms\n";
		if (total_blocks > 0) {
			std::cout << "Average: " << static_cast<double>(duration) / total_blocks << " ms per block\n";
		}
	}

   private:
	torch::jit::Module model_;
	torch::Tensor codebook_;
	torch::Device device_;
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
