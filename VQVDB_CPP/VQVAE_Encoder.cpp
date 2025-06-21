//
// Created by zphrfx on 20/06/2025.
//

// VQVAE_Encoder.cpp

#include <openvdb/io/File.h>
#include <openvdb/openvdb.h>
#include <openvdb/tools/GridOperators.h>  // For LeafCIter
#include <torch/cuda.h>
#include <torch/script.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

// Helper to write a simple type to a binary stream (assumes little-endian)
template <typename T>
void write_binary(std::ofstream& stream, const T& value) {
	stream.write(reinterpret_cast<const char*>(&value), sizeof(T));
}
constexpr uint8_t LEAF_DIM = 8;
constexpr uint16_t LEAF_VOXELS = LEAF_DIM * LEAF_DIM * LEAF_DIM;  // 512
// Struct to hold the extracted data from a VDB file
struct VDBData {
	torch::Tensor blocks;  // A single tensor of shape [N, 1, 8, 8, 8]
	std::vector<openvdb::Coord> origins;
};

// Load a FloatGrid from a VDB file
openvdb::FloatGrid::Ptr loadVDBGrid(const std::string& vdb_path, const std::string& grid_name) {
	std::cout << "Reading VDB file: " << vdb_path << std::endl;
	openvdb::io::File file(vdb_path);
	file.open();

	const openvdb::GridBase::Ptr baseGrid = file.readGrid(grid_name);
	file.close();

	auto grid = openvdb::gridPtrCast<openvdb::FloatGrid>(baseGrid);
	if (!grid) {
		throw std::runtime_error("Failed to cast grid to FloatGrid. Check grid name and type.");
	}

	return grid;
}

// Extract tensor blocks from a FloatGrid
VDBData extractBlocksFromGrid(const openvdb::FloatGrid::Ptr& grid) {
	const size_t leaf_count = grid->tree().leafCount();
	std::cout << "Found " << leaf_count << " leaf nodes." << std::endl;

	if (leaf_count == 0) {
		return {torch::empty({0, 1, LEAF_DIM, LEAF_DIM, LEAF_DIM}), {}};
	}

	std::vector<torch::Tensor> block_list;
	std::vector<openvdb::Coord> origins;

	// Create leaf manager for parallel processing
	const openvdb::tree::LeafManager<openvdb::FloatTree> leafManager(grid->tree());

	// Resize containers
	block_list.reserve(leaf_count);
	origins.reserve(leaf_count);

	// Process leaves in parallel
	tbb::parallel_for(tbb::blocked_range<size_t>(0, leaf_count), [&](const tbb::blocked_range<size_t>& range) {
		for (size_t idx = range.begin(); idx != range.end(); ++idx) {
			auto& leaf = leafManager.leaf(idx);
			if (leaf.isEmpty()) continue;  // skip empty leaves

			// 1. Store the origin of the 8×8×8 block
			origins[idx] = leaf.origin();

			// 2. Wrap the leaf's internal buffer (layout: x-major → (x,y,z))
			const float* buf = leaf.buffer().data();
			torch::Tensor block = torch::from_blob(const_cast<float*>(buf), {LEAF_DIM, LEAF_DIM, LEAF_DIM},  // (x, y, z)
			                                       torch::TensorOptions().dtype(torch::kFloat32));

			// 3. Convert to (z, y, x), make contiguous, and add [N, C] dims
			block = block
			            .permute({2, 1, 0})  // (z, y, x)
			            .unsqueeze_(0)       // channel dim
			            .unsqueeze_(0);      // batch dim

			block_list[idx] = std::move(block);
		}
	});

	std::cout << "Found " << block_list.size() << " non-empty leaf nodes." << std::endl;

	if (block_list.empty()) {
		return {torch::empty({0, 1, LEAF_DIM, LEAF_DIM, LEAF_DIM}), {}};
	}

	const torch::Tensor all_blocks = torch::cat(block_list, 0);

	return {all_blocks, origins};
}

// Wrapper function to maintain compatibility
VDBData extractBlocksFromVDB(const std::string& vdb_path, const std::string& grid_name) {
	const auto grid = loadVDBGrid(vdb_path, grid_name);
	return extractBlocksFromGrid(grid);
}



class VQVAEEncoder {
   public:
	explicit VQVAEEncoder(const std::string& model_path) : device_(torch::kCPU) {
		// Use GPU if available
		device_ = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
		std::cout << "Using device: " << device_ << std::endl;

		try {
			// Load the scripted model (contains code and weights)
			model_ = torch::jit::load(model_path);
			model_.eval();  // Set to evaluation mode
			model_.to(device_);
		} catch (const c10::Error& e) {
			std::cerr << "Error loading the model: " << e.what() << std::endl;
			throw;
		}
	}

	void compress(const std::string& input_vdb_path, const std::string& grid_name, const std::string& output_path) const {
		auto start_time_loading = std::chrono::high_resolution_clock::now();

		// 1. Extract data from the VDB file
		auto [blocks, origins] = extractBlocksFromVDB(input_vdb_path, grid_name);
		if (origins.empty()) {
			std::cout << "No active voxels found in VDB. Output will be empty." << std::endl;
			return;
		}

		auto start_time_compression = std::chrono::high_resolution_clock::now();


		const int64_t num_leaves = origins.size();
		std::ofstream out_file(output_path, std::ios::binary);
		if (!out_file) {
			throw std::runtime_error("Failed to open output file for writing: " + output_path);
		}

		// 2. Write file header (matching Python script's `compress_vdb`)
		out_file.write("VQVDB", 5);          // Magic
		write_binary<uint8_t>(out_file, 2);  // Version

		// Get index grid shape from a sample pass
		auto sample_indices = encode_batch(blocks.slice(0, 0, 1));
		const auto idx_dims = sample_indices.sizes();

		write_binary<uint32_t>(out_file, static_cast<uint32_t>(sample_indices.size(0)));  // num_embeddings not used, just to match format.
		write_binary<uint8_t>(out_file, static_cast<uint8_t>(idx_dims.size() - 1));  // num dimensions in index grid (e.g., 3 for [B,D,H,W])
		for (size_t i = 1; i < idx_dims.size(); ++i) {
			write_binary<uint16_t>(out_file, static_cast<uint16_t>(idx_dims[i]));
		}

		// 3. Write origins block
		write_binary<uint32_t>(out_file, static_cast<uint32_t>(num_leaves));
		for (const auto& origin : origins) {
			write_binary<int32_t>(out_file, origin.x());
			write_binary<int32_t>(out_file, origin.y());
			write_binary<int32_t>(out_file, origin.z());
		}

		// 4. Write indices block in chunks
		write_binary<uint32_t>(out_file, static_cast<uint32_t>(num_leaves));
		constexpr int64_t CHUNK_SIZE = 8192;  // Match Python script

		std::cout << "Compressing " << num_leaves << " blocks in chunks of " << CHUNK_SIZE << "..." << std::endl;
		for (int64_t i = 0; i < num_leaves; i += CHUNK_SIZE) {
			int64_t end = std::min(i + CHUNK_SIZE, num_leaves);
			auto batch_tensor = blocks.slice(0, i, end);

			// Encode the batch
			auto indices = encode_batch(batch_tensor);

			// Convert to uint8 for writing
			auto indices_u8 = indices.to(torch::kCPU, torch::kU8);

			// Write raw bytes to file
			out_file.write(reinterpret_cast<const char*>(indices_u8.data_ptr<uint8_t>()),
			               indices_u8.numel()  // Total number of bytes
			);
		}

		out_file.close();

		const auto end_time = std::chrono::high_resolution_clock::now();
		const auto duration_compression = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time_compression).count();
		const auto duration_loading =
		    std::chrono::duration_cast<std::chrono::milliseconds>(start_time_compression - start_time_loading).count();

		// 5. Final Report
		const size_t original_size = std::filesystem::file_size(input_vdb_path);
		const size_t compressed_size = std::filesystem::file_size(output_path);
		const double ratio = (compressed_size > 0) ? static_cast<double>(original_size) / compressed_size : 0.0;

		printf("\n--- Loading Complete in %lld ms ---\n", duration_loading);
		printf("\n--- Compression Complete in %lld ms ---\n", duration_compression);
		printf("Original data size: %.2f MB\n", original_size / (1024.0 * 1024.0));
		printf("Compressed file size: %.2f MB\n", compressed_size / (1024.0 * 1024.0));
		printf("Compression Ratio: %.2fx\n", ratio);
		printf("Saved to %s\n", output_path.c_str());
	}

   private:
	torch::Tensor encode_batch(const torch::Tensor& batch) const {
		torch::NoGradGuard no_grad;

		// Move batch to the correct device
		const torch::Tensor input_tensor = batch.to(device_);

		// Prepare input for the JIT method call
		std::vector<torch::jit::IValue> inputs;
		inputs.push_back(input_tensor);

		// Call the scripted 'encode' method
		torch::Tensor indices = model_.get_method("encode")(inputs).toTensor();

		return indices;
	}

	torch::jit::Module model_;
	torch::Device device_;
};


int main(int argc, char** argv) {
	if (argc < 5) {
		std::cout << "Usage: " << argv[0] << " <scripted_model.pt> <input.vdb> <grid_name> <output.vqvdb>" << std::endl;
		return 1;
	}

	const std::string model_path = argv[1];
	const std::string input_path = argv[2];
	const std::string grid_name = argv[3];
	const std::string output_path = argv[4];

	try {
		// Initialize OpenVDB
		openvdb::initialize();

		const VQVAEEncoder encoder(model_path);
		encoder.compress(input_path, grid_name, output_path);

	} catch (const std::exception& e) {
		std::cerr << "An error occurred: " << e.what() << std::endl;
		return 1;
	}

	return 0;
}