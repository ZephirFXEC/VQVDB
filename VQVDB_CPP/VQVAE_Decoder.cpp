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

#include "../Kernel.cuh"


// Size of a dense “mega-leaf” (= one code-book entry)
constexpr int BLOCK_DIM = 8;  // 32 × 32 × 32
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


// Header for compressed VDB index file
struct CompressedHeader {
	char magic[5];                        // "VQVDB"
	uint8_t version;                      // Version number
	uint32_t numEmbeddings;               // Number of codebook entries
	uint8_t numDimensions;                // Number of dimensions in the index tensor (typically 3 for [d,h,w])
	std::vector<uint16_t> shape;          // Shape of each index tensor
	uint32_t leafCount = 0;               // Number of leaf nodes (optional)
	std::vector<openvdb::Coord> origins;  // Leaf origins if present
};

// Read compressed indices from file
std::vector<torch::Tensor> readCompressedIndices(const std::string& filename, CompressedHeader& header) {
	std::ifstream file(filename, std::ios::binary);
	if (!file) {
		throw std::runtime_error("Failed to open compressed file: " + filename);
	}

	// Read magic number and version
	char magic[6] = {0};
	file.read(magic, 5);
	if (std::string(magic) != "VQVDB") {
		throw std::runtime_error("Invalid file format: incorrect magic number");
	}

	// Read version
	file.read(reinterpret_cast<char*>(&header.version), 1);
	if (header.version != 1 && header.version != 2) {
		throw std::runtime_error("Unsupported file version");
	}

	std::cout << "Reading compressed indices from " << filename << " (version " << static_cast<int>(header.version) << ")" << std::endl;

	// Read number of embeddings
	file.read(reinterpret_cast<char*>(&header.numEmbeddings), sizeof(uint32_t));

	// Read number of dimensions
	file.read(reinterpret_cast<char*>(&header.numDimensions), 1);

	// Read shape dimensions
	header.shape.resize(header.numDimensions);
	file.read(reinterpret_cast<char*>(header.shape.data()), header.numDimensions * sizeof(uint16_t));

	if (header.version >= 2) {
		file.read(reinterpret_cast<char*>(&header.leafCount), sizeof(uint32_t));
		header.origins.resize(header.leafCount);
		file.read(reinterpret_cast<char*>(header.origins.data()), header.leafCount * sizeof(openvdb::Coord));
	}

	// Calculate bits needed per index
	// (matches the Python bit_length implementation used when writing)
	unsigned int embedding_minus_one = header.numEmbeddings - 1;
	int bitsPerIndex = 32 - _lzcnt_u32(embedding_minus_one);
	int bytesPerIndex = (bitsPerIndex + 7) / 8;  // Round up to bytes

	// Read all batches
	std::vector<torch::Tensor> all_indices;

	while (file) {
		// Read batch size
		uint32_t batchSize;
		file.read(reinterpret_cast<char*>(&batchSize), sizeof(uint32_t));

		if (!file || file.eof()) break;

		// Calculate total elements in this batch
		size_t totalElements = batchSize;
		for (uint16_t dim : header.shape) {
			totalElements *= dim;
		}

		// Read indices
		std::vector<uint16_t> indices_data(totalElements);
		file.read(reinterpret_cast<char*>(indices_data.data()), totalElements * sizeof(uint16_t));

		// Create tensor and append to list
		std::vector<int64_t> tensor_shape;
		tensor_shape.push_back(batchSize);
		for (uint16_t dim : header.shape) {
			tensor_shape.push_back(dim);
		}

		torch::Tensor indices_tensor =
		    torch::from_blob(indices_data.data(), tensor_shape, torch::TensorOptions().dtype(torch::kInt16)).clone();

		all_indices.push_back(indices_tensor);
	}

	return all_indices;
}

// Write voxels back to OpenVDB grid
template <typename GridType>
void writeVoxelsToGrid(typename GridType::Ptr grid, const std::vector<openvdb::Coord>& origins, const torch::Tensor& voxelData) {
	using ValueT = typename GridType::ValueType;

	// Ensure tensor is on CPU and convert to correct data type
	const at::Tensor cpu_data = voxelData.to(torch::kCPU);
	const at::Tensor cpu_data_float = cpu_data.to(torch::kFloat32);

	const int batch_size = cpu_data.size(0);
	const int depth = cpu_data.size(2);
	const int height = cpu_data.size(3);
	const int width = cpu_data.size(4);

	if (depth != BLOCK_DIM || height != BLOCK_DIM || width != BLOCK_DIM)
		throw std::runtime_error("Decoded block is not 32³ – shape mismatch");


	openvdb::tree::ValueAccessor<openvdb::FloatTree> acc(grid->tree());
	// For each leaf
	for (int b = 0; b < batch_size && b < static_cast<int>(origins.size()); b++) {
		openvdb::Coord origin = origins[b];

		// For each voxel in the leaf
		for (int z = 0; z < BLOCK_DIM; z++) {
			for (int y = 0; y < BLOCK_DIM; y++) {
				for (int x = 0; x < BLOCK_DIM; x++) {
					// Get voxel value
					float value = cpu_data_float[b][0][z][y][x].item<float>();

					// Set voxel in grid
					openvdb::Coord xyz(origin.x() + x, origin.y() + y, origin.z() + z);
					acc.setValue(xyz, static_cast<ValueT>(value));
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
			// Load the TorchScript model
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
		// The codebook is stored as a buffer, not a parameter, in the EMA version.
		// We need to iterate through the model's named_buffers().
		for (const auto& buffer : model_.named_buffers()) {
			// The name in the scripted model will be 'vq.embedding'
			if (buffer.name == "quantizer.embedding") {
				codebook_ = buffer.value.clone().to(device_);
				std::cout << "Found codebook buffer with shape: [" << codebook_.size(0) << ", " << codebook_.size(1) << "]" << std::endl;
				return;
			}
		}

		// If we get here, neither worked.
		throw std::runtime_error("Could not find codebook in model's buffers ('vq.embedding') or parameters ('vq.embedding.weight')");
	}

	torch::Tensor decodeIndices(const torch::Tensor& indices) {
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
			throw; // Re-throw the exception so the program terminates.
		}

		return output;
	}

	template <typename GridType>
	void decodeToGrid(const std::string& compressed_file, typename GridType::Ptr output_grid, int batch_size = 64) {
		// Read compressed indices (also loads leaf origins if present)
		CompressedHeader header;
		std::vector<torch::Tensor> all_indices = readCompressedIndices(compressed_file, header);

		// Process each batch of indices
		int total_blocks = 0;
		auto start_time = std::chrono::high_resolution_clock::now();

		for (const auto& indices : all_indices) {
			// Decode indices to voxel data
			torch::Tensor voxels = decodeIndices(indices);

			// Write voxels to grid
			int batch_idx_offset = total_blocks;
			writeVoxelsToGrid<GridType>(
			    output_grid,
			    std::vector<openvdb::Coord>(
			        header.origins.begin() + batch_idx_offset,
			        header.origins.begin() + std::min<size_t>(batch_idx_offset + indices.size(0), header.origins.size())),
			    voxels);

			total_blocks += indices.size(0);
		}

		auto end_time = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

		std::cout << "Decoded " << total_blocks << " blocks (8,8,8) in " << duration << " ms\n";
		if (total_blocks) std::cout << "Average: " << duration / total_blocks << " ms per block\n";
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

		// Initialize decoder
		VQVAEDecoder decoder(model_path);

		// Create new grid for output
		auto output_grid = openvdb::FloatGrid::create();


		// Decode compressed data to grid (leaf origins are read from the .vqvdb file)
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
