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

#include "Kernel.cuh"

// Host function to launch lookup kernel
void lookupCodebook(const torch::Tensor& codebook, const torch::Tensor& indices, torch::Tensor& output) {
	// Get tensor shapes and sizes
	int batch_size = indices.size(0);
	int depth = indices.size(1);
	int height = indices.size(2);
	int width = indices.size(3);
	int embedding_dim = codebook.size(1);
	int num_embeddings = codebook.size(0);

	// Get raw pointers to tensor data
	const float* codebook_ptr = codebook.data_ptr<float>();
	const uint16_t* indices_ptr = reinterpret_cast<const uint16_t*>(indices.data_ptr<int>());
	float* output_ptr = output.data_ptr<float>();

	// Define block and grid dimensions
	dim3 block(8, 8, 4);

	// We'll encode the batch dimension into the z dimension by multiplying
	dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y,
	          batch_size * ((depth + block.z - 1) / block.z)  // Combine batch and depth dimensions
	);

	// Launch kernel
	lookupCodebookKernel<<<grid, block>>>(codebook_ptr, indices_ptr, output_ptr, batch_size, depth, height, width, embedding_dim,
	                                      num_embeddings);
	// Check for errors
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
		throw std::runtime_error("CUDA kernel launch failed");
	}
}


// Header for compressed VDB index file
struct CompressedHeader {
	char magic[5];                // "VQVDB"
	uint8_t version;              // Version number
	uint32_t numEmbeddings;       // Number of codebook entries
	uint8_t numDimensions;        // Number of dimensions in the index tensor (typically 3 for [d,h,w])
	std::vector<uint16_t> shape;  // Shape of each index tensor
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
	if (header.version != 1) {
		throw std::runtime_error("Unsupported file version");
	}

	// Read number of embeddings
	file.read(reinterpret_cast<char*>(&header.numEmbeddings), sizeof(uint32_t));

	// Read number of dimensions
	file.read(reinterpret_cast<char*>(&header.numDimensions), 1);

	// Read shape dimensions
	header.shape.resize(header.numDimensions);
	file.read(reinterpret_cast<char*>(header.shape.data()), header.numDimensions * sizeof(uint16_t));

	// Calculate bits needed per index
	int bitsPerIndex = 32 - _tzcnt_u32(header.numEmbeddings - 1);
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
		std::vector<int> indices_data(totalElements);

		if (bytesPerIndex == 1) {
			// Read as bytes and convert to int
			std::vector<uint8_t> temp(totalElements);
			file.read(reinterpret_cast<char*>(temp.data()), totalElements);
			std::transform(temp.begin(), temp.end(), indices_data.begin(), [](uint8_t val) { return static_cast<int>(val); });
		} else {
			// Read as uint16_t and convert to int
			std::vector<uint16_t> temp(totalElements);
			file.read(reinterpret_cast<char*>(temp.data()), totalElements * sizeof(uint16_t));
			std::transform(temp.begin(), temp.end(), indices_data.begin(), [](uint16_t val) { return static_cast<int>(val); });
		}

		// Create tensor and append to list
		std::vector<int64_t> tensor_shape;
		tensor_shape.push_back(batchSize);
		for (uint16_t dim : header.shape) {
			tensor_shape.push_back(dim);
		}

		torch::Tensor indices_tensor =
		    torch::from_blob(indices_data.data(), tensor_shape, torch::TensorOptions().dtype(torch::kInt32)).clone();

		all_indices.push_back(indices_tensor);
	}

	return all_indices;
}

// Write voxels back to OpenVDB grid
template <typename GridType>
void writeVoxelsToGrid(typename GridType::Ptr grid, const std::vector<openvdb::Coord>& origins, const torch::Tensor& voxelData) {
	using ValueT = typename GridType::ValueType;

	// Ensure tensor is on CPU and convert to correct data type
	auto cpu_data = voxelData.to(torch::kCPU);
	auto cpu_data_float = cpu_data.to(torch::kFloat32);

	int batch_size = cpu_data.size(0);
	int depth = cpu_data.size(2);
	int height = cpu_data.size(3);
	int width = cpu_data.size(4);

	// For each leaf
	for (int b = 0; b < batch_size && b < static_cast<int>(origins.size()); b++) {
		openvdb::Coord origin = origins[b];

		// For each voxel in the leaf
		for (int z = 0; z < depth; z++) {
			for (int y = 0; y < height; y++) {
				for (int x = 0; x < width; x++) {
					// Get voxel value
					float value = cpu_data_float[b][0][z][y][x].item<float>();

					// Set voxel in grid
					openvdb::Coord xyz(origin.x() + x, origin.y() + y, origin.z() + z);
					grid->setValue(xyz, static_cast<ValueT>(value));
				}
			}
		}
	}
}

// Main decoder class
class VQVAEDecoder {
   public:
	VQVAEDecoder(const std::string& model_path) : device_(torch::kCPU) {
		try {
			// Load the TorchScript model
			model_ = torch::jit::load(model_path);
			model_.eval();

			/*// Move model to GPU if available
			if (torch::jit::cuda::is_available()) {
			    model_.to(torch::kCUDA);
			    device_ = torch::kCUDA;
			    std::cout << "Using CUDA device for decoding" << std::endl;
			} else {
			}
			*/
			device_ = torch::kCPU;
			std::cout << "Using CPU for decoding (CUDA not available)" << std::endl;

			// Extract codebook from the model
			extractCodebook();

		} catch (const c10::Error& e) {
			std::cerr << "Error loading the model: " << e.what() << std::endl;
			throw;
		}
	}

	void extractCodebook() {
		// We need to extract the codebook weights from the model
		// This is model-specific and depends on how your model is structured
		for (const auto& p : model_.named_parameters()) {
			if (p.name.find("vq.embedding.weight") != std::string::npos) {
				codebook_ = p.value.clone().to(device_);
				std::cout << "Found codebook with shape: [" << codebook_.size(0) << ", " << codebook_.size(1) << "]" << std::endl;
				return;
			}
		}
		throw std::runtime_error("Could not find codebook in the model");
	}

	torch::Tensor decodeIndices(const torch::Tensor& indices) {
		torch::NoGradGuard no_grad;

		// Create output tensor to hold embeddings
		int batch_size = indices.size(0);
		int depth = indices.size(1);
		int height = indices.size(2);
		int width = indices.size(3);
		int embedding_dim = codebook_.size(1);

		auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device_);

		torch::Tensor embeddings = torch::empty({batch_size, embedding_dim, depth, height, width}, options);

		// Look up embeddings using CUDA kernel
		lookupCodebook(codebook_, indices.to(device_, torch::kInt32), embeddings);

		// Pass through decoder
		std::vector<torch::jit::IValue> inputs;
		inputs.push_back(embeddings);

		// Use the decode method from the TorchScript model
		torch::Tensor output;
		try {
			auto result = model_.get_method("decode")(inputs);
			output = result.toTensor();
		} catch (const c10::Error& e) {
			// Fallback if 'decode' method isn't available
			std::cout << "Decode method not found, using forward pass with embeddings" << std::endl;
			inputs.clear();
			inputs.push_back(embeddings);
			auto result = model_.forward(inputs);

			// Extract the first element if it's a tuple (common for VQ-VAE forward to return
			// (reconstruction, vq_loss, indices))
			if (result.isTuple()) {
				auto resultTuple = result.toTuple();
				if (resultTuple->elements().size() > 0) {
					output = resultTuple->elements()[0].toTensor();
				} else {
					throw std::runtime_error("Forward method returned empty tuple");
				}
			} else if (result.isTensor()) {
				output = result.toTensor();
			} else {
				throw std::runtime_error("Forward method returned unexpected type");
			}
		}

		return output;
	}

	template <typename GridType>
	void decodeToGrid(const std::string& compressed_file, const std::vector<openvdb::Coord>& leaf_origins,
	                  typename GridType::Ptr output_grid, int batch_size = 64) {
		// Read compressed indices
		CompressedHeader header;
		std::vector<torch::Tensor> all_indices = readCompressedIndices(compressed_file, header);

		// Process each batch of indices
		int total_leaves = 0;
		auto start_time = std::chrono::high_resolution_clock::now();

		for (const auto& indices : all_indices) {
			// Decode indices to voxel data
			torch::Tensor voxels = decodeIndices(indices);

			// Write voxels to grid
			int batch_idx_offset = total_leaves;
			writeVoxelsToGrid<GridType>(
			    output_grid,
			    std::vector<openvdb::Coord>(
			        leaf_origins.begin() + batch_idx_offset,
			        leaf_origins.begin() + std::min<size_t>(batch_idx_offset + indices.size(0), leaf_origins.size())),
			    voxels);

			total_leaves += indices.size(0);
		}

		auto end_time = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

		std::cout << "Decoded " << total_leaves << " leaves in " << duration << " ms" << std::endl;
		if (total_leaves > 0) {
			std::cout << "Average: " << duration / total_leaves << " ms per leaf" << std::endl;
		}
	}

   private:
	torch::jit::Module model_;
	torch::Tensor codebook_;
	torch::Device device_;
};

// Collect all leaf origins from a grid
template <typename GridType>
std::vector<openvdb::Coord> collectLeafOrigins(typename GridType::Ptr grid) {
	std::vector<openvdb::Coord> origins;

	// Iterate over all leaf nodes
	for (auto leaf_iter = grid->tree().beginLeaf(); leaf_iter; ++leaf_iter) {
		origins.push_back(leaf_iter->origin());
	}

	return origins;
}

int main(int argc, char** argv) {
	if (argc < 5) {
		std::cout << "Usage: " << argv[0] << " <model.pt> <compressed_indices.bin> <input_template.vdb> <output.vdb> [grid_name]"
		          << std::endl;
		return 1;
	}

	const std::string model_path = argv[1];
	const std::string compressed_path = argv[2];
	const std::string input_template_path = argv[3];
	const std::string output_path = argv[4];
	const std::string grid_name = (argc > 5) ? argv[5] : "";

	try {
		// Initialize OpenVDB
		openvdb::initialize();

		// Load template grid (for structure)
		openvdb::io::File template_file(input_template_path);
		template_file.open();

		openvdb::GridBase::Ptr base_grid;
		if (grid_name.empty()) {
			base_grid = template_file.readGrid(template_file.beginName().gridName());
		} else {
			base_grid = template_file.readGrid(grid_name);
		}

		if (!base_grid) {
			std::cerr << "Error: Could not read grid from template file" << std::endl;
			return 1;
		}

		// Initialize decoder
		VQVAEDecoder decoder(model_path);

		// Process based on grid type
		if (auto float_grid = openvdb::GridBase::grid<openvdb::FloatGrid>(base_grid)) {
			// Get leaf origins
			std::vector<openvdb::Coord> origins = collectLeafOrigins<openvdb::FloatGrid>(float_grid);

			// Create new grid for output
			auto output_grid = openvdb::FloatGrid::create();
			output_grid->setTransform(float_grid->transformPtr());
			output_grid->setGridClass(float_grid->getGridClass());
			output_grid->setName(float_grid->getName());

			// Decode compressed data to grid
			decoder.decodeToGrid<openvdb::FloatGrid>(compressed_path, origins, output_grid);

			// Write output grid
			openvdb::io::File file(output_path);
			file.write({output_grid});
			file.close();

		} else {
			std::cerr << "Error: Currently only supports FloatGrid" << std::endl;
			return 1;
		}

		template_file.close();
		std::cout << "Successfully decoded and saved to " << output_path << std::endl;

	} catch (const std::exception& e) {
		std::cerr << "Error: " << e.what() << std::endl;
		return 1;
	}

	return 0;
}