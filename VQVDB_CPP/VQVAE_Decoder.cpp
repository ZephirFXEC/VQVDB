//
// Created by zphrfx on 04/06/2025.
//
#include "VQVAE_Decoder.hpp"

#include "Utils/VQVDB_Reader.hpp"

void VQVAEDecoder::writeVoxelsToGrid(openvdb::tree::ValueAccessor<openvdb::FloatTree>& grid, const std::vector<openvdb::Coord>& origins,
                                     const float* __restrict__ raw, const int B) {
	// Precompute per‐block stride: B × C × D × H × W with C==1
	constexpr int64_t blockSize = BLOCK_DIM * BLOCK_DIM * BLOCK_DIM;  // 512

	for (int b = 0; b < B && b < (int)origins.size(); ++b) {
		const auto& origin = origins[b];
		const float* blockPtr = raw + b * blockSize;

		// Each voxel
		for (int z = 0; z < BLOCK_DIM; ++z) {
			int64_t zOff = z * (BLOCK_DIM * BLOCK_DIM);
			for (int y = 0; y < BLOCK_DIM; ++y) {
				int64_t yzOff = zOff + y * BLOCK_DIM;
				for (int x = 0; x < BLOCK_DIM; ++x) {
					float v = blockPtr[yzOff + x];
					grid.setValue(openvdb::Coord(origin.x() + x, origin.y() + y, origin.z() + z), v);
				}
			}
		}
	}
}

VQVAEDecoder::VQVAEDecoder(const std::string& model_path)
    : device_(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU),
      model_(torch::jit::load(model_path)),
      decodeMethod_(model_.get_method("decode")) {
	std::cout << "Using device: " << device_ << std::endl;

	model_.eval();
	model_.to(device_, torch::kFloat32);

	// Lookup method "decode" from the JIT module
	if (!model_.find_method("decode")) {
		throw std::runtime_error("The provided PyTorch JIT model has no 'decode' method.");
	}
}

// Core function: Decodes a batch of indices into a [B,1,D,H,W] voxel tensor
torch::Tensor VQVAEDecoder::decodeIndices(const torch::Tensor& indices) const {
	torch::NoGradGuard no_grad;
	// Convert indices to device-friendly type
	auto final_indices = indices.to(device_, torch::kInt64, /*non_blocking=*/true);
	auto result = decodeMethod_({final_indices}).toTensor();  // shape [B,1,D,H,W]
	return result;                                            // Still on device; final reshape done on CPU side
}


void VQVAEDecoder::decodeToGrid(const std::string& compressed_file, const openvdb::FloatGrid::Ptr& output_grid) const {
	CompressedIndexReader reader(compressed_file);
	int total_blocks = 0;
	auto start_time = std::chrono::high_resolution_clock::now();

	openvdb::tree::ValueAccessor<openvdb::FloatTree> accessor(output_grid->tree());

	while (auto opt = reader.readNextBatch(8192)) {
		torch::Tensor& indices = *opt;                        // [B, …]
		torch::Tensor voxelsDevice = decodeIndices(indices);  // [B,1,D,H,W] on GPU/CPU

		int B = indices.size(0);
		auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU).pinned_memory(true);
		auto voxels = voxelsDevice.to(opts);
		const float* raw = voxels.data_ptr<float>();

		// slice out exactly B origins
		int start = total_blocks;
		int end = std::min(total_blocks + B, static_cast<int>(reader.header.origins.size()));
		std::vector<openvdb::Coord> origins_slice(reader.header.origins.begin() + start, reader.header.origins.begin() + end);

		writeVoxelsToGrid(accessor, origins_slice, raw, B);
		total_blocks += B;
	}

	auto end_time = std::chrono::high_resolution_clock::now();
	auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
	std::cout << "Decoded " << total_blocks << " blocks in " << ms << " ms\n";
	if (total_blocks) {
		std::cout << "Average " << static_cast<double>(ms * 1000) / total_blocks << " us/block\n";
	}
}


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
		decoder.decodeToGrid(compressed_path, output_grid);

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
