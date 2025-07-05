// VQVAECodec.cpp
//
// Created by zphrfx on 23/06/2025.
//

#include "VQVAECodec.hpp"

#include <openvdb/tools/GridOperators.h>
#include <torch/cuda.h>

#include <chrono>
#include <iostream>
#include <vector>

#include "Profiler.hpp"
#include "VQVDB_Reader.hpp"

// Constants for VDB leaf nodes. A leaf is a dense 8x8x8 grid of voxels.
constexpr uint8_t LEAF_DIM = 8;
constexpr uint16_t LEAF_VOXELS = LEAF_DIM * LEAF_DIM * LEAF_DIM;  // 512

#include "bin_decoder.h"
#include "bin_encoder.h"


// =========================================================================================
// Helper Streamer for Reading VDB Leaf Blocks (for Compression)
// =========================================================================================
class VDBInputBlockStreamer {
   public:
	explicit VDBInputBlockStreamer(const openvdb::tree::LeafManager<openvdb::FloatTree>& leafManager)
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
				const float* src = leaf.buffer().data();
				float* dst = dstBase + i * LEAF_VOXELS;
				std::memcpy(dst, src, LEAF_VOXELS * sizeof(float));
			}
		});

		return {batch, origins};  // Add channel dim: [B, 1, D, D, D]
	}

   private:
	const openvdb::tree::LeafManager<openvdb::FloatTree>& leafManager_;
	size_t currentPos_;
	const size_t totalLeaves_;
};


// =========================================================================================
// VQVAECodec Method Implementations
// =========================================================================================

VQVAECodec::VQVAECodec()
    : env_(ORT_LOGGING_LEVEL_WARNING, "VQVAE-ONNX-Codec"), device_(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU) {
	PROFILE_SCOPE("Load ONNX Models");

	Ort::SessionOptions session_options;
	session_options.SetIntraOpNumThreads(1);
	session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

	if (device_.is_cuda()) {
		std::cout << "VQVAECodec: CUDA is available. Using CUDA Execution Provider." << std::endl;
		OrtCUDAProviderOptions cuda_options{};
		session_options.AppendExecutionProvider_CUDA(cuda_options);
		memory_info_cuda_ = Ort::MemoryInfo("Cuda", OrtArenaAllocator, 0, OrtMemTypeDefault);
	} else {
		std::cout << "VQVAECodec: CUDA not found. Using CPU Execution Provider." << std::endl;
	}

	// Load encoder model from embedded data
	try {
		encoder_session_ = std::make_unique<Ort::Session>(env_, g_encoder_data, g_encoder_data_size, session_options);
	} catch (const Ort::Exception& e) {
		throw std::runtime_error("Failed to load ONNX encoder model: " + std::string(e.what()));
	}

	// Load decoder model from embedded data
	try {
		decoder_session_ = std::make_unique<Ort::Session>(env_, g_decoder_data, g_decoder_data_size, session_options);
	} catch (const Ort::Exception& e) {
		throw std::runtime_error("Failed to load ONNX decoder model: " + std::string(e.what()));
	}

	// Set input/output names (these must match what you defined during ONNX export)
	encoder_input_names_ = {"input"};
	encoder_output_names_ = {"output"};
	decoder_input_names_ = {"input"};
	decoder_output_names_ = {"output"};

	std::cout << "VQVAECodec: ONNX encoder and decoder models loaded successfully." << std::endl;
}

void VQVAECodec::compress(const openvdb::FloatGrid::Ptr& grid, const std::string& outPath, const size_t batchSize) const {
	const auto t0 = std::chrono::high_resolution_clock::now();
	const openvdb::tree::LeafManager<openvdb::FloatTree> leafMgr(grid->tree());
	const int64_t N = leafMgr.leafCount();

	if (N == 0) {
		std::cout << "Grid has no active voxels. Nothing to compress.\n";
		return;
	}

	// --- Step 1: Get latent shape from a dummy tensor ---
	// (This part remains the same, but is necessary for the header)
	std::vector<int64_t> latentShapeVec;
	{
		torch::NoGradGuard nograd;
		auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
		torch::Tensor dummyInput = torch::randn({1, 1, LEAF_DIM, LEAF_DIM, LEAF_DIM}, opts);
		auto idx = encodeBatch(dummyInput).sizes();  // e.g., [1, H, W]
		latentShapeVec.assign(idx.begin() + 1, idx.end());
	}

	// --- Step 2: Create the optimized writer ---
	// The writer handles the header and buffered I/O automatically.
	VDBStreamWriter writer(outPath, 256, latentShapeVec, N);  // 256 numEmbeddings hardcoded

	// --- Step 3: Stream, encode, and write data blocks ---
	VDBInputBlockStreamer streamer(leafMgr);
	int64_t done = 0;
	const auto t1 = std::chrono::high_resolution_clock::now();
	std::cout << "Starting compression of " << N << " blocks using optimized writer...\n";

	while (streamer.hasNext()) {
		auto [hostTensor, origins] = streamer.nextBatch(batchSize);
		if (hostTensor.numel() == 0) break;

		torch::Tensor encodedIndices = encodeBatch(hostTensor);  // Returns uint8 on CPU

		// The writer handles buffering and interleaving origins and data
		writer.writeBatch(encodedIndices, origins);

		done += hostTensor.size(0);
		std::cout << "\rProcessed " << done << " / " << N << " blocks..." << std::flush;
	}

	const auto t2 = std::chrono::high_resolution_clock::now();
	const auto setup = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
	const auto comp = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
	printf("\n-- Setup  : %lld ms\n-- Encode : %lld ms\n", setup, comp);
}


void VQVAECodec::decompress(const std::string& inPath, openvdb::FloatGrid::Ptr& grid, const size_t batchSize) const {
	const auto t0 = std::chrono::high_resolution_clock::now();

	// --- Step 1: Use the optimized stream reader ---
	// The reader handles the header and buffered I/O automatically.
	VDBStreamReader streamer(inPath);
	std::cout << "Starting decompression using optimized reader...\n";

	// --- Step 2: Prepare the output grid ---
	grid = openvdb::FloatGrid::create();
	auto accessor = grid->getAccessor();

	// --- Step 3: Stream, decode, and write data to the new grid ---
	const auto t1 = std::chrono::high_resolution_clock::now();
	int64_t done = 0;

	while (streamer.hasNext()) {
		EncodedBatch batch = streamer.nextBatch(batchSize);
		if (batch.data.numel() == 0) break;

		torch::Tensor decodedData = decodeBatch(batch.data);  // Returns float32 on CPU

		const float* dataPtr = decodedData.data_ptr<float>();

		for (size_t i = 0; i < batch.origins.size(); ++i) {
			const openvdb::Coord& origin = batch.origins[i];

			// This creates the leaf in the tree if it doesn't exist.
			if (auto* leaf = accessor.touchLeaf(origin)) {
				const float* src = dataPtr + i * LEAF_VOXELS;
				std::memcpy(leaf->buffer().data(), src, LEAF_VOXELS * sizeof(float));
				leaf->setValuesOn();
			}
		}
		done += batch.origins.size();
		std::cout << "\rProcessed " << done << " blocks..." << std::flush;
	}

	std::cout << "\nDecompression finished.\n";

	const auto t2 = std::chrono::high_resolution_clock::now();
	const auto setup = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
	const auto decomp = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
	printf("\n-- Setup  : %lld ms\n-- Decode : %lld ms\n", setup, decomp);
}

// Helper function to map torch dtype to ONNX dtype
ONNXTensorElementDataType to_onnx_type(const c10::ScalarType& dtype) {
	switch (dtype) {
		case torch::kFloat32:
			return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
		case torch::kFloat16:
			return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
		case torch::kInt64:
			return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
		case torch::kInt32:
			return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
		case torch::kInt8:
			return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
		case torch::kUInt8:
			return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
		default:
			throw std::invalid_argument("Unsupported torch dtype for ONNX conversion");
	}
}

torch::Tensor VQVAECodec::encodeBatch(const torch::Tensor& cpuBatch) const {
	PROFILE_SCOPE("EncodeBatch Total");
	torch::NoGradGuard nograd;

	// --- H2D Copy ---
	// Note: If model is FP16, use .to(torch::kHalf)
	PROFILE_START("Encode::H2D Copy");
	torch::Tensor gpuTensor = cpuBatch.to(device_, /*non_blocking=*/true);
	if (device_.is_cuda()) torch::cuda::synchronize();
	PROFILE_END("Encode::H2D Copy");

	// --- Create ONNX Runtime Input Tensor ---
	std::vector<int64_t> input_shape = gpuTensor.sizes().vec();
	Ort::Value input_tensor =
	    Ort::Value::CreateTensor(memory_info_cuda_, gpuTensor.data_ptr(), gpuTensor.numel() * gpuTensor.element_size(), input_shape.data(),
	                             input_shape.size(), to_onnx_type(gpuTensor.scalar_type()));

	// --- Inference ---
	PROFILE_START("Encode::Inference");
	auto output_tensors =
	    encoder_session_->Run(Ort::RunOptions{nullptr}, encoder_input_names_.data(), &input_tensor, 1, encoder_output_names_.data(), 1);
	if (device_.is_cuda()) torch::cuda::synchronize();
	PROFILE_END("Encode::Inference");

	// --- Wrap ONNX Output in torch::Tensor (no copy) ---
	Ort::Value& output_tensor = output_tensors.front();
	auto* float_vals = output_tensor.GetTensorMutableData<int64_t>();  // Encoder outputs int64
	Ort::TensorTypeAndShapeInfo shape_info = output_tensor.GetTensorTypeAndShapeInfo();
	auto output_shape = shape_info.GetShape();

	// The output tensor is already on the GPU. We can wrap it without a D2H copy.
	auto gpu_result = torch::from_blob(float_vals, torch::IntArrayRef(output_shape), torch::kInt64).to(device_);

	// --- D2H Copy ---
	PROFILE_START("Encode::D2H Copy");
	torch::Tensor cpu_result = gpu_result.to(torch::kCPU, torch::kUInt8);  // Final conversion to uint8
	PROFILE_END("Encode::D2H Copy");

	return cpu_result;
}

torch::Tensor VQVAECodec::decodeBatch(const torch::Tensor& cpuBatch) const {
	PROFILE_SCOPE("DecodeBatch Total");
	torch::NoGradGuard nograd;

	// --- H2D Copy ---
	// Decoder input is int64 indices
	PROFILE_START("Decode::H2D Copy");
	torch::Tensor gpuTensor = cpuBatch.to(device_, torch::kInt64, /*non_blocking=*/true);
	if (device_.is_cuda()) torch::cuda::synchronize();
	PROFILE_END("Decode::H2D Copy");

	// --- Create ONNX Runtime Input Tensor ---
	std::vector<int64_t> input_shape = gpuTensor.sizes().vec();
	Ort::Value input_tensor =
	    Ort::Value::CreateTensor(memory_info_cuda_, gpuTensor.data_ptr<int64_t>(), gpuTensor.numel() * sizeof(int64_t), input_shape.data(),
	                             input_shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);

	// --- Inference ---
	PROFILE_START("Decode::Inference");
	auto output_tensors =
	    decoder_session_->Run(Ort::RunOptions{nullptr}, decoder_input_names_.data(), &input_tensor, 1, decoder_output_names_.data(), 1);
	if (device_.is_cuda()) torch::cuda::synchronize();
	PROFILE_END("Decode::Inference");

	// --- Wrap ONNX Output in torch::Tensor (no copy) ---
	Ort::Value& output_tensor = output_tensors.front();
	auto* float_vals = output_tensor.GetTensorMutableData<float>();  // Decoder outputs floats
	Ort::TensorTypeAndShapeInfo shape_info = output_tensor.GetTensorTypeAndShapeInfo();
	auto output_shape = shape_info.GetShape();

	// Note: If model is FP16, use torch::kHalf
	auto gpu_result = torch::from_blob(float_vals, torch::IntArrayRef(output_shape), torch::kFloat32).to(device_);

	// --- D2H Copy ---
	PROFILE_START("Decode::D2H Copy");
	torch::Tensor cpu_result = gpu_result.to(torch::kCPU);
	PROFILE_END("Decode::D2H Copy");

	return cpu_result;
}
