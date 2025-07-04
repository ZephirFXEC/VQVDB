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
#include <cuda_runtime.h>

#include "VQVDB_Reader.hpp"
#include "PerformanceProfiler.hpp"

// Constants for VDB leaf nodes. A leaf is a dense 8x8x8 grid of voxels.
constexpr uint8_t LEAF_DIM = 8;
constexpr uint16_t LEAF_VOXELS = LEAF_DIM * LEAF_DIM * LEAF_DIM;  // 512

#include "bin_model.h"

// =========================================================================================
// CUDA Error Checking Utility
// =========================================================================================
inline void checkCudaError(cudaError_t error, const char* message) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(message) + ": " + cudaGetErrorString(error));
    }
}

// =========================================================================================
// Performance Optimization Component Implementations  
// =========================================================================================

GPUMemoryPool::GPUMemoryPool(const torch::Device& device) : device_(device) {}

torch::Tensor GPUMemoryPool::getTensor(const std::vector<int64_t>& shape, torch::ScalarType dtype) {
    TensorKey key{shape, dtype};
    auto& cache = tensorCache_[key];
    
    if (!cache.empty()) {
        torch::Tensor tensor = cache.back();
        cache.pop_back();
        return tensor;
    }
    
    // Create new tensor if none available in cache
    auto opts = torch::TensorOptions().dtype(dtype).device(device_);
    return torch::empty(shape, opts);
}

void GPUMemoryPool::returnTensor(torch::Tensor tensor) {
    if (!tensor.defined() || tensor.device() != device_) return;
    
    TensorKey key{tensor.sizes().vec(), tensor.scalar_type()};
    tensorCache_[key].push_back(tensor);
}

void GPUMemoryPool::clear() {
    tensorCache_.clear();
}

void GPUMemoryPool::warmup(const std::vector<std::pair<std::vector<int64_t>, torch::ScalarType>>& commonShapes) {
    for (const auto& [shape, dtype] : commonShapes) {
        // Pre-allocate a few tensors of each common size
        for (int i = 0; i < 3; ++i) {
            auto opts = torch::TensorOptions().dtype(dtype).device(device_);
            torch::Tensor tensor = torch::empty(shape, opts);
            returnTensor(tensor);
        }
    }
}

size_t GPUMemoryPool::TensorKeyHash::operator()(const TensorKey& key) const {
    size_t hash = std::hash<int>{}(static_cast<int>(key.dtype));
    for (int64_t dim : key.shape) {
        hash ^= std::hash<int64_t>{}(dim) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    }
    return hash;
}

StreamManager::StreamManager(const torch::Device& device) 
    : device_(device), h2dStreamIndex_(0), d2hStreamIndex_(0) {
    if (device_.is_cuda()) {
        // Create multiple streams for overlapping transfers
        h2dStreams_.resize(2);
        d2hStreams_.resize(2);
        
        for (size_t i = 0; i < 2; ++i) {
            cudaStream_t stream;
            if (cudaStreamCreate(&stream) != cudaSuccess) {
                throw std::runtime_error("Failed to create H2D CUDA stream");
            }
            h2dStreams_[i] = stream;
            
            if (cudaStreamCreate(&stream) != cudaSuccess) {
                throw std::runtime_error("Failed to create D2H CUDA stream");
            }
            d2hStreams_[i] = stream;
        }
        
        cudaStream_t computeStream;
        if (cudaStreamCreate(&computeStream) != cudaSuccess) {
            throw std::runtime_error("Failed to create compute CUDA stream");
        }
        computeStream_ = computeStream;
    }
}

StreamManager::~StreamManager() {
    if (device_.is_cuda()) {
        for (void* stream : h2dStreams_) {
            cudaStreamDestroy(static_cast<cudaStream_t>(stream));
        }
        for (void* stream : d2hStreams_) {
            cudaStreamDestroy(static_cast<cudaStream_t>(stream));
        }
        if (computeStream_) {
            cudaStreamDestroy(static_cast<cudaStream_t>(computeStream_));
        }
    }
}

void* StreamManager::getH2DStream() {
    if (!device_.is_cuda()) return nullptr;
    void* stream = h2dStreams_[h2dStreamIndex_];
    h2dStreamIndex_ = (h2dStreamIndex_ + 1) % h2dStreams_.size();
    return stream;
}

void* StreamManager::getD2HStream() {
    if (!device_.is_cuda()) return nullptr;
    void* stream = d2hStreams_[d2hStreamIndex_];
    d2hStreamIndex_ = (d2hStreamIndex_ + 1) % d2hStreams_.size();
    return stream;
}

void* StreamManager::getComputeStream() {
    return device_.is_cuda() ? computeStream_ : nullptr;
}

void StreamManager::synchronizeAll() {
    if (device_.is_cuda()) {
        for (void* stream : h2dStreams_) {
            checkCudaError(cudaStreamSynchronize(static_cast<cudaStream_t>(stream)), 
                         "H2D stream synchronization failed");
        }
        for (void* stream : d2hStreams_) {
            checkCudaError(cudaStreamSynchronize(static_cast<cudaStream_t>(stream)), 
                         "D2H stream synchronization failed");
        }
        checkCudaError(cudaStreamSynchronize(static_cast<cudaStream_t>(computeStream_)), 
                     "Compute stream synchronization failed");
    }
}

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

		// OPTIMIZATION: Create tensor with channel dimension upfront to avoid unsqueeze copy
		auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU).pinned_memory(true);
		torch::Tensor batch = torch::empty({static_cast<long>(B), 1, LEAF_DIM, LEAF_DIM, LEAF_DIM}, opts);
		std::vector<openvdb::Coord> origins(B);

		float* dstBase = batch.data_ptr<float>();
		tbb::parallel_for(tbb::blocked_range<size_t>(0, B), [&](const tbb::blocked_range<size_t>& r) {
			for (size_t i = r.begin(); i != r.end(); ++i) {
				const auto& leaf = leafManager_.leaf(start + i);
				origins[i] = leaf.origin();
				const float* src = leaf.buffer().data();
				
				// OPTIMIZATION: Prefetch next leaf data for better cache performance
				if (i + 1 < r.end() && start + i + 1 < totalLeaves_) {
					const auto& nextLeaf = leafManager_.leaf(start + i + 1);
					__builtin_prefetch(nextLeaf.buffer().data(), 0, 3);
				}
				
				// Direct copy to the correct position in 5D tensor (skip channel dim offset)
				float* dst = dstBase + i * LEAF_VOXELS;
				std::memcpy(dst, src, LEAF_VOXELS * sizeof(float));
			}
		});

		return {batch, origins};  // Already has channel dim: [B, 1, D, D, D]
	}

   private:
	const openvdb::tree::LeafManager<openvdb::FloatTree>& leafManager_;
	size_t currentPos_;
	const size_t totalLeaves_;
};


// =========================================================================================
// VQVAECodec Method Implementations
// =========================================================================================

// --- Implementation of the private static helper function ---
std::tuple<torch::jit::Module, torch::jit::Method, torch::jit::Method> VQVAECodec::load_embedded_model(const torch::Device& device) {
	// Create a string stream from the embedded byte array
	std::string model_string(reinterpret_cast<const char*>(g_model_data), g_model_data_size);
	std::istringstream stream(model_string);

	torch::jit::Module module;
	try {
		// Load the model from the stream (onto CPU by default)
		module = torch::jit::load(stream);
	} catch (const c10::Error& e) {
		throw std::runtime_error("Failed to load TorchScript model from memory: " + std::string(e.what()));
	}

	// Move the loaded module to the target device
	module.to(device);
	module.eval();

	// Get the methods from the now-configured module
	torch::jit::Method encode_method = module.get_method("encode");
	torch::jit::Method decode_method = module.get_method("decode");

	std::cout << "VQVAECodec: Model successfully loaded from memory onto device: " << device << '\n';

	// Return all the constructed objects in a tuple
	return {std::move(module), std::move(encode_method), std::move(decode_method)};
}

VQVAECodec::VQVAECodec()
    : device_(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU),
      model_parts_(load_embedded_model(device_)),
      model_(std::get<0>(model_parts_)),
      encodeMethod_(std::get<1>(model_parts_)),
      decodeMethod_(std::get<2>(model_parts_)),
      memoryPool_(std::make_unique<GPUMemoryPool>(device_)),
      streamManager_(std::make_unique<StreamManager>(device_)) {
    
    // Pre-warm memory pool with common tensor sizes for VDB leaf blocks
    if (device_.is_cuda()) {
        std::vector<std::pair<std::vector<int64_t>, torch::ScalarType>> commonShapes = {
            {{16, 1, 8, 8, 8}, torch::kFloat32},  // Typical batch size with channel dim
            {{32, 1, 8, 8, 8}, torch::kFloat32},  // Larger batch
            {{16, 4, 4}, torch::kU8},             // Typical encoded output size
            {{32, 4, 4}, torch::kU8},             // Larger encoded batch
        };
        memoryPool_->warmup(commonShapes);
    }
}

VQVAECodec::~VQVAECodec() {
    // Synchronize all GPU operations before cleanup
    if (streamManager_ && device_.is_cuda()) {
        streamManager_->synchronizeAll();
    }
    // Clear GPU memory pool
    if (memoryPool_) {
        memoryPool_->clear();
    }
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
	
	// Print detailed performance report
	PerformanceProfiler::getInstance().printReport();
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
	
	// Print detailed performance report
	PerformanceProfiler::getInstance().printReport();
}

torch::Tensor VQVAECodec::encodeBatch(const torch::Tensor& cpuBatch) const {
	PROFILE_SCOPE("VQVAECodec::encodeBatch");
	torch::NoGradGuard nograd;
	
	// Get reusable GPU tensor from memory pool
	PROFILE_START("encode_memory_pool");
	auto gpuTensorShape = cpuBatch.sizes().vec();
	torch::Tensor gpuTensor = memoryPool_->getTensor(gpuTensorShape, torch::kFloat32);
	PROFILE_END("encode_memory_pool");
	
	// Use optimized copy if CUDA is available
	PROFILE_START("encode_h2d_transfer");
	if (device_.is_cuda()) {
		// Asynchronous H2D transfer using dedicated stream
		auto h2dStream = static_cast<cudaStream_t>(streamManager_->getH2DStream());
		gpuTensor.copy_(cpuBatch, /*non_blocking=*/true);
		
		// Synchronize H2D transfer before computation
		checkCudaError(cudaStreamSynchronize(h2dStream), "H2D stream synchronization failed");
	} else {
		gpuTensor.copy_(cpuBatch);
	}
	PROFILE_END("encode_h2d_transfer");
	
	// Perform encoding on compute stream
	PROFILE_START("encode_computation");
	const torch::Tensor result = encodeMethod_({gpuTensor}).toTensor();
	
	// Convert to uint8 on GPU to avoid redundant transfers
	torch::Tensor gpuResult = result.to(torch::kU8);
	PROFILE_END("encode_computation");
	
	// Create output tensor with pinned memory for faster D2H transfer
	PROFILE_START("encode_d2h_transfer");
	auto opts = torch::TensorOptions().dtype(torch::kU8).device(torch::kCPU).pinned_memory(device_.is_cuda());
	torch::Tensor cpuResult = torch::empty(gpuResult.sizes(), opts);
	
	// Asynchronous D2H transfer
	if (device_.is_cuda()) {
		auto d2hStream = static_cast<cudaStream_t>(streamManager_->getD2HStream());
		cpuResult.copy_(gpuResult, /*non_blocking=*/true);
		checkCudaError(cudaStreamSynchronize(d2hStream), "D2H stream synchronization failed");
	} else {
		cpuResult.copy_(gpuResult);
	}
	PROFILE_END("encode_d2h_transfer");
	
	// Return tensors to memory pool for reuse
	memoryPool_->returnTensor(gpuTensor);
	
	return cpuResult;
}

torch::Tensor VQVAECodec::decodeBatch(const torch::Tensor& cpuBatch) const {
	PROFILE_SCOPE("VQVAECodec::decodeBatch");
	torch::NoGradGuard nograd;
	
	// Optimize: Skip unnecessary uint8 -> int64 conversion
	// Convert directly to float on GPU if the model supports it
	PROFILE_START("decode_memory_pool");
	auto gpuTensorShape = cpuBatch.sizes().vec();
	torch::Tensor gpuTensor = memoryPool_->getTensor(gpuTensorShape, torch::kU8);
	PROFILE_END("decode_memory_pool");
	
	// Asynchronous H2D transfer
	PROFILE_START("decode_h2d_transfer");
	if (device_.is_cuda()) {
		auto h2dStream = static_cast<cudaStream_t>(streamManager_->getH2DStream());
		gpuTensor.copy_(cpuBatch, /*non_blocking=*/true);
		checkCudaError(cudaStreamSynchronize(h2dStream), "Decode H2D stream synchronization failed");
	} else {
		gpuTensor.copy_(cpuBatch);
	}
	PROFILE_END("decode_h2d_transfer");
	
	// Convert to long only if necessary for the model
	PROFILE_START("decode_computation");
	torch::Tensor gpuLongTensor = gpuTensor.to(torch::kLong);
	
	// Perform decoding
	const torch::Tensor result = decodeMethod_({gpuLongTensor}).toTensor();
	PROFILE_END("decode_computation");
	
	// Create output tensor with pinned memory
	PROFILE_START("decode_d2h_transfer");
	auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU).pinned_memory(device_.is_cuda());
	torch::Tensor cpuResult = torch::empty(result.sizes(), opts);
	
	// Asynchronous D2H transfer
	if (device_.is_cuda()) {
		auto d2hStream = static_cast<cudaStream_t>(streamManager_->getD2HStream());
		cpuResult.copy_(result, /*non_blocking=*/true);
		checkCudaError(cudaStreamSynchronize(d2hStream), "Decode D2H stream synchronization failed");
	} else {
		cpuResult.copy_(result);
	}
	PROFILE_END("decode_d2h_transfer");
	
	// Return tensors to memory pool
	memoryPool_->returnTensor(gpuTensor);
	
	return cpuResult;
}