// VQVAECodec.hpp
//
// Created by zphrfx on 23/06/2025.
//

#pragma once

#include <openvdb/openvdb.h>
#include <torch/script.h>

#include <filesystem>
#include <string>
#include <memory>
#include <unordered_map>

// Forward declarations for performance optimization components
class GPUMemoryPool;
class StreamManager;

/**
 * @class GPUMemoryPool
 * @brief Manages GPU tensor memory allocation and reuse for optimal performance
 */
class GPUMemoryPool {
public:
    explicit GPUMemoryPool(const torch::Device& device);
    ~GPUMemoryPool() = default;
    
    // Get a reusable tensor with the given shape and dtype
    torch::Tensor getTensor(const std::vector<int64_t>& shape, torch::ScalarType dtype);
    
    // Return tensor to pool for reuse
    void returnTensor(torch::Tensor tensor);
    
    // Clear all cached tensors
    void clear();

private:
    struct TensorKey {
        std::vector<int64_t> shape;
        torch::ScalarType dtype;
        
        bool operator==(const TensorKey& other) const {
            return shape == other.shape && dtype == other.dtype;
        }
    };
    
    struct TensorKeyHash {
        size_t operator()(const TensorKey& key) const;
    };
    
    torch::Device device_;
    std::unordered_map<TensorKey, std::vector<torch::Tensor>, TensorKeyHash> tensorCache_;
};

/**
 * @class StreamManager  
 * @brief Manages CUDA streams for overlapping memory transfers with computation
 */
class StreamManager {
public:
    explicit StreamManager(const torch::Device& device);
    ~StreamManager();
    
    // Get next available stream for H2D transfer
    void* getH2DStream();
    
    // Get next available stream for D2H transfer  
    void* getD2HStream();
    
    // Get compute stream
    void* getComputeStream();
    
    // Synchronize all streams
    void synchronizeAll();
    
private:
    torch::Device device_;
    std::vector<void*> h2dStreams_;
    std::vector<void*> d2hStreams_;
    void* computeStream_;
    size_t h2dStreamIndex_;
    size_t d2hStreamIndex_;
};

/**
 * @class VQVAECodec
 * @brief A class for compressing and decompressing OpenVDB FloatGrids using a VQ-VAE model.
 *
 * This class provides a high-level interface to stream VDB leaf nodes to a neural network
 * for compression, and to stream encoded data from a file for decompression. It handles
 * GPU acceleration via PyTorch/libtorch if a CUDA-enabled device is available.
 */
class VQVAECodec {
   public:
	/**
	 * @brief Constructs the VQVAECodec.
	 */
	explicit VQVAECodec();

	/**
	 * @brief Compresses an OpenVDB FloatGrid into a .vqvdb file.
	 *
	 * This method streams leaf blocks from the grid, encodes them in batches,
	 * and writes the result to a file. The file format stores all block origins
	 * first, followed by the concatenated stream of encoded data.
	 *
	 * @param grid The input grid to compress.
	 * @param outPath The path for the output .vqvdb file.
	 * @param batchSize The number of leaf blocks to process in a single GPU batch.
	 */
	void compress(const openvdb::FloatGrid::Ptr& grid, const std::string& outPath, size_t batchSize) const;

	/**
	 * @brief Decompresses a .vqvdb file into an OpenVDB FloatGrid.
	 *
	 * This method streams encoded data from the file, decodes it in batches,
	 * and reconstructs the grid. It's memory-efficient as it doesn't load all
	 * encoded data at once.
	 *
	 * @param inPath The path to the input .vqvdb file.
	 * @param grid A pointer to a FloatGrid which will be created and populated.
	 * @param batchSize The number of leaf blocks to process in a single GPU batch.
	 */
	void decompress(const std::string& inPath, openvdb::FloatGrid::Ptr& grid, size_t batchSize) const;

   private:
	/**
	 * @brief Encodes a batch of VDB leaf data.
	 * @param cpuBatch A pinned CPU tensor of shape [B, 1, D, D, D] with float32 data.
	 * @return A CPU tensor of shape [B, H, W] (or similar) with uint8 quantized indices.
	 */
	torch::Tensor encodeBatch(const torch::Tensor& cpuBatch) const;

	/**
	 * @brief Decodes a batch of quantized indices.
	 * @param cpuBatch A CPU tensor of shape [B, H, W] (or similar) with uint8 quantized indices.
	 * @return A CPU tensor of shape [B, 1, D, D, D] with float32 reconstructed data.
	 */
	torch::Tensor decodeBatch(const torch::Tensor& cpuBatch) const;

	torch::Device device_;

	// Performance optimization components
	mutable std::unique_ptr<GPUMemoryPool> memoryPool_;
	mutable std::unique_ptr<StreamManager> streamManager_;

	static std::tuple<torch::jit::Module, torch::jit::Method, torch::jit::Method> load_embedded_model(const torch::Device& device);
	std::tuple<torch::jit::Module, torch::jit::Method, torch::jit::Method> model_parts_;

	torch::jit::Module model_;
	torch::jit::Method encodeMethod_;
	torch::jit::Method decodeMethod_;
};