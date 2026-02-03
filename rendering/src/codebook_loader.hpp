#pragma once
/**
 * @file codebook_loader.hpp
 * @brief Load VQVAE codebook from binary format for GPU rendering.
 * 
 * Binary format (.bin):
 *   - Header (16 bytes):
 *       - uint32: magic number (0x56514342 = "VQCB")
 *       - uint32: num_embeddings (typically 256)
 *       - uint32: embedding_dim (typically 128)
 *       - uint32: reserved (0)
 *   - Data:
 *       - float32[num_embeddings * embedding_dim]: row-major codebook
 */

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace vqvdb {

constexpr uint32_t CODEBOOK_MAGIC = 0x56514342;  // "VQCB"

struct CodebookHeader {
    uint32_t magic;
    uint32_t numEmbeddings;
    uint32_t embeddingDim;
    uint32_t reserved;
};

struct Codebook {
    uint32_t numEmbeddings = 0;
    uint32_t embeddingDim = 0;
    std::vector<float> data;  // Row-major: [numEmbeddings][embeddingDim]
    
    [[nodiscard]] bool isValid() const noexcept {
        return numEmbeddings > 0 && embeddingDim > 0 && 
               data.size() == static_cast<size_t>(numEmbeddings) * embeddingDim;
    }
    
    [[nodiscard]] const float* embedding(uint32_t index) const noexcept {
        return data.data() + static_cast<size_t>(index) * embeddingDim;
    }
    
    [[nodiscard]] size_t sizeBytes() const noexcept {
        return data.size() * sizeof(float);
    }
};

/**
 * @brief Load a codebook from a binary file.
 * @param path Path to the .bin file created by export_codebook.py
 * @return Codebook structure with loaded data
 * @throws std::runtime_error on file or format errors
 */
inline Codebook loadCodebook(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open codebook file: " + path);
    }
    
    // Read header
    CodebookHeader header{};
    file.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!file || file.gcount() != sizeof(header)) {
        throw std::runtime_error("Failed to read codebook header");
    }
    
    // Validate magic
    if (header.magic != CODEBOOK_MAGIC) {
        throw std::runtime_error("Invalid codebook file: bad magic number");
    }
    
    // Validate dimensions
    if (header.numEmbeddings == 0 || header.embeddingDim == 0) {
        throw std::runtime_error("Invalid codebook dimensions");
    }
    if (header.numEmbeddings > 65536 || header.embeddingDim > 4096) {
        throw std::runtime_error("Codebook dimensions exceed reasonable limits");
    }
    
    // Allocate and read data
    Codebook codebook;
    codebook.numEmbeddings = header.numEmbeddings;
    codebook.embeddingDim = header.embeddingDim;
    
    const size_t numFloats = static_cast<size_t>(header.numEmbeddings) * header.embeddingDim;
    codebook.data.resize(numFloats);
    
    file.read(reinterpret_cast<char*>(codebook.data.data()), numFloats * sizeof(float));
    if (!file) {
        throw std::runtime_error("Failed to read codebook data");
    }
    
    return codebook;
}

}  // namespace vqvdb
