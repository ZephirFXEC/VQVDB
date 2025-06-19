#pragma once

#include <openvdb/Types.h>
#include <torch/torch.h>

#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>

struct CompressedHeader {
	char magic[5];                        // "VQVDB"
	uint8_t version;                      // Version number
	uint32_t numEmbeddings;               // Number of codebook entries
	uint8_t numDimensions;                // Number of dimensions in the index tensor (typically 3 for [d,h,w])
	std::vector<uint16_t> shape;          // Shape of each index tensor
	uint32_t leafCount = 0;               // Number of leaf nodes (optional)
	std::vector<openvdb::Coord> origins;  // Leaf origins if present
};

class CompressedIndexReader {
   public:
	// Make the header public so the decoder can access it for shape/origins.
	CompressedHeader header;

	// Constructor: Opens the file and reads the header.
	explicit CompressedIndexReader(const std::string& filename) : file_(filename, std::ios::binary) {
		if (!file_) {
			throw std::runtime_error("Failed to open compressed file: " + filename);
		}
		readHeader();
	}

	// Reads the next batch of indices from the file.
	// Returns an empty optional if the end of the file is reached.
	std::optional<torch::Tensor> readNextBatch(size_t batchSize);

   private:
	void readHeader();
	std::ifstream file_;
};