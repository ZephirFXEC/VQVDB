#pragma once

#include <openvdb/Types.h>
#include <torch/torch.h>

#include <fstream>
#include <string>

#pragma pack(push, 1)
struct VQVDBHeader {
	char magic[5] = {'V', 'Q', 'V', 'D', 'B'};
	uint8_t version = 2;
	uint32_t numEmbeddings = 0;
	uint8_t latentDimCount = 0;
	// Followed immediately by latentShape[latentDimCount]
	// Followed immediately by totalBlockCount
};
#pragma pack(pop)

// A struct to hold a batch of data read from the file.
struct EncodedBatch {
	torch::Tensor data;
	std::vector<openvdb::Coord> origins;
};


class VDBStreamWriter {
   public:
	VDBStreamWriter(const std::string& outPath, uint32_t numEmbeddings, const std::vector<int64_t>& latentShape, size_t totalBlocks);

	~VDBStreamWriter() {
		flush();
		fileStream_.close();
	}

	// Writes a batch of encoded data and their origins to the internal buffer,
	// flushing to disk when the buffer is full.
	void writeBatch(const torch::Tensor& encodedIndices, const std::vector<openvdb::Coord>& origins);

   private:
	void flush() {
		if (bufferOffset_ > 0) {
			fileStream_.write(buffer_.data(), bufferOffset_);
			bufferOffset_ = 0;
		}
	}

	std::ofstream fileStream_;
	std::string filePath_;
	const size_t blockDataSize_;
	const size_t chunkSize_; // sizeof(Coord) + blockDataSize_

	// Use a large buffer (e.g., 4MB) for optimized disk writes
	static constexpr size_t IO_BUFFER_SIZE = 4 * 1024 * 1024;
	std::vector<char> buffer_;
	size_t bufferOffset_ = 0;
};


class VDBStreamReader {
   public:
	explicit VDBStreamReader(const std::string& inPath);

	[[nodiscard]] bool hasNext() const noexcept { return blocksRead_ < totalBlocks_; }

	EncodedBatch nextBatch(size_t maxBatch);

	// Accessors for metadata
	[[nodiscard]] size_t totalBlocks() const { return totalBlocks_; }
	[[nodiscard]] const std::vector<int64_t>& latentShape() const { return tensorShapeSuffix_; }


   private:
	void refillBuffer() {
		// Move any remaining partial data to the start of the buffer
		const size_t remainingBytes = bufferBytes_ - bufferOffset_;
		if (remainingBytes > 0) {
			std::memmove(buffer_.data(), buffer_.data() + bufferOffset_, remainingBytes);
		}
		bufferOffset_ = 0;

		// Fill the rest of the buffer from the file
		fileStream_.read(buffer_.data() + remainingBytes, buffer_.size() - remainingBytes);
		bufferBytes_ = remainingBytes + fileStream_.gcount();
	}

	std::ifstream fileStream_;
	size_t totalBlocks_ = 0;
	size_t blocksRead_ = 0;
	uint32_t numEmbeddings_ = 0;
	std::vector<uint16_t> latentShape_;       // As read from file
	std::vector<int64_t> tensorShapeSuffix_;  // As int64_t for torch

	size_t blockDataSize_ = 0;
	size_t chunkSize_ = 0;  // sizeof(Coord) + blockDataSize_

	// I/O Buffer
	static constexpr size_t IO_BUFFER_SIZE = 4 * 1024 * 1024;
	std::vector<char> buffer_;
	size_t bufferOffset_ = 0;
	size_t bufferBytes_ = 0;  // Valid bytes in buffer
};