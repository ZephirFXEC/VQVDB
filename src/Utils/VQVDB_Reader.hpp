#pragma once

#include <openvdb/Types.h>
#include "core/IVQVAECodec.hpp" // Include to get Tensor & TensorView

#include <fstream>
#include <string>


struct VQVDBMetadata {
	std::string name;
	uint8_t fileVersion = 3;
	uint32_t numEmbeddings = 0;
	std::vector<int64_t> latentShape;
	size_t totalBlocks = 0;
	openvdb::math::Mat4s transform;

	VQVDBMetadata() { transform.identity(); }
};

#pragma pack(push, 1)

struct VQVDBHeaderExtension {
	float transform[16];
};

struct VQVDBFileHeader {
	char magic[5] = {'V', 'Q', 'V', 'D', 'B'};
	uint8_t version = 3;
	uint8_t numGrids = 0;
	uint32_t numEmbeddings = 0;
	uint8_t latentDimCount = 0;
};
#pragma pack(pop)

struct EncodedBatch {
	Tensor data; // Changed from torch::Tensor to our generic Tensor
	std::vector<openvdb::Coord> origins;
};


class VDBStreamWriter {
   public:
	explicit VDBStreamWriter(std::string_view outPath);
	~VDBStreamWriter() noexcept;

	// Non-copyable and non-movable
	VDBStreamWriter(const VDBStreamWriter&) = delete;
	VDBStreamWriter& operator=(const VDBStreamWriter&) = delete;
	VDBStreamWriter(VDBStreamWriter&&) = delete;
	VDBStreamWriter& operator=(VDBStreamWriter&&) = delete;

	void startGrid(const VQVDBMetadata& metadata);
	void writeBatch(const Tensor& encodedTensor, const std::vector<openvdb::Coord>& origins);
	void endGrid();
	void close();

   private:
	void flush();
	void finalizeHeader();

	std::ofstream fileStream_;
	size_t blockDataSize_ = 0;
	size_t chunkSize_ = 0;

	// State for deferred header writing
	uint8_t numGrids_ = 0;
	bool headerFinalized_ = false;
	uint32_t sharedNumEmbeddings_ = 0;
	uint8_t sharedLatentDimCount_ = 0;

	static constexpr size_t IO_BUFFER_SIZE = 4 * 1024 * 1024;
	std::vector<char> buffer_;
	size_t bufferOffset_ = 0;
};


class VDBStreamReader {
   public:
	explicit VDBStreamReader(std::string_view inPath);
	~VDBStreamReader() noexcept = default;

	// Non-copyable and non-movable
	VDBStreamReader(const VDBStreamReader&) = delete;
	VDBStreamReader& operator=(const VDBStreamReader&) = delete;
	VDBStreamReader(VDBStreamReader&&) = delete;
	VDBStreamReader& operator=(VDBStreamReader&&) = delete;

	[[nodiscard]] bool hasNextGrid() const noexcept { return currentGrid_ < numGrids_; }
	VQVDBMetadata nextGridMetadata();
	[[nodiscard]] bool hasNext() const noexcept { return blocksRead_ < currentMetadata_.totalBlocks; }
	EncodedBatch nextBatch(size_t maxBatch);

   private:
	void refillBuffer();

	std::ifstream fileStream_;
	uint32_t numGrids_ = 0;
	uint32_t currentGrid_ = 0;
	uint32_t sharedNumEmbeddings_ = 0;
	uint8_t sharedLatentDimCount_ = 0;
	VQVDBMetadata currentMetadata_;
	size_t blocksRead_ = 0;
	size_t blockDataSize_ = 0;
	size_t chunkSize_ = 0;
	size_t remainingDataBytes_ = 0;

	static constexpr size_t IO_BUFFER_SIZE = 64 * 1024 * 1024;
	std::vector<char> buffer_;
	size_t bufferOffset_ = 0;
	size_t bufferBytes_ = 0;
};