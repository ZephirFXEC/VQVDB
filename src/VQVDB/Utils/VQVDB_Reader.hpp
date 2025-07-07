#pragma once

#include <openvdb/Types.h>
#include <torch/torch.h>

#include <fstream>
#include <string>


struct VQVDBMetadata {
	uint8_t fileVersion = 0;
	uint32_t numEmbeddings = 0;
	std::vector<int64_t> latentShape;
	size_t totalBlocks = 0;

	openvdb::Vec3f voxelSize{1.0, 1.0, 1.0};
	openvdb::math::Mat4f transform;

	VQVDBMetadata() { transform.identity(); }
};

#pragma pack(push, 1)
struct VQVDBHeader {
	char magic[5] = {'V', 'Q', 'V', 'D', 'B'};
	uint8_t version = 2;
	uint32_t numEmbeddings = 0;
	uint8_t latentDimCount = 0;
	uint32_t headerExtensionSize = 0;
};

struct VQVDBHeaderExtension {
	float voxelSize[3];
	double transform[16];
};
#pragma pack(pop)

struct EncodedBatch {
	torch::Tensor data;
	std::vector<openvdb::Coord> origins;
};


class VDBStreamWriter {
   public:
	VDBStreamWriter(std::string_view outPath, const VQVDBMetadata& metadata);

	~VDBStreamWriter() noexcept;

	// Writes a batch of encoded data and their origins to the internal buffer,
	// flushing to disk when the buffer is full.
	void writeBatch(const torch::Tensor& encodedIndices, const std::vector<openvdb::Coord>& origins);

   private:
	void flush();


	std::ofstream fileStream_;
	const size_t blockDataSize_;
	const size_t chunkSize_;  // sizeof(Coord) + blockDataSize_

	// Use a large buffer (e.g., 4MB) for optimized disk writes
	static constexpr size_t IO_BUFFER_SIZE = 4 * 1024 * 1024;
	std::vector<char> buffer_;
	size_t bufferOffset_ = 0;
};


class VDBStreamReader {
   public:
	explicit VDBStreamReader(std::string_view inPath);
	~VDBStreamReader() noexcept = default;

	[[nodiscard]] bool hasNext() const noexcept { return blocksRead_ < metadata_.totalBlocks; }
	EncodedBatch nextBatch(size_t maxBatch);

	// Accessors for metadata
	[[nodiscard]] size_t totalBlocks() const noexcept { return metadata_.totalBlocks; }
	[[nodiscard]] const VQVDBMetadata& getMetadata() const noexcept { return metadata_; }


   private:
	void refillBuffer();

	VQVDBMetadata metadata_;
	std::ifstream fileStream_;
	size_t blocksRead_ = 0;

	size_t blockDataSize_ = 0;
	size_t chunkSize_ = 0;  // sizeof(Coord) + blockDataSize_

	// I/O Buffer
	static constexpr size_t IO_BUFFER_SIZE = 4 * 1024 * 1024;
	std::vector<char> buffer_;
	size_t bufferOffset_ = 0;
	size_t bufferBytes_ = 0;  // Valid bytes in buffer
};