#include "VQVDB_Reader.hpp"

VDBStreamWriter::VDBStreamWriter(const std::string& outPath, uint32_t numEmbeddings, const std::vector<int64_t>& latentShape,
                                 size_t totalBlocks)
    : filePath_(outPath),
      blockDataSize_(std::accumulate(latentShape.begin(), latentShape.end(), 1, std::multiplies<int64_t>())),
      chunkSize_(sizeof(openvdb::Coord) + blockDataSize_),
      buffer_(IO_BUFFER_SIZE) {
	fileStream_.open(outPath, std::ios::binary);
	if (!fileStream_) {
		throw std::runtime_error("Cannot open output file: " + outPath);
	}

	// --- Write Header ---
	VQVDBHeader header;
	header.numEmbeddings = numEmbeddings;
	header.latentDimCount = static_cast<uint8_t>(latentShape.size());

	fileStream_.write(reinterpret_cast<const char*>(&header), sizeof(header));

	// Write variable-sized parts of the header
	std::vector<uint16_t> shape_u16(latentShape.begin(), latentShape.end());
	fileStream_.write(reinterpret_cast<const char*>(shape_u16.data()), shape_u16.size() * sizeof(uint16_t));
	uint32_t totalBlockCount = static_cast<uint32_t>(totalBlocks);
	fileStream_.write(reinterpret_cast<const char*>(&totalBlockCount), sizeof(totalBlockCount));
}

void VDBStreamWriter::writeBatch(const torch::Tensor& encodedIndices, const std::vector<openvdb::Coord>& origins) {
	const size_t numBlocksInBatch = origins.size();
	const uint8_t* dataPtr = encodedIndices.data_ptr<uint8_t>();

	for (size_t i = 0; i < numBlocksInBatch; ++i) {
		if (bufferOffset_ + chunkSize_ > buffer_.size()) {
			flush();
		}
		// Interleave origin and data into the buffer
		char* currentDst = buffer_.data() + bufferOffset_;
		std::memcpy(currentDst, &origins[i], sizeof(openvdb::Coord));
		std::memcpy(currentDst + sizeof(openvdb::Coord), dataPtr + i * blockDataSize_, blockDataSize_);
		bufferOffset_ += chunkSize_;
	}
}


VDBStreamReader::VDBStreamReader(const std::string& inPath) : buffer_(IO_BUFFER_SIZE) {
	fileStream_.open(inPath, std::ios::binary);
	if (!fileStream_) {
		throw std::runtime_error("Cannot open input file: " + inPath);
	}

	// --- Read Header ---
	VQVDBHeader header;
	fileStream_.read(reinterpret_cast<char*>(&header), sizeof(header));
	if (std::string(header.magic, 5) != "VQVDB") throw std::runtime_error("Invalid VQVDB magic number.");
	if (header.version != 2) throw std::runtime_error("Unsupported VQVDB version. Expected 2, got " + std::to_string(header.version));

	numEmbeddings_ = header.numEmbeddings;

	latentShape_.resize(header.latentDimCount);
	fileStream_.read(reinterpret_cast<char*>(latentShape_.data()), header.latentDimCount * sizeof(uint16_t));

	uint32_t totalBlockCount;
	fileStream_.read(reinterpret_cast<char*>(&totalBlockCount), sizeof(totalBlockCount));
	totalBlocks_ = totalBlockCount;

	// --- Prepare for reading data blocks ---
	blockDataSize_ = 1;
	for (uint16_t dim : latentShape_) {
		blockDataSize_ *= dim;
		tensorShapeSuffix_.push_back(dim);
	}
	chunkSize_ = sizeof(openvdb::Coord) + blockDataSize_;
}

EncodedBatch VDBStreamReader::nextBatch(size_t maxBatch) {
	if (!hasNext()) return {torch::empty({0}), {}};

	const size_t blocksToRead = std::min(maxBatch, totalBlocks_ - blocksRead_);

	std::vector<int64_t> tensorShape = {static_cast<int64_t>(blocksToRead)};
	tensorShape.insert(tensorShape.end(), tensorShapeSuffix_.begin(), tensorShapeSuffix_.end());
	auto opts = torch::TensorOptions().dtype(torch::kU8).device(torch::kCPU).pinned_memory(true);
	torch::Tensor data = torch::empty(tensorShape, opts);

	std::vector<openvdb::Coord> originsBatch;
	originsBatch.reserve(blocksToRead);
	uint8_t* dataPtr = data.data_ptr<uint8_t>();

	size_t blocksProcessed = 0;
	while (blocksProcessed < blocksToRead) {
		if (bufferOffset_ >= bufferBytes_) {
			refillBuffer();
			if (bufferBytes_ == 0) break;  // EOF
		}

		size_t remainingBytesInBuffer = bufferBytes_ - bufferOffset_;
		size_t blocksAvailableInBuffer = remainingBytesInBuffer / chunkSize_;
		size_t blocksToProcessNow = std::min(blocksToRead - blocksProcessed, blocksAvailableInBuffer);

		if (blocksToProcessNow == 0) {  // Not enough data for a full chunk, refill
			refillBuffer();
			if (bufferBytes_ < chunkSize_) break;  // EOF and not enough data for one chunk
			continue;
		}

		// OPTIMIZATION: Process all available blocks in the buffer in a tight loop.
		char* currentSrc = buffer_.data() + bufferOffset_;
		for (size_t i = 0; i < blocksToProcessNow; ++i) {
			originsBatch.emplace_back();
			std::memcpy(&originsBatch.back(), currentSrc, sizeof(openvdb::Coord));
			std::memcpy(dataPtr, currentSrc + sizeof(openvdb::Coord), blockDataSize_);
			currentSrc += chunkSize_;
			dataPtr += blockDataSize_;
		}

		bufferOffset_ += blocksToProcessNow * chunkSize_;
		blocksProcessed += blocksToProcessNow;
	}

	// Handle truncated files where we read fewer blocks than requested
	if (blocksProcessed < blocksToRead) {
		data = data.slice(0, 0, blocksProcessed);
		originsBatch.resize(blocksProcessed);
	}

	blocksRead_ += blocksProcessed;
	return {data, originsBatch};
}
