#include "VQVDB_Reader.hpp"

VDBStreamWriter::VDBStreamWriter(const std::string& outPath, const VQVDBMetadata& metadata)
    : filePath_(outPath),
      blockDataSize_(std::accumulate(metadata.latentShape.begin(), metadata.latentShape.end(), 1, std::multiplies<int64_t>())),
      chunkSize_(sizeof(openvdb::Coord) + blockDataSize_),
      buffer_(IO_BUFFER_SIZE) {
	fileStream_.open(outPath, std::ios::binary);
	if (!fileStream_) {
		throw std::runtime_error("Cannot open output file: " + outPath);
	}

	// --- Write Header ---
	VQVDBHeader header;
	header.numEmbeddings = metadata.numEmbeddings;
	header.latentDimCount = static_cast<uint8_t>(metadata.latentShape.size());
	header.headerExtensionSize = sizeof(VQVDBHeaderExtension);
	fileStream_.write(reinterpret_cast<const char*>(&header), sizeof(header));


	// Write header extension
	VQVDBHeaderExtension extension;
	std::memcpy(extension.voxelSize, metadata.voxelSize.asPointer(), 3 * sizeof(float));
	std::memcpy(extension.transform, metadata.transform.asPointer(), 16 * sizeof(double));
	fileStream_.write(reinterpret_cast<const char*>(&extension), sizeof(extension));

	// Write remaining variable-sized parts of the header
	std::vector<uint16_t> shape_u16(metadata.latentShape.begin(), metadata.latentShape.end());
	fileStream_.write(reinterpret_cast<const char*>(shape_u16.data()), shape_u16.size() * sizeof(uint16_t));
	uint32_t totalBlockCount = static_cast<uint32_t>(metadata.totalBlocks);
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


	metadata_.fileVersion = header.version;
	metadata_.numEmbeddings = header.numEmbeddings;

	// Read header extension
	if (header.headerExtensionSize > 0) {
		VQVDBHeaderExtension extension;
		// Ensure we don't read more than the struct we have defined
		fileStream_.read(reinterpret_cast<char*>(&extension), std::min((uint32_t)sizeof(extension), header.headerExtensionSize));
		metadata_.voxelSize = openvdb::Vec3d(extension.voxelSize);
		metadata_.transform = openvdb::math::Mat4d(extension.transform);
	}

	std::vector<uint16_t> latentShape_u16(header.latentDimCount);
	fileStream_.read(reinterpret_cast<char*>(latentShape_u16.data()), header.latentDimCount * sizeof(uint16_t));
	metadata_.latentShape.assign(latentShape_u16.begin(), latentShape_u16.end());

	uint32_t totalBlockCount;
	fileStream_.read(reinterpret_cast<char*>(&totalBlockCount), sizeof(totalBlockCount));
	metadata_.totalBlocks = totalBlockCount;

	// --- Prepare for reading data blocks ---
	blockDataSize_ = 1;
	for (uint16_t dim : metadata_.latentShape) {
		blockDataSize_ *= dim;
	}
	chunkSize_ = sizeof(openvdb::Coord) + blockDataSize_;
}

EncodedBatch VDBStreamReader::nextBatch(size_t maxBatch) {
	if (!hasNext()) return {torch::empty({0}), {}};

	const size_t blocksToRead = std::min(maxBatch, metadata_.totalBlocks - blocksRead_);

	std::vector<int64_t> tensorShape = {static_cast<int64_t>(blocksToRead)};
	tensorShape.insert(tensorShape.end(), metadata_.latentShape.begin(), metadata_.latentShape.end());
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
