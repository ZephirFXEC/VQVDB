#include "VQVDB_Reader.hpp"

VDBStreamWriter::VDBStreamWriter(std::string_view outPath, const VQVDBMetadata& metadata)
    : blockDataSize_(std::accumulate(metadata.latentShape.begin(), metadata.latentShape.end(), 1, std::multiplies<int64_t>())),
      chunkSize_(sizeof(openvdb::Coord) + blockDataSize_),
      buffer_(IO_BUFFER_SIZE) {
	fileStream_.open(outPath.data(), std::ios::binary | std::ios::out);
	if (!fileStream_) {
		throw std::runtime_error("Cannot open output file: " + std::string(outPath));
	}

	// Write Header
	VQVDBHeader header;
	header.numEmbeddings = metadata.numEmbeddings;
	header.latentDimCount = static_cast<uint8_t>(metadata.latentShape.size());
	header.headerExtensionSize = sizeof(VQVDBHeaderExtension);

	// Write header extension
	VQVDBHeaderExtension extension{};
	std::memcpy(extension.voxelSize, metadata.voxelSize.asPointer(), 3 * sizeof(float));
	std::memcpy(extension.transform, metadata.transform.asPointer(), 16 * sizeof(double));

	// Write remaining variable-sized parts of the header
	const std::vector<uint16_t> shape_u16(metadata.latentShape.begin(), metadata.latentShape.end());
	const uint32_t totalBlockCount = static_cast<uint32_t>(metadata.totalBlocks);

	fileStream_.write(reinterpret_cast<const char*>(&header), sizeof(header));
	fileStream_.write(reinterpret_cast<const char*>(&extension), sizeof(extension));
	fileStream_.write(reinterpret_cast<const char*>(shape_u16.data()), shape_u16.size() * sizeof(uint16_t));
	fileStream_.write(reinterpret_cast<const char*>(&totalBlockCount), sizeof(totalBlockCount));
}

VDBStreamWriter::~VDBStreamWriter() noexcept {
	try {
		flush();
	} catch (const std::exception& e) {
		std::cerr << "Error during VDBStreamWriter destruction: " << e.what() << std::endl;
	}
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

void VDBStreamWriter::flush() {
	if (bufferOffset_ > 0) {
		fileStream_.write(buffer_.data(), bufferOffset_);
		if (!fileStream_) {
			throw std::runtime_error("Failed to write buffer to file.");
		}
		bufferOffset_ = 0;
	}
}


VDBStreamReader::VDBStreamReader(const std::string_view inPath) : buffer_(IO_BUFFER_SIZE) {
	fileStream_.open(inPath.data(), std::ios::binary | std::ios::in);
	if (!fileStream_) {
		throw std::runtime_error("Cannot open input file: " + std::string(inPath));
	}

	// --- Read Header ---
	VQVDBHeader header;
	fileStream_.read(reinterpret_cast<char*>(&header), sizeof(header));
	if (!fileStream_) throw std::runtime_error("Failed to read VQVDB header.");
	if (std::string(header.magic, 5) != "VQVDB") throw std::runtime_error("Invalid VQVDB magic number.");
	if (header.version != 2) throw std::runtime_error("Unsupported VQVDB version. Expected 2, got " + std::to_string(header.version));


	metadata_.fileVersion = header.version;
	metadata_.numEmbeddings = header.numEmbeddings;

	// Read header extension
	if (header.headerExtensionSize > 0) {
		VQVDBHeaderExtension extension{};
		// Ensure we don't read more than the struct we have defined
		const auto bytesToRead = std::min(static_cast<uint32_t>(sizeof(extension)), header.headerExtensionSize);
		fileStream_.read(reinterpret_cast<char*>(&extension), bytesToRead);

		// Seek past any future unknown extension data
		fileStream_.seekg(header.headerExtensionSize - bytesToRead, std::ios_base::cur);

		metadata_.voxelSize = openvdb::Vec3f(extension.voxelSize);
		metadata_.transform = openvdb::math::Mat4d(extension.transform);
	}

	std::vector<uint16_t> latentShape_u16(header.latentDimCount);
	fileStream_.read(reinterpret_cast<char*>(latentShape_u16.data()), header.latentDimCount * sizeof(uint16_t));
	metadata_.latentShape.assign(latentShape_u16.begin(), latentShape_u16.end());

	uint32_t totalBlockCount;
	fileStream_.read(reinterpret_cast<char*>(&totalBlockCount), sizeof(totalBlockCount));
	metadata_.totalBlocks = totalBlockCount;
	if (!fileStream_) throw std::runtime_error("File appears truncated, failed to read full header.");

	// --- Prepare for reading data blocks ---
	blockDataSize_ = std::accumulate(metadata_.latentShape.begin(), metadata_.latentShape.end(), 1LL, std::multiplies<int64_t>());
	chunkSize_ = sizeof(openvdb::Coord) + blockDataSize_;

	// Initial buffer fill
	refillBuffer();
}

EncodedBatch VDBStreamReader::nextBatch(const size_t maxBatch) {
	if (!hasNext()) return {torch::empty({0}), {}};

	const size_t blocksToRequest = std::min(maxBatch, metadata_.totalBlocks - blocksRead_);

	std::vector<int64_t> tensorShape = {static_cast<int64_t>(blocksToRequest)};
	tensorShape.insert(tensorShape.end(), metadata_.latentShape.begin(), metadata_.latentShape.end());
	auto opts = torch::TensorOptions().dtype(torch::kU8).device(torch::kCPU).pinned_memory(true);
	torch::Tensor data = torch::empty(tensorShape, opts);

	std::vector<openvdb::Coord> originsBatch;
	originsBatch.reserve(blocksToRequest);
	uint8_t* dataPtr = data.data_ptr<uint8_t>();


	size_t blocksProcessed = 0;
	while (blocksProcessed < blocksToRequest) {
		const size_t remainingBytesInBuffer = bufferBytes_ - bufferOffset_;
		size_t blocksAvailableInBuffer = remainingBytesInBuffer / chunkSize_;

		if (blocksAvailableInBuffer == 0) {
			if (fileStream_.eof()) break;  // End of file, no more complete blocks
			refillBuffer();
			if (bufferBytes_ < chunkSize_ && fileStream_.eof()) break;  // Not enough data for even one chunk
			continue;
		}

		const size_t blocksToProcessNow = std::min(blocksToRequest - blocksProcessed, blocksAvailableInBuffer);

		const char* currentSrc = buffer_.data() + bufferOffset_;
		for (size_t i = 0; i < blocksToProcessNow; ++i) {
			// De-interleave origin and data
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
	if (blocksProcessed < blocksToRequest) {
		data = data.slice(0, 0, blocksProcessed);
		originsBatch.resize(blocksProcessed);  // Should already be correct but good practice
	}

	blocksRead_ += blocksProcessed;
	return {data, originsBatch};
}


void VDBStreamReader::refillBuffer() {
	// Move any remaining partial data to the start of the buffer
	const size_t remainingBytes = bufferBytes_ - bufferOffset_;
	if (remainingBytes > 0) {
		std::memmove(buffer_.data(), buffer_.data() + bufferOffset_, remainingBytes);
	}

	// Fill the rest with the buffer from the file
	fileStream_.read(buffer_.data() + remainingBytes, buffer_.size() - remainingBytes);
	bufferBytes_ = remainingBytes + fileStream_.gcount();
	bufferOffset_ = 0;
}
