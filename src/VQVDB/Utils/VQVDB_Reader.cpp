#include "VQVDB_Reader.hpp"

#include <cstring>
#include <iostream>
#include <numeric>
#include <stdexcept>

// ===================================================================================
// VDBStreamWriter Implementation
// ===================================================================================

VDBStreamWriter::VDBStreamWriter(const std::string_view outPath) : buffer_(IO_BUFFER_SIZE) {
	fileStream_.open(outPath.data(), std::ios::binary | std::ios::out);
	if (!fileStream_) {
		throw std::runtime_error("Cannot open output file: " + std::string(outPath));
	}

	// Write a placeholder header. It will be finalized later.
	VQVDBFileHeader placeholderHeader{};
	fileStream_.write(reinterpret_cast<const char*>(&placeholderHeader), sizeof(placeholderHeader));
	if (!fileStream_) {
		throw std::runtime_error("Failed to write placeholder file header.");
	}
}

VDBStreamWriter::~VDBStreamWriter() noexcept {
	try {
		// Ensure file is closed and header is finalized even if close() isn't called explicitly.
		if (fileStream_.is_open()) {
			close();
		}
	} catch (const std::exception& e) {
		std::cerr << "Error during VDBStreamWriter destruction: " << e.what() << std::endl;
	}
}

void VDBStreamWriter::close() {
	if (!fileStream_.is_open()) {
		return;
	}

	flush();
	finalizeHeader();
	fileStream_.close();
	if (fileStream_.fail()) {
		throw std::runtime_error("Error closing the output file.");
	}
}

void VDBStreamWriter::finalizeHeader() {
	if (numGrids_ == 0) {
		return;  // No grids were written, header remains empty.
	}

	// Seek to the beginning of the file to update the header with final grid count.
	fileStream_.seekp(0, std::ios::beg);
	if (!fileStream_) {
		throw std::runtime_error("Failed to seek to update header.");
	}

	VQVDBFileHeader header;
	header.numGrids = numGrids_;
	header.numEmbeddings = sharedNumEmbeddings_;
	header.latentDimCount = sharedLatentDimCount_;
	fileStream_.write(reinterpret_cast<const char*>(&header), sizeof(header));

	if (!fileStream_) {
		throw std::runtime_error("Failed to write final file header.");
	}
}


void VDBStreamWriter::startGrid(const VQVDBMetadata& metadata) {
	flush();  // Flush any previous grid data

	if (!headerFinalized_) {
		// First grid: use its metadata to set the shared file properties.
		sharedNumEmbeddings_ = metadata.numEmbeddings;
		sharedLatentDimCount_ = static_cast<uint8_t>(metadata.latentShape.size());

		// Temporarily seek to the start to write the shared properties.
		auto currentPos = fileStream_.tellp();
		finalizeHeader();               // Writes the shared properties, with numGrids=0 for now.
		fileStream_.seekp(currentPos);  // Return to the end of the file to continue writing.
		if (!fileStream_) {
			throw std::runtime_error("Failed to seek back after writing header.");
		}
		headerFinalized_ = true;
	} else {
		// For subsequent grids, ensure their properties are consistent.
		if (metadata.numEmbeddings != sharedNumEmbeddings_) {
			throw std::runtime_error("Inconsistent number of embeddings across grids.");
		}
		if (metadata.latentShape.size() != sharedLatentDimCount_) {
			throw std::runtime_error("Inconsistent latent dimension count across grids.");
		}
	}

	blockDataSize_ = std::accumulate(metadata.latentShape.begin(), metadata.latentShape.end(), 1LL, std::multiplies<>());
	chunkSize_ = sizeof(openvdb::Coord) + blockDataSize_;

	// Write grid-specific metadata
	// Write name
	uint32_t nameLength = static_cast<uint32_t>(metadata.name.size());
	fileStream_.write(reinterpret_cast<const char*>(&nameLength), sizeof(nameLength));
	fileStream_.write(metadata.name.data(), nameLength);

	// Write extension
	VQVDBHeaderExtension extension{};
	std::memcpy(extension.transform, metadata.transform.asPointer(), 16 * sizeof(float));
	fileStream_.write(reinterpret_cast<const char*>(&extension), sizeof(extension));

	// Write latent shape (vector of uint16_t)
	const std::vector<uint16_t> shape_u16(metadata.latentShape.begin(), metadata.latentShape.end());
	fileStream_.write(reinterpret_cast<const char*>(shape_u16.data()), shape_u16.size() * sizeof(uint16_t));

	// Write total blocks
	const uint32_t totalBlockCount = static_cast<uint32_t>(metadata.totalBlocks);
	fileStream_.write(reinterpret_cast<const char*>(&totalBlockCount), sizeof(totalBlockCount));

	if (!fileStream_) {
		throw std::runtime_error("Failed to write grid metadata.");
	}

	numGrids_++;  // A grid is successfully started.
	bufferOffset_ = 0;
}

void VDBStreamWriter::writeBatch(const torch::Tensor& encodedIndices, const std::vector<openvdb::Coord>& origins) {
	const size_t numBlocksInBatch = origins.size();
	const uint8_t* dataPtr = encodedIndices.data_ptr<uint8_t>();

	for (size_t i = 0; i < numBlocksInBatch; ++i) {
		if (bufferOffset_ + chunkSize_ > buffer_.size()) {
			flush();
		}
		char* currentDst = buffer_.data() + bufferOffset_;
		std::memcpy(currentDst, &origins[i], sizeof(openvdb::Coord));
		std::memcpy(currentDst + sizeof(openvdb::Coord), dataPtr + i * blockDataSize_, blockDataSize_);
		bufferOffset_ += chunkSize_;
	}
}

void VDBStreamWriter::endGrid() { flush(); }

void VDBStreamWriter::flush() {
	if (bufferOffset_ > 0) {
		fileStream_.write(buffer_.data(), bufferOffset_);
		if (!fileStream_) {
			throw std::runtime_error("Failed to write buffer to file.");
		}
		bufferOffset_ = 0;
	}
}

// ===================================================================================
// VDBStreamReader Implementation
// ===================================================================================

VDBStreamReader::VDBStreamReader(std::string_view inPath) : buffer_(IO_BUFFER_SIZE) {
	fileStream_.open(inPath.data(), std::ios::binary | std::ios::in);
	if (!fileStream_) {
		throw std::runtime_error("Cannot open input file: " + std::string(inPath));
	}

	VQVDBFileHeader header;
	fileStream_.read(reinterpret_cast<char*>(&header), sizeof(header));
	if (!fileStream_) throw std::runtime_error("Failed to read file header.");
	if (std::string(header.magic, 5) != "VQVDB") throw std::runtime_error("Invalid VQVDB magic number.");
	if (header.version != 3) throw std::runtime_error("Unsupported VQVDB version. Expected 3, got " + std::to_string(header.version));

	sharedNumEmbeddings_ = header.numEmbeddings;
	sharedLatentDimCount_ = header.latentDimCount;
	numGrids_ = header.numGrids;

	currentGrid_ = 0;
	bufferOffset_ = 0;
	bufferBytes_ = 0;
	remainingDataBytes_ = 0;
}

VQVDBMetadata VDBStreamReader::nextGridMetadata() {
	if (!hasNextGrid()) throw std::runtime_error("No more grids available.");

	VQVDBMetadata metadata;
	metadata.fileVersion = 3;
	metadata.numEmbeddings = sharedNumEmbeddings_;

	// Read name
	uint32_t nameLength;
	fileStream_.read(reinterpret_cast<char*>(&nameLength), sizeof(nameLength));
	if (!fileStream_) throw std::runtime_error("Failed to read grid name length.");
	metadata.name.resize(nameLength);
	fileStream_.read(metadata.name.data(), nameLength);
	if (!fileStream_) throw std::runtime_error("Failed to read grid name.");

	// Read extension
	VQVDBHeaderExtension extension{};
	fileStream_.read(reinterpret_cast<char*>(&extension), sizeof(extension));
	if (!fileStream_) throw std::runtime_error("Failed to read header extension.");

	metadata.transform = openvdb::math::Mat4s(extension.transform);

	// Read latent shape
	if (sharedLatentDimCount_ > 0) {
		std::vector<uint16_t> latentShape_u16(sharedLatentDimCount_);
		fileStream_.read(reinterpret_cast<char*>(latentShape_u16.data()), sharedLatentDimCount_ * sizeof(uint16_t));
		if (!fileStream_) throw std::runtime_error("Failed to read latent shape.");
		metadata.latentShape.assign(latentShape_u16.begin(), latentShape_u16.end());
	}

	// Read total blocks
	uint32_t totalBlockCount;
	fileStream_.read(reinterpret_cast<char*>(&totalBlockCount), sizeof(totalBlockCount));
	if (!fileStream_) throw std::runtime_error("File appears truncated, failed to read total block count.");
	metadata.totalBlocks = totalBlockCount;

	// Prepare for data reading
	blockDataSize_ = std::accumulate(metadata.latentShape.begin(), metadata.latentShape.end(), 1LL, std::multiplies<>());
	chunkSize_ = sizeof(openvdb::Coord) + blockDataSize_;
	blocksRead_ = 0;
	currentMetadata_ = metadata;
	remainingDataBytes_ = metadata.totalBlocks * chunkSize_;

	refillBuffer();

	currentGrid_++;
	return metadata;
}


EncodedBatch VDBStreamReader::nextBatch(const size_t maxBatch) {
	if (!hasNext()) return {torch::empty({0}), {}};

	const size_t blocksToRequest = std::min(maxBatch, currentMetadata_.totalBlocks - blocksRead_);

	std::vector<int64_t> tensorShape = {static_cast<int64_t>(blocksToRequest)};
	tensorShape.insert(tensorShape.end(), currentMetadata_.latentShape.begin(), currentMetadata_.latentShape.end());
	auto opts = torch::TensorOptions().dtype(torch::kU8).device(torch::kCPU).pinned_memory(true);
	torch::Tensor data = torch::empty(tensorShape, opts);

	std::vector<openvdb::Coord> originsBatch;
	originsBatch.reserve(blocksToRequest);
	auto* dataPtr = data.data_ptr<uint8_t>();

	size_t blocksProcessed = 0;
	while (blocksProcessed < blocksToRequest) {
		const size_t remainingBytesInBuffer = bufferBytes_ - bufferOffset_;
		size_t blocksAvailableInBuffer = (chunkSize_ > 0) ? (remainingBytesInBuffer / chunkSize_) : 0;

		if (blocksAvailableInBuffer == 0) {
			if (fileStream_.eof() || (bufferBytes_ < buffer_.size() && remainingBytesInBuffer < chunkSize_)) {
				break;
			}
			refillBuffer();
			continue;
		}

		const size_t blocksToProcessNow = std::min(blocksToRequest - blocksProcessed, blocksAvailableInBuffer);

		const char* currentSrc = buffer_.data() + bufferOffset_;
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

	remainingDataBytes_ -= blocksProcessed * chunkSize_;

	if (blocksProcessed < blocksToRequest) {
		if (remainingDataBytes_ > 0) {
			throw std::runtime_error("File truncated: Fewer blocks read than expected by metadata.");
		}
		tensorShape[0] = static_cast<int64_t>(blocksProcessed);  // Update first dim
		data = data.resize_(tensorShape);                        // Resize to new shape vector
	}

	blocksRead_ += blocksProcessed;
	return {data, originsBatch};
}


void VDBStreamReader::refillBuffer() {
	const size_t remainingBytes = bufferBytes_ - bufferOffset_;
	if (remainingBytes > 0) {
		std::memmove(buffer_.data(), buffer_.data() + bufferOffset_, remainingBytes);
	}

	const size_t maxPossibleRead = buffer_.size() - remainingBytes;
	const size_t bytesToRead = std::min(maxPossibleRead, remainingDataBytes_);

	if (bytesToRead == 0) {
		bufferBytes_ = remainingBytes;
		bufferOffset_ = 0;
		return;  // Nothing left to read for this grid
	}

	fileStream_.read(buffer_.data() + remainingBytes, bytesToRead);

	const size_t bytesRead = fileStream_.gcount();

	remainingDataBytes_ -= bytesRead;

	if (fileStream_.fail() && !fileStream_.eof()) {
		throw std::runtime_error("Failed to read from file.");
	}

	// NEW: Check for truncation (read less than requested, but not at EOF and still expecting more)
	if (bytesRead < bytesToRead && !fileStream_.eof() && remainingDataBytes_ > 0) {
		throw std::runtime_error("File truncated: Incomplete read during refill.");
	}

	bufferBytes_ = remainingBytes + bytesRead;
	bufferOffset_ = 0;
}