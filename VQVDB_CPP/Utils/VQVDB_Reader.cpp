//
// Created by zphrfx on 17/06/2025.
//

#include "VQVDB_Reader.hpp"

void CompressedIndexReader::readHeader() {
	file_.read(header.magic, 5);
	if (std::string(header.magic, 5) != "VQVDB") {
		throw std::runtime_error("Invalid file format: incorrect magic number");
	}

	file_.read(reinterpret_cast<char*>(&header.version), 1);
	if (header.version != 2) {  // The python script specifically writes version 2
		throw std::runtime_error("Unsupported file version. Expected version 2.");
	}
	std::cout << "Reading VQVDB file (version " << static_cast<int>(header.version) << ")" << std::endl;

	file_.read(reinterpret_cast<char*>(&header.numEmbeddings), sizeof(uint32_t));
	file_.read(reinterpret_cast<char*>(&header.numDimensions), 1);
	header.shape.resize(header.numDimensions);
	file_.read(reinterpret_cast<char*>(header.shape.data()), header.numDimensions * sizeof(uint16_t));

	// Read Origins Block
	uint32_t num_origins;
	file_.read(reinterpret_cast<char*>(&num_origins), sizeof(uint32_t));
	header.origins.resize(num_origins);
	// BUG FIX: The C++ side expects int32_t for Coords. The Python script MUST write int32, not float32.
	// This read will produce garbage if the writing script is not also fixed.
	file_.read(reinterpret_cast<char*>(header.origins.data()), num_origins * sizeof(openvdb::Coord));

	// Read Indices Block Header
	file_.read(reinterpret_cast<char*>(&header.leafCount), sizeof(uint32_t));

	// The current position of the file stream is now at the beginning of the continuous index data.
}

// This function now correctly reads from a continuous block of data.
std::optional<torch::Tensor> CompressedIndexReader::readNextBatch(size_t batchSize) {
	if (!file_.good() || file_.eof()) {
		return std::nullopt;
	}

	size_t elementsPerBlock = 1;
	for (uint16_t dim : header.shape) {
		elementsPerBlock *= dim;
	}
	if (elementsPerBlock == 0) return std::nullopt;  // Avoid division by zero

	size_t elementsToRead = batchSize * elementsPerBlock;
	std::vector<uint16_t> indices_data(elementsToRead);

	file_.read(reinterpret_cast<char*>(indices_data.data()), elementsToRead * sizeof(uint16_t));

	const size_t bytesRead = file_.gcount();
	if (bytesRead == 0) {
		return std::nullopt;
	}

	const size_t elementsRead = bytesRead / sizeof(uint16_t);
	const size_t actualBatchSize = elementsRead / elementsPerBlock;
	if (actualBatchSize == 0) return std::nullopt;

	indices_data.resize(elementsRead);

	std::vector<int64_t> tensor_shape;
	tensor_shape.push_back(actualBatchSize);
	for (uint16_t dim : header.shape) {
		tensor_shape.push_back(dim);
	}

	return torch::from_blob(indices_data.data(), tensor_shape, torch::TensorOptions().dtype(torch::kInt16)).clone();
}