//
// Created by zphrfx on 17/06/2025.
//

#include "VQVDB_Reader.hpp"

void CompressedIndexReader::readHeader() {
	// Magic
	file_.read(header.magic, 5);
	if (std::string(header.magic, 5) != "VQVDB") {
		throw std::runtime_error("Invalid file format: bad magic");
	}

	// Version
	file_.read(reinterpret_cast<char*>(&header.version), 1);
	if (header.version != 2) {
		throw std::runtime_error("Unsupported file version: expected 2, got " + std::to_string(int(header.version)));
	}
	std::cout << "Reading VQVDB file (version " << int(header.version) << ")\n";

	// numEmbeddings
	file_.read(reinterpret_cast<char*>(&header.numEmbeddings), sizeof(header.numEmbeddings));

	// numDimensions + shape[]
	file_.read(reinterpret_cast<char*>(&header.numDimensions), 1);
	header.shape.resize(header.numDimensions);
	file_.read(reinterpret_cast<char*>(header.shape.data()), header.numDimensions * sizeof(uint16_t));

	// Origins block
	uint32_t nOrig;
	file_.read(reinterpret_cast<char*>(&nOrig), sizeof(nOrig));
	header.origins.resize(nOrig);
	file_.read(reinterpret_cast<char*>(header.origins.data()), nOrig * sizeof(openvdb::Coord));

	// Leaf‐count
	file_.read(reinterpret_cast<char*>(&header.leafCount), sizeof(header.leafCount));
}

// This function now correctly reads from a continuous block of data.
std::optional<torch::Tensor> CompressedIndexReader::readNextBatch(size_t batchSize) {
	if (!file_.good() || file_.eof()) {
		return std::nullopt;
	}

	// Allocate a pinned‐CPU tensor [batchSize, elementsPerBlock_]
	auto options = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU).pinned_memory(true);
	torch::Tensor tensor = torch::empty({int64_t(batchSize), int64_t(elementsPerBlock_)}, options);

	// Read raw bytes directly into that tensor
	file_.read(reinterpret_cast<char*>(tensor.data_ptr<uint8_t>()), std::streamsize(batchSize * elementsPerBlock_));
	size_t bytesRead = file_.gcount();
	if (bytesRead == 0) {
		return std::nullopt;
	}

	// Compute how many *full* blocks we actually got
	size_t elemsRead = bytesRead;  // since one uint8_t = one byte
	size_t blocksRead = elemsRead / elementsPerBlock_;
	if (blocksRead == 0) {
		return std::nullopt;  // dropped a partial block?
	}

	// Narrow to [blocksRead, elementsPerBlock_]
	tensor = tensor.narrow(0, 0, int64_t(blocksRead));

	// View to [blocksRead, D, H, W, …]
	std::vector<int64_t> fullShape;
	fullShape.reserve(1 + tensorShapeSuffix_.size());
	fullShape.push_back(int64_t(blocksRead));
	fullShape.insert(fullShape.end(), tensorShapeSuffix_.begin(), tensorShapeSuffix_.end());
	return tensor.view(fullShape);
}