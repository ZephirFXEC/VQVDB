/*
 * Copyright (c) 2025, Enzo Crema
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "vqvdb/codebook_loader.hpp"

#include <fstream>

namespace vqvdb {

CodebookResult<Codebook> loadCodebookFile(const std::filesystem::path& path) {
	// Check file exists
	if (!std::filesystem::exists(path)) {
		return std::unexpected(CodebookError::FileNotFound);
	}

	// Open file
	std::ifstream file(path, std::ios::binary);
	if (!file) {
		return std::unexpected(CodebookError::FileOpenFailed);
	}

	// Read header
	CodebookFileHeader header{};
	file.read(reinterpret_cast<char*>(&header), sizeof(header));
	if (!file || file.gcount() != sizeof(header)) {
		return std::unexpected(CodebookError::HeaderReadFailed);
	}

	// Validate magic
	if (header.magic != kCodebookMagic) {
		return std::unexpected(CodebookError::InvalidMagic);
	}

	// Validate dimensions
	if (header.numEmbeddings == 0 || header.embeddingDim == 0) {
		return std::unexpected(CodebookError::InvalidDimensions);
	}

	// Sanity check - reasonable limits
	if (header.numEmbeddings > 65536 || header.embeddingDim > 4096) {
		return std::unexpected(CodebookError::InvalidDimensions);
	}

	// Calculate expected data size
	const size_t numFloats = static_cast<size_t>(header.numEmbeddings) * header.embeddingDim;
	const size_t dataBytes = numFloats * sizeof(float);

	// Check file has enough data
	const auto currentPos = file.tellg();
	file.seekg(0, std::ios::end);
	const auto fileSize = file.tellg();
	file.seekg(currentPos);

	if (static_cast<size_t>(fileSize - currentPos) < dataBytes) {
		return std::unexpected(CodebookError::FileTruncated);
	}

	// Allocate and read data
	Codebook result;
	result.numEmbeddings = header.numEmbeddings;
	result.embeddingDim = header.embeddingDim;
	result.data.resize(numFloats);

	file.read(reinterpret_cast<char*>(result.data.data()), static_cast<std::streamsize>(dataBytes));
	if (!file) {
		return std::unexpected(CodebookError::DataReadFailed);
	}

	return result;
}

bool isCodebookFile(const std::filesystem::path& path) {
	if (!std::filesystem::exists(path)) {
		return false;
	}

	std::ifstream file(path, std::ios::binary);
	if (!file) {
		return false;
	}

	uint32_t magic = 0;
	file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
	if (!file || file.gcount() != sizeof(magic)) {
		return false;
	}

	return magic == kCodebookMagic;
}

}  // namespace vqvdb
