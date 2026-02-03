/*
 * Copyright (c) 2025, Enzo Crema
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Standalone VQVDB file loader implementation.
 * Matches the file format from VQVDB_Reader.hpp/cpp exactly.
 */

#include "vqvdb_loader.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <numeric>

namespace vqvdb {

// ============================================================================
// File Format Structures (must match VQVDB_Reader.hpp exactly)
// ============================================================================

#pragma pack(push, 1)

/// Main file header for v3 format (matches VQVDBFileHeader in VQVDB_Reader.hpp)
struct FileHeaderV3 {
	char magic[5];           // "VQVDB"
	uint8_t version;         // 3
	uint8_t numGrids;        // Number of grids in file
	uint32_t numEmbeddings;  // Codebook entries (typically 256)
	uint8_t latentDimCount;  // Number of latent dimensions (typically 3)
};

static_assert(sizeof(FileHeaderV3) == 12, "FileHeaderV3 size mismatch");

/// Per-grid header extension (transform matrix)
struct HeaderExtension {
	float transform[16];  // 4x4 matrix
};

static_assert(sizeof(HeaderExtension) == 64, "HeaderExtension size mismatch");

#pragma pack(pop)

// ============================================================================
// Error Handling
// ============================================================================

const char* errorToString(LoadError error) noexcept {
	switch (error) {
		case LoadError::FileNotFound:
			return "File not found";
		case LoadError::FileOpenFailed:
			return "Failed to open file";
		case LoadError::InvalidMagic:
			return "Invalid VQVDB magic number";
		case LoadError::UnsupportedVersion:
			return "Unsupported file version";
		case LoadError::HeaderReadFailed:
			return "Failed to read file header";
		case LoadError::GridMetadataReadFailed:
			return "Failed to read grid metadata";
		case LoadError::BlockDataReadFailed:
			return "Failed to read block data";
		case LoadError::FileTruncated:
			return "File appears to be truncated";
		case LoadError::AllocationFailed:
			return "Memory allocation failed";
		case LoadError::InvalidData:
			return "Invalid data in file";
	}
	return "Unknown error";
}

// ============================================================================
// Internal Loader Class (matches VDBStreamReader pattern)
// ============================================================================

class VQVDBLoader {
   public:
	explicit VQVDBLoader(const std::filesystem::path& path, const LoaderConfig& config) : config_(config), buffer_(config.ioBufferSize) {
		file_.open(path, std::ios::binary);
	}

	[[nodiscard]] bool isOpen() const { return file_.is_open(); }

	LoadResult<VQVDBFile> load() {
		// First, read 7 bytes to detect magic format
		char magicBuf[7] = {0};
		file_.read(magicBuf, 7);
		if (!file_) {
			return std::unexpected(LoadError::HeaderReadFailed);
		}

		VQVDBFile result;
		uint8_t numGrids = 0;

		// Check for v3 format ("VQVDB" + version byte)
		if (std::string_view(magicBuf, 5) == "VQVDB") {
			// v3 format - we already read 7 bytes, need to parse correctly
			// magicBuf[5] = version, magicBuf[6] = numGrids
			uint8_t version = static_cast<uint8_t>(magicBuf[5]);
			if (version != 3) {
				return std::unexpected(LoadError::UnsupportedVersion);
			}

			// Read remaining header (numEmbeddings + latentDimCount = 5 bytes)
			uint32_t numEmbeddings = 0;
			uint8_t latentDimCount = 0;
			file_.read(reinterpret_cast<char*>(&numEmbeddings), sizeof(numEmbeddings));
			file_.read(reinterpret_cast<char*>(&latentDimCount), sizeof(latentDimCount));
			if (!file_) {
				return std::unexpected(LoadError::HeaderReadFailed);
			}

			result.version = 3;
			numGrids = static_cast<uint8_t>(magicBuf[6]);
			result.codebook.numEmbeddings = static_cast<int32_t>(numEmbeddings);
			result.codebook.embeddingDim = kDefaultEmbeddingDim;
			latentDimCount_ = latentDimCount;
		} else {
			return std::unexpected(LoadError::InvalidMagic);
		}

		// Load each grid
		result.grids.reserve(numGrids);

		for (uint8_t gridIdx = 0; gridIdx < numGrids; ++gridIdx) {
			auto gridResult = loadGrid();
			if (!gridResult) {
				return std::unexpected(gridResult.error());
			}
			result.grids.push_back(std::move(*gridResult));
		}

		return result;
	}

   private:
	LoadResult<VQVDBGrid> loadGrid() {
		VQVDBGrid grid;

		// === Read Grid Metadata (direct read, matches original) ===

		// Read name length
		uint32_t nameLength = 0;
		file_.read(reinterpret_cast<char*>(&nameLength), sizeof(nameLength));
		if (!file_) {
			return std::unexpected(LoadError::GridMetadataReadFailed);
		}

		// Read name
		grid.metadata.name.resize(nameLength);
		if (nameLength > 0) {
			file_.read(grid.metadata.name.data(), nameLength);
			if (!file_) {
				return std::unexpected(LoadError::GridMetadataReadFailed);
			}
		}

		// Read transform extension
		HeaderExtension extension{};
		file_.read(reinterpret_cast<char*>(&extension), sizeof(extension));
		if (!file_) {
			return std::unexpected(LoadError::GridMetadataReadFailed);
		}

		// Copy transform data
		for (int i = 0; i < 16; ++i) {
			grid.metadata.transform.data[i] = extension.transform[i];
		}

		// Read latent shape
		if (latentDimCount_ > 0) {
			std::vector<uint16_t> latentShape(latentDimCount_);
			file_.read(reinterpret_cast<char*>(latentShape.data()), latentDimCount_ * sizeof(uint16_t));
			if (!file_) {
				return std::unexpected(LoadError::GridMetadataReadFailed);
			}

			for (size_t i = 0; i < 3 && i < latentDimCount_; ++i) {
				grid.metadata.latentShape[i] = static_cast<int32_t>(latentShape[i]);
			}
		}

		// Read total block count
		uint32_t totalBlocks = 0;
		file_.read(reinterpret_cast<char*>(&totalBlocks), sizeof(totalBlocks));
		if (!file_) {
			return std::unexpected(LoadError::GridMetadataReadFailed);
		}
		grid.metadata.totalBlocks = totalBlocks;

		// === Calculate sizes for block data ===
		blockDataSize_ = static_cast<size_t>(grid.metadata.latentShape[0]) * static_cast<size_t>(grid.metadata.latentShape[1]) *
		                 static_cast<size_t>(grid.metadata.latentShape[2]);
		chunkSize_ = sizeof(BlockOrigin) + blockDataSize_;
		remainingDataBytes_ = totalBlocks * chunkSize_;

		// === Load block data (buffered read, matches original) ===
		auto blocksResult = loadBlockData(grid.metadata);
		if (!blocksResult) {
			return std::unexpected(blocksResult.error());
		}
		grid.blocks = std::move(*blocksResult);

		// Compute world bounds
		grid.metadata.worldBounds = grid.blocks.computeWorldBounds(grid.metadata.transform);

		return grid;
	}

	LoadResult<BlockData> loadBlockData(const GridMetadata& metadata) {
		BlockData blocks;

		// Determine how many blocks to load
		size_t blocksToLoad = metadata.totalBlocks;
		if (config_.maxBlocksPerGrid > 0) {
			blocksToLoad = std::min(blocksToLoad, config_.maxBlocksPerGrid);
		}

		// Skip if metadata only
		if (config_.metadataOnly) {
			file_.seekg(static_cast<std::streamoff>(remainingDataBytes_), std::ios::cur);
			if (!file_) {
				return std::unexpected(LoadError::BlockDataReadFailed);
			}
			return blocks;
		}

		// Reserve memory
		try {
			blocks.reserve(blocksToLoad);
			blocks.indices.resize(blocksToLoad * blockDataSize_);
		} catch (...) {
			return std::unexpected(LoadError::AllocationFailed);
		}

		// Fill buffer initially
		refillBuffer();

		size_t blocksProcessed = 0;
		uint8_t* indicesPtr = blocks.indices.data();

		while (blocksProcessed < blocksToLoad) {
			// Check if we need to refill buffer
			const size_t availableBytes = bufferFilled_ - bufferOffset_;
			const size_t availableBlocks = (chunkSize_ > 0) ? (availableBytes / chunkSize_) : 0;

			if (availableBlocks == 0) {
				if (file_.eof() || (bufferFilled_ < buffer_.size() && availableBytes < chunkSize_)) {
					break;  // No more data
				}
				refillBuffer();
				continue;
			}

			const size_t blocksToProcess = std::min(blocksToLoad - blocksProcessed, availableBlocks);
			const char* src = buffer_.data() + bufferOffset_;

			for (size_t i = 0; i < blocksToProcess; ++i) {
				// Read origin
				BlockOrigin origin{};
				std::memcpy(&origin, src, sizeof(BlockOrigin));
				blocks.origins.push_back(origin);

				// Read indices
				std::memcpy(indicesPtr, src + sizeof(BlockOrigin), blockDataSize_);

				src += chunkSize_;
				indicesPtr += blockDataSize_;
			}

			bufferOffset_ += blocksToProcess * chunkSize_;
			blocksProcessed += blocksToProcess;
		}

		// Resize indices to actual size if we read fewer blocks
		if (blocksProcessed < blocksToLoad) {
			blocks.indices.resize(blocksProcessed * blockDataSize_);
		}

		// Skip remaining blocks if we hit the limit
		if (blocksProcessed < metadata.totalBlocks) {
			const size_t remainingBlocks = metadata.totalBlocks - blocksProcessed;
			const size_t bytesToSkip = remainingBlocks * chunkSize_;

			// Skip what's left in buffer
			const size_t availableInBuffer = bufferFilled_ - bufferOffset_;
			if (bytesToSkip <= availableInBuffer) {
				bufferOffset_ += bytesToSkip;
			} else {
				// Need to seek in file
				const size_t additionalSkip = bytesToSkip - availableInBuffer;
				file_.seekg(static_cast<std::streamoff>(additionalSkip), std::ios::cur);
				bufferOffset_ = 0;
				bufferFilled_ = 0;
			}
		}

		return blocks;
	}

	void refillBuffer() {
		// Move remaining bytes to start
		const size_t remaining = bufferFilled_ - bufferOffset_;
		if (remaining > 0) {
			std::memmove(buffer_.data(), buffer_.data() + bufferOffset_, remaining);
		}

		// Calculate how much to read
		const size_t spaceAvailable = buffer_.size() - remaining;
		const size_t toRead = std::min(spaceAvailable, remainingDataBytes_);

		if (toRead == 0) {
			bufferFilled_ = remaining;
			bufferOffset_ = 0;
			return;
		}

		file_.read(buffer_.data() + remaining, static_cast<std::streamsize>(toRead));
		const size_t bytesRead = static_cast<size_t>(file_.gcount());

		remainingDataBytes_ -= bytesRead;
		bufferFilled_ = remaining + bytesRead;
		bufferOffset_ = 0;
	}

	std::ifstream file_;
	LoaderConfig config_;
	std::vector<char> buffer_;

	uint8_t latentDimCount_ = 0;
	size_t blockDataSize_ = 0;
	size_t chunkSize_ = 0;
	size_t remainingDataBytes_ = 0;
	size_t bufferOffset_ = 0;
	size_t bufferFilled_ = 0;
};

// ============================================================================
// Public API Implementation
// ============================================================================

LoadResult<VQVDBFile> loadFile(const std::filesystem::path& path, const LoaderConfig& config) {
	if (!std::filesystem::exists(path)) {
		return std::unexpected(LoadError::FileNotFound);
	}

	VQVDBLoader loader(path, config);
	if (!loader.isOpen()) {
		return std::unexpected(LoadError::FileOpenFailed);
	}

	return loader.load();
}

LoadResult<VQVDBFile> loadMetadata(const std::filesystem::path& path) {
	LoaderConfig config;
	config.metadataOnly = true;
	return loadFile(path, config);
}

bool isVQVDBFile(const std::filesystem::path& path) {
	if (!std::filesystem::exists(path)) {
		return false;
	}

	std::ifstream file(path, std::ios::binary);
	if (!file) {
		return false;
	}

	// Read enough bytes to detect both formats
	char magic[7] = {0};
	file.read(magic, 7);
	if (!file.good() && !file.eof()) {
		return false;
	}

	// Check for v3 format ("VQVDB")
	if (std::string_view(magic, 5) == "VQVDB") {
		return true;
	}

	return false;
}

LoadResult<uint8_t> getFileVersion(const std::filesystem::path& path) {
	if (!std::filesystem::exists(path)) {
		return std::unexpected(LoadError::FileNotFound);
	}

	std::ifstream file(path, std::ios::binary);
	if (!file) {
		return std::unexpected(LoadError::FileOpenFailed);
	}

	// Read enough bytes to detect format
	char magic[7] = {0};
	file.read(magic, 7);
	if (!file) {
		return std::unexpected(LoadError::HeaderReadFailed);
	}


	// Check for v3 format ("VQVDB" + version byte)
	if (std::string_view(magic, 5) == "VQVDB") {
		return static_cast<uint8_t>(magic[5]);
	}

	return std::unexpected(LoadError::InvalidMagic);
}

}  // namespace vqvdb
