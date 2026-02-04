/*
 * Copyright (c) 2025, Enzo Crema
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Codebook file loader for GPU rendering.
 * Loads .bin files exported from Python training scripts.
 */

#pragma once

#include <cstdint>
#include <filesystem>

#include "vqvdb_types.hpp"

namespace vqvdb {

/// Magic number for codebook files ("VQCB" in little-endian)
inline constexpr uint32_t kCodebookMagic = 0x56514342;

/// Binary file header (16 bytes)
struct CodebookFileHeader {
	uint32_t magic;          ///< Must be kCodebookMagic
	uint32_t numEmbeddings;  ///< Number of codebook entries (typically 256)
	uint32_t embeddingDim;   ///< Dimension of each embedding (typically 128)
	uint32_t reserved;       ///< Reserved for future use
};

static_assert(sizeof(CodebookFileHeader) == 16, "Header must be 16 bytes");

/// Load a codebook from a binary file.
/// @param path Path to the .bin file created by export_codebook.py
/// @return CodebookData on success, or CodebookError on failure
[[nodiscard]] CodebookResult<Codebook> loadCodebookFile(const std::filesystem::path& path);

/// Check if a file appears to be a valid codebook file (magic number check)
/// @param path Path to check
/// @return true if file has valid codebook magic number
[[nodiscard]] bool isCodebookFile(const std::filesystem::path& path);

}  // namespace vqvdb
