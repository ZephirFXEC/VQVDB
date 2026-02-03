/*
 * Copyright (c) 2025, Enzo Crema
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Standalone VQVDB file loader for GPU rendering.
 * No Houdini or OpenVDB dependencies.
 */

#pragma once

#include <expected>
#include <filesystem>

#include "vqvdb_types.hpp"

namespace vqvdb {

// ============================================================================
// Error Handling
// ============================================================================

/// Error codes for VQVDB loading
enum class LoadError {
	FileNotFound,
	FileOpenFailed,
	InvalidMagic,
	UnsupportedVersion,
	HeaderReadFailed,
	GridMetadataReadFailed,
	BlockDataReadFailed,
	FileTruncated,
	AllocationFailed,
	InvalidData
};

/// Convert error code to human-readable string
[[nodiscard]] const char* errorToString(LoadError error) noexcept;

/// Result type for load operations
template <typename T>
using LoadResult = std::expected<T, LoadError>;

// ============================================================================
// Loader Configuration
// ============================================================================

/// Configuration options for the VQVDB loader
struct LoaderConfig {
	/// Read buffer size for file I/O (default: 64 MB)
	size_t ioBufferSize = 64 * 1024 * 1024;

	/// Maximum number of blocks to load per grid (0 = unlimited)
	size_t maxBlocksPerGrid = 0;

	/// Skip loading block data (metadata only)
	bool metadataOnly = false;

	/// Validate data integrity during load
	bool validateData = true;
};

// ============================================================================
// Public API
// ============================================================================

/// Load a complete VQVDB file from disk.
/// @param path Path to the .vqvdb file
/// @param config Optional loader configuration
/// @return VQVDBFile on success, or LoadError on failure
[[nodiscard]] LoadResult<VQVDBFile> loadFile(const std::filesystem::path& path, const LoaderConfig& config = {});

/// Load only the file metadata (grid names, block counts, transforms)
/// without reading the actual block data.
/// @param path Path to the .vqvdb file
/// @return VQVDBFile with empty block data on success, or LoadError on failure
[[nodiscard]] LoadResult<VQVDBFile> loadMetadata(const std::filesystem::path& path);

/// Check if a file appears to be a valid VQVDB file (magic number check)
/// @param path Path to check
/// @return true if file has valid VQVDB magic number
[[nodiscard]] bool isVQVDBFile(const std::filesystem::path& path);

/// Get the version number from a VQVDB file without fully loading it
/// @param path Path to the .vqvdb file
/// @return Version number on success, or LoadError on failure
[[nodiscard]] LoadResult<uint8_t> getFileVersion(const std::filesystem::path& path);

}  // namespace vqvdb
