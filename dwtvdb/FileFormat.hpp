/*
* Copyright (c) 2025, Enzo Crema
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See the LICENSE file in the project root for full license text.
 */

#ifndef DWTVDB_FILEFORMAT_HPP
#define DWTVDB_FILEFORMAT_HPP

#include <openvdb/Types.h>

#pragma pack(push, 1)

enum class CompressionCodec : uint32_t { Zstd = 1, Zlib = 2, None = 0 };

struct FileHeader {
	char magic[4] = {'V', '4', 'D', 'C'};
	uint32_t version = 2; // streaming, per-run records
	float qstep{};
	uint32_t codec = static_cast<uint32_t>(CompressionCodec::Zlib);
	uint32_t reserved[4] = {0, 0, 0, 0};
};

struct RunMetadata {
	openvdb::Coord origin;
	int32_t start_frame{};
	int32_t num_frames{};
	uint32_t blob_size{}; // compressed coefficient bytes (including size prefix)
	uint32_t mask_bytes{}; // T * 64
};

#pragma pack(pop)

#endif //DWTVDB_FILEFORMAT_HPP