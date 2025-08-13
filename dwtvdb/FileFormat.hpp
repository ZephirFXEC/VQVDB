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

#include <cstdint>

#pragma pack(push, 1)

// --- On-disk codec enum ------------------------------------------------------

enum class CompressionCodec : uint32_t { None = 0, Zstd = 1, Zlib = 2, Bitpack = 3 };

// Band-weight model used by decoder to recompute per-frequency weights.
enum class BandWeightModel : uint32_t {
	L1T = 0,  // L1 in (kx,ky,kz, kt*scale)
	L2T = 1   // (reserved)
};

// Version 3 header (GPU-friendly, bitpacked layout)
struct FileHeader {
	char magic[4] = {'V', '4', 'D', 'C'};  // "V4DC"
	uint32_t version = 3;
	uint32_t codec = static_cast<uint32_t>(CompressionCodec::Bitpack);

	// Global base quantization step (q_b). Decoder multiplies by per-brick
	// adapt_scale and by band weights to reconstruct each coefficient's step.
	float qstep = 0.01f;

	// Band model + parameters for decoder-side recomputation of w(k).
	uint32_t band_model = static_cast<uint32_t>(BandWeightModel::L1T);
	float bw_time_scale = -1.0f;  // t-axis contribution in the L1 metric
	float bw_dc = -1.0f;          // DC protection multiplier
	float bw_low = -1.0f;         // low band multiplier
	float bw_mid = -1.0f;         // mid band multiplier
	float bw_high = -1.0f;        // high band multiplier

	// Adaptive quantization parameters (for analysis/telemetry; decode only
	// requires per-brick adapt_scale value).
	float adapt_ref_rms = -1.0f;
	float adapt_floor = -1.0f;
	float adapt_ceil = -1.0f;
	float empty_rms_thresh = -1.0f;

	// Index table pointer + brick count.
	uint64_t index_offset = 0;
	uint32_t brick_count = 0;

	uint32_t reserved[9] = {0};
};

// Flags for index entries
enum : uint32_t { BRICK_FLAG_SKIP = 1u << 0 };

// One index record per encoded brick/run.
struct BrickIndexEntry {
	openvdb::Coord origin;  // 12B (3×int32)
	int32_t start_frame = 0;
	int32_t num_frames = 0;

	uint32_t flags = 0;        // BRICK_FLAG_*
	float adapt_scale = 1.0f;  // per-brick scale s

	uint32_t coeff_mask_bytes = 0;    // bitmask over (8*8*8*T) coeffs
	uint32_t coeff_values_bytes = 0;  // nnz * sizeof(int16)
	uint32_t value_mask_bytes = 0;    // T * 64 (OpenVDB valueMask)

	// Payload layout at data_offset:
	//   uint32_t nnz
	//   uint8_t  coeff_mask[coeff_mask_bytes]
	//   int16_t  coeff_values[nnz]
	//   uint8_t  value_masks[value_mask_bytes]
	uint64_t data_offset = 0;
};

#pragma pack(pop)

#endif  // DWTVDB_FILEFORMAT_HPP