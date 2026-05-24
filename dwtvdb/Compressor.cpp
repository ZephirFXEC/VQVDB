#include "Compressor.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <stdexcept>

#include "CompressionStats.hpp"
#include "FileFormat.hpp"
#include "Logger.hpp"
#include "pocketfft_hdronly.h"

extern CompressionStats g_stats;

// ------------------------- small helpers ------------------------------
namespace {
inline bool test_bit(const uint8_t* buf, size_t i) { return (buf[i >> 3] >> (i & 7)) & 1u; }

inline void set_bit(uint8_t* buf, size_t i) { buf[i >> 3] |= static_cast<uint8_t>(1u << (i & 7)); }

inline size_t bytes_for_bits(size_t nbits) { return (nbits + 7) >> 3; }

// Band weight using an L1 metric with scaled time.
inline float bandWeightL1T(int kx, int ky, int kz, int kt, int T, float tscale, float wdc, float wlow, float wmid, float whigh) {
	const bool is_dc = (kx | ky | kz | kt) == 0;
	if (is_dc) return wdc;

	const float denomX = float(DCTCompressor::LeafDim > 1 ? (DCTCompressor::LeafDim - 1) : 1);
	const float denomT = float(T > 1 ? (T - 1) : 1);
	const float fx = float(kx) / denomX, fy = float(ky) / denomX, fz = float(kz) / denomX, ft = float(kt) / denomT;
	const float l1n = fx + fy + fz + tscale * ft;    // in [0,3+tscale]
	const float l1u = l1n / (3.0f + tscale);         // map to [0,1]
	constexpr float thrLow = 0.20f, thrMid = 0.55f;  // tunable
	return (l1u <= thrLow) ? wlow : (l1u <= thrMid ? wmid : whigh);
}

// RMS of active voxels only (uses OpenVDB value masks saved per frame)
inline float masked_rms(const Eigen::Tensor<float, 4>& block, const std::vector<uint8_t>& value_masks) {
	const int T = int(block.dimension(3));
	double sumsq = 0.0;
	size_t cnt = 0;
	for (int t = 0; t < T; ++t) {
		const uint8_t* m = value_masks.data() + size_t(t) * 64;
		for (int z = 0; z < DCTCompressor::LeafDim; ++z)
			for (int y = 0; y < DCTCompressor::LeafDim; ++y)
				for (int x = 0; x < DCTCompressor::LeafDim; ++x) {
					const size_t bit = size_t((z * DCTCompressor::LeafDim + y) * DCTCompressor::LeafDim + x);
					if (!test_bit(m, bit)) continue;
					const float v = block(x, y, z, t);
					sumsq += double(v) * double(v);
					++cnt;
				}
	}
	return cnt ? float(std::sqrt(sumsq / double(cnt))) : 0.0f;
}
}  // namespace

// ------------------------- Header -------------------------------------
FileHeader DCTCompressor::makeHeader() const {
	FileHeader h;
	h.version = 3;
	h.codec = static_cast<uint32_t>(CompressionCodec::Bitpack);
	h.qstep = par_.qstep;
	h.band_model = static_cast<uint32_t>(BandWeightModel::L1T);
	h.bw_time_scale = par_.timeWeight;
	h.bw_dc = par_.bwDc;
	h.bw_low = par_.bwLow;
	h.bw_mid = par_.bwMid;
	h.bw_high = par_.bwHigh;
	h.adapt_ref_rms = par_.adaptRefRms;
	h.adapt_floor = par_.adaptFloor;
	h.adapt_ceil = par_.adaptCeil;
	h.empty_rms_thresh = par_.emptyRmsThresh;
	// index_offset & brick_count are filled by the file writer elsewhere
	return h;
}

// ------------------------- Quantization --------------------------------
namespace {
struct QuantizationResult {
	bool skip = false;
	float adaptScale = 1.0f;       // per-brick s
	std::vector<int16_t> qcoeffs;  // dense quantized buffer (int16)
};

struct PackedCoeffs {
	bool skip = false;
	float adaptScale = 1.0f;
	uint32_t nnz = 0;
	std::vector<uint8_t> coeffMask;    // N bits
	std::vector<int16_t> coeffValues;  // nnz values
};

// Core forward quantization with adaptive + band weights.
// IMPORTANT (bug fix): deadzone is applied in *value domain* using
//   |c| < deadzoneQ * step  ==> q = 0
// This fixes previous logic where deadzone compared to rounded Q-domain
// values, which could inconsistently keep/remove ±1 bins.
static QuantizationResult quantize_block(const Eigen::Tensor<float, 4>& dct, int T, float spatialRms, const DCTCompressor::Params& par) {
	QuantizationResult res;
	res.qcoeffs.resize(size_t(DCTCompressor::LeafVoxelCount) * size_t(T), 0);

	// Empty block skipping
	if (spatialRms < par.emptyRmsThresh) {
		res.skip = true;
		return res;
	}

	// Adaptive scale s = clamp(rms/ref, floor, ceil)
	const float ref = std::max(par.adaptRefRms, 1e-12f);
	const float s = std::clamp(spatialRms / ref, par.adaptFloor, par.adaptCeil);
	res.adaptScale = s;

	const float qbase = par.qstep;
	const float tscale = par.timeWeight;
	const float wdc = par.bwDc, wlow = par.bwLow, wmid = par.bwMid, whigh = par.bwHigh;

	size_t i = 0;
	for (int t = 0; t < T; ++t)
		for (int z = 0; z < DCTCompressor::LeafDim; ++z)
			for (int y = 0; y < DCTCompressor::LeafDim; ++y)
				for (int x = 0; x < DCTCompressor::LeafDim; ++x, ++i) {
					const float w = bandWeightL1T(x, y, z, t, T, tscale, wdc, wlow, wmid, whigh);
					const float step = qbase * s * w;
					const float c = dct(x, y, z, t);

					// Deadzone in value domain
					if (std::fabs(c) < par.deadzoneQ * step) {
						res.qcoeffs[i] = 0;
						continue;
					}

					// Uniform scalar quantizer (mid-rise around 0)
					const float qf = std::round(c / step);
					const float clamped = std::max(-32768.0f, std::min(32767.0f, qf));
					res.qcoeffs[i] = static_cast<int16_t>(clamped);
				}

	return res;
}

static PackedCoeffs pack_sparsity(const std::vector<int16_t>& q, float adaptScale) {
	PackedCoeffs p;
	p.adaptScale = adaptScale;
	const size_t N = q.size();
	p.coeffMask.assign(bytes_for_bits(N), 0);
	p.coeffValues.reserve(N);
	for (size_t i = 0; i < N; ++i)
		if (q[i] != 0) {
			set_bit(p.coeffMask.data(), i);
			p.coeffValues.push_back(q[i]);
		}
	p.nnz = static_cast<uint32_t>(p.coeffValues.size());
	return p;
}
}  // namespace


// ------------------------- Encode -------------------------------------
DCTCompressor::EncodedRun DCTCompressor::encodeRun(const RunInput& in) const {
	if (in.frames.empty()) throw std::runtime_error("encodeRun: empty frames");
	const int T = int(in.frames.size());

	Eigen::Tensor<float, 4> block(LeafDim, LeafDim, LeafDim, T);
	std::vector<uint8_t> value_masks(size_t(T) * 64);

	// Copy voxels + value masks
	for (int t = 0; t < T; ++t) {
		const auto& grid = in.frames[(size_t)t];
		const LeafType* leaf = grid->tree().probeConstLeaf(in.origin);
		if (!leaf) throw std::runtime_error("encodeRun: missing leaf despite activity mask");

		Eigen::TensorMap<Eigen::Tensor<const float, 3>> src(leaf->buffer().data(), LeafDim, LeafDim, LeafDim);
		block.chip(t, 3) = src;  // copy into time slice

		const auto& mask = leaf->valueMask();
		uint8_t* dstMask = value_masks.data() + size_t(t) * 64;
		std::memset(dstMask, 0, 64);
		for (int bit = 0; bit < LeafVoxelCount; ++bit)
			if (mask.isOn(bit)) set_bit(dstMask, size_t(bit));
	}

	// Compute masked spatial RMS in *spatial* domain (before DCT)
	const float spatialRms = masked_rms(block, value_masks);

	// Forward 4-D DCT
	dct4d(block, /*inverse=*/false);

	// Quantize with the corrected deadzone logic
	auto qres = quantize_block(block, T, spatialRms, par_);

	// Stats (optional)
	const uint64_t coeffCount = qres.qcoeffs.size();
	uint64_t nnzCount = 0, deadzoneCount = 0;
	for (auto v : qres.qcoeffs) {
		if (v != 0)
			++nnzCount;
		else
			++deadzoneCount;
	}
	g_stats.recordBrick(qres.skip, qres.adaptScale, coeffCount, nnzCount, deadzoneCount);

	// Pack sparsity
	PackedCoeffs packed;
	packed.skip = qres.skip;
	packed.adaptScale = qres.adaptScale;
	if (!packed.skip) packed = pack_sparsity(qres.qcoeffs, qres.adaptScale);

	// Build result
	EncodedRun out;
	out.origin = in.origin;
	out.start_frame = in.start_frame;
	out.num_frames = T;
	out.v3.skip = packed.skip;
	out.v3.adaptScale = packed.adaptScale;
	out.v3.nnz = packed.nnz;
	out.v3.coeffMask = std::move(packed.coeffMask);
	out.v3.coeffValues = std::move(packed.coeffValues);
	out.valueMasks = std::move(value_masks);
	return out;
}

// ------------------------- Dequant + inverse DCT ----------------------
Eigen::Tensor<float, 4> DCTCompressor::dequantizeAndInverseTransform(const std::vector<int16_t>& q_coeffs, int T, float adaptScale) const {
	if (q_coeffs.size() != size_t(T) * LeafVoxelCount) throw std::runtime_error("dequantize: size mismatch");

	Eigen::Tensor<float, 4> block(LeafDim, LeafDim, LeafDim, T);

	const float qbase = par_.qstep;
	const float tscale = par_.timeWeight;
	size_t i = 0;
	for (int t = 0; t < T; ++t)
		for (int z = 0; z < LeafDim; ++z)
			for (int y = 0; y < LeafDim; ++y)
				for (int x = 0; x < LeafDim; ++x, ++i) {
					const float w = bandWeightL1T(x, y, z, t, T, tscale, par_.bwDc, par_.bwLow, par_.bwMid, par_.bwHigh);
					const float step = qbase * adaptScale * w;
					block(x, y, z, t) = float(q_coeffs[i]) * step;
				}

	dct4d(block, /*inverse=*/true);
	return block;
}

// ------------------------- DCT4D --------------------------------------
void DCTCompressor::dct4d(Eigen::Tensor<float, 4>& block, const bool inverse) {
	const size_t T = block.dimension(3);

	const pocketfft::shape_t s{(size_t)LeafDim, (size_t)LeafDim, (size_t)LeafDim, T};
	const pocketfft::stride_t st{(long long)sizeof(float), (long long)(sizeof(float) * s[0]), (long long)(sizeof(float) * s[0] * s[1]),
	                             (long long)(sizeof(float) * s[0] * s[1] * s[2])};
	const pocketfft::shape_t axes{0, 1, 2, 3};

	if (!inverse)
		pocketfft::dct<float>(s, st, st, axes, 2, block.data(), block.data(), 1.0f, true, 1);
	else
		pocketfft::dct<float>(s, st, st, axes, 3, block.data(), block.data(), 1.0f, true, 1);
}

// ------------------------- Decompress (v3) ----------------------------
VDBSequence DCTCompressor::decompress(const std::string& input_path) const {
	logger::info("Starting decompression from {}", input_path);

	std::ifstream file(input_path, std::ios::binary);
	if (!file) throw std::runtime_error("Failed to open input file");

	FileHeader header{};
	file.read(reinterpret_cast<char*>(&header), sizeof(header));
	if (std::string(header.magic, 4) != "V4DC") throw std::runtime_error("Invalid file magic");
	if (header.version != 3 || header.codec != static_cast<uint32_t>(CompressionCodec::Bitpack))
		throw std::runtime_error("Unsupported format (expect v3 Bitpack)");

	if (header.brick_count == 0) return VDBSequence{0};

	file.seekg((std::streamoff)header.index_offset, std::ios::beg);
	std::vector<BrickIndexEntry> index(header.brick_count);
	file.read(reinterpret_cast<char*>(index.data()), (std::streamsize)(index.size() * sizeof(BrickIndexEntry)));
	if (!file) throw std::runtime_error("Failed to read index table");

	// Determine frame count
	int max_frame = 0;
	for (const auto& e : index) max_frame = std::max(max_frame, e.start_frame + e.num_frames);
	if (max_frame <= 0) return VDBSequence{0};

	// Prepare output sequence
	VDBSequence seq(max_frame);
	for (int i = 0; i < max_frame; ++i) seq[i].grid = openvdb::FloatGrid::create();

	// Decoder params: only band/step settings matter for dequantization
	DCTCompressor tmp = *this;
	tmp.par_.qstep = header.qstep;
	tmp.par_.timeWeight = header.bw_time_scale;
	tmp.par_.bwDc = header.bw_dc;
	tmp.par_.bwLow = header.bw_low;
	tmp.par_.bwMid = header.bw_mid;
	tmp.par_.bwHigh = header.bw_high;

	// Decode bricks
	for (const auto& ent : index) {
		const int T = ent.num_frames;
		const bool skip = (ent.flags & BRICK_FLAG_SKIP) != 0;

		// Read payload
		file.seekg((std::streamoff)ent.data_offset, std::ios::beg);
		uint32_t nnz = 0;
		file.read(reinterpret_cast<char*>(&nnz), sizeof(uint32_t));

		std::vector<uint8_t> coeffMask(ent.coeff_mask_bytes);
		if (ent.coeff_mask_bytes) file.read(reinterpret_cast<char*>(coeffMask.data()), (std::streamsize)ent.coeff_mask_bytes);

		std::vector<int16_t> coeffValues(nnz);
		if (nnz) file.read(reinterpret_cast<char*>(coeffValues.data()), (std::streamsize)(nnz * sizeof(int16_t)));

		std::vector<uint8_t> value_masks(ent.value_mask_bytes);
		if (ent.value_mask_bytes) file.read(reinterpret_cast<char*>(value_masks.data()), (std::streamsize)ent.value_mask_bytes);

		// Expand & dequantize
		Eigen::Tensor<float, 4> block(LeafDim, LeafDim, LeafDim, T);
		if (!skip) {
			const size_t N = (size_t)LeafVoxelCount * (size_t)T;
			std::vector<int16_t> q_dense(N, 0);
			size_t k = 0;
			for (size_t i = 0; i < N; ++i)
				if (ent.coeff_mask_bytes && test_bit(coeffMask.data(), i)) {
					if (k >= coeffValues.size()) throw std::runtime_error("Corrupt payload: nnz mismatch");
					q_dense[i] = coeffValues[k++];
				}
			if (k != coeffValues.size()) throw std::runtime_error("Corrupt payload: nnz tail mismatch");
			block = tmp.dequantizeAndInverseTransform(q_dense, T, ent.adapt_scale);
		} else {
			block.setZero();
		}

		// Scatter back to frames
		for (int t = 0; t < T; ++t) {
			const int frame_idx = ent.start_frame + t;
			auto& grid = seq[frame_idx].grid;
			LeafType* leaf = grid->tree().touchLeaf(ent.origin);

			// restore valueMask
			std::memcpy((void*)&leaf->valueMask(), value_masks.data() + (size_t)t * 64, 64);
			Eigen::TensorMap<Eigen::Tensor<float, 3>> leaf_map(leaf->buffer().data(), LeafDim, LeafDim, LeafDim);

			const auto& mask = leaf->valueMask();
			if (mask.isOn()) {
				leaf_map = block.chip(t, 3);
			} else {
				for (int z = 0; z < LeafDim; ++z) {
					for (int y = 0; y < LeafDim; ++y) {
						for (int x = 0; x < LeafDim; ++x) {
							const int bit = (z * LeafDim + y) * LeafDim + x;
							if (mask.isOn(bit)) leaf_map(x, y, z) = block(x, y, z, t);
						}
					}
				}
			}
		}
	}

	logger::info("Decompression finished. Reconstructed {} frames", (int)seq.size());
	return seq;
}