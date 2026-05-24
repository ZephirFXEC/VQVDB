#pragma once

#include <openvdb/openvdb.h>

#include <memory>
#include <string>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>

#include "FileFormat.hpp"
#include "VDBStreamReader.hpp"

// DCT-based compressor for OpenVDB leaf runs. The implementation uses:
//   - separable 4-D DCT (8x8x8 spatial, T temporal)
//   - adaptive + band-weighted scalar quantization to int16
//   - sparsity bitmask + packed values
class DCTCompressor {
   public:
	// Tunable parameters (filled from dataset stats via paramsFromKnob).
	struct Params {
		// Base quantization step (q_b). Real step per coeff is
		//   q = q_b * adaptScale * bandWeight(k).
		float qstep = 1.0f;

		// Skip bricks with masked spatial RMS below this threshold.
		float emptyRmsThresh = 1e-2f;

		// Adaptive quantization scale s = clamp(rms/ref, floor, ceil).
		float adaptRefRms = 0.5f;
		float adaptFloor = 0.1f;
		float adaptCeil = 1.0f;

		// Band-weight parameters (see bandWeightL1T).
		float bwDc = 0.25f;
		float bwLow = 0.5f;
		float bwMid = 0.75f;
		float bwHigh = 1.0f;
		float timeWeight = 1.0f;  // time factor in L1 metric

		float deadzoneQ = 0.5f;

		void toString(std::string& out) const {
			out = "DCTCompressor Params:\n";
			out += "  qstep: " + std::to_string(qstep) + "\n";
			out += "  emptyRmsThresh: " + std::to_string(emptyRmsThresh) + "\n";
			out += "  adaptRefRms: " + std::to_string(adaptRefRms) + "\n";
			out += "  adaptFloor: " + std::to_string(adaptFloor) + "\n";
			out += "  adaptCeil: " + std::to_string(adaptCeil) + "\n";
			out += "  bwDc: " + std::to_string(bwDc) + "\n";
			out += "  bwLow: " + std::to_string(bwLow) + "\n";
			out += "  bwMid: " + std::to_string(bwMid) + "\n";
			out += "  bwHigh: " + std::to_string(bwHigh) + "\n";
			out += "  timeWeight: " + std::to_string(timeWeight) + "\n";
			out += "  deadzoneQ: " + std::to_string(deadzoneQ) + "\n";
		}
	};

	// Lightweight dataset statistics used to derive Params from a user knob.
	struct DatasetStats {
		float minVal = 0.f, maxVal = 0.f;
		float spatialRms = 1e-6f;         // σ_spatial (robust)
		float bandVar[4] = {0, 0, 0, 0};  // energy in DC/Low/Mid/High
		float temporalToSpatialVar = 0.f;
		float medianLeafRms = 0.f;

		static DatasetStats estimateStats(const std::vector<openvdb::FloatGrid::Ptr>& frames, int sampleFrames, int sampleLeaves);
	};

	explicit DCTCompressor(Params p) : par_(p) {}

	void setParams(const Params& p) { par_ = p; }
	std::string getParams() const {
		std::string s;
		par_.toString(s);
		return s;
	}

	// Encoding input for one leaf run [start_frame, start_frame+frames.size())
	struct RunInput {
		openvdb::Coord origin;
		int start_frame = 0;
		std::vector<openvdb::FloatGrid::Ptr> frames;  // size = T
	};

	// Result of encoding one run
	struct EncodedRun {
		openvdb::Coord origin;
		int32_t start_frame = 0;
		int32_t num_frames = 0;
		std::vector<uint8_t> valueMasks;  // T * 64 (OpenVDB valueMask per frame)
		struct V3Payload {
			bool skip = false;
			float adaptScale = 1.0f;
			uint32_t nnz = 0;
			std::vector<uint8_t> coeffMask;    // bitmask over 8*8*8*T
			std::vector<int16_t> coeffValues;  // non-zero values
		} v3;
	};

	// Encode a single leaf run (no file I/O here).
	[[nodiscard]] EncodedRun encodeRun(const RunInput& in) const;

	// Streaming decompressor for a v3 file in GPU-friendly layout.
	[[nodiscard]] VDBSequence decompress(const std::string& input_path) const;

	// Build an on-disk header from current parameters.
	[[nodiscard]] FileHeader makeHeader() const;

	// Public constants
	using LeafType = openvdb::FloatGrid::TreeType::LeafNodeType;
	static constexpr int LeafDim = LeafType::DIM;          // usually 8
	static constexpr int LeafVoxelCount = LeafType::SIZE;  // 512

	// Separable 4-D DCT using pocketfft (DCT-II forward, DCT-III inverse).
	static void dct4d(Eigen::Tensor<float, 4>& block, bool inverse);

	// Map a [0,1] quality knob to Params using dataset statistics.
	static Params paramsFromKnob(float k, const DatasetStats& S);

   private:
	Params par_{};

	[[nodiscard]] Eigen::Tensor<float, 4> dequantizeAndInverseTransform(const std::vector<int16_t>& q_coeffs, int T,
	                                                                    float adaptScale) const;
};
