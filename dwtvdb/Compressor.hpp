#pragma once

#include <memory>
#include <string>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>

#include <openvdb/openvdb.h>

#include "FileFormat.hpp"
#include "VDBStreamReader.hpp"

class DCTCompressor {
public:
	struct Params {
		float qstep = 0.5f;
		int zstdLevel = 3;
	};

	explicit DCTCompressor(Params p) : par_(p) {
	}

	// Input for encoding one leaf run inside a GOP
	struct RunInput {
		openvdb::Coord origin;
		int start_frame;
		std::vector<openvdb::FloatGrid::Ptr> frames;
	};

	// Output of encoding a single leaf run
	struct EncodedRun {
		RunMetadata meta;
		std::vector<char> blob; // zstd payload with size prefix
		std::vector<uint8_t> valueMasks; // T * 64
	};

	// Encode a single leaf run
	EncodedRun encodeRun(const RunInput& in) const;

	// Streaming decompressor for a file in new format (version 2)
	[[nodiscard]] VDBSequence decompress(const std::string& input_path) const;

	// File header creation
	FileHeader makeHeader() const;

private:
	using LeafType = openvdb::FloatGrid::TreeType::LeafNodeType;
	static constexpr int LeafDim = LeafType::DIM;
	static constexpr int LeafVoxelCount = LeafType::SIZE;

	Params par_;

	// Transform core
	std::vector<int16_t> transformAndQuantize(
		Eigen::Tensor<float, 4>& block) const;

	Eigen::Tensor<float, 4> dequantizeAndInverseTransform(
		const std::vector<int16_t>& q_coeffs, int T) const;

	static void dct4d(Eigen::Tensor<float, 4>& block, bool inverse);
};