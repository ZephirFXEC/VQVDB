#include "VDBStreamReader.hpp"

#include <openvdb/io/File.h>

#include <cstring>

VDBFrame VDBLoader::loadFrame(const std::string& path, const std::string& gridName) const {
	openvdb::io::File file(path);
	file.open();
	openvdb::GridBase::Ptr base = file.readGrid(gridName);
	file.close();

	auto grid = openvdb::gridPtrCast<openvdb::FloatGrid>(base);
	if (!grid) throw std::runtime_error("Grid is not FloatGrid");

	VDBFrame frame;
	frame.grid = grid;

	constexpr int B = 8;

	for (auto leafIt = grid->tree().cbeginLeaf(); leafIt; ++leafIt) {
		DenseBlock block;
		block.origin = leafIt->origin();
		block.data = Eigen::Tensor<float, 3>(B, B, B);

		std::memcpy(block.data.data(), leafIt->buffer().data(), B * B * B * sizeof(float));
		frame.blocks.emplace_back(std::move(block));
	}
	return frame;
}

VDBSequence VDBLoader::loadSequence(const std::vector<std::string>& paths, const std::string& gridName) const {
	std::vector<VDBFrame> frames;
	frames.reserve(paths.size());
	for (const auto& p : paths) frames.emplace_back(loadFrame(p, gridName));
	return VDBSequence(std::move(frames));
}