#pragma once

#include <openvdb/openvdb.h>
#include <openvdb/tools/GridOperators.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

// Simple NPY file writer for float32 and int32 arrays
class NpyWriter {
   public:
	static bool writeFloat32(const std::filesystem::path& filename, const std::vector<float>& data, const std::vector<size_t>& shape) {
		std::ofstream file(filename, std::ios::binary);
		if (!file) return false;

		// Calculate total elements and verify data size
		size_t total_elements = 1;
		for (auto dim : shape) total_elements *= dim;
		if (data.size() != total_elements) {
			std::cerr << "Data size mismatch with shape dimensions\n";
			return false;
		}

		// Construct header
		std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': (";
		for (size_t i = 0; i < shape.size(); ++i) {
			header += std::to_string(shape[i]);
			if (i < shape.size() - 1) header += ", ";
		}
		header += "), }";

		// Pad header to multiple of 64 bytes (NPY format requirement)
		size_t header_len = header.size() + 1;  // +1 for null terminator
		size_t padding_needed = (64 - ((10 + header_len) % 64)) % 64;
		header.append(padding_needed, ' ');
		header.push_back('\n');

		// Write NPY magic number and version
		file.write("\x93NUMPY", 6);
		const uint8_t major = 1, minor = 0;
		file.write(reinterpret_cast<const char*>(&major), 1);
		file.write(reinterpret_cast<const char*>(&minor), 1);

		// Write header length (little endian) and header
		uint16_t header_size = static_cast<uint16_t>(header.size());
		file.write(reinterpret_cast<char*>(&header_size), 2);
		file.write(header.data(), header_size);

		// Write data
		file.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));

		return file.good();
	}

	static bool writeInt32(const std::filesystem::path& filename, const std::vector<int>& data, const std::vector<size_t>& shape) {
		std::ofstream file(filename, std::ios::binary);
		if (!file) return false;

		size_t total_elements = 1;
		for (auto dim : shape) total_elements *= dim;
		if (data.size() != total_elements) {
			std::cerr << "Data size mismatch with shape dimensions\n";
			return false;
		}

		std::string header = "{'descr': '<i4', 'fortran_order': False, 'shape': (";
		for (size_t i = 0; i < shape.size(); ++i) {
			header += std::to_string(shape[i]);
			if (i < shape.size() - 1) header += ", ";
		}
		header += "), }";

		size_t header_len = header.size() + 1;
		size_t padding_needed = (64 - ((10 + header_len) % 64)) % 64;
		header.append(padding_needed, ' ');
		header.push_back('\n');

		file.write("\x93NUMPY", 6);
		const uint8_t major = 1, minor = 0;
		file.write(reinterpret_cast<const char*>(&major), 1);
		file.write(reinterpret_cast<const char*>(&minor), 1);

		uint16_t header_size = static_cast<uint16_t>(header.size());
		file.write(reinterpret_cast<char*>(&header_size), 2);
		file.write(header.data(), header_size);

		file.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(int));

		return file.good();
	}
};

template <typename GridType>
bool extractLeafData(const std::string& vdbFilePath, const std::string& gridName, const std::string& outputPath) {
	using namespace openvdb;

	// Initialize OpenVDB
	openvdb::initialize();

	// Load VDB file
	io::File file(vdbFilePath);
	try {
		file.open();
	} catch (const std::exception& e) {
		std::cerr << "Error opening VDB file: " << e.what() << std::endl;
		return false;
	}

	// Get grid
	typename FloatGrid::Ptr grid;

	if (gridName.empty()) {
		// Use first grid if name not specified
		GridBase::Ptr baseGrid = file.readGrid(file.beginName().gridName());
		grid = std::dynamic_pointer_cast<GridType>(baseGrid);
		if (!grid) {
			std::cerr << "First grid is not of expected type\n";
			return false;
		}
	} else {
		// Get grid by name
		GridBase::Ptr baseGrid = file.readGrid(gridName);
		if (!baseGrid) {
			std::cerr << "Grid '" << gridName << "' not found\n";
			return false;
		}
		grid = std::dynamic_pointer_cast<GridType>(baseGrid);
		if (!grid) {
			std::cerr << "Grid '" << gridName << "' is not of expected type\n";
			return false;
		}
	}

	file.close();

	const auto& tree = grid->tree();
	const tree::LeafManager leafManager{tree};

	const uint32_t leafCount = leafManager.leafCount();
	std::cout << "Found " << leafCount << " leaf nodes\n";

	// Extract leaf data and origins
	std::vector<float> leafData(leafCount * GridType::TreeType::LeafNodeType::SIZE);
	std::vector<int> originsData(leafCount * 3);

	size_t leafIndex = 0;
	constexpr int LEAF_DIM = 8;
	for (auto iter = grid->tree().cbeginLeaf(); iter; ++iter) {
		const auto& leaf = *iter;

		if (leaf.isEmpty()) {
			continue;
		}

		// Get origin of this leaf
		Coord origin = leaf.origin();
		originsData[leafIndex * 3 + 0] = origin.x();
		originsData[leafIndex * 3 + 1] = origin.y();
		originsData[leafIndex * 3 + 2] = origin.z();

		// Extract all voxels in the leaf (8x8x8)
		for (int z = 0; z < LEAF_DIM; z++) {
			for (int y = 0; y < LEAF_DIM; y++) {
				for (int x = 0; x < LEAF_DIM; x++) {
					Coord xyz(origin.x() + x, origin.y() + y, origin.z() + z);
					float value = static_cast<float>(leaf.getValue(xyz));

					// Store in flattened array in NCHW format:
					// [leafIndex][z][y][x]
					size_t offset = leafIndex * LEAF_DIM * LEAF_DIM * LEAF_DIM + z * LEAF_DIM * LEAF_DIM + y * LEAF_DIM + x;
					leafData[offset] = value;
				}
			}
		}

		leafIndex++;
	}

	// Write voxel data
	std::vector<size_t> shape = {leafCount, LEAF_DIM, LEAF_DIM, LEAF_DIM};
	bool success = NpyWriter::writeFloat32(outputPath, leafData, shape);

	// Also write origins in a separate `/origin` folder
	std::filesystem::path outDir = std::filesystem::path(outputPath).parent_path();
	std::filesystem::path originDir = outDir / "origin";
	std::filesystem::create_directories(originDir);
	std::filesystem::path originPath = originDir / std::filesystem::path(outputPath).filename().replace_extension("_origins.npy");
	std::vector<size_t> originShape = {leafCount, 3};
	bool originSuccess = NpyWriter::writeInt32(originPath, originsData, originShape);

	if (success && originSuccess) {
		std::cout << "Successfully wrote " << leafCount << " leaf nodes to " << outputPath << " and origins to " << originPath << "\n";
	} else {
		std::cerr << "Failed to write NPY files" << "\n";
	}

	return success && originSuccess;
}
