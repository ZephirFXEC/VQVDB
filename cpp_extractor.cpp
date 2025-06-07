//
// Created by zphrfx on 07/06/2025.
//
#include <openvdb/io/File.h>
#include <openvdb/openvdb.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

// This function extracts all active leaf nodes from a float grid in a VDB file.
// It returns a list of NumPy arrays, where each array is a flattened 8x8x8 leaf.
py::list extract_vdb_leaves(const std::string& file_path, const std::string& grid_name) {
	// Initialize OpenVDB
	openvdb::initialize();

	// Open the VDB file
	openvdb::io::File file(file_path);
	try {
		file.open();
	} catch (const openvdb::IoError& e) {
		file.close();
		throw std::runtime_error("Could not open VDB file: " + std::string(e.what()));
	}

	// Get the grid by name
	openvdb::GridBase::Ptr baseGrid = file.readGrid(grid_name);
	file.close();

	if (!baseGrid) {
		throw std::runtime_error("Grid '" + grid_name + "' not found in file '" + file_path + "'.");
	}

	// Cast to a float grid
	openvdb::FloatGrid::Ptr grid = openvdb::gridPtrCast<openvdb::FloatGrid>(baseGrid);
	if (!grid) {
		throw std::runtime_error("Grid '" + grid_name + "' is not a FloatGrid.");
	}

	// The dimension of a standard float leaf node (8x8x8)
	constexpr int leaf_dim = openvdb::FloatGrid::TreeType::LeafNodeType::DIM;
	constexpr int leaf_voxel_count = leaf_dim * leaf_dim * leaf_dim;  // 512

	py::list all_leaves;
	auto accessor = grid->getConstAccessor();

	// Iterate over all active leaf nodes in the tree
	for (auto leaf_it = grid->tree().cbeginLeaf(); leaf_it; ++leaf_it) {
		const auto& leaf = *leaf_it;

		// Create a NumPy array from the C++ vector (this copies the data)
		auto np_array = py::array_t<float>(leaf_voxel_count, leaf.buffer().data());

		// Reshape to 3D (8, 8, 8) for easier use in PyTorch
		np_array = np_array.reshape({leaf_dim, leaf_dim, leaf_dim});

		all_leaves.append(np_array);
	}

	return all_leaves;
}

// pybind11 module definition
PYBIND11_MODULE(vdb_leaf_extractor, m) {
	m.doc() = "A C++ module to extract VDB leaf nodes efficiently";
	m.def("extract_leaves", &extract_vdb_leaves, "Extracts active float leaves from a VDB file", py::arg("file_path"),
	      py::arg("grid_name"));
}