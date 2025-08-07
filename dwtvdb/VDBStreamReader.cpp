#include "VDBStreamReader.hpp"

#include <openvdb/io/File.h>

#include <stdexcept>

#include "logger.hpp"

VDBFrame VDBLoader::loadFrame(const std::string& path, const std::string& gridName) const {
	try {
		logger::debug("Loading grid '{}' from '{}'", gridName, path);
		openvdb::io::File file(path);
		file.open();

		// Check if the grid exists before trying to read it.
		if (!file.hasGrid(gridName)) {
			file.close();
			throw std::runtime_error("Grid '" + gridName + "' not found in file: " + path);
		}

		openvdb::GridBase::Ptr base = file.readGrid(gridName);
		file.close();

		// Cast to the grid type we expect.
		auto grid = openvdb::gridPtrCast<openvdb::FloatGrid>(base);
		if (!grid) {
			throw std::runtime_error("Grid '" + gridName + "' in file '" + path + "' is not a FloatGrid.");
		}

		VDBFrame frame;
		frame.grid = grid;
		return frame;
	} catch (const openvdb::IoError& e) {
		throw std::runtime_error("OpenVDB I/O error loading file " + path + ": " + e.what());
	}
}

VDBSequence VDBLoader::loadSequence(const std::vector<std::string>& paths, const std::string& gridName) const {
	VDBSequence sequence;
	sequence.reserve(paths.size());
	for (const auto& p : paths) {
		sequence.emplace_back(loadFrame(p, gridName));
	}
	logger::info("Successfully loaded {} frames.", sequence.size());
	return sequence;
}