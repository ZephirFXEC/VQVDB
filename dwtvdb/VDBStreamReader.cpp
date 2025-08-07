#include "VDBStreamReader.hpp"

#include <openvdb/io/File.h>

#include <stdexcept>

#include "logger.hpp"
#include <filesystem>
#include <iomanip>
#include <sstream>

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

void VDBWriter::saveSequence(const VDBSequence& seq, const std::string& outputPath) {
	if (seq.empty()) {
		logger::warn("VDBWriter::saveSequence called with an empty sequence. Nothing to write.");
		return;
	}

	// 1. Ensure the output directory exists. Create it if it doesn't.
	try {
		std::filesystem::create_directories(outputPath);
	} catch (const std::filesystem::filesystem_error& e) {
		logger::error("Failed to create output directory '{}': {}", outputPath, e.what());
		return;
	}

	logger::info("Saving {} frames to directory '{}'", seq.size(), outputPath);

	// 2. Loop through each frame in the sequence.
	for (size_t i = 0; i < seq.size(); ++i) {
		const auto& grid = seq[i].grid;

		// Skip frames that have no grid data.
		if (!grid || grid->empty()) {
			logger::warn("Frame {} is empty, skipping.", i);
			continue;
		}

		// 3. Generate a unique, padded filename for this frame.
		std::stringstream ss;
		ss << "frame_" << std::setw(4) << std::setfill('0') << i << ".vdb";
		std::string filename = ss.str();

		// 4. Combine the directory path and the filename.
		std::filesystem::path full_path = std::filesystem::path(outputPath) / filename;

		// 5. Write this single grid to its own file.
		try {
			openvdb::io::File file(full_path.string());

			// To write a single grid, it's often best to put it in a container
			// to avoid the template overload ambiguity we saw before.
			openvdb::GridPtrVec grids_to_write;
			grids_to_write.push_back(grid);

			file.write(grids_to_write);
			file.close();
		} catch (const openvdb::IoError& e) {
			logger::error("Failed to write frame {} to '{}': {}", i, full_path.string(), e.what());
		}
	}

	logger::info("Finished writing VDB sequence.");
}


openvdb::FloatGrid::Ptr VDBStreamReader::readFrame(
	const int index) const {
	if (index < 0 || index >= numFrames()) {
		std::ostringstream oss;
		oss << "readFrame: index " << index << " out of range [0, "
			<< (numFrames() - 1) << "]";
		throw std::out_of_range(oss.str());
	}

	const std::string& path = paths_[static_cast<size_t>(index)];
	openvdb::io::File file(path);
	try {
		file.open();

		if (!file.hasGrid(gridName_)) {
			file.close();
			std::ostringstream oss;
			oss << "Grid '" << gridName_ << "' not found in file: "
				<< path;
			throw std::runtime_error(oss.str());
		}

		openvdb::GridBase::Ptr base = file.readGrid(gridName_);
		file.close();

		auto grid = openvdb::gridPtrCast<openvdb::FloatGrid>(base);
		if (!grid) {
			std::ostringstream oss;
			oss << "Grid '" << gridName_ << "' in '" << path
				<< "' is not a FloatGrid.";
			throw std::runtime_error(oss.str());
		}
		return grid;
	} catch (const openvdb::IoError& e) {
		// Ensure file is closed on exceptions
		try {
			file.close();
		} catch (...) {
		}
		std::ostringstream oss;
		oss << "OpenVDB I/O error loading file '" << path
			<< "': " << e.what();
		throw std::runtime_error(oss.str());
	}
}