/*
 * Copyright (c) 2025, Enzo Crema
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See the LICENSE file in the project root for full license text.
 */

#pragma once

#include <openvdb/openvdb.h>

#include <string>
#include <vector>

// --- This is the ONLY part we need for the initial loading ---

/**
 * @brief Represents a single frame in the sequence.
 * For the initial analysis, it only needs to hold a pointer to the grid structure.
 */
struct VDBFrame {
	openvdb::FloatGrid::Ptr grid;
};

/**
 * @brief A sequence of VDB frames.
 */
class VDBSequence : public std::vector<VDBFrame> {
   public:
	using std::vector<VDBFrame>::vector;  // Inherit constructors
};

// --- This loader will now be much simpler and more efficient ---

class VDBLoader {
   public:
	/**
	 * @brief Loads only the grid structure for a single frame.
	 */
	[[nodiscard]] VDBFrame loadFrame(const std::string& path, const std::string& gridName) const;

	/**
	 * @brief Loads an entire sequence of grid structures.
	 */
	[[nodiscard]] VDBSequence loadSequence(const std::vector<std::string>& paths, const std::string& gridName) const;
};

class VDBWriter {
   public:
	/**
	 * @brief Saves a sequence of VDB frames to disk.
	 *
	 * @param seq The sequence of frames to save.
	 * @param outputPath The path where the sequence will be saved.
	 */
	static void saveSequence(const VDBSequence& seq, const std::string& outputPath);
};


class VDBStreamReader {
   public:
	VDBStreamReader(std::vector<std::string> paths, std::string gridName) : paths_(std::move(paths)), gridName_(std::move(gridName)) {}

	// Number of frames available
	int numFrames() const { return static_cast<int>(paths_.size()); }

	// Thread-safe random-access frame read. Throws on error.
	openvdb::FloatGrid::Ptr readFrame(int index) const;

	const std::string& gridName() const { return gridName_; }

	const std::vector<std::string>& paths() const { return paths_; }

   private:
	std::vector<std::string> paths_;
	std::string gridName_;
};