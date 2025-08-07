/*
 * Copyright (c) 2025, Enzo Crema
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See the LICENSE file in the project root for full license text.
 */
#pragma once

#include <openvdb/openvdb.h>
#include <openvdb/tools/GridTransformer.h>

#include <algorithm> // Required for std::count
#include <cassert>
#include <map>
#include <set>
#include <string> // Required for std::string
#include <sstream> // Required for std::stringstream
#include <utility>
#include <vector>

#include "VDBStreamReader.hpp"


template <typename T>
struct array_view {
	const T* ptr = nullptr;
	size_t count = 0;

	array_view() = default;

	array_view(const T* p, const size_t c) : ptr(p), count(c) {
	}

	template <size_t N>
	explicit array_view(const T (&arr)[N]) : ptr(arr), count(N) {
	}

	[[nodiscard]] const float* data() const { return ptr; }

	[[nodiscard]] float* data() {
		return const_cast<float*>(ptr);
	}

	[[nodiscard]] size_t size() const { return count; }
};

/**
 * @brief Describes a single temporal run of consecutive frames for compression.
 */
struct RunDescriptor {
	int startFrame;
	int numFrames;

	[[nodiscard]] int endFrame() const { return startFrame + numFrames - 1; }
};

/**
 * @brief Represents the presence of a leaf within a specific run.
 *        Each leaf can have multiple runs across the sequence.
 */
struct LeafRun {
	RunDescriptor run;
	std::vector<bool> presenceMask; // Mask within this specific run
};

/**
 * @brief A leaf-centric database where each leaf has its own optimal
 *        temporal grouping based on its actual presence pattern.
 */
class GOPLayout {
public:
	int targetRunSize = 0;
	std::map<openvdb::Coord, std::vector<LeafRun>> leafRuns;

	/**
	 * @brief Checks if the layout is empty.
	 */
	[[nodiscard]] bool empty() const { return leafRuns.empty(); }

	/**
	 * @brief Gets the total number of unique leaves across the entire sequence.
	 */
	[[nodiscard]] size_t totalUniqueLeaves() const { return leafRuns.size(); }

	/**
	 * @brief Helper function to format a boolean vector into a 0/1 string.
	 */
	static std::string vectorBoolToString(const std::vector<bool>& mask) {
		std::string s;
		s.reserve(mask.size());
		for (bool bit : mask) {
			s += (bit ? '1' : '0');
		}
		return s;
	}

	/**
	 * @brief Generates a string summary of the layout.
	 */
	[[nodiscard]] std::string toString(size_t max_leaves_to_print = 5) const {
		if (empty()) {
			return "Empty GOP Layout";
		}

		std::stringstream ss;

		// --- Summary Section ---
		size_t total_runs = 0;
		size_t total_leaf_instances = 0;
		for (const auto& [origin, runs] : leafRuns) {
			total_runs += runs.size();
			for (const auto& run : runs) {
				total_leaf_instances += std::count(run.presenceMask.begin(), run.presenceMask.end(), true);
			}
		}

		ss << "GOP Layout Summary:\n";
		ss << "  - Target Run Size: " << targetRunSize << "\n";
		ss << "  - Total Unique Leaves: " << totalUniqueLeaves() << "\n";
		ss << "  - Total Runs: " << total_runs << "\n";
		ss << "  - Total Leaf Instances (sum of presences): " << total_leaf_instances << "\n";

		if (max_leaves_to_print == 0 || leafRuns.empty()) {
			return ss.str();
		}

		// --- Detailed Timeline Section ---
		ss << "\n--- Leaf Runs (showing first " << std::min(max_leaves_to_print, leafRuns.size()) << ") ---\n";

		size_t count = 0;
		for (const auto& [origin, runs] : leafRuns) {
			if (count >= max_leaves_to_print) {
				break;
			}

			ss << "Leaf at " << origin << ":\n";
			for (size_t run_idx = 0; run_idx < runs.size(); ++run_idx) {
				const auto& leaf_run = runs[run_idx];
				ss << "  Run " << run_idx << " (Frames " << leaf_run.run.startFrame
					<< "-" << leaf_run.run.endFrame() << "):\n";
				ss << "    Presence: " << vectorBoolToString(leaf_run.presenceMask) << "\n";
			}
			ss << "\n";
			count++;
		}

		return ss.str();
	}

	// Compatibility methods for existing compressor code
	struct GOPDescriptor {
		int startFrame;
		int numFrames;
	};

	struct LeafGOPMasks {
		std::vector<bool> presenceMask;
	};

	// Convert leaf-centric runs to GOP-centric format for compression
	std::vector<GOPDescriptor> gops;
	std::map<openvdb::Coord, std::vector<LeafGOPMasks>> timelines;

	void buildCompatibilityView() {
		// Clear existing compatibility data
		gops.clear();
		timelines.clear();

		// Collect all unique runs
		std::set<std::pair<int, int>> unique_runs; // (startFrame, numFrames)
		for (const auto& [origin, runs] : leafRuns) {
			for (const auto& leaf_run : runs) {
				unique_runs.insert({leaf_run.run.startFrame, leaf_run.run.numFrames});
			}
		}

		// Convert to GOP descriptors
		for (const auto& [start, num] : unique_runs) {
			gops.push_back({start, num});
		}

		// Sort GOPs by start frame
		std::sort(gops.begin(), gops.end(),
		          [](const GOPDescriptor& a, const GOPDescriptor& b) {
			          return a.startFrame < b.startFrame;
		          });

		// Build timelines in GOP-centric format
		for (const auto& [origin, runs] : leafRuns) {
			std::vector<LeafGOPMasks>& timeline = timelines[origin];
			timeline.resize(gops.size());

			for (const auto& leaf_run : runs) {
				// Find the GOP index for this run
				auto it = std::find_if(gops.begin(), gops.end(),
				                       [&](const GOPDescriptor& gop) {
					                       return gop.startFrame == leaf_run.run.startFrame &&
					                              gop.numFrames == leaf_run.run.numFrames;
				                       });

				if (it != gops.end()) {
					size_t gop_idx = std::distance(gops.begin(), it);
					timeline[gop_idx].presenceMask = leaf_run.presenceMask;
				}
			}
		}
	}
};

class GOPAnalyzer {
public:
	/**
	 * @brief Analyzes a VDB sequence and creates leaf-specific temporal runs
	 *        that maximize compression efficiency for each individual leaf.
	 *
	 * @param seq The input VDB sequence.
	 * @param targetRunSize The target number of frames per run.
	 * @param minRunSize Minimum run size (default: 4).
	 * @param maxRunSize Maximum run size (default: targetRunSize * 2).
	 * @return A GOPLayout object with optimized leaf-specific runs.
	 */
	[[nodiscard]] static GOPLayout analyze(const VDBSequence& seq, int targetRunSize = 16, int minRunSize = 4, int maxRunSize = -1);

private:
	/**
	 * @brief Find optimal runs for a single leaf based on its presence pattern.
	 */
	static std::vector<LeafRun> findOptimalRuns(const std::vector<bool>& presence, int targetRunSize, int minRunSize, int maxRunSize);
};

// ====================================================================
// DATA EXTRACTION STRUCTURES (GOP-CENTRIC PAYLOAD)
// ====================================================================

// One dense 8³ leaf buffer.
struct LeafBuffer {
	array_view<float> buffer;
};

/**
 * @brief Holds the actual voxel data for one leaf across all frames it was
 *        present within a single run.
 */
struct LeafDataSeries {
	openvdb::Coord origin;
	std::vector<LeafBuffer> buffers; // Data only for present frames.
};

/**
 * @brief Contains all the extracted leaf data for a single run.
 */
struct GOPData {
	int startFrame;
	int numFrames;
	std::vector<LeafDataSeries> leafSeries;
};


// ====================================================================
// IMPLEMENTATIONS
// ====================================================================

inline std::vector<LeafRun> GOPAnalyzer::findOptimalRuns(const std::vector<bool>& presence, int targetRunSize, int minRunSize,
                                                         int maxRunSize) {
	std::vector<LeafRun> runs;
	const int numFrames = static_cast<int>(presence.size());

	int i = 0;
	while (i < numFrames) {
		// Skip frames where leaf is not present
		while (i < numFrames && !presence[i]) {
			i++;
		}

		if (i >= numFrames) break;

		// Found start of a consecutive run
		int runStart = i;
		int runEnd = i;

		// Extend run as long as leaf is present and we haven't exceeded maxRunSize
		while (runEnd < numFrames && presence[runEnd] && (runEnd - runStart + 1) <= maxRunSize) {
			runEnd++;
		}
		runEnd--; // runEnd is now the last frame where leaf is present

		int runLength = runEnd - runStart + 1;

		// If the run is too long, split it into smaller chunks
		if (runLength > maxRunSize) {
			int currentStart = runStart;
			while (currentStart <= runEnd) {
				int chunkEnd = std::min(currentStart + maxRunSize - 1, runEnd);
				int chunkLength = chunkEnd - currentStart + 1;

				if (chunkLength >= minRunSize) {
					LeafRun run;
					run.run.startFrame = currentStart;
					run.run.numFrames = chunkLength;
					run.presenceMask.resize(chunkLength, true); // All frames in chunk are present
					runs.push_back(run);
				}

				currentStart = chunkEnd + 1;
			}
		} else if (runLength >= minRunSize) {
			// Run is acceptable size
			LeafRun run;
			run.run.startFrame = runStart;
			run.run.numFrames = runLength;
			run.presenceMask.resize(runLength, true); // All frames in run are present
			runs.push_back(run);
		}

		i = runEnd + 1;
	}

	return runs;
}

inline GOPLayout GOPAnalyzer::analyze(const VDBSequence& seq, const int targetRunSize, const int minRunSize, int maxRunSize) {
	assert(targetRunSize > 0 && "Target run size must be positive.");
	assert(minRunSize > 0 && "Minimum run size must be positive.");

	if (maxRunSize <= 0) {
		maxRunSize = targetRunSize * 2;
	}
	assert(maxRunSize >= minRunSize && "Maximum run size must be >= minimum run size.");

	GOPLayout layout;
	layout.targetRunSize = targetRunSize;

	if (seq.empty()) {
		return layout;
	}

	const int numFrames = static_cast<int>(seq.size());

	// 1. Collect presence information for each leaf across all frames
	std::map<openvdb::Coord, std::vector<bool>> leaf_presence;

	for (int t = 0; t < numFrames; ++t) {
		const auto& [grid] = seq[t];
		if (!grid) continue;

		std::set<openvdb::Coord> frame_leaves;
		for (auto leafIt = grid->tree().cbeginLeaf(); leafIt; ++leafIt) {
			frame_leaves.insert(leafIt->origin());
		}

		// Update presence for all leaves we've seen so far
		for (auto& [origin, presence] : leaf_presence) {
			if (presence.size() <= static_cast<size_t>(t)) {
				presence.resize(t + 1, false);
			}
			presence[t] = frame_leaves.count(origin) > 0;
		}

		// Add new leaves
		for (const auto& origin : frame_leaves) {
			if (leaf_presence.find(origin) == leaf_presence.end()) {
				std::vector<bool> presence(t + 1, false);
				presence[t] = true;
				leaf_presence[origin] = presence;
			}
		}
	}

	// Ensure all presence vectors are the same length
	for (auto& [origin, presence] : leaf_presence) {
		presence.resize(numFrames, false);
	}

	// 2. Find optimal runs for each leaf independently
	for (const auto& [origin, presence] : leaf_presence) {
		std::vector<LeafRun> runs = findOptimalRuns(presence, targetRunSize, minRunSize, maxRunSize);
		if (!runs.empty()) {
			layout.leafRuns[origin] = runs;
		}
	}

	// 3. Build compatibility view for existing compressor code
	layout.buildCompatibilityView();

	return layout;
}