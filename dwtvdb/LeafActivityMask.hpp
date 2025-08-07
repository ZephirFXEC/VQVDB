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

#include <cassert>
#include <map>
#include <utility>
#include <vector>

#include "VDBStreamReader.hpp"
#include "Logger.hpp"

// Intrinsic for popcount for performance, with a fallback.
#if defined(__GNUC__) || defined(__clang__)
#define popcount(x) __builtin_popcountll(x)
#else
// A simple fallback for other compilers (e.g., MSVC, which has __popcnt64)
// This can be adapted for higher performance on other platforms.
#define popcount(x) __popcnt64(x)
#endif


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
 * @brief Describes a continuous sequence of frames where a single leaf is present.
 */
struct LeafRun {
	int startFrame;
	int numFrames;
};


/**
 * @brief A normalized, leaf-centric database of the entire sequence's layout,
 *        decomposed into continuous runs of presence for each leaf.
 *
 * This structure maps each unique leaf's coordinate to a vector of its
 * continuous runs. Each run is a unit of compression.
 */
class GOPLayout {
public:
	// A map from each unique leaf's origin to its timeline of continuous runs.
	std::map<openvdb::Coord, std::vector<LeafRun>> leaf_runs;

	/**
	 * @brief Checks if the layout is empty.
	 */
	[[nodiscard]] bool empty() const { return leaf_runs.empty(); }

	/**
	 * @brief Gets the total number of unique leaves across the entire sequence.
	 */
	[[nodiscard]] size_t totalUniqueLeaves() const { return leaf_runs.size(); }


	/**
	 * @brief Generates a string summary of the layout.
	 */
	[[nodiscard]] std::string toString(size_t max_leaves_to_print = 5) const {
		if (empty()) {
			return "Empty GOP Layout (no leaves found)";
		}

		std::stringstream ss;
		size_t total_runs = 0;
		for (const auto& pair : leaf_runs) {
			total_runs += pair.second.size();
		}

		ss << "Leaf Run Layout Summary:\n";
		ss << "  - Total Unique Leaves: " << totalUniqueLeaves() << "\n";
		ss << "  - Total Compression Runs: " << total_runs << "\n";

		if (max_leaves_to_print == 0 || leaf_runs.empty()) {
			return ss.str();
		}

		ss << "\n--- Leaf Run Details (showing first " << std::min(max_leaves_to_print, leaf_runs.size()) << ") ---\n";

		size_t count = 0;
		for (const auto& [origin, runs] : leaf_runs) {
			if (count >= max_leaves_to_print) break;
			ss << "Leaf at " << origin << ":\n";
			for (const auto& run : runs) {
				ss << "  - Run: " << run.numFrames << " frames starting at frame " << run.startFrame << "\n";
			}
			ss << "\n";
			count++;
		}
		return ss.str();
	}
};

class GOPAnalyzer {
public:
	/**
	 * @brief Analyzes a VDB sequence and creates a layout based on the continuous
	 *        presence of each individual leaf.
	 *
	 * This method finds every continuous "run" of frames for each leaf. Each run
	 * will become an independent unit for compression, which is the correct way
	 * to handle temporal transforms on sparse, dynamic data.
	 *
	 * @param seq The input VDB sequence.
	 * @param gopSize (Unused) Kept for API compatibility. The analysis is now data-driven.
	 * @return A GOPLayout object containing the leaf run metadata.
	 */
	[[nodiscard]] static GOPLayout analyze(const VDBSequence& seq, int gopSize = 16);
};


// ====================================================================
// IMPLEMENTATIONS
// ====================================================================

inline GOPLayout GOPAnalyzer::analyze(const VDBSequence& seq, [[maybe_unused]] const int gopSize) {
	GOPLayout layout;
	if (seq.empty()) {
		return layout;
	}
	const int numFrames = static_cast<int>(seq.size());

	// 1. First pass: Collect a full presence timeline for every unique leaf.
	std::map<openvdb::Coord, std::vector<bool>> presence_timelines;
	for (int t = 0; t < numFrames; ++t) {
		const auto& [grid] = seq[t];
		for (auto leafIt = grid->tree().cbeginLeaf(); leafIt; ++leafIt) {
			const openvdb::Coord origin = leafIt->origin();
			// Ensure the timeline vector exists and is the correct size.
			if (presence_timelines.find(origin) == presence_timelines.end()) {
				presence_timelines[origin].resize(numFrames, false);
			}
			presence_timelines[origin][t] = true;
		}
	}

	// 2. Second pass: For each leaf's timeline, find its continuous runs.
	for (auto const& [origin, timeline] : presence_timelines) {
		int current_run_start = -1;
		for (int t = 0; t < numFrames; ++t) {
			if (timeline[t] && current_run_start == -1) {
				// Start of a new run
				current_run_start = t;
			} else if (!timeline[t] && current_run_start != -1) {
				// End of the current run
				layout.leaf_runs[origin].push_back({current_run_start, t - current_run_start});
				current_run_start = -1;
			}
		}
		// If a run was active until the very last frame
		if (current_run_start != -1) {
			layout.leaf_runs[origin].push_back({current_run_start, numFrames - current_run_start});
		}
	}

	return layout;
}