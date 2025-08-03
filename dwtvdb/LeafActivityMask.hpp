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

#include <map>
#include <set>
#include <utility>
#include <vector>

#include "VDBStreamReader.hpp"


template <typename T>
struct array_view {
	const T* ptr = nullptr;
	size_t count = 0;

	array_view() = default;
	array_view(const T* p, const size_t c) : ptr(p), count(c) {}

	template <size_t N>
	explicit array_view(const T (&arr)[N]) : ptr(arr), count(N) {}

	const T* data() const { return ptr; }
	size_t size() const { return count; }
};

class ActivityAnalyzer {
   public:
	struct Window {
		int start = -1;
		int end = -1;
	};

	[[nodiscard]] static std::map<openvdb::Coord, std::vector<Window>> computeWindows(const VDBSequence& seq,
	                                                                                  int inactivityThresholdFrames = 3);
};

// One dense 8³ leaf buffer at one time step.
struct LeafSlice {
	array_view<float> buffer;
	int frameIndex;
};

// All slices that belong to one activity window of one leaf.
struct LeafWindowSeries {
	openvdb::Coord origin;
	ActivityAnalyzer::Window window;
	std::vector<LeafSlice> slices;
};

class DataSeriesExtractor {
   public:
	[[nodiscard]] static std::vector<LeafWindowSeries> extract(
	    const VDBSequence& seq, const std::map<openvdb::Coord, std::vector<ActivityAnalyzer::Window>>& windows);
};


inline std::vector<LeafWindowSeries> DataSeriesExtractor::extract(
    const VDBSequence& seq, const std::map<openvdb::Coord, std::vector<ActivityAnalyzer::Window>>& windows) {
	std::vector<LeafWindowSeries> result;

	size_t total_windows = 0;
	for (const auto& pair : windows) {
		total_windows += pair.second.size();
	}
	result.reserve(total_windows);

	// This buffer is read-only and can be safely shared across threads if you ever parallelize.
	static constexpr float zeros[512] = {};
	const array_view<float> zero_buffer_view(zeros);  // Create a view once

	for (const auto& [origin, winList] : windows) {
		for (const auto& w : winList) {
			LeafWindowSeries series;
			series.origin = origin;
			series.window = w;

			if (w.start <= w.end) {
				series.slices.reserve(w.end - w.start + 1);
			}

			for (int t = w.start; t <= w.end; ++t) {
				const auto& [blocks, grid] = seq[t];
				const openvdb::tree::ValueAccessor<const openvdb::FloatTree> acc(grid->tree());

				if (const auto leaf = acc.probeLeaf(origin)) {
					series.slices.push_back({array_view<float>(leaf->buffer().data(), 512), t});
				} else {
					series.slices.push_back({zero_buffer_view, t});
				}
			}
			result.emplace_back(std::move(series));
		}
	}
	return result;
}

inline std::map<openvdb::Coord, std::vector<ActivityAnalyzer::Window>> ActivityAnalyzer::computeWindows(
    const VDBSequence& seq, const int inactivityThresholdFrames) {
	struct LeafState {
		std::vector<Window> windows;
		int currentStart = -1;
		int inactiveCount = 0;
	};

	std::map<openvdb::Coord, LeafState> table;

	for (int t = 0; t < static_cast<int>(seq.size()); ++t) {
		const auto& frame = seq[t];
		std::set<openvdb::Coord> seen;

		// 1. iterate over every leaf present in this frame
		openvdb::tree::ValueAccessor<const openvdb::FloatTree> acc(frame.grid->tree());
		for (auto leafIt = acc.tree().cbeginLeaf(); leafIt; ++leafIt) {
			const auto& leaf = *leafIt;
			openvdb::Coord key = leaf.origin();
			seen.insert(key);

			LeafState& st = table[key];
			if (!leaf.isInactive()) {
				st.inactiveCount = 0;
				if (st.currentStart == -1) st.currentStart = t;
			} else {
				++st.inactiveCount;
				if (st.currentStart != -1 && st.inactiveCount >= inactivityThresholdFrames) {
					st.windows.push_back({st.currentStart, t - st.inactiveCount});
					st.currentStart = -1;
				}
			}
		}

		// 2. handle leaves that disappeared
		for (auto& [key, st] : table) {
			if (seen.count(key) > 0) continue;  // still active
			++st.inactiveCount;
			if (st.currentStart != -1 && st.inactiveCount >= inactivityThresholdFrames) {
				st.windows.push_back({st.currentStart, t - st.inactiveCount});
				st.currentStart = -1;
			}
		}
	}

	// 3. close any open window at end of sequence
	const int last = static_cast<int>(seq.size()) - 1;
	std::map<openvdb::Coord, std::vector<Window>> result;
	for (auto& [key, st] : table) {
		if (st.currentStart != -1) {
			st.windows.push_back({st.currentStart, last});
		}
		if (!st.windows.empty()) {
			result.emplace(key, std::move(st.windows));
		}
	}
	return result;
}
