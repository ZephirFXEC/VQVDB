#include "Pipeline.hpp"

#include <algorithm>
#include <fstream>
#include <map>
#include <stdexcept>

#include "Logger.hpp"

void CompressorPipeline::run(VDBStreamReader& reader,
                             const std::string& outputPath) {
	const int totalFrames = reader.numFrames();
	if (totalFrames <= 0) {
		logger::warn("No frames to compress. Writing empty file.");
		std::ofstream f(outputPath, std::ios::binary);
		if (!f)
			throw std::runtime_error(
				"Failed to open output file for writing");
		const FileHeader header = comp_.makeHeader();
		f.write(reinterpret_cast<const char*>(&header),
		        sizeof(header));
		return;
	}

	logger::info(
		"Pipeline start: {} frames, target GOP size={}, loaders={}, "
		"compressors={}",
		totalFrames, opts_.targetRunSize, opts_.loaderThreads,
		opts_.compressorThreads);

	std::thread writerTh(&CompressorPipeline::stageWrite, this,
	                     outputPath);

	std::vector<std::thread> compThreads;
	compThreads.reserve(
		static_cast<size_t>(opts_.compressorThreads));
	for (int i = 0; i < opts_.compressorThreads; ++i) {
		compThreads.emplace_back(
			&CompressorPipeline::stageCompressWorker, this);
	}

	std::thread analyzerTh(&CompressorPipeline::stageAnalyze, this,
	                       totalFrames);

	std::atomic<int> nextIndex{0};
	std::vector<std::thread> loaderThreads;
	loaderThreads.reserve(
		static_cast<size_t>(opts_.loaderThreads));
	for (int i = 0; i < opts_.loaderThreads; ++i) {
		loaderThreads.emplace_back(
			&CompressorPipeline::stageLoad, this, std::ref(reader),
			i, std::ref(nextIndex));
	}

	for (auto& th : loaderThreads) th.join();
	frameQ_.close();

	analyzerTh.join();
	gopQ_.close();

	for (auto& th : compThreads) th.join();
	outQ_.close();

	writerTh.join();

	logger::info("Pipeline finished: wrote file {}", outputPath);
}

void CompressorPipeline::stageLoad(VDBStreamReader& reader,
                                   int /*threadId*/,
                                   std::atomic<int>& nextIndex) {
	const int total = reader.numFrames();
	while (true) {
		const int idx = nextIndex.fetch_add(1);
		if (idx >= total) break;

		try {
			auto grid = reader.readFrame(idx);
			FrameItem item{idx, grid};
			if (!frameQ_.push(std::move(item))) {
				break;
			}
		} catch (const std::exception& e) {
			logger::error("Loader failed for frame {}: {}", idx, e.what());
			break;
		}
	}
}

void CompressorPipeline::stageAnalyze(const int totalFrames) {
	// Reorder buffer to ensure we emit frames in sequence.
	std::map<int, openvdb::FloatGrid::Ptr> reorder;
	int nextEmit = 0;

	std::vector<openvdb::FloatGrid::Ptr> gopFrames;
	gopFrames.reserve(static_cast<size_t>(opts_.targetRunSize));
	int currentGopStart = 0;

	auto flushGOP = [&](int startIdx,
	                    std::vector<std::shared_ptr<openvdb::FloatGrid>>&
	                    frames) {
		if (frames.empty()) return;
		const int T = static_cast<int>(frames.size());

		GOPJob job;
		job.startFrame = startIdx;
		job.numFrames = T;
		job.frames = frames;

		// Build presence masks for all leaves across the GOP
		for (int t = 0; t < T; ++t) {
			const auto& grid = job.frames[t];
			if (!grid) continue;

			for (auto it = grid->tree().cbeginLeaf(); it; ++it) {
				const auto origin = it->origin();
				auto& mask = job.presence[origin];
				if (mask.empty()) mask.assign(static_cast<size_t>(T), false);
				mask[static_cast<size_t>(t)] = true;
			}
		}

		if (!gopQ_.push(GOPJob{std::move(job)})) {
			// Queue closed; drop remaining work
			return;
		}
		frames.clear();
	};

	FrameItem item;
	while (frameQ_.pop(item)) {
		reorder[item.index] = std::move(item.grid);

		// Drain any newly contiguous frames
		while (true) {
			auto it = reorder.find(nextEmit);
			if (it == reorder.end()) break;

			if (gopFrames.empty()) currentGopStart = nextEmit;
			gopFrames.push_back(std::move(it->second));
			reorder.erase(it);
			++nextEmit;

			if (static_cast<int>(gopFrames.size()) >=
			    opts_.targetRunSize) {
				flushGOP(currentGopStart, gopFrames);
			}
		}
	}

	// After frameQ_ is closed, flush remaining in-order frames
	while (true) {
		auto it = reorder.find(nextEmit);
		if (it == reorder.end()) break;
		if (gopFrames.empty()) currentGopStart = nextEmit;
		gopFrames.push_back(std::move(it->second));
		reorder.erase(it);
		++nextEmit;

		if (static_cast<int>(gopFrames.size()) >=
		    opts_.targetRunSize) {
			flushGOP(currentGopStart, gopFrames);
		}
	}

	// Flush final partial GOP
	if (!gopFrames.empty()) {
		flushGOP(currentGopStart, gopFrames);
	}

	if (nextEmit != totalFrames) {
		logger::warn(
			"Analyzer ended with nextEmit={} but totalFrames={}. "
			"Some frames may be missing.",
			nextEmit, totalFrames);
	}
}

void CompressorPipeline::stageCompressWorker() {
	GOPJob job;
	while (gopQ_.pop(job)) {
		const int T = job.numFrames;
		if (T <= 0) continue;

		for (const auto& kv : job.presence) {
			const auto& origin = kv.first;
			const auto& mask = kv.second;

			int cursor = 0;
			while (cursor < T) {
				while (cursor < T && !mask[static_cast<size_t>(cursor)])
					++cursor;
				if (cursor >= T) break;

				const int runStart = cursor;
				int len = 0;
				while (cursor < T &&
				       mask[static_cast<size_t>(cursor)]) {
					++len;
					++cursor;
				}
				if (len <= 0) continue;

				DCTCompressor::RunInput in;
				in.origin = origin;
				in.start_frame = job.startFrame + runStart;
				in.frames.reserve(static_cast<size_t>(len));
				for (int t = 0; t < len; ++t) {
					in.frames.push_back(
						job.frames[static_cast<size_t>(runStart + t)]);
				}

				try {
					auto encoded = comp_.encodeRun(in);

					EncodedRunOut out;
					out.meta = encoded.meta;
					out.blob = std::move(encoded.blob);
					out.masks = std::move(encoded.valueMasks);

					if (!outQ_.push(std::move(out))) {
						return; // queue closed
					}
				} catch (const std::exception& e) {
					logger::error(
						"Compression failed for leaf ({}, {}, {}) at "
						"frame {} len {}: {}",
						origin.x(), origin.y(), origin.z(),
						in.start_frame, len, e.what());
					// Skip this run and continue with others
				}
			}
		}
	}
}

void CompressorPipeline::stageWrite(
	const std::string& outputPath) {
	std::ofstream file(outputPath, std::ios::binary);
	if (!file) {
		throw std::runtime_error("Failed to open output file");
	}

	const FileHeader header = comp_.makeHeader();
	file.write(reinterpret_cast<const char*>(&header),
	           sizeof(header));

	size_t runsWritten = 0;
	EncodedRunOut out;
	while (outQ_.pop(out)) {
		// Write metadata
		file.write(reinterpret_cast<const char*>(&out.meta),
		           sizeof(out.meta));

		// Write payload blob and masks
		if (!out.blob.empty())
			file.write(out.blob.data(),
			           static_cast<std::streamsize>(out.blob.size()));
		if (!out.masks.empty())
			file.write(
				reinterpret_cast<const char*>(out.masks.data()),
				static_cast<std::streamsize>(out.masks.size()));

		++runsWritten;
		if (runsWritten % 500 == 0) {
			logger::debug("Writer: {} runs written...", runsWritten);
		}
	}

	file.flush();
	logger::info("Writer completed: {} runs written.", runsWritten);
}