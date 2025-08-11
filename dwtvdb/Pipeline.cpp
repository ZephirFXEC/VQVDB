#include "Pipeline.hpp"

#include <algorithm>
#include <fstream>
#include <map>
#include <stdexcept>

#include "CompressionStats.hpp"
#include "LeafActivityMask.hpp"
#include "Logger.hpp"

CompressionStats g_stats;

void CompressorPipeline::run(VDBStreamReader& reader, const std::string& outputPath) {
	const int totalFrames = reader.numFrames();
	if (totalFrames <= 0) {
		logger::warn("No frames to compress. Writing empty file.");
		FileHeader header = comp_.makeHeader();
		header.index_offset = sizeof(FileHeader);
		header.brick_count = 0;
		std::ofstream f(outputPath, std::ios::binary);
		f.write(reinterpret_cast<const char*>(&header), sizeof(header));
		return;
	}

	logger::info(
	    "Pipeline start: {} frames, target GOP size={}, loaders={}, "
	    "compressors={}",
	    totalFrames, opts_.targetRunSize, opts_.loaderThreads, opts_.compressorThreads);

	std::thread writerTh(&CompressorPipeline::stageWrite, this, outputPath);

	std::vector<std::thread> compThreads;
	compThreads.reserve(static_cast<size_t>(opts_.compressorThreads));
	for (int i = 0; i < opts_.compressorThreads; ++i) {
		compThreads.emplace_back(&CompressorPipeline::stageCompressWorker, this);
	}

	std::thread analyzerTh(&CompressorPipeline::stageAnalyze, this, totalFrames);

	std::atomic<int> nextIndex{0};
	std::vector<std::thread> loaderThreads;
	loaderThreads.reserve(static_cast<size_t>(opts_.loaderThreads));
	for (int i = 0; i < opts_.loaderThreads; ++i) {
		loaderThreads.emplace_back(&CompressorPipeline::stageLoad, this, std::ref(reader), i, std::ref(nextIndex));
	}

	for (auto& th : loaderThreads) th.join();
	frameQ_.close();

	analyzerTh.join();
	gopQ_.close();

	for (auto& th : compThreads) th.join();
	outQ_.close();

	writerTh.join();

	logger::info("Pipeline finished: wrote file {}", outputPath);

	g_stats.printSummary();
}

void CompressorPipeline::stageLoad(VDBStreamReader& reader, int /*threadId*/, std::atomic<int>& nextIndex) {
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

void CompressorPipeline::stageAnalyze(const int totalFramesInput) {
	// --- Phase A: ingest + reorder all frames into a contiguous vector ---
	std::map<int, openvdb::FloatGrid::Ptr> reorder;
	int nextEmit = 0;

	std::vector<openvdb::FloatGrid::Ptr> allFrames;
	allFrames.resize(static_cast<size_t>(totalFramesInput));

	FrameItem item;
	while (frameQ_.pop(item)) {
		reorder[item.index] = std::move(item.grid);
		for (;;) {
			auto it = reorder.find(nextEmit);
			if (it == reorder.end()) break;
			allFrames[size_t(nextEmit)] = std::move(it->second);
			reorder.erase(it);
			++nextEmit;
		}
	}
	// Drain leftovers (won't fill gaps in the middle)
	for (;;) {
		auto it = reorder.find(nextEmit);
		if (it == reorder.end()) break;
		allFrames[size_t(nextEmit)] = std::move(it->second);
		reorder.erase(it);
		++nextEmit;
	}

	if (nextEmit != totalFramesInput) {
		logger::warn("Analyzer got {} frames out of {}. Trimming to available frames.", nextEmit, totalFramesInput);
	}

	// IMPORTANT: from here on, only use the frames we actually have.
	allFrames.resize(size_t(nextEmit));
	const int totalFrames = nextEmit;
	if (totalFrames == 0) {
		gopQ_.close();
		return;
	}

	// --- Phase B: build sequence over available frames ---
	VDBSequence seq;
	seq.reserve(size_t(totalFrames));
	for (int t = 0; t < totalFrames; ++t) {
		if (!allFrames[size_t(t)]) {
			logger::error("Null frame at t={}, aborting.", t);
			gopQ_.close();
			return;
		}
		seq.push_back({allFrames[size_t(t)]});
	}

	auto layout = GOPAnalyzer::analyze(seq, opts_.targetRunSize);
	if (layout.gops.empty()) {
		gopQ_.close();
		return;
	}

	logger::debug(layout.toString());

	/*// --- Phase C: stats + autotune (NULL-SAFE) ---
	const int sampleFrames = std::min(totalFrames, 16);
	const int sampleLeaves = 4096;

	auto stats = DCTCompressor::DatasetStats::estimateStats(allFrames, sampleFrames, sampleLeaves);  // see safe impl below
	const float k = opts_.compression;                                                               // [0,1]
	auto tuned = DCTCompressor::paramsFromKnob(k, stats);
	comp_.setParams(tuned);*/

	logger::debug("Dataset stats: {}", comp_.getParams());

	// --- Phase D: enqueue GOP jobs ---
	for (size_t g = 0; g < layout.gops.size(); ++g) {
		const auto& gdesc = layout.gops[g];

		GOPJob job;
		job.startFrame = gdesc.startFrame;
		job.numFrames = gdesc.numFrames;

		job.frames.reserve(size_t(gdesc.numFrames));
		for (int t = 0; t < gdesc.numFrames; ++t) {
			auto& grid = allFrames[size_t(gdesc.startFrame + t)];
			if (!grid) {
				logger::error("Null grid in GOP build.");
				gopQ_.close();
				return;
			}
			job.frames.push_back(grid);
		}

		for (const auto& [origin, masksVec] : layout.timelines) {
			if (g < masksVec.size()) {
				const auto& m = masksVec[g].presenceMask;
				if (!m.empty()) job.presence[origin] = m;
			}
		}

		if (!gopQ_.push(std::move(job))) return;
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
				while (cursor < T && !mask[static_cast<size_t>(cursor)]) ++cursor;
				if (cursor >= T) break;

				const int runStart = cursor;
				int len = 0;
				while (cursor < T && mask[static_cast<size_t>(cursor)]) {
					++len;
					++cursor;
				}
				if (len <= 0) continue;

				DCTCompressor::RunInput in;
				in.origin = origin;
				in.start_frame = job.startFrame + runStart;
				in.frames.reserve(static_cast<size_t>(len));
				for (int t = 0; t < len; ++t) {
					in.frames.push_back(job.frames[static_cast<size_t>(runStart + t)]);
				}

				try {
					auto encoded = comp_.encodeRun(in);

					EncodedRunOut out;
					out.origin = encoded.origin;
					out.start_frame = encoded.start_frame;
					out.num_frames = encoded.num_frames;

					out.v3.skip = encoded.v3.skip;
					out.v3.adaptScale = encoded.v3.adaptScale;
					out.v3.nnz = encoded.v3.nnz;
					out.v3.coeffMask = std::move(encoded.v3.coeffMask);
					out.v3.coeffValues = std::move(encoded.v3.coeffValues);

					out.masks = std::move(encoded.valueMasks);

					if (!outQ_.push(std::move(out))) {
						return;  // queue closed
					}
				} catch (const std::exception& e) {
					logger::error("Compression failed for leaf ({}, {}, {}) at frame {} len {}: {}", origin.x(), origin.y(), origin.z(),
					              in.start_frame, len, e.what());
					// Skip this run and continue with others
				}
			}
		}
	}
}

void CompressorPipeline::stageWrite(const std::string& outputPath) {
	// Open for read/write to patch header at the end.
	std::fstream file(outputPath, std::ios::binary | std::ios::in | std::ios::out | std::ios::trunc);
	if (!file) {
		throw std::runtime_error("Failed to open output file");
	}

	// Write placeholder header
	FileHeader header = comp_.makeHeader();
	file.write(reinterpret_cast<const char*>(&header), sizeof(header));

	auto tellp64 = [&]() -> uint64_t { return static_cast<uint64_t>(file.tellp()); };

	auto pad_to_alignment = [&](uint64_t alignment) {
		const uint64_t pos = tellp64();
		const uint64_t pad = (alignment - (pos % alignment)) % alignment;
		if (pad) {
			static const uint8_t zeros[64] = {0};
			uint64_t remaining = pad;
			while (remaining > 0) {
				const uint64_t chunk = std::min<uint64_t>(remaining, sizeof(zeros));
				file.write(reinterpret_cast<const char*>(zeros), static_cast<std::streamsize>(chunk));
				remaining -= chunk;
			}
		}
	};

	// Collect index entries in memory (lightweight).
	std::vector<BrickIndexEntry> index;
	index.reserve(1 << 20);  // optional pre-reserve

	size_t runsWritten = 0;
	EncodedRunOut out;
	while (outQ_.pop(out)) {
		// 64-byte align each brick payload for coalesced GPU reads
		pad_to_alignment(64);

		BrickIndexEntry ent{};
		ent.origin = out.origin;
		ent.start_frame = out.start_frame;
		ent.num_frames = out.num_frames;
		ent.flags = 0;
		if (out.v3.skip) ent.flags |= BRICK_FLAG_SKIP;
		ent.adapt_scale = out.v3.adaptScale;

		const uint32_t nnz = out.v3.nnz;

		const uint8_t* coeffMask = out.v3.coeffMask.data();
		const size_t coeffMaskBytes = out.v3.coeffMask.size();

		const int16_t* coeffValues = out.v3.coeffValues.data();
		const size_t coeffValuesBytes = out.v3.coeffValues.size() * sizeof(int16_t);

		const uint8_t* valueMasks = out.masks.data();
		const size_t valueMaskBytes = out.masks.size();

		ent.coeff_mask_bytes = static_cast<uint32_t>(coeffMaskBytes);
		ent.coeff_values_bytes = static_cast<uint32_t>(coeffValuesBytes);
		ent.value_mask_bytes = static_cast<uint32_t>(valueMaskBytes);

		ent.data_offset = tellp64();

		// Payload layout:
		// [nnz:u32][coeffMask][coeffValues][valueMasks]
		file.write(reinterpret_cast<const char*>(&nnz), sizeof(uint32_t));

		if (coeffMaskBytes > 0) {
			file.write(reinterpret_cast<const char*>(coeffMask), static_cast<std::streamsize>(coeffMaskBytes));
		}

		if (coeffValuesBytes > 0) {
			file.write(reinterpret_cast<const char*>(coeffValues), static_cast<std::streamsize>(coeffValuesBytes));
		}

		if (valueMaskBytes > 0) {
			file.write(reinterpret_cast<const char*>(valueMasks), static_cast<std::streamsize>(valueMaskBytes));
		}

		index.push_back(ent);
		++runsWritten;
		if (runsWritten % 500 == 0) {
			logger::debug("Writer: {} runs written...", runsWritten);
		}
	}

	// Append index table, 64-byte aligned
	pad_to_alignment(64);
	const uint64_t indexOffset = tellp64();

	if (!index.empty()) {
		file.write(reinterpret_cast<const char*>(index.data()), static_cast<std::streamsize>(index.size() * sizeof(BrickIndexEntry)));
	}

	// Patch header with index metadata
	header.index_offset = indexOffset;
	header.brick_count = static_cast<uint32_t>(index.size());

	file.seekp(0, std::ios::beg);
	file.write(reinterpret_cast<const char*>(&header), sizeof(header));
	file.flush();

	logger::info("Writer completed: {} runs written. Index @ {} bytes.", runsWritten, indexOffset);
}