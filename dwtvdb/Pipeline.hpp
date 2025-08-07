//
// Created by zphrfx on 07/08/2025.
//

#ifndef DWTVDB_PIPELINE_HPP
#define DWTVDB_PIPELINE_HPP

#include <atomic>
#include <map>
#include <memory>
#include <thread>
#include <unordered_map>
#include <vector>

#include <openvdb/openvdb.h>

#include "Compressor.hpp"
#include "FileFormat.hpp"
#include "Logger.hpp"
#include "ThreadSafeQueue.hpp"
#include "VDBStreamReader.hpp"

struct FrameItem {
	int index = -1;
	std::shared_ptr<openvdb::FloatGrid> grid;
};

struct GOPJob {
	int startFrame = 0;
	int numFrames = 0;
	std::vector<openvdb::FloatGrid::Ptr> frames;

	// presence masks per leaf in this GOP
	std::unordered_map<openvdb::Coord, std::vector<bool>> presence;
};

struct EncodedRunOut {
	RunMetadata meta;
	std::vector<char> blob;
	std::vector<uint8_t> masks;
};

class CompressorPipeline {
public:
	struct Options {
		int targetRunSize = 16;
		int loaderThreads = 2;
		int analyzerThreads = 1; // kept 1 for ordering correctness
		int compressorThreads = std::max(1u,
		                                 std::thread::hardware_concurrency());
		size_t frameQueueCapacity = 64;
		size_t gopQueueCapacity = 8;
		size_t outputQueueCapacity = 64;
	};

	CompressorPipeline(DCTCompressor::Params dctPar,
	                   Options opts)
		: comp_(dctPar),
		  opts_(opts),
		  frameQ_(opts.frameQueueCapacity),
		  gopQ_(opts.gopQueueCapacity),
		  outQ_(opts.outputQueueCapacity) {
	}

	void run(VDBStreamReader& reader,
	         const std::string& outputPath);

private:
	DCTCompressor comp_;
	Options opts_;

	ThreadSafeQueue<FrameItem> frameQ_;
	ThreadSafeQueue<GOPJob> gopQ_;
	ThreadSafeQueue<EncodedRunOut> outQ_;

	void stageLoad(VDBStreamReader& reader, int threadId,
	               std::atomic<int>& nextIndex);

	void stageAnalyze(int totalFrames);

	void stageCompressWorker();

	void stageWrite(const std::string& outputPath);
};
#endif //DWTVDB_PIPELINE_HPP