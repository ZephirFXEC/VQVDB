#include <cxxopts.hpp>
#include <filesystem>
#include <iostream>

#include <openvdb/openvdb.h>

#include "Compressor.hpp"
#include "Logger.hpp"
#include "Pipeline.hpp"
#include "VDBStreamReader.hpp"

int main(int argc, char** argv) {
	openvdb::initialize();

	cxxopts::Options cli("DWTVDB", "OpenVDB sequence compressor");

	cli.add_options()
		("c,compress", "Compress mode", cxxopts::value<bool>()->default_value("false"))
		("d,decompress", "Decompress mode", cxxopts::value<bool>()->default_value("false"))
		("v,verbose", "Verbose / debug logging",
		 cxxopts::value<bool>()->default_value("false"))
		("i,input", "Input pattern (compress) or file (decompress)",
		 cxxopts::value<std::string>())
		("o,output", "Output file (compress) or directory (decompress)",
		 cxxopts::value<std::string>())
		("s,start", "Start frame (compress)",
		 cxxopts::value<int>()->default_value("0"))
		("e,end", "End frame (compress)", cxxopts::value<int>())
		("g,grid", "Grid name",
		 cxxopts::value<std::string>()->default_value("density"))
		("q,qstep", "Quantization step",
		 cxxopts::value<float>()->default_value("0.5"))
		("gop_size", "GOP size (target run size in pipeline)",
		 cxxopts::value<int>()->default_value("16"))
		("zstd", "Zstd compression level (1..22)",
		 cxxopts::value<int>()->default_value("3"))
		("loaders", "Number of loader threads",
		 cxxopts::value<int>()->default_value("2"))
		("compressors", "Number of compressor threads (0=auto)",
		 cxxopts::value<int>()->default_value("0"))
		("h,help", "Show help");

	auto args = cli.parse(argc, argv);

	// --- Initialise logging ----------------------------------------------------
	logger::init(args["verbose"].as<bool>());
	logger::info("DWTVDB started");

	if (args.count("help") || argc == 1) {
		std::cout << cli.help() << '\n';
		return 0;
	}

	const bool doCompress = args["compress"].as<bool>();
	const bool doDecompress = args["decompress"].as<bool>();

	if (doCompress == doDecompress) {
		std::cerr << "Specify exactly one of --compress or --decompress\n";
		return 1;
	}

	try {
		if (doCompress) {
			// Required args
			if (!args.count("input") || !args.count("output") ||
			    !args.count("end")) {
				throw std::runtime_error(
					"Compress mode requires --input, --output, --start, --end");
			}

			const std::string pattern = args["input"].as<std::string>();
			const std::string outputFile = args["output"].as<std::string>();
			const int start = args["start"].as<int>();
			const int end = args["end"].as<int>();
			const std::string gridName = args["grid"].as<std::string>();
			const int gopSize = args["gop_size"].as<int>();
			const float qstep = args["q"].as<float>();
			const int zstdLevel = args["zstd"].as<int>();
			int loaderThreads = args["loaders"].as<int>();
			int compressorThreads = args["compressors"].as<int>();

			if (start > end) {
				throw std::runtime_error("--start must be <= --end");
			}
			if (gopSize <= 0) {
				throw std::runtime_error("--gop_size must be > 0");
			}
			if (loaderThreads <= 0) loaderThreads = 1;
			if (compressorThreads <= 0) {
				const unsigned hc = std::thread::hardware_concurrency();
				compressorThreads = static_cast<int>(hc ? hc : 1);
			}

			// Expand input pattern into paths
			std::vector<std::string> paths;
			paths.reserve(static_cast<size_t>(end - start + 1));
			for (int f = start; f <= end; ++f) {
				char buf[1024];
				std::snprintf(buf, sizeof(buf), pattern.c_str(), f);
				paths.emplace_back(buf);
			}

			// Ensure output directory exists
			try {
				const std::filesystem::path outp(outputFile);
				if (outp.has_parent_path()) {
					std::filesystem::create_directories(outp.parent_path());
				}
			} catch (const std::filesystem::filesystem_error& e) {
				logger::warn("Failed to create output parent directory: {}", e.what());
			}

			// Build reader and pipeline
			VDBStreamReader reader(paths, gridName);

			DCTCompressor::Params dpar;
			dpar.qstep = qstep;
			dpar.zstdLevel = zstdLevel;

			CompressorPipeline::Options popt;
			popt.targetRunSize = gopSize;
			popt.loaderThreads = loaderThreads;
			popt.compressorThreads = compressorThreads;

			CompressorPipeline pipeline(dpar, popt);

			const auto t0 = std::chrono::high_resolution_clock::now();
			pipeline.run(reader, outputFile);
			const auto t1 = std::chrono::high_resolution_clock::now();
			const auto ms =
				std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0)
				.count();
			logger::info("Compression completed in {} ms", ms);
		} else if (doDecompress) {
			// Required args
			if (!args.count("input") || !args.count("output")) {
				throw std::runtime_error(
					"Decompress mode requires --input and --output");
			}
			const std::string inFile = args["input"].as<std::string>();
			const std::string outDir = args["output"].as<std::string>();

			if (outDir.empty()) {
				throw std::runtime_error(
					"Output directory must be specified for decompression.");
			}

			// Ensure output directory exists
			try {
				std::filesystem::create_directories(outDir);
			} catch (const std::filesystem::filesystem_error& e) {
				logger::warn("Failed to create output directory '{}': {}", outDir,
				             e.what());
			}

			// qstep is taken from file header during decode; we keep CLI qstep
			// only to construct the object, but it will be overridden.
			DCTCompressor::Params dpar;
			dpar.qstep = args["q"].as<float>();
			dpar.zstdLevel = args["zstd"].as<int>();
			DCTCompressor comp(dpar);

			const auto t0 = std::chrono::high_resolution_clock::now();
			VDBSequence out_seq = comp.decompress(inFile);
			const auto t1 = std::chrono::high_resolution_clock::now();
			const auto ms =
				std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0)
				.count();
			logger::info("Decompression completed in {} ms", ms);

			if (out_seq.empty()) {
				logger::error("Decompression resulted in an empty sequence.");
				logger::info(
					"This can happen if the source was empty or the file was invalid.");
				return 1;
			}

			logger::info("--- WRITING RECONSTRUCTED SEQUENCE TO DISK ---");
			VDBWriter::saveSequence(out_seq, outDir);
			logger::info("Successfully saved reconstructed sequence to '{}'", outDir);
		}
	} catch (const std::exception& e) {
		logger::error("Fatal error: {}", e.what());
		return 1;
	}

	return 0;
}