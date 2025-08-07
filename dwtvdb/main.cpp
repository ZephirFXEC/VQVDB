/*
 * Copyright (c) 2025, Enzo Crema
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See the LICENSE file in the project root for full license text.
 */

#include <cxxopts.hpp>
#include <iostream>

#include "LeafActivityMask.hpp"
#include "Logger.hpp"
#include "Compressor.hpp"

int main(int argc, char** argv) {
	openvdb::initialize();

	cxxopts::Options cli("DWTVDB", "OpenVDB sequence compressor");

	cli.add_options()
		("c,compress", "Compress mode", cxxopts::value<bool>())("d,decompress", "Decompress mode", cxxopts::value<bool>())
		("v,verbose", "Verbose / debug logging", cxxopts::value<bool>()->default_value("false"))
		("i,input", "Input pattern / file", cxxopts::value<std::string>())
		("o,output", "Output file / dir", cxxopts::value<std::string>())
		("s,start", "Start frame", cxxopts::value<int>()->default_value("0"))
		("e,end", "End frame", cxxopts::value<int>())
		("b,block", "Block size", cxxopts::value<int>()->default_value("8"))
		("g,grid", "Grid name", cxxopts::value<std::string>()->default_value("density"))
		("q,qstep", "Quantization step", cxxopts::value<float>()->default_value("0.5"))
		("gop_size", "GOP Size", cxxopts::value<int>()->default_value("8"))


		("h,help", "Show help");

	auto args = cli.parse(argc, argv);

	// --- Initialise logging --------------------------------------------------
	logger::init(args["verbose"].as<bool>());
	logger::info("DWTVDB started");

	if (args.count("help") || argc == 1) {
		std::cout << cli.help() << '\n';
	}

	DCTCompressor::Params opt{};
	opt.qstep = args["q"].as<float>();


	DCTCompressor comp(opt);

	if (args["compress"].as<bool>()) {
		int start = args["start"].as<int>();
		int end = args["end"].as<int>();
		std::string pattern = args["input"].as<std::string>();

		std::vector<std::string> paths;
		for (int f = start; f <= end; ++f) {
			char buf[512];
			std::snprintf(buf, sizeof(buf), pattern.c_str(), f);
			paths.emplace_back(buf);
		}

		VDBLoader loader;
		VDBSequence seq = loader.loadSequence(paths, args["grid"].as<std::string>());
		logger::debug("Loaded {} frames from {} files", seq.size(), paths.size());

		std::chrono::high_resolution_clock::time_point startTime = std::chrono::high_resolution_clock::now();
		GOPLayout layout = GOPAnalyzer::analyze(seq, args["gop_size"].as<int>());
		logger::debug("GOP layout: {}", layout.toString());

		std::chrono::high_resolution_clock::time_point endTime = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
		logger::info("GOP analysis completed in {} ms", duration.count());

		comp.compress(seq, layout, args["output"].as<std::string>());
	} else if (args["decompress"].as<bool>()) {
		std::string inFile = args["input"].as<std::string>();
		std::string outDir = args["output"].as<std::string>();

		if (outDir.empty()) {
			logger::error("Output directory must be specified for decompression.");
			return 1;
		}

		std::chrono::high_resolution_clock::time_point startTime = std::chrono::high_resolution_clock::now();
		VDBSequence out_seq = comp.decompress(inFile);
		std::chrono::high_resolution_clock::time_point endTime = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
		logger::info("Decompression completed in {} ms", duration.count());

		if (out_seq.empty()) {
			logger::error("Decompression resulted in an empty sequence. Cannot save to VDB file.");
			logger::info("This is expected if the original VDBs were empty or the compressed file was invalid.");
			return 1;
		} else {
			logger::info("Decompression successful. Reconstructed {} frames.", out_seq.size());
		}

		logger::info("--- WRITING RECONSTRUCTED SEQUENCE TO DISK ---");
		VDBWriter writer;
		writer.saveSequence(out_seq, outDir);
		logger::info("Successfully saved reconstructed sequence to '{}'", outDir);
	} else {
		std::cerr << "Specify --compress or --decompress\n";
		return 1;
	}
}