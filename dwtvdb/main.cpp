/*
 * Copyright (c) 2025, Enzo Crema
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See the LICENSE file in the project root for full license text.
 */

#include "WaveletCompressor.hpp"
#include <cxxopts.hpp>
#include <filesystem>
#include <iostream>
#include "Logger.hpp"

int main(int argc, char** argv)
{
    cxxopts::Options cli("DWTVDB", "OpenVDB sequence compressor (C++17)");

    cli.add_options()
        ("c,compress",   "Compress mode", cxxopts::value<bool>())
        ("d,decompress", "Decompress mode", cxxopts::value<bool>())
		("v,verbose",    "Verbose / debug logging", cxxopts::value<bool>()->default_value("false"))
        ("i,input",      "Input pattern / file",  cxxopts::value<std::string>())
        ("o,output",     "Output file / dir",     cxxopts::value<std::string>())
        ("s,start",      "Start frame",           cxxopts::value<int>()->default_value("0"))
        ("e,end",        "End frame",             cxxopts::value<int>())
        ("b,block",      "Block size",            cxxopts::value<int>()->default_value("8"))
        ("l,levels",     "Wavelet levels",        cxxopts::value<int>()->default_value("2"))
        ("r,rank",       "CP rank",               cxxopts::value<int>()->default_value("32"))
        ("g,grid",       "Grid name",             cxxopts::value<std::string>()->default_value("density"))
        ("h,help",       "Show help");

    auto args = cli.parse(argc, argv);

	// --- Initialise logging --------------------------------------------------
	logger::init(args["verbose"].as<bool>());
	logger::info("DWTVDB started");

    if (args.count("help") || argc==1) {
        std::cout << cli.help() << '\n';
        return 0;
    }

    WaveletCompressor::Options opt;
    opt.block   = args["block"].as<int>();
    opt.levels  = args["levels"].as<int>();
    opt.rank    = args["rank"].as<int>();
    opt.gridName= args["grid"].as<std::string>();

    WaveletCompressor comp(opt);

    if (args["compress"].as<bool>()) {
        int start = args["start"].as<int>();
        int end   = args["end"].as<int>();
        std::string pattern = args["input"].as<std::string>();

        std::vector<std::string> paths;
        for (int f=start; f<=end; ++f) {
            char buf[512];
            std::snprintf(buf,sizeof(buf), pattern.c_str(), f);
            paths.emplace_back(buf);
        }
        comp.compress(paths, args["output"].as<std::string>());

    } else if (args["decompress"].as<bool>()) {
        comp.decompress(args["input"].as<std::string>(),
                        args["output"].as<std::string>());
    } else {
        std::cerr << "Specify --compress or --decompress\n";
        return 1;
    }
}
