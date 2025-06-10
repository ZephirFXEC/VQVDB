//
// Created by zphrfx on 04/06/2025.
//

#include "../../LeafExtractor.hpp"

int main(int argc, char** argv) {
	if (argc < 3) {
		std::cout << "Usage: " << argv[0] << " <input_folder> <output_parent_folder> [gridName]\n";
		return 1;
	}

	namespace fs = std::filesystem;
	fs::path inputDir = argv[1];
	fs::path outputParent = argv[2];
	std::string gridName = (argc > 3) ? argv[3] : "";

	if (!fs::is_directory(inputDir)) {
		std::cerr << "Error: input path is not a directory\n";
		return 1;
	}
	if (!fs::exists(outputParent)) {
		fs::create_directories(outputParent);
	}

	for (auto& entry : fs::directory_iterator(inputDir)) {
		if (entry.path().extension() == ".vdb") {
			auto base = entry.path().stem().string();
			fs::path outFile = outputParent / (base + ".npy");

			if (!extractLeafData<openvdb::FloatGrid>(entry.path().string(), gridName, outFile.string())) {
				std::cerr << "Failed processing " << entry.path() << "\n";
			}
		}
	}

	return 0;
}
