//
// Created by zphrfx on 04/06/2025.
//

#include "LeafExtractor.hpp"

int main(int argc, char** argv) {
	if (argc < 3) {
		std::cout << "Usage: " << argv[0] << " <input.vdb> <output.npy> [gridName]\n";
		return 1;
	}

	std::string inputPath = argv[1];
	std::string outputPath = argv[2];
	std::string gridName = (argc > 3) ? argv[3] : "";

	return extractLeafData<openvdb::FloatGrid>(inputPath, gridName, outputPath) ? 0 : 1;
}
