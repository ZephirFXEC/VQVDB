//
// Created by zphrfx on 07/08/2025.
//

#ifndef DWTVDB_COMPRESSIONSTATS_HPP
#define DWTVDB_COMPRESSIONSTATS_HPP

#include <atomic>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <vector>

struct CompressionStats {
	std::atomic<uint64_t> totalBricks{0};
	std::atomic<uint64_t> skippedBricks{0};
	std::atomic<uint64_t> totalCoeffs{0};
	std::atomic<uint64_t> totalNonZero{0};
	std::atomic<uint64_t> deadzoneZeroed{0};

	// Adaptive scale tracking
	std::mutex scaleMutex;
	std::vector<float> adaptScales;

	void recordBrick(bool skipped, float adaptScale, uint64_t coeffCount, uint64_t nnz, uint64_t deadzoneCount) {
		totalBricks++;
		if (skipped) skippedBricks++;
		totalCoeffs += coeffCount;
		totalNonZero += nnz;
		deadzoneZeroed += deadzoneCount;

		{
			std::lock_guard<std::mutex> lock(scaleMutex);
			adaptScales.push_back(adaptScale);
		}
	}

	void printSummary() {
		uint64_t bricks = totalBricks.load();
		uint64_t skipped = skippedBricks.load();
		uint64_t coeffs = totalCoeffs.load();
		uint64_t nnz = totalNonZero.load();
		uint64_t dz = deadzoneZeroed.load();

		std::cout << "\n=== Compression Stats ===\n";
		std::cout << "Total bricks: " << bricks << "\n";
		std::cout << "Skipped bricks: " << skipped << " (" << (100.0 * skipped / std::max<uint64_t>(1, bricks)) << "%)\n";
		std::cout << "Total coeffs: " << coeffs << "\n";
		std::cout << "Non-zero coeffs: " << nnz << " (" << (100.0 * nnz / std::max<uint64_t>(1, coeffs)) << "%)\n";
		std::cout << "Zeroed by deadzone: " << dz << " (" << (100.0 * dz / std::max<uint64_t>(1, coeffs)) << "%)\n";

		// Adaptive scale stats
		{
			std::lock_guard<std::mutex> lock(scaleMutex);
			if (!adaptScales.empty()) {
				float minS = adaptScales[0], maxS = adaptScales[0], sumS = 0.0f;
				for (float s : adaptScales) {
					if (s < minS) minS = s;
					if (s > maxS) maxS = s;
					sumS += s;
				}
				float avgS = sumS / adaptScales.size();
				std::cout << "Adaptive scale: min=" << minS << " avg=" << avgS << " max=" << maxS << "\n";
			}
		}
		std::cout << "=========================\n";
	}
};

#endif  // DWTVDB_COMPRESSIONSTATS_HPP