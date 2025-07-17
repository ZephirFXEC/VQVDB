#pragma once

#include <chrono>
#include <iostream>
#include <string>
#include <unordered_map>

/**
 * @class PerformanceProfiler
 * @brief Simple performance profiling utility for GPU/CPU operations
 */
class PerformanceProfiler {
   public:
	static PerformanceProfiler& getInstance();

	void startTimer(const std::string& name) { startTimes_[name] = std::chrono::high_resolution_clock::now(); }

	void endTimer(const std::string& name) {
		auto endTime = std::chrono::high_resolution_clock::now();
		auto it = startTimes_.find(name);
		if (it != startTimes_.end()) {
			auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - it->second).count();
			totalTimes_[name] += duration;
			callCounts_[name]++;
			startTimes_.erase(it);
		}
	}

	void printReport() const {
		std::cout << "\n=== Performance Report ===\n";
		for (const auto& [name, totalTime] : totalTimes_) {
			auto count = callCounts_.at(name);
			auto avgTime = totalTime / count;
			std::cout << name << ": " << count << " calls, " << totalTime << " μs total, " << avgTime << " μs average\n";
		}
		std::cout << "========================\n\n";
	}

	void reset() {
		startTimes_.clear();
		totalTimes_.clear();
		callCounts_.clear();
	}

   private:
	std::unordered_map<std::string, std::chrono::high_resolution_clock::time_point> startTimes_;
	std::unordered_map<std::string, long long> totalTimes_;
	std::unordered_map<std::string, int> callCounts_;
};

// RAII timer for automatic timing
class ScopedTimer {
   public:
	explicit ScopedTimer(const std::string& name) : name_(name) { PerformanceProfiler::getInstance().startTimer(name_); }

	~ScopedTimer() { PerformanceProfiler::getInstance().endTimer(name_); }

   private:
	std::string name_;
};

#define PROFILE_SCOPE(name) ScopedTimer _timer(name)
#define PROFILE_START(name) PerformanceProfiler::getInstance().startTimer(name)
#define PROFILE_END(name) PerformanceProfiler::getInstance().endTimer(name)