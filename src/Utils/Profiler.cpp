#include "Profiler.hpp"

PerformanceProfiler& PerformanceProfiler::getInstance() {
    static PerformanceProfiler instance;
    return instance;
}