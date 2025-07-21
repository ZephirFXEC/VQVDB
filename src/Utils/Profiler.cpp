/*
 * Copyright (c) 2025, Enzo Crema
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See the LICENSE file in the project root for full license text.
 */

#include "Profiler.hpp"

PerformanceProfiler& PerformanceProfiler::getInstance() {
	static PerformanceProfiler instance;
	return instance;
}