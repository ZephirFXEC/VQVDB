/*
 * Copyright (c) 2025, Enzo Crema
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See the LICENSE file in the project root for full license text.
 */

#pragma once
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

namespace logger {
inline void init(bool verbose = false) {
	static bool once = [] {
		auto sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
		sink->set_pattern("[%H:%M:%S] [%^%l%$] %v");

		auto logger = std::make_shared<spdlog::logger>("DWTVDB", sink);
		spdlog::register_logger(logger);
		spdlog::set_default_logger(logger);
		return true;
	}();

	spdlog::set_level(verbose ? spdlog::level::debug : spdlog::level::info);
}

// Convenience wrappers (so you can switch backend later with 0 churn)
template <typename... T>
inline void debug(T&&... t) {
	spdlog::debug(std::forward<T>(t)...);
}

template <typename... T>
inline void info(T&&... t) {
	spdlog::info(std::forward<T>(t)...);
}

template <typename... T>
inline void warn(T&&... t) {
	spdlog::warn(std::forward<T>(t)...);
}

template <typename... T>
inline void error(T&&... t) {
	spdlog::error(std::forward<T>(t)...);
}
}  // namespace logger