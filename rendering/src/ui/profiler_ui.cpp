#include "ui/profiler_ui.hpp"

#include <imgui.h>

#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <format>
#include <functional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "ui/ui.hpp"

namespace profiler_ui {

namespace {

constexpr float kFlameRowHeight = 18.0f;
constexpr size_t kFlameAverageWindowFrames = 240;

template <typename T>
void hashCombine(size_t& seed, const T& value) noexcept {
	seed ^= std::hash<T>{}(value) + 0x9e3779b9u + (seed << 6u) + (seed >> 2u);
}

struct ScopeEventBaseKey {
	std::string_view name{};
	profiler::ScopeType type{profiler::ScopeType::CPU};
	int depth{0};
	profiler::TransferDirection transferDirection{profiler::TransferDirection::CPUToCPU};
	std::string_view sourceFile{};
	int sourceLine{0};

	[[nodiscard]] bool operator==(const ScopeEventBaseKey& other) const noexcept = default;
};

struct ScopeEventBaseKeyHash {
	[[nodiscard]] size_t operator()(const ScopeEventBaseKey& key) const noexcept {
		size_t seed = 0;
		hashCombine(seed, key.name);
		hashCombine(seed, static_cast<int>(key.type));
		hashCombine(seed, key.depth);
		hashCombine(seed, static_cast<int>(key.transferDirection));
		hashCombine(seed, key.sourceFile);
		hashCombine(seed, key.sourceLine);
		return seed;
	}
};

struct ScopeEventKey {
	ScopeEventBaseKey base{};
	size_t occurrence{0};

	[[nodiscard]] bool operator==(const ScopeEventKey& other) const noexcept = default;
};

struct ScopeEventKeyHash {
	[[nodiscard]] size_t operator()(const ScopeEventKey& key) const noexcept {
		size_t seed = ScopeEventBaseKeyHash{}(key.base);
		hashCombine(seed, key.occurrence);
		return seed;
	}
};

struct FlameGraphAggregate {
	std::vector<profiler::ScopeEvent> events;
	double totalTimeMs{0.0};
	size_t frameCount{0};
};

[[nodiscard]] const char* directionLabel(profiler::TransferDirection direction) noexcept {
	switch (direction) {
		case profiler::TransferDirection::CPUToGPU:
			return "CPU -> GPU";
		case profiler::TransferDirection::GPUToCPU:
			return "GPU -> CPU";
		case profiler::TransferDirection::CPUToCPU:
			return "CPU only";
	}
	return "Unknown";
}

[[nodiscard]] std::string formatMiB(size_t bytes) {
	const double value = static_cast<double>(bytes) / (1024.0 * 1024.0);
	return std::format("{:.3f} MiB", value);
}

[[nodiscard]] ImU32 colorForScope(const profiler::ScopeEvent& event) {
	const size_t hashSeed = std::hash<std::string>{}(event.name);
	const float hue = static_cast<float>(hashSeed % 360) / 360.0f;
	float saturation = 0.65f;
	float value = 0.85f;

	if (event.type == profiler::ScopeType::GPU) {
		saturation = 0.80f;
		value = 0.88f;
	} else if (event.type == profiler::ScopeType::Transfer) {
		saturation = 0.70f;
		value = 0.92f;
	}

	return ImGui::ColorConvertFloat4ToU32(ImColor::HSV(hue, saturation, value));
}

[[nodiscard]] int maxDepth(const std::vector<profiler::ScopeEvent>& events) {
	int depth = 0;
	for (const profiler::ScopeEvent& event : events) {
		depth = std::max(depth, event.depth);
	}
	return depth;
}

template <typename Predicate>
[[nodiscard]] std::vector<const profiler::FrameProfile*> collectRecentFrames(const std::vector<const profiler::FrameProfile*>& frames,
                                                                              size_t maxCount, Predicate&& predicate) {
	std::vector<const profiler::FrameProfile*> selected;
	selected.reserve(std::min(maxCount, frames.size()));
	for (auto it = frames.rbegin(); it != frames.rend() && selected.size() < maxCount; ++it) {
		const profiler::FrameProfile* frame = *it;
		if (frame != nullptr && predicate(*frame)) {
			selected.push_back(frame);
		}
	}
	std::reverse(selected.begin(), selected.end());
	return selected;
}

template <typename EventGetter, typename TotalGetter>
[[nodiscard]] FlameGraphAggregate averageFlameGraph(const std::vector<const profiler::FrameProfile*>& frames, EventGetter&& eventGetter,
                                                    TotalGetter&& totalGetter) {
	FlameGraphAggregate aggregate;
	if (frames.empty()) {
		return aggregate;
	}

	struct EventAccum {
		profiler::ScopeEvent sample;
		double startMsSum{0.0};
		double endMsSum{0.0};
		double bytesSum{0.0};
		size_t samples{0};
	};

	std::unordered_map<ScopeEventKey, EventAccum, ScopeEventKeyHash> accumByKey;
	double totalTimeMsSum = 0.0;

	for (const profiler::FrameProfile* frame : frames) {
		if (frame == nullptr) continue;

		totalTimeMsSum += std::max(0.0, totalGetter(*frame));
		++aggregate.frameCount;

		std::unordered_map<ScopeEventBaseKey, size_t, ScopeEventBaseKeyHash> occurrenceByBase;
		const auto& events = eventGetter(*frame);
		for (const profiler::ScopeEvent& event : events) {
			if (event.durationMs() <= 0.0) continue;

			ScopeEventBaseKey baseKey;
			baseKey.name = event.name;
			baseKey.type = event.type;
			baseKey.depth = event.depth;
			baseKey.transferDirection = event.transferDirection;
			baseKey.sourceFile = (event.source.file != nullptr) ? std::string_view(event.source.file) : std::string_view{};
			baseKey.sourceLine = event.source.line;

			const size_t occurrence = occurrenceByBase[baseKey]++;
			ScopeEventKey eventKey;
			eventKey.base = baseKey;
			eventKey.occurrence = occurrence;

			auto [accumIt, inserted] = accumByKey.try_emplace(eventKey);
			EventAccum& accum = accumIt->second;
			if (inserted) {
				accum.sample = event;
			}

			accum.startMsSum += event.startMs;
			accum.endMsSum += event.endMs;
			accum.bytesSum += static_cast<double>(event.bytes);
			++accum.samples;
		}
	}

	if (aggregate.frameCount == 0) {
		return aggregate;
	}

	aggregate.totalTimeMs = totalTimeMsSum / static_cast<double>(aggregate.frameCount);
	aggregate.events.reserve(accumByKey.size());
	for (auto& [_, accum] : accumByKey) {
		if (accum.samples == 0) continue;

		profiler::ScopeEvent event = accum.sample;
		const double invSampleCount = 1.0 / static_cast<double>(accum.samples);
		event.startMs = accum.startMsSum * invSampleCount;
		event.endMs = accum.endMsSum * invSampleCount;
		if (event.endMs < event.startMs) {
			std::swap(event.startMs, event.endMs);
		}
		event.bytes = static_cast<size_t>(std::llround(accum.bytesSum * invSampleCount));
		aggregate.events.push_back(std::move(event));
	}

	std::sort(aggregate.events.begin(), aggregate.events.end(), [](const profiler::ScopeEvent& a, const profiler::ScopeEvent& b) {
		if (a.startMs != b.startMs) return a.startMs < b.startMs;
		if (a.depth != b.depth) return a.depth < b.depth;
		return a.name < b.name;
	});

	return aggregate;
}

void drawFlameGraph(const char* childId, const std::vector<profiler::ScopeEvent>& events, double totalTimeMs) {
	if (events.empty()) {
		ImGui::TextDisabled("No scope events to display.");
		return;
	}

	const int depth = maxDepth(events);
	const float graphHeight = (static_cast<float>(depth) + 1.0f) * kFlameRowHeight + 8.0f;

	if (!ImGui::BeginChild(childId, ImVec2(0.0f, graphHeight + 12.0f), true, ImGuiWindowFlags_HorizontalScrollbar)) {
		ImGui::EndChild();
		return;
	}

	const ImVec2 canvasPos = ImGui::GetCursorScreenPos();
	const ImVec2 avail = ImGui::GetContentRegionAvail();
	const float width = std::max(avail.x, 60.0f);
	const float height = std::max(graphHeight, 24.0f);
	const double safeTotalMs = std::max(totalTimeMs, 0.001);
	ImDrawList* drawList = ImGui::GetWindowDrawList();

	drawList->AddRectFilled(canvasPos, ImVec2(canvasPos.x + width, canvasPos.y + height), IM_COL32(14, 14, 18, 255));
	drawList->AddRect(canvasPos, ImVec2(canvasPos.x + width, canvasPos.y + height), IM_COL32(40, 40, 48, 255));

	for (int tick = 1; tick <= 5; ++tick) {
		const float x = canvasPos.x + (static_cast<float>(tick) / 6.0f) * width;
		drawList->AddLine(ImVec2(x, canvasPos.y), ImVec2(x, canvasPos.y + height), IM_COL32(45, 45, 50, 255), 1.0f);
	}

	const profiler::ScopeEvent* hoveredEvent = nullptr;
	for (const profiler::ScopeEvent& event : events) {
		const double duration = event.durationMs();
		if (duration <= 0.0) continue;

		float x0 = canvasPos.x + static_cast<float>((event.startMs / safeTotalMs) * width);
		float x1 = canvasPos.x + static_cast<float>((event.endMs / safeTotalMs) * width);
		if (x1 < x0 + 1.0f) x1 = x0 + 1.0f;

		const float y0 = canvasPos.y + static_cast<float>(event.depth) * kFlameRowHeight + 2.0f;
		const float y1 = y0 + kFlameRowHeight - 4.0f;
		const ImU32 color = colorForScope(event);

		drawList->AddRectFilled(ImVec2(x0, y0), ImVec2(x1, y1), color, 2.0f);
		drawList->AddRect(ImVec2(x0, y0), ImVec2(x1, y1), IM_COL32(18, 18, 18, 230), 2.0f);

		const float labelPadding = 4.0f;
		if ((x1 - x0) > 36.0f) {
			ImGui::PushClipRect(ImVec2(x0 + 1.0f, y0 + 1.0f), ImVec2(x1 - 1.0f, y1 - 1.0f), true);
			drawList->AddText(ImVec2(x0 + labelPadding, y0 + 1.0f), IM_COL32(20, 20, 20, 255), event.name.c_str());
			ImGui::PopClipRect();
		}

		if (ImGui::IsMouseHoveringRect(ImVec2(x0, y0), ImVec2(x1, y1))) {
			hoveredEvent = &event;
		}
	}

	if (hoveredEvent != nullptr) {
		ImGui::BeginTooltip();
		ImGui::TextUnformatted(hoveredEvent->name.c_str());
		ImGui::Separator();
		ImGui::Text("Duration: %.3f ms", hoveredEvent->durationMs());
		ImGui::Text("Start: %.3f ms", hoveredEvent->startMs);
		ImGui::Text("Depth: %d", hoveredEvent->depth);
		if (hoveredEvent->type == profiler::ScopeType::Transfer) {
			ImGui::Text("Direction: %s", directionLabel(hoveredEvent->transferDirection));
			ImGui::Text("Bytes: %zu", hoveredEvent->bytes);
		}
		ImGui::Text("Source: %s:%d", hoveredEvent->source.file, hoveredEvent->source.line);
		ImGui::EndTooltip();
	}

	ImGui::Dummy(ImVec2(width, height));
	ImGui::EndChild();
}

void renderHistoryPlots(const std::vector<const profiler::FrameProfile*>& frames) {
	if (frames.empty()) {
		ImGui::TextDisabled("No profiler frames captured yet.");
		return;
	}

	std::vector<float> cpuMs;
	std::vector<float> gpuMs;
	std::vector<float> transferMs;
	std::vector<float> upMiB;
	std::vector<float> downMiB;
	cpuMs.reserve(frames.size());
	gpuMs.reserve(frames.size());
	transferMs.reserve(frames.size());
	upMiB.reserve(frames.size());
	downMiB.reserve(frames.size());

	for (const profiler::FrameProfile* frame : frames) {
		if (frame == nullptr) continue;
		cpuMs.push_back(static_cast<float>(frame->cpuFrameMs));
		gpuMs.push_back(static_cast<float>(frame->gpuResolved ? frame->gpuFrameMs : 0.0));
		transferMs.push_back(static_cast<float>(frame->transferCpuMs));
		upMiB.push_back(static_cast<float>(static_cast<double>(frame->bytesCpuToGpu) / (1024.0 * 1024.0)));
		downMiB.push_back(static_cast<float>(static_cast<double>(frame->bytesGpuToCpu) / (1024.0 * 1024.0)));
	}

	const int count = static_cast<int>(frames.size());
	ImGui::PlotLines("CPU Frame (ms)", cpuMs.data(), count, 0, nullptr, 0.0f, FLT_MAX, ImVec2(0.0f, 60.0f));
	ImGui::PlotLines("GPU Frame (ms)", gpuMs.data(), count, 0, "0 = unresolved", 0.0f, FLT_MAX, ImVec2(0.0f, 60.0f));
	ImGui::PlotLines("Transfer CPU (ms)", transferMs.data(), count, 0, nullptr, 0.0f, FLT_MAX, ImVec2(0.0f, 45.0f));
	ImGui::PlotLines("CPU->GPU (MiB)", upMiB.data(), count, 0, nullptr, 0.0f, FLT_MAX, ImVec2(0.0f, 45.0f));
	ImGui::PlotLines("GPU->CPU (MiB)", downMiB.data(), count, 0, nullptr, 0.0f, FLT_MAX, ImVec2(0.0f, 45.0f));
}

void renderTransferTable(const profiler::FrameProfile& frame) {
	if (frame.transfers.empty()) {
		ImGui::TextDisabled("No transfer scopes in this frame.");
		return;
	}

	if (ImGui::BeginTable("transferTable", 4, ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp)) {
		ImGui::TableSetupColumn("Label");
		ImGui::TableSetupColumn("Direction");
		ImGui::TableSetupColumn("Bytes");
		ImGui::TableSetupColumn("CPU ms");
		ImGui::TableHeadersRow();

		const size_t start = frame.transfers.size() > 16 ? frame.transfers.size() - 16 : 0;
		for (size_t i = start; i < frame.transfers.size(); ++i) {
			const profiler::TransferEvent& event = frame.transfers[i];
			ImGui::TableNextRow();
			ImGui::TableSetColumnIndex(0);
			ImGui::TextUnformatted(event.name.c_str());
			ImGui::TableSetColumnIndex(1);
			ImGui::TextUnformatted(directionLabel(event.direction));
			ImGui::TableSetColumnIndex(2);
			ImGui::TextUnformatted(formatMiB(event.bytes).c_str());
			ImGui::TableSetColumnIndex(3);
			ImGui::Text("%.3f", event.cpuTimeMs);
		}

		ImGui::EndTable();
	}
}

}  // namespace

void renderProfilerTab(UIState& state) noexcept {
	auto& profiler = state.profiler;
	profiler.collectFinishedGpuQueries();

	const auto& history = profiler.frameHistory();
	std::vector<const profiler::FrameProfile*> closedFrames;
	closedFrames.reserve(history.size());
	for (const profiler::FrameProfile& frame : history) {
		if (frame.closed) {
			closedFrames.push_back(&frame);
		}
	}

	ImGui::Checkbox("Enable instrumentation", &state.profilerEnabled);
	ImGui::SameLine();
	ImGui::Checkbox("Gathering", &state.profilerGathering);
	ImGui::SameLine();
	if (ImGui::Button("Clear Session")) {
		profiler.clearSession();
		state.profilerSelectedClosedFrame = 0;
		return;
	}

	ImGui::TextDisabled("Closed frames: %zu | Total frames: %zu | Pending GPU queries: %zu", closedFrames.size(), history.size(),
	                    profiler.pendingGpuScopeCount());
	ImGui::TextDisabled("GPU timer query support: %s", profiler.gpuTimingSupported() ? "available" : "missing");

	if (closedFrames.empty()) {
		ImGui::SeparatorText("Frame Summary");
		ImGui::TextDisabled("No completed frames in session yet.");
		return;
	}

	ImGui::Checkbox("Follow latest completed frame", &state.profilerFollowLatest);
	if (state.profilerFollowLatest) {
		state.profilerSelectedClosedFrame = static_cast<int>(closedFrames.size()) - 1;
	}

	state.profilerSelectedClosedFrame = std::clamp(state.profilerSelectedClosedFrame, 0, static_cast<int>(closedFrames.size()) - 1);

	if (!state.profilerFollowLatest) {
		ImGui::SliderInt("Selected frame", &state.profilerSelectedClosedFrame, 0, static_cast<int>(closedFrames.size()) - 1);
	}

	const profiler::FrameProfile* selectedFrame = closedFrames[state.profilerSelectedClosedFrame];
	const profiler::FrameProfile* selectedGpuFrame = selectedFrame->gpuResolved ? selectedFrame : nullptr;

	ImGui::SeparatorText("Frame Summary");
	ImGui::Text("Frame #%llu", static_cast<unsigned long long>(selectedFrame->frameId));
	ImGui::Text("CPU frame: %.3f ms", selectedFrame->cpuFrameMs);
	if (selectedGpuFrame != nullptr) {
		ImGui::Text("GPU frame: %.3f ms", selectedGpuFrame->gpuFrameMs);
	} else {
		ImGui::Text("GPU frame: pending");
	}
	ImGui::Text("Transfers CPU: %.3f ms", selectedFrame->transferCpuMs);
	ImGui::Text("CPU->GPU: %s | GPU->CPU: %s", formatMiB(selectedFrame->bytesCpuToGpu).c_str(),
	            formatMiB(selectedFrame->bytesGpuToCpu).c_str());

	ImGui::SeparatorText("History");
	renderHistoryPlots(closedFrames);

	ImGui::SeparatorText("Transfer Events");
	renderTransferTable(*selectedFrame);

	const std::vector<const profiler::FrameProfile*> cpuAverageFrames =
	    collectRecentFrames(closedFrames, kFlameAverageWindowFrames, [](const profiler::FrameProfile&) { return true; });
	const FlameGraphAggregate averagedCpuFlame =
	    averageFlameGraph(cpuAverageFrames, [](const profiler::FrameProfile& frame) -> const std::vector<profiler::ScopeEvent>& {
		    return frame.cpuEvents;
	    }, [](const profiler::FrameProfile& frame) { return frame.cpuFrameMs; });

	ImGui::SeparatorText("CPU Flame Graph");
	ImGui::TextDisabled("Averaged over last %zu frame(s)", averagedCpuFlame.frameCount);
	drawFlameGraph("CPUFlameGraphAverage", averagedCpuFlame.events, averagedCpuFlame.totalTimeMs);

	const std::vector<const profiler::FrameProfile*> gpuAverageFrames = collectRecentFrames(
	    closedFrames, kFlameAverageWindowFrames, [](const profiler::FrameProfile& frame) { return frame.gpuResolved; });
	const FlameGraphAggregate averagedGpuFlame =
	    averageFlameGraph(gpuAverageFrames, [](const profiler::FrameProfile& frame) -> const std::vector<profiler::ScopeEvent>& {
		    return frame.gpuEvents;
	    }, [](const profiler::FrameProfile& frame) { return frame.gpuFrameMs; });

	ImGui::SeparatorText("GPU Flame Graph");
	if (!profiler.gpuTimingSupported()) {
		ImGui::TextDisabled("GPU timestamp queries are not supported on this OpenGL context.");
	} else if (averagedGpuFlame.frameCount > 0) {
		ImGui::TextDisabled("Averaged over last %zu resolved frame(s)", averagedGpuFlame.frameCount);
		drawFlameGraph("GPUFlameGraphAverage", averagedGpuFlame.events, averagedGpuFlame.totalTimeMs);
	} else {
		ImGui::TextDisabled("Waiting for GPU query results...");
	}
}

}  // namespace profiler_ui
