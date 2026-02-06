#include "core/profiler.hpp"

#include <glad/glad.h>

#include <algorithm>
#include <cstddef>

namespace profiler {

namespace {

[[nodiscard]] double toMilliseconds(std::chrono::steady_clock::duration duration) noexcept {
	return std::chrono::duration<double, std::milli>(duration).count();
}

}  // namespace

Profiler::Profiler(size_t maxFrames) : maxFrames_(std::max<size_t>(maxFrames, 8)) {}

Profiler::~Profiler() noexcept { shutdown(); }

void Profiler::shutdown() noexcept {
	if (shutdown_) return;

	for (const auto& pair : allocatedQueryPairs_) {
		GLuint queries[2] = {pair.first, pair.second};
		glDeleteQueries(2, queries);
	}
	freeQueryPairs_.clear();
	allocatedQueryPairs_.clear();
	pendingGpuScopes_.clear();
	activeGpuScopes_.clear();
	activeCpuScopes_.clear();
	frameHistory_.clear();
	activeFrameId_ = 0;
	shutdown_ = true;
}

void Profiler::setEnabled(bool enabled) noexcept { enabled_ = enabled; }

bool Profiler::isEnabled() const noexcept { return enabled_; }

void Profiler::beginFrame() {
	if (shutdown_) return;

	collectFinishedGpuQueries();

	if (!enabled_) {
		activeFrameId_ = 0;
		return;
	}

	if (activeFrameId_ != 0) {
		endFrame();
	}

	FrameProfile frame;
	frame.frameId = nextFrameId_++;
	frame.cpuFrameStart = std::chrono::steady_clock::now();
	frameHistory_.push_back(std::move(frame));
	activeFrameId_ = frameHistory_.back().frameId;

	trimHistory();
}

void Profiler::endFrame() {
	if (activeFrameId_ == 0) return;

	closeDanglingCpuScopes(activeFrameId_);

	FrameProfile* frame = findFrame(activeFrameId_);
	if (frame == nullptr) {
		activeFrameId_ = 0;
		return;
	}

	frame->cpuFrameMs = toMilliseconds(std::chrono::steady_clock::now() - frame->cpuFrameStart);
	frame->closed = true;
	markFrameGpuResolved(frame->frameId);

	activeFrameId_ = 0;
	collectFinishedGpuQueries();
	trimHistory();
}

void Profiler::collectFinishedGpuQueries() noexcept {
	if (shutdown_) return;
	if (pendingGpuScopes_.empty()) return;

	size_t i = 0;
	while (i < pendingGpuScopes_.size()) {
		const PendingGpuScope pending = pendingGpuScopes_[i];

		GLint startAvailable = 0;
		GLint endAvailable = 0;
		glGetQueryObjectiv(pending.startQuery, GL_QUERY_RESULT_AVAILABLE, &startAvailable);
		glGetQueryObjectiv(pending.endQuery, GL_QUERY_RESULT_AVAILABLE, &endAvailable);

		if (startAvailable == 0 || endAvailable == 0) {
			++i;
			continue;
		}

		GLuint64 startTimestampNs = 0;
		GLuint64 endTimestampNs = 0;
		glGetQueryObjectui64v(pending.startQuery, GL_QUERY_RESULT, &startTimestampNs);
		glGetQueryObjectui64v(pending.endQuery, GL_QUERY_RESULT, &endTimestampNs);

		FrameProfile* frame = findFrame(pending.frameId);
		if (frame != nullptr && pending.eventIndex < frame->gpuEvents.size()) {
			ScopeEvent& event = frame->gpuEvents[pending.eventIndex];

			if (!frame->gpuTimestampBaseNs.has_value()) {
				frame->gpuTimestampBaseNs = startTimestampNs;
			}

			if (startTimestampNs < frame->gpuTimestampBaseNs.value()) {
				const double deltaMs = static_cast<double>(frame->gpuTimestampBaseNs.value() - startTimestampNs) * 1.0e-6;
				for (ScopeEvent& existingEvent : frame->gpuEvents) {
					existingEvent.startMs += deltaMs;
					existingEvent.endMs += deltaMs;
				}
				frame->gpuTimestampBaseNs = startTimestampNs;
			}

			const uint64_t baseTimestampNs = frame->gpuTimestampBaseNs.value();
			const uint64_t clampedEnd = std::max<uint64_t>(endTimestampNs, startTimestampNs);
			event.startMs = static_cast<double>(startTimestampNs - baseTimestampNs) * 1.0e-6;
			event.endMs = static_cast<double>(clampedEnd - baseTimestampNs) * 1.0e-6;
			frame->gpuFrameMs = std::max(frame->gpuFrameMs, event.endMs);
		}

		// Remove from pending BEFORE calling markFrameGpuResolved, otherwise
		// hasPendingQueries() still finds this scope and the frame is never resolved.
		const uint64_t resolvedFrameId = pending.frameId;
		releaseQueryPair(pending.startQuery, pending.endQuery);
		pendingGpuScopes_[i] = pendingGpuScopes_.back();
		pendingGpuScopes_.pop_back();

		markFrameGpuResolved(resolvedFrameId);
	}

	trimHistory();
}

void Profiler::clearSession() noexcept {
	activeFrameId_ = 0;
	frameHistory_.clear();
	activeCpuScopes_.clear();
	activeGpuScopes_.clear();
	for (const PendingGpuScope& pending : pendingGpuScopes_) {
		releaseQueryPair(pending.startQuery, pending.endQuery);
	}
	pendingGpuScopes_.clear();
}

const std::deque<FrameProfile>& Profiler::frameHistory() const noexcept { return frameHistory_; }

const FrameProfile* Profiler::latestFrame() const noexcept {
	if (frameHistory_.empty()) return nullptr;
	return &frameHistory_.back();
}

const FrameProfile* Profiler::latestClosedFrame() const noexcept {
	for (auto it = frameHistory_.rbegin(); it != frameHistory_.rend(); ++it) {
		if (it->closed) return &(*it);
	}
	return nullptr;
}

const FrameProfile* Profiler::latestResolvedGpuFrame() const noexcept {
	for (auto it = frameHistory_.rbegin(); it != frameHistory_.rend(); ++it) {
		if (it->gpuResolved) return &(*it);
	}
	return nullptr;
}

size_t Profiler::pendingGpuScopeCount() const noexcept { return pendingGpuScopes_.size(); }

bool Profiler::gpuTimingSupported() const noexcept { return GLAD_GL_VERSION_3_3 && glQueryCounter != nullptr; }

Profiler::CpuScopeHandle Profiler::beginCpuScope(std::string_view name, SourceLocation source) noexcept {
	if (!enabled_ || activeFrameId_ == 0) return {};

	FrameProfile* frame = activeFrame();
	if (frame == nullptr) return {};

	const auto now = std::chrono::steady_clock::now();
	ScopeEvent event;
	event.name = std::string(name);
	event.type = ScopeType::CPU;
	event.startMs = toMilliseconds(now - frame->cpuFrameStart);
	event.endMs = event.startMs;
	event.depth = static_cast<int>(activeCpuScopes_.size());
	event.source = source;

	const size_t eventIndex = frame->cpuEvents.size();
	frame->cpuEvents.push_back(std::move(event));

	ActiveCpuScope activeScope;
	activeScope.frameId = frame->frameId;
	activeScope.eventIndex = eventIndex;
	activeScope.stackIndex = activeCpuScopes_.size();
	activeScope.type = ScopeType::CPU;
	activeScope.startTime = now;
	activeCpuScopes_.push_back(activeScope);

	CpuScopeHandle handle;
	handle.frameId = activeScope.frameId;
	handle.eventIndex = activeScope.eventIndex;
	handle.stackIndex = activeScope.stackIndex;
	handle.type = activeScope.type;
	handle.startTime = activeScope.startTime;
	handle.active = true;
	return handle;
}

Profiler::CpuScopeHandle Profiler::beginTransferScope(std::string_view name, size_t bytes, TransferDirection direction,
                                                      SourceLocation source) noexcept {
	if (!enabled_ || activeFrameId_ == 0) return {};

	FrameProfile* frame = activeFrame();
	if (frame == nullptr) return {};

	const auto now = std::chrono::steady_clock::now();
	ScopeEvent event;
	event.name = std::string(name);
	event.type = ScopeType::Transfer;
	event.startMs = toMilliseconds(now - frame->cpuFrameStart);
	event.endMs = event.startMs;
	event.depth = static_cast<int>(activeCpuScopes_.size());
	event.bytes = bytes;
	event.transferDirection = direction;
	event.source = source;

	const size_t eventIndex = frame->cpuEvents.size();
	frame->cpuEvents.push_back(std::move(event));

	ActiveCpuScope activeScope;
	activeScope.frameId = frame->frameId;
	activeScope.eventIndex = eventIndex;
	activeScope.stackIndex = activeCpuScopes_.size();
	activeScope.type = ScopeType::Transfer;
	activeScope.bytes = bytes;
	activeScope.direction = direction;
	activeScope.startTime = now;
	activeCpuScopes_.push_back(activeScope);

	CpuScopeHandle handle;
	handle.frameId = activeScope.frameId;
	handle.eventIndex = activeScope.eventIndex;
	handle.stackIndex = activeScope.stackIndex;
	handle.type = activeScope.type;
	handle.bytes = activeScope.bytes;
	handle.direction = activeScope.direction;
	handle.startTime = activeScope.startTime;
	handle.active = true;
	return handle;
}

void Profiler::endCpuScope(const CpuScopeHandle& handle) noexcept {
	if (!handle.active) return;

	FrameProfile* frame = findFrame(handle.frameId);
	if (frame == nullptr) return;
	if (handle.eventIndex >= frame->cpuEvents.size()) return;

	const auto now = std::chrono::steady_clock::now();
	ScopeEvent& event = frame->cpuEvents[handle.eventIndex];
	event.endMs = toMilliseconds(now - frame->cpuFrameStart);

	for (size_t index = activeCpuScopes_.size(); index > 0; --index) {
		const size_t activeIndex = index - 1;
		if (activeCpuScopes_[activeIndex].frameId == handle.frameId && activeCpuScopes_[activeIndex].eventIndex == handle.eventIndex) {
			const ActiveCpuScope active = activeCpuScopes_[activeIndex];
			activeCpuScopes_.erase(activeCpuScopes_.begin() + static_cast<std::ptrdiff_t>(activeIndex));

			if (active.type == ScopeType::Transfer) {
				const double durationMs = toMilliseconds(now - active.startTime);
				TransferEvent transfer;
				transfer.name = event.name;
				transfer.bytes = active.bytes;
				transfer.direction = active.direction;
				transfer.cpuTimeMs = durationMs;
				frame->transfers.push_back(std::move(transfer));
				frame->transferCpuMs += durationMs;

				if (active.direction == TransferDirection::CPUToGPU) {
					frame->bytesCpuToGpu += active.bytes;
				} else if (active.direction == TransferDirection::GPUToCPU) {
					frame->bytesGpuToCpu += active.bytes;
				}
			}

			break;
		}
	}
}

Profiler::GpuScopeHandle Profiler::beginGpuScope(std::string_view name, SourceLocation source) noexcept {
	if (!enabled_ || activeFrameId_ == 0) return {};
	if (!gpuTimingSupported()) return {};

	FrameProfile* frame = activeFrame();
	if (frame == nullptr) return {};

	const auto [startQuery, endQuery] = acquireQueryPair();
	if (startQuery == 0 || endQuery == 0) return {};

	ScopeEvent event;
	event.name = std::string(name);
	event.type = ScopeType::GPU;
	event.depth = static_cast<int>(activeGpuScopes_.size());
	event.source = source;
	const size_t eventIndex = frame->gpuEvents.size();
	frame->gpuEvents.push_back(std::move(event));

	glQueryCounter(startQuery, GL_TIMESTAMP);

	GpuScopeHandle handle;
	handle.frameId = frame->frameId;
	handle.eventIndex = eventIndex;
	handle.stackIndex = activeGpuScopes_.size();
	handle.startQuery = startQuery;
	handle.endQuery = endQuery;
	handle.active = true;
	activeGpuScopes_.push_back(handle);

	return handle;
}

void Profiler::endGpuScope(const GpuScopeHandle& handle) noexcept {
	if (!handle.active) return;

	glQueryCounter(handle.endQuery, GL_TIMESTAMP);

	PendingGpuScope pending;
	pending.frameId = handle.frameId;
	pending.eventIndex = handle.eventIndex;
	pending.startQuery = handle.startQuery;
	pending.endQuery = handle.endQuery;
	pendingGpuScopes_.push_back(pending);

	for (size_t index = activeGpuScopes_.size(); index > 0; --index) {
		const size_t activeIndex = index - 1;
		if (activeGpuScopes_[activeIndex].frameId == handle.frameId && activeGpuScopes_[activeIndex].eventIndex == handle.eventIndex) {
			activeGpuScopes_.erase(activeGpuScopes_.begin() + static_cast<std::ptrdiff_t>(activeIndex));
			break;
		}
	}
}

FrameProfile* Profiler::activeFrame() noexcept { return findFrame(activeFrameId_); }

FrameProfile* Profiler::findFrame(uint64_t frameId) noexcept {
	for (FrameProfile& frame : frameHistory_) {
		if (frame.frameId == frameId) return &frame;
	}
	return nullptr;
}

const FrameProfile* Profiler::findFrame(uint64_t frameId) const noexcept {
	for (const FrameProfile& frame : frameHistory_) {
		if (frame.frameId == frameId) return &frame;
	}
	return nullptr;
}

bool Profiler::hasPendingQueries(uint64_t frameId) const noexcept {
	for (const PendingGpuScope& pending : pendingGpuScopes_) {
		if (pending.frameId == frameId) return true;
	}
	return false;
}

std::pair<uint32_t, uint32_t> Profiler::acquireQueryPair() noexcept {
	if (!freeQueryPairs_.empty()) {
		const auto pair = freeQueryPairs_.back();
		freeQueryPairs_.pop_back();
		return pair;
	}

	GLuint queries[2] = {0, 0};
	glGenQueries(2, queries);
	if (queries[0] == 0 || queries[1] == 0) {
		if (queries[0] != 0) glDeleteQueries(1, &queries[0]);
		if (queries[1] != 0) glDeleteQueries(1, &queries[1]);
		return {0, 0};
	}

	const std::pair<uint32_t, uint32_t> pair{queries[0], queries[1]};
	allocatedQueryPairs_.push_back(pair);
	return pair;
}

void Profiler::releaseQueryPair(uint32_t startQuery, uint32_t endQuery) noexcept {
	if (startQuery == 0 || endQuery == 0) return;
	freeQueryPairs_.push_back({startQuery, endQuery});
}

void Profiler::closeDanglingCpuScopes(uint64_t frameId) noexcept {
	if (frameId == 0) return;

	FrameProfile* frame = findFrame(frameId);
	if (frame == nullptr) return;

	const auto now = std::chrono::steady_clock::now();
	for (size_t index = activeCpuScopes_.size(); index > 0; --index) {
		const size_t activeIndex = index - 1;
		const ActiveCpuScope active = activeCpuScopes_[activeIndex];
		if (active.frameId != frameId) continue;

		if (active.eventIndex < frame->cpuEvents.size()) {
			ScopeEvent& event = frame->cpuEvents[active.eventIndex];
			event.endMs = toMilliseconds(now - frame->cpuFrameStart);

			if (active.type == ScopeType::Transfer) {
				TransferEvent transfer;
				transfer.name = event.name;
				transfer.bytes = active.bytes;
				transfer.direction = active.direction;
				transfer.cpuTimeMs = toMilliseconds(now - active.startTime);
				frame->transfers.push_back(std::move(transfer));
				frame->transferCpuMs += transfer.cpuTimeMs;

				if (active.direction == TransferDirection::CPUToGPU) {
					frame->bytesCpuToGpu += active.bytes;
				} else if (active.direction == TransferDirection::GPUToCPU) {
					frame->bytesGpuToCpu += active.bytes;
				}
			}
		}

		activeCpuScopes_.erase(activeCpuScopes_.begin() + static_cast<std::ptrdiff_t>(activeIndex));
	}
}

void Profiler::markFrameGpuResolved(uint64_t frameId) noexcept {
	FrameProfile* frame = findFrame(frameId);
	if (frame == nullptr) return;

	if (!frame->closed) return;
	if (hasPendingQueries(frameId)) return;
	frame->gpuResolved = true;
}

void Profiler::trimHistory() noexcept {
	while (frameHistory_.size() > maxFrames_) {
		if (!frameHistory_.front().gpuResolved && hasPendingQueries(frameHistory_.front().frameId)) {
			break;
		}
		frameHistory_.pop_front();
	}
}

CpuScope::CpuScope(Profiler& profiler, std::string_view name, const char* file, int line) noexcept : profiler_(&profiler) {
	handle_ = profiler_->beginCpuScope(name, {.file = file, .line = line});
}

CpuScope::~CpuScope() noexcept {
	if (profiler_ == nullptr) return;
	profiler_->endCpuScope(handle_);
}

GpuScope::GpuScope(Profiler& profiler, std::string_view name, const char* file, int line) noexcept : profiler_(&profiler) {
	handle_ = profiler_->beginGpuScope(name, {.file = file, .line = line});
}

GpuScope::~GpuScope() noexcept {
	if (profiler_ == nullptr) return;
	profiler_->endGpuScope(handle_);
}

TransferScope::TransferScope(Profiler& profiler, std::string_view name, size_t bytes, TransferDirection direction, const char* file,
                             int line) noexcept
    : profiler_(&profiler) {
	handle_ = profiler_->beginTransferScope(name, bytes, direction, {.file = file, .line = line});
}

TransferScope::~TransferScope() noexcept {
	if (profiler_ == nullptr) return;
	profiler_->endCpuScope(handle_);
}

}  // namespace profiler
