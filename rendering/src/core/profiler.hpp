#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace profiler {

enum class ScopeType { CPU, GPU, Transfer };

enum class TransferDirection {
	CPUToGPU,
	GPUToCPU,
	CPUToCPU
};

struct SourceLocation {
	const char* file{""};
	int line{0};
};

struct ScopeEvent {
	std::string name;
	ScopeType type{ScopeType::CPU};
	double startMs{0.0};
	double endMs{0.0};
	int depth{0};
	size_t bytes{0};
	TransferDirection transferDirection{TransferDirection::CPUToCPU};
	SourceLocation source;

	[[nodiscard]] double durationMs() const noexcept { return endMs - startMs; }
};

struct TransferEvent {
	std::string name;
	size_t bytes{0};
	TransferDirection direction{TransferDirection::CPUToCPU};
	double cpuTimeMs{0.0};
};

struct FrameProfile {
	uint64_t frameId{0};
	double cpuFrameMs{0.0};
	double gpuFrameMs{0.0};
	double transferCpuMs{0.0};
	size_t bytesCpuToGpu{0};
	size_t bytesGpuToCpu{0};
	bool closed{false};
	bool gpuResolved{false};
	std::vector<ScopeEvent> cpuEvents;
	std::vector<ScopeEvent> gpuEvents;
	std::vector<TransferEvent> transfers;

	// Internal frame timing anchors used while data is live in history.
	std::chrono::steady_clock::time_point cpuFrameStart{};
	std::optional<uint64_t> gpuTimestampBaseNs{};
};

class Profiler {
   public:
	struct CpuScopeHandle {
		uint64_t frameId{0};
		size_t eventIndex{0};
		size_t stackIndex{0};
		ScopeType type{ScopeType::CPU};
		size_t bytes{0};
		TransferDirection direction{TransferDirection::CPUToCPU};
		std::chrono::steady_clock::time_point startTime{};
		bool active{false};
	};

	struct GpuScopeHandle {
		uint64_t frameId{0};
		size_t eventIndex{0};
		size_t stackIndex{0};
		uint32_t startQuery{0};
		uint32_t endQuery{0};
		bool active{false};
	};

	explicit Profiler(size_t maxFrames = 240);
	~Profiler() noexcept;

	void shutdown() noexcept;

	void setEnabled(bool enabled) noexcept;
	[[nodiscard]] bool isEnabled() const noexcept;

	void beginFrame();
	void endFrame();
	void collectFinishedGpuQueries() noexcept;
	void clearSession() noexcept;

	[[nodiscard]] const std::deque<FrameProfile>& frameHistory() const noexcept;
	[[nodiscard]] const FrameProfile* latestFrame() const noexcept;
	[[nodiscard]] const FrameProfile* latestClosedFrame() const noexcept;
	[[nodiscard]] const FrameProfile* latestResolvedGpuFrame() const noexcept;
	[[nodiscard]] size_t pendingGpuScopeCount() const noexcept;
	[[nodiscard]] bool gpuTimingSupported() const noexcept;

	CpuScopeHandle beginCpuScope(std::string_view name, SourceLocation source = {}) noexcept;
	CpuScopeHandle beginTransferScope(std::string_view name, size_t bytes, TransferDirection direction, SourceLocation source = {}) noexcept;
	void endCpuScope(const CpuScopeHandle& handle) noexcept;

	GpuScopeHandle beginGpuScope(std::string_view name, SourceLocation source = {}) noexcept;
	void endGpuScope(const GpuScopeHandle& handle) noexcept;

   private:
	struct ActiveCpuScope {
		uint64_t frameId{0};
		size_t eventIndex{0};
		size_t stackIndex{0};
		ScopeType type{ScopeType::CPU};
		size_t bytes{0};
		TransferDirection direction{TransferDirection::CPUToCPU};
		std::chrono::steady_clock::time_point startTime{};
	};

	struct PendingGpuScope {
		uint64_t frameId{0};
		size_t eventIndex{0};
		uint32_t startQuery{0};
		uint32_t endQuery{0};
	};

	[[nodiscard]] FrameProfile* activeFrame() noexcept;
	[[nodiscard]] FrameProfile* findFrame(uint64_t frameId) noexcept;
	[[nodiscard]] const FrameProfile* findFrame(uint64_t frameId) const noexcept;
	[[nodiscard]] bool hasPendingQueries(uint64_t frameId) const noexcept;
	[[nodiscard]] std::pair<uint32_t, uint32_t> acquireQueryPair() noexcept;
	void releaseQueryPair(uint32_t startQuery, uint32_t endQuery) noexcept;
	void closeDanglingCpuScopes(uint64_t frameId) noexcept;
	void markFrameGpuResolved(uint64_t frameId) noexcept;
	void trimHistory() noexcept;

	size_t maxFrames_{240};
	uint64_t nextFrameId_{1};
	uint64_t activeFrameId_{0};
	bool enabled_{true};
	bool shutdown_{false};

	std::deque<FrameProfile> frameHistory_;
	std::vector<ActiveCpuScope> activeCpuScopes_;
	std::vector<GpuScopeHandle> activeGpuScopes_;
	std::vector<PendingGpuScope> pendingGpuScopes_;

	std::vector<std::pair<uint32_t, uint32_t>> freeQueryPairs_;
	std::vector<std::pair<uint32_t, uint32_t>> allocatedQueryPairs_;
};

class CpuScope {
   public:
	CpuScope(Profiler& profiler, std::string_view name, const char* file, int line) noexcept;
	~CpuScope() noexcept;

	CpuScope(const CpuScope&) = delete;
	CpuScope& operator=(const CpuScope&) = delete;

   private:
	Profiler* profiler_{nullptr};
	Profiler::CpuScopeHandle handle_{};
};

class GpuScope {
   public:
	GpuScope(Profiler& profiler, std::string_view name, const char* file, int line) noexcept;
	~GpuScope() noexcept;

	GpuScope(const GpuScope&) = delete;
	GpuScope& operator=(const GpuScope&) = delete;

   private:
	Profiler* profiler_{nullptr};
	Profiler::GpuScopeHandle handle_{};
};

class TransferScope {
   public:
	TransferScope(Profiler& profiler, std::string_view name, size_t bytes, TransferDirection direction, const char* file, int line) noexcept;
	~TransferScope() noexcept;

	TransferScope(const TransferScope&) = delete;
	TransferScope& operator=(const TransferScope&) = delete;

   private:
	Profiler* profiler_{nullptr};
	Profiler::CpuScopeHandle handle_{};
};

}  // namespace profiler

#define VQVDB_PROFILE_CONCAT_INNER(x, y) x##y
#define VQVDB_PROFILE_CONCAT(x, y) VQVDB_PROFILE_CONCAT_INNER(x, y)

#define CPU_PROFILE_SCOPE(prof, label) \
	::profiler::CpuScope VQVDB_PROFILE_CONCAT(_cpu_profile_scope_, __COUNTER__)((prof), (label), __FILE__, __LINE__)

#define GPU_PROFILE_SCOPE(prof, label) \
	::profiler::GpuScope VQVDB_PROFILE_CONCAT(_gpu_profile_scope_, __COUNTER__)((prof), (label), __FILE__, __LINE__)

#define TRANSFER_PROFILE_SCOPE(prof, label, bytes, direction) \
	::profiler::TransferScope VQVDB_PROFILE_CONCAT(_transfer_profile_scope_, __COUNTER__)((prof), (label), (bytes), (direction), __FILE__, __LINE__)
