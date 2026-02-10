#pragma once

#include <deque>
#include <future>
#include <optional>
#include <string>
#include <vector>

#include "core/profiler.hpp"
#include "vqvdb/brick_cache.hpp"
#include "vqvdb/decoder_backend.hpp"
#include "vqvdb/gpu_resources.hpp"
#include "vqvdb/vqvdb_types.hpp"

struct GLFWwindow;
struct CameraState;
struct CameraLimits;
struct Timing;

// VQVDB Volume State - data loaded from .vqvdb files
struct VQVDBVolumeState {
	// Loaded file data
	std::optional<vqvdb::VQVDBFile> file;

	// Computed statistics (cached)
	vqvdb::VQVDBStats stats;

	// Whether data is currently loaded
	bool isLoaded{false};

	// Whether renderer needs to update block bboxes
	bool needsRendererUpdate{false};
	// Load error message (if any)
	std::string loadError;

	// Path of the loaded file
	std::string loadedPath;

	void clear() noexcept {
		file.reset();
		stats = {};
		isLoaded = false;
		needsRendererUpdate = false;
		loadError.clear();
		loadedPath.clear();
	}
};

// GPU Resources State - Milestone 1.2/1.3
struct DecoderInitTaskResult {
	std::unique_ptr<vqvdb::DecoderBackend> backend;
	std::string readyModelPath;
	std::string errorMessage;
	bool success{false};
};

struct GPUResourcesState {
	// GPU resources handle
	vqvdb::GPUResources resources;

	// Verification results
	vqvdb::VerificationResult blockIndicesVerification;
	vqvdb::VerificationResult blockMetadataVerification;

	// Grid transform info for GPU instanced rendering
	float voxelSize{1.0f};
	float blockSize{8.0f};
	glm::mat4 gridTransform{1.0f};
	bool rendererNeedsUpdate{false};  // Signal render loop to update renderer state

	// Brick cache state (Task 3 debug visualization)
	vqvdb::BrickCache brickCache;
	bool brickCacheInitialized{false};
	int brickCacheCapacity{2048};
	bool brickCacheAllocateTexture{true};
	int brickCachePrimeCount{256};
	int brickCacheHeatmapSlice{0};
	int brickCacheHeatmapMode{0};  // 0 age, 1 touches, 2 LRU
	bool autoUpdateVisibleCache{true};
	bool enableFrustumCulling{true};
	bool enableDepthOcclusion{true};
	bool colorBlocksByVisibility{true};
	int decodeBudgetPerFrame{64};
	float maxDecodeDistance{0.0f};  // 0 = no distance limit
	float occlusionDepthBias{0.001f};
	uint32_t visibleBlocksLastFrame{0};
	uint32_t visibleCachedLastFrame{0};
	uint32_t visibleMissingLastFrame{0};
	uint32_t scheduledDecodesLastFrame{0};
	uint32_t occludedRequestsLastFrame{0};
	uint32_t cacheTouchedLastFrame{0};
	uint32_t cacheInsertedLastFrame{0};
	uint32_t cacheEvictedLastFrame{0};
	uint32_t decodedBlocksLastFrame{0};
	std::string schedulerError;
	std::unique_ptr<vqvdb::DecoderBackend> decoderBackend;
	std::string decoderModelPath{};
	bool decoderInitRequested{false};
	bool decoderInitInProgress{false};
	std::future<DecoderInitTaskResult> decoderInitTask;
	bool decoderReady{false};
	std::string decoderStatus{"Not initialized"};
	std::string decoderError;

	// Block display limit
	int maxDisplayBlocks{0};    // 0 = show all blocks
	bool useBlockLimit{false};  // Toggle for block limit

	// State flags
	bool blockIndicesUploaded{false};
	bool blockIndicesVerified{false};
	bool blockMetadataUploaded{false};
	bool blockMetadataVerified{false};

	// Error messages
	std::string blockDataError;
	std::string brickCacheError;

	// Request flags (set by UI, processed by render loop)
	bool blockDataUploadRequested{false};
	bool blockDataVerifyRequested{false};
	bool brickCacheReinitRequested{false};
	bool brickCachePrimeRequested{false};
	bool brickCacheClearRequested{false};

	void clear() noexcept {
		blockIndicesVerification = {};
		blockMetadataVerification = {};
		voxelSize = 1.0f;
		blockSize = 8.0f;
		gridTransform = glm::mat4(1.0f);
		rendererNeedsUpdate = false;
		brickCacheCapacity = 2048;
		brickCachePrimeCount = 256;
		brickCacheHeatmapSlice = 0;
		brickCacheHeatmapMode = 0;
		autoUpdateVisibleCache = true;
		enableFrustumCulling = true;
		enableDepthOcclusion = true;
		colorBlocksByVisibility = true;
		decodeBudgetPerFrame = 64;
		maxDecodeDistance = 0.0f;
		occlusionDepthBias = 0.001f;
		visibleBlocksLastFrame = 0;
		visibleCachedLastFrame = 0;
		visibleMissingLastFrame = 0;
		scheduledDecodesLastFrame = 0;
		occludedRequestsLastFrame = 0;
		cacheTouchedLastFrame = 0;
		cacheInsertedLastFrame = 0;
		cacheEvictedLastFrame = 0;
		decodedBlocksLastFrame = 0;
		schedulerError.clear();
		if (decoderInitTask.valid()) {
			decoderInitTask.wait();
		}
		decoderBackend.reset();
		decoderModelPath = "models/onnx_models/decoder.onnx";
		decoderInitRequested = false;
		decoderInitInProgress = false;
		decoderReady = false;
		decoderStatus = "Not initialized";
		decoderError.clear();
		maxDisplayBlocks = 0;
		useBlockLimit = false;
		blockIndicesUploaded = false;
		blockIndicesVerified = false;
		blockMetadataUploaded = false;
		blockMetadataVerified = false;
		blockDataError.clear();
		brickCacheError.clear();
	}
};

// UI State - holds all UI-related data
struct UIState {
	// Layout dimensions (computed each frame based on window size)
	float leftPanelWidth{280.0f};
	float rightPanelWidth{340.0f};
	float bottomPanelHeight{180.0f};

	// Viewport region (computed)
	int viewportX{0};
	int viewportY{0};
	int viewportWidth{1};
	int viewportHeight{1};
	bool viewportHovered{false};
	bool viewportFocused{false};

	// File loading
	std::string loadedFilePath{"(No file loaded)"};
	bool fileLoadRequested{false};

	// VQVDB volume data
	VQVDBVolumeState volumeState;

	// GPU resources (Milestone 1.2/1.3)
	GPUResourcesState gpuState;

	// Performance tracking
	static constexpr size_t kMaxFrameSamples = 60;
	std::deque<float> frameTimes;
	std::deque<float> fpsHistory;
	size_t frameTimeIndex{0};
	size_t fpsHistoryIndex{0};
	float avgFrameTime{0.0f};
	float avgFps{0.0f};
	float minFps{0.0f};
	float maxFps{0.0f};

	// Debug log
	std::vector<std::string> debugLog;
	static constexpr size_t kMaxLogLines = 100;
	bool scrollLogToBottom{true};

	// UI visibility toggles
	bool showCameraInfo{true};
	bool showPerformance{true};
	bool showProfiler{true};
	bool showDebugLog{true};
	bool showPhase1Data{true};

	// CPU/GPU profiler
	profiler::Profiler profiler{};
	bool profilerEnabled{true};
	bool profilerGathering{true};
	bool profilerFollowLatest{true};
	int profilerSelectedClosedFrame{0};
};

namespace ui {

// Initialize ImGui context and backend
[[nodiscard]] bool init(GLFWwindow* window) noexcept;

// Shutdown ImGui
void shutdown() noexcept;

// Begin a new ImGui frame
void beginFrame() noexcept;

// End ImGui frame and render
void endFrame() noexcept;

// Render the full UI layout and return viewport dimensions
void renderUI(UIState& state, CameraState& camera, CameraLimits& limits, const Timing& timing, int windowWidth, int windowHeight) noexcept;

// Add a message to the debug log
void logMessage(UIState& state, const std::string& message) noexcept;

// Load a VQVDB file and update the volume state
// Returns true on success, false on failure (error stored in volumeState.loadError)
bool loadVQVDBFile(UIState& state, const std::string& filePath) noexcept;

// Upload block data to GPU (Milestone 1.3)
bool uploadBlockDataToGPU(UIState& state) noexcept;

// Verify block data on GPU (Milestone 1.3)
void verifyBlockDataOnGPU(UIState& state) noexcept;

// Brick cache control helpers (Task 3 debug)
bool reinitBrickCache(UIState& state) noexcept;
void primeBrickCacheFromLoadedBlocks(UIState& state) noexcept;
void clearBrickCache(UIState& state) noexcept;

// Initialize GPU resources (call after GL context is ready)
void initGPUResources(UIState& state) noexcept;

// Initialize decoder backend with configured model path.
[[nodiscard]] bool initDecoderBackend(UIState& state) noexcept;

// Poll decoder async initialization; finalizes ready/error state when complete.
void pollDecoderBackendInit(UIState& state) noexcept;

// Shutdown decoder backend and release decoder resources.
void shutdownDecoderBackend(UIState& state) noexcept;

// Shutdown GPU resources
void shutdownGPUResources(UIState& state) noexcept;

// Check if mouse is over ImGui windows (for input blocking)
[[nodiscard]] bool wantCaptureMouse() noexcept;

// Check if keyboard is captured by ImGui
[[nodiscard]] bool wantCaptureKeyboard() noexcept;

}  // namespace ui
