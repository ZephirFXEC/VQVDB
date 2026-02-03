#pragma once

#include <deque>
#include <optional>
#include <string>
#include <vector>

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

// UI State - holds all UI-related data
struct UIState {
	// Layout dimensions (computed each frame based on window size)
	float leftPanelWidth{280.0f};
	float bottomPanelHeight{180.0f};

	// Viewport region (computed)
	int viewportX{0};
	int viewportY{0};
	int viewportWidth{1};
	int viewportHeight{1};

	// File loading
	std::string loadedFilePath{"(No file loaded)"};
	bool fileLoadRequested{false};

	// VQVDB volume data
	VQVDBVolumeState volumeState;

	// Performance tracking
	static constexpr size_t kMaxFrameSamples = 120;
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
	bool showDebugLog{true};
	bool showPhase1Data{true};
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

// Check if mouse is over ImGui windows (for input blocking)
[[nodiscard]] bool wantCaptureMouse() noexcept;

// Check if keyboard is captured by ImGui
[[nodiscard]] bool wantCaptureKeyboard() noexcept;

}  // namespace ui
