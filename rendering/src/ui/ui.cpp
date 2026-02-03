#include "ui/ui.hpp"

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <numeric>
#include <sstream>

#include "core/camera.hpp"
#include "core/types.hpp"
#include "vqvdb/vqvdb_loader.hpp"

namespace ui {

bool init(GLFWwindow* window) noexcept {
	// Create ImGui context
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();

	ImGuiIO& io = ImGui::GetIO();
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
	io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

	// Set up dark style
	ImGui::StyleColorsDark();
	ImGuiStyle& style = ImGui::GetStyle();
	style.WindowRounding = 4.0f;
	style.FrameRounding = 2.0f;
	style.GrabRounding = 2.0f;
	style.WindowBorderSize = 1.0f;
	style.FrameBorderSize = 0.0f;
	style.PopupBorderSize = 1.0f;
	style.AntiAliasedLines = true;

	// Customize colors for a more professional look
	ImVec4* colors = style.Colors;
	colors[ImGuiCol_WindowBg] = ImVec4(0.08f, 0.08f, 0.10f, 0.95f);
	colors[ImGuiCol_TitleBg] = ImVec4(0.10f, 0.10f, 0.12f, 1.00f);
	colors[ImGuiCol_TitleBgActive] = ImVec4(0.15f, 0.15f, 0.18f, 1.00f);
	colors[ImGuiCol_FrameBg] = ImVec4(0.12f, 0.12f, 0.15f, 1.00f);
	colors[ImGuiCol_FrameBgHovered] = ImVec4(0.18f, 0.18f, 0.22f, 1.00f);
	colors[ImGuiCol_FrameBgActive] = ImVec4(0.22f, 0.22f, 0.27f, 1.00f);
	colors[ImGuiCol_Button] = ImVec4(0.20f, 0.40f, 0.60f, 1.00f);
	colors[ImGuiCol_ButtonHovered] = ImVec4(0.25f, 0.50f, 0.75f, 1.00f);
	colors[ImGuiCol_ButtonActive] = ImVec4(0.30f, 0.60f, 0.90f, 1.00f);
	colors[ImGuiCol_Header] = ImVec4(0.20f, 0.40f, 0.60f, 0.70f);
	colors[ImGuiCol_HeaderHovered] = ImVec4(0.25f, 0.50f, 0.75f, 0.80f);
	colors[ImGuiCol_HeaderActive] = ImVec4(0.30f, 0.60f, 0.90f, 1.00f);
	colors[ImGuiCol_PlotLines] = ImVec4(0.40f, 0.80f, 0.40f, 1.00f);
	colors[ImGuiCol_PlotHistogram] = ImVec4(0.40f, 0.70f, 0.90f, 1.00f);

	// Initialize platform/renderer backends
	// Pass false to NOT install callbacks - we'll forward events manually
	if (!ImGui_ImplGlfw_InitForOpenGL(window, false)) {
		return false;
	}

	if (!ImGui_ImplOpenGL3_Init("#version 450")) {
		ImGui_ImplGlfw_Shutdown();
		return false;
	}

	return true;
}

void shutdown() noexcept {
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
}

void beginFrame() noexcept {
	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();
}

void endFrame() noexcept {
	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

namespace {

void updatePerformanceStats(UIState& state, const Timing& timing) {
	// Add current frame time
	const float frameTimeMs = timing.deltaTime * 1000.0f;
	state.frameTimes.push_back(frameTimeMs);
	if (state.frameTimes.size() > UIState::kMaxFrameSamples) {
		state.frameTimes.pop_front();
	}

	// Calculate FPS
	const float fps = timing.deltaTime > 0.0f ? 1.0f / timing.deltaTime : 0.0f;
	state.fpsHistory.push_back(fps);
	if (state.fpsHistory.size() > UIState::kMaxFrameSamples) {
		state.fpsHistory.pop_front();
	}

	// Compute statistics
	if (!state.frameTimes.empty()) {
		float sum = 0.0f;
		for (float ft : state.frameTimes) {
			sum += ft;
		}
		state.avgFrameTime = sum / static_cast<float>(state.frameTimes.size());
	}

	if (!state.fpsHistory.empty()) {
		float sum = 0.0f;
		state.minFps = state.fpsHistory.front();
		state.maxFps = state.fpsHistory.front();
		for (float f : state.fpsHistory) {
			sum += f;
			state.minFps = std::min(state.minFps, f);
			state.maxFps = std::max(state.maxFps, f);
		}
		state.avgFps = sum / static_cast<float>(state.fpsHistory.size());
	}
}

void renderLeftPanel(UIState& state, CameraState& camera, CameraLimits& limits, int windowHeight) {
	ImGui::SetNextWindowPos(ImVec2(0, 0));
	ImGui::SetNextWindowSize(ImVec2(state.leftPanelWidth, static_cast<float>(windowHeight)));

	ImGuiWindowFlags flags =
	    ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoBringToFrontOnFocus;

	if (ImGui::Begin("Control Panel", nullptr, flags)) {
		// === File Section ===
		if (ImGui::CollapsingHeader("File", ImGuiTreeNodeFlags_DefaultOpen)) {
			ImGui::Text("Current File:");
			ImGui::TextWrapped("%s", state.loadedFilePath.c_str());
			ImGui::Spacing();

			if (ImGui::Button("Load VQVDB File...", ImVec2(-1, 0))) {
				state.fileLoadRequested = true;
			}
			ImGui::Spacing();
			ImGui::Separator();
		}

		// === Camera Section ===
		if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
			ImGui::Text("Position:");
			ImGui::Text("  X: %.2f", camera.position.x);
			ImGui::Text("  Y: %.2f", camera.position.y);
			ImGui::Text("  Z: %.2f", camera.position.z);
			ImGui::Spacing();

			ImGui::Text("Target:");
			ImGui::Text("  X: %.2f", camera.target.x);
			ImGui::Text("  Y: %.2f", camera.target.y);
			ImGui::Text("  Z: %.2f", camera.target.z);
			ImGui::Spacing();

			ImGui::Text("Orientation:");
			ImGui::Text("  Yaw:   %.1f deg", camera.yawDegrees);
			ImGui::Text("  Pitch: %.1f deg", camera.pitchDegrees);
			ImGui::Text("  Distance: %.2f", camera.distance);
			ImGui::Spacing();

			// Editable camera parameters
			ImGui::SliderFloat("FOV", &camera.fovDegrees, 30.0f, 120.0f, "%.0f deg");
			ImGui::SliderFloat("Near", &camera.nearPlane, 0.01f, 1.0f, "%.3f");
			ImGui::SliderFloat("Far", &camera.farPlane, 100.0f, 10000.0f, "%.0f");
			ImGui::Spacing();

			// Camera limits
			if (ImGui::TreeNode("Movement Settings")) {
				ImGui::SliderFloat("Move Speed", &limits.moveSpeed, 1.0f, 100.0f);
				ImGui::SliderFloat("Orbit Sens.", &limits.orbitSensitivity, 0.05f, 1.0f);
				ImGui::SliderFloat("Zoom Sens.", &limits.zoomSensitivity, 0.05f, 0.5f);
				ImGui::TreePop();
			}

			ImGui::Spacing();
			if (ImGui::Button("Reset Camera", ImVec2(-1, 0))) {
				camera.yawDegrees = 0.0f;
				camera.pitchDegrees = 20.0f;
				camera.distance = 5.0f;
				camera.target = glm::vec3(0.0f);
				camera.velocity = glm::vec3(0.0f);
			}

			ImGui::Separator();
		}

		// === Rendering Section ===
		if (ImGui::CollapsingHeader("Rendering", ImGuiTreeNodeFlags_DefaultOpen)) {
			ImGui::Text("Viewport: %d x %d", state.viewportWidth, state.viewportHeight);
			ImGui::Text("Aspect: %.3f", camera.aspectRatio);
			ImGui::Spacing();
			ImGui::Separator();
		}

		// === View Toggles ===
		if (ImGui::CollapsingHeader("View Options")) {
			ImGui::Checkbox("Show Camera Info", &state.showCameraInfo);
			ImGui::Checkbox("Show Performance", &state.showPerformance);
			ImGui::Checkbox("Show Debug Log", &state.showDebugLog);
			ImGui::Checkbox("Show Phase 1 Data", &state.showPhase1Data);
			ImGui::Separator();
		}

		// === Help Section ===
		if (ImGui::CollapsingHeader("Controls")) {
			ImGui::BulletText("LMB + Drag: Orbit camera");
			ImGui::BulletText("Scroll: Zoom in/out");
			ImGui::BulletText("W/S: Move forward/back");
			ImGui::BulletText("A/D: Move left/right");
			ImGui::BulletText("Q/E: Move down/up");
		}
	}
	ImGui::End();
}

void renderBottomPanel(UIState& state, int windowWidth, int windowHeight) {
	const float bottomY = static_cast<float>(windowHeight) - state.bottomPanelHeight;

	ImGui::SetNextWindowPos(ImVec2(state.leftPanelWidth, bottomY));
	ImGui::SetNextWindowSize(ImVec2(static_cast<float>(windowWidth) - state.leftPanelWidth, state.bottomPanelHeight));

	ImGuiWindowFlags flags =
	    ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoBringToFrontOnFocus;

	if (ImGui::Begin("Info Panel", nullptr, flags)) {
		// Use tabs for different sections
		if (ImGui::BeginTabBar("BottomTabs")) {
			// === Performance Tab ===
			if (ImGui::BeginTabItem("Performance")) {
				ImGui::Columns(2, "perfColumns", true);

				// Left column: stats
				ImGui::Text("Frame Time: %.2f ms", state.avgFrameTime);
				ImGui::Text("FPS: %.1f (%.0f - %.0f)", state.avgFps, state.minFps, state.maxFps);
				ImGui::Spacing();
				ImGui::Text("Samples: %zu", state.frameTimes.size());

				ImGui::NextColumn();

				// Right column: graph
				if (!state.fpsHistory.empty()) {
					std::vector<float> fpsData(state.fpsHistory.begin(), state.fpsHistory.end());
					ImGui::PlotLines("##FPS", fpsData.data(), static_cast<int>(fpsData.size()), 0, "FPS", 0.0f, 120.0f, ImVec2(-1, 80));
				}

				ImGui::Columns(1);
				ImGui::EndTabItem();
			}

			// === Debug Log Tab ===
			if (ImGui::BeginTabItem("Debug Log")) {
				const float footerHeight = ImGui::GetStyle().ItemSpacing.y + ImGui::GetFrameHeightWithSpacing();

				if (ImGui::BeginChild("LogScrollRegion", ImVec2(0, -footerHeight), ImGuiChildFlags_None,
				                      ImGuiWindowFlags_HorizontalScrollbar)) {
					for (const auto& line : state.debugLog) {
						ImGui::TextUnformatted(line.c_str());
					}
					if (state.scrollLogToBottom && ImGui::GetScrollY() >= ImGui::GetScrollMaxY()) {
						ImGui::SetScrollHereY(1.0f);
					}
				}
				ImGui::EndChild();

				if (ImGui::Button("Clear Log")) {
					state.debugLog.clear();
				}
				ImGui::SameLine();
				ImGui::Checkbox("Auto-scroll", &state.scrollLogToBottom);

				ImGui::EndTabItem();
			}

			// === Phase 1 Data Tab (VQVDB Loading - Milestone 1.1) ===
			if (ImGui::BeginTabItem("Phase 1: VQVDB Data")) {
				const auto& vol = state.volumeState;

				if (vol.isLoaded && vol.file.has_value()) {
					ImGui::TextColored(ImVec4(0.3f, 0.8f, 0.3f, 1.0f), "VQVDB File Loaded Successfully");
					ImGui::Spacing();
					ImGui::Separator();
					ImGui::Spacing();

					// File info
					ImGui::Text("File: %s", vol.loadedPath.c_str());
					ImGui::Text("Format Version: v%d", static_cast<int>(vol.file->version));
					ImGui::Spacing();

					ImGui::Columns(2, "phase1Cols", true);

					// Block stats
					ImGui::Text("Total Grids:");
					ImGui::Text("Total Blocks:");
					ImGui::Text("Index Data:");
					ImGui::Text("Codebook:");
					ImGui::Text("Voxel Size:");

					ImGui::NextColumn();

					ImGui::Text("%zu", vol.stats.totalGrids);
					ImGui::Text("%zu", vol.stats.totalBlocks);
					ImGui::Text("%.2f KB", static_cast<float>(vol.stats.indexDataBytes) / 1024.0f);
					ImGui::Text("%.2f KB", static_cast<float>(vol.stats.codebookBytes) / 1024.0f);
					ImGui::Text("%.4f", vol.stats.voxelSize);

					ImGui::Columns(1);
					ImGui::Spacing();
					ImGui::Separator();
					ImGui::Spacing();

					// World bounds
					ImGui::Text("World Bounds:");
					const auto& bounds = vol.stats.worldBounds;
					ImGui::Text("  Min: (%.3f, %.3f, %.3f)", bounds.min.x, bounds.min.y, bounds.min.z);
					ImGui::Text("  Max: (%.3f, %.3f, %.3f)", bounds.max.x, bounds.max.y, bounds.max.z);
					const auto size = bounds.size();
					ImGui::Text("  Size: (%.3f, %.3f, %.3f)", size.x, size.y, size.z);
					ImGui::Spacing();

					// Per-grid details
					if (ImGui::TreeNode("Grid Details")) {
						for (size_t i = 0; i < vol.file->grids.size(); ++i) {
							const auto& grid = vol.file->grids[i];
							if (ImGui::TreeNode(reinterpret_cast<void*>(i), "Grid %zu: %s", i, grid.metadata.name.c_str())) {
								ImGui::Text("Blocks: %u", grid.metadata.totalBlocks);
								ImGui::Text("Latent Shape: [%d, %d, %d]", grid.metadata.latentShape[0], grid.metadata.latentShape[1],
								            grid.metadata.latentShape[2]);

								const auto& gridBounds = grid.metadata.worldBounds;
								ImGui::Text("World Bounds:");
								ImGui::Text("  Min: (%.2f, %.2f, %.2f)", gridBounds.min.x, gridBounds.min.y, gridBounds.min.z);
								ImGui::Text("  Max: (%.2f, %.2f, %.2f)", gridBounds.max.x, gridBounds.max.y, gridBounds.max.z);

								ImGui::TreePop();
							}
						}
						ImGui::TreePop();
					}

					// Codebook info
					ImGui::Spacing();
					ImGui::Separator();
					ImGui::Spacing();

					const auto& cb = vol.file->codebook;
					if (cb.empty()) {
						ImGui::TextColored(ImVec4(0.8f, 0.6f, 0.2f, 1.0f), "Codebook: [%d x %d] - Not embedded in file", cb.numEmbeddings,
						                   cb.embeddingDim);
						ImGui::TextWrapped("Note: Codebook must be loaded separately from the model file.");
					} else {
						ImGui::Text("Codebook: [%d x %d] (%.2f KB)", cb.numEmbeddings, cb.embeddingDim,
						            static_cast<float>(cb.sizeBytes()) / 1024.0f);
					}

				} else if (!vol.loadError.empty()) {
					ImGui::TextColored(ImVec4(0.9f, 0.3f, 0.3f, 1.0f), "Load Error:");
					ImGui::TextWrapped("%s", vol.loadError.c_str());
					ImGui::Spacing();
					ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "Use the 'Load VQVDB File...' button to try again.");
				} else {
					ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.3f, 1.0f), "No VQVDB File Loaded");
					ImGui::Spacing();
					ImGui::Separator();
					ImGui::Spacing();
					ImGui::TextWrapped("Load a .vqvdb file using the 'Load VQVDB File...' button in the File section.");
					ImGui::Spacing();
					ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f),
					                   "Milestone 1.1: File loading is implemented. Block visualization coming in Milestone 1.4.");
				}

				ImGui::EndTabItem();
			}

			ImGui::EndTabBar();
		}
	}
	ImGui::End();
}

void renderViewportOverlay(const UIState& state, const CameraState& camera) {
	// Optional: render a small overlay in the viewport corner
	const float padding = 10.0f;
	ImVec2 overlayPos(static_cast<float>(state.viewportX) + padding, static_cast<float>(state.viewportY) + padding);

	ImGui::SetNextWindowPos(overlayPos);
	ImGui::SetNextWindowBgAlpha(0.5f);

	ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings |
	                         ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoMove;

	if (ImGui::Begin("ViewportOverlay", nullptr, flags)) {
		ImGui::Text("Pos: (%.1f, %.1f, %.1f)", camera.position.x, camera.position.y, camera.position.z);
	}
	ImGui::End();
}

}  // namespace

void renderUI(UIState& state, CameraState& camera, CameraLimits& limits, const Timing& timing, int windowWidth, int windowHeight) noexcept {
	// Update performance stats
	updatePerformanceStats(state, timing);

	// Render UI panels
	renderLeftPanel(state, camera, limits, windowHeight);
	renderBottomPanel(state, windowWidth, windowHeight);

	// Compute viewport region (top-right area not covered by panels)
	state.viewportX = static_cast<int>(state.leftPanelWidth);
	state.viewportY = static_cast<int>(state.bottomPanelHeight);
	state.viewportWidth = windowWidth - state.viewportX;
	state.viewportHeight = windowHeight - static_cast<int>(state.bottomPanelHeight);

	// Ensure minimum viewport size
	state.viewportWidth = std::max(state.viewportWidth, 1);
	state.viewportHeight = std::max(state.viewportHeight, 1);

	// Render viewport overlay
	if (state.showCameraInfo) {
		renderViewportOverlay(state, camera);
	}
}

void logMessage(UIState& state, const std::string& message) noexcept {
	// Add timestamp
	auto now = std::chrono::system_clock::now();
	auto time = std::chrono::system_clock::to_time_t(now);
	char timeStr[32];
	std::strftime(timeStr, sizeof(timeStr), "[%H:%M:%S] ", std::localtime(&time));

	state.debugLog.push_back(std::string(timeStr) + message);

	// Limit log size
	while (state.debugLog.size() > UIState::kMaxLogLines) {
		state.debugLog.erase(state.debugLog.begin());
	}
}

bool loadVQVDBFile(UIState& state, const std::string& filePath) noexcept {
	// Clear previous state
	state.volumeState.clear();

	logMessage(state, "Loading VQVDB file: " + filePath);

	// Attempt to load the file
	auto result = vqvdb::loadFile(filePath);

	if (!result.has_value()) {
		state.volumeState.loadError = std::string("Failed to load file: ") + vqvdb::errorToString(result.error());
		logMessage(state, "ERROR: " + state.volumeState.loadError);
		return false;
	}

	// Store the loaded file
	state.volumeState.file = std::move(*result);
	state.volumeState.isLoaded = true;
	state.volumeState.loadedPath = filePath;
	state.volumeState.needsRendererUpdate = true;  // Signal renderer to update block bboxes

	// Compute statistics
	state.volumeState.stats = vqvdb::computeStats(*state.volumeState.file);

	// Update UI state
	state.loadedFilePath = filePath;

	// Log success
	std::stringstream ss;
	ss << "Loaded successfully: " << state.volumeState.stats.totalGrids << " grid(s), " << state.volumeState.stats.totalBlocks << " blocks";
	logMessage(state, ss.str());

	// Print statistics to console as well
	state.volumeState.stats.print();

	return true;
}

bool wantCaptureMouse() noexcept { return ImGui::GetIO().WantCaptureMouse; }

bool wantCaptureKeyboard() noexcept { return ImGui::GetIO().WantCaptureKeyboard; }

}  // namespace ui
