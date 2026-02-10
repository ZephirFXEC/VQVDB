#include "ui/ui.hpp"

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <imgui_internal.h>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <exception>
#include <filesystem>
#include <format>
#include <numeric>
#include <sstream>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__linux__)
#include <unistd.h>
#endif

#include "core/camera.hpp"
#include "core/profiler.hpp"
#include "core/types.hpp"
#include "ui/profiler_ui.hpp"
#include "vqvdb/gpu_resources.hpp"
#include "vqvdb/vqvdb_loader.hpp"

namespace ui {

bool init(GLFWwindow* window) noexcept {
	// Create ImGui context
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();

	ImGuiIO& io = ImGui::GetIO();
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
	io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
	io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;

	// Use an explicitly configured default font atlas for crisper text.
	io.Fonts->Clear();
	ImFontConfig fontConfig;
	fontConfig.OversampleH = 2;
	fontConfig.OversampleV = 2;
	fontConfig.PixelSnapH = false;
	fontConfig.SizePixels = 13.0f;
	io.Fonts->AddFontDefault(&fontConfig);

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
	style.AntiAliasedFill = true;

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

	if ((io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) != 0) {
		style.WindowRounding = 2.0f;
		style.Colors[ImGuiCol_WindowBg].w = 1.0f;
	}

	// Initialize platform/renderer backends
	// Pass false to NOT install callbacks - we'll forward events manually
	if (!ImGui_ImplGlfw_InitForOpenGL(window, false)) {
		return false;
	}

	if (!ImGui_ImplOpenGL3_Init("#version 460")) {
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

	ImGuiIO& io = ImGui::GetIO();
	if ((io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) != 0) {
		GLFWwindow* backupCurrentContext = glfwGetCurrentContext();
		ImGui::UpdatePlatformWindows();
		ImGui::RenderPlatformWindowsDefault();
		glfwMakeContextCurrent(backupCurrentContext);
	}
}

namespace {

void beginDockspace() {
	ImGuiViewport* viewport = ImGui::GetMainViewport();
	ImGui::SetNextWindowPos(viewport->Pos);
	ImGui::SetNextWindowSize(viewport->Size);
	ImGui::SetNextWindowViewport(viewport->ID);

	ImGuiWindowFlags hostFlags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize |
	                             ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus |
	                             ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoDocking;

	ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
	ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
	ImGui::Begin("MainDockspaceHost", nullptr, hostFlags);
	ImGui::PopStyleVar(2);

	const ImGuiID dockspaceId = ImGui::GetID("MainDockspace");
	ImGui::DockSpace(dockspaceId, ImVec2(0.0f, 0.0f), ImGuiDockNodeFlags_None);
	ImGui::End();
}

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

// --- UI helpers focused on compact, status-first presentation ---

ImVec4 colorOk() { return {0.30f, 0.80f, 0.30f, 1.0f}; }
ImVec4 colorWarn() { return {0.85f, 0.65f, 0.25f, 1.0f}; }
ImVec4 colorError() { return {0.90f, 0.30f, 0.30f, 1.0f}; }
ImVec4 colorMuted() { return {0.65f, 0.65f, 0.70f, 1.0f}; }

constexpr const char* kDefaultDecoderModelPath = "C:/Users/zphrfx/Desktop/hdk/VQVDB/models/onnx_models/decoder_opt.onnx";
constexpr uint32_t kDefaultDecoderMaxBatchSize = 4096;

std::filesystem::path executableDirectory() {
#if defined(_WIN32)
	char modulePath[MAX_PATH] = {};
	const DWORD length = GetModuleFileNameA(nullptr, modulePath, MAX_PATH);
	if (length > 0 && length < MAX_PATH) {
		return std::filesystem::path(modulePath).parent_path();
	}
#elif defined(__linux__)
	std::array<char, 4096> modulePath{};
	const ssize_t length = readlink("/proc/self/exe", modulePath.data(), modulePath.size() - 1);
	if (length > 0) {
		modulePath[static_cast<size_t>(length)] = '\0';
		return std::filesystem::path(modulePath.data()).parent_path();
	}
#endif
	return std::filesystem::current_path();
}

std::filesystem::path resolveDecoderModelPath(const std::string& configuredPath) {
	const std::filesystem::path rawPath =
	    configuredPath.empty() ? std::filesystem::path(kDefaultDecoderModelPath) : std::filesystem::path(configuredPath);
	return rawPath.is_relative() ? (executableDirectory() / rawPath) : rawPath;
}

DecoderInitTaskResult runDecoderInitTask(const std::filesystem::path& resolvedPath) {
	DecoderInitTaskResult result{};
	std::filesystem::path readyPath = resolvedPath;

	auto backend = std::make_unique<vqvdb::DecoderBackend>();
	auto initResult = backend->init(resolvedPath, kDefaultDecoderMaxBatchSize);
	std::string primaryFailureDetails;
	if (!initResult.has_value()) {
		primaryFailureDetails = backend->lastErrorMessage();
	}

	// Fallback: if the default decoder fails to build, try decoder_opt.onnx in the same folder.
	if (!initResult.has_value() && resolvedPath.filename() == "decoder.onnx") {
		const std::filesystem::path fallbackPath = resolvedPath.parent_path() / "decoder_opt.onnx";
		if (std::filesystem::exists(fallbackPath)) {
			backend = std::make_unique<vqvdb::DecoderBackend>();
			initResult = backend->init(fallbackPath, kDefaultDecoderMaxBatchSize);
			if (initResult.has_value()) {
				readyPath = fallbackPath;
			}
		}
	}

	if (!initResult.has_value()) {
		result.success = false;
		result.errorMessage = std::format("Decoder init failed: {}", vqvdb::errorToString(initResult.error()));
		if (!primaryFailureDetails.empty()) {
			result.errorMessage += std::format(" | primary: {}", primaryFailureDetails);
		}
		const std::string details = backend->lastErrorMessage();
		if (!details.empty()) {
			result.errorMessage += std::format(" | details: {}", details);
		}
		return result;
	}

	result.success = true;
	result.readyModelPath = readyPath.string();
	result.backend = std::move(backend);
	return result;
}

std::string formatKB(size_t bytes) {
	const float kb = static_cast<float>(bytes) / 1024.0f;
	return std::format("{:.2f} KB", kb);
}

void tableStatusRow(const char* label, const char* value, const ImVec4& color) {
	ImGui::TableNextRow();
	ImGui::TableSetColumnIndex(0);
	ImGui::TextColored(color, "● %s", label);
	ImGui::TableSetColumnIndex(1);
	ImGui::TextUnformatted(value);
}

void tableStatRow(const char* label, const char* value) {
	ImGui::TableNextRow();
	ImGui::TableSetColumnIndex(0);
	ImGui::TextUnformatted(label);
	ImGui::TableSetColumnIndex(1);
	ImGui::TextUnformatted(value);
}

void sectionHeader(const char* label) {
	ImGui::Separator();
	ImGui::TextUnformatted(label);
}

ImU32 heatColor(float t) {
	t = std::clamp(t, 0.0f, 1.0f);
	// Cool -> warm ramp: blue -> cyan -> yellow -> red.
	const ImVec4 c0{0.10f, 0.18f, 0.35f, 1.0f};
	const ImVec4 c1{0.15f, 0.55f, 0.75f, 1.0f};
	const ImVec4 c2{0.92f, 0.78f, 0.25f, 1.0f};
	const ImVec4 c3{0.88f, 0.25f, 0.20f, 1.0f};
	ImVec4 c{};
	if (t < 0.33f) {
		const float u = t / 0.33f;
		c = ImLerp(c0, c1, u);
	} else if (t < 0.66f) {
		const float u = (t - 0.33f) / 0.33f;
		c = ImLerp(c1, c2, u);
	} else {
		const float u = (t - 0.66f) / 0.34f;
		c = ImLerp(c2, c3, u);
	}
	return ImGui::GetColorU32(c);
}

void renderBrickCacheHeatmap(UIState& state) {
	auto& gpu = state.gpuState;
	if (!gpu.brickCacheInitialized) {
		ImGui::TextColored(colorWarn(), "Brick cache is not initialized");
		return;
	}

	const vqvdb::BrickCacheDebugSnapshot snapshot = vqvdb::buildBrickCacheDebugSnapshot(gpu.brickCache);
	if (snapshot.capacitySlots == 0 || snapshot.slotGridDims.x <= 0 || snapshot.slotGridDims.y <= 0 || snapshot.slotGridDims.z <= 0) {
		ImGui::TextColored(colorWarn(), "Brick cache has no slots");
		return;
	}

	const vqvdb::BrickCacheStats stats = vqvdb::getCacheStats(gpu.brickCache);
	ImGui::Text(
	    "Resident: %u / %u  (%.1f%%)", stats.residentBricks, stats.capacitySlots,
	    (stats.capacitySlots > 0) ? (100.0f * static_cast<float>(stats.residentBricks) / static_cast<float>(stats.capacitySlots)) : 0.0f);
	ImGui::Text("Lookups: %llu  Hits: %llu  Hit rate: %.1f%%  Evictions: %llu", static_cast<unsigned long long>(stats.lookupCount),
	            static_cast<unsigned long long>(stats.hitCount), 100.0f * stats.hitRate,
	            static_cast<unsigned long long>(stats.evictionCount));

	const char* modes[] = {"Least Used (Age)", "Least Used (Touches)", "LRU Tail Rank"};
	gpu.brickCacheHeatmapMode = std::clamp(gpu.brickCacheHeatmapMode, 0, 2);
	ImGui::Combo("Heatmap Mode", &gpu.brickCacheHeatmapMode, modes, IM_ARRAYSIZE(modes));

	gpu.brickCacheHeatmapSlice = std::clamp(gpu.brickCacheHeatmapSlice, 0, snapshot.slotGridDims.z - 1);
	if (snapshot.slotGridDims.z > 1) {
		ImGui::SliderInt("Z Slice", &gpu.brickCacheHeatmapSlice, 0, snapshot.slotGridDims.z - 1);
	}

	const int sx = snapshot.slotGridDims.x;
	const int sy = snapshot.slotGridDims.y;
	const int z = gpu.brickCacheHeatmapSlice;

	ImGui::TextColored(colorMuted(), "Slice dimensions: %d x %d (z=%d)", sx, sy, z);

	const ImVec2 avail = ImGui::GetContentRegionAvail();
	const float cellW = std::max(4.0f, std::min(16.0f, (avail.x - 4.0f) / static_cast<float>(sx)));
	const float cellH = cellW;
	const float pad = 1.0f;
	const ImVec2 start = ImGui::GetCursorScreenPos();
	const ImVec2 canvasSize(cellW * static_cast<float>(sx), cellH * static_cast<float>(sy));
	ImGui::InvisibleButton("brickCacheHeatmapCanvas", canvasSize);
	ImDrawList* draw = ImGui::GetWindowDrawList();
	draw->AddRectFilled(start, ImVec2(start.x + canvasSize.x, start.y + canvasSize.y),
	                    ImGui::GetColorU32(ImVec4(0.07f, 0.07f, 0.09f, 1.0f)));
	draw->AddRect(start, ImVec2(start.x + canvasSize.x, start.y + canvasSize.y), ImGui::GetColorU32(ImVec4(0.25f, 0.25f, 0.3f, 1.0f)));

	std::optional<uint32_t> hoveredSlot;
	const ImVec2 mouse = ImGui::GetIO().MousePos;

	const int slotsPerLayer = sx * sy;
	for (int y = 0; y < sy; ++y) {
		for (int x = 0; x < sx; ++x) {
			const uint32_t slot = static_cast<uint32_t>(z * slotsPerLayer + y * sx + x);
			if (slot >= snapshot.slots.size()) continue;
			const auto& info = snapshot.slots[slot];

			float norm = 0.0f;
			if (info.occupied) {
				switch (gpu.brickCacheHeatmapMode) {
					case 0: {
						const uint64_t age =
						    (snapshot.accessCounter >= info.lastAccessCounter) ? (snapshot.accessCounter - info.lastAccessCounter) : 0ull;
						norm = (snapshot.maxAge > 0) ? static_cast<float>(age) / static_cast<float>(snapshot.maxAge) : 0.0f;
						break;
					}
					case 1: {
						norm = (snapshot.maxTouchCount > 0)
						           ? (1.0f - static_cast<float>(info.touchCount) / static_cast<float>(snapshot.maxTouchCount))
						           : 0.0f;
						break;
					}
					case 2: {
						norm = (snapshot.residentBricks > 1 && info.lruRank != 0xFFFFFFFFu)
						           ? static_cast<float>(info.lruRank) / static_cast<float>(snapshot.residentBricks - 1u)
						           : 0.0f;
						break;
					}
				}
			}

			const ImU32 color = info.occupied ? heatColor(norm) : ImGui::GetColorU32(ImVec4(0.12f, 0.12f, 0.14f, 1.0f));
			const ImVec2 p0(start.x + static_cast<float>(x) * cellW + pad, start.y + static_cast<float>(y) * cellH + pad);
			const ImVec2 p1(start.x + static_cast<float>(x + 1) * cellW - pad, start.y + static_cast<float>(y + 1) * cellH - pad);
			draw->AddRectFilled(p0, p1, color);

			if (mouse.x >= p0.x && mouse.x <= p1.x && mouse.y >= p0.y && mouse.y <= p1.y) {
				hoveredSlot = slot;
			}
		}
	}

	if (hoveredSlot.has_value() && hoveredSlot.value() < snapshot.slots.size()) {
		const uint32_t slot = hoveredSlot.value();
		const auto& info = snapshot.slots[slot];
		ImGui::BeginTooltip();
		ImGui::Text("Slot: %u", slot);
		ImGui::Text("Occupied: %s", info.occupied ? "yes" : "no");
		if (info.occupied) {
			ImGui::Text("Morton: %llu", static_cast<unsigned long long>(info.mortonCode));
			ImGui::Text("Touches: %u", info.touchCount);
			const uint64_t age =
			    (snapshot.accessCounter >= info.lastAccessCounter) ? (snapshot.accessCounter - info.lastAccessCounter) : 0ull;
			ImGui::Text("Age: %llu", static_cast<unsigned long long>(age));
			if (info.lruRank != 0xFFFFFFFFu) {
				ImGui::Text("LRU rank: %u (0 = MRU)", info.lruRank);
			}
		}
		ImGui::EndTooltip();
	}
}

void renderLeftPanel(UIState& state, CameraState& camera, CameraLimits& limits) {
	ImGui::SetNextWindowSize(ImVec2(state.leftPanelWidth, 720.0f), ImGuiCond_FirstUseEver);
	if (ImGui::Begin("Control Panel")) {
		// === File / Data Section ===
		if (ImGui::CollapsingHeader("Data & Files", ImGuiTreeNodeFlags_DefaultOpen)) {
			const bool fileLoaded = state.volumeState.isLoaded;
			ImGui::TextColored(fileLoaded ? colorOk() : colorWarn(), fileLoaded ? "VQVDB loaded" : "No VQVDB file");
			ImGui::TextWrapped("%s", state.loadedFilePath.c_str());

			if (ImGui::Button("Load VQVDB File...", ImVec2(-1, 0))) {
				state.fileLoadRequested = true;
			}

			sectionHeader("Blocks");
			const bool blocksOnGpu = state.gpuState.blockIndicesUploaded;
			ImGui::TextColored(blocksOnGpu ? colorOk() : colorWarn(), blocksOnGpu ? "GPU block data uploaded" : "Blocks not on GPU");
			if (ImGui::Button(blocksOnGpu ? "Re-upload Blocks" : "Upload Blocks to GPU", ImVec2(-1, 0))) {
				state.gpuState.blockDataUploadRequested = true;
			}
			if (blocksOnGpu && (!state.gpuState.blockIndicesVerified || !state.gpuState.blockMetadataVerified)) {
				if (ImGui::Button("Verify Block Data", ImVec2(-1, 0))) {
					state.gpuState.blockDataVerifyRequested = true;
				}
			}
			if (!state.gpuState.blockDataError.empty()) {
				ImGui::TextColored(colorError(), "Block upload error:");
				ImGui::TextWrapped("%s", state.gpuState.blockDataError.c_str());
			}

			ImGui::Spacing();
			ImGui::TextColored(colorMuted(), "Detailed GPU debug is on the right panel.");
			ImGui::Separator();
		}

		// === Camera section (compact, collapsed by default) ===
		if (ImGui::CollapsingHeader("Camera & View")) {
			ImGui::Text("Pos: (%.1f, %.1f, %.1f)", camera.position.x, camera.position.y, camera.position.z);
			ImGui::Text("Yaw/Pitch: %.1f / %.1f deg", camera.yawDegrees, camera.pitchDegrees);
			ImGui::Text("Dist: %.2f | FOV: %.0f°", camera.distance, camera.fovDegrees);
			ImGui::Spacing();

			ImGui::SliderFloat("FOV", &camera.fovDegrees, 30.0f, 120.0f, "%.0f deg");

			if (ImGui::TreeNode("Projection & Clipping")) {
				ImGui::SliderFloat("Near", &camera.nearPlane, 0.01f, 1.0f, "%.3f");
				ImGui::SliderFloat("Far", &camera.farPlane, 100.0f, 10000.0f, "%.0f");
				ImGui::TreePop();
			}

			if (ImGui::TreeNode("Movement Tuning")) {
				ImGui::SliderFloat("Move Speed", &limits.moveSpeed, 1.0f, 100.0f);
				ImGui::SliderFloat("Orbit Sens.", &limits.orbitSensitivity, 0.05f, 1.0f);
				ImGui::SliderFloat("Zoom Sens.", &limits.zoomSensitivity, 0.05f, 0.5f);
				ImGui::TreePop();
			}

			if (ImGui::Button("Reset Camera", ImVec2(-1, 0))) {
				camera.yawDegrees = 0.0f;
				camera.pitchDegrees = 20.0f;
				camera.distance = 5.0f;
				camera.target = glm::vec3(0.0f);
				camera.velocity = glm::vec3(0.0f);
			}
			ImGui::Separator();
		}

		// === Overlays / visibility ===
		if (ImGui::CollapsingHeader("Panels & Overlays", ImGuiTreeNodeFlags_DefaultOpen)) {
			ImGui::Checkbox("Show Camera Overlay", &state.showCameraInfo);
			ImGui::Checkbox("Show Performance Tab", &state.showPerformance);
			ImGui::Checkbox("Show Profiler Tab", &state.showProfiler);
			ImGui::Checkbox("Show Debug Log Tab", &state.showDebugLog);
			ImGui::Checkbox("Show Dataset Tab", &state.showPhase1Data);
			ImGui::Checkbox("Enable Profiler Instrumentation", &state.profilerEnabled);
			ImGui::Separator();
		}

		// === Shortcuts ===
		if (ImGui::CollapsingHeader("Controls / Shortcuts")) {
			ImGui::BulletText("LMB + Drag: Orbit camera");
			ImGui::BulletText("Scroll: Zoom in/out");
			ImGui::BulletText("W/S: Move forward/back");
			ImGui::BulletText("A/D: Move left/right");
			ImGui::BulletText("Q/E: Move down/up");
		}
	}
	state.leftPanelWidth = std::max(220.0f, ImGui::GetWindowWidth());
	ImGui::End();
}

void renderGpuDebugPanel(UIState& state) {
	ImGui::SetNextWindowSize(ImVec2(state.rightPanelWidth, 720.0f), ImGuiCond_FirstUseEver);
	if (ImGui::Begin("GPU Debug")) {
		auto& gpu = state.gpuState;
		const auto memStats = vqvdb::getGPUMemoryStats(gpu.resources);
		const bool blocksReady = gpu.blockIndicesUploaded;
		const bool blocksVerified = gpu.blockIndicesVerified && gpu.blockIndicesVerification.passed;
		const bool metadataVerified = !gpu.blockMetadataUploaded || (gpu.blockMetadataVerified && gpu.blockMetadataVerification.passed);

		sectionHeader("Summary");
		if (ImGui::BeginTable("gpuSummary", 2, ImGuiTableFlags_SizingStretchSame)) {
			tableStatusRow("Blocks", blocksReady ? (blocksVerified && metadataVerified ? "Verified" : "Uploaded") : "Missing",
			               blocksReady ? (blocksVerified && metadataVerified ? colorOk() : colorWarn()) : colorError());
			tableStatusRow("Renderer", gpu.rendererNeedsUpdate ? "Pending update" : "Ready",
			               gpu.rendererNeedsUpdate ? colorWarn() : colorOk());
			ImGui::EndTable();
		}

		sectionHeader("Memory");
		if (ImGui::BeginTable("gpuMemory", 2, ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_SizingStretchSame)) {
			tableStatRow("Block indices", formatKB(memStats.blockIndicesBytes).c_str());
			tableStatRow("Block origins", formatKB(memStats.blockOriginsBytes).c_str());
			tableStatRow("Block metadata", formatKB(memStats.blockMetadataBytes).c_str());
			tableStatRow("Decoder weights", formatKB(memStats.decoderWeightsBytes).c_str());
			tableStatRow("Total", formatKB(memStats.totalBytes).c_str());
			ImGui::EndTable();
		}

		sectionHeader("Blocks");
		ImGui::Text("Blocks on GPU: %zu", gpu.resources.numBlocks);
		ImGui::Text("Voxel size: %.4f | Block size: %.1f", gpu.voxelSize, gpu.blockSize);

		if (ImGui::Button(blocksReady ? "Re-upload Blocks" : "Upload Blocks to GPU", ImVec2(-1, 0))) {
			gpu.blockDataUploadRequested = true;
		}
		if (blocksReady && (!gpu.blockIndicesVerified || !gpu.blockMetadataVerified)) {
			if (ImGui::Button("Verify Block Data", ImVec2(-1, 0))) {
				gpu.blockDataVerifyRequested = true;
			}
		}
		if (gpu.blockIndicesVerified) {
			ImGui::TextColored(blocksVerified ? colorOk() : colorError(), "Indices: %s", blocksVerified ? "PASSED" : "FAILED");
			if (!gpu.blockIndicesVerification.message.empty()) {
				ImGui::TextWrapped("%s", gpu.blockIndicesVerification.message.c_str());
			}
		}
		if (gpu.blockMetadataUploaded && gpu.blockMetadataVerified) {
			ImGui::TextColored(metadataVerified ? colorOk() : colorError(), "Metadata: %s", metadataVerified ? "PASSED" : "FAILED");
			if (!gpu.blockMetadataVerification.message.empty()) {
				ImGui::TextWrapped("%s", gpu.blockMetadataVerification.message.c_str());
			}
		}
		if (!gpu.blockDataError.empty()) {
			ImGui::TextColored(colorError(), "%s", gpu.blockDataError.c_str());
		}

		// Visualization tuning
		sectionHeader("Block Visualization");
		ImGui::Checkbox("Limit displayed blocks", &gpu.useBlockLimit);
		ImGui::Checkbox("Color by visibility/cache state", &gpu.colorBlocksByVisibility);
		if (gpu.resources.numBlocks == 0) {
			ImGui::TextColored(colorMuted(), "No blocks uploaded yet");
		} else if (gpu.useBlockLimit) {
			const int totalBlocks = static_cast<int>(gpu.resources.numBlocks);
			gpu.maxDisplayBlocks = std::clamp(gpu.maxDisplayBlocks, 1, totalBlocks);
			ImGui::SliderInt("Max blocks", &gpu.maxDisplayBlocks, 1, totalBlocks);
			ImGui::Text("Showing %d / %d blocks", gpu.maxDisplayBlocks, totalBlocks);
		} else {
			ImGui::Text("Showing all %zu blocks", gpu.resources.numBlocks);
		}

		sectionHeader("Decoder");
		gpu.decoderReady = gpu.decoderBackend != nullptr && gpu.decoderBackend->isReady();
		ImGui::TextWrapped("Model path: %s", gpu.decoderModelPath.c_str());
		if (gpu.decoderInitInProgress) {
			ImGui::BeginDisabled();
		}
		if (ImGui::Button(gpu.decoderReady ? "Reinitialize Decoder" : "Initialize Decoder", ImVec2(-1, 0))) {
			gpu.decoderInitRequested = true;
		}
		if (gpu.decoderInitInProgress) {
			ImGui::EndDisabled();
		}

		const ImVec4 decoderStatusColor = gpu.decoderInitInProgress ? colorMuted() : (gpu.decoderReady ? colorOk() : colorWarn());
		const char* decoderSummary = gpu.decoderInitInProgress ? "initializing" : (gpu.decoderReady ? "ready" : "not ready");
		ImGui::TextColored(decoderStatusColor, "Decoder status: %s", decoderSummary);
		if (!gpu.decoderStatus.empty()) {
			ImGui::TextWrapped("%s", gpu.decoderStatus.c_str());
		}
		if (!gpu.decoderError.empty()) {
			ImGui::TextColored(colorError(), "%s", gpu.decoderError.c_str());
		}

		sectionHeader("Visibility Scheduler");
		ImGui::Checkbox("Enable frustum culling", &gpu.enableFrustumCulling);
		ImGui::Checkbox("Enable depth occlusion (Hi-Z)", &gpu.enableDepthOcclusion);
		ImGui::Checkbox("Auto update cache from visible set", &gpu.autoUpdateVisibleCache);
		ImGui::InputInt("Decode budget / frame", &gpu.decodeBudgetPerFrame);
		gpu.decodeBudgetPerFrame = std::max(0, gpu.decodeBudgetPerFrame);
		ImGui::InputFloat("Max decode distance", &gpu.maxDecodeDistance, 10.0f, 100.0f, "%.1f");
		gpu.maxDecodeDistance = std::max(0.0f, gpu.maxDecodeDistance);
		ImGui::InputFloat("Occlusion depth bias", &gpu.occlusionDepthBias, 0.0001f, 0.001f, "%.5f");
		gpu.occlusionDepthBias = std::max(0.0f, gpu.occlusionDepthBias);
		if (gpu.maxDecodeDistance <= 0.0f) {
			ImGui::TextColored(colorMuted(), "Distance limit: disabled");
		}
		ImGui::Text("Visible: %u (cached=%u, missing=%u)", gpu.visibleBlocksLastFrame, gpu.visibleCachedLastFrame,
		            gpu.visibleMissingLastFrame);
		ImGui::Text("Scheduled decodes: %u (occluded filtered=%u)", gpu.scheduledDecodesLastFrame, gpu.occludedRequestsLastFrame);
		ImGui::Text("Decoded this frame: %u", gpu.decodedBlocksLastFrame);
		ImGui::Text("Cache update: touched=%u, inserted=%u, evicted=%u", gpu.cacheTouchedLastFrame, gpu.cacheInsertedLastFrame,
		            gpu.cacheEvictedLastFrame);
		if (!gpu.schedulerError.empty()) {
			ImGui::TextColored(colorError(), "%s", gpu.schedulerError.c_str());
		}

		sectionHeader("Brick Cache");
		ImGui::InputInt("Cache capacity (slots)", &gpu.brickCacheCapacity);
		gpu.brickCacheCapacity = std::max(1, gpu.brickCacheCapacity);
		ImGui::Checkbox("Allocate atlas texture", &gpu.brickCacheAllocateTexture);
		if (ImGui::Button("Recreate Cache", ImVec2(-1, 0))) {
			gpu.brickCacheReinitRequested = true;
		}
		if (ImGui::Button("Clear Cache", ImVec2(-1, 0))) {
			gpu.brickCacheClearRequested = true;
		}

		const bool canPrime = state.volumeState.isLoaded && state.volumeState.file.has_value() && !state.volumeState.file->grids.empty();
		ImGui::InputInt("Prime count", &gpu.brickCachePrimeCount);
		gpu.brickCachePrimeCount = std::max(1, gpu.brickCachePrimeCount);
		if (!canPrime) {
			ImGui::BeginDisabled();
		}
		if (ImGui::Button("Prime Cache From Loaded Blocks", ImVec2(-1, 0))) {
			gpu.brickCachePrimeRequested = true;
		}
		if (!canPrime) {
			ImGui::EndDisabled();
			ImGui::TextColored(colorMuted(), "Load a VQVDB file first to prime cache.");
		}

		if (!gpu.brickCacheError.empty()) {
			ImGui::TextColored(colorError(), "%s", gpu.brickCacheError.c_str());
		}
		renderBrickCacheHeatmap(state);

		sectionHeader("Troubleshooting");
		if (!state.volumeState.loadError.empty()) {
			ImGui::TextColored(colorWarn(), "Last load error:");
			ImGui::TextWrapped("%s", state.volumeState.loadError.c_str());
		}
		if (!gpu.blockDataError.empty()) {
			ImGui::TextColored(colorWarn(), "GPU errors are also shown above.");
		}
	}
	state.rightPanelWidth = std::max(280.0f, ImGui::GetWindowWidth());
	ImGui::End();
}

void renderBottomPanel(UIState& state) {
	ImGui::SetNextWindowSize(ImVec2(1000.0f, state.bottomPanelHeight), ImGuiCond_FirstUseEver);
	if (ImGui::Begin("Info Panel")) {
		// Use tabs for different sections
		if (ImGui::BeginTabBar("BottomTabs")) {
			// === Performance Tab ===
			if (state.showPerformance && ImGui::BeginTabItem("Performance")) {
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

			// === Profiler Tab ===
			if (state.showProfiler && ImGui::BeginTabItem("Profiler")) {
				profiler_ui::renderProfilerTab(state);
				ImGui::EndTabItem();
			}

			// === Debug Log Tab ===
			if (state.showDebugLog && ImGui::BeginTabItem("Debug Log")) {
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

			// === Dataset Tab (compact) ===
			if (state.showPhase1Data && ImGui::BeginTabItem("Dataset")) {
				const auto& vol = state.volumeState;

				if (vol.isLoaded && vol.file.has_value()) {
					ImGui::TextColored(colorOk(), "VQVDB File Loaded");

					if (ImGui::BeginTable("datasetSummary", 2, ImGuiTableFlags_SizingStretchSame)) {
						tableStatRow("File", vol.loadedPath.c_str());
						tableStatRow("Version", std::format("v{}", static_cast<int>(vol.file->version)).c_str());
						tableStatRow("Grids", std::format("{}", vol.stats.totalGrids).c_str());
						tableStatRow("Blocks", std::format("{}", vol.stats.totalBlocks).c_str());
						tableStatRow("Voxel size", std::format("{:.4f}", vol.stats.voxelSize).c_str());
						ImGui::EndTable();
					}

					const auto& bounds = vol.stats.worldBounds;
					ImGui::Text("World Bounds");
					ImGui::Text("Min (%.3f, %.3f, %.3f)", bounds.min.x, bounds.min.y, bounds.min.z);
					ImGui::Text("Max (%.3f, %.3f, %.3f)", bounds.max.x, bounds.max.y, bounds.max.z);

					if (ImGui::TreeNode("Grid Details")) {
						for (size_t i = 0; i < vol.file->grids.size(); ++i) {
							const auto& grid = vol.file->grids[i];
							if (ImGui::TreeNode(reinterpret_cast<void*>(i), "Grid %zu: %s", i, grid.metadata.name.c_str())) {
								if (ImGui::BeginTable("gridTable", 2, ImGuiTableFlags_SizingStretchSame)) {
									tableStatRow("Blocks", std::format("{}", grid.metadata.totalBlocks).c_str());
									tableStatRow("Latent shape", std::format("[%d, %d, %d]", grid.metadata.latentShape[0],
									                                         grid.metadata.latentShape[1], grid.metadata.latentShape[2])
									                                 .c_str());
									ImGui::EndTable();
								}

								const auto& gridBounds = grid.metadata.worldBounds;
								ImGui::Text("Bounds Min (%.2f, %.2f, %.2f)", gridBounds.min.x, gridBounds.min.y, gridBounds.min.z);
								ImGui::Text("Bounds Max (%.2f, %.2f, %.2f)", gridBounds.max.x, gridBounds.max.y, gridBounds.max.z);
								ImGui::TreePop();
							}
						}
						ImGui::TreePop();
					}

				} else if (!vol.loadError.empty()) {
					ImGui::TextColored(colorError(), "Load Error:");
					ImGui::TextWrapped("%s", vol.loadError.c_str());
					ImGui::Spacing();
					ImGui::TextColored(colorMuted(), "Use the 'Load VQVDB File...' button to try again.");
				} else {
					ImGui::TextColored(colorWarn(), "No VQVDB file loaded.");
					ImGui::Spacing();
					ImGui::TextWrapped("Load a .vqvdb file using the left panel. GPU debug remains available for previous resources.");
				}

				ImGui::EndTabItem();
			}

			ImGui::EndTabBar();
		}
	}
	state.bottomPanelHeight = std::max(140.0f, ImGui::GetWindowHeight());
	ImGui::End();
}

void renderViewportWindow(UIState& state) {
	ImGui::SetNextWindowSize(ImVec2(1000.0f, 700.0f), ImGuiCond_FirstUseEver);
	ImGuiWindowClass viewportClass;
	viewportClass.ParentViewportId = ImGui::GetMainViewport()->ID;
	ImGui::SetNextWindowClass(&viewportClass);
	ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
	constexpr ImGuiWindowFlags viewportFlags = ImGuiWindowFlags_NoBackground;
	if (ImGui::Begin("Viewport", nullptr, viewportFlags)) {
		state.viewportHovered = ImGui::IsWindowHovered();
		state.viewportFocused = ImGui::IsWindowFocused();

		const ImVec2 contentMin = ImGui::GetWindowContentRegionMin();
		const ImVec2 contentMax = ImGui::GetWindowContentRegionMax();
		const ImVec2 pos = ImGui::GetWindowPos();
		const ImVec2 mainPos = ImGui::GetMainViewport()->Pos;

		// Keep render viewport coordinates in main-window framebuffer space.
		state.viewportX = static_cast<int>(pos.x + contentMin.x - mainPos.x);
		state.viewportY = static_cast<int>(pos.y + contentMin.y - mainPos.y);
		state.viewportWidth = std::max(1, static_cast<int>(contentMax.x - contentMin.x));
		state.viewportHeight = std::max(1, static_cast<int>(contentMax.y - contentMin.y));
	} else {
		state.viewportHovered = false;
		state.viewportFocused = false;
	}
	ImGui::End();
	ImGui::PopStyleVar();
}

void renderViewportOverlay(const UIState& state, const CameraState& camera) {
	// Optional: render a small overlay in the viewport corner
	const float padding = 10.0f;
	const ImVec2 mainPos = ImGui::GetMainViewport()->Pos;
	ImVec2 overlayPos(mainPos.x + static_cast<float>(state.viewportX) + padding, mainPos.y + static_cast<float>(state.viewportY) + padding);

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
	(void)windowWidth;
	(void)windowHeight;

	// Update performance stats
	updatePerformanceStats(state, timing);

	beginDockspace();

	// Render UI panels
	renderViewportWindow(state);
	renderLeftPanel(state, camera, limits);
	renderGpuDebugPanel(state);
	renderBottomPanel(state);

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

	// New dataset invalidates previous cache contents.
	if (state.gpuState.brickCacheInitialized) {
		clearBrickCache(state);
	}

	return true;
}

bool uploadBlockDataToGPU(UIState& state) noexcept {
	auto& gpu = state.gpuState;
	const auto& vol = state.volumeState;

	if (!vol.isLoaded || !vol.file.has_value() || vol.file->grids.empty()) {
		gpu.blockDataError = "No VQVDB file loaded";
		return false;
	}

	// Upload indices from first grid (for now)
	const auto& grid = vol.file->grids[0];
	const auto& blocks = grid.blocks;

	logMessage(state, std::format("Uploading {} blocks ({} bytes) to GPU...", blocks.count(), blocks.indices.size()));

	// Upload indices
	vqvdb::GPUResult<void> indicesResult;
	{
		TRANSFER_PROFILE_SCOPE(state.profiler, "Transfer Block Indices CPU->GPU", blocks.indices.size(),
		                       profiler::TransferDirection::CPUToGPU);
		indicesResult = vqvdb::uploadBlockIndices(gpu.resources, blocks);
	}
	if (!indicesResult.has_value()) {
		gpu.blockDataError = std::format("Indices upload failed: {}", vqvdb::errorToString(indicesResult.error()));
		logMessage(state, "ERROR: " + gpu.blockDataError);
		return false;
	}

	// Upload origins
	const size_t paddedOriginBytes = blocks.origins.size() * sizeof(int32_t) * 4;
	vqvdb::GPUResult<void> originsResult;
	{
		TRANSFER_PROFILE_SCOPE(state.profiler, "Transfer Block Origins CPU->GPU", paddedOriginBytes, profiler::TransferDirection::CPUToGPU);
		originsResult = vqvdb::uploadBlockOrigins(gpu.resources, blocks);
	}
	if (!originsResult.has_value()) {
		gpu.blockDataError = std::format("Origins upload failed: {}", vqvdb::errorToString(originsResult.error()));
		logMessage(state, "ERROR: " + gpu.blockDataError);
		return false;
	}

	// Upload block metadata with morton codes (Milestone 1.3)
	const size_t metadataBytes = blocks.origins.size() * sizeof(vqvdb::BlockMetadata);
	vqvdb::GPUResult<void> metadataResult;
	{
		TRANSFER_PROFILE_SCOPE(state.profiler, "Transfer Block Metadata CPU->GPU", metadataBytes, profiler::TransferDirection::CPUToGPU);
		metadataResult = vqvdb::uploadBlockMetadata(gpu.resources, blocks);
	}
	if (!metadataResult.has_value()) {
		gpu.blockDataError = std::format("Metadata upload failed: {}", vqvdb::errorToString(metadataResult.error()));
		logMessage(state, "ERROR: " + gpu.blockDataError);
		return false;
	}
	gpu.blockMetadataUploaded = true;

	// Store grid transform info for GPU instanced rendering
	gpu.voxelSize = grid.metadata.transform.voxelSize();
	gpu.blockSize = static_cast<float>(vqvdb::kBlockSize);
	gpu.gridTransform = grid.metadata.transform.toMat4();
	gpu.rendererNeedsUpdate = true;

	// Initialize block display limit to total blocks
	gpu.maxDisplayBlocks = static_cast<int>(blocks.count());

	gpu.blockIndicesUploaded = true;
	gpu.blockIndicesVerified = false;
	gpu.blockMetadataVerified = false;
	gpu.blockDataError.clear();

	logMessage(state, std::format("Block data uploaded to GPU (voxelSize={:.4f}, {} morton codes)", gpu.voxelSize, blocks.count()));

	return true;
}

void verifyBlockDataOnGPU(UIState& state) noexcept {
	auto& gpu = state.gpuState;
	const auto& vol = state.volumeState;

	if (!gpu.blockIndicesUploaded) {
		gpu.blockDataError = "Cannot verify: block data not uploaded";
		return;
	}

	if (!vol.isLoaded || !vol.file.has_value() || vol.file->grids.empty()) {
		gpu.blockDataError = "Cannot verify: no VQVDB file loaded";
		return;
	}

	const auto& blocks = vol.file->grids[0].blocks;

	// Verify block indices
	logMessage(state, "Verifying block indices on GPU via readback...");
	{
		TRANSFER_PROFILE_SCOPE(state.profiler, "Transfer Block Indices GPU->CPU", blocks.indices.size(),
		                       profiler::TransferDirection::GPUToCPU);
		gpu.blockIndicesVerification = vqvdb::verifyBlockIndices(gpu.resources, blocks);
	}
	gpu.blockIndicesVerified = true;

	if (gpu.blockIndicesVerification.passed) {
		logMessage(state, std::format("Block indices verification PASSED: {} bytes match", gpu.blockIndicesVerification.testedElements));
	} else {
		logMessage(state, "ERROR: " + gpu.blockIndicesVerification.message);
	}

	// Verify block metadata (morton codes)
	if (gpu.blockMetadataUploaded) {
		logMessage(state, "Verifying block metadata (morton codes) on GPU...");
		const size_t metadataBytes = blocks.origins.size() * sizeof(vqvdb::BlockMetadata);
		{
			TRANSFER_PROFILE_SCOPE(state.profiler, "Transfer Block Metadata GPU->CPU", metadataBytes,
			                       profiler::TransferDirection::GPUToCPU);
			gpu.blockMetadataVerification = vqvdb::verifyBlockMetadata(gpu.resources, blocks);
		}
		gpu.blockMetadataVerified = true;

		if (gpu.blockMetadataVerification.passed) {
			logMessage(state,
			           std::format("Block metadata verification PASSED: {} morton codes", gpu.blockMetadataVerification.testedElements));
		} else {
			logMessage(state, "ERROR: " + gpu.blockMetadataVerification.message);
		}
	}
}

bool reinitBrickCache(UIState& state) noexcept {
	auto& gpu = state.gpuState;

	vqvdb::BrickCacheConfig config{};
	config.capacitySlots = static_cast<uint32_t>(std::max(1, gpu.brickCacheCapacity));
	config.brickSizeVoxels = static_cast<uint32_t>(vqvdb::kBlockSize);
	config.allocateAtlasTexture = gpu.brickCacheAllocateTexture;

	logMessage(state, std::format("Initializing brick cache: {} slots, atlasTexture={}", config.capacitySlots,
	                              config.allocateAtlasTexture ? "on" : "off"));

	const auto result = vqvdb::initBrickCache(gpu.brickCache, config);
	if (!result.has_value()) {
		gpu.brickCacheInitialized = false;
		gpu.brickCacheError = std::format("Brick cache init failed: {}", vqvdb::errorToString(result.error()));
		logMessage(state, "ERROR: " + gpu.brickCacheError);
		return false;
	}

	gpu.brickCacheInitialized = true;
	gpu.brickCacheError.clear();
	gpu.brickCacheHeatmapSlice = 0;

	logMessage(state, std::format("Brick cache ready: slotGrid={}x{}x{}, atlas={}x{}x{}", gpu.brickCache.slotGridDims.x,
	                              gpu.brickCache.slotGridDims.y, gpu.brickCache.slotGridDims.z, gpu.brickCache.atlasDimsVoxels.x,
	                              gpu.brickCache.atlasDimsVoxels.y, gpu.brickCache.atlasDimsVoxels.z));
	return true;
}

void clearBrickCache(UIState& state) noexcept {
	auto& gpu = state.gpuState;
	if (!gpu.brickCacheInitialized) {
		gpu.brickCacheError = "Brick cache is not initialized";
		return;
	}

	logMessage(state, "Clearing brick cache...");
	if (!reinitBrickCache(state)) {
		return;
	}
	logMessage(state, "Brick cache cleared");
}

void primeBrickCacheFromLoadedBlocks(UIState& state) noexcept {
	auto& gpu = state.gpuState;
	const auto& vol = state.volumeState;

	if (!gpu.brickCacheInitialized && !reinitBrickCache(state)) {
		return;
	}
	if (!vol.isLoaded || !vol.file.has_value() || vol.file->grids.empty()) {
		gpu.brickCacheError = "Cannot prime cache: no VQVDB file loaded";
		return;
	}

	const auto& origins = vol.file->grids[0].blocks.origins;
	if (origins.empty()) {
		gpu.brickCacheError = "Cannot prime cache: loaded grid has no blocks";
		return;
	}

	const size_t count = std::min<size_t>(static_cast<size_t>(std::max(1, gpu.brickCachePrimeCount)), origins.size());
	size_t inserted = 0;
	for (size_t i = 0; i < count; ++i) {
		const auto alloc = vqvdb::allocateSlot(gpu.brickCache, vqvdb::encodeMorton64(origins[i]));
		if (!alloc.has_value()) {
			gpu.brickCacheError = std::format("Cache prime failed at block {}: {}", i, vqvdb::errorToString(alloc.error()));
			logMessage(state, "ERROR: " + gpu.brickCacheError);
			return;
		}
		++inserted;
	}

	const auto hashUpload = vqvdb::uploadCacheHashTable(gpu.brickCache);
	if (!hashUpload.has_value()) {
		gpu.brickCacheError = std::format("Cache hash table upload failed: {}", vqvdb::errorToString(hashUpload.error()));
		logMessage(state, "ERROR: " + gpu.brickCacheError);
		return;
	}

	gpu.brickCacheError.clear();
	const auto stats = vqvdb::getCacheStats(gpu.brickCache);
	logMessage(state, std::format("Primed brick cache with {} blocks (resident={}/{}, evictions={})", inserted, stats.residentBricks,
	                              stats.capacitySlots, static_cast<unsigned long long>(stats.evictionCount)));
}

void initGPUResources(UIState& state) noexcept {
	logMessage(state, "Initializing GPU resources...");
	vqvdb::initGPUResources(state.gpuState.resources);
	state.gpuState.decoderModelPath = kDefaultDecoderModelPath;
	state.gpuState.decoderBackend.reset();
	state.gpuState.decoderInitInProgress = false;
	state.gpuState.decoderReady = false;
	state.gpuState.decoderStatus = "Not initialized";
	state.gpuState.decoderError.clear();
	state.gpuState.decodedBlocksLastFrame = 0;
	(void)reinitBrickCache(state);
	logMessage(state, "GPU resources initialized");
}

bool initDecoderBackend(UIState& state) noexcept {
	auto& gpu = state.gpuState;
	if (gpu.decoderInitInProgress) {
		return false;
	}

	const std::filesystem::path resolvedPath = resolveDecoderModelPath(gpu.decoderModelPath);
	gpu.decoderError.clear();
	gpu.decoderStatus = std::format("Initializing ({})", resolvedPath.string());
	gpu.decoderInitInProgress = true;
	logMessage(state, std::format("Initializing decoder backend in background: {}", resolvedPath.string()));
	try {
		gpu.decoderInitTask = std::async(std::launch::async, [resolvedPath]() { return runDecoderInitTask(resolvedPath); });
	} catch (const std::exception& e) {
		gpu.decoderInitInProgress = false;
		gpu.decoderStatus = "Initialization failed";
		gpu.decoderError = std::format("Failed to launch decoder initialization task: {}", e.what());
		logMessage(state, "ERROR: " + gpu.decoderError);
		return false;
	}
	return true;
}

void pollDecoderBackendInit(UIState& state) noexcept {
	auto& gpu = state.gpuState;
	if (!gpu.decoderInitInProgress || !gpu.decoderInitTask.valid()) {
		return;
	}

	if (gpu.decoderInitTask.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready) {
		return;
	}

	DecoderInitTaskResult result{};
	try {
		result = gpu.decoderInitTask.get();
	} catch (const std::exception& e) {
		gpu.decoderInitInProgress = false;
		gpu.decoderReady = gpu.decoderBackend != nullptr && gpu.decoderBackend->isReady();
		gpu.decoderStatus = "Initialization failed";
		gpu.decoderError = std::format("Decoder initialization task failed: {}", e.what());
		logMessage(state, "ERROR: " + gpu.decoderError);
		return;
	}
	gpu.decoderInitInProgress = false;
	if (!result.success || result.backend == nullptr) {
		gpu.decoderReady = gpu.decoderBackend != nullptr && gpu.decoderBackend->isReady();
		gpu.decoderStatus = "Initialization failed";
		gpu.decoderError = result.errorMessage.empty() ? "Decoder init failed" : result.errorMessage;
		logMessage(state, "ERROR: " + gpu.decoderError);
		return;
	}

	gpu.decoderBackend = std::move(result.backend);
	gpu.decoderReady = gpu.decoderBackend->isReady();
	gpu.decoderStatus = std::format("Ready ({})", result.readyModelPath);
	gpu.decoderError.clear();
	logMessage(state, "Decoder backend ready");
}

void shutdownDecoderBackend(UIState& state) noexcept {
	auto& gpu = state.gpuState;
	if (gpu.decoderInitTask.valid()) {
		try {
			gpu.decoderInitTask.wait();
			(void)gpu.decoderInitTask.get();
		} catch (...) {
			// Swallow task teardown failures during shutdown.
		}
	}
	gpu.decoderInitInProgress = false;
	gpu.decoderBackend.reset();
	gpu.decoderReady = false;
	gpu.decoderStatus = "Not initialized";
	gpu.decoderError.clear();
	gpu.decodedBlocksLastFrame = 0;
}

void shutdownGPUResources(UIState& state) noexcept {
	shutdownDecoderBackend(state);
	vqvdb::shutdownBrickCache(state.gpuState.brickCache);
	state.gpuState.brickCacheInitialized = false;
	vqvdb::shutdownGPUResources(state.gpuState.resources);
}

bool wantCaptureMouse() noexcept {
	// When the mouse is over the Viewport docked window, let the 3D camera
	// handle input instead of ImGui.
	if (ImGui::GetIO().WantCaptureMouse) {
		// Check if the hovered window is the Viewport
		ImGuiWindow* hovered = ImGui::GetCurrentContext()->HoveredWindow;
		if (hovered && strcmp(hovered->Name, "Viewport") == 0) {
			return false;
		}
		return true;
	}
	return false;
}

bool wantCaptureKeyboard() noexcept {
	if (ImGui::GetIO().WantCaptureKeyboard) {
		// Allow keyboard input when the Viewport window is focused
		ImGuiWindow* focused = ImGui::GetCurrentContext()->NavWindow;
		if (focused && strcmp(focused->Name, "Viewport") == 0) {
			return false;
		}
		return true;
	}
	return false;
}

}  // namespace ui
