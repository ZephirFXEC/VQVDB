#include "render_loop.hpp"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <commdlg.h>  // DO NOT INCLUDE BEFORE windows.h
#endif

#include <glad/glad.h>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <chrono>
#include <cstdio>
#include <glm/gtc/type_ptr.hpp>
#include <string>

#include "camera.hpp"
#include "renderer.hpp"
#include "types.hpp"
#include "ui.hpp"
#include "window.hpp"

namespace {

// Simple file dialog for Windows
std::string openFileDialog(const char* filter, const char* title) {
#ifdef _WIN32
	char filename[MAX_PATH] = {0};

	OPENFILENAMEA ofn;
	ZeroMemory(&ofn, sizeof(ofn));
	ofn.lStructSize = sizeof(ofn);
	ofn.hwndOwner = nullptr;
	ofn.lpstrFilter = filter;
	ofn.lpstrFile = filename;
	ofn.nMaxFile = MAX_PATH;
	ofn.lpstrTitle = title;
	ofn.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR;

	if (GetOpenFileNameA(&ofn)) {
		return std::string(filename);
	}
#else
	(void)filter;
	(void)title;
#endif
	return "";
}

}  // namespace

RenderLoop::RenderLoop(Window& window, CameraState& cam, CameraLimits& limits, InputState& input, RendererState& renderer, Timing& timing,
                       UIState& ui)
    : windowRef(window), cameraRef(cam), limitsRef(limits), inputRef(input), rendererRef(renderer), timingRef(timing), uiRef(ui) {}

void RenderLoop::updateTiming() {
	const auto now = std::chrono::steady_clock::now();
	const auto duration = std::chrono::duration<float>(now - timingRef.lastFrameTime);
	timingRef.deltaTime = duration.count();
	timingRef.lastFrameTime = now;
	timingRef.frameCount++;
}

void RenderLoop::drawFrame() {
	updateTiming();

	// Handle file load request
	if (uiRef.fileLoadRequested) {
		uiRef.fileLoadRequested = false;

		std::string filePath = openFileDialog("VQVDB Files (*.vqvdb)\0*.vqvdb\0All Files (*.*)\0*.*\0", "Open VQVDB File");

		if (!filePath.empty()) {
			ui::loadVQVDBFile(uiRef, filePath);
		}
	}

	// Update renderer if VQVDB file was loaded
	if (uiRef.volumeState.needsRendererUpdate) {
		uiRef.volumeState.needsRendererUpdate = false;
		if (uiRef.volumeState.isLoaded && uiRef.volumeState.file.has_value()) {
			renderer::updateBlockBBoxes(rendererRef, *uiRef.volumeState.file);
		} else {
			renderer::clearBlockBBoxes(rendererRef);
		}
	}
	// Get window size for UI layout
	int windowWidth, windowHeight;
	glfwGetFramebufferSize(windowRef.raw(), &windowWidth, &windowHeight);

	// Update camera with movement input
	const glm::vec3 movementDir = getMovementDirection(inputRef);
	camera::update(cameraRef, limitsRef, movementDir, timingRef.deltaTime);

	// Begin ImGui frame
	ui::beginFrame();

	// Render UI and get viewport dimensions
	ui::renderUI(uiRef, cameraRef, limitsRef, timingRef, windowWidth, windowHeight);

	// Update camera aspect ratio based on viewport (not full window)
	camera::setAspectRatio(cameraRef, static_cast<float>(uiRef.viewportWidth), static_cast<float>(uiRef.viewportHeight));
	camera::computeProjectionMatrix(cameraRef);
	camera::computeViewProjectionMatrix(cameraRef);

	// Clear the entire screen
	glViewport(0, 0, windowWidth, windowHeight);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// Set viewport to the UI-defined region (top-right area)
	// Note: OpenGL viewport origin is bottom-left, so we need to flip Y
	const int viewportY = windowHeight - uiRef.viewportY - uiRef.viewportHeight;
	glViewport(uiRef.viewportX, viewportY, uiRef.viewportWidth, uiRef.viewportHeight);

	// Enable scissor test to clip rendering to viewport
	glEnable(GL_SCISSOR_TEST);
	glScissor(uiRef.viewportX, viewportY, uiRef.viewportWidth, uiRef.viewportHeight);

	// Enable depth testing for proper occlusion
	glEnable(GL_DEPTH_TEST);

	// Draw scene with current view-projection matrix
	renderer::drawScene(rendererRef, cameraRef.viewProjectionMatrix);

	// Disable scissor for UI rendering
	glDisable(GL_SCISSOR_TEST);

	// Reset viewport for UI rendering
	glViewport(0, 0, windowWidth, windowHeight);

	// Render ImGui
	ui::endFrame();

	updateTitle();
}

void RenderLoop::updateTitle() {
	// Only update title every 30 frames to avoid performance hit
	if (timingRef.frameCount % 30 != 0) return;

	const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - timingRef.startTime).count();

	char title[256];
	std::snprintf(title, sizeof(title), "VQVDB GPU Viewer - Milestone 0.2/0.3  |  t=%llds  |  pos=(%.1f, %.1f, %.1f)  |  dist=%.1f",
	              static_cast<long long>(elapsed), cameraRef.position.x, cameraRef.position.y, cameraRef.position.z, cameraRef.distance);

	windowRef.setTitle(title);
}
