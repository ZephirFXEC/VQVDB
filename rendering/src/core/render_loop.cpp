#include "core/render_loop.hpp"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <commdlg.h>
#endif

#include <glad/glad.h>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <chrono>
#include <cstdio>
#include <glm/gtc/type_ptr.hpp>
#include <string>

#include "core/camera.hpp"
#include "core/profiler.hpp"
#include "core/types.hpp"
#include "core/window.hpp"
#include "graphics/renderer.hpp"
#include "ui/ui.hpp"

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
	uiRef.profiler.setEnabled(uiRef.profilerEnabled && uiRef.profilerGathering);
	uiRef.profiler.beginFrame();

	{
		CPU_PROFILE_SCOPE(uiRef.profiler, "Frame");

		{
			CPU_PROFILE_SCOPE(uiRef.profiler, "Handle UI Requests");

			// Handle file load request
			if (uiRef.fileLoadRequested) {
				uiRef.fileLoadRequested = false;

				std::string filePath = openFileDialog("VQVDB Files (*.vqvdb)\0*.vqvdb\0All Files (*.*)\0*.*\0", "Open VQVDB File");

				if (!filePath.empty()) {
					ui::loadVQVDBFile(uiRef, filePath);
				}
			}

			// Handle codebook load request (Milestone 1.2)
			if (uiRef.gpuState.codebookLoadRequested) {
				uiRef.gpuState.codebookLoadRequested = false;

				std::string filePath = openFileDialog("Codebook Files (*.bin)\0*.bin\0All Files (*.*)\0*.*\0", "Open Codebook File");

				if (!filePath.empty()) {
					ui::loadAndUploadCodebook(uiRef, filePath);
				}
			}

			// Handle codebook verify request (Milestone 1.2)
			if (uiRef.gpuState.codebookVerifyRequested) {
				uiRef.gpuState.codebookVerifyRequested = false;
				ui::verifyCodebookOnGPU(uiRef);
			}

			// Handle block data upload request (Milestone 1.3)
			if (uiRef.gpuState.blockDataUploadRequested) {
				uiRef.gpuState.blockDataUploadRequested = false;
				ui::uploadBlockDataToGPU(uiRef);
			}

			// Handle block data verify request (Milestone 1.3)
			if (uiRef.gpuState.blockDataVerifyRequested) {
				uiRef.gpuState.blockDataVerifyRequested = false;
				ui::verifyBlockDataOnGPU(uiRef);
			}

			// Update renderer with grid transform if GPU data was uploaded (Milestone 1.4)
			if (uiRef.gpuState.rendererNeedsUpdate) {
				uiRef.gpuState.rendererNeedsUpdate = false;
				renderer::setGridTransform(rendererRef, uiRef.gpuState.voxelSize, uiRef.gpuState.blockSize);
			}
		}

		// Get window size for UI layout
		int windowWidth, windowHeight;
		glfwGetFramebufferSize(windowRef.raw(), &windowWidth, &windowHeight);

		{
			CPU_PROFILE_SCOPE(uiRef.profiler, "Camera Update");
			// Update camera with movement input
			const glm::vec3 movementDir = getMovementDirection(inputRef);
			camera::update(cameraRef, limitsRef, movementDir, timingRef.deltaTime);
		}

		{
			CPU_PROFILE_SCOPE(uiRef.profiler, "Build UI");
			// Begin ImGui frame
			ui::beginFrame();

			// Render UI and get viewport dimensions
			ui::renderUI(uiRef, cameraRef, limitsRef, timingRef, windowWidth, windowHeight);
		}

		// Update camera aspect ratio based on viewport (not full window)
		camera::setAspectRatio(cameraRef, static_cast<float>(uiRef.viewportWidth), static_cast<float>(uiRef.viewportHeight));
		camera::computeProjectionMatrix(cameraRef);
		camera::computeViewProjectionMatrix(cameraRef);

		{
			CPU_PROFILE_SCOPE(uiRef.profiler, "Render 3D Scene");
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

			{
				GPU_PROFILE_SCOPE(uiRef.profiler, "Scene Draw");
				renderer::drawScene(rendererRef, cameraRef.viewProjectionMatrix);
			}

			// Draw block bboxes using GPU instancing if data is uploaded (Milestone 1.4)
			// This replaces the CPU-generated bbox mesh when GPU data is available
			if (uiRef.gpuState.blockIndicesUploaded && uiRef.gpuState.resources.hasBlockData()) {
				// Compute block limit based on UI settings
				size_t maxBlocks = 0;  // 0 = all blocks
				if (uiRef.gpuState.useBlockLimit && uiRef.gpuState.maxDisplayBlocks > 0) {
					maxBlocks = static_cast<size_t>(uiRef.gpuState.maxDisplayBlocks);
				}

				GPU_PROFILE_SCOPE(uiRef.profiler, "Block BBoxes Draw");
				renderer::drawBlockBBoxesInstanced(rendererRef, uiRef.gpuState.resources, cameraRef.viewProjectionMatrix, maxBlocks);
			}

			// Disable scissor for UI rendering
			glDisable(GL_SCISSOR_TEST);

			// Reset viewport for UI rendering
			glViewport(0, 0, windowWidth, windowHeight);
		}

		{
			CPU_PROFILE_SCOPE(uiRef.profiler, "Render ImGui");
			GPU_PROFILE_SCOPE(uiRef.profiler, "ImGui Draw");
			ui::endFrame();
		}

		updateTitle();
	}

	uiRef.profiler.endFrame();
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
