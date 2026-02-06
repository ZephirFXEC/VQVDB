#include "core/app.hpp"

#include "core/camera.hpp"
#include "core/gl_context.hpp"
#include "core/input_controller.hpp"
#include "core/render_loop.hpp"
#include "core/types.hpp"
#include "core/window.hpp"
#include "graphics/renderer.hpp"
#include "ui/ui.hpp"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <iostream>
#include <memory>

namespace {
struct AppConfig {
	int width{1280};
	int height{720};
	const char* title{"VQVDB GPU Viewer"};
};
}  // namespace

int App::run() {
	AppConfig config{};
	Window window{};
	GlContext glContext{};
	CameraState cameraState{};
	CameraLimits cameraLimits{};
	InputState input{};
	RendererState rendererState{};
	Timing timing{};
	UIState uiState{};

	if (!window.create(config.width, config.height, config.title)) {
		return EXIT_FAILURE;
	}

	if (!glContext.load()) {
		window.destroy();
		return EXIT_FAILURE;
	}

	// Initialize ImGui
	if (!ui::init(window.raw())) {
		std::cerr << "[VQVDB] Failed to initialize ImGui\n";
		window.destroy();
		return EXIT_FAILURE;
	}

	// Initialize camera aspect ratio
	camera::setAspectRatio(cameraState, static_cast<float>(config.width), static_cast<float>(config.height));

	// Initialize renderer (shaders, meshes)
	if (!renderer::init(rendererState)) {
		std::cerr << "[VQVDB] Failed to initialize renderer\n";
		ui::shutdown();
		window.destroy();
		return EXIT_FAILURE;
	}

	// Initialize GPU resources for VQVDB
	ui::initGPUResources(uiState);

	// Add welcome message to debug log
	ui::logMessage(uiState, "VQVDB GPU Viewer initialized");

	auto inputController = std::make_unique<InputController>(cameraState, cameraLimits, input);
	window.setUserPointer(inputController.get());

	glfwSetFramebufferSizeCallback(window.raw(), [](GLFWwindow* win, int w, int h) {
		glViewport(0, 0, w, h);
		// Update camera aspect ratio on resize
		auto* controller = static_cast<InputController*>(glfwGetWindowUserPointer(win));
		if (controller) {
			// We need access to camera state - this is a limitation of the callback approach
			// For now, we'll handle this in the render loop by checking window size
		}
	});

	inputController->bindCallbacks(window.raw());

	glViewport(0, 0, config.width, config.height);
	glClearColor(0.05f, 0.08f, 0.12f, 1.0f);

	RenderLoop renderLoop(window, cameraState, cameraLimits, input, rendererState, timing, uiState);

	while (!window.shouldClose()) {
		renderLoop.drawFrame();
		window.swapBuffers();
		window.pollEvents();
	}

	uiState.profiler.shutdown();
	ui::shutdownGPUResources(uiState);
	renderer::shutdown(rendererState);
	ui::shutdown();
	window.destroy();
	return EXIT_SUCCESS;
}
