#include "core/input_controller.hpp"

#include "core/camera.hpp"
#include "core/types.hpp"
#include "ui/ui.hpp"

#include <imgui_impl_glfw.h>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

InputController::InputController(CameraState& cam, CameraLimits& limits, InputState& input)
    : cameraRef(cam), limitsRef(limits), inputRef(input) {}

void InputController::bindCallbacks(GLFWwindow* window) {
	glfwSetMouseButtonCallback(window, [](GLFWwindow* win, int button, int action, int mods) {
		// Forward to ImGui first
		ImGui_ImplGlfw_MouseButtonCallback(win, button, action, mods);
		auto* self = static_cast<InputController*>(glfwGetWindowUserPointer(win));
		if (self) self->onMouseButton(win, button, action, mods);
	});

	glfwSetCursorPosCallback(window, [](GLFWwindow* win, double x, double y) {
		// Forward to ImGui first
		ImGui_ImplGlfw_CursorPosCallback(win, x, y);
		auto* self = static_cast<InputController*>(glfwGetWindowUserPointer(win));
		if (self) self->onCursorMove(win, x, y);
	});

	glfwSetScrollCallback(window, [](GLFWwindow* win, double xoff, double yoff) {
		// Forward to ImGui first
		ImGui_ImplGlfw_ScrollCallback(win, xoff, yoff);
		auto* self = static_cast<InputController*>(glfwGetWindowUserPointer(win));
		if (self) self->onScroll(win, xoff, yoff);
	});

	glfwSetKeyCallback(window, [](GLFWwindow* win, int key, int scancode, int action, int mods) {
		// Forward to ImGui first
		ImGui_ImplGlfw_KeyCallback(win, key, scancode, action, mods);
		auto* self = static_cast<InputController*>(glfwGetWindowUserPointer(win));
		if (self) self->onKey(win, key, scancode, action, mods);
	});
	
	glfwSetCharCallback(window, [](GLFWwindow* win, unsigned int c) {
		// Forward to ImGui for text input
		ImGui_ImplGlfw_CharCallback(win, c);
	});
	
	glfwSetWindowFocusCallback(window, [](GLFWwindow* win, int focused) {
		ImGui_ImplGlfw_WindowFocusCallback(win, focused);
	});
	
	glfwSetCursorEnterCallback(window, [](GLFWwindow* win, int entered) {
		ImGui_ImplGlfw_CursorEnterCallback(win, entered);
	});
}

void InputController::onMouseButton(GLFWwindow* window, int button, int action, int /*mods*/) {
	// Don't process if ImGui wants the mouse
	if (ui::wantCaptureMouse()) {
		inputRef.orbiting = false;
		return;
	}
	
	if (button != GLFW_MOUSE_BUTTON_LEFT) return;

	if (action == GLFW_PRESS) {
		inputRef.orbiting = true;
		glfwGetCursorPos(window, &inputRef.lastMouseX, &inputRef.lastMouseY);
	} else if (action == GLFW_RELEASE) {
		inputRef.orbiting = false;
	}
}

void InputController::onCursorMove(GLFWwindow* /*window*/, double xpos, double ypos) {
	if (!inputRef.orbiting || ui::wantCaptureMouse()) return;

	const float dx = static_cast<float>(xpos - inputRef.lastMouseX);
	const float dy = static_cast<float>(ypos - inputRef.lastMouseY);

	camera::applyOrbitDelta(cameraRef, limitsRef, dx, dy);

	inputRef.lastMouseX = xpos;
	inputRef.lastMouseY = ypos;
}

void InputController::onScroll(GLFWwindow* /*window*/, double /*xoffset*/, double yoffset) {
	if (ui::wantCaptureMouse()) return;
	camera::applyZoomDelta(cameraRef, limitsRef, static_cast<float>(yoffset));
}

void InputController::onKey(GLFWwindow* /*window*/, int key, int /*scancode*/, int action, int /*mods*/) {
	// Don't process if ImGui wants the keyboard
	if (ui::wantCaptureKeyboard()) return;
	
	// We use key press/release to set movement direction
	// This allows smooth movement while key is held
	const int value = (action == GLFW_RELEASE) ? 0 : 1;

	switch (key) {
		case GLFW_KEY_W:
			inputRef.moveForward = (action == GLFW_RELEASE) ? 0 : 1;
			break;
		case GLFW_KEY_S:
			inputRef.moveForward = (action == GLFW_RELEASE) ? 0 : -1;
			break;
		case GLFW_KEY_D:
			inputRef.moveRight = (action == GLFW_RELEASE) ? 0 : 1;
			break;
		case GLFW_KEY_A:
			inputRef.moveRight = (action == GLFW_RELEASE) ? 0 : -1;
			break;
		case GLFW_KEY_E:
			inputRef.moveUp = (action == GLFW_RELEASE) ? 0 : 1;
			break;
		case GLFW_KEY_Q:
			inputRef.moveUp = (action == GLFW_RELEASE) ? 0 : -1;
			break;
		default:
			break;
	}
}
