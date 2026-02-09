#include "core/window.hpp"

#include <GLFW/glfw3.h>

#include <iostream>

bool Window::create(int width, int height, const std::string& title) {
	if (glfwInit() == GLFW_FALSE) {
		std::cerr << "[VQVDB] Failed to initialize GLFW\n";
		return false;
	}

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif
	glfwWindowHint(GLFW_DOUBLEBUFFER, GLFW_TRUE);
	glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

	handle = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);
	if (!handle) {
		std::cerr << "[VQVDB] Failed to create GLFW window\n";
		glfwTerminate();
		return false;
	}

	glfwMakeContextCurrent(handle);
	glfwSwapInterval(1);
	return true;
}

void Window::destroy() {
	if (handle) {
		glfwDestroyWindow(handle);
		handle = nullptr;
	}
	glfwTerminate();
}

bool Window::shouldClose() const { return glfwWindowShouldClose(handle); }

void Window::swapBuffers() const { glfwSwapBuffers(handle); }

void Window::pollEvents() const { glfwPollEvents(); }

void Window::setUserPointer(void* ptr) { glfwSetWindowUserPointer(handle, ptr); }

void Window::setTitle(const std::string& title) { glfwSetWindowTitle(handle, title.c_str()); }
