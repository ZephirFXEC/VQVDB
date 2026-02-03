#include "core/gl_context.hpp"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <glad/glad.h>

#include <iostream>

bool GlContext::load() {
	if (!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress))) {
		std::cerr << "[VQVDB] Failed to load OpenGL symbols via GLAD\n";
		return false;
	}
	return true;
}
