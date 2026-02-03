#pragma once

#include "types.hpp"

struct GLFWwindow;

class InputController {
   public:
	InputController(CameraState& cam, CameraLimits& limits, InputState& input);

	static void bindCallbacks(GLFWwindow* window);

   private:
	void onMouseButton(GLFWwindow* window, int button, int action, int mods);
	void onCursorMove(GLFWwindow* window, double xpos, double ypos);
	void onScroll(GLFWwindow* window, double xoffset, double yoffset);
	void onKey(GLFWwindow* window, int key, int scancode, int action, int mods);

	CameraState& cameraRef;
	CameraLimits& limitsRef;
	InputState& inputRef;
};
