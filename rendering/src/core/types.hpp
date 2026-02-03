#pragma once

#include <chrono>

#include <glm/glm.hpp>

#include "core/camera.hpp"

// Input state for mouse/keyboard handling
struct InputState {
	// Mouse orbiting state
	bool orbiting{false};
	double lastMouseX{0.0};
	double lastMouseY{0.0};

	// Keyboard movement input (-1, 0, or 1 for each axis)
	int moveForward{0};  // W/S
	int moveRight{0};    // A/D
	int moveUp{0};       // Q/E
};

// Timing information for frame timing
struct Timing {
	std::chrono::steady_clock::time_point startTime{std::chrono::steady_clock::now()};
	std::chrono::steady_clock::time_point lastFrameTime{std::chrono::steady_clock::now()};
	float deltaTime{0.0f};
	uint64_t frameCount{0};
};

// Compute movement direction from input state
inline glm::vec3 getMovementDirection(const InputState& input) noexcept {
	return glm::vec3{static_cast<float>(input.moveRight), static_cast<float>(input.moveUp), static_cast<float>(input.moveForward)};
}
