#pragma once

#include <algorithm>
#include <glm/glm.hpp>

#include "core/math_utils.hpp"

// Data-oriented camera representation
// Separates camera state (data) from operations (functions)

struct CameraState {
	// Orbit parameters
	float yawDegrees{0.0f};
	float pitchDegrees{20.0f};
	float distance{5.0f};

	// Target point (orbit center / focus point)
	glm::vec3 target{0.0f, 0.0f, 0.0f};

	// Free movement velocity (for smooth WASD movement)
	glm::vec3 velocity{0.0f};

	// Computed values (updated each frame)
	glm::vec3 position{0.0f, 0.0f, 5.0f};
	glm::vec3 forward{0.0f, 0.0f, -1.0f};
	glm::vec3 right{1.0f, 0.0f, 0.0f};
	glm::vec3 up{0.0f, 1.0f, 0.0f};

	// View and projection matrices
	glm::mat4 viewMatrix{1.0f};
	glm::mat4 projectionMatrix{1.0f};
	glm::mat4 viewProjectionMatrix{1.0f};

	// Projection parameters
	float fovDegrees{60.0f};
	float aspectRatio{16.0f / 9.0f};
	float nearPlane{0.1f};
	float farPlane{1000.0f};
};

struct CameraLimits {
	float minPitch{-89.0f};
	float maxPitch{89.0f};
	float minDistance{0.5f};
	float maxDistance{100.0f};
	float orbitSensitivity{0.25f};
	float zoomSensitivity{0.15f};
	float moveSpeed{25.0f};
	float moveDamping{10.0f};
};

namespace camera {

// Clamp camera parameters to valid ranges
inline void clampValues(CameraState& state, const CameraLimits& limits) noexcept {
	state.pitchDegrees = std::clamp(state.pitchDegrees, limits.minPitch, limits.maxPitch);
	state.distance = std::clamp(state.distance, limits.minDistance, limits.maxDistance);
}

// Apply orbit rotation delta (from mouse movement)
inline void applyOrbitDelta(CameraState& state, const CameraLimits& limits, float deltaYaw, float deltaPitch) noexcept {
	state.yawDegrees -= deltaYaw * limits.orbitSensitivity;
	state.pitchDegrees += deltaPitch * limits.orbitSensitivity;
	clampValues(state, limits);
}

// Apply zoom delta (from scroll wheel)
inline void applyZoomDelta(CameraState& state, const CameraLimits& limits, float delta) noexcept {
	state.distance *= 1.0f - delta * limits.zoomSensitivity;
	clampValues(state, limits);
}

// Compute camera vectors from orbit angles
inline void computeOrbitVectors(CameraState& state) noexcept {
	const float yawRad = math::toRadians(state.yawDegrees);
	const float pitchRad = math::toRadians(state.pitchDegrees);

	const float cosPitch = std::cos(pitchRad);
	const float sinPitch = std::sin(pitchRad);
	const float cosYaw = std::cos(yawRad);
	const float sinYaw = std::sin(yawRad);

	// Direction from target to camera (orbit offset direction)
	const glm::vec3 orbitDir{cosPitch * sinYaw, sinPitch, cosPitch * cosYaw};

	state.position = state.target + orbitDir * state.distance;
	state.forward = -orbitDir;
	state.right = glm::normalize(glm::cross(state.forward, glm::vec3(0.0f, 1.0f, 0.0f)));
	state.up = glm::normalize(glm::cross(state.right, state.forward));
}

// Apply keyboard movement input to velocity
inline void applyMovementInput(CameraState& state, const CameraLimits& limits, const glm::vec3& inputDir, float deltaTime) noexcept {
	// Accumulate velocity based on input
	if (glm::length(inputDir) > 0.001f) {
		const glm::vec3 normalizedInput = glm::normalize(inputDir);
		// Move target, which moves the orbit center
		const glm::vec3 moveDir =
		    state.right * normalizedInput.x + glm::vec3(0.0f, 1.0f, 0.0f) * normalizedInput.y + state.forward * normalizedInput.z;
		state.velocity += moveDir * limits.moveSpeed * deltaTime;
	}

	// Apply velocity to target
	state.target += state.velocity * deltaTime;

	// Dampen velocity
	const float damping = std::exp(-limits.moveDamping * deltaTime);
	state.velocity *= damping;
}

// Update view matrix from current camera state
inline void computeViewMatrix(CameraState& state) noexcept {
	state.viewMatrix = glm::lookAt(state.position, state.target, glm::vec3(0.0f, 1.0f, 0.0f));
}

// Update projection matrix
inline void computeProjectionMatrix(CameraState& state) noexcept {
	state.projectionMatrix = glm::perspective(math::toRadians(state.fovDegrees), state.aspectRatio, state.nearPlane, state.farPlane);
}

// Update combined view-projection matrix
inline void computeViewProjectionMatrix(CameraState& state) noexcept {
	state.viewProjectionMatrix = state.projectionMatrix * state.viewMatrix;
}

// Full camera update for the frame
inline void update(CameraState& state, const CameraLimits& limits, const glm::vec3& movementInput, float deltaTime) noexcept {
	applyMovementInput(state, limits, movementInput, deltaTime);
	computeOrbitVectors(state);
	computeViewMatrix(state);
	computeProjectionMatrix(state);
	computeViewProjectionMatrix(state);
}

// Set aspect ratio (on window resize)
inline void setAspectRatio(CameraState& state, float width, float height) noexcept {
	if (height > 0.0f) {
		state.aspectRatio = width / height;
	}
}

}  // namespace camera
