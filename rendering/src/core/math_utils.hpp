#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <numbers>

namespace math {

constexpr float kPi = std::numbers::pi_v<float>;
constexpr float kDegToRad = kPi / 180.0f;
constexpr float kRadToDeg = 180.0f / kPi;

[[nodiscard]] inline float toRadians(float degrees) noexcept { return degrees * kDegToRad; }

[[nodiscard]] inline float toDegrees(float radians) noexcept { return radians * kRadToDeg; }

}  // namespace math
