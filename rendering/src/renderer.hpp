#pragma once

#include <glm/glm.hpp>

#include "mesh.hpp"
#include "shader.hpp"
#include "vqvdb/vqvdb_types.hpp"

// Scene renderer state - holds all GPU resources
struct RendererState {
	ShaderProgram lineShader;
	Mesh gridMesh;
	Mesh axisMesh;
	Mesh blockBBoxMesh;  // Block bounding boxes from loaded VQVDB
	bool initialized{false};
};

namespace renderer {

// Initialize renderer resources (shaders, meshes)
[[nodiscard]] bool init(RendererState& state) noexcept;

// Shutdown and release GPU resources
void shutdown(RendererState& state) noexcept;

// Draw the scene (grid, axes, block bboxes)
void drawScene(const RendererState& state, const glm::mat4& viewProjection) noexcept;

// Update block bounding boxes from loaded VQVDB file
void updateBlockBBoxes(RendererState& state, const vqvdb::VQVDBFile& file) noexcept;

// Clear block bounding boxes
void clearBlockBBoxes(RendererState& state) noexcept;

}  // namespace renderer

// Embedded shader sources
namespace shaders {

constexpr const char* kLineVertexShader = R"(
#version 450 core

layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aColor;

uniform mat4 uViewProjection;
uniform mat4 uModel;

out vec3 vColor;

void main()
{
    vColor = aColor;
    gl_Position = uViewProjection * uModel * vec4(aPosition, 1.0);
}
)";

constexpr const char* kLineFragmentShader = R"(
#version 450 core

in vec3 vColor;

out vec4 fragColor;

void main()
{
    fragColor = vec4(vColor, 1.0);
}
)";

}  // namespace shaders
