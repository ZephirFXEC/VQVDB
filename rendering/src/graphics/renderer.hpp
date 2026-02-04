#pragma once

#include <glm/glm.hpp>

#include "graphics/mesh.hpp"
#include "graphics/shader.hpp"
#include "vqvdb/gpu_resources.hpp"
#include "vqvdb/vqvdb_types.hpp"

// Scene renderer state - holds all GPU resources
struct RendererState {
	ShaderProgram lineShader;
	ShaderProgram instancedBBoxShader;  // Instanced block bbox shader (reads from SSBO)
	Mesh gridMesh;
	Mesh axisMesh;
	Mesh unitCubeWireframe;  // Unit cube for instanced rendering
	bool initialized{false};

	// GPU instanced rendering state
	bool useGPUInstancing{true};  // Prefer GPU instancing when block data is on GPU
	float blockSize{8.0f};        // Block size in voxels
	float voxelSize{1.0f};        // Voxel size for current grid
};

namespace renderer {

// Initialize renderer resources (shaders, meshes)
[[nodiscard]] bool init(RendererState& state) noexcept;

// Shutdown and release GPU resources
void shutdown(RendererState& state) noexcept;

// Draw the scene (grid, axes, block bboxes)
void drawScene(const RendererState& state, const glm::mat4& viewProjection) noexcept;

// Draw block bounding boxes using GPU instancing (Milestone 1.4)
// Uses block origins SSBO for instanced rendering
// @param maxBlocks Maximum number of blocks to render (0 = all blocks)
void drawBlockBBoxesInstanced(const RendererState& state, const vqvdb::GPUResources& gpuResources,
                               const glm::mat4& viewProjection, size_t maxBlocks = 0) noexcept;

// Update renderer with grid transform info for GPU instancing
void setGridTransform(RendererState& state, float voxelSize, float blockSize = 8.0f) noexcept;

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

// Instanced block bounding box shader - reads origins from SSBO
constexpr const char* kInstancedBBoxVertexShader = R"(
#version 450 core

// Unit cube vertex (local space, 0-1 range)
layout(location = 0) in vec3 aPosition;

// Block origins SSBO - ivec3 packed as 3 int32s per block
layout(std430, binding = 0) readonly buffer BlockOrigins {
    ivec4 origins[];  // Padded to 16 bytes for alignment (x, y, z, padding)
};

uniform mat4 uViewProjection;
uniform float uBlockSize;   // Block size in voxels (8)
uniform float uVoxelSize;   // Voxel size in world units

out vec3 vColor;
flat out int vInstanceID;

// HSV to RGB conversion for color coding
vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

void main()
{
    // Get block origin for this instance
    ivec3 origin = origins[gl_InstanceID].xyz;
    
    // Compute world position
    float worldBlockSize = uBlockSize * uVoxelSize;
    vec3 worldOrigin = vec3(origin) * uVoxelSize;
    vec3 worldPos = worldOrigin + aPosition * worldBlockSize;
    
    gl_Position = uViewProjection * vec4(worldPos, 1.0);
    
    // Color based on instance ID (creates a nice rainbow pattern)
    float hue = fract(float(gl_InstanceID) * 0.00037);  // Golden ratio for good distribution
    vColor = hsv2rgb(vec3(hue, 0.7, 0.9));
    vInstanceID = gl_InstanceID;
}
)";

constexpr const char* kInstancedBBoxFragmentShader = R"(
#version 450 core

in vec3 vColor;
flat in int vInstanceID;

out vec4 fragColor;

void main()
{
    fragColor = vec4(vColor, 1.0);
}
)";

}  // namespace shaders
