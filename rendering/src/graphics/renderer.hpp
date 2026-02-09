#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#include <glm/glm.hpp>

#include "graphics/mesh.hpp"
#include "graphics/shader.hpp"
#include "vqvdb/gpu_resources.hpp"

// Scene renderer state - holds all GPU resources
struct RendererState {
	ShaderProgram lineShader;
	ShaderProgram instancedBBoxShader;  // Instanced block bbox shader (reads from SSBO)
	ShaderProgram instancedDepthShader; // Instanced solid-cube depth prepass shader
	Mesh gridMesh;
	Mesh axisMesh;
	Mesh unitCubeWireframe;  // Unit cube for instanced rendering
	Mesh unitCubeSolid;      // Unit cube triangles for depth prepass
	bool initialized{false};

	// GPU instanced rendering state
	bool useGPUInstancing{true};   // Prefer GPU instancing when block data is on GPU
	uint8_t blockSize{8};          // Block size in voxels
	float voxelSize{1.0f};         // Voxel size for current grid (UI/stats)
	glm::mat4 gridTransform{1.0f}; // Index -> world transform for block rendering
	bool useBlockDebugColors{true};
	uint32_t blockDebugStateBuffer{0};
	size_t blockDebugStateBufferSize{0};
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
void drawBlockBBoxesInstanced(const RendererState& state, const vqvdb::GPUResources& gpuResources, const glm::mat4& viewProjection,
                              size_t maxBlocks = 0) noexcept;

// Draw depth-only solid cubes for occlusion depth pyramid generation.
// When cullNonVisible is true, uses the blockDebugStateBuffer to skip non-visible blocks on the GPU.
void drawBlockDepthPrepassInstanced(const RendererState& state, const vqvdb::GPUResources& gpuResources,
                                    const glm::mat4& viewProjection, size_t maxBlocks = 0,
                                    bool cullNonVisible = false) noexcept;

// Update renderer with grid transform info for GPU instancing
void setGridTransform(RendererState& state, const glm::mat4& gridTransform, float voxelSize, uint8_t blockSize = 8) noexcept;

// Upload per-block debug state (0=not visible, 1=visible+missing, 2=visible+cached)
[[nodiscard]] bool uploadBlockDebugStates(RendererState& state, std::span<const uint32_t> states) noexcept;

// Delete the optional block debug-state buffer
void clearBlockDebugStates(RendererState& state) noexcept;

}  // namespace renderer

// Embedded shader sources
namespace shaders {

constexpr const char* kLineVertexShader = R"(
#version 460 core

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
#version 460 core

in vec3 vColor;

out vec4 fragColor;

void main()
{
    fragColor = vec4(vColor, 1.0);
}
)";

// Instanced block bounding box shader - reads origins from SSBO
constexpr const char* kInstancedBBoxVertexShader = R"(
#version 460 core

// Unit cube vertex (local space, 0-1 range)
layout(location = 0) in vec3 aPosition;

// Block origins SSBO - ivec3 packed as 3 int32s per block
layout(std430, binding = 0) readonly buffer BlockOrigins {
    ivec4 origins[];  // Padded to 16 bytes for alignment (x, y, z, padding)
};
layout(std430, binding = 1) readonly buffer BlockDebugState {
    uint states[];
};

uniform mat4 uViewProjection;
uniform mat4 uGridTransform;
uniform int uBlockSize;   // Block size in voxels (8)
uniform int uUseDebugState;

out vec3 vColor;

// HSV to RGB conversion for color coding
vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

vec3 blockStateColor(uint stateValue) {
    if (stateValue == 2u) {
        return vec3(0.20, 0.85, 0.30); // visible + cached
    }
    if (stateValue == 1u) {
        return vec3(0.95, 0.45, 0.12); // visible + missing
    }
    return vec3(0.35, 0.37, 0.40); // not visible
}

void main()
{
    // Get block origin for this instance
    ivec3 origin = origins[gl_InstanceID].xyz;
    
    // Compute world position
    vec3 indexPos = vec3(origin) + aPosition * float(uBlockSize);
    vec3 worldPos = vec3(uGridTransform * vec4(indexPos, 1.0));
    
    gl_Position = uViewProjection * vec4(worldPos, 1.0);

    if (uUseDebugState != 0) {
        vColor = blockStateColor(states[gl_InstanceID]);
    } else {
        // Fallback: color based on instance ID
        float hue = fract(float(gl_InstanceID) * 0.00037);  // Golden ratio for good distribution
        vColor = hsv2rgb(vec3(hue, 0.7, 0.9));
    }
}
)";

constexpr const char* kInstancedDepthVertexShader = R"(
#version 460 core

layout(location = 0) in vec3 aPosition;

layout(std430, binding = 0) readonly buffer BlockOrigins {
    ivec4 origins[];
};

layout(std430, binding = 1) readonly buffer BlockDebugState {
    uint states[];
};

uniform mat4 uViewProjection;
uniform mat4 uGridTransform;
uniform int uBlockSize;
uniform int uCullNonVisible;

void main()
{
    // When culling is enabled, discard blocks not marked visible (state == 0).
    if (uCullNonVisible != 0 && states[gl_InstanceID] == 0u) {
        gl_Position = vec4(0.0, 0.0, 0.0, 0.0);
        return;
    }

    ivec3 origin = origins[gl_InstanceID].xyz;
    vec3 indexPos = vec3(origin) + aPosition * float(uBlockSize);
    vec3 worldPos = vec3(uGridTransform * vec4(indexPos, 1.0));
    gl_Position = uViewProjection * vec4(worldPos, 1.0);
}
)";

constexpr const char* kInstancedDepthFragmentShader = R"(
#version 460 core

out vec4 fragColor;

void main()
{
    fragColor = vec4(0.0);
}
)";

constexpr const char* kInstancedBBoxFragmentShader = R"(
#version 460 core

in vec3 vColor;

out vec4 fragColor;

void main()
{
    fragColor = vec4(vColor, 1.0);
}
)";

}  // namespace shaders
