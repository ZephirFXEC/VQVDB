#include "graphics/renderer.hpp"

#include <glad/glad.h>

#include <glm/gtc/type_ptr.hpp>
#include <iostream>

#include "graphics/primitives.hpp"

namespace renderer {

bool init(RendererState& state) noexcept {
	// Create line shader
	state.lineShader = shader::create(shaders::kLineVertexShader, shaders::kLineFragmentShader);
	if (!state.lineShader.valid) {
		std::cerr << "[VQVDB] Failed to create line shader\n";
		return false;
	}

	// Create instanced bbox shader
	state.instancedBBoxShader = shader::create(shaders::kInstancedBBoxVertexShader, shaders::kInstancedBBoxFragmentShader);
	if (!state.instancedBBoxShader.valid) {
		std::cerr << "[VQVDB] Failed to create instanced bbox shader\n";
		return false;
	}

	// Create grid mesh
	MeshData gridData = primitives::createGrid(10, 1.0f, 0.25f, 0.25f, 0.25f);
	state.gridMesh = mesh::upload(gridData);
	if (!state.gridMesh.valid) {
		std::cerr << "[VQVDB] Failed to create grid mesh\n";
		return false;
	}

	// Create axis lines
	MeshData axisData = primitives::createAxisLines(2.0f);
	state.axisMesh = mesh::upload(axisData);
	if (!state.axisMesh.valid) {
		std::cerr << "[VQVDB] Failed to create axis mesh\n";
		return false;
	}

	// Create unit cube for instanced rendering
	MeshData unitCubeData = primitives::createUnitCubeWireframe();
	state.unitCubeWireframe = mesh::upload(unitCubeData);
	if (!state.unitCubeWireframe.valid) {
		std::cerr << "[VQVDB] Failed to create unit cube wireframe mesh\n";
		return false;
	}

	state.initialized = true;
	return true;
}

void shutdown(RendererState& state) noexcept {
	mesh::destroy(state.gridMesh);
	mesh::destroy(state.axisMesh);
	mesh::destroy(state.unitCubeWireframe);
	shader::destroy(state.lineShader);
	shader::destroy(state.instancedBBoxShader);
	state.initialized = false;
}

void drawScene(const RendererState& state, const glm::mat4& viewProjection) noexcept {
	if (!state.initialized) return;

	shader::use(state.lineShader);
	shader::setMat4(state.lineShader, "uViewProjection", glm::value_ptr(viewProjection));

	// Identity model matrix for grid and axes (world space)
	const glm::mat4 identity{1.0f};
	shader::setMat4(state.lineShader, "uModel", glm::value_ptr(identity));

	// Draw grid floor
	mesh::draw(state.gridMesh);

	// Draw axis lines
	mesh::draw(state.axisMesh);
}

void setGridTransform(RendererState& state, float voxelSize, float blockSize) noexcept {
	state.voxelSize = voxelSize;
	state.blockSize = blockSize;
}

void drawBlockBBoxesInstanced(const RendererState& state, const vqvdb::GPUResources& gpuResources,
                              const glm::mat4& viewProjection, size_t maxBlocks) noexcept {
	if (!state.initialized) return;
	if (!state.unitCubeWireframe.valid) return;
	if (!gpuResources.blockOriginsBuffer.isValid()) return;
	if (gpuResources.numBlocks == 0) return;

	// Determine how many blocks to render
	size_t blocksToRender = gpuResources.numBlocks;
	if (maxBlocks > 0 && maxBlocks < blocksToRender) {
		blocksToRender = maxBlocks;
	}

	// Use instanced shader
	shader::use(state.instancedBBoxShader);
	shader::setMat4(state.instancedBBoxShader, "uViewProjection", glm::value_ptr(viewProjection));
	shader::setFloat(state.instancedBBoxShader, "uBlockSize", state.blockSize);
	shader::setFloat(state.instancedBBoxShader, "uVoxelSize", state.voxelSize);

	// Bind block origins SSBO to binding point 0
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, gpuResources.blockOriginsBuffer.id);

	// Draw instanced cubes
	glBindVertexArray(state.unitCubeWireframe.vao);
	glDrawElementsInstanced(state.unitCubeWireframe.primitiveType, static_cast<GLsizei>(state.unitCubeWireframe.indexCount),
	                        GL_UNSIGNED_INT, nullptr, static_cast<GLsizei>(blocksToRender));

	// Unbind SSBO
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, 0);
}

}  // namespace renderer
