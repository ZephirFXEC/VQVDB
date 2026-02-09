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

	// Create instanced depth prepass shader
	state.instancedDepthShader = shader::create(shaders::kInstancedDepthVertexShader, shaders::kInstancedDepthFragmentShader);
	if (!state.instancedDepthShader.valid) {
		std::cerr << "[VQVDB] Failed to create instanced depth shader\n";
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

	MeshData unitCubeSolidData = primitives::createUnitCubeSolid();
	state.unitCubeSolid = mesh::upload(unitCubeSolidData);
	if (!state.unitCubeSolid.valid) {
		std::cerr << "[VQVDB] Failed to create unit cube solid mesh\n";
		return false;
	}

	state.initialized = true;
	return true;
}

void shutdown(RendererState& state) noexcept {
	mesh::destroy(state.gridMesh);
	mesh::destroy(state.axisMesh);
	mesh::destroy(state.unitCubeWireframe);
	mesh::destroy(state.unitCubeSolid);
	clearBlockDebugStates(state);
	shader::destroy(state.lineShader);
	shader::destroy(state.instancedBBoxShader);
	shader::destroy(state.instancedDepthShader);
	state.initialized = false;
}

void drawScene(const RendererState& state, const glm::mat4& viewProjection) noexcept {
	if (!state.initialized) return;

	shader::use(state.lineShader);
	shader::setMat4(state.lineShader, "uViewProjection", glm::value_ptr(viewProjection));

	// Identity model matrix for grid and axes (world space)
	constexpr glm::mat4 identity{1.0f};
	shader::setMat4(state.lineShader, "uModel", glm::value_ptr(identity));

	// Draw grid floor
	mesh::draw(state.gridMesh);

	// Draw axis lines
	mesh::draw(state.axisMesh);
}

void setGridTransform(RendererState& state, const glm::mat4& gridTransform, float voxelSize, uint8_t blockSize) noexcept {
	state.gridTransform = gridTransform;
	state.voxelSize = voxelSize;
	state.blockSize = blockSize;
}

bool uploadBlockDebugStates(RendererState& state, std::span<const uint32_t> states) noexcept {
	if (states.empty()) {
		clearBlockDebugStates(state);
		return true;
	}

	const size_t bytes = states.size_bytes();
	if (state.blockDebugStateBuffer == 0 || state.blockDebugStateBufferSize != bytes) {
		clearBlockDebugStates(state);

		glCreateBuffers(1, &state.blockDebugStateBuffer);
		if (state.blockDebugStateBuffer == 0) {
			return false;
		}

		glNamedBufferStorage(state.blockDebugStateBuffer, static_cast<GLsizeiptr>(bytes), states.data(), GL_DYNAMIC_STORAGE_BIT);
		if (glGetError() != GL_NO_ERROR) {
			clearBlockDebugStates(state);
			return false;
		}

		state.blockDebugStateBufferSize = bytes;
		return true;
	}

	glNamedBufferSubData(state.blockDebugStateBuffer, 0, static_cast<GLsizeiptr>(bytes), states.data());
	return glGetError() == GL_NO_ERROR;
}

void clearBlockDebugStates(RendererState& state) noexcept {
	if (state.blockDebugStateBuffer != 0) {
		glDeleteBuffers(1, &state.blockDebugStateBuffer);
		state.blockDebugStateBuffer = 0;
	}
	state.blockDebugStateBufferSize = 0;
}

void drawBlockBBoxesInstanced(const RendererState& state, const vqvdb::GPUResources& gpuResources, const glm::mat4& viewProjection,
                              size_t maxBlocks) noexcept {
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
	shader::setMat4(state.instancedBBoxShader, "uGridTransform", glm::value_ptr(state.gridTransform));
	shader::setInt(state.instancedBBoxShader, "uBlockSize", state.blockSize);
	const bool useDebugStates = state.useBlockDebugColors && state.blockDebugStateBuffer != 0;
	shader::setInt(state.instancedBBoxShader, "uUseDebugState", useDebugStates ? 1 : 0);

	// Bind block origins SSBO to binding point 0
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, gpuResources.blockOriginsBuffer.id);
	if (useDebugStates) {
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, state.blockDebugStateBuffer);
	}

	// Draw instanced cubes
	glBindVertexArray(state.unitCubeWireframe.vao);
	glDrawElementsInstanced(state.unitCubeWireframe.primitiveType, static_cast<GLsizei>(state.unitCubeWireframe.indexCount),
	                        GL_UNSIGNED_INT, nullptr, static_cast<GLsizei>(blocksToRender));

	// Unbind SSBO
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, 0);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, 0);
}

void drawBlockDepthPrepassInstanced(const RendererState& state, const vqvdb::GPUResources& gpuResources, const glm::mat4& viewProjection,
                                    size_t maxBlocks, bool cullNonVisible) noexcept {
	if (!state.initialized) return;
	if (!state.unitCubeSolid.valid) return;
	if (!state.instancedDepthShader.valid) return;
	if (!gpuResources.blockOriginsBuffer.isValid()) return;
	if (gpuResources.numBlocks == 0) return;

	size_t blocksToRender = gpuResources.numBlocks;
	if (maxBlocks > 0 && maxBlocks < blocksToRender) {
		blocksToRender = maxBlocks;
	}

	const bool doCull = cullNonVisible && state.blockDebugStateBuffer != 0;

	shader::use(state.instancedDepthShader);
	shader::setMat4(state.instancedDepthShader, "uViewProjection", glm::value_ptr(viewProjection));
	shader::setMat4(state.instancedDepthShader, "uGridTransform", glm::value_ptr(state.gridTransform));
	shader::setInt(state.instancedDepthShader, "uBlockSize", state.blockSize);
	shader::setInt(state.instancedDepthShader, "uCullNonVisible", doCull ? 1 : 0);

	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, gpuResources.blockOriginsBuffer.id);
	if (doCull) {
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, state.blockDebugStateBuffer);
	}

	GLboolean colorMask[4] = {GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE};
	glGetBooleanv(GL_COLOR_WRITEMASK, colorMask);
	glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);

	glBindVertexArray(state.unitCubeSolid.vao);
	glDrawElementsInstanced(state.unitCubeSolid.primitiveType, static_cast<GLsizei>(state.unitCubeSolid.indexCount), GL_UNSIGNED_INT, nullptr,
	                        static_cast<GLsizei>(blocksToRender));

	glColorMask(colorMask[0], colorMask[1], colorMask[2], colorMask[3]);

	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, 0);
	if (doCull) {
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, 0);
	}
}

}  // namespace renderer
