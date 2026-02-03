#include "renderer.hpp"

#include <glm/gtc/type_ptr.hpp>
#include <iostream>

#include "primitives.hpp"

namespace renderer {

bool init(RendererState& state) noexcept {
	// Create line shader
	state.lineShader = shader::create(shaders::kLineVertexShader, shaders::kLineFragmentShader);
	if (!state.lineShader.valid) {
		std::cerr << "[VQVDB] Failed to create line shader\n";
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

	state.initialized = true;
	return true;
}

void shutdown(RendererState& state) noexcept {
	mesh::destroy(state.gridMesh);
	mesh::destroy(state.axisMesh);
	mesh::destroy(state.blockBBoxMesh);
	shader::destroy(state.lineShader);
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

	// Draw block bounding boxes if loaded
	if (state.blockBBoxMesh.valid) {
		mesh::draw(state.blockBBoxMesh);
	}
}

namespace {

// Add a wireframe box to mesh data at the given position with size
void addWireframeBox(MeshData& data, const glm::vec3& min, const glm::vec3& max, float r, float g, float b) {
	const uint32_t baseIdx = static_cast<uint32_t>(data.vertices.size());

	// 8 corners of the box
	data.vertices.push_back({min.x, min.y, min.z, r, g, b});  // 0: front-bottom-left
	data.vertices.push_back({max.x, min.y, min.z, r, g, b});  // 1: front-bottom-right
	data.vertices.push_back({min.x, max.y, min.z, r, g, b});  // 2: front-top-left
	data.vertices.push_back({max.x, max.y, min.z, r, g, b});  // 3: front-top-right
	data.vertices.push_back({min.x, min.y, max.z, r, g, b});  // 4: back-bottom-left
	data.vertices.push_back({max.x, min.y, max.z, r, g, b});  // 5: back-bottom-right
	data.vertices.push_back({min.x, max.y, max.z, r, g, b});  // 6: back-top-left
	data.vertices.push_back({max.x, max.y, max.z, r, g, b});  // 7: back-top-right

	// 12 edges
	// Bottom face
	data.indices.push_back(baseIdx + 0); data.indices.push_back(baseIdx + 1);
	data.indices.push_back(baseIdx + 1); data.indices.push_back(baseIdx + 5);
	data.indices.push_back(baseIdx + 5); data.indices.push_back(baseIdx + 4);
	data.indices.push_back(baseIdx + 4); data.indices.push_back(baseIdx + 0);
	// Top face
	data.indices.push_back(baseIdx + 2); data.indices.push_back(baseIdx + 3);
	data.indices.push_back(baseIdx + 3); data.indices.push_back(baseIdx + 7);
	data.indices.push_back(baseIdx + 7); data.indices.push_back(baseIdx + 6);
	data.indices.push_back(baseIdx + 6); data.indices.push_back(baseIdx + 2);
	// Vertical edges
	data.indices.push_back(baseIdx + 0); data.indices.push_back(baseIdx + 2);
	data.indices.push_back(baseIdx + 1); data.indices.push_back(baseIdx + 3);
	data.indices.push_back(baseIdx + 4); data.indices.push_back(baseIdx + 6);
	data.indices.push_back(baseIdx + 5); data.indices.push_back(baseIdx + 7);
}

}  // anonymous namespace

void updateBlockBBoxes(RendererState& state, const vqvdb::VQVDBFile& file) noexcept {
	// Destroy existing mesh if any
	if (state.blockBBoxMesh.valid) {
		mesh::destroy(state.blockBBoxMesh);
	}

	if (file.empty()) {
		return;
	}

	MeshData bboxData;
	bboxData.primitiveType = GL_LINES;

	// Reserve space for all blocks (8 vertices, 24 indices per block)
	size_t totalBlocks = file.totalBlockCount();
	bboxData.vertices.reserve(totalBlocks * 8);
	bboxData.indices.reserve(totalBlocks * 24);

	// Generate a color for each block - use a gradient based on position
	for (const auto& grid : file.grids) {
		const auto& transform = grid.metadata.transform;
		const float blockSizeWorld = static_cast<float>(vqvdb::kBlockSize) * transform.voxelSize();

		for (const auto& origin : grid.blocks.origins) {
			// Convert block origin to world space
			const glm::vec3 originWorld = transform.indexToWorld(origin.toVec3());
			const glm::vec3 minWorld = originWorld;
			const glm::vec3 maxWorld = originWorld + glm::vec3(blockSizeWorld);

			// Color based on normalized position within bounds
			const auto& bounds = grid.metadata.worldBounds;
			const glm::vec3 normalizedPos = (originWorld - bounds.min) / (bounds.max - bounds.min + 0.001f);

			// Create a nice color gradient (cyan to magenta)
			float r = 0.3f + 0.5f * normalizedPos.x;
			float g = 0.6f + 0.3f * normalizedPos.y;
			float b = 0.8f + 0.2f * normalizedPos.z;

			addWireframeBox(bboxData, minWorld, maxWorld, r, g, b);
		}
	}

	if (!bboxData.vertices.empty()) {
		state.blockBBoxMesh = mesh::upload(bboxData);
		std::cout << "[VQVDB] Created block bbox mesh with " << totalBlocks << " blocks\n";
	}
}

void clearBlockBBoxes(RendererState& state) noexcept {
	if (state.blockBBoxMesh.valid) {
		mesh::destroy(state.blockBBoxMesh);
		state.blockBBoxMesh = {};
	}
}

}  // namespace renderer
