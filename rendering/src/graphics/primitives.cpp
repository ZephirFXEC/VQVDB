#include "graphics/primitives.hpp"

namespace primitives {

MeshData createUnitCubeWireframe() noexcept {
	MeshData data;
	data.primitiveType = GL_LINES;

	// 8 vertices of a unit cube from (0,0,0) to (1,1,1)
	// Color is white - will be overridden by instance shader
	const float r = 1.0f, g = 1.0f, b = 1.0f;

	// Vertices: 8 corners
	//     6-------7
	//    /|      /|
	//   2-------3 |
	//   | 4-----|-5
	//   |/      |/
	//   0-------1

	data.vertices = {
	    // Bottom face (y = 0)
	    {0.0f, 0.0f, 0.0f, r, g, b},  // 0: front-left-bottom
	    {1.0f, 0.0f, 0.0f, r, g, b},  // 1: front-right-bottom
	    {0.0f, 0.0f, 1.0f, r, g, b},  // 2: back-left-bottom
	    {1.0f, 0.0f, 1.0f, r, g, b},  // 3: back-right-bottom
	    // Top face (y = 1)
	    {0.0f, 1.0f, 0.0f, r, g, b},  // 4: front-left-top
	    {1.0f, 1.0f, 0.0f, r, g, b},  // 5: front-right-top
	    {0.0f, 1.0f, 1.0f, r, g, b},  // 6: back-left-top
	    {1.0f, 1.0f, 1.0f, r, g, b},  // 7: back-right-top
	};

	// 12 edges as line segments
	data.indices = {
	    // Bottom face edges
	    0, 1,  // front
	    1, 3,  // right
	    3, 2,  // back
	    2, 0,  // left
	    // Top face edges
	    4, 5,  // front
	    5, 7,  // right
	    7, 6,  // back
	    6, 4,  // left
	    // Vertical edges
	    0, 4,  // front-left
	    1, 5,  // front-right
	    2, 6,  // back-left
	    3, 7,  // back-right
	};

	return data;
}

MeshData createUnitCubeSolid() noexcept {
	MeshData data;
	data.primitiveType = GL_TRIANGLES;

	const float r = 1.0f, g = 1.0f, b = 1.0f;

	data.vertices = {
	    {0.0f, 0.0f, 0.0f, r, g, b},  // 0
	    {1.0f, 0.0f, 0.0f, r, g, b},  // 1
	    {0.0f, 0.0f, 1.0f, r, g, b},  // 2
	    {1.0f, 0.0f, 1.0f, r, g, b},  // 3
	    {0.0f, 1.0f, 0.0f, r, g, b},  // 4
	    {1.0f, 1.0f, 0.0f, r, g, b},  // 5
	    {0.0f, 1.0f, 1.0f, r, g, b},  // 6
	    {1.0f, 1.0f, 1.0f, r, g, b},  // 7
	};

	data.indices = {
	    // Bottom (y=0)
	    0, 2, 1, 1, 2, 3,
	    // Top (y=1)
	    4, 5, 6, 5, 7, 6,
	    // Front (z=0)
	    0, 1, 4, 1, 5, 4,
	    // Back (z=1)
	    2, 6, 3, 3, 6, 7,
	    // Left (x=0)
	    0, 4, 2, 2, 4, 6,
	    // Right (x=1)
	    1, 3, 5, 3, 7, 5,
	};

	return data;
}

MeshData createGrid(int gridSize, float cellSize, float r, float g, float b) noexcept {
	MeshData data;
	data.primitiveType = GL_LINES;

	const float halfExtent = static_cast<float>(gridSize) * cellSize;
	uint32_t vertexIndex = 0;

	// Lines parallel to X axis (varying Z)
	for (int i = -gridSize; i <= gridSize; ++i) {
		const float z = static_cast<float>(i) * cellSize;

		// Use brighter color for center lines
		float lineR = r;
		float lineG = g;
		float lineB = b;
		if (i == 0) {
			lineR = 0.5f;
			lineG = 0.5f;
			lineB = 0.5f;
		}

		data.vertices.push_back({-halfExtent, 0.0f, z, lineR, lineG, lineB});
		data.vertices.push_back({halfExtent, 0.0f, z, lineR, lineG, lineB});
		data.indices.push_back(vertexIndex++);
		data.indices.push_back(vertexIndex++);
	}

	// Lines parallel to Z axis (varying X)
	for (int i = -gridSize; i <= gridSize; ++i) {
		const float x = static_cast<float>(i) * cellSize;

		float lineR = r;
		float lineG = g;
		float lineB = b;
		if (i == 0) {
			lineR = 0.5f;
			lineG = 0.5f;
			lineB = 0.5f;
		}

		data.vertices.push_back({x, 0.0f, -halfExtent, lineR, lineG, lineB});
		data.vertices.push_back({x, 0.0f, halfExtent, lineR, lineG, lineB});
		data.indices.push_back(vertexIndex++);
		data.indices.push_back(vertexIndex++);
	}

	return data;
}

MeshData createAxisLines(float length) noexcept {
	MeshData data;
	data.primitiveType = GL_LINES;

	// X axis (red)
	data.vertices.push_back({0.0f, 0.0f, 0.0f, 1.0f, 0.2f, 0.2f});
	data.vertices.push_back({length, 0.0f, 0.0f, 1.0f, 0.2f, 0.2f});

	// Y axis (green)
	data.vertices.push_back({0.0f, 0.0f, 0.0f, 0.2f, 1.0f, 0.2f});
	data.vertices.push_back({0.0f, length, 0.0f, 0.2f, 1.0f, 0.2f});

	// Z axis (blue)
	data.vertices.push_back({0.0f, 0.0f, 0.0f, 0.2f, 0.2f, 1.0f});
	data.vertices.push_back({0.0f, 0.0f, length, 0.2f, 0.2f, 1.0f});

	data.indices = {0, 1, 2, 3, 4, 5};

	return data;
}

}  // namespace primitives
