#pragma once

#include <glad/glad.h>

#include <cstdint>
#include <vector>

// Vertex with position and color
struct Vertex {
	float x, y, z;
	float r, g, b;
};

// GPU mesh handle - stores VAO, VBO, EBO
struct Mesh {
	GLuint vao{0};
	GLuint vbo{0};
	GLuint ebo{0};
	uint32_t indexCount{0};
	GLenum primitiveType{GL_TRIANGLES};
	bool valid{false};
};

// CPU-side mesh data for building geometry
struct MeshData {
	std::vector<Vertex> vertices;
	std::vector<uint32_t> indices;
	GLenum primitiveType{GL_TRIANGLES};
};

namespace mesh {

// Upload mesh data to GPU
[[nodiscard]] Mesh upload(const MeshData& data) noexcept;

// Destroy mesh GPU resources
void destroy(Mesh& mesh) noexcept;

// Draw mesh
inline void draw(const Mesh& mesh) noexcept {
	if (!mesh.valid) return;
	glBindVertexArray(mesh.vao);
	glDrawElements(mesh.primitiveType, static_cast<GLsizei>(mesh.indexCount), GL_UNSIGNED_INT, nullptr);
}

}  // namespace mesh
