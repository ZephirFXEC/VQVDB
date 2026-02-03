#include "graphics/mesh.hpp"

namespace mesh {

Mesh upload(const MeshData& data) noexcept {
	Mesh mesh{};

	if (data.vertices.empty() || data.indices.empty()) {
		return mesh;
	}

	glGenVertexArrays(1, &mesh.vao);
	glGenBuffers(1, &mesh.vbo);
	glGenBuffers(1, &mesh.ebo);

	glBindVertexArray(mesh.vao);

	// Upload vertex data
	glBindBuffer(GL_ARRAY_BUFFER, mesh.vbo);
	glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(data.vertices.size() * sizeof(Vertex)), data.vertices.data(), GL_STATIC_DRAW);

	// Upload index data
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh.ebo);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, static_cast<GLsizeiptr>(data.indices.size() * sizeof(uint32_t)), data.indices.data(),
	             GL_STATIC_DRAW);

	// Position attribute (location 0)
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void*>(offsetof(Vertex, x)));

	// Color attribute (location 1)
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void*>(offsetof(Vertex, r)));

	glBindVertexArray(0);

	mesh.indexCount = static_cast<uint32_t>(data.indices.size());
	mesh.primitiveType = data.primitiveType;
	mesh.valid = true;

	return mesh;
}

void destroy(Mesh& mesh) noexcept {
	if (mesh.vao != 0) {
		glDeleteVertexArrays(1, &mesh.vao);
		mesh.vao = 0;
	}
	if (mesh.vbo != 0) {
		glDeleteBuffers(1, &mesh.vbo);
		mesh.vbo = 0;
	}
	if (mesh.ebo != 0) {
		glDeleteBuffers(1, &mesh.ebo);
		mesh.ebo = 0;
	}
	mesh.indexCount = 0;
	mesh.valid = false;
}

}  // namespace mesh
