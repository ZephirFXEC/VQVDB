#pragma once

#include "graphics/mesh.hpp"

// Functions to generate primitive geometry data

namespace primitives {

// Create a unit cube wireframe (edges only) centered at origin
// Cube spans from (-0.5, -0.5, -0.5) to (0.5, 0.5, 0.5)
[[nodiscard]] MeshData createWireframeCube(float r = 1.0f, float g = 1.0f, float b = 1.0f) noexcept;

// Create a grid on the XZ plane centered at origin
// gridSize: number of cells in each direction (total grid is 2*gridSize x 2*gridSize)
// cellSize: size of each cell
[[nodiscard]] MeshData createGrid(int gridSize = 10, float cellSize = 1.0f, float r = 0.3f, float g = 0.3f, float b = 0.3f) noexcept;

// Create axis lines (X=red, Y=green, Z=blue)
[[nodiscard]] MeshData createAxisLines(float length = 1.0f) noexcept;

}  // namespace primitives
