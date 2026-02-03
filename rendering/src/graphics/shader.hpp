#pragma once

#include <glad/glad.h>

#include <string>
#include <string_view>

// Shader program data
struct ShaderProgram {
	GLuint id{0};
	bool valid{false};
};

namespace shader {

// Compile a shader from source
[[nodiscard]] GLuint compile(GLenum type, std::string_view source) noexcept;

// Link vertex and fragment shaders into a program
[[nodiscard]] ShaderProgram link(GLuint vertexShader, GLuint fragmentShader) noexcept;

// Create a complete shader program from vertex and fragment source
[[nodiscard]] ShaderProgram create(std::string_view vertexSource, std::string_view fragmentSource) noexcept;

// Destroy a shader program
void destroy(ShaderProgram& program) noexcept;

// Use/bind shader program
inline void use(const ShaderProgram& program) noexcept { glUseProgram(program.id); }

// Uniform setters
inline void setMat4(const ShaderProgram& program, const char* name, const float* value) noexcept {
	glUniformMatrix4fv(glGetUniformLocation(program.id, name), 1, GL_FALSE, value);
}

}  // namespace shader
