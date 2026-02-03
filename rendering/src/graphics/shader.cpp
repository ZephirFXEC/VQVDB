#include "graphics/shader.hpp"

#include <iostream>
#include <vector>

namespace shader {

namespace {

void logCompileError(GLuint shader, std::string_view shaderType) {
	GLint length{0};
	glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &length);
	if (length > 0) {
		std::vector<char> log(static_cast<size_t>(length));
		glGetShaderInfoLog(shader, length, nullptr, log.data());
		std::cerr << "[VQVDB] " << shaderType << " shader compile error:\n" << log.data() << '\n';
	}
}

void logLinkError(GLuint program) {
	GLint length{0};
	glGetProgramiv(program, GL_INFO_LOG_LENGTH, &length);
	if (length > 0) {
		std::vector<char> log(static_cast<size_t>(length));
		glGetProgramInfoLog(program, length, nullptr, log.data());
		std::cerr << "[VQVDB] Program link error:\n" << log.data() << '\n';
	}
}

}  // namespace

GLuint compile(GLenum type, std::string_view source) noexcept {
	GLuint shader = glCreateShader(type);
	const char* srcPtr = source.data();
	const GLint srcLen = static_cast<GLint>(source.size());
	glShaderSource(shader, 1, &srcPtr, &srcLen);
	glCompileShader(shader);

	GLint success{0};
	glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
	if (!success) {
		const char* typeStr = (type == GL_VERTEX_SHADER) ? "Vertex" : "Fragment";
		logCompileError(shader, typeStr);
		glDeleteShader(shader);
		return 0;
	}

	return shader;
}

ShaderProgram link(GLuint vertexShader, GLuint fragmentShader) noexcept {
	ShaderProgram program{};
	program.id = glCreateProgram();

	glAttachShader(program.id, vertexShader);
	glAttachShader(program.id, fragmentShader);
	glLinkProgram(program.id);

	GLint success{0};
	glGetProgramiv(program.id, GL_LINK_STATUS, &success);
	if (!success) {
		logLinkError(program.id);
		glDeleteProgram(program.id);
		program.id = 0;
		program.valid = false;
		return program;
	}

	program.valid = true;
	return program;
}

ShaderProgram create(std::string_view vertexSource, std::string_view fragmentSource) noexcept {
	GLuint vs = compile(GL_VERTEX_SHADER, vertexSource);
	if (vs == 0) {
		return ShaderProgram{};
	}

	GLuint fs = compile(GL_FRAGMENT_SHADER, fragmentSource);
	if (fs == 0) {
		glDeleteShader(vs);
		return ShaderProgram{};
	}

	ShaderProgram program = link(vs, fs);

	// Shaders can be deleted after linking
	glDeleteShader(vs);
	glDeleteShader(fs);

	return program;
}

void destroy(ShaderProgram& program) noexcept {
	if (program.id != 0) {
		glDeleteProgram(program.id);
		program.id = 0;
		program.valid = false;
	}
}

}  // namespace shader
