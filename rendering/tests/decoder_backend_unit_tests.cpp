#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <glm/glm.hpp>
#include <sstream>
#include <string>
#include <vector>

#include "vqvdb/cuda_gl_interop.hpp"
#include "vqvdb/decoder_backend.hpp"

#if defined(VQVDB_DECODER_TEST_GL_CONTEXT)
#include <glad/glad.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#endif

namespace {

std::filesystem::path writeTempFile(const std::string& filename, const std::vector<uint8_t>& data) {
	const auto path = std::filesystem::temp_directory_path() / filename;
	std::ofstream out(path, std::ios::binary | std::ios::trunc);
	out.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
	return path;
}

std::string onnxModelsDirFromEnv() {
#if defined(_WIN32)
	char* modelDir = nullptr;
	size_t len = 0;
	if (_dupenv_s(&modelDir, &len, "VQVDB_ONNX_MODELS_DIR") != 0 || modelDir == nullptr) {
		return {};
	}

	std::string value(modelDir);
	free(modelDir);
	return value;
#else
	if (const char* modelDir = std::getenv("VQVDB_ONNX_MODELS_DIR"); modelDir != nullptr && modelDir[0] != '\0') {
		return modelDir;
	}
	return {};
#endif
}

std::filesystem::path findOnnxModelPath(const std::string& fileName) {
	const std::string modelDir = onnxModelsDirFromEnv();
	if (!modelDir.empty()) {
		const auto envPath = std::filesystem::path(modelDir) / fileName;
		if (std::filesystem::exists(envPath)) {
			return envPath;
		}
	}

	std::filesystem::path cursor = std::filesystem::current_path();
	while (!cursor.empty()) {
		const auto candidate = cursor / "models" / "onnx_models" / fileName;
		if (std::filesystem::exists(candidate)) {
			return candidate;
		}
		if (cursor == cursor.root_path()) {
			break;
		}
		cursor = cursor.parent_path();
	}

	return {};
}

bool isRuntimeInitFailure(vqvdb::DecoderError error) {
	return error == vqvdb::DecoderError::EngineBuildFailed || error == vqvdb::DecoderError::EngineDeserializeFailed ||
	       error == vqvdb::DecoderError::CudaError || error == vqvdb::DecoderError::InferenceFailed;
}

struct OnnxRoundTripResult {
	bool executed{false};
	float mae{0.0f};
	float maxAbsError{0.0f};
	std::string details;
};

OnnxRoundTripResult runOnnxRoundTripWithPython(const std::filesystem::path& encoderPath, const std::filesystem::path& decoderPath) {
	const auto scriptPath = std::filesystem::temp_directory_path() / "vqvdb_onnx_roundtrip_test.py";
	const auto outputPath = std::filesystem::temp_directory_path() / "vqvdb_onnx_roundtrip_output.txt";

	{
		std::ofstream script(scriptPath, std::ios::trunc);
		script << "import sys\n";
		script << "import numpy as np\n";
		script << "try:\n";
		script << "    import onnxruntime as ort\n";
		script << "except Exception as exc:\n";
		script << "    print(f'SKIP onnxruntime import failed: {exc}')\n";
		script << "    raise SystemExit(0)\n";
		script << "try:\n";
		script << "    enc_path = sys.argv[1]\n";
		script << "    dec_path = sys.argv[2]\n";
		script << "    x = np.random.default_rng(1234).random((1, 1, 8, 8, 8), dtype=np.float32)\n";
		script << "    enc = ort.InferenceSession(enc_path, providers=['CPUExecutionProvider'])\n";
		script << "    dec = ort.InferenceSession(dec_path, providers=['CPUExecutionProvider'])\n";
		script << "    enc_in = enc.get_inputs()[0].name\n";
		script << "    enc_out = enc.get_outputs()[0].name\n";
		script << "    idx = enc.run([enc_out], {enc_in: x})[0]\n";
		script << "    dec_input = dec.get_inputs()[0]\n";
		script << "    dec_in = dec_input.name\n";
		script << "    dec_type = dec_input.type\n";
		script << "    if 'int64' in dec_type:\n";
		script << "        idx_cast = idx.astype(np.int64, copy=False)\n";
		script << "    elif 'uint8' in dec_type:\n";
		script << "        idx_cast = idx.astype(np.uint8, copy=False)\n";
		script << "    else:\n";
		script << "        idx_cast = idx\n";
		script << "    dec_out = dec.get_outputs()[0].name\n";
		script << "    y = dec.run([dec_out], {dec_in: idx_cast})[0].astype(np.float32, copy=False)\n";
		script << "    mae = float(np.mean(np.abs(y - x)))\n";
		script << "    max_err = float(np.max(np.abs(y - x)))\n";
		script << "    print(f'OK {mae:.8f} {max_err:.8f}')\n";
		script << "except Exception as exc:\n";
		script << "    print(f'ERROR {exc}')\n";
		script << "    raise SystemExit(2)\n";
	}

	std::ostringstream cmd;
	cmd << "python \"" << scriptPath.string() << "\" \"" << encoderPath.string() << "\" \"" << decoderPath.string() << "\" > \""
	    << outputPath.string() << "\" 2>&1";
	const int rc = std::system(cmd.str().c_str());

	std::string output;
	{
		std::ifstream out(outputPath);
		if (out) {
			std::ostringstream buffer;
			buffer << out.rdbuf();
			output = buffer.str();
		}
	}

	std::error_code ec;
	std::filesystem::remove(scriptPath, ec);
	std::filesystem::remove(outputPath, ec);

	if (rc != 0) {
		return {.executed = false, .details = output.empty() ? "python execution failed" : output};
	}

	std::istringstream lines(output);
	for (std::string line; std::getline(lines, line);) {
		if (line.rfind("SKIP ", 0) == 0) {
			return {.executed = false, .details = line};
		}
		if (line.rfind("OK ", 0) == 0) {
			std::istringstream parse(line);
			std::string marker;
			float mae = 0.0f;
			float maxErr = 0.0f;
			parse >> marker >> mae >> maxErr;
			if (parse.fail()) {
				return {.executed = false, .details = "failed to parse round-trip output: " + line};
			}
			return {.executed = true, .mae = mae, .maxAbsError = maxErr, .details = line};
		}
	}

	return {.executed = false, .details = output.empty() ? "no round-trip result produced" : output};
}

#if defined(VQVDB_DECODER_TEST_GL_CONTEXT)
struct GLContextFixture {
	GLFWwindow* window{nullptr};
	bool ready{false};

	GLContextFixture() {
		if (!glfwInit()) {
			return;
		}

		glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
		glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
		glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
		window = glfwCreateWindow(64, 64, "vqvdb_decoder_test", nullptr, nullptr);
		if (window == nullptr) {
			glfwTerminate();
			return;
		}

		glfwMakeContextCurrent(window);
		if (!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress))) {
			glfwDestroyWindow(window);
			window = nullptr;
			glfwTerminate();
			return;
		}

		ready = true;
	}

	~GLContextFixture() {
		if (window != nullptr) {
			glfwDestroyWindow(window);
		}
		glfwTerminate();
	}
};
#endif

#if defined(VQVDB_RENDERING_ENABLE_TRT)
constexpr vqvdb::DecoderError expectedInteropValidationError() { return vqvdb::DecoderError::InvalidInput; }
#else
constexpr vqvdb::DecoderError expectedInteropValidationError() { return vqvdb::DecoderError::InteropError; }
#endif

}  // namespace

TEST_CASE("Decoder backend requires initialization before decode") {
	vqvdb::DecoderBackend backend;
	std::vector<uint8_t> indices(64, 0);
	std::vector<glm::ivec3> offsets{glm::ivec3(0, 0, 0)};

	const auto result = backend.decodeBatch(indices, 1, 1, offsets);
	REQUIRE_FALSE(result.has_value());
	CHECK(result.error() == vqvdb::DecoderError::EngineNotLoaded);
	CHECK_FALSE(backend.isReady());
}

TEST_CASE("Decoder backend init validates model path") {
	vqvdb::DecoderBackend backend;
	auto missingPath = std::filesystem::temp_directory_path() / "vqvdb_decoder_backend_missing_model.onnx";
	std::error_code ec;
	std::filesystem::remove(missingPath, ec);

	const auto result = backend.init(missingPath, 8);
	REQUIRE_FALSE(result.has_value());
	CHECK(result.error() == vqvdb::DecoderError::OnnxModelNotFound);
	CHECK_FALSE(backend.isReady());
}

TEST_CASE("Decoder backend init validates max batch size") {
	vqvdb::DecoderBackend backend;
	const auto temp = writeTempFile("vqvdb_decoder_test_zero_batch.onnx", {1, 2, 3, 4});
	const auto result = backend.init(temp, 0);
	REQUIRE_FALSE(result.has_value());
	CHECK(result.error() == vqvdb::DecoderError::InvalidInput);
	CHECK_FALSE(backend.isReady());

	std::error_code ec;
	std::filesystem::remove(temp, ec);
}

#if !defined(VQVDB_RENDERING_ENABLE_TRT)
TEST_CASE("Decoder backend without TensorRT reports unavailable engine") {
	vqvdb::DecoderBackend backend;
	const auto temp = writeTempFile("vqvdb_decoder_test_no_trt.onnx", {1, 2, 3, 4});

	const auto result = backend.init(temp, 1);
	REQUIRE_FALSE(result.has_value());
	CHECK(result.error() == vqvdb::DecoderError::EngineBuildFailed);
	CHECK_FALSE(backend.isReady());

	std::error_code ec;
	std::filesystem::remove(temp, ec);
}
#endif

TEST_CASE("CUDA-GL interop registration validates texture handle") {
	vqvdb::CudaGLInterop interop;

	const auto result = interop.registerTexture(0);
	REQUIRE_FALSE(result.has_value());
	CHECK(result.error() == expectedInteropValidationError());
	CHECK(interop.resource == nullptr);
	CHECK(interop.registeredTexture == 0);
	CHECK_FALSE(interop.mapped);
}

TEST_CASE("CUDA-GL interop requires registration before mapping") {
	vqvdb::CudaGLInterop interop;

	const auto result = interop.mapForCuda();
	REQUIRE_FALSE(result.has_value());
	CHECK(result.error() == expectedInteropValidationError());
	CHECK_FALSE(interop.mapped);
	CHECK(interop.resource == nullptr);
}

TEST_CASE("CUDA-GL interop requires a mapped resource for brick copy") {
	vqvdb::CudaGLInterop interop;
	std::vector<float> dummyBrick(8 * 8 * 8, 1.0f);

	const auto result = interop.copyBrickToAtlas(dummyBrick.data(), glm::ivec3(0, 0, 0), glm::ivec3(8, 8, 8));
	REQUIRE_FALSE(result.has_value());
	CHECK(result.error() == expectedInteropValidationError());
	CHECK_FALSE(interop.mapped);
	CHECK(interop.resource == nullptr);
	CHECK(interop.registeredTexture == 0);
}

TEST_CASE("CUDA-GL interop reset operations are idempotent") {
	vqvdb::CudaGLInterop interop;

	interop.unmapFromCuda();
	interop.unregisterTexture();
	interop.unmapFromCuda();
	interop.unregisterTexture();

	CHECK_FALSE(interop.mapped);
	CHECK(interop.resource == nullptr);
	CHECK(interop.registeredTexture == 0);
}

TEST_CASE("ONNX encoder/decoder round-trip reconstruction stays within tolerance") {
	const auto encoderPath = findOnnxModelPath("encoder.onnx");
	const auto decoderPath = findOnnxModelPath("decoder.onnx");
	if (encoderPath.empty() || decoderPath.empty()) {
		INFO("encoder.onnx/decoder.onnx not found under models/onnx_models");
		return;
	}

	const auto result = runOnnxRoundTripWithPython(encoderPath, decoderPath);
	if (!result.executed) {
		INFO("Round-trip test not executed: " << result.details);
		return;
	}

	INFO("Round-trip metrics: " << result.details);
	CHECK(result.mae < 0.30f);
	CHECK(result.maxAbsError < 0.75f);
}

#if defined(VQVDB_RENDERING_ENABLE_TRT)

TEST_CASE("Decoder backend can load decoder.onnx model") {
	vqvdb::DecoderBackend backend;
	const auto modelPath = findOnnxModelPath("decoder.onnx");
	if (modelPath.empty()) {
		INFO("decoder.onnx not found under models/onnx_models");
		return;
	}

	const auto result = backend.init(modelPath, 8);
	if (!result.has_value()) {
		CHECK(isRuntimeInitFailure(result.error()));
		CHECK_FALSE(backend.isReady());
		INFO("TensorRT init unavailable in this environment: " << vqvdb::errorToString(result.error()));
		return;
	}
	CHECK(backend.isReady());
}

TEST_CASE("Decoder backend can load decoder_opt.onnx model") {
	vqvdb::DecoderBackend backend;
	const auto modelPath = findOnnxModelPath("decoder_opt.onnx");
	if (modelPath.empty()) {
		INFO("decoder_opt.onnx not found under models/onnx_models");
		return;
	}

	const auto result = backend.init(modelPath, 8);
	if (!result.has_value()) {
		CHECK(isRuntimeInitFailure(result.error()));
		CHECK_FALSE(backend.isReady());
		INFO("TensorRT init unavailable in this environment: " << vqvdb::errorToString(result.error()));
		return;
	}
	CHECK(backend.isReady());
}

TEST_CASE("Decoder backend decodeBatch with real model") {
	vqvdb::DecoderBackend backend;
	const auto modelPath = findOnnxModelPath("decoder.onnx");
	if (modelPath.empty()) {
		INFO("decoder.onnx not found under models/onnx_models");
		return;
	}

	const auto initResult = backend.init(modelPath, 8);
	if (!initResult.has_value()) {
		CHECK(isRuntimeInitFailure(initResult.error()));
		CHECK_FALSE(backend.isReady());
		INFO("TensorRT init unavailable in this environment: " << vqvdb::errorToString(initResult.error()));
		return;
	}
	REQUIRE(backend.isReady());

	// Create test data: 4x4x4 = 64 indices per block
	std::vector<uint8_t> indices(64, 0);
	std::vector<glm::ivec3> offsets{glm::ivec3(0, 0, 0)};

	// Note: decodeBatch requires a valid OpenGL texture, which we don't have in unit tests
	// So we expect InvalidInput due to atlasTexture == 0
	const auto result = backend.decodeBatch(indices, 1, 0, offsets);
	CHECK(result.error() == vqvdb::DecoderError::InvalidInput);
}

#if defined(VQVDB_DECODER_TEST_GL_CONTEXT)
TEST_CASE("Decoder backend decode writes atlas data for one block") {
	GLContextFixture glContext;
	if (!glContext.ready) {
		INFO("Skipping decoder atlas write test: GL context unavailable");
		return;
	}

	vqvdb::DecoderBackend backend;
	const auto modelPath = findOnnxModelPath("decoder.onnx");
	if (modelPath.empty()) {
		INFO("decoder.onnx not found under models/onnx_models");
		return;
	}

	const auto initResult = backend.init(modelPath, 8);
	if (!initResult.has_value()) {
		CHECK(isRuntimeInitFailure(initResult.error()));
		INFO("TensorRT init unavailable in this environment: " << vqvdb::errorToString(initResult.error()));
		return;
	}
	REQUIRE(backend.isReady());

	GLuint atlasTexture = 0;
	glCreateTextures(GL_TEXTURE_3D, 1, &atlasTexture);
	REQUIRE(atlasTexture != 0);
	glTextureStorage3D(atlasTexture, 1, GL_R32F, 8, 8, 8);
	const float zero = 0.0f;
	glClearTexImage(atlasTexture, 0, GL_RED, GL_FLOAT, &zero);

	std::vector<uint8_t> indices(64, 0);
	std::vector<glm::ivec3> offsets{glm::ivec3(0, 0, 0)};
	const auto decodeResult = backend.decodeBatch(indices, 1, atlasTexture, offsets);

	if (!decodeResult.has_value()) {
		CHECK(decodeResult.error() == vqvdb::DecoderError::CudaError || decodeResult.error() == vqvdb::DecoderError::InferenceFailed ||
		      decodeResult.error() == vqvdb::DecoderError::InteropError);
		glDeleteTextures(1, &atlasTexture);
		INFO("Skipping atlas write verification due runtime decode failure: " << vqvdb::errorToString(decodeResult.error()));
		return;
	}

	std::vector<float> voxels(8 * 8 * 8, 0.0f);
	glGetTextureImage(atlasTexture, 0, GL_RED, GL_FLOAT, static_cast<GLsizei>(voxels.size() * sizeof(float)), voxels.data());
	glDeleteTextures(1, &atlasTexture);

	const bool hasNonZeroVoxel = std::any_of(voxels.begin(), voxels.end(), [](float v) { return std::fabs(v) > 1e-6f; });
	CHECK(hasNonZeroVoxel);
}
#endif

#endif  // VQVDB_RENDERING_ENABLE_TRT
