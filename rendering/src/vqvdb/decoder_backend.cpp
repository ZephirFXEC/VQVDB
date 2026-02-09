/*
 * Copyright (c) 2025, Enzo Crema
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "vqvdb/decoder_backend.hpp"

#include "vqvdb/cuda_gl_interop.hpp"

#if defined(VQVDB_RENDERING_ENABLE_TRT)

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>
#include <glad/glad.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

namespace vqvdb {

namespace {

constexpr uint32_t kLatentElementsPerBlock = 4u * 4u * 4u;
constexpr uint32_t kDecodedElementsPerBlock = 8u * 8u * 8u;

bool cudaOk(cudaError_t err) { return err == cudaSuccess; }

uint64_t fnv1a64Update(uint64_t state, const uint8_t* data, size_t len) {
	constexpr uint64_t kPrime = 1099511628211ull;
	for (size_t i = 0; i < len; ++i) {
		state ^= static_cast<uint64_t>(data[i]);
		state *= kPrime;
	}
	return state;
}

std::string hashFileFNV1aHex(const std::filesystem::path& path) {
	std::ifstream file(path, std::ios::binary);
	if (!file) return "missing";

	constexpr size_t kBufSize = 1u << 15;
	std::array<uint8_t, kBufSize> buffer{};
	uint64_t state = 14695981039346656037ull;

	while (file) {
		file.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(buffer.size()));
		const size_t got = static_cast<size_t>(file.gcount());
		if (got == 0) break;
		state = fnv1a64Update(state, buffer.data(), got);
	}

	std::ostringstream oss;
	oss << std::hex << state;
	return oss.str();
}

std::string gpuArchTag() {
	int device = 0;
	if (!cudaOk(cudaGetDevice(&device))) return "gpu_unknown";

	cudaDeviceProp prop{};
	if (!cudaOk(cudaGetDeviceProperties(&prop, device))) return "gpu_unknown";

	std::ostringstream oss;
	oss << "sm" << prop.major << prop.minor;
	return oss.str();
}

class TRTLogger final : public nvinfer1::ILogger {
   public:
	void log(Severity severity, const char* msg) noexcept override {
		if (severity <= Severity::kWARNING) {
			(void)msg;
		}
	}
};

struct TRTDeleter {
	template <typename T>
	void operator()(T* ptr) const {
		if (ptr == nullptr) {
			return;
		}
		if constexpr (requires(T* p) { p->destroy(); }) {
			ptr->destroy();
		} else {
			delete ptr;
		}
	}
};

}  // namespace

struct DecoderBackend::Impl {
	TRTLogger logger;
	std::unique_ptr<nvinfer1::IRuntime, TRTDeleter> runtime;
	std::unique_ptr<nvinfer1::ICudaEngine, TRTDeleter> engine;
	std::unique_ptr<nvinfer1::IExecutionContext, TRTDeleter> context;

	cudaStream_t stream{nullptr};
	void* dInput{nullptr};
	void* dOutput{nullptr};
	size_t dInputBytes{0};
	size_t dOutputBytes{0};

	std::string inputTensorName;
	std::string outputTensorName;
	uint32_t maxBatchSize{0};
	bool ready{false};

	CudaGLInterop interop;
	bool interopUsable{true};

	~Impl() {
		if (dInput != nullptr) cudaFree(dInput);
		if (dOutput != nullptr) cudaFree(dOutput);
		if (stream != nullptr) cudaStreamDestroy(stream);
		interop.unregisterTexture();
	}
};

DecoderBackend::DecoderBackend() : impl_(std::make_unique<Impl>()) {}

DecoderBackend::~DecoderBackend() = default;

DecoderResult<void> DecoderBackend::init(const std::filesystem::path& onnxModelPath, uint32_t maxBatchSize) {
	if (!std::filesystem::exists(onnxModelPath)) {
		return std::unexpected(DecoderError::OnnxModelNotFound);
	}
	if (maxBatchSize == 0) {
		return std::unexpected(DecoderError::InvalidInput);
	}

	impl_->ready = false;
	impl_->maxBatchSize = maxBatchSize;

	impl_->runtime.reset(nvinfer1::createInferRuntime(impl_->logger));
	if (!impl_->runtime) {
		return std::unexpected(DecoderError::EngineBuildFailed);
	}

	const auto cachePath = engineCachePath(onnxModelPath);
	auto loadResult = loadCachedEngine(cachePath);
	if (!loadResult.has_value()) {
		auto buildResult = buildEngine(onnxModelPath, maxBatchSize);
		if (!buildResult.has_value()) {
			return buildResult;
		}
	}

	impl_->context.reset(impl_->engine->createExecutionContext());
	if (!impl_->context) {
		return std::unexpected(DecoderError::EngineDeserializeFailed);
	}

	const int ioCount = impl_->engine->getNbIOTensors();
	for (int i = 0; i < ioCount; ++i) {
		const char* name = impl_->engine->getIOTensorName(i);
		if (impl_->engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
			impl_->inputTensorName = name;
		} else {
			impl_->outputTensorName = name;
		}
	}
	if (impl_->inputTensorName.empty() || impl_->outputTensorName.empty()) {
		return std::unexpected(DecoderError::EngineDeserializeFailed);
	}

	if (impl_->stream == nullptr && !cudaOk(cudaStreamCreate(&impl_->stream))) {
		return std::unexpected(DecoderError::CudaError);
	}

	impl_->dInputBytes = static_cast<size_t>(maxBatchSize) * kLatentElementsPerBlock * sizeof(uint8_t);
	impl_->dOutputBytes = static_cast<size_t>(maxBatchSize) * kDecodedElementsPerBlock * sizeof(float);

	if (impl_->dInput != nullptr) {
		cudaFree(impl_->dInput);
		impl_->dInput = nullptr;
	}
	if (impl_->dOutput != nullptr) {
		cudaFree(impl_->dOutput);
		impl_->dOutput = nullptr;
	}

	if (!cudaOk(cudaMalloc(&impl_->dInput, impl_->dInputBytes)) || !cudaOk(cudaMalloc(&impl_->dOutput, impl_->dOutputBytes))) {
		return std::unexpected(DecoderError::CudaError);
	}

	impl_->ready = true;
	impl_->interopUsable = true;
	return {};
}

DecoderResult<void> DecoderBackend::decodeBatch(std::span<const uint8_t> indices, uint32_t batchSize, uint32_t atlasTexture,
                                                std::span<const glm::ivec3> slotOffsets) {
	if (!impl_->ready || !impl_->context || !impl_->engine) {
		return std::unexpected(DecoderError::EngineNotLoaded);
	}
	if (batchSize == 0 || batchSize > impl_->maxBatchSize) {
		return std::unexpected(DecoderError::InvalidInput);
	}
	if (slotOffsets.size() != batchSize) {
		return std::unexpected(DecoderError::InvalidInput);
	}
	const size_t expectedInputSize = static_cast<size_t>(batchSize) * kLatentElementsPerBlock;
	if (indices.size() < expectedInputSize) {
		return std::unexpected(DecoderError::InvalidInput);
	}
	if (atlasTexture == 0) {
		return std::unexpected(DecoderError::InvalidInput);
	}

	const size_t inputBytes = expectedInputSize * sizeof(uint8_t);
	if (!cudaOk(cudaMemcpyAsync(impl_->dInput, indices.data(), inputBytes, cudaMemcpyHostToDevice, impl_->stream))) {
		return std::unexpected(DecoderError::CudaError);
	}

	nvinfer1::Dims inputDims{};
	inputDims.nbDims = 4;
	inputDims.d[0] = static_cast<int>(batchSize);
	inputDims.d[1] = 4;
	inputDims.d[2] = 4;
	inputDims.d[3] = 4;

	if (!impl_->context->setInputShape(impl_->inputTensorName.c_str(), inputDims)) {
		return std::unexpected(DecoderError::InferenceFailed);
	}
	if (!impl_->context->setTensorAddress(impl_->inputTensorName.c_str(), impl_->dInput) ||
	    !impl_->context->setTensorAddress(impl_->outputTensorName.c_str(), impl_->dOutput)) {
		return std::unexpected(DecoderError::InferenceFailed);
	}
	if (!impl_->context->enqueueV3(impl_->stream)) {
		return std::unexpected(DecoderError::InferenceFailed);
	}
	if (!cudaOk(cudaStreamSynchronize(impl_->stream))) {
		return std::unexpected(DecoderError::CudaError);
	}

	GLint width = 0;
	GLint height = 0;
	GLint depth = 0;
	glGetTextureLevelParameteriv(atlasTexture, 0, GL_TEXTURE_WIDTH, &width);
	glGetTextureLevelParameteriv(atlasTexture, 0, GL_TEXTURE_HEIGHT, &height);
	glGetTextureLevelParameteriv(atlasTexture, 0, GL_TEXTURE_DEPTH, &depth);
	if (width <= 0 || height <= 0 || depth <= 0) {
		return std::unexpected(DecoderError::InvalidInput);
	}

	const glm::ivec3 atlasDims{width, height, depth};
	bool usedInterop = false;
	if (impl_->interopUsable) {
		auto regResult = impl_->interop.registerTexture(atlasTexture);
		if (regResult.has_value()) {
			auto mapResult = impl_->interop.mapForCuda();
			if (mapResult.has_value()) {
				usedInterop = true;
				for (uint32_t i = 0; i < batchSize; ++i) {
					const float* brickPtr = static_cast<const float*>(impl_->dOutput) + static_cast<size_t>(i) * kDecodedElementsPerBlock;
					auto copyResult = impl_->interop.copyBrickToAtlas(brickPtr, slotOffsets[i], atlasDims);
					if (!copyResult.has_value()) {
						usedInterop = false;
						break;
					}
				}
				impl_->interop.unmapFromCuda();
			}
		}

		if (!usedInterop) {
			impl_->interopUsable = false;
			impl_->interop.unregisterTexture();
		}
	}

	if (usedInterop) {
		return {};
	}

	// Fallback path: readback to CPU + glTextureSubImage3D upload.
	std::vector<float> hostOutput(static_cast<size_t>(batchSize) * kDecodedElementsPerBlock);
	const size_t outputBytes = hostOutput.size() * sizeof(float);
	if (!cudaOk(cudaMemcpyAsync(hostOutput.data(), impl_->dOutput, outputBytes, cudaMemcpyDeviceToHost, impl_->stream))) {
		return std::unexpected(DecoderError::CudaError);
	}
	if (!cudaOk(cudaStreamSynchronize(impl_->stream))) {
		return std::unexpected(DecoderError::CudaError);
	}

	for (uint32_t i = 0; i < batchSize; ++i) {
		const glm::ivec3& off = slotOffsets[i];
		const void* brickData = hostOutput.data() + static_cast<size_t>(i) * kDecodedElementsPerBlock;
		glTextureSubImage3D(atlasTexture, 0, off.x, off.y, off.z, 8, 8, 8, GL_RED, GL_FLOAT, brickData);
	}
	if (glGetError() != GL_NO_ERROR) {
		return std::unexpected(DecoderError::InteropError);
	}

	return {};
}

bool DecoderBackend::isReady() const noexcept { return impl_ && impl_->ready; }

DecoderResult<void> DecoderBackend::buildEngine(const std::filesystem::path& onnxPath, uint32_t maxBatchSize) {
	auto builder = std::unique_ptr<nvinfer1::IBuilder, TRTDeleter>(nvinfer1::createInferBuilder(impl_->logger));
	if (!builder) {
		return std::unexpected(DecoderError::EngineBuildFailed);
	}

	const uint32_t explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
	auto network = std::unique_ptr<nvinfer1::INetworkDefinition, TRTDeleter>(builder->createNetworkV2(explicitBatch));
	if (!network) {
		return std::unexpected(DecoderError::EngineBuildFailed);
	}

	auto parser = std::unique_ptr<nvonnxparser::IParser, TRTDeleter>(nvonnxparser::createParser(*network, impl_->logger));
	if (!parser) {
		return std::unexpected(DecoderError::EngineBuildFailed);
	}
	if (!parser->parseFromFile(onnxPath.string().c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
		return std::unexpected(DecoderError::EngineBuildFailed);
	}

	auto config = std::unique_ptr<nvinfer1::IBuilderConfig, TRTDeleter>(builder->createBuilderConfig());
	if (!config) {
		return std::unexpected(DecoderError::EngineBuildFailed);
	}

	config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30);
	if (builder->platformHasFastFp16()) {
		config->setFlag(nvinfer1::BuilderFlag::kFP16);
	}

	auto profile = builder->createOptimizationProfile();
	bool profileUsed = false;
	const int networkInputs = network->getNbInputs();
	for (int i = 0; i < networkInputs; ++i) {
		const auto* input = network->getInput(i);
		const auto dims = input->getDimensions();
		bool hasDynamic = false;
		for (int d = 0; d < dims.nbDims; ++d) {
			if (dims.d[d] == -1) {
				hasDynamic = true;
				break;
			}
		}
		if (!hasDynamic) {
			continue;
		}

		nvinfer1::Dims minDims = dims;
		nvinfer1::Dims optDims = dims;
		nvinfer1::Dims maxDims = dims;

		for (int d = 0; d < dims.nbDims; ++d) {
			if (dims.d[d] == -1) {
				minDims.d[d] = 1;
				optDims.d[d] = static_cast<int>(std::min<uint32_t>(maxBatchSize, 32));
				maxDims.d[d] = static_cast<int>(maxBatchSize);
				break;
			}
		}

		if (!profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, minDims) ||
		    !profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, optDims) ||
		    !profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, maxDims)) {
			return std::unexpected(DecoderError::EngineBuildFailed);
		}
		profileUsed = true;
	}

	if (profileUsed) {
		config->addOptimizationProfile(profile);
	}

	auto serialized = std::unique_ptr<nvinfer1::IHostMemory, TRTDeleter>(builder->buildSerializedNetwork(*network, *config));
	if (!serialized || serialized->size() == 0) {
		return std::unexpected(DecoderError::EngineBuildFailed);
	}

	std::error_code ec;
	std::filesystem::create_directories(engineCachePath(onnxPath).parent_path(), ec);

	{
		std::ofstream out(engineCachePath(onnxPath), std::ios::binary | std::ios::trunc);
		if (out) {
			out.write(static_cast<const char*>(serialized->data()), static_cast<std::streamsize>(serialized->size()));
		}
	}

	impl_->engine.reset(impl_->runtime->deserializeCudaEngine(serialized->data(), serialized->size()));
	if (!impl_->engine) {
		return std::unexpected(DecoderError::EngineDeserializeFailed);
	}

	return {};
}

DecoderResult<void> DecoderBackend::loadCachedEngine(const std::filesystem::path& enginePath) {
	if (!std::filesystem::exists(enginePath)) {
		return std::unexpected(DecoderError::EngineDeserializeFailed);
	}

	std::ifstream in(enginePath, std::ios::binary | std::ios::ate);
	if (!in) {
		return std::unexpected(DecoderError::EngineDeserializeFailed);
	}

	const auto size = static_cast<size_t>(in.tellg());
	in.seekg(0, std::ios::beg);
	if (size == 0) {
		return std::unexpected(DecoderError::EngineDeserializeFailed);
	}

	std::vector<uint8_t> data(size);
	in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size()));
	if (!in) {
		return std::unexpected(DecoderError::EngineDeserializeFailed);
	}

	impl_->engine.reset(impl_->runtime->deserializeCudaEngine(data.data(), data.size()));
	if (!impl_->engine) {
		return std::unexpected(DecoderError::EngineDeserializeFailed);
	}

	return {};
}

std::filesystem::path DecoderBackend::engineCachePath(const std::filesystem::path& onnxPath) const {
	const std::string hash = hashFileFNV1aHex(onnxPath);
	const std::string arch = gpuArchTag();
	std::ostringstream fileName;
	fileName << "decoder_" << hash << "_" << arch << "_trt" << NV_TENSORRT_MAJOR << NV_TENSORRT_MINOR << ".engine";
	return onnxPath.parent_path() / ".trt_cache" / fileName.str();
}

}  // namespace vqvdb

#else

namespace vqvdb {

struct DecoderBackend::Impl {};

DecoderBackend::DecoderBackend() : impl_(std::make_unique<Impl>()) {}

DecoderBackend::~DecoderBackend() = default;

DecoderResult<void> DecoderBackend::init(const std::filesystem::path& onnxModelPath, uint32_t maxBatchSize) {
	if (!std::filesystem::exists(onnxModelPath)) {
		return std::unexpected(DecoderError::OnnxModelNotFound);
	}
	if (maxBatchSize == 0) {
		return std::unexpected(DecoderError::InvalidInput);
	}
	return std::unexpected(DecoderError::EngineBuildFailed);
}

DecoderResult<void> DecoderBackend::decodeBatch(std::span<const uint8_t> indices, uint32_t batchSize, uint32_t atlasTexture,
                                                std::span<const glm::ivec3> slotOffsets) {
	(void)indices;
	(void)batchSize;
	(void)atlasTexture;
	(void)slotOffsets;
	return std::unexpected(DecoderError::EngineNotLoaded);
}

bool DecoderBackend::isReady() const noexcept { return false; }

DecoderResult<void> DecoderBackend::buildEngine(const std::filesystem::path& onnxPath, uint32_t maxBatchSize) {
	(void)onnxPath;
	(void)maxBatchSize;
	return std::unexpected(DecoderError::EngineBuildFailed);
}

DecoderResult<void> DecoderBackend::loadCachedEngine(const std::filesystem::path& enginePath) {
	(void)enginePath;
	return std::unexpected(DecoderError::EngineDeserializeFailed);
}

std::filesystem::path DecoderBackend::engineCachePath(const std::filesystem::path& onnxPath) const { return onnxPath; }

}  // namespace vqvdb

#endif
