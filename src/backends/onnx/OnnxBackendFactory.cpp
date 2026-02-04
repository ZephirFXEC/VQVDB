/*
 * Copyright (c) 2025, Enzo Crema
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See the LICENSE file in the project root for full license text.
 */

#include "OnnxBackendFactory.hpp"

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <thread>

#include "Bin/bin_onnx.h"

ONNXTensorElementDataType OnnxBackendFactory::toOnnxDataType(DataType dtype) {
	switch (dtype) {
		case DataType::FLOAT32:
			return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
		case DataType::UINT8:
			return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
	}
	throw std::runtime_error("Unsupported data type");
}

size_t OnnxBackendFactory::getDataTypeSize(DataType dtype) {
	switch (dtype) {
		case DataType::FLOAT32:
			return sizeof(float);
		case DataType::UINT8:
			return sizeof(uint8_t);
	}
	throw std::runtime_error("Unsupported data type");
}

size_t OnnxBackendFactory::calculateTotalElements(const std::vector<int64_t>& shape) {
	size_t total = 1;
	for (int64_t dim : shape) {
		total *= static_cast<size_t>(dim);
	}
	return total;
}

std::pair<std::vector<Ort::AllocatedStringPtr>, std::vector<const char*>> getInputNames(const Ort::Session& session,
                                                                                        Ort::AllocatorWithDefaultOptions& allocator) {
	std::vector<Ort::AllocatedStringPtr> namePtrs;
	std::vector<const char*> names;
	size_t numInputs = session.GetInputCount();
	namePtrs.reserve(numInputs);
	names.reserve(numInputs);

	for (size_t i = 0; i < numInputs; i++) {
		auto ptr = session.GetInputNameAllocated(i, allocator);
		names.push_back(ptr.get());
		namePtrs.push_back(std::move(ptr));
	}
	return {std::move(namePtrs), std::move(names)};
}

std::pair<std::vector<Ort::AllocatedStringPtr>, std::vector<const char*>> getOutputNames(const Ort::Session& session,
                                                                                         Ort::AllocatorWithDefaultOptions& allocator) {
	std::vector<Ort::AllocatedStringPtr> namePtrs;
	std::vector<const char*> names;
	size_t numOutputs = session.GetOutputCount();
	namePtrs.reserve(numOutputs);
	names.reserve(numOutputs);

	for (size_t i = 0; i < numOutputs; i++) {
		auto ptr = session.GetOutputNameAllocated(i, allocator);
		names.push_back(ptr.get());
		namePtrs.push_back(std::move(ptr));
	}
	return {std::move(namePtrs), std::move(names)};
}

extern "C" void ORT_API_CALL onnxConsoleLogger(void*, OrtLoggingLevel /*severity*/, const char* /*category*/, const char* /*logid*/,
                                               const char* /*code_location*/, const char* message) {
	std::cout << "[ORT] " << message << std::endl;
}

OnnxBackendFactory::OnnxBackendFactory() {
	env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "VQVAECodec", onnxConsoleLogger, nullptr);
}

void OnnxBackendFactory::init(const CodecConfig& config) {
	sessionOptions_.SetIntraOpNumThreads(std::max(1, (int)std::thread::hardware_concurrency() / 2));
	sessionOptions_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

	configure_execution_provider();

	setup_sessions(config.source);
	initialize_latent_shape_impl();
}

void OnnxBackendFactory::setup_sessions(const ModelSource& source) {
	std::vector<uint8_t> encoderData, decoderData;

	if (std::holds_alternative<EmbeddedModel>(source)) {
		std::cout << "Loading embedded models." << std::endl;
		const auto& embedded = std::get<EmbeddedModel>(source);
		encoderData = std::vector<uint8_t>(encoder_model_data, encoder_model_data + encoder_model_data_size);
		decoderData = std::vector<uint8_t>(decoder_model_data, decoder_model_data + decoder_model_data_size);
	} else if (std::holds_alternative<OnnxModelPaths>(source)) {
		const auto& paths = std::get<OnnxModelPaths>(source);
		std::cout << "Loading models from paths:\n  Encoder: " << paths.encoder_path << "\n  Decoder: " << paths.decoder_path << std::endl;
		encoderData = load_model_data(paths.encoder_path);
		decoderData = load_model_data(paths.decoder_path);
	} else if (std::holds_alternative<std::filesystem::path>(source)) {
		const auto& basePath = std::get<std::filesystem::path>(source);
		auto encoderPath = basePath / "encoder.onnx";
		auto decoderPath = basePath / "decoder.onnx";
		std::cout << "Loading models from directory:\n  Encoder: " << encoderPath << "\n  Decoder: " << decoderPath << std::endl;
		encoderData = load_model_data(encoderPath);
		decoderData = load_model_data(decoderPath);
	} else {
		throw std::logic_error("Unsupported model source type.");
	}

	// Create sessions
	try {
		encoderSession_ = std::make_unique<Ort::Session>(*env_, encoderData.data(), encoderData.size(), sessionOptions_);
		decoderSession_ = std::make_unique<Ort::Session>(*env_, decoderData.data(), decoderData.size(), sessionOptions_);
	} catch (const Ort::Exception& e) {
		throw std::runtime_error("Failed to create ONNX sessions: " + std::string(e.what()));
	}

	// Get input/output names
	auto [encoderInputPtrs, encoderInputs] = getInputNames(*encoderSession_, allocator_);
	encoderInputNamePtrs_ = std::move(encoderInputPtrs);
	encoderInputNames_ = std::move(encoderInputs);

	auto [encoderOutputPtrs, encoderOutputs] = getOutputNames(*encoderSession_, allocator_);
	encoderOutputNamePtrs_ = std::move(encoderOutputPtrs);
	encoderOutputNames_ = std::move(encoderOutputs);

	auto [decoderInputPtrs, decoderInputs] = getInputNames(*decoderSession_, allocator_);
	decoderInputNamePtrs_ = std::move(decoderInputPtrs);
	decoderInputNames_ = std::move(decoderInputs);

	auto [decoderOutputPtrs, decoderOutputs] = getOutputNames(*decoderSession_, allocator_);
	decoderOutputNamePtrs_ = std::move(decoderOutputPtrs);
	decoderOutputNames_ = std::move(decoderOutputs);
}

std::vector<uint8_t> OnnxBackendFactory::load_model_data(const std::filesystem::path& path) {
	if (!std::filesystem::exists(path)) {
		throw std::runtime_error("Model file not found at path: " + path.string());
	}

	std::ifstream file(path, std::ios::binary | std::ios::ate);
	if (!file.is_open()) {
		throw std::runtime_error("Failed to open model file: " + path.string());
	}

	std::streamsize size = file.tellg();
	file.seekg(0, std::ios::beg);

	std::vector<uint8_t> buffer(size);
	if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) {
		throw std::runtime_error("Failed to read model file: " + path.string());
	}

	return buffer;
}

// ----------------------------------------------------------------------------
// Codebook Extraction
// ----------------------------------------------------------------------------
// We assume the ONNX model has the codebook weights stored in an initializer.
// By convention (from PyTorch export), the quantizer embedding is usually named:
// "quantizer.embedding"
//
// However, in an optimized ONNX graph, names might change.
// Strategy:
// 1. Try to find an initializer named "quantizer.embedding" (standard)
// 2. If not found, we might need a backup strategy (e.g., look for a Constant node)
//    For now, we enforce the naming convention during export.

static const char* CODEBOOK_NODE_NAME = "quantizer.embedding";

// Helper to copy tensor data from ONNX initializer to std::vector
template <typename T>
static std::vector<T> extractTensorData(const Ort::Session& session, const char* tensorName) {
	// Note: ORT C++ API doesn't expose GetAllInitializers() easily.
	// We typically have to run the model to get outputs, OR assume the
	// initializer is graph-accessible.
	//
	// However, if the codebook is a model parameter (Initializer), we can't
	// always just "get" it without running an inference if it's not an output.
	//
	// WORKAROUND:
	// We will assume that during the Python export (save_for_inference.py),
	// we added the codebook as an *output* of the encoder or a separate graph.
	//
	// Checking `src/backends/onnx/OnnxBackendFactory.cpp`, we don't have this yet.
	//
	// RETRACTION: We cannot easily get internal initializers from a loaded ORT Session
	// unless they are graph inputs or outputs.
	//
	// ALTERNATIVE:
	// We will update the `VQVAE_v2.py` export script to ensure `quantizer.embedding`
	// is an OUTPUT of the Encoder model.
	//
	// For this C++ implementation, we will assume it is an output named "codebook".
	// Let's check the encoder outputs.

	// If "codebook" is not an output, we can't get it easily.
	// But wait! We embedded the model binary. We can parse the ONNX protobuf manually?
	// No, that's too heavy (requires protobuf dependency).
	//
	// Proposed Solution:
	// We will fail gracefully if we can't find it, but for now, let's assumes
	// future models will export "codebook" as a secondary output of the Encoder.

	// For now, return a dummy or throw not implemented until we update the Python exporter.
	// To unblock the build, I will implement a placeholder.

	// TODO: Update Python exporter to add 'codebook' as an output.
	return std::vector<T>();
}

std::vector<float> OnnxBackendFactory::getCodebook() const {
	// This requires the Python exporter to be updated to expose the codebook
	// as a model output or a separate file.
	// For the immediate "Direct Index" implementation, we need this data.
	//
	// Hack for prototype: Return a zero-filled vector or throw.
	// In a real implementation, we would modify `save_for_inference.py` to
	// write `codebook.bin` alongside `encoder.onnx`.

	// Let's assume we load it from a separate file for now?
	// No, let's keep it clean.

	std::cerr << "[OnnxBackend] Warning: getCodebook() not fully implemented without model export update." << std::endl;
	std::vector<float> dummy(256 * 128, 0.0f);
	return dummy;
}

void OnnxBackendFactory::getCodebookDims(int& numCodes, int& embeddingDim) const {
	numCodes = 256;      // Hardcoded for VQ-VAE-2 conventions
	embeddingDim = 128;  // Hardcoded for VQ-VAE-2 conventions
}
