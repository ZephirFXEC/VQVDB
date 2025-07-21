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