#include "OnnxBackend.hpp"

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <thread>

// Include the binary data for the embedded model.
// This header should define g_model_data (unsigned char*) and g_model_data_size (size_t).
#include "bin/bin_model.h"

namespace {

// Helper to convert our DataType enum to ONNX tensor element type
ONNXTensorElementDataType toOnnxDataType(DataType dtype) {
	switch (dtype) {
		case DataType::FLOAT32:
			return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
		case DataType::UINT8:
			return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
	}
	throw std::runtime_error("Unsupported data type");
}

// Helper to get size of data type in bytes
size_t getDataTypeSize(DataType dtype) {
	switch (dtype) {
		case DataType::FLOAT32:
			return sizeof(float);
		case DataType::UINT8:
			return sizeof(uint8_t);
	}
	throw std::runtime_error("Unsupported data type");
}

// Helper to calculate total number of elements from shape
size_t calculateTotalElements(const std::vector<int64_t>& shape) {
	size_t total = 1;
	for (int64_t dim : shape) {
		total *= static_cast<size_t>(dim);
	}
	return total;
}

}  // namespace

OnnxBackend::OnnxBackend(const CodecConfig& config)
    : env_(ORT_LOGGING_LEVEL_WARNING, "VQVAECodec"), useGpu_(config.device == CodecConfig::Device::CUDA) {
	// Configure session options
	sessionOptions_.SetIntraOpNumThreads(std::max(1, (int)std::thread::hardware_concurrency() / 2));
	sessionOptions_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

	// Configure execution provider
	if (useGpu_) {
		try {
			Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions_, 0));
			std::cout << "OnnxBackend: CUDA execution provider enabled." << std::endl;
		} catch (const Ort::Exception& e) {
			std::cerr << "Warning: Failed to enable CUDA execution provider: " << e.what() << ". Falling back to CPU." << std::endl;
			useGpu_ = false;
		}
	}

	if (!useGpu_) {
		std::cout << "OnnxBackend: Using CPU execution provider." << std::endl;
	}

	// Load model data
	std::vector<uint8_t> modelData = load_model_data(config.source);

	// Create session
	try {
		session_ = std::make_unique<Ort::Session>(env_, modelData.data(), modelData.size(), sessionOptions_);
	} catch (const Ort::Exception& e) {
		throw std::runtime_error("Failed to create ONNX session: " + std::string(e.what()));
	}

	// Get input/output names
	size_t numInputNodes = session_->GetInputCount();
	size_t numOutputNodes = session_->GetOutputCount();

	inputNames_.reserve(numInputNodes);
	outputNames_.reserve(numOutputNodes);

	for (size_t i = 0; i < numInputNodes; i++) {
		char* inputName = session_->GetInputNameAllocated(i, allocator_).get();
		inputNames_.push_back(inputName);
	}

	for (size_t i = 0; i < numOutputNodes; i++) {
		char* outputName = session_->GetOutputNameAllocated(i, allocator_).get();
		outputNames_.push_back(outputName);
	}

	std::cout << "OnnxBackend: Model loaded with " << numInputNodes << " inputs and " << numOutputNodes << " outputs." << std::endl;

	// Initialize latent shape
	initialize_latent_shape();
}

std::vector<uint8_t> OnnxBackend::load_model_data(const ModelSource& source) {
	if (std::holds_alternative<EmbeddedModel>(source)) {
		std::cout << "OnnxBackend: Loading embedded model." << std::endl;
		return std::vector<uint8_t>(g_model_data, g_model_data + g_model_data_size);
	}

	if (std::holds_alternative<std::filesystem::path>(source)) {
		const auto& path = std::get<std::filesystem::path>(source);
		std::cout << "OnnxBackend: Loading model from path: " << path << std::endl;

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

	throw std::logic_error("Unsupported model source type.");
}

void OnnxBackend::initialize_latent_shape() {
	// Create a dummy input tensor to probe the model
	std::vector<int64_t> dummyShape = {1, 1, 8, 8, 8};
	size_t totalElements = calculateTotalElements(dummyShape);
	std::vector<float> dummyData(totalElements, 0.0f);

	// Create input tensor
	Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	Ort::Value inputTensor =
	    Ort::Value::CreateTensor<float>(memoryInfo, dummyData.data(), totalElements, dummyShape.data(), dummyShape.size());

	// Run inference
	try {
		std::vector<Ort::Value> inputTensors;
		inputTensors.push_back(std::move(inputTensor));

		auto outputTensors = session_->Run(Ort::RunOptions{nullptr}, inputNames_.data(), inputTensors.data(), 1, outputNames_.data(), 1);

		// Get the shape of the first output (should be the encoded indices)
		auto tensorInfo = outputTensors[0].GetTensorTypeAndShapeInfo();
		std::vector<int64_t> outputShape = tensorInfo.GetShape();

		if (outputShape.size() <= 1) {
			throw std::runtime_error("Encoder output has invalid dimensions.");
		}

		// The latent shape is the shape excluding the batch dimension
		latentShape_.assign(outputShape.begin() + 1, outputShape.end());

		std::cout << "OnnxBackend: Detected latent shape: (";
		for (size_t i = 0; i < latentShape_.size(); ++i) {
			std::cout << latentShape_[i] << (i == latentShape_.size() - 1 ? "" : ", ");
		}
		std::cout << ")" << std::endl;

	} catch (const Ort::Exception& e) {
		throw std::runtime_error("Failed to initialize latent shape: " + std::string(e.what()));
	}
}

Tensor OnnxBackend::encode(const TensorView& leafBatch) const {
	if (leafBatch.dtype != DataType::FLOAT32) {
		throw std::runtime_error("encode expects FLOAT32 data.");
	}

	// Create input tensor
	Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	Ort::Value inputTensor =
	    Ort::Value::CreateTensor<float>(memoryInfo, const_cast<float*>(static_cast<const float*>(leafBatch.data)),
	                                    calculateTotalElements(leafBatch.shape), leafBatch.shape.data(), leafBatch.shape.size());

	// Run inference
	try {
		std::vector<Ort::Value> inputTensors;
		inputTensors.push_back(std::move(inputTensor));

		auto outputTensors = session_->Run(Ort::RunOptions{nullptr}, inputNames_.data(), inputTensors.data(), 1, outputNames_.data(), 1);

		// Get output tensor info
		auto tensorInfo = outputTensors[0].GetTensorTypeAndShapeInfo();
		std::vector<int64_t> outputShape = tensorInfo.GetShape();

		// Create result tensor
		Tensor result;
		result.shape = outputShape;
		result.dtype = DataType::UINT8;

		// Copy data from ONNX tensor to result buffer
		const uint8_t* outputData = outputTensors[0].GetTensorData<uint8_t>();
		size_t totalElements = calculateTotalElements(outputShape);
		size_t totalBytes = totalElements * getDataTypeSize(DataType::UINT8);

		result.buffer.resize(totalBytes);
		std::memcpy(result.buffer.data(), outputData, totalBytes);

		return result;

	} catch (const Ort::Exception& e) {
		throw std::runtime_error("Failed to encode: " + std::string(e.what()));
	}
}

Tensor OnnxBackend::decode(const TensorView& indices) const {
	if (indices.dtype != DataType::UINT8) {
		throw std::runtime_error("decode expects UINT8 data.");
	}

	// Create input tensor
	Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	Ort::Value inputTensor =
	    Ort::Value::CreateTensor<uint8_t>(memoryInfo, const_cast<uint8_t*>(static_cast<const uint8_t*>(indices.data)),
	                                      calculateTotalElements(indices.shape), indices.shape.data(), indices.shape.size());

	// Run inference
	try {
		std::vector<Ort::Value> inputTensors;
		inputTensors.push_back(std::move(inputTensor));

		auto outputTensors = session_->Run(Ort::RunOptions{nullptr}, inputNames_.data(), inputTensors.data(), 1, outputNames_.data(), 1);

		// Get output tensor info
		auto tensorInfo = outputTensors[0].GetTensorTypeAndShapeInfo();
		std::vector<int64_t> outputShape = tensorInfo.GetShape();

		// Create result tensor
		Tensor result;
		result.shape = outputShape;
		result.dtype = DataType::FLOAT32;

		// Copy data from ONNX tensor to result buffer
		const float* outputData = outputTensors[0].GetTensorData<float>();
		size_t totalElements = calculateTotalElements(outputShape);
		size_t totalBytes = totalElements * getDataTypeSize(DataType::FLOAT32);

		result.buffer.resize(totalBytes);
		std::memcpy(result.buffer.data(), outputData, totalBytes);

		return result;

	} catch (const Ort::Exception& e) {
		throw std::runtime_error("Failed to decode: " + std::string(e.what()));
	}
}