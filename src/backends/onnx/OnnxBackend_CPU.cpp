#include "OnnxBackend_CPU.hpp"

#include <iostream>
#include <stdexcept>


OnnxCpuBackend::OnnxCpuBackend(const CodecConfig& config) {
	init(config);
	std::cout << "OnnxCpuBackend: Using CPU execution provider." << std::endl;
}

void OnnxCpuBackend::configure_execution_provider() {}

void OnnxCpuBackend::initialize_latent_shape_impl() {
	std::vector<int64_t> dummyShape = {1, 1, 8, 8, 8};
	size_t totalElements = calculateTotalElements(dummyShape);
	std::vector<float> dummyData(totalElements, 0.0f);

	try {
		if (encoderInputNames_.empty()) {
			throw std::runtime_error("No input names found for encoder");
		}
		if (encoderOutputNames_.empty()) {
			throw std::runtime_error("No output names found for encoder");
		}

		std::cout << "OnnxCpuBackend: Running encoder with input name: '" << encoderInputNames_[0] << "'" << std::endl;

		Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
		Ort::Value inputTensor =
		    Ort::Value::CreateTensor<float>(memoryInfo, dummyData.data(), totalElements, dummyShape.data(), dummyShape.size());

		std::vector<Ort::Value> inputTensors;
		inputTensors.push_back(std::move(inputTensor));

		auto outputTensors = encoderSession_->Run(Ort::RunOptions{nullptr}, encoderInputNames_.data(), inputTensors.data(), 1,
		                                          encoderOutputNames_.data(), 1);

		auto tensorInfo = outputTensors[0].GetTensorTypeAndShapeInfo();
		std::vector<int64_t> outputShape = tensorInfo.GetShape();

		if (outputShape.size() <= 1) {
			throw std::runtime_error("Encoder output has invalid dimensions.");
		}

		latentShape_.assign(outputShape.begin() + 1, outputShape.end());

		std::cout << "OnnxCpuBackend: Detected latent shape: (";
		for (size_t i = 0; i < latentShape_.size(); ++i) {
			std::cout << latentShape_[i] << (i == latentShape_.size() - 1 ? "" : ", ");
		}
		std::cout << ")" << std::endl;
	} catch (const Ort::Exception& e) {
		throw std::runtime_error("Failed to initialize latent shape: " + std::string(e.what()));
	}
}

Tensor OnnxCpuBackend::encode_impl(const TensorView& leafBatch) const {
	if (leafBatch.dtype != DataType::FLOAT32) {
		throw std::runtime_error("encode expects FLOAT32 data.");
	}

	try {
		Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
		Ort::Value inputTensor =
		    Ort::Value::CreateTensor<float>(memoryInfo, const_cast<float*>(static_cast<const float*>(leafBatch.data)),
		                                    calculateTotalElements(leafBatch.shape), leafBatch.shape.data(), leafBatch.shape.size());

		std::vector<Ort::Value> inputTensors;
		inputTensors.push_back(std::move(inputTensor));

		auto outputTensors = encoderSession_->Run(Ort::RunOptions{nullptr}, encoderInputNames_.data(), inputTensors.data(), 1,
		                                          encoderOutputNames_.data(), 1);

		auto tensorInfo = outputTensors[0].GetTensorTypeAndShapeInfo();
		std::vector<int64_t> outputShape = tensorInfo.GetShape();

		Tensor result;
		result.shape = outputShape;
		result.dtype = DataType::UINT8;

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

Tensor OnnxCpuBackend::decode_impl(const TensorView& indices) const {
	if (indices.dtype != DataType::UINT8) {
		throw std::runtime_error("decode expects UINT8 data.");
	}

	try {
		Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
		Ort::Value inputTensor =
		    Ort::Value::CreateTensor<uint8_t>(memoryInfo, const_cast<uint8_t*>(static_cast<const uint8_t*>(indices.data)),
		                                      calculateTotalElements(indices.shape), indices.shape.data(), indices.shape.size());

		std::vector<Ort::Value> inputTensors;
		inputTensors.push_back(std::move(inputTensor));

		auto outputTensors = decoderSession_->Run(Ort::RunOptions{nullptr}, decoderInputNames_.data(), inputTensors.data(), 1,
		                                          decoderOutputNames_.data(), 1);

		auto tensorInfo = outputTensors[0].GetTensorTypeAndShapeInfo();
		std::vector<int64_t> outputShape = tensorInfo.GetShape();

		Tensor result;
		result.shape = outputShape;
		result.dtype = DataType::FLOAT32;

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