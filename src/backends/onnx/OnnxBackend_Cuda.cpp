/*
 * Copyright (c) 2025, Enzo Crema
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See the LICENSE file in the project root for full license text.
 */

#include "OnnxBackend_Cuda.hpp"

#include <iostream>
#include <stdexcept>

OnnxCudaBackend::OnnxCudaBackend(const CodecConfig& config) {
	std::cout << "OnnxCudaBackend: Initialising …" << std::endl;
	init(config);
}

void OnnxCudaBackend::configure_execution_provider() {
	OrtCUDAProviderOptions cuda_options{};
	cuda_options.device_id = 0;
	cuda_options.arena_extend_strategy = 1;
	cuda_options.gpu_mem_limit = SIZE_MAX;
	cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchHeuristic;
	cuda_options.do_copy_in_default_stream = 1;

	try {
		sessionOptions_.AppendExecutionProvider_CUDA(cuda_options);
		std::cout << "OnnxCudaBackend: CUDA execution provider enabled." << std::endl;
	} catch (const std::exception& e) {
		std::cerr << "OnnxCudaBackend: FATAL – could not enable CUDA: " << e.what() << std::endl;
		throw;
	}
}

void OnnxCudaBackend::initialize_latent_shape_impl() {
	// Create dummy input
	std::vector<int64_t> dummyShape = {1, 1, 8, 8, 8};
	size_t totalElements = calculateTotalElements(dummyShape);
	std::vector<float> dummyData(totalElements, 0.0f);

	try {
		if (encoderInputNames_.empty() || encoderOutputNames_.empty()) {
			throw std::runtime_error("No input/output names for encoder");
		}

		std::cout << "OnnxCudaBackend: Running encoder with input name: " << encoderInputNames_[0] << std::endl;

		Ort::IoBinding ioBinding(*encoderSession_);

		// Input on CPU (ORT will copy to GPU)
		Ort::MemoryInfo cpuMemInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
		Ort::Value inputTensor =
		    Ort::Value::CreateTensor<float>(cpuMemInfo, dummyData.data(), totalElements, dummyShape.data(), dummyShape.size());
		ioBinding.BindInput(encoderInputNames_[0], inputTensor);

		// Bind output to CPU (auto-copies from GPU)
		ioBinding.BindOutput(encoderOutputNames_[0], cpuMemInfo);

		encoderSession_->Run(Ort::RunOptions{nullptr}, ioBinding);

		std::vector<Ort::Value> outputTensors = ioBinding.GetOutputValues();

		auto tensorInfo = outputTensors[0].GetTensorTypeAndShapeInfo();
		std::vector<int64_t> outputShape = tensorInfo.GetShape();

		if (outputShape.size() <= 1) {
			throw std::runtime_error("Encoder output has invalid dimensions.");
		}

		latentShape_.assign(outputShape.begin() + 1, outputShape.end());

		std::cout << "OnnxCudaBackend: Detected latent shape: (";
		for (size_t i = 0; i < latentShape_.size(); ++i) {
			std::cout << latentShape_[i] << (i == latentShape_.size() - 1 ? "" : ", ");
		}
		std::cout << ")" << std::endl;
	} catch (const Ort::Exception& e) {
		throw std::runtime_error("Failed to initialize latent shape: " + std::string(e.what()));
	}
}

Tensor OnnxCudaBackend::encode_impl(const TensorView& leafBatch) const {
	if (leafBatch.dtype != DataType::FLOAT32) {
		throw std::runtime_error("encode expects FLOAT32 data.");
	}

	try {
		Ort::IoBinding ioBinding(*encoderSession_);

		// Input on CPU
		Ort::MemoryInfo cpuMemInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
		Ort::Value inputTensor =
		    Ort::Value::CreateTensor<float>(cpuMemInfo, const_cast<float*>(static_cast<const float*>(leafBatch.data)),
		                                    calculateTotalElements(leafBatch.shape), leafBatch.shape.data(), leafBatch.shape.size());
		ioBinding.BindInput(encoderInputNames_[0], inputTensor);

		// Bind output to CPU (auto-copy from GPU)
		ioBinding.BindOutput(encoderOutputNames_[0], cpuMemInfo);

		encoderSession_->Run(Ort::RunOptions{nullptr}, ioBinding);

		std::vector<Ort::Value> outputTensors = ioBinding.GetOutputValues();

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

Tensor OnnxCudaBackend::decode_impl(const TensorView& indices) const {
	if (indices.dtype != DataType::UINT8) {
		throw std::runtime_error("decode expects UINT8 data.");
	}

	try {
		Ort::IoBinding ioBinding(*decoderSession_);

		// Input on CPU
		Ort::MemoryInfo cpuMemInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
		Ort::Value inputTensor =
		    Ort::Value::CreateTensor<uint8_t>(cpuMemInfo, const_cast<uint8_t*>(static_cast<const uint8_t*>(indices.data)),
		                                      calculateTotalElements(indices.shape), indices.shape.data(), indices.shape.size());
		ioBinding.BindInput(decoderInputNames_[0], inputTensor);

		// Bind output to CPU (auto-copy from GPU)
		ioBinding.BindOutput(decoderOutputNames_[0], cpuMemInfo);

		decoderSession_->Run(Ort::RunOptions{nullptr}, ioBinding);

		std::vector<Ort::Value> outputTensors = ioBinding.GetOutputValues();

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