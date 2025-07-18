#include "OnnxBackend.hpp"

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <thread>

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

// Helper to get input/output names from session
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

}  // namespace

OnnxBackend::OnnxBackend(const CodecConfig& config)
    : env_(ORT_LOGGING_LEVEL_WARNING, "VQVAECodec"), useGpu_(config.device == CodecConfig::Device::CUDA) {
	// Configure session options
	sessionOptions_.SetIntraOpNumThreads(std::max(1, (int)std::thread::hardware_concurrency() / 2));
	sessionOptions_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

	// Configure execution provider
	if (useGpu_) {
		try {
			OrtCUDAProviderOptions cuda_options{};
			cuda_options.device_id = 0;
			cuda_options.arena_extend_strategy = 1;
			cuda_options.gpu_mem_limit = SIZE_MAX;
			cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchDefault;
			cuda_options.do_copy_in_default_stream = 1;

			sessionOptions_.AppendExecutionProvider_CUDA(cuda_options);
			std::cout << "OnnxBackend: CUDA execution provider enabled." << std::endl;
		} catch (const std::exception& e) {
			std::cerr << "Warning: Failed to enable CUDA execution provider: " << e.what() << ". Falling back to CPU." << std::endl;
			useGpu_ = false;
		}
	}

	if (!useGpu_) {
		std::cout << "OnnxBackend: Using CPU execution provider." << std::endl;
	}

	// Setup encoder and decoder sessions
	setup_sessions(config.source);

	// Initialize latent shape
	initialize_latent_shape();
}

void OnnxBackend::setup_sessions(const ModelSource& source) {
	std::vector<uint8_t> encoderData, decoderData;

	if (std::holds_alternative<EmbeddedModel>(source)) {
		/*std::cout << "OnnxBackend: Loading embedded models." << std::endl;
		encoderData = std::vector<uint8_t>(g_encoder_data, g_encoder_data + g_encoder_data_size);
		decoderData = std::vector<uint8_t>(g_decoder_data, g_decoder_data + g_decoder_data_size);*/
	} else if (std::holds_alternative<OnnxModelPaths>(source)) {
		const auto& paths = std::get<OnnxModelPaths>(source);
		std::cout << "OnnxBackend: Loading models from paths:" << std::endl;
		std::cout << "  Encoder: " << paths.encoder_path << std::endl;
		std::cout << "  Decoder: " << paths.decoder_path << std::endl;

		encoderData = load_model_data(paths.encoder_path);
		decoderData = load_model_data(paths.decoder_path);
	} else if (std::holds_alternative<std::filesystem::path>(source)) {
		const auto& basePath = std::get<std::filesystem::path>(source);
		auto encoderPath = basePath / "encoder.onnx";
		auto decoderPath = basePath / "decoder.onnx";

		std::cout << "OnnxBackend: Loading models from directory:" << std::endl;
		std::cout << "  Encoder: " << encoderPath << std::endl;
		std::cout << "  Decoder: " << decoderPath << std::endl;

		encoderData = load_model_data(encoderPath);
		decoderData = load_model_data(decoderPath);
	} else {
		throw std::logic_error("Unsupported model source type.");
	}

	// Create sessions
	try {
		encoderSession_ = std::make_unique<Ort::Session>(env_, encoderData.data(), encoderData.size(), sessionOptions_);
		decoderSession_ = std::make_unique<Ort::Session>(env_, decoderData.data(), decoderData.size(), sessionOptions_);
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

std::vector<uint8_t> OnnxBackend::load_model_data(const std::filesystem::path& path) {
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

void OnnxBackend::initialize_latent_shape() {
	// Create a dummy input tensor to probe the encoder
	std::vector<int64_t> dummyShape = {1, 1, 8, 8, 8};
	size_t totalElements = calculateTotalElements(dummyShape);
	std::vector<float> dummyData(totalElements, 0.0f);

	// Create input tensor
	Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	Ort::Value inputTensor =
	    Ort::Value::CreateTensor<float>(memoryInfo, dummyData.data(), totalElements, dummyShape.data(), dummyShape.size());

	// Run encoder inference
	try {
		if (encoderInputNames_.empty()) {
			throw std::runtime_error("No input names found for encoder");
		}
		if (encoderOutputNames_.empty()) {
			throw std::runtime_error("No output names found for encoder");
		}

		std::vector<Ort::Value> inputTensors;
		inputTensors.push_back(std::move(inputTensor));

		std::cout << "OnnxBackend: Running encoder with input name: '" << encoderInputNames_[0] << "'" << std::endl;

		auto outputTensors = encoderSession_->Run(Ort::RunOptions{nullptr}, encoderInputNames_.data(), inputTensors.data(), 1,
		                                          encoderOutputNames_.data(), 1);

		// Get the shape of the encoder output
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

	// Run encoder inference
	try {
		std::vector<Ort::Value> inputTensors;
		inputTensors.push_back(std::move(inputTensor));

		auto outputTensors = encoderSession_->Run(Ort::RunOptions{nullptr}, encoderInputNames_.data(), inputTensors.data(), 1,
		                                          encoderOutputNames_.data(), 1);

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

	// Run decoder inference
	try {
		std::vector<Ort::Value> inputTensors;
		inputTensors.push_back(std::move(inputTensor));

		auto outputTensors = decoderSession_->Run(Ort::RunOptions{nullptr}, decoderInputNames_.data(), inputTensors.data(), 1,
		                                          decoderOutputNames_.data(), 1);

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