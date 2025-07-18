//
// Created by zphrfx on 05/07/2025.
//

#include "TorchBackend.hpp"

#include <torch/script.h>
#include <torch/torch.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

// Include the binary data for the embedded model.
// This header should define g_model_data (unsigned char*) and g_model_data_size (size_t).
#include "bin/bin_model.h"

namespace {

// --- Model Loading ---

torch::jit::Module load_model_from_stream(std::istream& stream, const torch::Device& device) {
	torch::jit::Module module;
	try {
		// Load from the stream. The device argument tells libtorch where to place the model directly.
		module = torch::jit::load(stream, device);
	} catch (const c10::Error& e) {
		throw std::runtime_error("Failed to load TorchScript model from stream: " + std::string(e.what()));
	}
	module.eval();  // Set to evaluation mode
	return module;
}
torch::jit::Module load_model(const ModelSource& source, const torch::Device& device) {
	if (std::holds_alternative<EmbeddedModel>(source)) {
		std::cout << "TorchBackend: Loading embedded model." << std::endl;

		std::stringstream model_stream;
		model_stream.write(reinterpret_cast<const char*>(g_model_data), g_model_data_size);

		// The stream is now ready to be used by libtorch.
		return load_model_from_stream(model_stream, device);
	}

	if (std::holds_alternative<std::filesystem::path>(source)) {
		const auto& path = std::get<std::filesystem::path>(source);
		std::cout << "TorchBackend: Loading model from path: " << path << std::endl;
		if (!std::filesystem::exists(path)) {
			throw std::runtime_error("Model file not found at path: " + path.string());
		}
		std::ifstream stream(path, std::ios::binary);
		return load_model_from_stream(stream, device);
	}

	throw std::logic_error("Unsupported model source type.");
}

// --- Device and Threading ---

torch::Device resolve_torch_device(const CodecConfig::Device device_enum) {
	if (device_enum == CodecConfig::Device::CUDA) {
		if (torch::cuda::is_available()) {
			return torch::kCUDA;
		}
		std::cerr << "Warning: CUDA requested but not available. Falling back to CPU." << std::endl;
	}
	return torch::kCPU;
}

void configure_cpu_threads() {
	const int num_threads = std::max(1, (int)std::thread::hardware_concurrency() / 2);
	torch::set_num_threads(num_threads);
	std::cout << "TorchBackend: Set CPU threads to " << torch::get_num_threads() << std::endl;
}

}  // namespace


// --- TorchBackend Implementation ---
TorchBackend::TorchBackend(const CodecConfig& config)
    : device_(resolve_torch_device(config.device)),
      module_(load_model(config.source, device_)),
      encodeMethod_(module_.get_method("encode")),
      decodeMethod_(module_.get_method("decode")) {
	if (device_.is_cpu()) {
		configure_cpu_threads();
	}

	initialize_latent_shape();
	std::cout << "TorchBackend: Model successfully loaded onto device: " << device_ << std::endl;
}

void TorchBackend::initialize_latent_shape() {
	torch::NoGradGuard nograd;
	// Create a dummy input tensor on the target device to probe the model
	const auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(device_);
	torch::Tensor dummyInput = torch::zeros({1, 1, 8, 8, 8}, opts);

	// Run a dummy encode pass to get the output shape
	const auto idx = encodeMethod_({dummyInput}).toTensor();
	const auto& sizes = idx.sizes();

	if (sizes.size() <= 1) {
		throw std::runtime_error("Encoder output has invalid dimensions.");
	}

	// The latent shape is the shape of the tensor, excluding the batch dimension
	latentShape_.assign(sizes.begin() + 1, sizes.end());

	std::cout << "TorchBackend: Detected latent shape: (";
	for (size_t i = 0; i < latentShape_.size(); ++i) {
		std::cout << latentShape_[i] << (i == latentShape_.size() - 1 ? "" : ", ");
	}
	std::cout << ")" << std::endl;
}


// Helper to convert our DataType enum to a torch::ScalarType
static torch::ScalarType toTorchDType(DataType dtype) {
	switch (dtype) {
		case DataType::FLOAT32:
			return torch::kFloat32;
		case DataType::UINT8:
			return torch::kUInt8;
	}
	throw std::runtime_error("Unsupported data type");
}

Tensor TorchBackend::encode(const TensorView& leafBatch) const {
	if (leafBatch.dtype != DataType::FLOAT32) {
		throw std::runtime_error("encode expects FLOAT32 data.");
	}

	torch::NoGradGuard g;

	// 1. Create a non-owning torch::Tensor from the input TensorView.
	//    This does not copy the data, it only wraps the existing memory.
	auto options = torch::TensorOptions().dtype(toTorchDType(leafBatch.dtype)).device(torch::kCPU);

	torch::Tensor cpuBatch = torch::from_blob(const_cast<void*>(leafBatch.data),  // from_blob needs a non-const ptr
	                                          leafBatch.shape, options);

	// 2. Run the original logic: move to device, execute, move back to CPU.
	torch::Tensor deviceBatch = cpuBatch.to(device_, /*non_blocking=*/true);
	torch::IValue output = encodeMethod_({deviceBatch});
	torch::Tensor outputCpuTensor = output.toTensor().to(torch::kCPU, torch::kUInt8).contiguous();

	// 3. Create an owning `Tensor` for the return value.
	Tensor result;
	result.shape = outputCpuTensor.sizes().vec();
	result.dtype = DataType::UINT8;

	// 4. Copy the data from the torch tensor into the result's owning buffer.
	//    This is the primary source of overhead.
	const size_t numBytes = outputCpuTensor.nbytes();
	result.buffer.resize(numBytes);
	std::memcpy(result.buffer.data(), outputCpuTensor.data_ptr(), numBytes);

	return result;
}

Tensor TorchBackend::decode(const TensorView& indices) const {
	if (indices.dtype != DataType::UINT8) {
		throw std::runtime_error("decode expects UINT8 data.");
	}

	torch::NoGradGuard g;

	// 1. Create a non-owning torch::Tensor from the input TensorView.
	auto options = torch::TensorOptions().dtype(toTorchDType(indices.dtype)).device(torch::kCPU);

	torch::Tensor cpuIndices = torch::from_blob(const_cast<void*>(indices.data), indices.shape, options);

	// 2. Run the original logic: move to device, cast to Long for embedding, execute.
	torch::Tensor deviceBatch = cpuIndices.to(device_, torch::kLong, /*non_blocking=*/true);
	torch::IValue output = decodeMethod_({deviceBatch});
	torch::Tensor outputCpuTensor = output.toTensor().to(torch::kCPU, torch::kFloat32).contiguous();

	// 3. Create an owning `Tensor` for the return value.
	Tensor result;
	result.shape = outputCpuTensor.sizes().vec();
	result.dtype = DataType::FLOAT32;

	// 4. Copy the data from the torch tensor into the result's owning buffer.
	const size_t numBytes = outputCpuTensor.nbytes();
	result.buffer.resize(numBytes);
	std::memcpy(result.buffer.data(), outputCpuTensor.data_ptr(), numBytes);

	return result;
}