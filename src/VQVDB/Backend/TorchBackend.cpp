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
#include "../Bin/bin_model.h"

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

		// *** THE FIX: Use std::stringstream ***
		// This creates an in-memory stream that is fully seekable, which torch::jit::load requires.
		// It involves one copy of the model data, which is acceptable for a one-time load.
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
	// Note: This is a GLOBAL setting in libtorch. Calling this can affect
	// other libtorch-using parts of the application. It's best to configure
	// this once at application startup.
	const int num_threads = std::max(1, (int)std::thread::hardware_concurrency() / 2);
	torch::set_num_threads(num_threads);
	std::cout << "TorchBackend: Set CPU threads to " << torch::get_num_threads() << std::endl;
}

}  // namespace

// --- Factory Function Implementation ---
std::unique_ptr<IVQVAECodec> IVQVAECodec::create(const CodecConfig& config) {
	try {
		return std::unique_ptr<IVQVAECodec>(new TorchBackend(config));
	} catch (const std::exception& e) {
		std::cerr << "Failed to create TorchBackend: " << e.what() << std::endl;
		return nullptr;
	}
}

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


torch::Tensor TorchBackend::encode(const torch::Tensor& cpuBatch) const {
	torch::NoGradGuard g;
	// Move input tensor to the model's device, non-blocking for GPU transfers
	torch::Tensor deviceBatch = cpuBatch.to(device_, /*non_blocking=*/true);
	torch::IValue output = encodeMethod_({deviceBatch});
	// Move result back to CPU and cast to uint8 as per the interface contract
	return output.toTensor().to(torch::kCPU, torch::kUInt8);
}

torch::Tensor TorchBackend::decode(const torch::Tensor& cpuIndices) const {
	torch::NoGradGuard g;
	// Indices must be long for embedding lookup; move to device
	torch::Tensor deviceBatch = cpuIndices.to(device_, torch::kLong, /*non_blocking=*/true);
	torch::IValue output = decodeMethod_({deviceBatch});
	// Move result back to CPU and cast to float32 as per the interface contract
	return output.toTensor().to(torch::kCPU, torch::kFloat32);
}