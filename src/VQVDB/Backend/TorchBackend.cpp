//
// Created by zphrfx on 05/07/2025.
//

#include "TorchBackend.hpp"

#include <torch/script.h>

#include <sstream>

#include "../Bin/bin_model.h"

using torch::kCPU;
using torch::kFloat32;
using torch::kLong;
using torch::kU8;

std::tuple<torch::jit::Module, torch::jit::Method, torch::jit::Method> TorchBackend::load_embedded_model(const torch::Device& device) {
	// Create a string stream from the embedded byte array
	const std::string model_string(reinterpret_cast<const char*>(g_model_data), g_model_data_size);
	std::istringstream stream(model_string);

	torch::jit::Module module;
	try {
		// Load the model from the stream (onto CPU by default)
		module = torch::jit::load(stream);
	} catch (const c10::Error& e) {
		throw std::runtime_error("Failed to load TorchScript model from memory: " + std::string(e.what()));
	}

	module.to(device);
	module.eval();

	// Get the methods from the now-configured module
	torch::jit::Method encode_method = module.get_method("encode");
	torch::jit::Method decode_method = module.get_method("decode");

	std::cout << "TorchBackend: Model successfully loaded onto device: " << device << std::endl;

	// Return all the constructed objects in a tuple
	return {std::move(module), std::move(encode_method), std::move(decode_method)};
}

TorchBackend::TorchBackend()
    : device_(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU),
      model_parts_(load_embedded_model(device_)),
      module_(std::get<0>(model_parts_)),
      encodeMethod_(std::get<1>(model_parts_)),
      decodeMethod_(std::get<2>(model_parts_)) {
	if (device_.is_cpu()) {
		const int num_threads = std::max(1, (int)std::thread::hardware_concurrency() / 2);
		torch::set_num_threads(num_threads);
		std::cout << "TorchBackend: Set CPU threads to " << torch::get_num_threads() << std::endl;
	}

	{
		torch::NoGradGuard nograd;
		auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(device_);
		torch::Tensor dummyInput = torch::zeros({1, 1, 8, 8, 8}, opts);
		auto idx = encodeMethod_({dummyInput}).toTensor();

		auto sizes = idx.sizes();
		latentShape_.assign(sizes.begin() + 1, sizes.end());
	}
}


torch::Tensor TorchBackend::encode(const torch::Tensor& cpuBatch) {
	torch::NoGradGuard g;
	// Move batch to the target device (GPU or CPU)
	torch::Tensor deviceBatch = cpuBatch.to(device_, /*non_blocking=*/true);
	torch::Tensor result = encodeMethod_({deviceBatch}).toTensor();
	// Return result on CPU as uint8
	return result.to(torch::kCPU, torch::kU8);
}

torch::Tensor TorchBackend::decode(const torch::Tensor& cpuBatch) {
	torch::NoGradGuard g;
	// Move indices to target device, casting to long
	torch::Tensor deviceBatch = cpuBatch.to(device_, torch::kLong, /*non_blocking=*/true);
	torch::Tensor result = decodeMethod_({deviceBatch}).toTensor();
	// Return result on CPU as float32
	return result.to(torch::kCPU, torch::kFloat32);
}