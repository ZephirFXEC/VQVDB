//
// Created by zphrfx on 18/07/2025.
//


#include "IVQVAECodec.hpp"

#include <iostream>
#include <stdexcept>

#ifdef ENABLE_TORCH_BACKEND
#include "backends/torch/TorchBackend.hpp"
#endif

#ifdef ENABLE_ONNX_BACKEND
#include "backends/onnx/OnnxBackend.hpp"
#endif


// The factory implementation with an explicit choice
std::unique_ptr<IVQVAECodec> IVQVAECodec::create(const CodecConfig& config, BackendType type) {
	try {
		switch (type) {
#ifdef ENABLE_TORCH_BACKEND
			case BackendType::LibTorch:
				return std::unique_ptr<IVQVAECodec>(new TorchBackend(config));
#endif

#ifdef ENABLE_ONNX_BACKEND
			case BackendType::ONNX:
				return std::unique_ptr<IVQVAECodec>(new OnnxBackend(config));
#endif

			default:
				throw std::runtime_error("Requested backend type is not available or disabled in the build configuration.");
		}
	} catch (const std::exception& e) {
		std::cerr << "Failed to create VQ-VAE backend: " << e.what() << std::endl;
		return nullptr;  // Return nullptr on any creation failure
	}
}