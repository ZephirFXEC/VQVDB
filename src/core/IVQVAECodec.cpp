/*
 * Copyright (c) 2025, Enzo Crema
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See the LICENSE file in the project root for full license text.
 */

#include "IVQVAECodec.hpp"

#include <iostream>
#include <stdexcept>

#ifdef ENABLE_TORCH_BACKEND
#include "backends/torch/TorchBackend.hpp"
#endif

#ifdef ENABLE_ONNX_BACKEND
#include "backends/onnx/OnnxBackend_CPU.hpp"
#include "backends/onnx/OnnxBackend_Cuda.hpp"
#endif

std::unique_ptr<IVQVAECodec> IVQVAECodec::create(const CodecConfig& config, BackendType type) {
	try {
		switch (type) {
#ifdef ENABLE_TORCH_BACKEND
			case BackendType::LibTorch:
				return std::unique_ptr<IVQVAECodec>(new TorchBackend(config));
#endif

#ifdef ENABLE_ONNX_BACKEND
			case BackendType::ONNX: {
				switch (config.device) {
					case CodecConfig::Device::CPU:
						return std::make_unique<OnnxCpuBackend>(config);
					case CodecConfig::Device::CUDA:
						return std::make_unique<OnnxCudaBackend>(config);

					default:
						throw std::runtime_error("Unsupported device for ONNX backend: " + std::to_string(static_cast<int>(config.device)));
				}
			}
#endif
			default:
				throw std::runtime_error(
				    "Requested backend type is not available or disabled in the "
				    "build configuration.");
		}
	} catch (const std::exception& e) {
		std::cerr << "Failed to create VQ-VAE backend: " << e.what() << std::endl;
		return nullptr;  // Return nullptr on any creation failure
	}
}