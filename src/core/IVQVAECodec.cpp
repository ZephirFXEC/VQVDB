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
#include <windows.h>
#include "backends/onnx/OnnxBackend_CPU.hpp"
#include "backends/onnx/OnnxBackend_Cuda.hpp"

void ManualInitOnnxRuntime(const std::string& dllPath) {
	static HMODULE dllHandle = nullptr;
	dllHandle = LoadLibrary(dllPath.c_str());
	if (!dllHandle) {
		DWORD error = GetLastError();
		std::cerr << "Failed to load ONNX Runtime DLL from " << dllPath << ". Error: " << error << std::endl;
		throw std::runtime_error("Failed to load ONNX Runtime DLL");
	}

	using PFN_OrtGetApiBase = const OrtApiBase*(ORT_API_CALL*)();
	auto getApiBase = reinterpret_cast<PFN_OrtGetApiBase>(GetProcAddress(dllHandle, "OrtGetApiBase"));
	if (!getApiBase) {
		FreeLibrary(dllHandle);
		dllHandle = nullptr;
		throw std::runtime_error("Failed to get OrtGetApiBase from DLL");
	}

	const OrtApiBase* apiBase = getApiBase();
	if (!apiBase) {
		FreeLibrary(dllHandle);
		dllHandle = nullptr;
		throw std::runtime_error("OrtGetApiBase returned null");
	}

	const OrtApi* ortApi = apiBase->GetApi(ORT_API_VERSION);  // ORT_API_VERSION is defined in the header
	if (!ortApi) {
		FreeLibrary(dllHandle);
		dllHandle = nullptr;
		throw std::runtime_error("Failed to get OrtApi (possible version mismatch?)");
	}

	Ort::InitApi(ortApi);
}
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
				std::filesystem::path dllPath =
				    std::filesystem::path("C:/Program Files/Side Effects Software/Houdini 20.5.613/bin/onnxruntime.dll");

				ManualInitOnnxRuntime(dllPath.string());

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