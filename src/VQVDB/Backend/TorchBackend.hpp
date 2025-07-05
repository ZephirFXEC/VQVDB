//
// Created by zphrfx on 05/07/2025.
//

#pragma once

#include <torch/torch.h>

#include "IVQVAECodec.hpp"

class TorchBackend : public IVQVAECodec {
   public:
	// Accept either an embedded buffer (as you do today) or a .pt path
	explicit TorchBackend();
	~TorchBackend() override = default;

	torch::Tensor encode(const torch::Tensor& cpuBatch) override;
	torch::Tensor decode(const torch::Tensor& cpuBatch) override;
	const std::vector<int64_t>& getLatentShape() const override { return latentShape_; }

	static std::tuple<torch::jit::Module, torch::jit::Method, torch::jit::Method> load_embedded_model(const torch::Device& device);

   private:
	torch::Device device_;
	std::tuple<torch::jit::Module, torch::jit::Method, torch::jit::Method> model_parts_;
	torch::jit::Module module_;
	torch::jit::Method encodeMethod_;
	torch::jit::Method decodeMethod_;
	std::vector<int64_t> latentShape_;

};