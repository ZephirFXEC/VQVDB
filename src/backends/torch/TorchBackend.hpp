/*
 * Copyright (c) 2025, Enzo Crema
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See the LICENSE file in the project root for full license text.
 */

#pragma once

#include <torch/torch.h>

#include "core/IVQVAECodec.hpp"


/**
 * @brief A libtorch-based implementation of the IVQVAECodec interface.
 *
 * NOTE: This class is not meant to be instantiated directly.
 * Use the `IVQVAECodec::create` factory function.
 */
class TorchBackend final : public IVQVAECodec {
   public:
	// The factory function in IVQVAECodec is a friend to access the private constructor.
	friend std::unique_ptr<IVQVAECodec> IVQVAECodec::create(const CodecConfig& config, BackendType type);

	~TorchBackend() override = default;

	// Deleted copy and move semantics to prevent slicing and ensure unique ownership.
	TorchBackend(const TorchBackend&) = delete;
	TorchBackend& operator=(const TorchBackend&) = delete;
	TorchBackend(TorchBackend&&) = delete;
	TorchBackend& operator=(TorchBackend&&) = delete;

	// IVQVAECodec overrides
	Tensor encode(const TensorView& leafBatch) const override;
	Tensor decode(const TensorView& leafBatch) const override;
	const std::vector<int64_t>& getLatentShape() const override { return latentShape_; }


   private:
	explicit TorchBackend(const CodecConfig& config);

	void initialize_latent_shape();

	const torch::Device device_;
	torch::jit::Module module_;
	const torch::jit::Method encodeMethod_;
	const torch::jit::Method decodeMethod_;
	std::vector<int64_t> latentShape_;
};