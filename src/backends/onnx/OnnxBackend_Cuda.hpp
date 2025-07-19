#pragma once

#include "OnnxBackendFactory.hpp"

class OnnxCudaBackend final : public OnnxBackendFactory {
   public:
	friend std::unique_ptr<IVQVAECodec> IVQVAECodec::create(const CodecConfig& config, BackendType type);
	explicit OnnxCudaBackend(const CodecConfig& config);

   protected:
	void configure_execution_provider() override;
	void initialize_latent_shape_impl() override;
	Tensor encode_impl(const TensorView& leafBatch) const override;
	Tensor decode_impl(const TensorView& indices) const override;
};