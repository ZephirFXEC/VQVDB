#pragma once

#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

#include <memory>
#include <vector>

#include "core/IVQVAECodec.hpp"

class OnnxBackend final : public IVQVAECodec {
   public:
	// The factory function in IVQVAECodec is a friend to access the private constructor.
	friend std::unique_ptr<IVQVAECodec> IVQVAECodec::create(const CodecConfig& config, BackendType type);

	~OnnxBackend() override = default;

	// Deleted copy and move semantics to prevent slicing and ensure unique ownership.
	OnnxBackend(const OnnxBackend&) = delete;
	OnnxBackend& operator=(const OnnxBackend&) = delete;
	OnnxBackend(OnnxBackend&&) = delete;
	OnnxBackend& operator=(OnnxBackend&&) = delete;

	// IVQVAECodec overrides
	Tensor encode(const TensorView& leafBatch) const override;
	Tensor decode(const TensorView& indices) const override;
	const std::vector<int64_t>& getLatentShape() const override { return latentShape_; }

   private:
	explicit OnnxBackend(const CodecConfig& config);

	void initialize_latent_shape();
	static std::vector<uint8_t> load_model_data(const std::filesystem::path& path);
	void setup_sessions(const ModelSource& source);

	Ort::Env env_;
	Ort::SessionOptions sessionOptions_;
	std::unique_ptr<Ort::Session> encoderSession_;
	std::unique_ptr<Ort::Session> decoderSession_;
	Ort::AllocatorWithDefaultOptions allocator_;

	// Input/output names for the ONNX models
	std::vector<const char*> encoderInputNames_;
	std::vector<const char*> encoderOutputNames_;
	std::vector<const char*> decoderInputNames_;
	std::vector<const char*> decoderOutputNames_;

	// Pointer holders to keep allocations alive
	std::vector<Ort::AllocatedStringPtr> encoderInputNamePtrs_;
	std::vector<Ort::AllocatedStringPtr> encoderOutputNamePtrs_;
	std::vector<Ort::AllocatedStringPtr> decoderInputNamePtrs_;
	std::vector<Ort::AllocatedStringPtr> decoderOutputNamePtrs_;

	// Model metadata
	std::vector<int64_t> latentShape_;
	bool useGpu_;
};