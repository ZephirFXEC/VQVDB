#pragma once

#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

#include <filesystem>
#include <memory>
#include <vector>

#include "core/IVQVAECodec.hpp"

class OnnxBackendFactory : public IVQVAECodec {
   public:
	~OnnxBackendFactory() override = default;

	// Deleted copy and move semantics
	OnnxBackendFactory(const OnnxBackendFactory&) = delete;
	OnnxBackendFactory& operator=(const OnnxBackendFactory&) = delete;
	OnnxBackendFactory(OnnxBackendFactory&&) = delete;
	OnnxBackendFactory& operator=(OnnxBackendFactory&&) = delete;

	const std::vector<int64_t>& getLatentShape() const override { return latentShape_; }

   protected:
	explicit OnnxBackendFactory();

	// Common setup methods
	void setup_sessions(const ModelSource& source);
	static std::vector<uint8_t> load_model_data(const std::filesystem::path& path);

	static size_t getDataTypeSize(DataType dtype);
	static size_t calculateTotalElements(const std::vector<int64_t>& shape);
	static ONNXTensorElementDataType toOnnxDataType(DataType dtype);

	// Provider-specific implementations must override these
	virtual void configure_execution_provider() = 0;
	virtual void initialize_latent_shape_impl() = 0;
	virtual Tensor encode_impl(const TensorView& leafBatch) const = 0;
	virtual Tensor decode_impl(const TensorView& indices) const = 0;

	// Common members
	Ort::Env env_;
	Ort::SessionOptions sessionOptions_;
	std::unique_ptr<Ort::Session> encoderSession_;
	std::unique_ptr<Ort::Session> decoderSession_;
	Ort::AllocatorWithDefaultOptions allocator_;

	// Input/output names
	std::vector<const char*> encoderInputNames_;
	std::vector<const char*> encoderOutputNames_;
	std::vector<const char*> decoderInputNames_;
	std::vector<const char*> decoderOutputNames_;

	// Pointer holders
	std::vector<Ort::AllocatedStringPtr> encoderInputNamePtrs_;
	std::vector<Ort::AllocatedStringPtr> encoderOutputNamePtrs_;
	std::vector<Ort::AllocatedStringPtr> decoderInputNamePtrs_;
	std::vector<Ort::AllocatedStringPtr> decoderOutputNamePtrs_;

	// Model metadata
	std::vector<int64_t> latentShape_;

	void init(const CodecConfig& config);

   private:
	Tensor encode(const TensorView& leafBatch) const override { return encode_impl(leafBatch); }
	Tensor decode(const TensorView& indices) const override { return decode_impl(indices); }
};