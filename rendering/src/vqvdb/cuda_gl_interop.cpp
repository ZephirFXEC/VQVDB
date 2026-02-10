/*
 * Copyright (c) 2025, Enzo Crema
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "vqvdb/cuda_gl_interop.hpp"

#if defined(VQVDB_RENDERING_ENABLE_TRT)
#include <glad/glad.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include "vqvdb/cuda_utils.hpp"

namespace vqvdb {

namespace {

cudaGraphicsResource_t toCudaResource(void* ptr) { return reinterpret_cast<cudaGraphicsResource_t>(ptr); }

using cuda_utils::cudaOk;

}  // namespace

DecoderResult<void> CudaGLInterop::registerTexture(uint32_t texture) {
	if (texture == 0) {
		return std::unexpected(DecoderError::InvalidInput);
	}

	if (registeredTexture == texture && resource != nullptr) {
		return {};
	}

	unregisterTexture();

	cudaGraphicsResource_t res = nullptr;
	const auto err = cudaGraphicsGLRegisterImage(&res, texture, GL_TEXTURE_3D, cudaGraphicsRegisterFlagsWriteDiscard);
	if (!cudaOk(err)) {
		return std::unexpected(DecoderError::InteropError);
	}

	resource = reinterpret_cast<void*>(res);
	registeredTexture = texture;
	return {};
}

void CudaGLInterop::unregisterTexture() noexcept {
	if (mapped) {
		unmapFromCuda();
	}
	if (resource != nullptr) {
		cudaGraphicsUnregisterResource(toCudaResource(resource));
	}
	resource = nullptr;
	registeredTexture = 0;
}

DecoderResult<void> CudaGLInterop::mapForCuda() {
	if (resource == nullptr) {
		return std::unexpected(DecoderError::InvalidInput);
	}
	if (mapped) {
		return {};
	}

	cudaGraphicsResource_t res = toCudaResource(resource);
	const auto err = cudaGraphicsMapResources(1, &res, 0);
	if (!cudaOk(err)) {
		return std::unexpected(DecoderError::InteropError);
	}

	mapped = true;
	return {};
}

void CudaGLInterop::unmapFromCuda() noexcept {
	if (resource != nullptr && mapped) {
		cudaGraphicsResource_t res = toCudaResource(resource);
		cudaGraphicsUnmapResources(1, &res, 0);
	}
	mapped = false;
}

DecoderResult<void> CudaGLInterop::copyBrickToAtlas(const float* d_brickData, glm::ivec3 atlasOffset, glm::ivec3 atlasDims) {
	(void)atlasDims;

	if (!mapped || resource == nullptr || d_brickData == nullptr) {
		return std::unexpected(DecoderError::InvalidInput);
	}

	cudaArray_t array = nullptr;
	const auto getErr = cudaGraphicsSubResourceGetMappedArray(&array, toCudaResource(resource), 0, 0);
	if (!cudaOk(getErr) || array == nullptr) {
		return std::unexpected(DecoderError::InteropError);
	}

	cudaMemcpy3DParms params{};
	params.srcPtr = make_cudaPitchedPtr(const_cast<float*>(d_brickData), 8 * sizeof(float), 8, 8);
	params.dstArray = array;
	params.dstPos = make_cudaPos(atlasOffset.x, atlasOffset.y, atlasOffset.z);
	params.extent = make_cudaExtent(8 * sizeof(float), 8, 8);
	params.kind = cudaMemcpyDeviceToDevice;

	const auto copyErr = cudaMemcpy3D(&params);
	if (!cudaOk(copyErr)) {
		return std::unexpected(DecoderError::InteropError);
	}

	return {};
}

}  // namespace vqvdb

#else

namespace vqvdb {

DecoderResult<void> CudaGLInterop::registerTexture(uint32_t texture) {
	(void)texture;
	return std::unexpected(DecoderError::InteropError);
}

void CudaGLInterop::unregisterTexture() noexcept {}

DecoderResult<void> CudaGLInterop::mapForCuda() { return std::unexpected(DecoderError::InteropError); }

void CudaGLInterop::unmapFromCuda() noexcept {}

DecoderResult<void> CudaGLInterop::copyBrickToAtlas(const float* d_brickData, glm::ivec3 atlasOffset, glm::ivec3 atlasDims) {
	(void)d_brickData;
	(void)atlasOffset;
	(void)atlasDims;
	return std::unexpected(DecoderError::InteropError);
}

}  // namespace vqvdb

#endif
