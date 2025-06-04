#pragma once

#include <cuda_runtime.h>


#ifdef __cplusplus
extern "C" {
#endif

// Declare the kernel launcher function (callable from C++)
void lookupCodebook_launch(const float* codebook,    // [num_embeddings, embedding_dim]
                           const uint16_t* indices,  // [batch_size, depth, height, width]
                           float* output,            // [batch_size, embedding_dim, depth, height, width]
                           int batch_size, int depth, int height, int width, int embedding_dim, int num_embeddings);

#ifdef __cplusplus
}
#endif

// Kernel declaration (only used within CUDA files)
__global__ void lookupCodebookKernel(const float* __restrict__ codebook, const uint16_t* __restrict__ indices, float* __restrict__ output,
                                     int batch_size, int depth, int height, int width, int embedding_dim, int num_embeddings);
