#include "Kernel.cuh"

__global__ void lookupCodebookKernel(const float* __restrict__ codebook,    // [num_embeddings, embedding_dim]
                                     const uint16_t* __restrict__ indices,  // [batch_size, depth, height, width]
                                     float* __restrict__ output,            // [batch_size, embedding_dim, depth, height, width]
                                     int batch_size, int depth, int height, int width, int embedding_dim, int num_embeddings) {
	// In lookupCodebookKernel
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	// blockIdx.z now maps directly to the batch item
	unsigned int b = blockIdx.z;

	// Loop over depth inside the kernel for this (x, y) column
	for (int z = 0; z < depth; ++z) {
		if (x >= width || y >= height || b >= batch_size) continue;

		const int flat_pos = b * (depth * height * width) + z * (height * width) + y * width + x;
		const uint16_t index = indices[flat_pos];

		if (index >= num_embeddings) continue;  // Safety check

		// This loop is a performance bottleneck (see optimization below)
		for (int d = 0; d < embedding_dim; d++) {
			const float val = codebook[index * embedding_dim + d];
			output[b * (embedding_dim * depth * height * width) + d * (depth * height * width) + z * (height * width) + y * width + x] =
			    val;
		}
	}
}

extern "C" void lookupCodebook_launch(const float* codebook, const uint16_t* indices, float* output, int batch_size, int depth, int height,
                                      int width, int embedding_dim, int num_embeddings) {
	// In lookupCodebook_launch
	dim3 block(16, 16, 1);                                                                     // 2D block
	dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y, batch_size);  // Grid is (WxH) x Batch
	lookupCodebookKernel<<<grid, block>>>(codebook, indices, output, batch_size, depth, height, width, embedding_dim, num_embeddings);
}
