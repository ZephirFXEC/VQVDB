#include <cuda_runtime.h>
__global__ void lookupCodebookKernel(const float* __restrict__ codebook,    // [num_embeddings, embedding_dim]
									 const uint16_t* __restrict__ indices,  // [batch_size, depth, height, width]
									 float* __restrict__ output,            // [batch_size, embedding_dim, depth, height, width]
									 int batch_size, int depth, int height, int width, int embedding_dim, int num_embeddings) {
	// Calculate global thread indices
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;
	int b = blockIdx.w;  // Batch index

	if (x >= width || y >= height || z >= depth || b >= batch_size) return;

	// Get index from indices tensor
	int flat_pos = b * (depth * height * width) + z * (height * width) + y * width + x;
	uint16_t index = indices[flat_pos];

	// Bounds check
	if (index >= num_embeddings) return;

	// Copy embedding vector to output
	for (int d = 0; d < embedding_dim; d++) {
		float val = codebook[index * embedding_dim + d];
		// Output in BCHWD format
		output[b * (embedding_dim * depth * height * width) + d * (depth * height * width) + z * (height * width) + y * width + x] = val;
	}
}

