#include "Kernel.cuh"

__global__ void lookupCodebookKernel(const float* __restrict__ codebook,    // [num_embeddings, embedding_dim]
                                     const uint16_t* __restrict__ indices,  // [batch_size, depth, height, width]
                                     float* __restrict__ output,            // [batch_size, embedding_dim, depth, height, width]
                                     int batch_size, int depth, int height, int width, int embedding_dim, int num_embeddings) {
	// Calculate global thread indices
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;

	// Calculate batch index from blockIdx.z
	int b = blockIdx.z / ((depth + blockDim.z - 1) / blockDim.z);
	// Adjust z to be local to the current batch
	z = z % depth;

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

extern "C" void lookupCodebook_launch(const float* codebook, const uint16_t* indices, float* output,
                                       int batch_size, int depth, int height, int width, int embedding_dim,
                                       int num_embeddings) {
        dim3 block(8, 8, 4);
        dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y,
                  batch_size * ((depth + block.z - 1) / block.z));

        lookupCodebookKernel<<<grid, block>>>(codebook, indices, output, batch_size, depth, height, width,
                                              embedding_dim, num_embeddings);
}
