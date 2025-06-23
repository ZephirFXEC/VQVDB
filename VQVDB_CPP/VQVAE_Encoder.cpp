//
// Created by zphrfx on 20/06/2025.
//

// VQVAE_Encoder.cpp

#include "VQVAE_Encoder.hpp"

#include <torch/cuda.h>

// Helper to write a simple type to a binary stream (assumes little-endian)
template <typename T>
void write_binary(std::ofstream& stream, const T& value) {
	stream.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

torch::Tensor VDBBlockStreamer::nextBatch(const size_t maxBatch) {
	if (!hasNext()) return torch::empty({0});

	const size_t start = currentPos_;
	const size_t end = std::min(start + maxBatch, leafIndices_.size());
	currentPos_ = end;
	const size_t B = end - start;

	// Pinned host memory ⇒ async H2D possible later
	auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU).pinned_memory(true);

	torch::Tensor batch = torch::empty({static_cast<long>(B), 1, LEAF_DIM, LEAF_DIM, LEAF_DIM}, opts);

	constexpr size_t elemPerLeaf = LEAF_DIM * LEAF_DIM * LEAF_DIM;
	float* dstBase = batch.data_ptr<float>();

	tbb::parallel_for(tbb::blocked_range<size_t>(0, B), [&](auto r) {
		for (size_t i = r.begin(); i != r.end(); ++i) {
			const size_t leafIdx = leafIndices_[start + i];
			const auto& leaf = leafManager_.leaf(leafIdx);
			const float* src = leaf.buffer().data();
			float* dst = dstBase + i * elemPerLeaf;
			std::memcpy(dst, src, elemPerLeaf * sizeof(float));
		}
	});

	return batch;  // Already [B,1,D,H,W] – no permute/unsqueeze needed
}


VQVAEEncoder::VQVAEEncoder(const std::string& modelPath)
    : device_(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU),
      model_(torch::jit::load(modelPath)),
      encodeMethod_(model_.get_method("encode"))  // then grab method
{
	std::cout << "Using device: " << device_ << '\n';
	model_.eval();
	model_.to(device_, device_.is_cuda() ? torch::kHalf : torch::kFloat32);
}


/**  Compress one grid into a .vqvdb file  */
void VQVAEEncoder::compress(const openvdb::FloatGrid::Ptr& grid, const std::string& outPath) const {
	const auto t0 = std::chrono::high_resolution_clock::now();

	const openvdb::tree::LeafManager<openvdb::FloatTree> leafMgr(grid->tree());

	std::vector<openvdb::Coord> origins;
	origins.reserve(leafMgr.leafCount());
	std::vector<size_t> leafIdx;
	leafIdx.reserve(leafMgr.leafCount());

	for (size_t i = 0; i < leafMgr.leafCount(); ++i) {
		origins.push_back(leafMgr.leaf(i).origin());
		leafIdx.push_back(i);
	}

	const int64_t N = origins.size();
	if (N == 0) {
		std::cout << "No active voxels - nothing to do.\n";
		return;
	}

	// ---------- open output + header ----------
	std::ofstream out(outPath, std::ios::binary);
	if (!out) throw std::runtime_error("Cannot open output file " + outPath);

	out.write("VQVDB", 5);          // magic
	write_binary<uint8_t>(out, 2);  // version
	{                               // sample dims
		torch::Tensor s = torch::randn({1, 1, LEAF_DIM, LEAF_DIM, LEAF_DIM});
		auto idx = encodeBatch(s).sizes();
		write_binary<uint32_t>(out, 0);
		write_binary<uint8_t>(out, static_cast<uint8_t>(idx.size() - 1));
		for (size_t i = 1; i < idx.size(); ++i) write_binary<uint16_t>(out, static_cast<uint16_t>(idx[i]));
	}
	write_binary<uint32_t>(out, static_cast<uint32_t>(N));  // #origins
	for (const auto& o : origins) {
		write_binary<int32_t>(out, o.x());
		write_binary<int32_t>(out, o.y());
		write_binary<int32_t>(out, o.z());
	}
	write_binary<uint32_t>(out, static_cast<uint32_t>(N));  // #blocks

	// ---------- streaming ----------
	constexpr int64_t BATCH = 8192;
	VDBBlockStreamer stream(leafMgr, leafIdx);

	int64_t done = 0;
	const auto t1 = std::chrono::high_resolution_clock::now();

	while (stream.hasNext()) {
		torch::Tensor host = stream.nextBatch(BATCH);
		if (host.numel() == 0) break;

		torch::Tensor idx8 = encodeBatch(host);  // uint8 on CPU

		out.write(reinterpret_cast<const char*>(idx8.data_ptr<uint8_t>()), idx8.numel());
		done += host.size(0);
		std::cout << "\rProcessed " << done << '/' << N << " blocks..." << std::flush;
	}
	std::cout << '\n';
	out.close();

	// ---------- stats ----------
	const auto t2 = std::chrono::high_resolution_clock::now();
	const auto load = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
	const auto comp = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

	printf("\n-- Load  : %lld ms\n-- Encode : %lld ms\n", load, comp);
}

/** hostBatch: pinned FP32 on CPU.  Returns **uint8 on CPU**. */
torch::Tensor VQVAEEncoder::encodeBatch(const torch::Tensor& hostBatch) const {
	torch::NoGradGuard nograd;

	torch::Tensor gpu = hostBatch.to(device_, torch::kHalf, /*non_blocking=*/true);
	const torch::Tensor idx = encodeMethod_({gpu}).toTensor();  // → int64/32 on device

	return idx.to(torch::kCPU, torch::kU8, /*non_blocking=*/true);
}
