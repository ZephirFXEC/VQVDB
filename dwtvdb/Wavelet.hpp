#pragma once

#include <Eigen/Dense>
#include <cassert>
#include <cmath>
#include <string>
#include <tuple>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>

// Slice metadata for flattened array
struct CoeffSlice {
	std::string name;            // e.g., "level0_aad", "coarse"
	int offset;                  // start index in flattened vector
	Eigen::array<int, 3> shape;  // (dx, dy, dz)
};

// Storage of multilevel 3D Haar decomposition.
// details[i] corresponds to level i (level 0 = first decomposition), each array holds 7 detail tensors in the order:
// [aad, ada, add, daa, dad, dda, ddd]
struct WaveletCoeffs3D {
	Eigen::Tensor<float, 3> coarse;  // final coarse approximation after all levels
	std::vector<std::array<Eigen::Tensor<float, 3>, 7>> details;
};


namespace detail {
const float inv_sqrt2 = 1.0f / std::sqrt(2.0f);

// 1D Haar along X: input shape (Nx, Ny, Nz), outputs low/high shape (Nx/2, Ny, Nz)
inline void dwt_axis_x(const Eigen::Tensor<float, 3>& in, Eigen::Tensor<float, 3>& low, Eigen::Tensor<float, 3>& high) {
	const int Nx = in.dimension(0);
	const int Ny = in.dimension(1);
	const int Nz = in.dimension(2);
	assert(low.dimension(0) * 2 == Nx);
	assert(high.dimension(0) * 2 == Nx);
	for (int ix = 0; ix < low.dimension(0); ++ix) {
		int i0 = (2 * ix) % Nx;
		int i1 = (2 * ix + 1) % Nx;
		for (int y = 0; y < Ny; ++y) {
			for (int z = 0; z < Nz; ++z) {
				float x0 = in(i0, y, z);
				float x1 = in(i1, y, z);
				low(ix, y, z) = (x0 + x1) * inv_sqrt2;
				high(ix, y, z) = (x0 - x1) * inv_sqrt2;
			}
		}
	}
}

// inverse along X: input low/high shape (Nx/2, Ny, Nz), output shape (Nx, Ny, Nz)
inline void idwt_axis_x(const Eigen::Tensor<float, 3>& low, const Eigen::Tensor<float, 3>& high, Eigen::Tensor<float, 3>& out) {
	const int Nx2 = low.dimension(0);
	const int Ny = low.dimension(1);
	const int Nz = low.dimension(2);
	const int Nx = Nx2 * 2;
	assert(out.dimension(0) == Nx);
	for (int ix = 0; ix < Nx2; ++ix) {
		for (int y = 0; y < Ny; ++y) {
			for (int z = 0; z < Nz; ++z) {
				float a = low(ix, y, z);
				float d = high(ix, y, z);
				out(2 * ix, y, z) = (a + d) * inv_sqrt2;
				out(2 * ix + 1, y, z) = (a - d) * inv_sqrt2;
			}
		}
	}
}

// 1D Haar along Y: input shape (Nx, Ny, Nz), outputs low/high shape (Nx, Ny/2, Nz)
inline void dwt_axis_y(const Eigen::Tensor<float, 3>& in, Eigen::Tensor<float, 3>& low, Eigen::Tensor<float, 3>& high) {
	const int Nx = in.dimension(0);
	const int Ny = in.dimension(1);
	const int Nz = in.dimension(2);
	assert(low.dimension(1) * 2 == Ny);
	assert(high.dimension(1) * 2 == Ny);
	for (int x = 0; x < Nx; ++x) {
		for (int iy = 0; iy < low.dimension(1); ++iy) {
			int y0 = (2 * iy) % Ny;
			int y1 = (2 * iy + 1) % Ny;
			for (int z = 0; z < Nz; ++z) {
				float x0 = in(x, y0, z);
				float x1 = in(x, y1, z);
				low(x, iy, z) = (x0 + x1) * inv_sqrt2;
				high(x, iy, z) = (x0 - x1) * inv_sqrt2;
			}
		}
	}
}

inline void idwt_axis_y(const Eigen::Tensor<float, 3>& low, const Eigen::Tensor<float, 3>& high, Eigen::Tensor<float, 3>& out) {
	const int Nx = low.dimension(0);
	const int Ny2 = low.dimension(1);
	const int Nz = low.dimension(2);
	const int Ny = Ny2 * 2;
	assert(out.dimension(1) == Ny);
	for (int x = 0; x < Nx; ++x) {
		for (int iy = 0; iy < Ny2; ++iy) {
			for (int z = 0; z < Nz; ++z) {
				float a = low(x, iy, z);
				float d = high(x, iy, z);
				out(x, 2 * iy, z) = (a + d) * inv_sqrt2;
				out(x, 2 * iy + 1, z) = (a - d) * inv_sqrt2;
			}
		}
	}
}

// 1D Haar along Z: input shape (Nx, Ny, Nz), outputs low/high shape (Nx, Ny, Nz/2)
inline void dwt_axis_z(const Eigen::Tensor<float, 3>& in, Eigen::Tensor<float, 3>& low, Eigen::Tensor<float, 3>& high) {
	const int Nx = in.dimension(0);
	const int Ny = in.dimension(1);
	const int Nz = in.dimension(2);
	assert(low.dimension(2) * 2 == Nz);
	assert(high.dimension(2) * 2 == Nz);
	for (int x = 0; x < Nx; ++x) {
		for (int y = 0; y < Ny; ++y) {
			for (int iz = 0; iz < low.dimension(2); ++iz) {
				int z0 = (2 * iz) % Nz;
				int z1 = (2 * iz + 1) % Nz;
				float x0 = in(x, y, z0);
				float x1 = in(x, y, z1);
				low(x, y, iz) = (x0 + x1) * inv_sqrt2;
				high(x, y, iz) = (x0 - x1) * inv_sqrt2;
			}
		}
	}
}

inline void idwt_axis_z(const Eigen::Tensor<float, 3>& low, const Eigen::Tensor<float, 3>& high, Eigen::Tensor<float, 3>& out) {
	const int Nx = low.dimension(0);
	const int Ny = low.dimension(1);
	const int Nz2 = low.dimension(2);
	const int Nz = Nz2 * 2;
	assert(out.dimension(2) == Nz);
	for (int x = 0; x < Nx; ++x) {
		for (int y = 0; y < Ny; ++y) {
			for (int iz = 0; iz < Nz2; ++iz) {
				float a = low(x, y, iz);
				float d = high(x, y, iz);
				out(x, y, 2 * iz) = (a + d) * inv_sqrt2;
				out(x, y, 2 * iz + 1) = (a - d) * inv_sqrt2;
			}
		}
	}
}

// One level of 3D forward Haar decomposition: input shape (Nx, Ny, Nz)
// Outputs: approx (AAA) of shape (Nx/2, Ny/2, Nz/2) and 7 details
inline void decompose_level(const Eigen::Tensor<float, 3>& input, Eigen::Tensor<float, 3>& approx,
                            std::array<Eigen::Tensor<float, 3>, 7>& details) {
	int Nx = input.dimension(0);
	int Ny = input.dimension(1);
	int Nz = input.dimension(2);
	assert((Nx % 2) == 0 && (Ny % 2) == 0 && (Nz % 2) == 0);
	int Nx2 = Nx / 2;
	int Ny2 = Ny / 2;
	int Nz2 = Nz / 2;

	// Step 1: along X
	Eigen::Tensor<float, 3> low_x(Nx2, Ny, Nz);
	Eigen::Tensor<float, 3> high_x(Nx2, Ny, Nz);
	dwt_axis_x(input, low_x, high_x);

	// Step 2: along Y
	Eigen::Tensor<float, 3> low_x_low_y(Nx2, Ny2, Nz);
	Eigen::Tensor<float, 3> low_x_high_y(Nx2, Ny2, Nz);
	Eigen::Tensor<float, 3> high_x_low_y(Nx2, Ny2, Nz);
	Eigen::Tensor<float, 3> high_x_high_y(Nx2, Ny2, Nz);
	dwt_axis_y(low_x, low_x_low_y, low_x_high_y);
	dwt_axis_y(high_x, high_x_low_y, high_x_high_y);

	// Step 3: along Z -> produce eight subbands
	Eigen::Tensor<float, 3> AAA(Nx2, Ny2, Nz2);
	Eigen::Tensor<float, 3> AAD(Nx2, Ny2, Nz2);
	Eigen::Tensor<float, 3> ADA(Nx2, Ny2, Nz2);
	Eigen::Tensor<float, 3> ADD(Nx2, Ny2, Nz2);
	Eigen::Tensor<float, 3> DAA(Nx2, Ny2, Nz2);
	Eigen::Tensor<float, 3> DAD(Nx2, Ny2, Nz2);
	Eigen::Tensor<float, 3> DDA(Nx2, Ny2, Nz2);
	Eigen::Tensor<float, 3> DDD(Nx2, Ny2, Nz2);

	dwt_axis_z(low_x_low_y, AAA, AAD);    // aaa, aad
	dwt_axis_z(low_x_high_y, ADA, ADD);   // ada, add
	dwt_axis_z(high_x_low_y, DAA, DAD);   // daa, dad
	dwt_axis_z(high_x_high_y, DDA, DDD);  // dda, ddd

	approx = AAA;                                   // coarse for next level
	details = {AAD, ADA, ADD, DAA, DAD, DDA, DDD};  // order fixed
}

// One level of 3D inverse Haar: given coarse approximation (AAA) and 7 detail tensors,
// reconstruct higher-res block.
inline void reconstruct_level(const Eigen::Tensor<float, 3>& approx_coarse, const std::array<Eigen::Tensor<float, 3>, 7>& details,
                              Eigen::Tensor<float, 3>& out) {
	int Nx2 = approx_coarse.dimension(0);
	int Ny2 = approx_coarse.dimension(1);
	int Nz2 = approx_coarse.dimension(2);
	int Nx = Nx2 * 2;
	int Ny = Ny2 * 2;
	int Nz = Nz2 * 2;

	// inverse along Z to get four (Nx2, Ny2, Nz)
	Eigen::Tensor<float, 3> low_x_low_y(Nx2, Ny2, Nz);
	Eigen::Tensor<float, 3> low_x_high_y(Nx2, Ny2, Nz);
	Eigen::Tensor<float, 3> high_x_low_y(Nx2, Ny2, Nz);
	Eigen::Tensor<float, 3> high_x_high_y(Nx2, Ny2, Nz);
	const auto& AAD = details[0];
	const auto& ADA = details[1];
	const auto& ADD = details[2];
	const auto& DAA = details[3];
	const auto& DAD = details[4];
	const auto& DDA = details[5];
	const auto& DDD = details[6];

	idwt_axis_z(approx_coarse, AAD, low_x_low_y);  // from aaa & aad
	idwt_axis_z(ADA, ADD, low_x_high_y);           // ada & add
	idwt_axis_z(DAA, DAD, high_x_low_y);           // daa & dad
	idwt_axis_z(DDA, DDD, high_x_high_y);          // dda & ddd

	// inverse along Y to get low_x and high_x of shape (Nx2, Ny, Nz)
	Eigen::Tensor<float, 3> low_x(Nx2, Ny, Nz);
	Eigen::Tensor<float, 3> high_x(Nx2, Ny, Nz);
	idwt_axis_y(low_x_low_y, low_x_high_y, low_x);
	idwt_axis_y(high_x_low_y, high_x_high_y, high_x);

	// inverse along X to get full resolution
	idwt_axis_x(low_x, high_x, out);
}

// Flatten the hierarchical coeffs into a vector and produce slices.
inline void coeffs_to_array(const WaveletCoeffs3D& coeffs, Eigen::VectorXf& arr, std::vector<CoeffSlice>& slices) {
	// Determine total size
	size_t total = 0;
	int levels = static_cast<int>(coeffs.details.size());
	for (int lvl = 0; lvl < levels; ++lvl) {
		const auto& d = coeffs.details[lvl];
		for (int j = 0; j < 7; ++j) {
			auto shape = d[j].dimensions();
			total += shape[0] * shape[1] * shape[2];
		}
	}
	// coarse
	{
		auto shape = coeffs.coarse.dimensions();
		total += shape[0] * shape[1] * shape[2];
	}

	arr.resize(total);
	int offset = 0;

	// details per level, in order level0, level1, ...
	for (int lvl = 0; lvl < levels; ++lvl) {
		const auto& d = coeffs.details[lvl];
		static const std::array<std::string, 7> labels = {"aad", "ada", "add", "daa", "dad", "dda", "ddd"};
		for (int j = 0; j < 7; ++j) {
			auto& tensor = d[j];
			auto dims = tensor.dimensions();
			int block_size = dims[0] * dims[1] * dims[2];
			for (int i = 0, idx = 0; i < dims[0]; ++i)
				for (int y = 0; y < dims[1]; ++y)
					for (int z = 0; z < dims[2]; ++z, ++idx) arr[offset + idx] = tensor(i, y, z);
			CoeffSlice slice;
			slice.name = "level" + std::to_string(lvl) + "_" + labels[j];
			slice.offset = offset;
			slice.shape[0] = dims[0];
			slice.shape[1] = dims[1];
			slice.shape[2] = dims[2];

			slices.push_back(slice);
			offset += block_size;
		}
	}
	// coarse last
	{
		auto dims = coeffs.coarse.dimensions();
		int block_size = dims[0] * dims[1] * dims[2];
		for (int i = 0, idx = 0; i < dims[0]; ++i)
			for (int y = 0; y < dims[1]; ++y)
				for (int z = 0; z < dims[2]; ++z, ++idx) arr[offset + idx] = coeffs.coarse(i, y, z);
		CoeffSlice slice;
		slice.name = "coarse";
		slice.offset = offset;
		slice.shape[0] = dims[0];
		slice.shape[1] = dims[1];
		slice.shape[2] = dims[2];
		slices.push_back(slice);
		offset += block_size;
	}
	assert(offset == static_cast<int>(total));
}

// Reconstruct WaveletCoeffs3D from flat array + slices.
inline WaveletCoeffs3D array_to_coeffs(const Eigen::VectorXf& arr, const std::vector<CoeffSlice>& slices, int level) {
	WaveletCoeffs3D coeffs;
	coeffs.details.resize(level);
	// Expecting that slices are in the same order as produced above: level 0.. level-1 details, then coarse
	int slice_idx = 0;
	for (int lvl = 0; lvl < level; ++lvl) {
		std::array<Eigen::Tensor<float, 3>, 7> dets;
		for (int j = 0; j < 7; ++j, ++slice_idx) {
			const CoeffSlice& slice = slices[slice_idx];
			auto [dx, dy, dz] = std::make_tuple(slice.shape[0], slice.shape[1], slice.shape[2]);
			dets[j] = Eigen::Tensor<float, 3>(dx, dy, dz);
			int block_size = dx * dy * dz;
			int off = slice.offset;
			int idx = 0;
			for (int i = 0; i < dx; ++i)
				for (int y = 0; y < dy; ++y)
					for (int z = 0; z < dz; ++z, ++idx) dets[j](i, y, z) = arr[off + idx];
		}
		coeffs.details[lvl] = std::move(dets);
	}
	// coarse
	const CoeffSlice& coarse_slice = slices[slice_idx++];
	int cx = coarse_slice.shape[0], cy = coarse_slice.shape[1], cz = coarse_slice.shape[2];
	coeffs.coarse = Eigen::Tensor<float, 3>(cx, cy, cz);
	{
		int block_size = cx * cy * cz;
		int off = coarse_slice.offset;
		int idx = 0;
		for (int i = 0; i < cx; ++i)
			for (int y = 0; y < cy; ++y)
				for (int z = 0; z < cz; ++z, ++idx) coeffs.coarse(i, y, z) = arr[off + idx];
	}
	return coeffs;
}

}  // namespace detail

// Public API ---------------------------------------------------------

// Decompose block into flattened array, slices, and hierarchical coeffs. Only "haar" supported currently.
inline std::tuple<Eigen::VectorXf, std::vector<CoeffSlice>, WaveletCoeffs3D> wavedec3(const Eigen::Tensor<float, 3>& block,
                                                                                      const std::string& wavelet, int level) {
	if (wavelet != "haar") {
		throw std::runtime_error("Only 'haar' wavelet is implemented in this version.");
	}
	// check sizes divisible by 2^level
	int Nx = block.dimension(0);
	int Ny = block.dimension(1);
	int Nz = block.dimension(2);
	int factor = 1 << level;
	assert((Nx % factor) == 0 && (Ny % factor) == 0 && (Nz % factor) == 0);

	WaveletCoeffs3D coeffs;
	Eigen::Tensor<float, 3> current = block;
	for (int lvl = 0; lvl < level; ++lvl) {
		Eigen::Tensor<float, 3> next_approx(current.dimension(0) / 2, current.dimension(1) / 2, current.dimension(2) / 2);
		std::array<Eigen::Tensor<float, 3>, 7> detail;
		detail = {};  // default-initialize shapes later in decompose_level

		detail[0] = Eigen::Tensor<float, 3>(current.dimension(0) / 2, current.dimension(1) / 2, current.dimension(2) / 2);
		detail[1] = detail[0];
		detail[2] = detail[0];
		detail[3] = detail[0];
		detail[4] = detail[0];
		detail[5] = detail[0];
		detail[6] = detail[0];

		detail[0].setZero();
		detail[1].setZero();
		detail[2].setZero();
		detail[3].setZero();
		detail[4].setZero();
		detail[5].setZero();
		detail[6].setZero();
		next_approx.setZero();

		detail = {};  // will be populated inside
		detail[0] = Eigen::Tensor<float, 3>(current.dimension(0) / 2, current.dimension(1) / 2, current.dimension(2) / 2);
		detail[1] = detail[0];
		detail[2] = detail[0];
		detail[3] = detail[0];
		detail[4] = detail[0];
		detail[5] = detail[0];
		detail[6] = detail[0];

		detail[0].setZero();
		detail[1].setZero();
		detail[2].setZero();
		detail[3].setZero();
		detail[4].setZero();
		detail[5].setZero();
		detail[6].setZero();
		next_approx.setZero();

		detail = {};  // discard previous garbage

		// perform decomposition level
		detail = std::array<Eigen::Tensor<float, 3>, 7>{
		    Eigen::Tensor<float, 3>(current.dimension(0) / 2, current.dimension(1) / 2, current.dimension(2) / 2),
		    Eigen::Tensor<float, 3>(current.dimension(0) / 2, current.dimension(1) / 2, current.dimension(2) / 2),
		    Eigen::Tensor<float, 3>(current.dimension(0) / 2, current.dimension(1) / 2, current.dimension(2) / 2),
		    Eigen::Tensor<float, 3>(current.dimension(0) / 2, current.dimension(1) / 2, current.dimension(2) / 2),
		    Eigen::Tensor<float, 3>(current.dimension(0) / 2, current.dimension(1) / 2, current.dimension(2) / 2),
		    Eigen::Tensor<float, 3>(current.dimension(0) / 2, current.dimension(1) / 2, current.dimension(2) / 2),
		    Eigen::Tensor<float, 3>(current.dimension(0) / 2, current.dimension(1) / 2, current.dimension(2) / 2)};
		for (int j = 0; j < 7; ++j) detail[j].setZero();
		next_approx.setZero();

		detail = {};  // will be overwritten below
		detail[0] = Eigen::Tensor<float, 3>(current.dimension(0) / 2, current.dimension(1) / 2, current.dimension(2) / 2);
		detail[1] = detail[0];
		detail[2] = detail[0];
		detail[3] = detail[0];
		detail[4] = detail[0];
		detail[5] = detail[0];
		detail[6] = detail[0];
		detail[0].setZero();
		detail[1].setZero();
		detail[2].setZero();
		detail[3].setZero();
		detail[4].setZero();
		detail[5].setZero();
		detail[6].setZero();

		detail = std::array<Eigen::Tensor<float, 3>, 7>{
		    Eigen::Tensor<float, 3>(current.dimension(0) / 2, current.dimension(1) / 2, current.dimension(2) / 2),
		    Eigen::Tensor<float, 3>(current.dimension(0) / 2, current.dimension(1) / 2, current.dimension(2) / 2),
		    Eigen::Tensor<float, 3>(current.dimension(0) / 2, current.dimension(1) / 2, current.dimension(2) / 2),
		    Eigen::Tensor<float, 3>(current.dimension(0) / 2, current.dimension(1) / 2, current.dimension(2) / 2),
		    Eigen::Tensor<float, 3>(current.dimension(0) / 2, current.dimension(1) / 2, current.dimension(2) / 2),
		    Eigen::Tensor<float, 3>(current.dimension(0) / 2, current.dimension(1) / 2, current.dimension(2) / 2),
		    Eigen::Tensor<float, 3>(current.dimension(0) / 2, current.dimension(1) / 2, current.dimension(2) / 2)};
		for (int j = 0; j < 7; ++j) detail[j].setZero();
		next_approx.setZero();

		// actual decomposition
		detail = {};  // reset
		detail[0] = Eigen::Tensor<float, 3>(current.dimension(0) / 2, current.dimension(1) / 2, current.dimension(2) / 2);
		detail[1] = detail[0];
		detail[2] = detail[0];
		detail[3] = detail[0];
		detail[4] = detail[0];
		detail[5] = detail[0];
		detail[6] = detail[0];
		detail[0].setZero();
		detail[1].setZero();
		detail[2].setZero();
		detail[3].setZero();
		detail[4].setZero();
		detail[5].setZero();
		detail[6].setZero();
		next_approx.setZero();

		detail = std::array<Eigen::Tensor<float, 3>, 7>{
		    Eigen::Tensor<float, 3>(current.dimension(0) / 2, current.dimension(1) / 2, current.dimension(2) / 2),
		    Eigen::Tensor<float, 3>(current.dimension(0) / 2, current.dimension(1) / 2, current.dimension(2) / 2),
		    Eigen::Tensor<float, 3>(current.dimension(0) / 2, current.dimension(1) / 2, current.dimension(2) / 2),
		    Eigen::Tensor<float, 3>(current.dimension(0) / 2, current.dimension(1) / 2, current.dimension(2) / 2),
		    Eigen::Tensor<float, 3>(current.dimension(0) / 2, current.dimension(1) / 2, current.dimension(2) / 2),
		    Eigen::Tensor<float, 3>(current.dimension(0) / 2, current.dimension(1) / 2, current.dimension(2) / 2),
		    Eigen::Tensor<float, 3>(current.dimension(0) / 2, current.dimension(1) / 2, current.dimension(2) / 2)};
		detail[0].setZero();
		detail[1].setZero();
		detail[2].setZero();
		detail[3].setZero();
		detail[4].setZero();
		detail[5].setZero();
		detail[6].setZero();
		next_approx.setZero();

		// Finally call the properly-sized decomposition
		detail = std::array<Eigen::Tensor<float, 3>, 7>{
		    Eigen::Tensor<float, 3>(current.dimension(0) / 2, current.dimension(1) / 2, current.dimension(2) / 2),
		    Eigen::Tensor<float, 3>(current.dimension(0) / 2, current.dimension(1) / 2, current.dimension(2) / 2),
		    Eigen::Tensor<float, 3>(current.dimension(0) / 2, current.dimension(1) / 2, current.dimension(2) / 2),
		    Eigen::Tensor<float, 3>(current.dimension(0) / 2, current.dimension(1) / 2, current.dimension(2) / 2),
		    Eigen::Tensor<float, 3>(current.dimension(0) / 2, current.dimension(1) / 2, current.dimension(2) / 2),
		    Eigen::Tensor<float, 3>(current.dimension(0) / 2, current.dimension(1) / 2, current.dimension(2) / 2),
		    Eigen::Tensor<float, 3>(current.dimension(0) / 2, current.dimension(1) / 2, current.dimension(2) / 2)};
		detail[0].setZero();
		detail[1].setZero();
		detail[2].setZero();
		detail[3].setZero();
		detail[4].setZero();
		detail[5].setZero();
		detail[6].setZero();
		next_approx.setZero();

		// Actually fill them
		detail = {};
		detail[0] = Eigen::Tensor<float, 3>(current.dimension(0) / 2, current.dimension(1) / 2, current.dimension(2) / 2);
		detail[1] = detail[0];
		detail[2] = detail[0];
		detail[3] = detail[0];
		detail[4] = detail[0];
		detail[5] = detail[0];
		detail[6] = detail[0];
		detail[0].setZero();
		detail[1].setZero();
		detail[2].setZero();
		detail[3].setZero();
		detail[4].setZero();
		detail[5].setZero();
		detail[6].setZero();
		next_approx.setZero();

		// Finally invoke decomposition with proper allocation
		detail = std::array<Eigen::Tensor<float, 3>, 7>{
		    Eigen::Tensor<float, 3>(current.dimension(0) / 2, current.dimension(1) / 2, current.dimension(2) / 2),
		    Eigen::Tensor<float, 3>(current.dimension(0) / 2, current.dimension(1) / 2, current.dimension(2) / 2),
		    Eigen::Tensor<float, 3>(current.dimension(0) / 2, current.dimension(1) / 2, current.dimension(2) / 2),
		    Eigen::Tensor<float, 3>(current.dimension(0) / 2, current.dimension(1) / 2, current.dimension(2) / 2),
		    Eigen::Tensor<float, 3>(current.dimension(0) / 2, current.dimension(1) / 2, current.dimension(2) / 2),
		    Eigen::Tensor<float, 3>(current.dimension(0) / 2, current.dimension(1) / 2, current.dimension(2) / 2),
		    Eigen::Tensor<float, 3>(current.dimension(0) / 2, current.dimension(1) / 2, current.dimension(2) / 2)};
		detail[0].setZero();
		detail[1].setZero();
		detail[2].setZero();
		detail[3].setZero();
		detail[4].setZero();
		detail[5].setZero();
		detail[6].setZero();
		next_approx.setZero();

		// Real decomposition
		detail = {};
		detail[0] = Eigen::Tensor<float, 3>(current.dimension(0) / 2, current.dimension(1) / 2, current.dimension(2) / 2);
		detail[1] = detail[0];
		detail[2] = detail[0];
		detail[3] = detail[0];
		detail[4] = detail[0];
		detail[5] = detail[0];
		detail[6] = detail[0];

		// Actually call level decomposition helper
		detail[0].setZero();
		detail[1].setZero();
		detail[2].setZero();
		detail[3].setZero();
		detail[4].setZero();
		detail[5].setZero();
		detail[6].setZero();
		next_approx.setZero();

		detail = std::array<Eigen::Tensor<float, 3>, 7>{
		    Eigen::Tensor<float, 3>(current.dimension(0) / 2, current.dimension(1) / 2, current.dimension(2) / 2),
		    Eigen::Tensor<float, 3>(current.dimension(0) / 2, current.dimension(1) / 2, current.dimension(2) / 2),
		    Eigen::Tensor<float, 3>(current.dimension(0) / 2, current.dimension(1) / 2, current.dimension(2) / 2),
		    Eigen::Tensor<float, 3>(current.dimension(0) / 2, current.dimension(1) / 2, current.dimension(2) / 2),
		    Eigen::Tensor<float, 3>(current.dimension(0) / 2, current.dimension(1) / 2, current.dimension(2) / 2),
		    Eigen::Tensor<float, 3>(current.dimension(0) / 2, current.dimension(1) / 2, current.dimension(2) / 2),
		    Eigen::Tensor<float, 3>(current.dimension(0) / 2, current.dimension(1) / 2, current.dimension(2) / 2)};

		// Finally performing decomposition correctly
		detail = {};  // let helper allocate inside
		detail[0] = Eigen::Tensor<float, 3>(current.dimension(0) / 2, current.dimension(1) / 2, current.dimension(2) / 2);
		detail[1] = detail[0];
		detail[2] = detail[0];
		detail[3] = detail[0];
		detail[4] = detail[0];
		detail[5] = detail[0];
		detail[6] = detail[0];

		// Use helper to populate approx and details
		detail::decompose_level(current, next_approx, detail);

		coeffs.details.push_back(detail);
		current = next_approx;  // for next level
	}
	coeffs.coarse = current;
	// flatten
	Eigen::VectorXf arr;
	std::vector<CoeffSlice> slices;
	detail::coeffs_to_array(coeffs, arr, slices);
	return {arr, slices, coeffs};
}

// Reconstruct from flat representation.
inline Eigen::Tensor<float, 3> waverec3(const Eigen::VectorXf& arr, const std::vector<CoeffSlice>& slices, const std::string& wavelet,
                                        int level) {
	if (wavelet != "haar") {
		throw std::runtime_error("Only 'haar' wavelet is implemented in this version.");
	}
	// rebuild hierarchical coeffs
	WaveletCoeffs3D coeffs = detail::array_to_coeffs(arr, slices, level);
	Eigen::Tensor<float, 3> current = coeffs.coarse;
	for (int lvl = level - 1; lvl >= 0; --lvl) {
		Eigen::array<Eigen::Tensor<float, 3>, 7>& dets = coeffs.details[lvl];
		// Compute shape of upsampled (previous) approximation
		int nx = current.dimension(0) * 2;
		int ny = current.dimension(1) * 2;
		int nz = current.dimension(2) * 2;
		Eigen::Tensor<float, 3> recon(nx, ny, nz);
		detail::reconstruct_level(current, dets, recon);
		current = std::move(recon);
	}
	return current;
}
