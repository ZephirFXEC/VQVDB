/*
 * Copyright (c) 2025, Enzo Crema
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See the LICENSE file in the project root for full license text.
 */
#pragma once

#include <Eigen/Eigen>
#include <vector>

// Wavelet filters: decomposition low/high, reconstruction low/high
struct Wavelet {
	std::vector<float> Lo_D; // decomposition low-pass
	std::vector<float> Hi_D; // decomposition high-pass
	std::vector<float> Lo_R; // reconstruction low-pass
	std::vector<float> Hi_R; // reconstruction high-pass
};

namespace wavelet {
// Predefined Wavelets
inline Wavelet haar() {
	// Haar: [1/sqrt(2), 1/sqrt(2)]
	const float s = 1.0f / std::sqrt(2.0f);
	Wavelet w;
	w.Lo_D = {s, s};
	w.Hi_D = {-s, s};
	w.Lo_R = {s, s};
	w.Hi_R = {s, -s};
	return w;
}

inline static Wavelet db2() {
	// Daubechies 2 (db2), also known as D4
	// Decomposition low-pass (h0)
	Wavelet w;
	w.Lo_D = {(1 + std::sqrtf(3)) / (4 * std::sqrtf(2)), (3 + std::sqrtf(3)) / (4 * std::sqrtf(2)),
	          (3 - std::sqrtf(3)) / (4 * std::sqrtf(2)),
	          (1 - std::sqrtf(3)) / (4 * std::sqrtf(2))};
	// Decomposition high-pass (h1) alternating signs reverse
	w.Hi_D = {(1 - std::sqrtf(3)) / (4 * std::sqrtf(2)), -(3 - std::sqrtf(3)) / (4 * std::sqrtf(2)),
	          (3 + std::sqrtf(3)) / (4 * std::sqrtf(2)),
	          -(1 + std::sqrtf(3)) / (4 * std::sqrtf(2))};
	// Reconstruction low-pass is reverse of Lo_D
	w.Lo_R = {(1 - std::sqrtf(3)) / (4 * std::sqrtf(2)), (3 - std::sqrtf(3)) / (4 * std::sqrtf(2)),
	          (3 + std::sqrtf(3)) / (4 * std::sqrtf(2)),
	          (1 + std::sqrtf(3)) / (4 * std::sqrtf(2))};
	// Reconstruction high-pass is reverse of Hi_D with alternating signs
	w.Hi_R = {-(1 + std::sqrtf(3)) / (4 * std::sqrtf(2)), (3 + std::sqrtf(3)) / (4 * std::sqrtf(2)),
	          -(3 - std::sqrtf(3)) / (4 * std::sqrtf(2)),
	          (1 - std::sqrtf(3)) / (4 * std::sqrtf(2))};
	return w;
}

// Private helper for symmetric boundary handling. This mode is equivalent to
// `pywt`'s 'symmetric' or MATLAB's 'symh', where the boundary sample is repeated.
template <typename T>
static int get_symmetric_index(T index, T length) {
	if (length <= 1) return 0;
	// e.g., for signal [a,b,c,d], it's treated as ...b,a,|a,b,c,d|,d,c...
	while (index < 0 || index >= length) {
		if (index < 0) {
			index = -index - 1;
		} else {
			// index >= length
			index = 2 * length - 1 - index;
		}
	}
	return static_cast<int>(index);
}

// Convolution with symmetric padding at the boundaries.
template <typename Vec>
static Vec symmetricConv(const Vec& signal, const std::vector<float>& h) {
	const int N = signal.size();
	if (N == 0) return Vec();

	const int L = static_cast<int>(h.size());
	Vec out(N);
	out.setZero();

	for (int n = 0; n < N; ++n) {
		float sum = 0.f;
		for (int k = 0; k < L; ++k) {
			// The index into the signal for convolution
			int idx = n - k;
			// Get the corresponding index with symmetric padding
			int padded_idx = get_symmetric_index(idx, N);
			sum += h[k] * signal(padded_idx);
		}
		out(n) = sum;
	}
	return out;
}

// 1D DWT: input --> approx (low-pass) and detail (high-pass)
inline void dwt(const Eigen::VectorXf& input, const Wavelet& w, Eigen::VectorXf& approx, Eigen::VectorXf& detail, int framecount) {
	const int N = input.size();
	// filter + symmetric conv
	Eigen::VectorXf lo = symmetricConv(input, w.Lo_D);
	Eigen::VectorXf hi = symmetricConv(input, w.Hi_D);
	// downsample by 2
	int outSize = (framecount + 1) / 2;
	approx.resize(outSize);
	detail.resize(outSize);
	for (int i = 0; i < outSize; ++i) {
		approx(i) = lo(2 * i);
		detail(i) = hi(2 * i);
	}
}

// 1D inverse DWT: approx + detail --> reconstructed signal
inline Eigen::VectorXf idwt(const Eigen::VectorXf& approx, const Eigen::VectorXf& detail, const Wavelet& w, int framecount) {
	const int N = approx.size() * 2; // Full size before potential trimming
	// upsample
	Eigen::VectorXf upA(N);
	upA.setZero();
	Eigen::VectorXf upD(N);
	upD.setZero();
	for (int i = 0; i < approx.size(); ++i) {
		upA(2 * i) = approx(i);
	}
	for (int i = 0; i < detail.size(); ++i) {
		upD(2 * i) = detail(i);
	}
	// symmetric conv with reconstruction filters
	Eigen::VectorXf recA = symmetricConv(upA, w.Lo_R);
	Eigen::VectorXf recD = symmetricConv(upD, w.Hi_R);
	Eigen::VectorXf result = recA + recD;

	// Resize to match the requested framecount if necessary
	if (framecount != N) {
		Eigen::VectorXf trimmed(framecount);
		for (int i = 0; i < framecount; ++i) {
			trimmed(i) = result(i);
		}
		return trimmed;
	}
	return result;
}

inline void dwt2(const Eigen::VectorXf& input,
                 const Wavelet& w,
                 Eigen::VectorXf& approx2,
                 Eigen::VectorXf& detail2,
                 Eigen::VectorXf& detail1,
                 int framecount) {
	// 1st level
	Eigen::VectorXf approx1;
	dwt(input, w, approx1, detail1, framecount);

	// 2nd level: use approx1 as new input
	int framecount1 = approx1.size();
	dwt(approx1, w, approx2, detail2, framecount1);
}

/// 2-level inverse DWT:
/// stitches A₂ + D₂ → A₁, then A₁ + D₁ → original
inline Eigen::VectorXf idwt2(const Eigen::VectorXf& approx2,
                             const Eigen::VectorXf& detail2,
                             const Eigen::VectorXf& detail1,
                             const Wavelet& w,
                             int framecount) {
	// length of level-1 approximation
	int framecount1 = (framecount + 1) / 2;

	// reconstruct level-1 approx from level-2 coeffs
	Eigen::VectorXf approx1 = idwt(approx2, detail2, w, framecount1);

	// reconstruct original
	return idwt(approx1, detail1, w, framecount);
}
} // namespace wavelet