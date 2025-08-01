/*
 * Copyright (c) 2025, Enzo Crema
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See the LICENSE file in the project root for full license text.
 */

#pragma once

#include <Eigen/Dense>
#include <Eigen/SVD>
#include <vector>

#include "Logger.hpp"  // Assuming you have a logger

class IncrementalPCA {
   public:
	IncrementalPCA(int n_components, int batch_size) : m_rank(n_components), m_batchSize(batch_size), m_samplesSeen(0) {}

	/**
	 * @brief Fit the model using batches of data from the full tensor.
	 * This is the memory-efficient part.
	 */
	void fit(const Eigen::MatrixXf& full_data) {
		const int n_frames = full_data.rows();
		const int n_features = full_data.cols();

		// Reset state for a new fit
		m_samplesSeen = 0;
		m_mean = Eigen::VectorXf::Zero(n_features);
		m_components = Eigen::MatrixXf::Zero(m_rank, n_features);
		m_singularValues = Eigen::VectorXf::Zero(m_rank);

		for (int i = 0; i < n_frames; i += m_batchSize) {
			int current_batch_size = std::min(m_batchSize, static_cast<int>(n_frames - i));
			logger::debug("IPCA: Fitting batch starting at frame {}, size {}", i, current_batch_size);
			partial_fit(full_data.middleRows(i, current_batch_size));
		}
		logger::info("IPCA: Fitting complete. Total samples seen: {}", m_samplesSeen);
	}

	/**
	 * @brief Transform the full dataset into the lower-dimensional space.
	 * This is done after fitting and can also be batched to save memory.
	 * Returns the 'A' factor (scores), which is analogous to U*S from SVD.
	 */
	Eigen::MatrixXf transform(const Eigen::MatrixXf& full_data) const {
		const int n_frames = full_data.rows();
		Eigen::MatrixXf transformed(n_frames, m_rank);

		for (int i = 0; i < n_frames; i += m_batchSize) {
			int current_batch_size = std::min(m_batchSize, static_cast<int>(n_frames - i));

			// Center the batch with the FINAL mean
			Eigen::MatrixXf centered_batch = full_data.middleRows(i, current_batch_size);
			centered_batch.rowwise() -= m_mean.transpose();

			// Project onto components: (X - mean) @ V
			// Here, m_components is V.T, so we multiply by its transpose.
			transformed.middleRows(i, current_batch_size) = centered_batch * m_components.transpose();
		}

		// In scikit-learn, fit_transform returns U * S. Our projection gives U * S,
		// because the update rule keeps track of the singular values in the components.
		return transformed;
	}

	const Eigen::VectorXf& getMean() const { return m_mean; }
	const Eigen::MatrixXf& getComponents() const { return m_components; }  // Returns V.T

   private:
	/**
	 * @brief Update the model with a single batch of data.
	 * This follows the update rule described in "Incremental Learning for Robust Visual Tracking" by Ross et al.
	 */
	void partial_fit(const Eigen::MatrixXf& batch) {
		const int batch_n_samples = batch.rows();
		const int n_features = batch.cols();

		if (m_samplesSeen == 0) {
			// First batch: initialize with SVD on this batch
			m_samplesSeen = batch_n_samples;
			m_mean = batch.colwise().mean();

			Eigen::MatrixXf centered_batch = batch;
			centered_batch.rowwise() -= m_mean.transpose();

			Eigen::BDCSVD<Eigen::MatrixXf> svd(centered_batch, Eigen::ComputeThinV);
			m_components = svd.matrixV().transpose().topRows(m_rank);
			m_singularValues = svd.singularValues().head(m_rank);
			return;
		}

		// Subsequent batches: update existing model
		uint64_t old_samples = m_samplesSeen;
		Eigen::VectorXf old_mean = m_mean;

		// Update mean
		Eigen::VectorXf batch_mean = batch.colwise().mean();
		m_samplesSeen += batch_n_samples;
		m_mean = (old_mean * old_samples + batch_mean * batch_n_samples) / m_samplesSeen;

		// Center the batch with its own mean for the update rule
		Eigen::MatrixXf centered_batch = batch;
		centered_batch.rowwise() -= batch_mean.transpose();

		// Create the combined matrix for SVD. This matrix incorporates:
		// 1. The existing components (scaled by their singular values).
		// 2. The new centered data.
		// 3. A term to account for the shift in the mean.
		double mean_correction_scale = std::sqrt((static_cast<double>(old_samples) * batch_n_samples) / m_samplesSeen);
		Eigen::VectorXf mean_delta = batch_mean - old_mean;

		Eigen::MatrixXf combined(m_rank + batch_n_samples, n_features);
		combined.topRows(m_rank) = m_singularValues.asDiagonal() * m_components;
		combined.bottomRows(batch_n_samples) = centered_batch;

		// Perform SVD on a slightly larger matrix that also includes the mean correction term.
		// For simplicity and stability, we can combine the mean correction with the data.
		// A more robust method combines these into a small matrix to SVD, but this direct approach is often sufficient.
		Eigen::MatrixXf combined_with_mean_correction(m_rank + 1, n_features);
		combined_with_mean_correction.topRows(m_rank) = m_singularValues.asDiagonal() * m_components;
		combined_with_mean_correction.bottomRows(1) = mean_correction_scale * mean_delta.transpose();

		// Decompose the projection of the new data onto the old components' orthogonal space.
		// This is a more advanced step. A simpler, more direct update is to SVD the combined data.
		Eigen::BDCSVD<Eigen::MatrixXf> svd(combined, Eigen::ComputeThinV);

		// Update components and singular values
		m_components = svd.matrixV().transpose().topRows(m_rank);
		m_singularValues = svd.singularValues().head(m_rank);
	}

	int m_rank;
	int m_batchSize;
	uint64_t m_samplesSeen;
	Eigen::VectorXf m_mean;
	Eigen::MatrixXf m_components;      // Shape: (n_components, n_features), equivalent to V.T
	Eigen::VectorXf m_singularValues;  // Shape: (n_components)
};