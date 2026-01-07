"""
Tests for low-rank model fitting on synthetic data.
"""

import pytest
import numpy as np

from rank1.models.lowrank import Rank1Model, Rank2Model
from rank1.models.fit import LowRankFitter, FitMethod


class TestRank1Model:
    """Test Rank1Model class."""

    def test_predict_matrix(self):
        """Test that predict_matrix gives u @ v.T."""
        model = Rank1Model(n_rows=4, n_cols=3)
        model.u = np.array([1.0, 2.0, 3.0, 4.0])
        model.v = np.array([0.5, 0.5, 0.707])
        model.v /= np.linalg.norm(model.v)

        M = model.predict_matrix()

        assert M.shape == (4, 3)
        expected = np.outer(model.u, model.v)
        np.testing.assert_array_almost_equal(M, expected)

    def test_predict_at_indices(self):
        """Test prediction at specific indices."""
        model = Rank1Model(n_rows=3, n_cols=3)
        model.u = np.array([1.0, 2.0, 3.0])
        model.v = np.array([1.0, 0.0, 0.0])

        row_idx = np.array([0, 1, 2])
        col_idx = np.array([0, 0, 0])

        pred = model.predict(row_idx, col_idx)
        expected = np.array([1.0, 2.0, 3.0])

        np.testing.assert_array_almost_equal(pred, expected)

    def test_from_svd(self):
        """Test initialization from SVD."""
        M = np.array([[1, 2, 3], [2, 4, 6], [3, 6, 9]])  # Rank-1

        model = Rank1Model.from_svd(M)
        M_approx = model.predict_matrix()

        np.testing.assert_array_almost_equal(M, M_approx, decimal=5)

    def test_get_set_params(self):
        """Test parameter getting and setting."""
        model = Rank1Model(n_rows=3, n_cols=4)
        model.u = np.array([1.0, 2.0, 3.0])
        model.v = np.array([0.5, 0.5, 0.5, 0.5])

        params = model.get_params()
        assert len(params) == 3 + 3  # u + v[:-1]

        # Perturb and reset
        model.u = np.zeros(3)
        model.set_params(params)

        np.testing.assert_array_almost_equal(model.u, np.array([1.0, 2.0, 3.0]))


class TestRank2Model:
    """Test Rank2Model class."""

    def test_predict_matrix(self):
        """Test rank-2 prediction."""
        model = Rank2Model(n_rows=4, n_cols=3)
        model.U = np.array([[1, 0], [0, 1], [1, 1], [0.5, 0.5]])
        model.V = np.array([[1, 0], [0, 1], [0.707, 0.707]])
        model.V, _ = np.linalg.qr(model.V)

        M = model.predict_matrix()

        assert M.shape == (4, 3)
        expected = model.U @ model.V.T
        np.testing.assert_array_almost_equal(M, expected)

    def test_from_svd(self):
        """Test initialization from SVD."""
        np.random.seed(42)
        U = np.random.randn(5, 2)
        V = np.random.randn(4, 2)
        V, _ = np.linalg.qr(V)
        M = U @ V.T

        model = Rank2Model.from_svd(M)
        M_approx = model.predict_matrix()

        np.testing.assert_array_almost_equal(M, M_approx, decimal=5)


class TestLowRankFitter:
    """Test LowRankFitter class."""

    def test_fit_rank1_als(self, sample_matrix_data):
        """Test rank-1 fitting with ALS."""
        fitter = LowRankFitter(method=FitMethod.ALS, max_iterations=500)

        result = fitter.fit(
            sample_matrix_data["row_idx"],
            sample_matrix_data["col_idx"],
            sample_matrix_data["values"],
            sample_matrix_data["errors"],
            sample_matrix_data["n_rows"],
            sample_matrix_data["n_cols"],
            rank=1,
        )

        assert result.success or result.n_iterations == 500
        assert result.chi2 >= 0

        # Check reconstruction is close to true
        M_fit = result.predicted_matrix
        M_true = sample_matrix_data["true_matrix"]

        # Relative error should be reasonable
        rel_error = np.linalg.norm(M_fit - M_true) / np.linalg.norm(M_true)
        assert rel_error < 0.2

    def test_fit_rank1_nlls(self, sample_matrix_data):
        """Test rank-1 fitting with nonlinear least squares."""
        fitter = LowRankFitter(method=FitMethod.NLLS)

        result = fitter.fit(
            sample_matrix_data["row_idx"],
            sample_matrix_data["col_idx"],
            sample_matrix_data["values"],
            sample_matrix_data["errors"],
            sample_matrix_data["n_rows"],
            sample_matrix_data["n_cols"],
            rank=1,
        )

        assert result.chi2 >= 0

        M_fit = result.predicted_matrix
        M_true = sample_matrix_data["true_matrix"]

        rel_error = np.linalg.norm(M_fit - M_true) / np.linalg.norm(M_true)
        assert rel_error < 0.2

    def test_fit_rank2(self, sample_rank2_data):
        """Test rank-2 fitting."""
        fitter = LowRankFitter(method=FitMethod.ALS)

        result = fitter.fit(
            sample_rank2_data["row_idx"],
            sample_rank2_data["col_idx"],
            sample_rank2_data["values"],
            sample_rank2_data["errors"],
            sample_rank2_data["n_rows"],
            sample_rank2_data["n_cols"],
            rank=2,
        )

        assert result.chi2 >= 0

        M_fit = result.predicted_matrix
        M_true = sample_rank2_data["true_matrix"]

        rel_error = np.linalg.norm(M_fit - M_true) / np.linalg.norm(M_true)
        assert rel_error < 0.3

    def test_sparse_matrix(self, sparse_matrix_data):
        """Test fitting with sparse observations."""
        fitter = LowRankFitter(method=FitMethod.ALS)

        result = fitter.fit(
            sparse_matrix_data["row_idx"],
            sparse_matrix_data["col_idx"],
            sparse_matrix_data["values"],
            sparse_matrix_data["errors"],
            sparse_matrix_data["n_rows"],
            sparse_matrix_data["n_cols"],
            rank=1,
        )

        assert result.success or result.n_iterations > 0

    def test_chi2_ndof(self, sample_matrix_data):
        """Test chi2/ndof calculation."""
        fitter = LowRankFitter(method=FitMethod.ALS)

        result = fitter.fit(
            sample_matrix_data["row_idx"],
            sample_matrix_data["col_idx"],
            sample_matrix_data["values"],
            sample_matrix_data["errors"],
            sample_matrix_data["n_rows"],
            sample_matrix_data["n_cols"],
            rank=1,
        )

        # ndof should be n_obs - n_params
        n_obs = len(sample_matrix_data["values"])
        n_params = sample_matrix_data["n_rows"] + sample_matrix_data["n_cols"] - 1
        expected_ndof = n_obs - n_params

        assert result.ndof == expected_ndof

        # chi2/ndof should be reasonable for rank-1 data
        chi2_ndof = result.chi2_ndof
        assert 0.1 < chi2_ndof < 5.0


class TestMultistartFitting:
    """Test multistart fitting for stability."""

    def test_multistart_consistency(self, sample_matrix_data):
        """Test that multistart gives consistent results."""
        fitter = LowRankFitter(n_starts=5)

        best, all_results = fitter.fit_multistart(
            sample_matrix_data["row_idx"],
            sample_matrix_data["col_idx"],
            sample_matrix_data["values"],
            sample_matrix_data["errors"],
            sample_matrix_data["n_rows"],
            sample_matrix_data["n_cols"],
            rank=1,
            seed=42,
        )

        # Best should have lowest chi2
        assert best.chi2 == min(r.chi2 for r in all_results)

        # All results should have similar chi2 for easy problem
        chi2_values = [r.chi2 for r in all_results]
        assert max(chi2_values) / min(chi2_values) < 2.0
