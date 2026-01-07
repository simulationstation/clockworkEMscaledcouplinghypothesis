"""
Tests for parametric bootstrap hypothesis testing.
"""

import pytest
import numpy as np

from rank1.models.bootstrap import BootstrapTester, run_rank_test


class TestBootstrapTester:
    """Test BootstrapTester class."""

    def test_rank1_null_not_rejected(self, sample_matrix_data):
        """Test that rank-1 null is not rejected for rank-1 data."""
        tester = BootstrapTester(n_bootstrap=100, seed=42, use_parallel=False)

        result = tester.test(
            sample_matrix_data["row_idx"],
            sample_matrix_data["col_idx"],
            sample_matrix_data["values"],
            sample_matrix_data["errors"],
            sample_matrix_data["n_rows"],
            sample_matrix_data["n_cols"],
        )

        # p-value should be large (not rejected)
        assert result.p_value > 0.01

        # Lambda should be small
        assert result.lambda_obs < 20

    def test_rank2_data_may_be_rejected(self, sample_rank2_data):
        """Test that rank-2 data might be rejected as rank-1."""
        tester = BootstrapTester(n_bootstrap=100, seed=42, use_parallel=False)

        result = tester.test(
            sample_rank2_data["row_idx"],
            sample_rank2_data["col_idx"],
            sample_rank2_data["values"],
            sample_rank2_data["errors"],
            sample_rank2_data["n_rows"],
            sample_rank2_data["n_cols"],
        )

        # Lambda should be positive (rank-2 fits better)
        assert result.lambda_obs > 0

        # chi2_rank2 should be smaller
        assert result.chi2_rank2 <= result.chi2_rank1

    def test_bootstrap_distribution_length(self, sample_matrix_data):
        """Test bootstrap distribution has correct length."""
        n_bootstrap = 50
        tester = BootstrapTester(n_bootstrap=n_bootstrap, seed=42, use_parallel=False)

        result = tester.test(
            sample_matrix_data["row_idx"],
            sample_matrix_data["col_idx"],
            sample_matrix_data["values"],
            sample_matrix_data["errors"],
            sample_matrix_data["n_rows"],
            sample_matrix_data["n_cols"],
        )

        # Should have n_bootstrap - n_failed values
        assert len(result.lambda_null) >= n_bootstrap - 10

    def test_pvalue_ci_validity(self, sample_matrix_data):
        """Test p-value confidence interval is valid."""
        tester = BootstrapTester(n_bootstrap=100, seed=42, use_parallel=False)

        result = tester.test(
            sample_matrix_data["row_idx"],
            sample_matrix_data["col_idx"],
            sample_matrix_data["values"],
            sample_matrix_data["errors"],
            sample_matrix_data["n_rows"],
            sample_matrix_data["n_cols"],
        )

        # CI should contain p-value
        assert result.p_value_ci[0] <= result.p_value <= result.p_value_ci[1]

        # CI should be valid probabilities
        assert 0 <= result.p_value_ci[0] <= result.p_value_ci[1] <= 1

    def test_summary_dict(self, sample_matrix_data):
        """Test summary dictionary contains expected keys."""
        tester = BootstrapTester(n_bootstrap=50, seed=42, use_parallel=False)

        result = tester.test(
            sample_matrix_data["row_idx"],
            sample_matrix_data["col_idx"],
            sample_matrix_data["values"],
            sample_matrix_data["errors"],
            sample_matrix_data["n_rows"],
            sample_matrix_data["n_cols"],
        )

        summary = result.summary()

        expected_keys = [
            "chi2_rank1", "chi2_rank2", "lambda_obs",
            "p_value", "n_bootstrap", "is_significant"
        ]

        for key in expected_keys:
            assert key in summary


class TestRunRankTest:
    """Test convenience function run_rank_test."""

    def test_basic_call(self, sample_matrix_data):
        """Test basic function call."""
        result = run_rank_test(
            sample_matrix_data["row_idx"],
            sample_matrix_data["col_idx"],
            sample_matrix_data["values"],
            sample_matrix_data["errors"],
            sample_matrix_data["n_rows"],
            sample_matrix_data["n_cols"],
            n_bootstrap=50,
            seed=42,
            use_parallel=False,
        )

        assert result.p_value >= 0
        assert result.p_value <= 1
        assert result.n_bootstrap >= 40  # Allow some failures
