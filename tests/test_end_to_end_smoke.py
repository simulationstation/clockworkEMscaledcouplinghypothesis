"""
End-to-end smoke tests for the rank-1 analysis pipeline.

These tests verify that the full pipeline runs without errors,
but use minimal bootstrap iterations for speed.
"""

import pytest
from pathlib import Path
import json

import numpy as np


@pytest.mark.slow
class TestHiggsAnalysisSmoke:
    """Smoke test for Higgs analysis."""

    def test_higgs_run(self, temp_dir):
        """Test that Higgs analysis runs without errors."""
        from rank1.datasets import HiggsATLASDataset
        from rank1.analysis import HiggsRankAnalysis

        dataset = HiggsATLASDataset(
            cache_dir=temp_dir / "cache",
            raw_dir=temp_dir / "raw",
            processed_dir=temp_dir / "processed",
        )

        analysis = HiggsRankAnalysis(
            dataset=dataset,
            output_dir=temp_dir / "outputs",
        )

        # Run with minimal bootstrap
        try:
            result = analysis.run(n_bootstrap=50, seed=42)

            assert result.p_value >= 0
            assert result.p_value <= 1
            assert result.chi2_rank1 >= 0
            assert result.ndof_rank1 > 0

            # Check outputs exist
            assert (temp_dir / "outputs" / "results.json").exists()

        except Exception as e:
            # If network fails, that's OK for smoke test
            if "connection" in str(e).lower() or "network" in str(e).lower():
                pytest.skip(f"Network error: {e}")
            raise


@pytest.mark.slow
class TestElasticAnalysisSmoke:
    """Smoke test for elastic scattering analysis."""

    def test_elastic_run(self, temp_dir):
        """Test that elastic analysis runs without errors."""
        from rank1.datasets import ElasticTOTEMDataset
        from rank1.analysis import ElasticRankAnalysis

        dataset = ElasticTOTEMDataset(
            cache_dir=temp_dir / "cache",
            raw_dir=temp_dir / "raw",
            processed_dir=temp_dir / "processed",
        )

        analysis = ElasticRankAnalysis(
            dataset=dataset,
            output_dir=temp_dir / "outputs",
        )

        try:
            result = analysis.run(n_bootstrap=50, seed=42)

            assert result.p_value >= 0
            assert result.chi2_rank1 >= 0

        except Exception as e:
            if "connection" in str(e).lower() or "network" in str(e).lower():
                pytest.skip(f"Network error: {e}")
            raise


@pytest.mark.slow
class TestDiffractiveAnalysisSmoke:
    """Smoke test for diffractive DIS analysis."""

    def test_diffractive_run(self, temp_dir):
        """Test that diffractive analysis runs without errors."""
        from rank1.datasets import DiffractiveDISDataset
        from rank1.analysis import DiffractiveRankAnalysis

        dataset = DiffractiveDISDataset(
            cache_dir=temp_dir / "cache",
            raw_dir=temp_dir / "raw",
            processed_dir=temp_dir / "processed",
        )

        analysis = DiffractiveRankAnalysis(
            dataset=dataset,
            output_dir=temp_dir / "outputs",
        )

        try:
            result = analysis.run(n_bootstrap=50, seed=42)

            assert result.p_value >= 0

        except Exception as e:
            if "connection" in str(e).lower() or "network" in str(e).lower():
                pytest.skip(f"Network error: {e}")
            raise


class TestSyntheticPipeline:
    """Test full pipeline on synthetic data (no network required)."""

    def test_synthetic_rank1_pipeline(self, temp_dir, sample_matrix_data):
        """Test pipeline on synthetic rank-1 data."""
        from rank1.datasets.base import MatrixData, MatrixObservation
        from rank1.models.bootstrap import BootstrapTester

        # Create MatrixData from fixture
        observations = []
        for i in range(len(sample_matrix_data["values"])):
            obs = MatrixObservation(
                row_idx=sample_matrix_data["row_idx"][i],
                col_idx=sample_matrix_data["col_idx"][i],
                value=sample_matrix_data["values"][i],
                total_err=sample_matrix_data["errors"][i],
            )
            observations.append(obs)

        matrix_data = MatrixData(
            name="synthetic",
            description="Synthetic rank-1 test data",
            row_labels=[f"row_{i}" for i in range(sample_matrix_data["n_rows"])],
            col_labels=[f"col_{j}" for j in range(sample_matrix_data["n_cols"])],
            observations=observations,
        )

        # Run bootstrap test
        rows, cols, values, errors = matrix_data.to_vectors()

        tester = BootstrapTester(n_bootstrap=100, seed=42, use_parallel=False)
        result = tester.test(
            rows, cols, values, errors,
            matrix_data.n_rows, matrix_data.n_cols
        )

        # Rank-1 data should not be rejected
        assert result.p_value > 0.01

    def test_synthetic_rank2_detection(self, temp_dir, sample_rank2_data):
        """Test that pipeline can detect rank-2 data."""
        from rank1.models.bootstrap import BootstrapTester

        tester = BootstrapTester(n_bootstrap=100, seed=42, use_parallel=False)

        result = tester.test(
            sample_rank2_data["row_idx"],
            sample_rank2_data["col_idx"],
            sample_rank2_data["values"],
            sample_rank2_data["errors"],
            sample_rank2_data["n_rows"],
            sample_rank2_data["n_cols"],
        )

        # Lambda should be positive (rank-2 fits better than rank-1)
        assert result.lambda_obs > 0


class TestCLI:
    """Test CLI commands."""

    def test_info_command(self):
        """Test that info command runs."""
        from typer.testing import CliRunner
        from rank1.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["info"])

        assert result.exit_code == 0
        assert "rank1-factorization" in result.stdout

    def test_help_command(self):
        """Test that help command runs."""
        from typer.testing import CliRunner
        from rank1.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        assert "fetch" in result.stdout
        assert "run" in result.stdout
