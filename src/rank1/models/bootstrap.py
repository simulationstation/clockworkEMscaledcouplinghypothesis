"""
Parametric bootstrap for rank hypothesis testing.

Implements the test statistic:
Λ = χ²(rank-1) - χ²(rank-2)

Under H0 (rank-1 is true), we generate pseudo-data from the rank-1
fit and compute the null distribution of Λ.
"""

from dataclasses import dataclass, field
from typing import Optional, Callable

import numpy as np
from scipy import linalg

from rank1.models.lowrank import Rank1Model, Rank2Model
from rank1.models.fit import LowRankFitter, FitResult
from rank1.utils.parallel import parallel_map
from rank1.utils.stats import clopper_pearson_ci
from rank1.logging import get_logger

logger = get_logger()


@dataclass
class BootstrapResult:
    """Result of parametric bootstrap hypothesis test."""

    # Observed statistics
    chi2_rank1: float
    chi2_rank2: float
    lambda_obs: float

    # Bootstrap distribution
    lambda_null: np.ndarray
    n_bootstrap: int

    # P-value and confidence interval
    p_value: float
    p_value_ci: tuple[float, float]

    # Additional diagnostics
    fit_rank1: FitResult
    fit_rank2: FitResult

    # Metadata
    seed: int = 42
    n_failed: int = 0

    @property
    def is_significant(self) -> bool:
        """Check if p-value is below 0.05."""
        return self.p_value < 0.05

    def summary(self) -> dict:
        """Get summary statistics."""
        return {
            "chi2_rank1": self.chi2_rank1,
            "chi2_rank2": self.chi2_rank2,
            "lambda_obs": self.lambda_obs,
            "p_value": self.p_value,
            "p_value_ci_lower": self.p_value_ci[0],
            "p_value_ci_upper": self.p_value_ci[1],
            "n_bootstrap": self.n_bootstrap,
            "n_failed": self.n_failed,
            "is_significant": self.is_significant,
        }


class BootstrapTester:
    """
    Parametric bootstrap tester for rank-1 vs rank-2 hypothesis.

    The test statistic is:
    Λ = χ²(rank-1) - χ²(rank-2)

    Larger Λ indicates the data prefers rank-2 over rank-1.

    Under H0 (rank-1 is true), we simulate pseudo-data from the
    rank-1 fit and compute Λ for each replicate to build the
    null distribution.
    """

    def __init__(
        self,
        n_bootstrap: int = 1000,
        seed: int = 42,
        n_jobs: int = -1,
        use_parallel: bool = True,
    ):
        """
        Initialize bootstrap tester.

        Args:
            n_bootstrap: Number of bootstrap iterations
            seed: Random seed for reproducibility
            n_jobs: Number of parallel jobs (-1 for auto)
            use_parallel: Whether to use parallel execution
        """
        self.n_bootstrap = n_bootstrap
        self.seed = seed
        self.n_jobs = n_jobs
        self.use_parallel = use_parallel

        self.fitter = LowRankFitter(check_health=False)

    def test(
        self,
        row_idx: np.ndarray,
        col_idx: np.ndarray,
        values: np.ndarray,
        errors: np.ndarray,
        n_rows: int,
        n_cols: int,
        covariance: Optional[np.ndarray] = None,
    ) -> BootstrapResult:
        """
        Run the rank-1 vs rank-2 hypothesis test.

        Args:
            row_idx: Row indices of observations
            col_idx: Column indices of observations
            values: Observed values
            errors: Observation uncertainties
            n_rows: Number of matrix rows
            n_cols: Number of matrix columns
            covariance: Full covariance matrix (optional)

        Returns:
            BootstrapResult with test statistics and p-value
        """
        n_obs = len(values)

        logger.info(f"Starting bootstrap test with {self.n_bootstrap} iterations")

        # Fit rank-1 and rank-2 to observed data
        fit_rank1 = self.fitter.fit(
            row_idx, col_idx, values, errors,
            n_rows, n_cols, rank=1, covariance=covariance
        )
        fit_rank2 = self.fitter.fit(
            row_idx, col_idx, values, errors,
            n_rows, n_cols, rank=2, covariance=covariance
        )

        chi2_rank1 = fit_rank1.chi2
        chi2_rank2 = fit_rank2.chi2
        lambda_obs = chi2_rank1 - chi2_rank2

        logger.debug(
            f"Observed: chi2_rank1={chi2_rank1:.2f}, "
            f"chi2_rank2={chi2_rank2:.2f}, Λ={lambda_obs:.2f}"
        )

        # Generate bootstrap null distribution
        if self.use_parallel and self.n_bootstrap > 10:
            lambda_null, n_failed = self._bootstrap_parallel(
                fit_rank1.model, row_idx, col_idx, errors, n_rows, n_cols, covariance
            )
        else:
            lambda_null, n_failed = self._bootstrap_sequential(
                fit_rank1.model, row_idx, col_idx, errors, n_rows, n_cols, covariance
            )

        # Compute p-value using (k+1)/(B+1) estimator
        k = np.sum(lambda_null >= lambda_obs)
        B = len(lambda_null)
        p_value = (k + 1) / (B + 1)

        # Clopper-Pearson confidence interval
        p_ci = clopper_pearson_ci(k, B, alpha=0.05)

        logger.info(
            f"Bootstrap complete: Λ={lambda_obs:.2f}, "
            f"p-value={p_value:.4f} [{p_ci[0]:.4f}, {p_ci[1]:.4f}]"
        )

        return BootstrapResult(
            chi2_rank1=chi2_rank1,
            chi2_rank2=chi2_rank2,
            lambda_obs=lambda_obs,
            lambda_null=lambda_null,
            n_bootstrap=self.n_bootstrap,
            p_value=p_value,
            p_value_ci=p_ci,
            fit_rank1=fit_rank1,
            fit_rank2=fit_rank2,
            seed=self.seed,
            n_failed=n_failed,
        )

    def _bootstrap_sequential(
        self,
        null_model: Rank1Model,
        row_idx: np.ndarray,
        col_idx: np.ndarray,
        errors: np.ndarray,
        n_rows: int,
        n_cols: int,
        covariance: Optional[np.ndarray],
    ) -> tuple[np.ndarray, int]:
        """Sequential bootstrap iterations."""
        rng = np.random.default_rng(self.seed)
        lambda_null = []
        n_failed = 0

        # Precompute Cholesky factor if covariance provided
        if covariance is not None:
            L = linalg.cholesky(covariance, lower=True)
        else:
            L = None

        # Expected values under null
        y0 = null_model.predict(row_idx, col_idx)

        for b in range(self.n_bootstrap):
            try:
                # Generate pseudo-data
                if L is not None:
                    # Full covariance
                    z = rng.standard_normal(len(y0))
                    noise = L @ z
                else:
                    # Diagonal
                    noise = rng.standard_normal(len(y0)) * errors

                y_pseudo = y0 + noise

                # Fit both models
                fit1 = self.fitter.fit(
                    row_idx, col_idx, y_pseudo, errors,
                    n_rows, n_cols, rank=1, covariance=covariance
                )
                fit2 = self.fitter.fit(
                    row_idx, col_idx, y_pseudo, errors,
                    n_rows, n_cols, rank=2, covariance=covariance
                )

                lambda_b = fit1.chi2 - fit2.chi2
                lambda_null.append(lambda_b)

            except Exception as e:
                logger.debug(f"Bootstrap iteration {b} failed: {e}")
                n_failed += 1

        return np.array(lambda_null), n_failed

    def _bootstrap_parallel(
        self,
        null_model: Rank1Model,
        row_idx: np.ndarray,
        col_idx: np.ndarray,
        errors: np.ndarray,
        n_rows: int,
        n_cols: int,
        covariance: Optional[np.ndarray],
    ) -> tuple[np.ndarray, int]:
        """Parallel bootstrap iterations."""

        # Precompute what we need
        if covariance is not None:
            L = linalg.cholesky(covariance, lower=True)
        else:
            L = None

        y0 = null_model.predict(row_idx, col_idx)

        # Create work items with different seeds
        work_items = list(range(self.n_bootstrap))

        def bootstrap_one(b: int) -> Optional[float]:
            """Single bootstrap iteration."""
            try:
                rng = np.random.default_rng(self.seed + b)

                if L is not None:
                    z = rng.standard_normal(len(y0))
                    noise = L @ z
                else:
                    noise = rng.standard_normal(len(y0)) * errors

                y_pseudo = y0 + noise

                # Create fresh fitter for this worker
                fitter = LowRankFitter(check_health=False)

                fit1 = fitter.fit(
                    row_idx, col_idx, y_pseudo, errors,
                    n_rows, n_cols, rank=1, covariance=covariance
                )
                fit2 = fitter.fit(
                    row_idx, col_idx, y_pseudo, errors,
                    n_rows, n_cols, rank=2, covariance=covariance
                )

                return fit1.chi2 - fit2.chi2

            except Exception:
                return None

        # Run in parallel
        results = parallel_map(
            bootstrap_one,
            work_items,
            n_jobs=self.n_jobs,
            desc="Bootstrap",
            show_progress=True,
        )

        # Filter out failures
        lambda_null = [r for r in results if r is not None]
        n_failed = len(results) - len(lambda_null)

        return np.array(lambda_null), n_failed


def run_rank_test(
    row_idx: np.ndarray,
    col_idx: np.ndarray,
    values: np.ndarray,
    errors: np.ndarray,
    n_rows: int,
    n_cols: int,
    n_bootstrap: int = 1000,
    seed: int = 42,
    covariance: Optional[np.ndarray] = None,
    use_parallel: bool = True,
) -> BootstrapResult:
    """
    Convenience function to run the rank-1 hypothesis test.

    Tests H0: data is consistent with rank-1 model
    vs H1: rank-2 provides significantly better fit

    Args:
        row_idx, col_idx, values, errors: Observation data
        n_rows, n_cols: Matrix dimensions
        n_bootstrap: Number of bootstrap samples
        seed: Random seed
        covariance: Full covariance matrix (optional)
        use_parallel: Use parallel bootstrap

    Returns:
        BootstrapResult with p-value and diagnostics
    """
    tester = BootstrapTester(
        n_bootstrap=n_bootstrap,
        seed=seed,
        use_parallel=use_parallel,
    )

    return tester.test(
        row_idx, col_idx, values, errors,
        n_rows, n_cols, covariance=covariance
    )
