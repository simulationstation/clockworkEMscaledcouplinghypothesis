"""
Diffractive DIS Regge factorization rank-1 analysis.

Tests whether σ_r^D(β, Q², x_P) ≈ f_P(x_P) × σ_r(β, Q²) (separable).
"""

from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

from rank1.analysis.base import BaseRankAnalysis, AnalysisResult
from rank1.datasets.diffractive_dis import DiffractiveDISDataset
from rank1.logging import get_logger

logger = get_logger()


class DiffractiveRankAnalysis(BaseRankAnalysis):
    """
    Rank-1 analysis for diffractive DIS factorization.

    Tests whether the diffractive structure function factorizes
    into a Pomeron flux times a reduced cross section.
    """

    name = "diffractive_dis"
    description = "H1/ZEUS diffractive DIS Regge factorization test"

    def __init__(
        self,
        dataset: Optional[DiffractiveDISDataset] = None,
        experiment: str = "combined",
        **kwargs,
    ):
        if dataset is None:
            dataset = DiffractiveDISDataset(experiment=experiment)

        super().__init__(dataset=dataset, **kwargs)
        self.experiment = experiment

    def additional_cross_checks(self) -> list[dict]:
        """Diffractive DIS specific cross-checks."""
        checks = []

        # Check 1: x_P power law behavior
        xp_check = self._check_xp_power_law()
        if xp_check:
            checks.append(xp_check)

        # Check 2: Variance explained by rank-1
        var_check = self._check_variance_explained()
        if var_check:
            checks.append(var_check)

        # Check 3: Synthetic injection test
        synth_result = self._synthetic_injection_test()
        checks.append(synth_result)

        return checks

    def _check_xp_power_law(self) -> Optional[dict]:
        """Check if x_P dependence follows expected power law."""
        if self._result is None or self._result.fit_rank1 is None:
            return None

        try:
            model = self._result.fit_rank1.model

            if hasattr(model, "v"):
                v = model.v

                # Fit power law to v vs x_P
                col_labels = self._matrix_data.col_labels
                xp_values = []
                for label in col_labels:
                    # Parse x_P from label like "xP=0.001"
                    try:
                        xp = float(label.split("=")[1])
                        xp_values.append(xp)
                    except (IndexError, ValueError):
                        xp_values.append(np.nan)

                xp_values = np.array(xp_values)
                valid = np.isfinite(xp_values) & (xp_values > 0)

                if np.sum(valid) >= 3:
                    # Log-log linear fit
                    log_xp = np.log(xp_values[valid])
                    log_v = np.log(np.abs(v[valid]) + 1e-10)

                    coeffs = np.polyfit(log_xp, log_v, 1)
                    power = coeffs[0]

                    # Pomeron flux typically gives power ~ -1.1 to -1.2
                    expected_range = (-1.5, -0.8)
                    is_reasonable = expected_range[0] < power < expected_range[1]

                    return {
                        "name": "xp_power_law",
                        "passed": is_reasonable,
                        "message": f"x_P power: {power:.2f} (expected {expected_range})",
                        "details": {"power": float(power)},
                    }

        except Exception as e:
            logger.debug(f"x_P power check failed: {e}")

        return None

    def _check_variance_explained(self) -> Optional[dict]:
        """Check how much variance is explained by rank-1."""
        if self._result is None:
            return None

        try:
            values, errors, mask = self._matrix_data.to_matrix()
            M_hat = self._result.matrix_rank1

            # Compute variance explained
            residuals = np.where(mask > 0, values - M_hat, 0)
            total_var = np.nanvar(values[mask > 0])
            residual_var = np.nanvar(residuals[mask > 0])

            explained = 1 - residual_var / total_var if total_var > 0 else 0

            return {
                "name": "variance_explained",
                "passed": explained > 0.8,
                "message": f"Rank-1 explains {explained:.1%} of variance",
                "details": {"fraction_explained": float(explained)},
            }

        except Exception as e:
            logger.debug(f"Variance check failed: {e}")
            return None

    def _synthetic_injection_test(self) -> dict:
        """Inject synthetic Regge-factorizable data."""
        if self._matrix_data is None:
            return {
                "name": "synthetic_injection",
                "passed": False,
                "message": "No data available",
                "details": {},
            }

        try:
            from rank1.models.bootstrap import BootstrapTester

            n_rows = self._matrix_data.n_rows
            n_cols = self._matrix_data.n_cols

            rng = np.random.default_rng(12345)

            # Pomeron-like flux: f(x_P) ~ x_P^{-1.1}
            xp_grid = 10**np.linspace(-3, -1.5, n_cols)
            f_xP = xp_grid**(-1.1)

            # Structure-like: g(Q2, beta) random positive
            g = 0.5 + 0.5 * rng.random(n_rows)

            M_true = np.outer(g, f_xP)

            rows, cols, _, errors = self._matrix_data.to_vectors()
            values = M_true[rows, cols] + errors * rng.standard_normal(len(rows))
            values = np.abs(values)  # Keep positive

            tester = BootstrapTester(n_bootstrap=100, seed=12345, use_parallel=False)
            result = tester.test(rows, cols, values, errors, n_rows, n_cols)

            passed = result.p_value > 0.05

            return {
                "name": "synthetic_injection",
                "passed": passed,
                "message": f"Synthetic Regge: p-value = {result.p_value:.3f}",
                "details": {"p_value": result.p_value},
            }

        except Exception as e:
            return {
                "name": "synthetic_injection",
                "passed": False,
                "message": f"Test failed: {e}",
                "details": {},
            }

    def generate_figures(self) -> None:
        """Generate diffractive-specific figures."""
        fig_dir = self.output_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)

        if self._result is None or self._matrix_data is None:
            return

        self._plot_residual_heatmap(fig_dir)
        self._plot_bootstrap_distribution(fig_dir)
        self._plot_xp_dependence(fig_dir)
        self._plot_flux_structure(fig_dir)

    def _plot_residual_heatmap(self, fig_dir: Path) -> None:
        """Plot residual heatmap."""
        values, errors, mask = self._matrix_data.to_matrix()
        M_hat = self._result.matrix_rank1

        residuals = (values - M_hat) / np.where(errors > 0, errors, 1)
        residuals = np.where(mask > 0, residuals, np.nan)

        fig, ax = plt.subplots(figsize=(10, 8))

        im = ax.imshow(residuals, cmap="RdBu_r", vmin=-3, vmax=3, aspect="auto")
        plt.colorbar(im, ax=ax, label="Pull (σ)")

        ax.set_xlabel("x_P bin")
        ax.set_ylabel("(Q², β) bin")
        ax.set_title("Regge Factorization Residuals")

        plt.tight_layout()
        for fmt in ["png", "pdf"]:
            fig.savefig(fig_dir / f"residual_heatmap.{fmt}", dpi=150)
        plt.close(fig)

    def _plot_bootstrap_distribution(self, fig_dir: Path) -> None:
        """Plot bootstrap null distribution."""
        if self._result.bootstrap_result is None:
            return

        br = self._result.bootstrap_result

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.hist(br.lambda_null, bins=50, density=True, alpha=0.7, label="Null distribution")
        ax.axvline(br.lambda_obs, color="red", linestyle="--", linewidth=2,
                   label=f"Observed (Λ = {br.lambda_obs:.2f})")

        ax.set_xlabel("Λ = χ²(rank-1) - χ²(rank-2)")
        ax.set_ylabel("Density")
        ax.set_title(f"Regge Factorization Test (p = {br.p_value:.4f})")
        ax.legend()

        plt.tight_layout()
        for fmt in ["png", "pdf"]:
            fig.savefig(fig_dir / f"bootstrap_distribution.{fmt}", dpi=150)
        plt.close(fig)

    def _plot_xp_dependence(self, fig_dir: Path) -> None:
        """Plot cross section vs x_P for selected (Q², β) bins."""
        values, errors, mask = self._matrix_data.to_matrix()

        # Parse x_P values from column labels
        xp_values = []
        for label in self._matrix_data.col_labels:
            try:
                xp = float(label.split("=")[1])
                xp_values.append(xp)
            except (IndexError, ValueError):
                xp_values.append(np.nan)
        xp_values = np.array(xp_values)

        # Select a few representative rows
        n_rows = min(4, values.shape[0])
        row_indices = np.linspace(0, values.shape[0] - 1, n_rows).astype(int)

        fig, ax = plt.subplots(figsize=(8, 6))

        for i in row_indices:
            valid = ~np.isnan(values[i, :]) & (xp_values > 0)
            if np.sum(valid) < 2:
                continue

            ax.errorbar(
                xp_values[valid], values[i, valid],
                yerr=errors[i, valid],
                marker="o", capsize=2, alpha=0.7,
                label=self._matrix_data.row_labels[i][:20]
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("x_P")
        ax.set_ylabel("σ_r^D")
        ax.set_title("Diffractive Cross Section vs x_P")
        ax.legend(loc="best", fontsize=8)

        plt.tight_layout()
        for fmt in ["png", "pdf"]:
            fig.savefig(fig_dir / f"xp_dependence.{fmt}", dpi=150)
        plt.close(fig)

    def _plot_flux_structure(self, fig_dir: Path) -> None:
        """Plot extracted flux and structure factors."""
        if self._result.fit_rank1 is None:
            return

        model = self._result.fit_rank1.model

        if not hasattr(model, "u") or not hasattr(model, "v"):
            return

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Structure factor (row factors)
        ax = axes[0]
        ax.bar(range(len(model.u)), model.u, alpha=0.7)
        ax.set_xlabel("(Q², β) bin index")
        ax.set_ylabel("σ_r(β, Q²) factor")
        ax.set_title("Extracted Structure Factor")

        # Flux factor (column factors)
        ax = axes[1]

        xp_values = []
        for label in self._matrix_data.col_labels:
            try:
                xp = float(label.split("=")[1])
                xp_values.append(xp)
            except (IndexError, ValueError):
                xp_values.append(np.nan)
        xp_values = np.array(xp_values)

        valid = xp_values > 0
        if np.sum(valid) > 0:
            ax.loglog(xp_values[valid], np.abs(model.v[valid]), "o-")

            # Add reference power law
            xp_ref = np.logspace(np.log10(xp_values[valid].min()),
                                 np.log10(xp_values[valid].max()), 50)
            ax.loglog(xp_ref, 0.1 * xp_ref**(-1.1), "--", alpha=0.5,
                      label="x_P^{-1.1} reference")

        ax.set_xlabel("x_P")
        ax.set_ylabel("|f_P(x_P)| factor")
        ax.set_title("Extracted Pomeron Flux")
        ax.legend()

        plt.tight_layout()
        for fmt in ["png", "pdf"]:
            fig.savefig(fig_dir / f"flux_structure.{fmt}", dpi=150)
        plt.close(fig)
