"""
Higgs production × decay signal strength rank-1 analysis.

Tests whether μ_{prod,decay} ≈ μ_prod × μ_decay (rank-1 separability).
"""

from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

from rank1.analysis.base import BaseRankAnalysis, AnalysisResult
from rank1.datasets.higgs_atlas_mu import HiggsATLASDataset
from rank1.logging import get_logger

logger = get_logger()


class HiggsRankAnalysis(BaseRankAnalysis):
    """
    Rank-1 analysis for ATLAS Higgs signal strengths.

    The SM prediction is that μ_{prod,decay} = μ × κ factors,
    which gives a rank-1 matrix. Deviations would indicate BSM physics
    affecting specific production-decay combinations differently.
    """

    name = "higgs_atlas"
    description = "ATLAS Higgs μ_{prod,decay} rank-1 test"

    def __init__(
        self,
        dataset: Optional[HiggsATLASDataset] = None,
        **kwargs,
    ):
        if dataset is None:
            dataset = HiggsATLASDataset()

        super().__init__(dataset=dataset, **kwargs)

    def additional_cross_checks(self) -> list[dict]:
        """Higgs-specific cross-checks."""
        checks = []

        # Check 1: Mean signal strength near 1
        if self._matrix_data is not None:
            values, _, _ = self._matrix_data.to_matrix()
            mean_mu = np.nanmean(values)

            checks.append({
                "name": "mean_signal_strength",
                "passed": 0.8 < mean_mu < 1.2,
                "message": f"Mean μ = {mean_mu:.3f} (expected ~1 for SM)",
                "details": {"mean_mu": float(mean_mu)},
            })

        # Check 2: Synthetic rank-1 injection test
        synth_result = self._synthetic_injection_test()
        checks.append(synth_result)

        return checks

    def _synthetic_injection_test(self) -> dict:
        """
        Inject synthetic rank-1 data and verify recovery.

        This validates that our analysis correctly identifies rank-1 data.
        """
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

            # Create synthetic rank-1 matrix
            rng = np.random.default_rng(12345)
            u_true = 1.0 + 0.2 * rng.standard_normal(n_rows)
            v_true = 1.0 + 0.2 * rng.standard_normal(n_cols)
            v_true /= np.linalg.norm(v_true)

            M_true = np.outer(u_true, v_true)

            # Use same observation pattern
            rows, cols, _, errors = self._matrix_data.to_vectors()

            # Add noise
            values = M_true[rows, cols] + errors * rng.standard_normal(len(rows))

            # Run quick bootstrap (fewer samples)
            tester = BootstrapTester(n_bootstrap=100, seed=12345, use_parallel=False)
            result = tester.test(rows, cols, values, errors, n_rows, n_cols)

            # Rank-1 injection should NOT be rejected
            passed = result.p_value > 0.05

            return {
                "name": "synthetic_injection",
                "passed": passed,
                "message": f"Synthetic rank-1: p-value = {result.p_value:.3f}",
                "details": {
                    "p_value": result.p_value,
                    "lambda": result.lambda_obs,
                },
            }

        except Exception as e:
            return {
                "name": "synthetic_injection",
                "passed": False,
                "message": f"Test failed: {e}",
                "details": {},
            }

    def generate_figures(self) -> None:
        """Generate Higgs-specific figures."""
        fig_dir = self.output_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)

        if self._result is None or self._matrix_data is None:
            return

        # Figure 1: Residual heatmap
        self._plot_residual_heatmap(fig_dir)

        # Figure 2: Bootstrap distribution
        self._plot_bootstrap_distribution(fig_dir)

        # Figure 3: Ratio invariance plot
        self._plot_ratio_invariance(fig_dir)

        # Figure 4: Singular value spectrum
        self._plot_singular_values(fig_dir)

    def _plot_residual_heatmap(self, fig_dir: Path) -> None:
        """Plot residual heatmap (M - M_hat_rank1) / σ."""
        values, errors, mask = self._matrix_data.to_matrix()
        M_hat = self._result.matrix_rank1

        residuals = (values - M_hat) / np.where(errors > 0, errors, 1)
        residuals = np.where(mask > 0, residuals, np.nan)

        fig, ax = plt.subplots(figsize=(10, 6))

        im = ax.imshow(residuals, cmap="RdBu_r", vmin=-3, vmax=3, aspect="auto")
        plt.colorbar(im, ax=ax, label="Pull (σ)")

        ax.set_xticks(range(len(self._matrix_data.col_labels)))
        ax.set_xticklabels(self._matrix_data.col_labels, rotation=45, ha="right")
        ax.set_yticks(range(len(self._matrix_data.row_labels)))
        ax.set_yticklabels(self._matrix_data.row_labels)

        ax.set_xlabel("Decay Channel")
        ax.set_ylabel("Production Mode")
        ax.set_title("Rank-1 Fit Residuals (Pull)")

        plt.tight_layout()
        for fmt in ["png", "pdf"]:
            fig.savefig(fig_dir / f"residual_heatmap.{fmt}", dpi=150)
        plt.close(fig)

    def _plot_bootstrap_distribution(self, fig_dir: Path) -> None:
        """Plot bootstrap null distribution vs observed."""
        if self._result.bootstrap_result is None:
            return

        br = self._result.bootstrap_result

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.hist(br.lambda_null, bins=50, density=True, alpha=0.7, label="Null distribution")
        ax.axvline(br.lambda_obs, color="red", linestyle="--", linewidth=2,
                   label=f"Observed (Λ = {br.lambda_obs:.2f})")

        ax.set_xlabel("Λ = χ²(rank-1) - χ²(rank-2)")
        ax.set_ylabel("Density")
        ax.set_title(f"Bootstrap Test (p = {br.p_value:.4f})")
        ax.legend()

        plt.tight_layout()
        for fmt in ["png", "pdf"]:
            fig.savefig(fig_dir / f"bootstrap_distribution.{fmt}", dpi=150)
        plt.close(fig)

    def _plot_ratio_invariance(self, fig_dir: Path) -> None:
        """
        Plot ratio of signal strengths between decay channels.

        For rank-1: μ(prod, c1) / μ(prod, c2) should be constant across prod.
        """
        values, errors, mask = self._matrix_data.to_matrix()
        n_cols = values.shape[1]

        if n_cols < 2:
            return

        fig, axes = plt.subplots(1, min(3, n_cols - 1), figsize=(4 * min(3, n_cols - 1), 4))
        if n_cols == 2:
            axes = [axes]

        # Compare each column to the first
        for ax_idx, col_idx in enumerate(range(1, min(4, n_cols))):
            ax = axes[ax_idx] if n_cols > 2 else axes[ax_idx]

            ratio = values[:, col_idx] / np.where(values[:, 0] > 0, values[:, 0], np.nan)

            # Error propagation
            rel_err_0 = errors[:, 0] / np.where(values[:, 0] > 0, values[:, 0], 1)
            rel_err_c = errors[:, col_idx] / np.where(values[:, col_idx] > 0, values[:, col_idx], 1)
            ratio_err = ratio * np.sqrt(rel_err_0**2 + rel_err_c**2)

            valid = np.isfinite(ratio) & np.isfinite(ratio_err)

            x_pos = np.arange(len(self._matrix_data.row_labels))
            ax.errorbar(x_pos[valid], ratio[valid], yerr=ratio_err[valid],
                        fmt="o", capsize=3)

            # Rank-1 prediction: constant ratio
            mean_ratio = np.nanmean(ratio[valid])
            ax.axhline(mean_ratio, color="red", linestyle="--", alpha=0.7,
                       label=f"Mean = {mean_ratio:.2f}")

            ax.set_xticks(x_pos)
            ax.set_xticklabels(self._matrix_data.row_labels, rotation=45, ha="right")
            ax.set_ylabel(f"μ({self._matrix_data.col_labels[col_idx]}) / μ({self._matrix_data.col_labels[0]})")
            ax.legend(loc="best")

        fig.suptitle("Ratio Invariance (Rank-1 predicts constant)", y=1.02)
        plt.tight_layout()

        for fmt in ["png", "pdf"]:
            fig.savefig(fig_dir / f"ratio_invariance.{fmt}", dpi=150)
        plt.close(fig)

    def _plot_singular_values(self, fig_dir: Path) -> None:
        """Plot singular value spectrum of the matrix."""
        values, errors, mask = self._matrix_data.to_matrix()

        # Weight by inverse errors
        weights = 1.0 / np.where(errors > 0, errors, 1)**2
        weights = np.where(mask > 0, weights, 0)

        # Weighted matrix for SVD
        weighted = values * np.sqrt(weights)
        weighted = np.nan_to_num(weighted, nan=0.0)

        from scipy.linalg import svdvals
        s = svdvals(weighted)

        fig, ax = plt.subplots(figsize=(6, 4))

        ax.bar(range(1, len(s) + 1), s / s[0], color="steelblue", alpha=0.8)
        ax.axhline(0.1, color="red", linestyle="--", alpha=0.5, label="10% threshold")

        ax.set_xlabel("Singular Value Index")
        ax.set_ylabel("Relative Magnitude (normalized to σ₁)")
        ax.set_title("Singular Value Spectrum")
        ax.legend()

        plt.tight_layout()
        for fmt in ["png", "pdf"]:
            fig.savefig(fig_dir / f"singular_values.{fmt}", dpi=150)
        plt.close(fig)
