"""
Elastic pp scattering dσ/dt shape rank-1 analysis.

Tests whether dσ/dt(√s, t) ≈ A(√s) × f(t) (separable shape).
"""

from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

from rank1.analysis.base import BaseRankAnalysis, AnalysisResult
from rank1.datasets.elastic_totem import ElasticTOTEMDataset
from rank1.logging import get_logger

logger = get_logger()


class ElasticRankAnalysis(BaseRankAnalysis):
    """
    Rank-1 analysis for TOTEM elastic scattering shapes.

    Tests whether the dσ/dt shape is universal across √s values,
    with only the overall normalization varying.
    """

    name = "elastic_totem"
    description = "TOTEM elastic pp dσ/dt shape rank-1 test"

    def __init__(
        self,
        dataset: Optional[ElasticTOTEMDataset] = None,
        **kwargs,
    ):
        if dataset is None:
            dataset = ElasticTOTEMDataset()

        super().__init__(dataset=dataset, **kwargs)

    def additional_cross_checks(self) -> list[dict]:
        """Elastic scattering specific cross-checks."""
        checks = []

        # Check 1: Forward slope consistency
        slope_check = self._check_forward_slopes()
        if slope_check:
            checks.append(slope_check)

        # Check 2: Energy ordering
        order_check = self._check_energy_ordering()
        if order_check:
            checks.append(order_check)

        # Check 3: Synthetic injection test
        synth_result = self._synthetic_injection_test()
        checks.append(synth_result)

        return checks

    def _check_forward_slopes(self) -> Optional[dict]:
        """Check extracted forward slopes against published values."""
        if not hasattr(self.dataset, "_raw_data"):
            return None

        expected_slopes = {
            7.0: (19.9, 0.5),   # (value, tolerance)
            8.0: (19.9, 0.5),
            13.0: (20.4, 0.5),
        }

        slopes = {}
        for energy, df in self.dataset._raw_data.items():
            B = self.dataset._fit_forward_slope(df)
            if B is not None:
                slopes[energy] = B

        all_ok = True
        messages = []

        for energy, (expected, tol) in expected_slopes.items():
            if energy in slopes:
                actual = slopes[energy]
                ok = abs(actual - expected) < 2 * tol
                all_ok = all_ok and ok
                messages.append(f"{int(energy)} TeV: B = {actual:.1f} (expected {expected}±{tol})")

        return {
            "name": "forward_slope_consistency",
            "passed": all_ok,
            "message": "; ".join(messages),
            "details": {"slopes": slopes},
        }

    def _check_energy_ordering(self) -> Optional[dict]:
        """Check that cross sections decrease with |t| at all energies."""
        if self._matrix_data is None:
            return None

        values, _, mask = self._matrix_data.to_matrix()

        # Check each row (energy) has decreasing trend
        all_decreasing = True
        for i in range(values.shape[0]):
            row = values[i, :]
            valid = ~np.isnan(row)
            if np.sum(valid) > 2:
                # Check if mostly decreasing
                diffs = np.diff(row[valid])
                frac_decreasing = np.mean(diffs < 0)
                if frac_decreasing < 0.7:
                    all_decreasing = False

        return {
            "name": "energy_ordering",
            "passed": all_decreasing,
            "message": "Cross sections decrease with |t|" if all_decreasing else "Unexpected increase in some |t| ranges",
            "details": {},
        }

    def _synthetic_injection_test(self) -> dict:
        """Inject synthetic separable data and verify recovery."""
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

            # Create synthetic separable shape
            rng = np.random.default_rng(12345)

            # A(E) - energy-dependent normalization
            A = np.array([1.0, 1.05, 1.15])[:n_rows]  # Increasing with energy

            # f(t) - universal shape (exponential-like)
            t_grid = np.linspace(0.04, 0.20, n_cols)
            f = np.exp(-15 * t_grid)

            M_true = np.outer(A, f)

            rows, cols, _, errors = self._matrix_data.to_vectors()
            values = M_true[rows, cols] + errors * rng.standard_normal(len(rows))

            tester = BootstrapTester(n_bootstrap=100, seed=12345, use_parallel=False)
            result = tester.test(rows, cols, values, errors, n_rows, n_cols)

            passed = result.p_value > 0.05

            return {
                "name": "synthetic_injection",
                "passed": passed,
                "message": f"Synthetic separable: p-value = {result.p_value:.3f}",
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
        """Generate elastic scattering specific figures."""
        fig_dir = self.output_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)

        if self._result is None or self._matrix_data is None:
            return

        self._plot_residual_heatmap(fig_dir)
        self._plot_bootstrap_distribution(fig_dir)
        self._plot_shape_comparison(fig_dir)
        self._plot_ratio_vs_t(fig_dir)

    def _plot_residual_heatmap(self, fig_dir: Path) -> None:
        """Plot residual heatmap."""
        values, errors, mask = self._matrix_data.to_matrix()
        M_hat = self._result.matrix_rank1

        residuals = (values - M_hat) / np.where(errors > 0, errors, 1)
        residuals = np.where(mask > 0, residuals, np.nan)

        fig, ax = plt.subplots(figsize=(12, 4))

        im = ax.imshow(residuals, cmap="RdBu_r", vmin=-3, vmax=3, aspect="auto")
        plt.colorbar(im, ax=ax, label="Pull (σ)")

        ax.set_xlabel("|t| bin")
        ax.set_yticks(range(len(self._matrix_data.row_labels)))
        ax.set_yticklabels(self._matrix_data.row_labels)
        ax.set_title("Rank-1 Fit Residuals")

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
        ax.set_title(f"Shape Separability Test (p = {br.p_value:.4f})")
        ax.legend()

        plt.tight_layout()
        for fmt in ["png", "pdf"]:
            fig.savefig(fig_dir / f"bootstrap_distribution.{fmt}", dpi=150)
        plt.close(fig)

    def _plot_shape_comparison(self, fig_dir: Path) -> None:
        """Plot dσ/dt shapes at different energies overlaid."""
        values, errors, mask = self._matrix_data.to_matrix()

        # Get t-grid from metadata
        t_grid = self._matrix_data.metadata.get("t_grid")
        if t_grid is None:
            t_grid = np.arange(values.shape[1])

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Raw shapes
        ax = axes[0]
        for i, label in enumerate(self._matrix_data.row_labels):
            valid = ~np.isnan(values[i, :]) & np.isfinite(errors[i, :]) & (errors[i, :] >= 0)
            ax.errorbar(
                np.array(t_grid)[valid], values[i, valid],
                yerr=np.abs(errors[i, valid]),
                label=label, marker="o", capsize=2, alpha=0.8
            )

        ax.set_xlabel("|t| (GeV²)")
        ax.set_ylabel("dσ/dt (normalized)")
        ax.set_title("Raw Shapes")
        ax.legend()

        # Normalized shapes (divide by rank-1 fit row factor)
        ax = axes[1]
        fit = self._result.fit_rank1
        if fit and hasattr(fit.model, "u"):
            u = fit.model.u
            for i, label in enumerate(self._matrix_data.row_labels):
                valid = ~np.isnan(values[i, :])
                normalized = values[i, :] / u[i]
                norm_err = np.abs(errors[i, :] / u[i])  # Ensure positive errors
                # Also filter out invalid errors
                valid = valid & np.isfinite(norm_err)

                ax.errorbar(
                    np.array(t_grid)[valid], normalized[valid],
                    yerr=norm_err[valid],
                    label=label, marker="o", capsize=2, alpha=0.8
                )

        ax.set_xlabel("|t| (GeV²)")
        ax.set_ylabel("dσ/dt / A(√s)")
        ax.set_title("Normalized Shapes (Rank-1 predicts overlap)")
        ax.legend()

        plt.tight_layout()
        for fmt in ["png", "pdf"]:
            fig.savefig(fig_dir / f"shape_comparison.{fmt}", dpi=150)
        plt.close(fig)

    def _plot_ratio_vs_t(self, fig_dir: Path) -> None:
        """Plot ratio of shapes between energies vs |t|."""
        values, errors, mask = self._matrix_data.to_matrix()

        t_grid = self._matrix_data.metadata.get("t_grid")
        if t_grid is None:
            t_grid = np.arange(values.shape[1])

        n_rows = values.shape[0]
        if n_rows < 2:
            return

        fig, axes = plt.subplots(1, n_rows - 1, figsize=(5 * (n_rows - 1), 4))
        if n_rows == 2:
            axes = [axes]

        for i in range(1, n_rows):
            ax = axes[i - 1]

            ratio = values[i, :] / np.where(values[0, :] > 0, values[0, :], np.nan)
            ratio_err = ratio * np.sqrt(
                (errors[i, :] / np.where(values[i, :] > 0, values[i, :], 1))**2 +
                (errors[0, :] / np.where(values[0, :] > 0, values[0, :], 1))**2
            )

            valid = np.isfinite(ratio) & np.isfinite(ratio_err) & (ratio_err >= 0)

            ax.errorbar(
                np.array(t_grid)[valid], ratio[valid],
                yerr=np.abs(ratio_err[valid]),
                fmt="o", capsize=3
            )

            # Rank-1 predicts flat ratio
            mean_ratio = np.nanmean(ratio[valid])
            ax.axhline(mean_ratio, color="red", linestyle="--",
                       label=f"Mean = {mean_ratio:.3f}")

            ax.set_xlabel("|t| (GeV²)")
            ax.set_ylabel(f"{self._matrix_data.row_labels[i]} / {self._matrix_data.row_labels[0]}")
            ax.legend()

        fig.suptitle("Ratio vs |t| (Rank-1 predicts flat)", y=1.02)
        plt.tight_layout()

        for fmt in ["png", "pdf"]:
            fig.savefig(fig_dir / f"ratio_vs_t.{fmt}", dpi=150)
        plt.close(fig)
