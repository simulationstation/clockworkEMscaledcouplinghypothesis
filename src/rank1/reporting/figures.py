"""
Figure generation utilities for rank-1 analysis reports.
"""

from pathlib import Path
from typing import Optional, Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from rank1.analysis.base import AnalysisResult
from rank1.logging import get_logger

logger = get_logger()


class FigureGenerator:
    """Generate publication-quality figures from analysis results."""

    def __init__(
        self,
        output_dir: Path,
        formats: list[str] = ["png", "pdf"],
        dpi: int = 150,
    ):
        self.output_dir = Path(output_dir)
        self.formats = formats
        self.dpi = dpi

        # Set up matplotlib style
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.rcParams.update({
            "font.size": 10,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "figure.figsize": (8, 5),
        })

    def save_figure(self, fig: Figure, name: str) -> list[Path]:
        """Save figure in multiple formats."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        paths = []

        for fmt in self.formats:
            path = self.output_dir / f"{name}.{fmt}"
            fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
            paths.append(path)

        plt.close(fig)
        return paths

    def plot_residual_heatmap(
        self,
        values: np.ndarray,
        errors: np.ndarray,
        predicted: np.ndarray,
        mask: np.ndarray,
        row_labels: list[str],
        col_labels: list[str],
        title: str = "Residual Heatmap",
    ) -> Figure:
        """Create a residual heatmap figure."""
        residuals = (values - predicted) / np.where(errors > 0, errors, 1)
        residuals = np.where(mask > 0, residuals, np.nan)

        fig, ax = plt.subplots(figsize=(10, 6))

        im = ax.imshow(residuals, cmap="RdBu_r", vmin=-3, vmax=3, aspect="auto")
        plt.colorbar(im, ax=ax, label="Pull (σ)")

        if len(col_labels) <= 10:
            ax.set_xticks(range(len(col_labels)))
            ax.set_xticklabels(col_labels, rotation=45, ha="right")
        if len(row_labels) <= 15:
            ax.set_yticks(range(len(row_labels)))
            ax.set_yticklabels(row_labels)

        ax.set_title(title)
        plt.tight_layout()

        return fig

    def plot_bootstrap_distribution(
        self,
        lambda_null: np.ndarray,
        lambda_obs: float,
        p_value: float,
        title: str = "Bootstrap Distribution",
    ) -> Figure:
        """Create bootstrap distribution figure."""
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.hist(lambda_null, bins=50, density=True, alpha=0.7,
                color="steelblue", label="Null distribution")
        ax.axvline(lambda_obs, color="red", linestyle="--", linewidth=2,
                   label=f"Observed (Λ = {lambda_obs:.2f})")

        ax.set_xlabel("Λ = χ²(rank-1) - χ²(rank-2)")
        ax.set_ylabel("Density")
        ax.set_title(f"{title} (p = {p_value:.4f})")
        ax.legend()

        plt.tight_layout()
        return fig

    def plot_comparison_bar(
        self,
        results: list[AnalysisResult],
        metric: str = "p_value",
        title: str = "Comparison",
    ) -> Figure:
        """Create bar chart comparing multiple analyses."""
        fig, ax = plt.subplots(figsize=(8, 5))

        names = [r.dataset_name for r in results]
        values = [getattr(r, metric) for r in results]

        colors = ["green" if v > 0.05 else "red" for v in values]

        bars = ax.bar(names, values, color=colors, alpha=0.7)

        if metric == "p_value":
            ax.axhline(0.05, color="black", linestyle="--", alpha=0.5,
                       label="α = 0.05")
            ax.set_ylabel("p-value")
        else:
            ax.set_ylabel(metric)

        ax.set_title(title)
        ax.legend()

        plt.tight_layout()
        return fig

    def create_summary_figure(
        self,
        results: list[AnalysisResult],
    ) -> Figure:
        """Create a multi-panel summary figure."""
        n = len(results)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))

        if n == 1:
            axes = [axes]

        for ax, result in zip(axes, results):
            # Plot chi2/ndof comparison
            labels = ["Rank-1", "Rank-2"]
            chi2_ndof = [
                result.chi2_rank1 / max(1, result.ndof_rank1),
                result.chi2_rank2 / max(1, result.ndof_rank2),
            ]

            colors = ["steelblue", "coral"]
            ax.bar(labels, chi2_ndof, color=colors, alpha=0.7)
            ax.axhline(1, color="black", linestyle="--", alpha=0.5)

            ax.set_ylabel("χ²/ndof")
            ax.set_title(f"{result.dataset_name}\np = {result.p_value:.4f}")

        plt.tight_layout()
        return fig
