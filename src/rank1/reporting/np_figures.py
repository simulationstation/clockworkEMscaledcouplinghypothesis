"""
Figure generation for NP (New Physics Sensitive) analysis.

Generates diagnostic plots including:
1. Residual heatmaps under rank-1
2. v2 shape plots (column/observable dependence)
3. u2 dependence plots (row/condition dependence)
4. Localization metric panels
5. Sweep summary plots
6. Replication similarity matrices
"""

from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle

from rank1.logging import get_logger
from rank1.analysis.np_analysis import NPResult
from rank1.analysis.residual_mode import ResidualMode, ResidualMap, LocalizationMetrics
from rank1.analysis.sweeps import SweepResult, GlobalSignificance
from rank1.analysis.replication import ReplicationReport

logger = get_logger()


# Style settings
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 13,
})


class NPFigureGenerator:
    """Generate all figures for NP analysis."""

    def __init__(self, output_dir: Path):
        """
        Initialize figure generator.

        Args:
            output_dir: Directory for figure output
        """
        self.output_dir = Path(output_dir)
        self.figures_dir = self.output_dir / "figures"
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    def generate_all_figures(self, result: NPResult) -> List[Path]:
        """
        Generate all figures for an NP result.

        Args:
            result: NPResult object

        Returns:
            List of paths to generated figures
        """
        paths = []

        # Residual heatmap
        if result.residual_map is not None:
            path = self.plot_residual_heatmap(result)
            if path:
                paths.append(path)

        # v2 shape
        if result.residual_mode is not None:
            path = self.plot_v2_shape(result)
            if path:
                paths.append(path)

        # u2 dependence
        if result.residual_mode is not None:
            path = self.plot_u2_dependence(result)
            if path:
                paths.append(path)

        # Localization panel
        if result.localization_metrics is not None:
            path = self.plot_localization_panel(result)
            if path:
                paths.append(path)

        # Sweep summary
        if result.sweep_results and len(result.sweep_results) > 1:
            path = self.plot_sweep_summary(result)
            if path:
                paths.append(path)

        # Replication similarity
        if result.replication_report and result.replication_report.comparisons:
            path = self.plot_replication_similarity(result)
            if path:
                paths.append(path)

        # Chi² contributions
        if result.residual_mode is not None and result.residual_mode.chi2_contributions is not None:
            path = self.plot_chi2_contributions(result)
            if path:
                paths.append(path)

        logger.info(f"Generated {len(paths)} figures for {result.dataset}")
        return paths

    def plot_residual_heatmap(self, result: NPResult) -> Optional[Path]:
        """
        Plot residual heatmap under rank-1.

        Shows standardized residuals R_ij = (M_ij - Mhat1_ij) / σ_ij
        """
        rmap = result.residual_map
        if rmap is None:
            return None

        # Convert to dense matrix
        matrix, mask = rmap.to_dense_matrix("rank1")
        n_rows, n_cols = matrix.shape

        # Simple ordering by row/column index (could add clustering later)
        fig, ax = plt.subplots(figsize=(10, 8))

        # Mask NaN values
        masked_matrix = np.ma.array(matrix, mask=~mask)

        # Symmetric colormap around 0
        vmax = np.nanmax(np.abs(masked_matrix))
        vmax = max(vmax, 2.0)  # At least ±2 sigma

        im = ax.imshow(
            masked_matrix,
            cmap='RdBu_r',
            vmin=-vmax,
            vmax=vmax,
            aspect='auto',
        )

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, label='Standardized residual (σ)')

        # Labels
        if rmap.row_labels and len(rmap.row_labels) <= 20:
            ax.set_yticks(range(n_rows))
            ax.set_yticklabels(rmap.row_labels)
        if rmap.col_labels and len(rmap.col_labels) <= 20:
            ax.set_xticks(range(n_cols))
            ax.set_xticklabels(rmap.col_labels, rotation=45, ha='right')

        ax.set_xlabel('Column (observable)')
        ax.set_ylabel('Row (condition)')
        ax.set_title(f'{result.dataset}: Rank-1 Residuals')

        # Mark cells with |R| > 2
        for i in range(n_rows):
            for j in range(n_cols):
                if mask[i, j] and abs(matrix[i, j]) > 2:
                    ax.add_patch(Rectangle(
                        (j - 0.5, i - 0.5), 1, 1,
                        fill=False, edgecolor='black', linewidth=1.5
                    ))

        plt.tight_layout()

        path = self.figures_dir / f"{result.dataset}_residual_heatmap.png"
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()

        return path

    def plot_v2_shape(self, result: NPResult) -> Optional[Path]:
        """
        Plot v2 shape (column/observable dependence).

        Different visualization based on dataset type:
        - Higgs: bar plot by decay channel
        - Elastic: line plot vs t
        - Diffractive: line plot vs xP
        """
        rm = result.residual_mode
        if rm is None:
            return None

        v2 = rm.v2
        n_cols = len(v2)

        fig, ax = plt.subplots(figsize=(10, 5))

        if result.dataset == "higgs" and rm.col_labels:
            # Bar plot for categorical
            x = np.arange(n_cols)
            colors = ['green' if v > 0 else 'red' for v in v2]
            ax.bar(x, v2, color=colors, alpha=0.7, edgecolor='black')
            ax.set_xticks(x)
            ax.set_xticklabels(rm.col_labels, rotation=45, ha='right')
            ax.axhline(0, color='black', linewidth=0.5)
            ax.set_xlabel('Decay Channel')

        else:
            # Line plot for ordered bins
            if rm.col_labels:
                try:
                    # Try to parse numeric labels
                    x = np.array([float(l.split('=')[-1]) for l in rm.col_labels])
                except:
                    x = np.arange(n_cols)
            else:
                x = np.arange(n_cols)

            ax.plot(x, v2, 'b-', linewidth=2, marker='o', markersize=4)
            ax.fill_between(x, 0, v2, alpha=0.3, color='blue')
            ax.axhline(0, color='black', linewidth=0.5)

            # Highlight peak region
            if result.localization_metrics:
                peak_idx = result.localization_metrics.peak_index
                ax.axvline(x[peak_idx], color='red', linestyle='--', alpha=0.7, label='Peak')
                ax.legend()

            if result.dataset == "elastic":
                ax.set_xlabel('|t| (GeV²)')
            elif result.dataset == "diffractive":
                ax.set_xlabel('xP')
            else:
                ax.set_xlabel('Column index')

        ax.set_ylabel('v2 (normalized)')
        ax.set_title(f'{result.dataset}: Residual Mode v2 (Column Dependence)')

        # Add localization info
        if result.localization_metrics:
            loc = result.localization_metrics
            textstr = f'Gini={loc.gini:.2f}\nEntropy={loc.normalized_entropy:.2f}'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
                    verticalalignment='top', bbox=props)

        plt.tight_layout()

        path = self.figures_dir / f"{result.dataset}_v2_shape.png"
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()

        return path

    def plot_u2_dependence(self, result: NPResult) -> Optional[Path]:
        """
        Plot u2 dependence (row/condition dependence).
        """
        rm = result.residual_mode
        if rm is None:
            return None

        u2 = rm.u2
        n_rows = len(u2)

        fig, ax = plt.subplots(figsize=(10, 5))

        if rm.row_labels and n_rows <= 10:
            # Bar plot for small number of rows
            x = np.arange(n_rows)
            colors = ['green' if u > 0 else 'red' for u in u2]
            ax.barh(x, u2, color=colors, alpha=0.7, edgecolor='black')
            ax.set_yticks(x)
            ax.set_yticklabels(rm.row_labels)
            ax.axvline(0, color='black', linewidth=0.5)
            ax.set_xlabel('u2 (scale factor)')
            ax.set_ylabel('Row (condition)')

        else:
            # Line/scatter for many rows
            x = np.arange(n_rows)
            ax.bar(x, u2, color='blue', alpha=0.7, width=0.8)
            ax.axhline(0, color='black', linewidth=0.5)
            ax.set_xlabel('Row index')
            ax.set_ylabel('u2 (scale factor)')

            if rm.row_labels and n_rows <= 30:
                ax.set_xticks(x[::max(1, n_rows // 10)])
                ax.set_xticklabels([rm.row_labels[i] for i in range(0, n_rows, max(1, n_rows // 10))],
                                   rotation=45, ha='right')

        ax.set_title(f'{result.dataset}: Residual Mode u2 (Row Dependence)')

        plt.tight_layout()

        path = self.figures_dir / f"{result.dataset}_u2_dependence.png"
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()

        return path

    def plot_localization_panel(self, result: NPResult) -> Optional[Path]:
        """
        Plot localization metrics panel.
        """
        loc = result.localization_metrics
        if loc is None:
            return None

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        # Panel 1: Top-k mass
        ax1 = axes[0]
        if loc.top_k_mass:
            k_vals = sorted(loc.top_k_mass.keys())
            masses = [loc.top_k_mass[k] for k in k_vals]
            ax1.bar(range(len(k_vals)), masses, color='steelblue', alpha=0.7)
            ax1.set_xticks(range(len(k_vals)))
            ax1.set_xticklabels([str(k) for k in k_vals])
            ax1.set_xlabel('k (top elements)')
            ax1.set_ylabel('Fraction of ||v2||²')
            ax1.set_title('Top-k Mass')
            ax1.set_ylim(0, 1)
            ax1.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='50%')
            ax1.legend()

        # Panel 2: Window concentration
        ax2 = axes[1]
        if loc.window_concentration:
            w_vals = sorted(loc.window_concentration.keys())
            concs = [loc.window_concentration[w] for w in w_vals]
            ax2.plot(w_vals, concs, 'o-', color='forestgreen', linewidth=2)
            ax2.set_xlabel('Window size')
            ax2.set_ylabel('Max fraction in window')
            ax2.set_title('Window Concentration')
            ax2.set_ylim(0, 1)
            ax2.axhline(0.5, color='red', linestyle='--', alpha=0.5)

        # Panel 3: Summary metrics
        ax3 = axes[2]
        metrics = ['Gini', 'Entropy\n(norm)', '1 - Entropy']
        values = [loc.gini, loc.normalized_entropy, 1 - loc.normalized_entropy]
        colors = ['steelblue', 'coral', 'forestgreen']
        ax3.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black')
        ax3.set_ylim(0, 1)
        ax3.set_ylabel('Value')
        ax3.set_title('Localization Summary')

        # Add interpretation
        ax3.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
        ax3.text(0.5, 0.52, 'threshold', fontsize=8, color='gray')

        fig.suptitle(f'{result.dataset}: Localization Metrics', fontsize=13)
        plt.tight_layout()

        path = self.figures_dir / f"{result.dataset}_localization.png"
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()

        return path

    def plot_sweep_summary(self, result: NPResult) -> Optional[Path]:
        """
        Plot sweep summary: p_local (or Λ) per preset.
        """
        sweeps = result.sweep_results
        if not sweeps or len(sweeps) < 2:
            return None

        # Sort by Lambda (descending)
        sorted_sweeps = sorted(sweeps, key=lambda s: s.lambda_stat, reverse=True)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Panel 1: Lambda by preset
        ax1 = axes[0]
        names = [s.preset.name for s in sorted_sweeps]
        lambdas = [s.lambda_stat for s in sorted_sweeps]

        colors = ['red' if s.p_local < 0.05 else 'steelblue' for s in sorted_sweeps]
        ax1.barh(range(len(names)), lambdas, color=colors, alpha=0.7)
        ax1.set_yticks(range(len(names)))
        ax1.set_yticklabels(names)
        ax1.set_xlabel('Λ')
        ax1.set_title('Test Statistic by Preset')

        # Mark best
        ax1.barh(0, lambdas[0], color='gold', alpha=0.8, edgecolor='black', linewidth=2)

        # Panel 2: p_local by preset
        ax2 = axes[1]
        p_locals = [s.p_local for s in sorted_sweeps]

        ax2.barh(range(len(names)), p_locals, color=colors, alpha=0.7)
        ax2.set_yticks(range(len(names)))
        ax2.set_yticklabels(names)
        ax2.set_xlabel('p_local')
        ax2.set_title('Local P-Value by Preset')
        ax2.axvline(0.05, color='red', linestyle='--', label='α=0.05')
        ax2.legend()

        # Add global p-value annotation
        if result.global_significance:
            gs = result.global_significance
            ax2.text(0.98, 0.02, f'p_global = {gs.p_global:.4f}',
                     transform=ax2.transAxes, fontsize=10, ha='right',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        fig.suptitle(f'{result.dataset}: Sweep Summary', fontsize=13)
        plt.tight_layout()

        path = self.figures_dir / f"{result.dataset}_sweep_summary.png"
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()

        return path

    def plot_replication_similarity(self, result: NPResult) -> Optional[Path]:
        """
        Plot replication similarity matrix.
        """
        repl = result.replication_report
        if repl is None or not repl.comparisons:
            return None

        # Build similarity matrix
        sources = set()
        for c in repl.comparisons:
            sources.add(c.source_a)
            sources.add(c.source_b)
        sources = sorted(sources)
        n = len(sources)

        if n < 2:
            return None

        # Create matrix
        sim_matrix = np.eye(n)
        source_to_idx = {s: i for i, s in enumerate(sources)}

        for c in repl.comparisons:
            i = source_to_idx.get(c.source_a)
            j = source_to_idx.get(c.source_b)
            if i is not None and j is not None:
                sim_matrix[i, j] = c.v2_cosine
                sim_matrix[j, i] = c.v2_cosine

        fig, ax = plt.subplots(figsize=(8, 7))

        im = ax.imshow(sim_matrix, cmap='RdYlGn', vmin=-1, vmax=1)

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, label='v2 Cosine Similarity')

        # Labels
        ax.set_xticks(range(n))
        ax.set_xticklabels(sources, rotation=45, ha='right')
        ax.set_yticks(range(n))
        ax.set_yticklabels(sources)

        # Annotate cells
        for i in range(n):
            for j in range(n):
                val = sim_matrix[i, j]
                color = 'white' if abs(val) > 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=9)

        ax.set_title(f'{result.dataset}: Replication Similarity')

        plt.tight_layout()

        path = self.figures_dir / f"{result.dataset}_replication.png"
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()

        return path

    def plot_chi2_contributions(self, result: NPResult) -> Optional[Path]:
        """
        Plot chi² contributions per cell showing which cells drive the deviation.
        """
        rm = result.residual_mode
        if rm is None or rm.chi2_contributions is None:
            return None

        rmap = result.residual_map
        if rmap is None:
            return None

        # Build contribution matrix
        n_rows = rmap.row_indices.max() + 1
        n_cols = rmap.col_indices.max() + 1

        contrib_matrix = np.full((n_rows, n_cols), np.nan)
        for i, (r, c, v) in enumerate(zip(rmap.row_indices, rmap.col_indices, rm.chi2_contributions)):
            contrib_matrix[r, c] = v

        fig, ax = plt.subplots(figsize=(10, 8))

        # Use diverging colormap
        vmax = np.nanmax(np.abs(contrib_matrix))
        vmax = max(vmax, 1.0)

        im = ax.imshow(
            contrib_matrix,
            cmap='RdBu_r',
            vmin=-vmax,
            vmax=vmax,
            aspect='auto',
        )

        cbar = plt.colorbar(im, ax=ax, label='χ² contribution (rank-1 - rank-2)')

        # Labels
        if rmap.row_labels and len(rmap.row_labels) <= 20:
            ax.set_yticks(range(n_rows))
            ax.set_yticklabels(rmap.row_labels)
        if rmap.col_labels and len(rmap.col_labels) <= 20:
            ax.set_xticks(range(n_cols))
            ax.set_xticklabels(rmap.col_labels, rotation=45, ha='right')

        ax.set_xlabel('Column (observable)')
        ax.set_ylabel('Row (condition)')
        ax.set_title(f'{result.dataset}: χ² Contribution per Cell')

        plt.tight_layout()

        path = self.figures_dir / f"{result.dataset}_chi2_contributions.png"
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()

        return path

    def save_figure(self, fig, name: str, dpi: int = 150) -> Path:
        """Save a figure to the figures directory."""
        path = self.figures_dir / f"{name}.png"
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        return path
