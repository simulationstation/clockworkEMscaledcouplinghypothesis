"""
Replication scoring across independent datasets and conditions.

This module provides:
1. Mode comparison metrics (cosine similarity, Spearman correlation)
2. Grid interpolation for comparing modes on different grids
3. Peak location agreement
4. Replication quality assessment
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
from scipy import stats, interpolate

from rank1.logging import get_logger
from rank1.analysis.residual_mode import ResidualMode, apply_sign_convention

logger = get_logger()


@dataclass
class ReplicationMetrics:
    """Metrics comparing two residual modes."""

    # Source identification
    source_a: str
    source_b: str

    # Alignment info
    sign_aligned: bool = True

    # Vector similarity for v2
    v2_cosine: float = 0.0
    v2_spearman: float = 0.0
    v2_spearman_pvalue: float = 1.0

    # Vector similarity for u2
    u2_cosine: float = 0.0
    u2_spearman: float = 0.0
    u2_spearman_pvalue: float = 1.0

    # Peak agreement
    v2_peak_a: int = 0
    v2_peak_b: int = 0
    v2_peak_match: bool = False
    v2_peak_distance: int = 0  # |peak_a - peak_b|

    # Overall replication quality
    replication_score: float = 0.0
    replication_grade: str = "unknown"  # "excellent", "good", "moderate", "poor", "incompatible"

    # Comparability status
    comparable: bool = True
    comparability_note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_a": self.source_a,
            "source_b": self.source_b,
            "sign_aligned": self.sign_aligned,
            "v2_cosine": self.v2_cosine,
            "v2_spearman": self.v2_spearman,
            "v2_spearman_pvalue": self.v2_spearman_pvalue,
            "u2_cosine": self.u2_cosine,
            "u2_spearman": self.u2_spearman,
            "u2_spearman_pvalue": self.u2_spearman_pvalue,
            "v2_peak_a": self.v2_peak_a,
            "v2_peak_b": self.v2_peak_b,
            "v2_peak_match": self.v2_peak_match,
            "v2_peak_distance": self.v2_peak_distance,
            "replication_score": self.replication_score,
            "replication_grade": self.replication_grade,
            "comparable": self.comparable,
            "comparability_note": self.comparability_note,
        }


def align_signs(v_a: np.ndarray, v_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    Align signs of two vectors for comparison.

    Returns:
        (v_a, v_b_aligned, was_flipped)
    """
    dot_product = np.dot(v_a, v_b)
    if dot_product < 0:
        return v_a, -v_b, True
    return v_a, v_b, False


def cosine_similarity(v_a: np.ndarray, v_b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(v_a)
    norm_b = np.linalg.norm(v_b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return float(np.dot(v_a, v_b) / (norm_a * norm_b))


def spearman_correlation(v_a: np.ndarray, v_b: np.ndarray) -> Tuple[float, float]:
    """
    Compute Spearman rank correlation on absolute values.

    Returns:
        (correlation, p-value)
    """
    if len(v_a) < 3 or len(v_b) < 3:
        return 0.0, 1.0

    # Use absolute values for rank correlation
    corr, pval = stats.spearmanr(np.abs(v_a), np.abs(v_b))
    return float(corr), float(pval)


def interpolate_to_common_grid(
    v_a: np.ndarray,
    grid_a: np.ndarray,
    v_b: np.ndarray,
    grid_b: np.ndarray,
    n_common: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Interpolate two vectors to a common grid.

    Args:
        v_a: First vector
        grid_a: Grid points for v_a
        v_b: Second vector
        grid_b: Grid points for v_b
        n_common: Number of common grid points

    Returns:
        (v_a_interp, v_b_interp, common_grid)
    """
    # Find overlapping range
    min_common = max(grid_a.min(), grid_b.min())
    max_common = min(grid_a.max(), grid_b.max())

    if min_common >= max_common:
        # No overlap
        return np.array([]), np.array([]), np.array([])

    # Common grid
    common_grid = np.linspace(min_common, max_common, n_common)

    # Interpolate both
    interp_a = interpolate.interp1d(grid_a, v_a, kind='linear', fill_value='extrapolate')
    interp_b = interpolate.interp1d(grid_b, v_b, kind='linear', fill_value='extrapolate')

    v_a_interp = interp_a(common_grid)
    v_b_interp = interp_b(common_grid)

    return v_a_interp, v_b_interp, common_grid


class ModeComparator:
    """Compare residual modes across datasets/conditions."""

    def __init__(self):
        pass

    def compare_direct(
        self,
        mode_a: ResidualMode,
        mode_b: ResidualMode,
        source_a: str,
        source_b: str,
    ) -> ReplicationMetrics:
        """
        Directly compare two residual modes (same grid assumed).

        Args:
            mode_a: First residual mode
            mode_b: Second residual mode
            source_a: Source identifier for mode_a
            source_b: Source identifier for mode_b

        Returns:
            ReplicationMetrics
        """
        # Check dimensions match
        if len(mode_a.v2) != len(mode_b.v2):
            return ReplicationMetrics(
                source_a=source_a,
                source_b=source_b,
                comparable=False,
                comparability_note=f"v2 dimensions differ: {len(mode_a.v2)} vs {len(mode_b.v2)}",
                replication_grade="incompatible",
            )

        if len(mode_a.u2) != len(mode_b.u2):
            return ReplicationMetrics(
                source_a=source_a,
                source_b=source_b,
                comparable=False,
                comparability_note=f"u2 dimensions differ: {len(mode_a.u2)} vs {len(mode_b.u2)}",
                replication_grade="incompatible",
            )

        # Align signs
        v2_a, v2_b, v2_flipped = align_signs(mode_a.v2, mode_b.v2)
        u2_a = mode_a.u2
        u2_b = -mode_b.u2 if v2_flipped else mode_b.u2

        # Compute metrics
        v2_cos = cosine_similarity(v2_a, v2_b)
        v2_spear, v2_spear_p = spearman_correlation(v2_a, v2_b)

        u2_cos = cosine_similarity(u2_a, u2_b)
        u2_spear, u2_spear_p = spearman_correlation(u2_a, u2_b)

        # Peak agreement
        peak_a = mode_a.v2_localization.peak_index if mode_a.v2_localization else np.argmax(np.abs(v2_a))
        peak_b = mode_b.v2_localization.peak_index if mode_b.v2_localization else np.argmax(np.abs(v2_b))
        peak_match = peak_a == peak_b
        peak_dist = abs(peak_a - peak_b)

        # Overall score (weighted combination)
        # v2 is typically more important than u2 for residual interpretation
        score = 0.5 * abs(v2_cos) + 0.3 * abs(u2_cos) + 0.2 * (1.0 if peak_match else max(0, 1 - peak_dist / len(v2_a)))

        # Grade
        if score > 0.9:
            grade = "excellent"
        elif score > 0.75:
            grade = "good"
        elif score > 0.5:
            grade = "moderate"
        else:
            grade = "poor"

        return ReplicationMetrics(
            source_a=source_a,
            source_b=source_b,
            sign_aligned=not v2_flipped,
            v2_cosine=v2_cos,
            v2_spearman=v2_spear,
            v2_spearman_pvalue=v2_spear_p,
            u2_cosine=u2_cos,
            u2_spearman=u2_spear,
            u2_spearman_pvalue=u2_spear_p,
            v2_peak_a=peak_a,
            v2_peak_b=peak_b,
            v2_peak_match=peak_match,
            v2_peak_distance=peak_dist,
            replication_score=score,
            replication_grade=grade,
            comparable=True,
        )

    def compare_interpolated(
        self,
        mode_a: ResidualMode,
        grid_a: np.ndarray,
        mode_b: ResidualMode,
        grid_b: np.ndarray,
        source_a: str,
        source_b: str,
        n_common: int = 100,
    ) -> ReplicationMetrics:
        """
        Compare residual modes on different grids via interpolation.

        Args:
            mode_a: First residual mode
            grid_a: Grid points for mode_a.v2
            mode_b: Second residual mode
            grid_b: Grid points for mode_b.v2
            source_a: Source identifier for mode_a
            source_b: Source identifier for mode_b
            n_common: Number of common grid points

        Returns:
            ReplicationMetrics
        """
        # Interpolate v2 to common grid
        v2_a_interp, v2_b_interp, common_grid = interpolate_to_common_grid(
            mode_a.v2, grid_a, mode_b.v2, grid_b, n_common
        )

        if len(common_grid) < 5:
            return ReplicationMetrics(
                source_a=source_a,
                source_b=source_b,
                comparable=False,
                comparability_note="Insufficient grid overlap for interpolation",
                replication_grade="incompatible",
            )

        # Align signs
        v2_a, v2_b, v2_flipped = align_signs(v2_a_interp, v2_b_interp)

        # Compute metrics for v2
        v2_cos = cosine_similarity(v2_a, v2_b)
        v2_spear, v2_spear_p = spearman_correlation(v2_a, v2_b)

        # Peak agreement (in common grid)
        peak_a = np.argmax(np.abs(v2_a))
        peak_b = np.argmax(np.abs(v2_b))
        peak_match = peak_a == peak_b
        peak_dist = abs(peak_a - peak_b)

        # u2 comparison only if dimensions match
        u2_cos = 0.0
        u2_spear = 0.0
        u2_spear_p = 1.0

        if len(mode_a.u2) == len(mode_b.u2):
            u2_a = mode_a.u2
            u2_b = -mode_b.u2 if v2_flipped else mode_b.u2
            u2_cos = cosine_similarity(u2_a, u2_b)
            u2_spear, u2_spear_p = spearman_correlation(u2_a, u2_b)

        # Overall score
        score = 0.6 * abs(v2_cos) + 0.2 * abs(u2_cos) + 0.2 * (1.0 if peak_match else max(0, 1 - peak_dist / len(v2_a)))

        # Grade
        if score > 0.9:
            grade = "excellent"
        elif score > 0.75:
            grade = "good"
        elif score > 0.5:
            grade = "moderate"
        else:
            grade = "poor"

        return ReplicationMetrics(
            source_a=source_a,
            source_b=source_b,
            sign_aligned=not v2_flipped,
            v2_cosine=v2_cos,
            v2_spearman=v2_spear,
            v2_spearman_pvalue=v2_spear_p,
            u2_cosine=u2_cos,
            u2_spearman=u2_spear,
            u2_spearman_pvalue=u2_spear_p,
            v2_peak_a=peak_a,
            v2_peak_b=peak_b,
            v2_peak_match=peak_match,
            v2_peak_distance=peak_dist,
            replication_score=score,
            replication_grade=grade,
            comparable=True,
            comparability_note=f"Interpolated to {len(common_grid)} common points",
        )


@dataclass
class ReplicationReport:
    """Complete replication analysis report."""
    dataset: str
    n_comparisons: int
    comparisons: List[ReplicationMetrics]

    # Summary
    mean_v2_cosine: float = 0.0
    mean_replication_score: float = 0.0
    all_replicate: bool = False  # All comparisons have good+ grade

    # Key findings
    best_comparison: Optional[str] = None
    worst_comparison: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "n_comparisons": self.n_comparisons,
            "comparisons": [c.to_dict() for c in self.comparisons],
            "mean_v2_cosine": self.mean_v2_cosine,
            "mean_replication_score": self.mean_replication_score,
            "all_replicate": self.all_replicate,
            "best_comparison": self.best_comparison,
            "worst_comparison": self.worst_comparison,
        }


def compute_replication_report(
    dataset: str,
    comparisons: List[ReplicationMetrics],
) -> ReplicationReport:
    """Generate a replication report from pairwise comparisons."""
    if not comparisons:
        return ReplicationReport(
            dataset=dataset,
            n_comparisons=0,
            comparisons=[],
        )

    comparable = [c for c in comparisons if c.comparable]

    if not comparable:
        return ReplicationReport(
            dataset=dataset,
            n_comparisons=len(comparisons),
            comparisons=comparisons,
        )

    v2_cosines = [c.v2_cosine for c in comparable]
    scores = [c.replication_score for c in comparable]
    grades = [c.replication_grade for c in comparable]

    mean_v2_cos = float(np.mean(v2_cosines))
    mean_score = float(np.mean(scores))
    all_good = all(g in ["excellent", "good"] for g in grades)

    # Best and worst
    best_idx = np.argmax(scores)
    worst_idx = np.argmin(scores)

    best_comp = f"{comparable[best_idx].source_a} vs {comparable[best_idx].source_b}"
    worst_comp = f"{comparable[worst_idx].source_a} vs {comparable[worst_idx].source_b}"

    return ReplicationReport(
        dataset=dataset,
        n_comparisons=len(comparisons),
        comparisons=comparisons,
        mean_v2_cosine=mean_v2_cos,
        mean_replication_score=mean_score,
        all_replicate=all_good,
        best_comparison=best_comp,
        worst_comparison=worst_comp,
    )


# ============================================================================
# DATASET-SPECIFIC REPLICATION STRATEGIES
# ============================================================================

def replicate_elastic_modes(
    modes_by_energy: Dict[str, ResidualMode],
    t_grids_by_energy: Dict[str, np.ndarray],
    raw_mode: Optional[ResidualMode] = None,
    normalized_mode: Optional[ResidualMode] = None,
) -> ReplicationReport:
    """
    Replication analysis for elastic scattering.

    Compares:
    - v2(t) across different energies
    - raw vs normalized modes
    """
    comparator = ModeComparator()
    comparisons = []

    # Compare across energies
    energies = list(modes_by_energy.keys())
    for i in range(len(energies)):
        for j in range(i + 1, len(energies)):
            e_a, e_b = energies[i], energies[j]
            mode_a = modes_by_energy[e_a]
            mode_b = modes_by_energy[e_b]
            grid_a = t_grids_by_energy[e_a]
            grid_b = t_grids_by_energy[e_b]

            metrics = comparator.compare_interpolated(
                mode_a, grid_a,
                mode_b, grid_b,
                source_a=e_a,
                source_b=e_b,
            )
            comparisons.append(metrics)

    # Compare raw vs normalized if both available
    if raw_mode is not None and normalized_mode is not None:
        metrics = comparator.compare_direct(
            raw_mode, normalized_mode,
            source_a="raw",
            source_b="normalized",
        )
        comparisons.append(metrics)

    return compute_replication_report("elastic", comparisons)


def replicate_diffractive_modes(
    h1_mode: Optional[ResidualMode] = None,
    zeus_mode: Optional[ResidualMode] = None,
    h1_xP_grid: Optional[np.ndarray] = None,
    zeus_xP_grid: Optional[np.ndarray] = None,
    combined_mode: Optional[ResidualMode] = None,
) -> ReplicationReport:
    """
    Replication analysis for diffractive DIS.

    Compares:
    - H1 vs ZEUS v2(xP) shapes
    - Individual vs combined analysis
    """
    comparator = ModeComparator()
    comparisons = []

    # H1 vs ZEUS
    if h1_mode is not None and zeus_mode is not None:
        if h1_xP_grid is not None and zeus_xP_grid is not None:
            metrics = comparator.compare_interpolated(
                h1_mode, h1_xP_grid,
                zeus_mode, zeus_xP_grid,
                source_a="H1",
                source_b="ZEUS",
            )
        else:
            # Try direct comparison if grids not provided
            metrics = comparator.compare_direct(
                h1_mode, zeus_mode,
                source_a="H1",
                source_b="ZEUS",
            )
        comparisons.append(metrics)

    # Combined vs individual
    if combined_mode is not None:
        if h1_mode is not None:
            metrics = comparator.compare_direct(
                combined_mode, h1_mode,
                source_a="combined",
                source_b="H1",
            )
            comparisons.append(metrics)

        if zeus_mode is not None:
            metrics = comparator.compare_direct(
                combined_mode, zeus_mode,
                source_a="combined",
                source_b="ZEUS",
            )
            comparisons.append(metrics)

    return compute_replication_report("diffractive", comparisons)


def replicate_higgs_modes(
    full_mode: ResidualMode,
    loo_modes: Dict[str, ResidualMode],
) -> ReplicationReport:
    """
    Replication analysis for Higgs signal strengths.

    Compares:
    - Full dataset mode vs leave-one-out modes
    - Stability of residual when removing each decay channel
    """
    comparator = ModeComparator()
    comparisons = []

    # Full vs each leave-one-out
    for channel, loo_mode in loo_modes.items():
        metrics = comparator.compare_direct(
            full_mode, loo_mode,
            source_a="full",
            source_b=f"loo_{channel}",
        )
        comparisons.append(metrics)

    # Pairwise leave-one-out comparisons
    channels = list(loo_modes.keys())
    for i in range(len(channels)):
        for j in range(i + 1, len(channels)):
            ch_a, ch_b = channels[i], channels[j]
            metrics = comparator.compare_direct(
                loo_modes[ch_a], loo_modes[ch_b],
                source_a=f"loo_{ch_a}",
                source_b=f"loo_{ch_b}",
            )
            comparisons.append(metrics)

    return compute_replication_report("higgs", comparisons)
