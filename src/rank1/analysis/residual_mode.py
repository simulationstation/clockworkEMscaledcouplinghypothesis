"""
Residual mode extraction and analysis for rank-2 deviations.

This module provides:
1. Extraction of the rank-2 residual mode (u2, v2) from fitted models
2. Deterministic sign convention for reproducibility
3. Localization metrics to quantify where residuals concentrate
4. Residual maps for visualization
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
from scipy import stats

from rank1.logging import get_logger

logger = get_logger()


@dataclass
class LocalizationMetrics:
    """Metrics quantifying how localized a residual mode is."""

    # Top-k mass: fraction of ||v||² in top k elements
    top_k_mass: Dict[int, float] = field(default_factory=dict)

    # Gini coefficient (0 = uniform, 1 = maximally concentrated)
    gini: float = 0.0

    # Entropy over normalized |v|² (lower = more localized)
    entropy: float = 0.0

    # Max entropy for reference
    max_entropy: float = 0.0

    # Normalized entropy (0 = maximally localized, 1 = uniform)
    normalized_entropy: float = 1.0

    # For ordered bins: contiguous window concentration
    # window_concentration[w] = max fraction in any window of size w
    window_concentration: Dict[int, float] = field(default_factory=dict)

    # Peak location (index of max |v|)
    peak_index: int = 0
    peak_value: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "top_k_mass": self.top_k_mass,
            "gini": self.gini,
            "entropy": self.entropy,
            "max_entropy": self.max_entropy,
            "normalized_entropy": self.normalized_entropy,
            "window_concentration": self.window_concentration,
            "peak_index": self.peak_index,
            "peak_value": self.peak_value,
        }


@dataclass
class ResidualMode:
    """Extracted rank-2 residual mode with metadata."""

    # The residual mode vectors
    u2: np.ndarray  # Row dependence (n_rows,)
    v2: np.ndarray  # Column dependence (n_cols,)

    # Sign convention applied
    sign_convention: str = "max_positive"

    # Scale factor (absorbed into u2)
    scale: float = 1.0

    # Localization metrics for v2
    v2_localization: Optional[LocalizationMetrics] = None

    # Localization metrics for u2
    u2_localization: Optional[LocalizationMetrics] = None

    # Row and column labels if available
    row_labels: Optional[List[str]] = None
    col_labels: Optional[List[str]] = None

    # Contribution to chi² improvement per cell
    chi2_contributions: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "u2": self.u2.tolist(),
            "v2": self.v2.tolist(),
            "sign_convention": self.sign_convention,
            "scale": self.scale,
            "v2_localization": self.v2_localization.to_dict() if self.v2_localization else None,
            "u2_localization": self.u2_localization.to_dict() if self.u2_localization else None,
            "row_labels": self.row_labels,
            "col_labels": self.col_labels,
            "chi2_contributions": self.chi2_contributions.tolist() if self.chi2_contributions is not None else None,
        }


@dataclass
class ResidualMap:
    """Standardized residual maps under different models."""

    # Standardized residuals under rank-1: R_ij = (M_ij - Mhat1_ij) / σ_ij
    residuals_rank1: np.ndarray

    # Rank-2 correction component: Δ_ij = u2_i * v2_j
    rank2_correction: np.ndarray

    # Residuals under rank-2
    residuals_rank2: np.ndarray

    # Row and column indices for sparse data
    row_indices: np.ndarray
    col_indices: np.ndarray

    # Labels
    row_labels: Optional[List[str]] = None
    col_labels: Optional[List[str]] = None

    def to_dense_matrix(self, which: str = "rank1") -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert sparse residuals to dense matrix for visualization.

        Returns:
            (matrix, mask) where mask indicates valid cells
        """
        n_rows = self.row_indices.max() + 1
        n_cols = self.col_indices.max() + 1

        matrix = np.full((n_rows, n_cols), np.nan)
        mask = np.zeros((n_rows, n_cols), dtype=bool)

        if which == "rank1":
            values = self.residuals_rank1
        elif which == "rank2":
            values = self.residuals_rank2
        elif which == "correction":
            values = self.rank2_correction
        else:
            raise ValueError(f"Unknown residual type: {which}")

        for i, (r, c, v) in enumerate(zip(self.row_indices, self.col_indices, values)):
            matrix[r, c] = v
            mask[r, c] = True

        return matrix, mask


def apply_sign_convention(
    u: np.ndarray,
    v: np.ndarray,
    convention: str = "max_positive"
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Apply deterministic sign convention to (u, v) pair.

    Args:
        u: Row vector
        v: Column vector
        convention: One of:
            - "max_positive": Ensure max(|v|) element is positive
            - "dot_positive": Ensure v has positive dot product with all-ones
            - "first_positive": Ensure first non-zero element of v is positive

    Returns:
        (u_signed, v_signed, convention_used)
    """
    if convention == "max_positive":
        # Find index of maximum absolute value
        max_idx = np.argmax(np.abs(v))
        if v[max_idx] < 0:
            return -u, -v, convention
        return u.copy(), v.copy(), convention

    elif convention == "dot_positive":
        # Dot product with all-ones
        if np.sum(v) < 0:
            return -u, -v, convention
        return u.copy(), v.copy(), convention

    elif convention == "first_positive":
        # First non-zero element
        nonzero = np.nonzero(np.abs(v) > 1e-10)[0]
        if len(nonzero) > 0 and v[nonzero[0]] < 0:
            return -u, -v, convention
        return u.copy(), v.copy(), convention

    else:
        raise ValueError(f"Unknown sign convention: {convention}")


def normalize_mode(
    u: np.ndarray,
    v: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Normalize mode so v has unit L2 norm, scale absorbed into u.

    Returns:
        (u_normalized, v_normalized, scale_factor)
    """
    v_norm = np.linalg.norm(v)
    if v_norm < 1e-12:
        return u.copy(), v.copy(), 0.0

    v_normalized = v / v_norm
    u_normalized = u * v_norm

    return u_normalized, v_normalized, v_norm


def compute_localization_metrics(
    v: np.ndarray,
    k_values: Optional[List[int]] = None,
    window_sizes: Optional[List[int]] = None,
    ordered_bins: bool = True
) -> LocalizationMetrics:
    """
    Compute localization metrics for a vector.

    Args:
        v: Input vector
        k_values: Values of k for top-k mass computation
        window_sizes: Window sizes for contiguous concentration
        ordered_bins: Whether bins are ordered (for window metrics)

    Returns:
        LocalizationMetrics
    """
    n = len(v)
    if n == 0:
        return LocalizationMetrics()

    # Default k values
    if k_values is None:
        k_values = [1, 2, 3, min(5, n), min(n // 4, n)]
        k_values = sorted(set(k for k in k_values if k <= n))

    # Squared magnitudes (for energy distribution)
    v_sq = v ** 2
    total_energy = np.sum(v_sq)

    if total_energy < 1e-12:
        return LocalizationMetrics()

    # Normalized energy distribution
    p = v_sq / total_energy

    # Top-k mass
    sorted_sq = np.sort(v_sq)[::-1]
    cumsum = np.cumsum(sorted_sq)
    top_k_mass = {}
    for k in k_values:
        if k <= n:
            top_k_mass[k] = cumsum[k - 1] / total_energy

    # Gini coefficient
    sorted_p = np.sort(p)
    n_p = len(sorted_p)
    index = np.arange(1, n_p + 1)
    gini = (2 * np.sum(index * sorted_p) - (n_p + 1) * np.sum(sorted_p)) / (n_p * np.sum(sorted_p))
    gini = max(0, min(1, gini))  # Clamp to [0, 1]

    # Entropy
    # Avoid log(0) by filtering
    p_nonzero = p[p > 1e-12]
    if len(p_nonzero) > 0:
        entropy = -np.sum(p_nonzero * np.log(p_nonzero))
    else:
        entropy = 0.0
    max_entropy = np.log(n)
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 1.0

    # Peak location
    peak_idx = np.argmax(np.abs(v))
    peak_value = v[peak_idx]

    # Window concentration (for ordered bins)
    window_concentration = {}
    if ordered_bins and window_sizes is not None:
        for w in window_sizes:
            if w <= n:
                # Sliding window max
                max_frac = 0.0
                for start in range(n - w + 1):
                    window_energy = np.sum(v_sq[start:start + w])
                    frac = window_energy / total_energy
                    max_frac = max(max_frac, frac)
                window_concentration[w] = max_frac
    elif ordered_bins:
        # Default window sizes
        default_windows = [1, 2, 3, max(1, n // 4), max(1, n // 2)]
        default_windows = sorted(set(w for w in default_windows if w <= n))
        for w in default_windows:
            max_frac = 0.0
            for start in range(n - w + 1):
                window_energy = np.sum(v_sq[start:start + w])
                frac = window_energy / total_energy
                max_frac = max(max_frac, frac)
            window_concentration[w] = max_frac

    return LocalizationMetrics(
        top_k_mass=top_k_mass,
        gini=gini,
        entropy=entropy,
        max_entropy=max_entropy,
        normalized_entropy=normalized_entropy,
        window_concentration=window_concentration,
        peak_index=int(peak_idx),
        peak_value=float(peak_value),
    )


class ResidualModeExtractor:
    """Extract and analyze rank-2 residual modes from fitted models."""

    def __init__(
        self,
        sign_convention: str = "max_positive",
        ordered_cols: bool = True,
        ordered_rows: bool = False,
    ):
        """
        Initialize extractor.

        Args:
            sign_convention: Sign convention for v2
            ordered_cols: Whether columns are ordered (for window metrics)
            ordered_rows: Whether rows are ordered
        """
        self.sign_convention = sign_convention
        self.ordered_cols = ordered_cols
        self.ordered_rows = ordered_rows

    def extract_from_fits(
        self,
        fit_rank1,
        fit_rank2,
        row_labels: Optional[List[str]] = None,
        col_labels: Optional[List[str]] = None,
    ) -> ResidualMode:
        """
        Extract residual mode from rank-1 and rank-2 fits.

        Args:
            fit_rank1: Fitted rank-1 model OR FitResult (has model attribute with u, v)
            fit_rank2: Fitted rank-2 model OR FitResult (has model attribute with U, V with 2 columns)
            row_labels: Optional row labels
            col_labels: Optional column labels

        Returns:
            ResidualMode with extracted and normalized (u2, v2)
        """
        # Handle FitResult objects - extract the model
        model2 = fit_rank2
        if hasattr(fit_rank2, 'model'):
            model2 = fit_rank2.model
            logger.debug("Extracted model from FitResult for rank-2")

        # Get rank-2 second mode
        # Convention: U is (n_rows, rank), V is (n_cols, rank)
        u2_raw = None
        v2_raw = None

        if hasattr(model2, 'U') and hasattr(model2, 'V'):
            U = model2.U
            V = model2.V

            if U is None or V is None:
                raise ValueError("Rank-2 model has None for U or V")

            U = np.asarray(U)
            V = np.asarray(V)

            if U.ndim == 1:
                # Single mode, shouldn't happen for rank-2
                logger.warning("Rank-2 fit has only 1D U/V, using as second mode")
                u2_raw = U
                v2_raw = V
            elif U.shape[1] < 2 or V.shape[1] < 2:
                # Degenerate rank-2 fit - only 1 mode extracted
                logger.warning(
                    f"Degenerate rank-2 fit: U.shape={U.shape}, V.shape={V.shape}. "
                    "Using first mode (may be identical to rank-1)."
                )
                u2_raw = U[:, 0]
                v2_raw = V[:, 0]
            else:
                # Normal case: second column is the residual mode
                u2_raw = U[:, 1]
                v2_raw = V[:, 1]
        elif hasattr(model2, 'u2') and hasattr(model2, 'v2'):
            u2_raw = model2.u2
            v2_raw = model2.v2
        elif hasattr(model2, 'u') and hasattr(model2, 'v'):
            # Model only has rank-1 attributes - this is a degenerate case
            logger.warning(
                "Rank-2 model has rank-1 structure (u, v). "
                "This indicates the rank-2 fit collapsed to rank-1."
            )
            u2_raw = np.asarray(model2.u)
            v2_raw = np.asarray(model2.v)
        else:
            raise ValueError(
                f"Cannot extract second mode from rank-2 fit. "
                f"Model type: {type(model2)}, attrs: {dir(model2)}"
            )

        # Check for NaN/Inf values
        if not np.all(np.isfinite(u2_raw)) or not np.all(np.isfinite(v2_raw)):
            logger.warning("Residual mode contains NaN/Inf values, replacing with zeros")
            u2_raw = np.nan_to_num(u2_raw, nan=0.0, posinf=0.0, neginf=0.0)
            v2_raw = np.nan_to_num(v2_raw, nan=0.0, posinf=0.0, neginf=0.0)

        # Apply sign convention
        u2_signed, v2_signed, convention = apply_sign_convention(
            u2_raw, v2_raw, self.sign_convention
        )

        # Normalize
        u2_norm, v2_norm, scale = normalize_mode(u2_signed, v2_signed)

        # Compute localization metrics
        v2_loc = compute_localization_metrics(
            v2_norm, ordered_bins=self.ordered_cols
        )
        u2_loc = compute_localization_metrics(
            u2_norm, ordered_bins=self.ordered_rows
        )

        return ResidualMode(
            u2=u2_norm,
            v2=v2_norm,
            sign_convention=convention,
            scale=scale,
            v2_localization=v2_loc,
            u2_localization=u2_loc,
            row_labels=row_labels,
            col_labels=col_labels,
        )

    def compute_residual_map(
        self,
        fit_rank1,
        fit_rank2,
        values: np.ndarray,
        errors: np.ndarray,
        row_indices: np.ndarray,
        col_indices: np.ndarray,
        row_labels: Optional[List[str]] = None,
        col_labels: Optional[List[str]] = None,
    ) -> ResidualMap:
        """
        Compute residual maps under rank-1 and rank-2.

        Args:
            fit_rank1: Fitted rank-1 model OR FitResult
            fit_rank2: Fitted rank-2 model OR FitResult
            values: Observed values
            errors: Measurement errors
            row_indices: Row indices for each observation
            col_indices: Column indices for each observation
            row_labels: Optional row labels
            col_labels: Optional column labels

        Returns:
            ResidualMap with standardized residuals
        """
        # Handle FitResult objects - extract the model
        model1 = fit_rank1.model if hasattr(fit_rank1, 'model') else fit_rank1
        model2 = fit_rank2.model if hasattr(fit_rank2, 'model') else fit_rank2

        # Get predictions using the model's predict method
        if hasattr(model1, 'predict'):
            pred1 = model1.predict(row_indices, col_indices)
        elif hasattr(model1, 'predict_obs'):
            pred1 = model1.predict_obs(row_indices, col_indices)
        else:
            raise ValueError(f"Model1 has no predict method: {type(model1)}")

        if hasattr(model2, 'predict'):
            pred2 = model2.predict(row_indices, col_indices)
        elif hasattr(model2, 'predict_obs'):
            pred2 = model2.predict_obs(row_indices, col_indices)
        else:
            raise ValueError(f"Model2 has no predict method: {type(model2)}")

        # Standardized residuals
        residuals_1 = (values - pred1) / errors
        residuals_2 = (values - pred2) / errors

        # Rank-2 correction
        correction = pred2 - pred1

        return ResidualMap(
            residuals_rank1=residuals_1,
            rank2_correction=correction / errors,  # Standardized
            residuals_rank2=residuals_2,
            row_indices=row_indices,
            col_indices=col_indices,
            row_labels=row_labels,
            col_labels=col_labels,
        )

    def compute_chi2_contributions(
        self,
        fit_rank1,
        fit_rank2,
        values: np.ndarray,
        errors: np.ndarray,
        row_indices: np.ndarray,
        col_indices: np.ndarray,
    ) -> np.ndarray:
        """
        Compute per-cell contribution to χ² improvement.

        Args:
            fit_rank1: Fitted rank-1 model OR FitResult
            fit_rank2: Fitted rank-2 model OR FitResult
            values: Observed values
            errors: Measurement errors
            row_indices: Row indices for each observation
            col_indices: Column indices for each observation

        Returns:
            Array of χ²_1(cell) - χ²_2(cell) for each observation
        """
        # Handle FitResult objects - extract the model
        model1 = fit_rank1.model if hasattr(fit_rank1, 'model') else fit_rank1
        model2 = fit_rank2.model if hasattr(fit_rank2, 'model') else fit_rank2

        # Get predictions
        if hasattr(model1, 'predict'):
            pred1 = model1.predict(row_indices, col_indices)
        elif hasattr(model1, 'predict_obs'):
            pred1 = model1.predict_obs(row_indices, col_indices)
        else:
            raise ValueError(f"Model1 has no predict method: {type(model1)}")

        if hasattr(model2, 'predict'):
            pred2 = model2.predict(row_indices, col_indices)
        elif hasattr(model2, 'predict_obs'):
            pred2 = model2.predict_obs(row_indices, col_indices)
        else:
            raise ValueError(f"Model2 has no predict method: {type(model2)}")

        chi2_1_cells = ((values - pred1) / errors) ** 2
        chi2_2_cells = ((values - pred2) / errors) ** 2

        return chi2_1_cells - chi2_2_cells


@dataclass
class StabilityMetrics:
    """Metrics quantifying stability of residual mode."""

    # Across multi-start fits
    n_starts: int = 0
    v2_cosine_mean: float = 0.0
    v2_cosine_std: float = 0.0
    u2_cosine_mean: float = 0.0
    u2_cosine_std: float = 0.0

    # Across bootstrap resamples
    n_bootstrap: int = 0
    v2_bootstrap_cosine_mean: float = 0.0
    v2_bootstrap_cosine_std: float = 0.0
    u2_bootstrap_cosine_mean: float = 0.0
    u2_bootstrap_cosine_std: float = 0.0

    # Peak stability
    peak_index_mode: int = 0  # Most common peak index
    peak_index_consistency: float = 0.0  # Fraction of runs with mode peak

    # Overall stability grade
    is_stable: bool = False
    stability_grade: str = "unknown"  # "high", "medium", "low", "unstable"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_starts": self.n_starts,
            "v2_cosine_mean": self.v2_cosine_mean,
            "v2_cosine_std": self.v2_cosine_std,
            "u2_cosine_mean": self.u2_cosine_mean,
            "u2_cosine_std": self.u2_cosine_std,
            "n_bootstrap": self.n_bootstrap,
            "v2_bootstrap_cosine_mean": self.v2_bootstrap_cosine_mean,
            "v2_bootstrap_cosine_std": self.v2_bootstrap_cosine_std,
            "u2_bootstrap_cosine_mean": self.u2_bootstrap_cosine_mean,
            "u2_bootstrap_cosine_std": self.u2_bootstrap_cosine_std,
            "peak_index_mode": self.peak_index_mode,
            "peak_index_consistency": self.peak_index_consistency,
            "is_stable": self.is_stable,
            "stability_grade": self.stability_grade,
        }


def compute_stability_metrics(
    reference_mode: ResidualMode,
    multistart_modes: List[ResidualMode],
    bootstrap_modes: Optional[List[ResidualMode]] = None,
) -> StabilityMetrics:
    """
    Compute stability metrics for a residual mode.

    Args:
        reference_mode: The reference (e.g., best-fit) residual mode
        multistart_modes: Modes from different random initializations
        bootstrap_modes: Modes from bootstrap resamples

    Returns:
        StabilityMetrics
    """
    metrics = StabilityMetrics()

    # Multi-start stability
    if len(multistart_modes) > 0:
        metrics.n_starts = len(multistart_modes)

        v2_cosines = []
        u2_cosines = []
        peak_indices = []

        ref_v2 = reference_mode.v2
        ref_u2 = reference_mode.u2

        for mode in multistart_modes:
            # Align signs before comparison
            v2_aligned, _, _ = apply_sign_convention(
                mode.u2, mode.v2, reference_mode.sign_convention
            )
            _, u2_aligned = v2_aligned, mode.u2  # u2 gets same sign flip

            # Actually need to check sign alignment with reference
            if np.dot(mode.v2, ref_v2) < 0:
                v2_compare = -mode.v2
                u2_compare = -mode.u2
            else:
                v2_compare = mode.v2
                u2_compare = mode.u2

            # Cosine similarity
            v2_cos = np.abs(np.dot(v2_compare, ref_v2)) / (
                np.linalg.norm(v2_compare) * np.linalg.norm(ref_v2) + 1e-12
            )
            u2_cos = np.abs(np.dot(u2_compare, ref_u2)) / (
                np.linalg.norm(u2_compare) * np.linalg.norm(ref_u2) + 1e-12
            )

            v2_cosines.append(v2_cos)
            u2_cosines.append(u2_cos)

            # Peak index
            if mode.v2_localization:
                peak_indices.append(mode.v2_localization.peak_index)

        metrics.v2_cosine_mean = float(np.mean(v2_cosines))
        metrics.v2_cosine_std = float(np.std(v2_cosines))
        metrics.u2_cosine_mean = float(np.mean(u2_cosines))
        metrics.u2_cosine_std = float(np.std(u2_cosines))

        if peak_indices:
            from collections import Counter
            peak_counts = Counter(peak_indices)
            mode_peak, mode_count = peak_counts.most_common(1)[0]
            metrics.peak_index_mode = mode_peak
            metrics.peak_index_consistency = mode_count / len(peak_indices)

    # Bootstrap stability
    if bootstrap_modes and len(bootstrap_modes) > 0:
        metrics.n_bootstrap = len(bootstrap_modes)

        v2_boot_cosines = []
        u2_boot_cosines = []

        for mode in bootstrap_modes:
            if np.dot(mode.v2, ref_v2) < 0:
                v2_compare = -mode.v2
                u2_compare = -mode.u2
            else:
                v2_compare = mode.v2
                u2_compare = mode.u2

            v2_cos = np.abs(np.dot(v2_compare, ref_v2)) / (
                np.linalg.norm(v2_compare) * np.linalg.norm(ref_v2) + 1e-12
            )
            u2_cos = np.abs(np.dot(u2_compare, ref_u2)) / (
                np.linalg.norm(u2_compare) * np.linalg.norm(ref_u2) + 1e-12
            )

            v2_boot_cosines.append(v2_cos)
            u2_boot_cosines.append(u2_cos)

        metrics.v2_bootstrap_cosine_mean = float(np.mean(v2_boot_cosines))
        metrics.v2_bootstrap_cosine_std = float(np.std(v2_boot_cosines))
        metrics.u2_bootstrap_cosine_mean = float(np.mean(u2_boot_cosines))
        metrics.u2_bootstrap_cosine_std = float(np.std(u2_boot_cosines))

    # Determine stability grade
    if metrics.n_starts == 0:
        # No multistart data - stability is unknown
        metrics.stability_grade = "unknown"
        metrics.is_stable = False
    else:
        v2_stable = metrics.v2_cosine_mean > 0.9 and metrics.v2_cosine_std < 0.1
        u2_stable = metrics.u2_cosine_mean > 0.9 and metrics.u2_cosine_std < 0.1

        if v2_stable and u2_stable:
            metrics.stability_grade = "high"
            metrics.is_stable = True
        elif metrics.v2_cosine_mean > 0.8 and metrics.u2_cosine_mean > 0.8:
            metrics.stability_grade = "medium"
            metrics.is_stable = True
        elif metrics.v2_cosine_mean > 0.6:
            metrics.stability_grade = "low"
            metrics.is_stable = False
        else:
            metrics.stability_grade = "unstable"
            metrics.is_stable = False

    return metrics
