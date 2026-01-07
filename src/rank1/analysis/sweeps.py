"""
Predefined sweep presets and global look-elsewhere correction.

This module provides:
1. Predefined sweep configurations per dataset (no auto-generated large grids)
2. Sweep runner with parallelization
3. Global look-elsewhere correction via sweep-bootstrap
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable, Tuple
from enum import Enum
from multiprocessing import Pool, cpu_count
import json
from pathlib import Path
import numpy as np

from rank1.logging import get_logger

logger = get_logger()


class SweepType(Enum):
    """Types of sweep variations."""
    BASELINE = "baseline"
    LEAVE_ONE_OUT = "leave_one_out"
    RANGE_VARIATION = "range_variation"
    RESOLUTION = "resolution"
    MODE = "mode"
    KINEMATIC_CUT = "kinematic_cut"
    DATA_SUBSET = "data_subset"


@dataclass
class SweepPreset:
    """A single sweep configuration preset."""
    name: str
    sweep_type: SweepType
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)

    # For leave-one-out: which element to exclude
    exclude_element: Optional[str] = None

    # For range variations: min/max bounds
    range_min: Optional[float] = None
    range_max: Optional[float] = None

    # For resolution: number of points
    n_points: Optional[int] = None

    # For mode: raw vs normalized
    mode: Optional[str] = None

    # For kinematic cuts
    cuts: Dict[str, Any] = field(default_factory=dict)

    # For data subsets: which subset
    subset: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "sweep_type": self.sweep_type.value,
            "description": self.description,
            "parameters": self.parameters,
            "exclude_element": self.exclude_element,
            "range_min": self.range_min,
            "range_max": self.range_max,
            "n_points": self.n_points,
            "mode": self.mode,
            "cuts": self.cuts,
            "subset": self.subset,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SweepPreset":
        d = d.copy()
        d["sweep_type"] = SweepType(d["sweep_type"])
        return cls(**d)


# ============================================================================
# PREDEFINED PRESETS PER DATASET
# ============================================================================

HIGGS_PRESETS = [
    SweepPreset(
        name="baseline",
        sweep_type=SweepType.BASELINE,
        description="Full dataset, no modifications",
    ),
    # Leave-one-out decay channels
    SweepPreset(
        name="loo_gamgam",
        sweep_type=SweepType.LEAVE_ONE_OUT,
        description="Exclude γγ decay channel",
        exclude_element="γγ",
    ),
    SweepPreset(
        name="loo_ZZ",
        sweep_type=SweepType.LEAVE_ONE_OUT,
        description="Exclude ZZ decay channel",
        exclude_element="ZZ",
    ),
    SweepPreset(
        name="loo_WW",
        sweep_type=SweepType.LEAVE_ONE_OUT,
        description="Exclude WW decay channel",
        exclude_element="WW",
    ),
    SweepPreset(
        name="loo_tautau",
        sweep_type=SweepType.LEAVE_ONE_OUT,
        description="Exclude ττ decay channel",
        exclude_element="ττ",
    ),
    SweepPreset(
        name="loo_bb",
        sweep_type=SweepType.LEAVE_ONE_OUT,
        description="Exclude bb̄ decay channel",
        exclude_element="bb̄",
    ),
    # Production mode groupings
    SweepPreset(
        name="gluon_only",
        sweep_type=SweepType.DATA_SUBSET,
        description="Only gluon-initiated production modes",
        subset="gluon",
        parameters={"include_rows": ["ggF", "ttH"]},
    ),
    SweepPreset(
        name="vector_only",
        sweep_type=SweepType.DATA_SUBSET,
        description="Only vector boson production modes",
        subset="vector",
        parameters={"include_rows": ["VBF", "VH"]},
    ),
]

ELASTIC_PRESETS = [
    SweepPreset(
        name="baseline",
        sweep_type=SweepType.BASELINE,
        description="Full overlap t-range, default resolution",
    ),
    # Range variations
    SweepPreset(
        name="tight_range",
        sweep_type=SweepType.RANGE_VARIATION,
        description="Tight t-range: 0.05-0.15 GeV²",
        range_min=0.05,
        range_max=0.15,
    ),
    SweepPreset(
        name="medium_range",
        sweep_type=SweepType.RANGE_VARIATION,
        description="Medium t-range: 0.03-0.20 GeV²",
        range_min=0.03,
        range_max=0.20,
    ),
    SweepPreset(
        name="wide_range",
        sweep_type=SweepType.RANGE_VARIATION,
        description="Wide t-range: 0.02-0.30 GeV²",
        range_min=0.02,
        range_max=0.30,
    ),
    # Resolution variations
    SweepPreset(
        name="res_50",
        sweep_type=SweepType.RESOLUTION,
        description="50 interpolation points",
        n_points=50,
    ),
    SweepPreset(
        name="res_100",
        sweep_type=SweepType.RESOLUTION,
        description="100 interpolation points",
        n_points=100,
    ),
    SweepPreset(
        name="res_200",
        sweep_type=SweepType.RESOLUTION,
        description="200 interpolation points",
        n_points=200,
    ),
    # Mode variations
    SweepPreset(
        name="raw_mode",
        sweep_type=SweepType.MODE,
        description="Raw dσ/dt values",
        mode="raw",
    ),
    SweepPreset(
        name="normalized_mode",
        sweep_type=SweepType.MODE,
        description="Shape-only normalized",
        mode="normalized",
    ),
]

DIFFRACTIVE_PRESETS = [
    SweepPreset(
        name="baseline",
        sweep_type=SweepType.BASELINE,
        description="Full dataset with standard cuts",
    ),
    # Kinematic cut variations
    SweepPreset(
        name="high_Q2",
        sweep_type=SweepType.KINEMATIC_CUT,
        description="Q² > 10 GeV² only",
        cuts={"Q2_min": 10.0},
    ),
    SweepPreset(
        name="mid_beta",
        sweep_type=SweepType.KINEMATIC_CUT,
        description="0.1 < β < 0.9 (exclude extremes)",
        cuts={"beta_min": 0.1, "beta_max": 0.9},
    ),
    SweepPreset(
        name="dense_coverage",
        sweep_type=SweepType.KINEMATIC_CUT,
        description="Stricter coverage threshold",
        cuts={"min_xP_points": 5},
    ),
    # Data subsets
    SweepPreset(
        name="h1_only",
        sweep_type=SweepType.DATA_SUBSET,
        description="H1 data only",
        subset="H1",
    ),
    SweepPreset(
        name="zeus_only",
        sweep_type=SweepType.DATA_SUBSET,
        description="ZEUS data only",
        subset="ZEUS",
    ),
]


def get_presets_for_dataset(dataset: str) -> List[SweepPreset]:
    """Get predefined presets for a dataset."""
    presets_map = {
        "higgs": HIGGS_PRESETS,
        "higgs_atlas": HIGGS_PRESETS,
        "higgs_atlas_mu": HIGGS_PRESETS,
        "elastic": ELASTIC_PRESETS,
        "elastic_totem": ELASTIC_PRESETS,
        "diffractive": DIFFRACTIVE_PRESETS,
        "diffractive_dis": DIFFRACTIVE_PRESETS,
    }
    return presets_map.get(dataset, [])


def get_fast_presets(dataset: str) -> List[SweepPreset]:
    """Get reduced preset list for fast mode."""
    all_presets = get_presets_for_dataset(dataset)
    # Keep baseline + 2-3 key variations
    fast_map = {
        "higgs": ["baseline", "loo_gamgam", "loo_bb"],
        "higgs_atlas": ["baseline", "loo_gamgam", "loo_bb"],
        "higgs_atlas_mu": ["baseline", "loo_gamgam", "loo_bb"],
        "elastic": ["baseline", "tight_range", "normalized_mode"],
        "elastic_totem": ["baseline", "tight_range", "normalized_mode"],
        "diffractive": ["baseline", "high_Q2", "h1_only"],
        "diffractive_dis": ["baseline", "high_Q2", "h1_only"],
    }
    fast_names = fast_map.get(dataset, ["baseline"])
    return [p for p in all_presets if p.name in fast_names]


@dataclass
class SweepResult:
    """Result from a single sweep preset run."""
    preset: SweepPreset
    lambda_stat: float
    p_local: float
    chi2_rank1: float
    chi2_rank2: float
    ndof_rank1: int
    ndof_rank2: int
    fit_converged: bool = True
    error: Optional[str] = None

    # Residual mode info
    v2_peak_index: Optional[int] = None
    v2_gini: Optional[float] = None
    v2_entropy: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "preset": self.preset.to_dict(),
            "lambda_stat": self.lambda_stat,
            "p_local": self.p_local,
            "chi2_rank1": self.chi2_rank1,
            "chi2_rank2": self.chi2_rank2,
            "ndof_rank1": self.ndof_rank1,
            "ndof_rank2": self.ndof_rank2,
            "fit_converged": self.fit_converged,
            "error": self.error,
            "v2_peak_index": self.v2_peak_index,
            "v2_gini": self.v2_gini,
            "v2_entropy": self.v2_entropy,
        }


@dataclass
class GlobalSignificance:
    """Global significance after look-elsewhere correction."""
    T_obs: float  # max Lambda across presets
    best_preset: str  # Preset achieving T_obs
    p_local_best: float  # Local p-value at best preset
    p_global: float  # Global p-value after correction
    n_presets: int
    n_bootstrap: int
    T_bootstrap: List[float] = field(default_factory=list)  # Bootstrap T values

    def to_dict(self) -> Dict[str, Any]:
        return {
            "T_obs": self.T_obs,
            "best_preset": self.best_preset,
            "p_local_best": self.p_local_best,
            "p_global": self.p_global,
            "n_presets": self.n_presets,
            "n_bootstrap": self.n_bootstrap,
            "T_bootstrap_percentiles": {
                "50": float(np.percentile(self.T_bootstrap, 50)) if self.T_bootstrap else 0,
                "90": float(np.percentile(self.T_bootstrap, 90)) if self.T_bootstrap else 0,
                "95": float(np.percentile(self.T_bootstrap, 95)) if self.T_bootstrap else 0,
                "99": float(np.percentile(self.T_bootstrap, 99)) if self.T_bootstrap else 0,
            },
        }


def _run_single_preset(args: Tuple) -> SweepResult:
    """Run analysis for a single preset. Worker function for parallelization."""
    preset, analysis_factory, n_bootstrap, seed = args

    try:
        # Create analysis instance with preset configuration
        analysis = analysis_factory(preset)

        # Run the analysis
        result = analysis.run(n_bootstrap=n_bootstrap, seed=seed)

        # Extract residual mode info if available
        v2_peak = None
        v2_gini = None
        v2_entropy = None
        if hasattr(result, 'residual_mode') and result.residual_mode:
            rm = result.residual_mode
            if rm.v2_localization:
                v2_peak = rm.v2_localization.peak_index
                v2_gini = rm.v2_localization.gini
                v2_entropy = rm.v2_localization.normalized_entropy

        return SweepResult(
            preset=preset,
            lambda_stat=result.lambda_stat,
            p_local=result.p_value,
            chi2_rank1=result.chi2_rank1,
            chi2_rank2=result.chi2_rank2,
            ndof_rank1=result.ndof_rank1,
            ndof_rank2=result.ndof_rank2,
            fit_converged=True,
            v2_peak_index=v2_peak,
            v2_gini=v2_gini,
            v2_entropy=v2_entropy,
        )

    except Exception as e:
        logger.error(f"Preset {preset.name} failed: {e}")
        return SweepResult(
            preset=preset,
            lambda_stat=0.0,
            p_local=1.0,
            chi2_rank1=0.0,
            chi2_rank2=0.0,
            ndof_rank1=0,
            ndof_rank2=0,
            fit_converged=False,
            error=str(e),
        )


def _run_global_bootstrap_sample(args: Tuple) -> float:
    """
    Run a single global bootstrap sample.

    For each preset, generate pseudo-data from rank-1 null, fit both models,
    compute Lambda, then return max across presets.
    """
    boot_seed, presets, null_generator, fitter_factory = args

    rng = np.random.RandomState(boot_seed)
    lambdas = []

    for preset in presets:
        try:
            # Generate pseudo-data under null for this preset's observation set
            pseudo_data = null_generator(preset, rng)

            # Fit rank-1 and rank-2
            fitter = fitter_factory(preset)
            fit1 = fitter.fit_rank1(*pseudo_data)
            fit2 = fitter.fit_rank2(*pseudo_data)

            lambda_b = fit1.chi2 - fit2.chi2
            lambdas.append(lambda_b)

        except Exception:
            # If a preset fails, use 0 (conservative)
            lambdas.append(0.0)

    return max(lambdas) if lambdas else 0.0


class SweepRunner:
    """
    Run predefined sweeps with parallelization and global correction.
    """

    def __init__(
        self,
        dataset: str,
        presets: Optional[List[SweepPreset]] = None,
        n_jobs: Optional[int] = None,
        output_dir: Optional[Path] = None,
    ):
        """
        Initialize sweep runner.

        Args:
            dataset: Dataset name
            presets: List of presets (uses defaults if None)
            n_jobs: Number of parallel jobs
            output_dir: Output directory for results
        """
        self.dataset = dataset
        self.presets = presets or get_presets_for_dataset(dataset)
        self.n_jobs = n_jobs or max(1, cpu_count() - 1)
        self.output_dir = output_dir

    def run_all_presets(
        self,
        analysis_factory: Callable[[SweepPreset], Any],
        n_bootstrap: int = 500,
        seed: int = 42,
    ) -> List[SweepResult]:
        """
        Run analysis for all presets in parallel.

        Args:
            analysis_factory: Factory function that takes a preset and returns an analysis object
            n_bootstrap: Bootstrap samples per preset
            seed: Random seed

        Returns:
            List of SweepResult for each preset
        """
        logger.info(f"Running {len(self.presets)} sweep presets for {self.dataset}")

        # Prepare arguments
        args_list = [
            (preset, analysis_factory, n_bootstrap, seed + i)
            for i, preset in enumerate(self.presets)
        ]

        # Run in parallel
        with Pool(self.n_jobs) as pool:
            results = pool.map(_run_single_preset, args_list)

        # Log summary
        n_success = sum(1 for r in results if r.fit_converged)
        logger.info(f"  {n_success}/{len(results)} presets succeeded")

        return results

    def compute_global_significance(
        self,
        sweep_results: List[SweepResult],
        null_generator: Callable[[SweepPreset, np.random.RandomState], Tuple],
        fitter_factory: Callable[[SweepPreset], Any],
        n_global_bootstrap: int = 1000,
        seed: int = 42,
    ) -> GlobalSignificance:
        """
        Compute global significance with look-elsewhere correction.

        Args:
            sweep_results: Results from run_all_presets
            null_generator: Function to generate pseudo-data under null
            fitter_factory: Function to create fitter for a preset
            n_global_bootstrap: Number of global bootstrap samples
            seed: Random seed

        Returns:
            GlobalSignificance with corrected p-value
        """
        # Find observed max Lambda
        valid_results = [r for r in sweep_results if r.fit_converged]
        if not valid_results:
            return GlobalSignificance(
                T_obs=0.0,
                best_preset="none",
                p_local_best=1.0,
                p_global=1.0,
                n_presets=len(self.presets),
                n_bootstrap=0,
            )

        best_result = max(valid_results, key=lambda r: r.lambda_stat)
        T_obs = best_result.lambda_stat
        best_preset = best_result.preset.name
        p_local_best = best_result.p_local

        logger.info(f"Computing global significance: T_obs={T_obs:.2f} at preset '{best_preset}'")
        logger.info(f"  Running {n_global_bootstrap} global bootstrap samples...")

        # Prepare arguments for parallel bootstrap
        args_list = [
            (seed + i, self.presets, null_generator, fitter_factory)
            for i in range(n_global_bootstrap)
        ]

        # Run bootstrap in parallel
        with Pool(self.n_jobs) as pool:
            T_bootstrap = pool.map(_run_global_bootstrap_sample, args_list)

        # Compute global p-value
        k = sum(1 for t in T_bootstrap if t >= T_obs)
        p_global = (k + 1) / (n_global_bootstrap + 1)

        logger.info(f"  Global p-value: {p_global:.4f}")

        return GlobalSignificance(
            T_obs=T_obs,
            best_preset=best_preset,
            p_local_best=p_local_best,
            p_global=p_global,
            n_presets=len(self.presets),
            n_bootstrap=n_global_bootstrap,
            T_bootstrap=T_bootstrap,
        )

    def save_results(
        self,
        sweep_results: List[SweepResult],
        global_sig: Optional[GlobalSignificance] = None,
        filename: str = "sweep_results.json",
    ) -> Path:
        """Save sweep results to JSON."""
        if self.output_dir is None:
            raise ValueError("output_dir must be set to save results")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / filename

        data = {
            "dataset": self.dataset,
            "n_presets": len(self.presets),
            "results": [r.to_dict() for r in sweep_results],
            "global_significance": global_sig.to_dict() if global_sig else None,
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Sweep results saved: {output_path}")
        return output_path


@dataclass
class SweepSummary:
    """Summary statistics across sweep presets."""
    n_total: int
    n_converged: int
    lambda_range: Tuple[float, float]  # (min, max)
    p_local_range: Tuple[float, float]
    best_preset: str
    best_lambda: float
    best_p_local: float

    # Stability indicators
    lambda_std: float
    p_local_std: float
    all_agree_significance: bool  # Do all presets agree on significance at 0.05?

    # Peak consistency (for residual mode)
    peak_indices: List[int]
    peak_mode: Optional[int]
    peak_consistency: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_total": self.n_total,
            "n_converged": self.n_converged,
            "lambda_range": list(self.lambda_range),
            "p_local_range": list(self.p_local_range),
            "best_preset": self.best_preset,
            "best_lambda": self.best_lambda,
            "best_p_local": self.best_p_local,
            "lambda_std": self.lambda_std,
            "p_local_std": self.p_local_std,
            "all_agree_significance": self.all_agree_significance,
            "peak_indices": self.peak_indices,
            "peak_mode": self.peak_mode,
            "peak_consistency": self.peak_consistency,
        }


def compute_sweep_summary(results: List[SweepResult]) -> SweepSummary:
    """Compute summary statistics from sweep results."""
    valid = [r for r in results if r.fit_converged]

    if not valid:
        return SweepSummary(
            n_total=len(results),
            n_converged=0,
            lambda_range=(0.0, 0.0),
            p_local_range=(1.0, 1.0),
            best_preset="none",
            best_lambda=0.0,
            best_p_local=1.0,
            lambda_std=0.0,
            p_local_std=0.0,
            all_agree_significance=True,
            peak_indices=[],
            peak_mode=None,
            peak_consistency=0.0,
        )

    lambdas = [r.lambda_stat for r in valid]
    p_locals = [r.p_local for r in valid]

    best = max(valid, key=lambda r: r.lambda_stat)

    # Significance agreement
    significances = [r.p_local < 0.05 for r in valid]
    all_agree = len(set(significances)) == 1

    # Peak consistency
    peaks = [r.v2_peak_index for r in valid if r.v2_peak_index is not None]
    if peaks:
        from collections import Counter
        peak_counts = Counter(peaks)
        mode_peak, mode_count = peak_counts.most_common(1)[0]
        peak_consistency = mode_count / len(peaks)
    else:
        mode_peak = None
        peak_consistency = 0.0

    return SweepSummary(
        n_total=len(results),
        n_converged=len(valid),
        lambda_range=(min(lambdas), max(lambdas)),
        p_local_range=(min(p_locals), max(p_locals)),
        best_preset=best.preset.name,
        best_lambda=best.lambda_stat,
        best_p_local=best.p_local,
        lambda_std=float(np.std(lambdas)),
        p_local_std=float(np.std(p_locals)),
        all_agree_significance=all_agree,
        peak_indices=peaks,
        peak_mode=mode_peak,
        peak_consistency=peak_consistency,
    )
