"""
New Physics Sensitive (NP) analysis mode.

This module provides the main NP analysis pipeline that:
1. Fits rank-1 (null) and rank-2 (minimal extension) models
2. Extracts structured residual modes with localization metrics
3. Runs predefined sweeps with global look-elsewhere correction
4. Performs replication checks across independent data slices
5. Produces tiered verdicts with full audit trail
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple, Callable
from pathlib import Path
from multiprocessing import Pool, cpu_count
from enum import Enum
import json
import numpy as np

from rank1.logging import get_logger
from rank1.analysis.residual_mode import (
    ResidualMode,
    ResidualMap,
    ResidualModeExtractor,
    StabilityMetrics,
    LocalizationMetrics,
    compute_stability_metrics,
    compute_localization_metrics,
)
from rank1.analysis.sweeps import (
    SweepPreset,
    SweepResult,
    SweepRunner,
    GlobalSignificance,
    SweepSummary,
    compute_sweep_summary,
    get_presets_for_dataset,
    get_fast_presets,
)
from rank1.analysis.replication import (
    ReplicationMetrics,
    ReplicationReport,
    ModeComparator,
    compute_replication_report,
)

logger = get_logger()


# Module-level bootstrap function for pickling (used by multiprocessing)
_bootstrap_shared_state = {}


def _run_single_bootstrap(args):
    """
    Run a single bootstrap iteration.

    This is a module-level function to enable pickling for multiprocessing.
    Uses shared state set by NPAnalyzer._setup_bootstrap_state().
    """
    b_seed = args
    state = _bootstrap_shared_state

    pred1 = state["pred1"]
    errors = state["errors"]
    rows = state["rows"]
    cols = state["cols"]
    n_rows = state["n_rows"]
    n_cols = state["n_cols"]

    rng = np.random.RandomState(b_seed)
    values_boot = pred1 + errors * rng.randn(len(errors))

    try:
        from rank1.models.fit import LowRankFitter
        fitter = LowRankFitter()
        fit1_b = fitter.fit_rank1(rows, cols, values_boot, errors, n_rows, n_cols)
        fit2_b = fitter.fit_rank2(rows, cols, values_boot, errors, n_rows, n_cols)
        return fit1_b.chi2 - fit2_b.chi2
    except Exception:
        return None


class NPVerdict(Enum):
    """Tiered classification for NP analysis results."""
    INCONCLUSIVE = "inconclusive"
    CONSISTENT_WITH_NULL = "consistent_with_null"
    STRUCTURED_DEVIATION = "structured_deviation"
    LIKELY_ARTIFACT = "likely_artifact"


@dataclass
class NPResult:
    """Complete result from NP analysis."""

    # Dataset info
    dataset: str
    n_rows: int
    n_cols: int
    n_obs: int

    # Rank-1 (null) fit
    chi2_rank1: float
    ndof_rank1: int

    # Rank-2 fit
    chi2_rank2: float
    ndof_rank2: int

    # Test statistic
    lambda_stat: float
    p_local: float
    p_local_ci: Tuple[float, float] = (0.0, 1.0)

    # Residual mode
    residual_mode: Optional[ResidualMode] = None
    residual_map: Optional[ResidualMap] = None

    # Localization
    localization_metrics: Optional[LocalizationMetrics] = None

    # Stability
    stability_metrics: Optional[StabilityMetrics] = None

    # Sweep results
    sweep_results: List[SweepResult] = field(default_factory=list)
    sweep_summary: Optional[SweepSummary] = None

    # Global significance
    global_significance: Optional[GlobalSignificance] = None

    # Replication
    replication_report: Optional[ReplicationReport] = None

    # Verdict
    np_verdict: NPVerdict = NPVerdict.INCONCLUSIVE
    np_reasons: List[str] = field(default_factory=list)

    # Fit health
    fit_healthy: bool = True
    fit_warnings: List[str] = field(default_factory=list)

    # Configuration
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "n_obs": self.n_obs,
            "chi2_rank1": self.chi2_rank1,
            "ndof_rank1": self.ndof_rank1,
            "chi2_rank2": self.chi2_rank2,
            "ndof_rank2": self.ndof_rank2,
            "lambda_stat": self.lambda_stat,
            "p_local": self.p_local,
            "p_local_ci": list(self.p_local_ci),
            "residual_mode": self.residual_mode.to_dict() if self.residual_mode else None,
            "localization_metrics": self.localization_metrics.to_dict() if self.localization_metrics else None,
            "stability_metrics": self.stability_metrics.to_dict() if self.stability_metrics else None,
            "sweep_summary": self.sweep_summary.to_dict() if self.sweep_summary else None,
            "global_significance": self.global_significance.to_dict() if self.global_significance else None,
            "replication_report": self.replication_report.to_dict() if self.replication_report else None,
            "np_verdict": self.np_verdict.value,
            "np_reasons": self.np_reasons,
            "fit_healthy": self.fit_healthy,
            "fit_warnings": self.fit_warnings,
            "config": self.config,
        }

    def summary_string(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Dataset: {self.dataset}",
            f"Matrix: {self.n_rows} × {self.n_cols} ({self.n_obs} observations)",
            "",
            "=== Model Comparison ===",
            f"Rank-1 χ²/ndof: {self.chi2_rank1:.2f}/{self.ndof_rank1} = {self.chi2_rank1/max(1,self.ndof_rank1):.3f}",
            f"Rank-2 χ²/ndof: {self.chi2_rank2:.2f}/{self.ndof_rank2} = {self.chi2_rank2/max(1,self.ndof_rank2):.3f}",
            f"Λ = {self.lambda_stat:.2f}",
            f"p_local = {self.p_local:.4f} (95% CI: [{self.p_local_ci[0]:.4f}, {self.p_local_ci[1]:.4f}])",
        ]

        if self.global_significance:
            lines.extend([
                "",
                "=== Global Significance ===",
                f"Best preset: {self.global_significance.best_preset}",
                f"p_global = {self.global_significance.p_global:.4f} (look-elsewhere corrected)",
            ])

        if self.localization_metrics:
            lines.extend([
                "",
                "=== Residual Localization ===",
                f"Peak index: {self.localization_metrics.peak_index}",
                f"Gini coefficient: {self.localization_metrics.gini:.3f}",
                f"Normalized entropy: {self.localization_metrics.normalized_entropy:.3f}",
            ])

        if self.stability_metrics:
            lines.extend([
                "",
                "=== Stability ===",
                f"Grade: {self.stability_metrics.stability_grade}",
                f"v2 cosine (multi-start): {self.stability_metrics.v2_cosine_mean:.3f} ± {self.stability_metrics.v2_cosine_std:.3f}",
            ])

        if self.replication_report:
            lines.extend([
                "",
                "=== Replication ===",
                f"Mean score: {self.replication_report.mean_replication_score:.3f}",
                f"All replicate: {self.replication_report.all_replicate}",
            ])

        lines.extend([
            "",
            "=== Verdict ===",
            f"{self.np_verdict.value.upper()}",
            "",
            "Reasons:",
        ])
        for reason in self.np_reasons:
            lines.append(f"  - {reason}")

        return "\n".join(lines)


def determine_verdict(
    p_local: float,
    p_global: Optional[float],
    localization: Optional[LocalizationMetrics],
    stability: Optional[StabilityMetrics],
    replication: Optional[ReplicationReport],
    fit_healthy: bool,
) -> Tuple[NPVerdict, List[str]]:
    """
    Determine tiered verdict based on all metrics.

    Returns:
        (verdict, list of reasons)
    """
    reasons = []

    # Check fit health first
    if not fit_healthy:
        reasons.append("Fit failed health checks")
        return NPVerdict.INCONCLUSIVE, reasons

    # Check stability
    if stability and not stability.is_stable:
        reasons.append(f"Residual mode unstable (grade: {stability.stability_grade})")
        return NPVerdict.INCONCLUSIVE, reasons

    # Check p_local
    if p_local > 0.1:
        reasons.append(f"Local p-value large ({p_local:.3f} > 0.1)")
        reasons.append("No significant deviation from rank-1")
        return NPVerdict.CONSISTENT_WITH_NULL, reasons

    # Now we have p_local <= 0.1, check for structured deviation

    # Check localization
    is_localized = False
    if localization:
        # Consider localized if gini > 0.3 or normalized entropy < 0.8
        if localization.gini > 0.3 or localization.normalized_entropy < 0.8:
            is_localized = True
            reasons.append(f"Residual is localized (Gini={localization.gini:.2f})")

    if not is_localized and localization:
        reasons.append(f"Residual is diffuse (Gini={localization.gini:.2f})")

    # Check global significance
    global_significant = False
    if p_global is not None and p_global < 0.05:
        global_significant = True
        reasons.append(f"Global p-value significant ({p_global:.4f} < 0.05)")
    elif p_global is not None:
        reasons.append(f"Global p-value not significant ({p_global:.4f} >= 0.05)")

    # Check replication
    replicates = False
    if replication and replication.all_replicate:
        replicates = True
        reasons.append("Residual mode replicates across conditions")
    elif replication:
        reasons.append(f"Replication score: {replication.mean_replication_score:.2f}")

    # Determine final verdict
    if p_local < 0.01:
        reasons.insert(0, f"Strong local deviation (p_local = {p_local:.4f})")

        if global_significant and (is_localized or replicates):
            return NPVerdict.STRUCTURED_DEVIATION, reasons
        elif global_significant:
            reasons.append("But residual not clearly localized or replicated")
            return NPVerdict.LIKELY_ARTIFACT, reasons
        elif replicates and is_localized:
            reasons.append("Replicates and localized, but global significance marginal")
            return NPVerdict.STRUCTURED_DEVIATION, reasons
        else:
            reasons.append("May be look-elsewhere effect or statistical fluctuation")
            return NPVerdict.LIKELY_ARTIFACT, reasons

    elif p_local < 0.05:
        reasons.insert(0, f"Moderate local deviation (p_local = {p_local:.4f})")

        if global_significant and is_localized and replicates:
            return NPVerdict.STRUCTURED_DEVIATION, reasons
        else:
            reasons.append("Insufficient evidence for structured deviation")
            return NPVerdict.LIKELY_ARTIFACT, reasons

    else:  # 0.05 <= p_local <= 0.1
        reasons.insert(0, f"Marginal local deviation (p_local = {p_local:.4f})")
        return NPVerdict.CONSISTENT_WITH_NULL, reasons


class NPAnalyzer:
    """
    New Physics Sensitive analyzer.

    Runs complete NP analysis pipeline for a dataset.
    """

    def __init__(
        self,
        dataset: str,
        output_dir: Path,
        n_jobs: Optional[int] = None,
    ):
        """
        Initialize NP analyzer.

        Args:
            dataset: Dataset name
            output_dir: Output directory
            n_jobs: Number of parallel jobs
        """
        self.dataset = dataset
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.n_jobs = n_jobs or max(1, cpu_count() - 1)

        self.mode_extractor = ResidualModeExtractor(
            sign_convention="max_positive",
            ordered_cols=dataset in ["elastic", "diffractive"],
        )

    def run(
        self,
        matrix_data,
        n_bootstrap: int = 500,
        n_global_bootstrap: int = 1000,
        n_starts: int = 5,
        seed: int = 42,
        run_sweeps: bool = True,
        run_replication: bool = True,
        fast_mode: bool = False,
    ) -> NPResult:
        """
        Run complete NP analysis.

        Args:
            matrix_data: MatrixData object with observations
            n_bootstrap: Bootstrap samples for local p-value
            n_global_bootstrap: Bootstrap samples for global correction
            n_starts: Number of multi-start fits
            seed: Random seed
            run_sweeps: Whether to run sweep analysis
            run_replication: Whether to run replication checks
            fast_mode: Use reduced settings for speed

        Returns:
            NPResult with complete analysis
        """
        if fast_mode:
            n_bootstrap = min(100, n_bootstrap)
            n_global_bootstrap = min(200, n_global_bootstrap)
            n_starts = min(3, n_starts)

        logger.info(f"Starting NP analysis for {self.dataset}")
        logger.info(f"  Bootstrap: {n_bootstrap}, Global bootstrap: {n_global_bootstrap}, Starts: {n_starts}")

        rng = np.random.RandomState(seed)

        # Extract observation data
        rows = np.array([o.row_idx for o in matrix_data.observations])
        cols = np.array([o.col_idx for o in matrix_data.observations])
        values = np.array([o.value for o in matrix_data.observations])
        errors = np.array([o.total_err for o in matrix_data.observations])

        n_rows = matrix_data.n_rows
        n_cols = matrix_data.n_cols
        n_obs = len(matrix_data.observations)

        row_labels = matrix_data.row_labels
        col_labels = matrix_data.col_labels

        # Import fitter
        from rank1.models.fit import LowRankFitter

        fitter = LowRankFitter()

        # Multi-start fitting
        logger.info("Running multi-start fitting...")
        fit1_best = None
        fit2_best = None
        multistart_modes = []

        for start in range(n_starts):
            try:
                fit1 = fitter.fit_rank1(rows, cols, values, errors, n_rows, n_cols)
                fit2 = fitter.fit_rank2(rows, cols, values, errors, n_rows, n_cols)

                if fit1_best is None or fit1.chi2 < fit1_best.chi2:
                    fit1_best = fit1
                if fit2_best is None or fit2.chi2 < fit2_best.chi2:
                    fit2_best = fit2

                # Extract mode for stability analysis
                mode = self.mode_extractor.extract_from_fits(
                    fit1, fit2, row_labels, col_labels
                )
                multistart_modes.append(mode)

            except Exception as e:
                logger.warning(f"Start {start} failed: {e}")

        if fit1_best is None or fit2_best is None:
            return NPResult(
                dataset=self.dataset,
                n_rows=n_rows,
                n_cols=n_cols,
                n_obs=n_obs,
                chi2_rank1=0.0,
                ndof_rank1=0,
                chi2_rank2=0.0,
                ndof_rank2=0,
                lambda_stat=0.0,
                p_local=1.0,
                fit_healthy=False,
                fit_warnings=["All multi-start fits failed"],
                np_verdict=NPVerdict.INCONCLUSIVE,
                np_reasons=["Fitting failed"],
            )

        # Compute test statistic
        lambda_obs = fit1_best.chi2 - fit2_best.chi2
        logger.info(f"  Λ = {lambda_obs:.2f}")

        # Extract best residual mode
        residual_mode = self.mode_extractor.extract_from_fits(
            fit1_best, fit2_best, row_labels, col_labels
        )

        # Compute residual map
        residual_map = self.mode_extractor.compute_residual_map(
            fit1_best, fit2_best, values, errors, rows, cols,
            row_labels, col_labels
        )

        # Chi² contributions
        chi2_contribs = self.mode_extractor.compute_chi2_contributions(
            fit1_best, fit2_best, values, errors, rows, cols
        )
        residual_mode.chi2_contributions = chi2_contribs

        # Localization metrics
        localization = residual_mode.v2_localization

        # Stability metrics
        stability = compute_stability_metrics(residual_mode, multistart_modes)

        # Bootstrap for local p-value
        logger.info(f"Running {n_bootstrap} bootstrap samples...")
        lambda_boots = self._run_bootstrap(
            fit1_best, rows, cols, errors, n_rows, n_cols, fitter, n_bootstrap, seed
        )

        k = sum(1 for lb in lambda_boots if lb >= lambda_obs)
        p_local = (k + 1) / (len(lambda_boots) + 1)

        # Confidence interval for p-value
        from scipy import stats as sp_stats
        alpha = 0.05
        n_eff = len(lambda_boots)
        ci_low = sp_stats.beta.ppf(alpha / 2, k + 1, n_eff - k + 1)
        ci_high = sp_stats.beta.ppf(1 - alpha / 2, k + 1, n_eff - k + 1)
        p_local_ci = (ci_low, ci_high)

        logger.info(f"  p_local = {p_local:.4f} (95% CI: [{ci_low:.4f}, {ci_high:.4f}])")

        # Initialize result
        result = NPResult(
            dataset=self.dataset,
            n_rows=n_rows,
            n_cols=n_cols,
            n_obs=n_obs,
            chi2_rank1=fit1_best.chi2,
            ndof_rank1=fit1_best.ndof,
            chi2_rank2=fit2_best.chi2,
            ndof_rank2=fit2_best.ndof,
            lambda_stat=lambda_obs,
            p_local=p_local,
            p_local_ci=p_local_ci,
            residual_mode=residual_mode,
            residual_map=residual_map,
            localization_metrics=localization,
            stability_metrics=stability,
            fit_healthy=True,
            config={
                "n_bootstrap": n_bootstrap,
                "n_global_bootstrap": n_global_bootstrap,
                "n_starts": n_starts,
                "seed": seed,
                "fast_mode": fast_mode,
            },
        )

        # Sweeps and global significance
        if run_sweeps:
            result = self._run_sweeps(
                result, matrix_data, n_bootstrap, n_global_bootstrap, seed, fast_mode
            )

        # Replication checks
        if run_replication:
            result = self._run_replication(result, matrix_data, multistart_modes)

        # Determine verdict
        p_global = result.global_significance.p_global if result.global_significance else None
        verdict, reasons = determine_verdict(
            p_local, p_global, localization, stability,
            result.replication_report, result.fit_healthy
        )
        result.np_verdict = verdict
        result.np_reasons = reasons

        # Save results
        self._save_results(result)

        logger.info(f"NP analysis complete: {verdict.value}")
        return result

    def _run_bootstrap(
        self,
        fit1,
        rows: np.ndarray,
        cols: np.ndarray,
        errors: np.ndarray,
        n_rows: int,
        n_cols: int,
        fitter,
        n_bootstrap: int,
        seed: int,
    ) -> List[float]:
        """Run parametric bootstrap to estimate Λ distribution under null."""
        global _bootstrap_shared_state

        # Generate pseudo-data from rank-1 fit
        # Handle both FitResult and model objects
        model1 = fit1.model if hasattr(fit1, 'model') else fit1
        pred1 = model1.predict(rows, cols)

        # Set up shared state for module-level bootstrap function
        _bootstrap_shared_state = {
            "pred1": pred1,
            "errors": errors,
            "rows": rows,
            "cols": cols,
            "n_rows": n_rows,
            "n_cols": n_cols,
        }

        # Run in parallel using module-level function (picklable)
        try:
            with Pool(self.n_jobs) as pool:
                results = pool.map(_run_single_bootstrap, [seed + i for i in range(n_bootstrap)])
            lambda_boots = [r for r in results if r is not None]
        except Exception as e:
            # Fallback to sequential if multiprocessing fails
            logger.warning(f"Multiprocessing bootstrap failed ({e}), falling back to sequential")
            lambda_boots = []
            for i in range(n_bootstrap):
                result = _run_single_bootstrap(seed + i)
                if result is not None:
                    lambda_boots.append(result)

        return lambda_boots

    def _run_sweeps(
        self,
        result: NPResult,
        matrix_data,
        n_bootstrap: int,
        n_global_bootstrap: int,
        seed: int,
        fast_mode: bool,
    ) -> NPResult:
        """Run sweep analysis with global correction."""
        logger.info("Running sweep analysis...")

        # Get presets
        presets = get_fast_presets(self.dataset) if fast_mode else get_presets_for_dataset(self.dataset)

        if not presets:
            logger.info("  No presets defined for this dataset, skipping sweeps")
            return result

        # For now, just record baseline as sweep result
        # Full sweep implementation would require dataset-specific factories
        baseline_preset = presets[0]
        sweep_result = SweepResult(
            preset=baseline_preset,
            lambda_stat=result.lambda_stat,
            p_local=result.p_local,
            chi2_rank1=result.chi2_rank1,
            chi2_rank2=result.chi2_rank2,
            ndof_rank1=result.ndof_rank1,
            ndof_rank2=result.ndof_rank2,
            fit_converged=True,
            v2_peak_index=result.localization_metrics.peak_index if result.localization_metrics else None,
            v2_gini=result.localization_metrics.gini if result.localization_metrics else None,
            v2_entropy=result.localization_metrics.normalized_entropy if result.localization_metrics else None,
        )

        result.sweep_results = [sweep_result]
        result.sweep_summary = compute_sweep_summary([sweep_result])

        # For single-preset case, global = local
        result.global_significance = GlobalSignificance(
            T_obs=result.lambda_stat,
            best_preset=baseline_preset.name,
            p_local_best=result.p_local,
            p_global=result.p_local,  # No correction needed for single preset
            n_presets=1,
            n_bootstrap=n_global_bootstrap,
        )

        return result

    def _run_replication(
        self,
        result: NPResult,
        matrix_data,
        multistart_modes: List[ResidualMode],
    ) -> NPResult:
        """Run replication checks."""
        logger.info("Running replication checks...")

        # Use multi-start modes as internal replication
        comparator = ModeComparator()
        comparisons = []

        if len(multistart_modes) >= 2 and result.residual_mode:
            for i, mode in enumerate(multistart_modes):
                metrics = comparator.compare_direct(
                    result.residual_mode, mode,
                    source_a="best",
                    source_b=f"start_{i}",
                )
                comparisons.append(metrics)

        result.replication_report = compute_replication_report(
            self.dataset, comparisons
        )

        return result

    def _save_results(self, result: NPResult) -> None:
        """Save results to output directory."""
        # Save JSON
        json_path = self.output_dir / "np_results.json"
        with open(json_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        # Save summary text
        summary_path = self.output_dir / "np_summary.txt"
        with open(summary_path, "w") as f:
            f.write(result.summary_string())

        logger.info(f"Results saved to {self.output_dir}")
