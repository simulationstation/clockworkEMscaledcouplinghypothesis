"""
Statistical calibration for the rank-1 test.

This module provides:
1. Null uniformity checks: Verify p-values are uniform under the null
2. Power analysis: Measure detection power for rank-2 alternatives

These calibrations verify that the statistical procedure is well-calibrated
and has adequate power to detect true departures from rank-1.
"""

from dataclasses import dataclass, field
from typing import Any, Optional
from multiprocessing import Pool, cpu_count
import warnings

import numpy as np
from scipy import stats

from rank1.logging import get_logger
from rank1.models.lowrank import Rank1Model, Rank2Model
from rank1.models.fit import LowRankFitter

logger = get_logger()


@dataclass
class CalibrationReport:
    """Report from calibration analysis."""
    n_simulations: int
    seed: int
    null_passed: bool
    power_adequate: bool
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_simulations": self.n_simulations,
            "seed": self.seed,
            "null_passed": self.null_passed,
            "power_adequate": self.power_adequate,
            **self.details,
        }


def _run_single_null_sim(args: tuple) -> float:
    """Run a single null simulation and return p-value."""
    seed, n_rows, n_cols, noise_level, n_bootstrap = args

    rng = np.random.RandomState(seed)

    # Generate true rank-1 data
    u_true = rng.randn(n_rows)
    v_true = rng.randn(n_cols)
    v_true = v_true / np.linalg.norm(v_true)

    M_true = np.outer(u_true, v_true)
    errors = noise_level * np.ones((n_rows, n_cols))
    M_obs = M_true + errors * rng.randn(n_rows, n_cols)

    # Create observation arrays
    rows, cols = np.meshgrid(range(n_rows), range(n_cols), indexing='ij')
    rows = rows.flatten()
    cols = cols.flatten()
    values = M_obs.flatten()
    errs = errors.flatten()

    # Fit rank-1 and rank-2
    fitter = LowRankFitter()

    try:
        fit1 = fitter.fit_rank1(rows, cols, values, errs)
        fit2 = fitter.fit_rank2(rows, cols, values, errs)

        lambda_obs = fit1.chi2 - fit2.chi2
    except Exception:
        return np.nan

    # Bootstrap under null
    lambda_boots = []
    for b in range(n_bootstrap):
        boot_rng = np.random.RandomState(seed * 1000 + b)

        # Generate from fitted rank-1
        M_boot = fit1.predict()
        values_boot = M_boot.flatten() + errs * boot_rng.randn(len(errs))

        try:
            fit1_b = fitter.fit_rank1(rows, cols, values_boot, errs)
            fit2_b = fitter.fit_rank2(rows, cols, values_boot, errs)
            lambda_boots.append(fit1_b.chi2 - fit2_b.chi2)
        except Exception:
            continue

    if len(lambda_boots) < n_bootstrap // 2:
        return np.nan

    k = sum(1 for lb in lambda_boots if lb >= lambda_obs)
    p_value = (k + 1) / (len(lambda_boots) + 1)

    return p_value


class NullCalibrator:
    """
    Verify null uniformity of p-values.

    Under the null hypothesis (true rank-1), p-values should be
    uniformly distributed on [0, 1].
    """

    def __init__(
        self,
        n_simulations: int = 200,
        n_bootstrap: int = 200,
        n_rows: int = 5,
        n_cols: int = 6,
        noise_level: float = 0.1,
        alpha: float = 0.05,
        seed: int = 42,
        n_jobs: Optional[int] = None,
    ):
        self.n_simulations = n_simulations
        self.n_bootstrap = n_bootstrap
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.noise_level = noise_level
        self.alpha = alpha
        self.seed = seed
        self.n_jobs = n_jobs or max(1, cpu_count() - 1)

    def run(self) -> CalibrationReport:
        """Run null calibration."""
        logger.info(
            f"Running null calibration: {self.n_simulations} simulations, "
            f"{self.n_bootstrap} bootstraps each"
        )

        # Prepare arguments for parallel runs
        args_list = [
            (self.seed + i, self.n_rows, self.n_cols, self.noise_level, self.n_bootstrap)
            for i in range(self.n_simulations)
        ]

        # Run simulations in parallel
        with Pool(self.n_jobs) as pool:
            p_values = pool.map(_run_single_null_sim, args_list)

        p_values = np.array([p for p in p_values if not np.isnan(p)])

        if len(p_values) < self.n_simulations // 2:
            logger.error(f"Too many failed simulations: {self.n_simulations - len(p_values)}")
            return CalibrationReport(
                n_simulations=self.n_simulations,
                seed=self.seed,
                null_passed=False,
                power_adequate=False,
                details={"error": "Too many failed simulations"},
            )

        # Test for uniformity using Kolmogorov-Smirnov
        ks_stat, ks_pvalue = stats.kstest(p_values, 'uniform')

        # Check if rejection rate at alpha is calibrated
        rejection_rate = np.mean(p_values < self.alpha)
        rejection_se = np.sqrt(self.alpha * (1 - self.alpha) / len(p_values))

        # Allow 3 SE deviation
        rate_calibrated = abs(rejection_rate - self.alpha) < 3 * rejection_se

        # Overall pass: KS test doesn't reject AND rejection rate is calibrated
        null_passed = ks_pvalue > 0.01 and rate_calibrated

        logger.info(f"  KS test: D={ks_stat:.4f}, p={ks_pvalue:.4f}")
        logger.info(f"  Rejection rate at α={self.alpha}: {rejection_rate:.3f} (expected: {self.alpha})")
        logger.info(f"  Null calibration: {'PASSED' if null_passed else 'FAILED'}")

        return CalibrationReport(
            n_simulations=len(p_values),
            seed=self.seed,
            null_passed=null_passed,
            power_adequate=True,  # Will be set by power analysis
            details={
                "ks_statistic": float(ks_stat),
                "ks_pvalue": float(ks_pvalue),
                "rejection_rate": float(rejection_rate),
                "expected_rate": self.alpha,
                "rate_se": float(rejection_se),
                "n_successful": len(p_values),
                "p_values": p_values.tolist(),
            },
        )


def _run_single_power_sim(args: tuple) -> float:
    """Run a single power simulation and return p-value."""
    seed, n_rows, n_cols, noise_level, rank2_strength, n_bootstrap = args

    rng = np.random.RandomState(seed)

    # Generate rank-2 data (alternative hypothesis)
    u1 = rng.randn(n_rows)
    v1 = rng.randn(n_cols)
    v1 = v1 / np.linalg.norm(v1)

    u2 = rng.randn(n_rows)
    v2 = rng.randn(n_cols)
    v2 = v2 / np.linalg.norm(v2)

    # Mix: mostly rank-1 with small rank-2 component
    M_true = np.outer(u1, v1) + rank2_strength * np.outer(u2, v2)

    errors = noise_level * np.ones((n_rows, n_cols))
    M_obs = M_true + errors * rng.randn(n_rows, n_cols)

    # Create observation arrays
    rows, cols = np.meshgrid(range(n_rows), range(n_cols), indexing='ij')
    rows = rows.flatten()
    cols = cols.flatten()
    values = M_obs.flatten()
    errs = errors.flatten()

    # Fit models
    fitter = LowRankFitter()

    try:
        fit1 = fitter.fit_rank1(rows, cols, values, errs)
        fit2 = fitter.fit_rank2(rows, cols, values, errs)

        lambda_obs = fit1.chi2 - fit2.chi2
    except Exception:
        return np.nan

    # Bootstrap under null (rank-1)
    lambda_boots = []
    for b in range(n_bootstrap):
        boot_rng = np.random.RandomState(seed * 1000 + b)

        M_boot = fit1.predict()
        values_boot = M_boot.flatten() + errs * boot_rng.randn(len(errs))

        try:
            fit1_b = fitter.fit_rank1(rows, cols, values_boot, errs)
            fit2_b = fitter.fit_rank2(rows, cols, values_boot, errs)
            lambda_boots.append(fit1_b.chi2 - fit2_b.chi2)
        except Exception:
            continue

    if len(lambda_boots) < n_bootstrap // 2:
        return np.nan

    k = sum(1 for lb in lambda_boots if lb >= lambda_obs)
    p_value = (k + 1) / (len(lambda_boots) + 1)

    return p_value


class PowerAnalyzer:
    """
    Analyze power to detect rank-2 alternatives.

    Measures the probability of rejecting the null when the true
    data is rank-2 with varying signal strengths.
    """

    def __init__(
        self,
        n_simulations: int = 100,
        n_bootstrap: int = 200,
        n_rows: int = 5,
        n_cols: int = 6,
        noise_level: float = 0.1,
        rank2_strengths: Optional[list[float]] = None,
        alpha: float = 0.05,
        seed: int = 42,
        n_jobs: Optional[int] = None,
    ):
        self.n_simulations = n_simulations
        self.n_bootstrap = n_bootstrap
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.noise_level = noise_level
        self.rank2_strengths = rank2_strengths or [0.1, 0.2, 0.3, 0.5]
        self.alpha = alpha
        self.seed = seed
        self.n_jobs = n_jobs or max(1, cpu_count() - 1)

    def run(self) -> CalibrationReport:
        """Run power analysis across different signal strengths."""
        logger.info(
            f"Running power analysis: {self.n_simulations} simulations per strength"
        )

        power_results = {}

        for strength in self.rank2_strengths:
            logger.info(f"  Rank-2 strength = {strength}...")

            args_list = [
                (self.seed + i, self.n_rows, self.n_cols,
                 self.noise_level, strength, self.n_bootstrap)
                for i in range(self.n_simulations)
            ]

            with Pool(self.n_jobs) as pool:
                p_values = pool.map(_run_single_power_sim, args_list)

            p_values = np.array([p for p in p_values if not np.isnan(p)])

            if len(p_values) > 0:
                power = np.mean(p_values < self.alpha)
                power_se = np.sqrt(power * (1 - power) / len(p_values))
            else:
                power = 0.0
                power_se = 0.0

            power_results[strength] = {
                "power": float(power),
                "power_se": float(power_se),
                "n_successful": len(p_values),
            }

            logger.info(f"    Power = {power:.3f} ± {power_se:.3f}")

        # Check if power is adequate at moderate signal
        moderate_strength = 0.3
        if moderate_strength in power_results:
            power_adequate = power_results[moderate_strength]["power"] > 0.5
        else:
            # Use closest available strength
            closest = min(self.rank2_strengths, key=lambda x: abs(x - moderate_strength))
            power_adequate = power_results[closest]["power"] > 0.5

        logger.info(f"  Power adequacy: {'PASSED' if power_adequate else 'FAILED'}")

        return CalibrationReport(
            n_simulations=self.n_simulations,
            seed=self.seed,
            null_passed=True,  # Set by null calibrator
            power_adequate=power_adequate,
            details={
                "power_curve": power_results,
                "alpha": self.alpha,
                "rank2_strengths": self.rank2_strengths,
            },
        )


def run_full_calibration(
    n_null_sims: int = 200,
    n_power_sims: int = 100,
    n_bootstrap: int = 200,
    seed: int = 42,
    n_jobs: Optional[int] = None,
) -> dict[str, CalibrationReport]:
    """Run complete calibration suite."""
    logger.info("=" * 60)
    logger.info("Statistical Calibration Suite")
    logger.info("=" * 60)

    results = {}

    # Null calibration
    logger.info("\n1. Null Uniformity Check")
    null_cal = NullCalibrator(
        n_simulations=n_null_sims,
        n_bootstrap=n_bootstrap,
        seed=seed,
        n_jobs=n_jobs,
    )
    results["null"] = null_cal.run()

    # Power analysis
    logger.info("\n2. Power Analysis")
    power_anal = PowerAnalyzer(
        n_simulations=n_power_sims,
        n_bootstrap=n_bootstrap,
        seed=seed + 10000,
        n_jobs=n_jobs,
    )
    results["power"] = power_anal.run()

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("CALIBRATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Null uniformity: {'PASSED' if results['null'].null_passed else 'FAILED'}")
    logger.info(f"  Power adequacy: {'PASSED' if results['power'].power_adequate else 'FAILED'}")

    return results
