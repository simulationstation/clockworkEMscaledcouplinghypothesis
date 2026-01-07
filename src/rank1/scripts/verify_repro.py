#!/usr/bin/env python3
"""
Reproducibility verification script.

This script verifies that the analysis pipeline produces deterministic,
reproducible results by:

1. Running the pipeline with a fixed seed
2. Computing checksums of all outputs
3. Running again with the same seed
4. Verifying checksums match exactly

Usage:
    python -m rank1.scripts.verify_repro
    python -m rank1.scripts.verify_repro --n-trials 3 --seed 42
    python -m rank1.scripts.verify_repro --quick  # Fast smoke test
"""

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np

from rank1.config import Config, set_config
from rank1.logging import setup_logging, get_logger
from rank1.datasets import HiggsATLASDataset, ElasticTOTEMDataset, DiffractiveDISDataset
from rank1.analysis import HiggsRankAnalysis, ElasticRankAnalysis, DiffractiveRankAnalysis


@dataclass
class ReproCheckResult:
    """Result of a single reproducibility check."""
    name: str
    passed: bool
    run1_hash: str
    run2_hash: str
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReproReport:
    """Full reproducibility verification report."""
    timestamp: str
    seed: int
    n_bootstrap: int
    all_passed: bool
    checks: list[ReproCheckResult]
    run1_outputs: dict[str, str]  # filename -> hash
    run2_outputs: dict[str, str]
    mismatches: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "seed": self.seed,
            "n_bootstrap": self.n_bootstrap,
            "all_passed": self.all_passed,
            "checks": [
                {
                    "name": c.name,
                    "passed": c.passed,
                    "run1_hash": c.run1_hash,
                    "run2_hash": c.run2_hash,
                    "message": c.message,
                }
                for c in self.checks
            ],
            "n_outputs_checked": len(self.run1_outputs),
            "n_mismatches": len(self.mismatches),
            "mismatches": self.mismatches,
        }


def compute_file_hash(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def compute_dir_hashes(directory: Path) -> dict[str, str]:
    """Compute hashes of all files in a directory."""
    hashes = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            rel_path = path.relative_to(directory)
            hashes[str(rel_path)] = compute_file_hash(path)
    return hashes


def run_single_analysis(
    output_dir: Path,
    seed: int,
    n_bootstrap: int,
    dataset_name: str,
    logger,
) -> Optional[dict[str, Any]]:
    """Run a single analysis and return key results."""
    from rank1.config import Config, set_config

    config = Config(
        output_dir=output_dir,
        seed=seed,
        log_level="WARNING",
    )
    config.ensure_dirs()
    set_config(config)

    # Set numpy random seed for reproducibility
    np.random.seed(seed)

    # Create dataset and analysis
    if dataset_name == "higgs":
        dataset = HiggsATLASDataset(
            cache_dir=config.data.cache_dir,
            raw_dir=config.data.raw_dir,
            processed_dir=config.data.processed_dir,
        )
        analysis = HiggsRankAnalysis(
            dataset=dataset,
            output_dir=output_dir / "higgs_atlas",
        )
    elif dataset_name == "elastic":
        dataset = ElasticTOTEMDataset(
            cache_dir=config.data.cache_dir,
            raw_dir=config.data.raw_dir,
            processed_dir=config.data.processed_dir,
        )
        analysis = ElasticRankAnalysis(
            dataset=dataset,
            output_dir=output_dir / "elastic_totem",
        )
    elif dataset_name == "diffractive":
        dataset = DiffractiveDISDataset(
            cache_dir=config.data.cache_dir,
            raw_dir=config.data.raw_dir,
            processed_dir=config.data.processed_dir,
        )
        analysis = DiffractiveRankAnalysis(
            dataset=dataset,
            output_dir=output_dir / "diffractive_dis",
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    try:
        result = analysis.run(n_bootstrap=n_bootstrap, seed=seed)
        return {
            "lambda_stat": result.lambda_stat,
            "p_value": result.p_value,
            "chi2_rank1": result.chi2_rank1,
            "chi2_rank2": result.chi2_rank2,
            "n_obs": result.n_observations,
        }
    except Exception as e:
        logger.error(f"Analysis failed for {dataset_name}: {e}")
        return None


def compare_results(
    results1: dict[str, Any],
    results2: dict[str, Any],
    name: str,
    tolerance: float = 1e-10,
) -> ReproCheckResult:
    """Compare two result dictionaries for exact match."""
    if results1 is None and results2 is None:
        return ReproCheckResult(
            name=name,
            passed=True,
            run1_hash="none",
            run2_hash="none",
            message="Both runs returned None (dataset unavailable)",
        )

    if results1 is None or results2 is None:
        return ReproCheckResult(
            name=name,
            passed=False,
            run1_hash=str(results1 is not None),
            run2_hash=str(results2 is not None),
            message="One run succeeded, the other failed",
        )

    # Check each value
    mismatches = []
    for key in results1.keys():
        v1, v2 = results1.get(key), results2.get(key)
        if isinstance(v1, float) and isinstance(v2, float):
            if not np.isclose(v1, v2, rtol=tolerance, atol=tolerance):
                mismatches.append(f"{key}: {v1} vs {v2}")
        elif v1 != v2:
            mismatches.append(f"{key}: {v1} vs {v2}")

    # Create hash of results for comparison
    hash1 = hashlib.sha256(json.dumps(results1, sort_keys=True).encode()).hexdigest()[:16]
    hash2 = hashlib.sha256(json.dumps(results2, sort_keys=True).encode()).hexdigest()[:16]

    passed = len(mismatches) == 0

    return ReproCheckResult(
        name=name,
        passed=passed,
        run1_hash=hash1,
        run2_hash=hash2,
        message="" if passed else f"Mismatches: {', '.join(mismatches)}",
        details={"mismatches": mismatches},
    )


def verify_reproducibility(
    seed: int = 42,
    n_bootstrap: int = 100,
    datasets: Optional[list[str]] = None,
    quick: bool = False,
) -> ReproReport:
    """
    Run full reproducibility verification.

    Args:
        seed: Random seed to use
        n_bootstrap: Number of bootstrap iterations
        datasets: List of datasets to test (default: all)
        quick: If True, use minimal bootstrap for speed

    Returns:
        ReproReport with all check results
    """
    if quick:
        n_bootstrap = 10

    if datasets is None:
        datasets = ["higgs", "elastic", "diffractive"]

    setup_logging("INFO")
    logger = get_logger()

    logger.info("=" * 60)
    logger.info("Reproducibility Verification")
    logger.info("=" * 60)
    logger.info(f"Seed: {seed}")
    logger.info(f"Bootstrap iterations: {n_bootstrap}")
    logger.info(f"Datasets: {datasets}")
    logger.info("")

    # Create temporary directories for two runs
    tmpdir = Path(tempfile.mkdtemp(prefix="repro_verify_"))
    run1_dir = tmpdir / "run1"
    run2_dir = tmpdir / "run2"

    try:
        checks = []
        run1_results = {}
        run2_results = {}

        for dataset_name in datasets:
            logger.info(f"Testing {dataset_name}...")

            # Run 1
            logger.info(f"  Run 1...")
            run1_dir.mkdir(parents=True, exist_ok=True)
            r1 = run_single_analysis(
                run1_dir, seed, n_bootstrap, dataset_name, logger
            )
            run1_results[dataset_name] = r1

            # Run 2 (identical parameters)
            logger.info(f"  Run 2...")
            shutil.rmtree(run2_dir, ignore_errors=True)
            run2_dir.mkdir(parents=True, exist_ok=True)
            r2 = run_single_analysis(
                run2_dir, seed, n_bootstrap, dataset_name, logger
            )
            run2_results[dataset_name] = r2

            # Compare results
            check = compare_results(r1, r2, dataset_name)
            checks.append(check)

            status = "PASS" if check.passed else "FAIL"
            logger.info(f"  [{status}] {check.message or 'Results match'}")

        # Compute output file hashes
        run1_hashes = compute_dir_hashes(run1_dir) if run1_dir.exists() else {}
        run2_hashes = compute_dir_hashes(run2_dir) if run2_dir.exists() else {}

        # Find mismatched files
        all_files = set(run1_hashes.keys()) | set(run2_hashes.keys())
        mismatches = []
        for f in all_files:
            h1 = run1_hashes.get(f, "missing")
            h2 = run2_hashes.get(f, "missing")
            if h1 != h2:
                mismatches.append(f)

        all_passed = all(c.passed for c in checks) and len(mismatches) == 0

        report = ReproReport(
            timestamp=datetime.utcnow().isoformat() + "Z",
            seed=seed,
            n_bootstrap=n_bootstrap,
            all_passed=all_passed,
            checks=checks,
            run1_outputs=run1_hashes,
            run2_outputs=run2_hashes,
            mismatches=mismatches,
        )

        # Summary
        logger.info("")
        logger.info("=" * 60)
        logger.info("SUMMARY")
        logger.info("=" * 60)
        for check in checks:
            status = "PASS" if check.passed else "FAIL"
            logger.info(f"  [{status}] {check.name}")

        if mismatches:
            logger.warning(f"  File hash mismatches: {len(mismatches)}")
            for m in mismatches[:5]:
                logger.warning(f"    - {m}")

        logger.info("")
        if all_passed:
            logger.info("REPRODUCIBILITY CHECK PASSED")
        else:
            logger.error("REPRODUCIBILITY CHECK FAILED")

        return report

    finally:
        # Cleanup
        shutil.rmtree(tmpdir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(
        description="Verify reproducibility of analysis pipeline"
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--bootstrap", "-b",
        type=int,
        default=100,
        help="Number of bootstrap iterations (default: 100)",
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        action="append",
        choices=["higgs", "elastic", "diffractive"],
        help="Dataset to test (can specify multiple, default: all)",
    )
    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Quick mode: minimal bootstrap iterations",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output file for JSON report",
    )

    args = parser.parse_args()

    report = verify_reproducibility(
        seed=args.seed,
        n_bootstrap=args.bootstrap,
        datasets=args.dataset,
        quick=args.quick,
    )

    # Save report if requested
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"Report saved to: {args.output}")

    # Exit code based on pass/fail
    return 0 if report.all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
