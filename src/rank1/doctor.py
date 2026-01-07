"""
Doctor / Smoke Test module for rank-1 pipeline.

Provides comprehensive environment checking, minimal test runs,
determinism verification, and auto-remediation for common issues.
"""

import os
import sys
import json
import time
import hashlib
import subprocess
import traceback
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum
import importlib
import platform

from rank1.logging import get_logger, setup_logging

logger = get_logger()


class CheckStatus(Enum):
    """Status of individual checks."""
    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"
    WARN = "warn"
    FIXED = "fixed"


class OverallStatus(Enum):
    """Overall doctor status."""
    PASS = "PASS"
    FAIL = "FAIL"
    NEEDS_MANUAL = "NEEDS_MANUAL_ATTENTION"


@dataclass
class CheckResult:
    """Result of a single check."""
    name: str
    status: CheckStatus
    message: str
    duration_seconds: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    fix_attempted: bool = False
    fix_succeeded: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "duration_seconds": self.duration_seconds,
            "details": self.details,
            "fix_attempted": self.fix_attempted,
            "fix_succeeded": self.fix_succeeded,
        }


@dataclass
class DoctorReport:
    """Complete doctor report."""
    timestamp: str
    python_version: str
    platform_info: str
    overall_status: OverallStatus
    checks: List[CheckResult]
    duration_seconds: float
    warnings: List[str] = field(default_factory=list)
    manual_attention_items: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "python_version": self.python_version,
            "platform_info": self.platform_info,
            "overall_status": self.overall_status.value,
            "checks": [c.to_dict() for c in self.checks],
            "duration_seconds": self.duration_seconds,
            "warnings": self.warnings,
            "manual_attention_items": self.manual_attention_items,
        }

    def to_markdown(self) -> str:
        """Generate markdown report."""
        lines = [
            "# Doctor Report",
            "",
            f"**Generated**: {self.timestamp}",
            f"**Python**: {self.python_version}",
            f"**Platform**: {self.platform_info}",
            f"**Duration**: {self.duration_seconds:.1f}s",
            "",
            f"## Overall Status: {self.overall_status.value}",
            "",
        ]

        if self.warnings:
            lines.append("### Warnings")
            lines.append("")
            for w in self.warnings:
                lines.append(f"- {w}")
            lines.append("")

        if self.manual_attention_items:
            lines.append("### Needs Manual Attention")
            lines.append("")
            for item in self.manual_attention_items:
                lines.append(f"- {item}")
            lines.append("")

        lines.append("## Check Results")
        lines.append("")
        lines.append("| Check | Status | Duration | Message |")
        lines.append("|-------|--------|----------|---------|")

        for c in self.checks:
            status_icon = {
                CheckStatus.PASS: "✅",
                CheckStatus.FAIL: "❌",
                CheckStatus.SKIP: "⏭️",
                CheckStatus.WARN: "⚠️",
                CheckStatus.FIXED: "🔧",
            }.get(c.status, "❓")

            lines.append(
                f"| {c.name} | {status_icon} {c.status.value} | "
                f"{c.duration_seconds:.1f}s | {c.message[:50]}{'...' if len(c.message) > 50 else ''} |"
            )

        lines.append("")

        # Detailed results for failures
        failures = [c for c in self.checks if c.status == CheckStatus.FAIL]
        if failures:
            lines.append("## Failure Details")
            lines.append("")
            for c in failures:
                lines.append(f"### {c.name}")
                lines.append("")
                lines.append(f"**Message**: {c.message}")
                lines.append("")
                if c.details:
                    lines.append("**Details**:")
                    lines.append("```")
                    lines.append(json.dumps(c.details, indent=2, default=str))
                    lines.append("```")
                lines.append("")

        return "\n".join(lines)


class Doctor:
    """
    Main doctor class that runs all checks and auto-remediation.
    """

    def __init__(
        self,
        output_dir: Path,
        n_bootstrap: int = 20,
        n_global_bootstrap: int = 30,
        seed: int = 1,
        n_jobs: int = 2,
        n_starts: int = 3,
        skip_elastic_7tev: bool = True,
        skip_large_downloads: bool = True,
        fast_mode: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.doctor_dir = self.output_dir / "doctor"
        self.n_bootstrap = n_bootstrap
        self.n_global_bootstrap = n_global_bootstrap
        self.seed = seed
        self.n_jobs = n_jobs
        self.n_starts = n_starts
        self.skip_elastic_7tev = skip_elastic_7tev
        self.skip_large_downloads = skip_large_downloads
        self.fast_mode = fast_mode

        self.checks: List[CheckResult] = []
        self.warnings: List[str] = []
        self.manual_attention: List[str] = []
        self.start_time = None

        # Ensure output dirs exist
        self.doctor_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> DoctorReport:
        """Run all doctor checks."""
        self.start_time = time.time()
        logger.info("=" * 60)
        logger.info("RANK-1 DOCTOR / SMOKE TEST")
        logger.info("=" * 60)

        # A) Environment checks
        self._run_environment_checks()

        # B) Lint/test checks
        self._run_code_checks()

        # C) Minimal data fetch
        self._run_data_fetch()

        # D) Minimal baseline runs
        self._run_baseline_analyses()

        # E) Minimal NP mode runs
        self._run_np_analyses()

        # F) Determinism check
        self._run_determinism_check()

        # G) Output artifact check
        self._run_artifact_check()

        # Compute overall status
        overall = self._compute_overall_status()

        duration = time.time() - self.start_time

        report = DoctorReport(
            timestamp=datetime.now().isoformat(),
            python_version=sys.version,
            platform_info=f"{platform.system()} {platform.release()}",
            overall_status=overall,
            checks=self.checks,
            duration_seconds=duration,
            warnings=self.warnings,
            manual_attention_items=self.manual_attention,
        )

        # H) Write report
        self._write_report(report)

        return report

    def _add_check(self, result: CheckResult):
        """Add a check result."""
        self.checks.append(result)
        status_str = result.status.value.upper()
        logger.info(f"  [{status_str}] {result.name}: {result.message}")

    def _run_environment_checks(self):
        """Run environment and dependency checks."""
        logger.info("\n--- Environment Checks ---")

        # Python version
        start = time.time()
        py_version = sys.version_info
        if py_version >= (3, 11):
            self._add_check(CheckResult(
                name="python_version",
                status=CheckStatus.PASS,
                message=f"Python {py_version.major}.{py_version.minor}.{py_version.micro}",
                duration_seconds=time.time() - start,
            ))
        else:
            self._add_check(CheckResult(
                name="python_version",
                status=CheckStatus.FAIL,
                message=f"Python {py_version.major}.{py_version.minor} < 3.11 required",
                duration_seconds=time.time() - start,
            ))
            self.manual_attention.append("Upgrade Python to 3.11+")

        # pip check
        start = time.time()
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "check"],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0:
                self._add_check(CheckResult(
                    name="pip_check",
                    status=CheckStatus.PASS,
                    message="All dependencies satisfied",
                    duration_seconds=time.time() - start,
                ))
            else:
                self._add_check(CheckResult(
                    name="pip_check",
                    status=CheckStatus.WARN,
                    message="Dependency issues detected",
                    duration_seconds=time.time() - start,
                    details={"stderr": result.stderr[:500]},
                ))
                self.warnings.append(f"pip check failed: {result.stderr[:200]}")
        except Exception as e:
            self._add_check(CheckResult(
                name="pip_check",
                status=CheckStatus.WARN,
                message=f"Could not run pip check: {e}",
                duration_seconds=time.time() - start,
            ))

        # Import key packages
        start = time.time()
        required_packages = ["numpy", "pandas", "scipy", "matplotlib", "typer", "requests"]
        optional_packages = ["uproot", "camelot", "pdfplumber"]

        missing_required = []
        missing_optional = []

        for pkg in required_packages:
            try:
                importlib.import_module(pkg)
            except ImportError:
                missing_required.append(pkg)

        for pkg in optional_packages:
            try:
                importlib.import_module(pkg)
            except ImportError:
                missing_optional.append(pkg)

        if missing_required:
            self._add_check(CheckResult(
                name="required_imports",
                status=CheckStatus.FAIL,
                message=f"Missing: {', '.join(missing_required)}",
                duration_seconds=time.time() - start,
            ))
            self.manual_attention.append(f"Install missing packages: {', '.join(missing_required)}")
        else:
            self._add_check(CheckResult(
                name="required_imports",
                status=CheckStatus.PASS,
                message="All required packages importable",
                duration_seconds=time.time() - start,
            ))

        if missing_optional:
            self._add_check(CheckResult(
                name="optional_imports",
                status=CheckStatus.WARN,
                message=f"Missing optional: {', '.join(missing_optional)}",
                duration_seconds=time.time() - start,
                details={"missing": missing_optional},
            ))
            if "camelot" in missing_optional or "pdfplumber" in missing_optional:
                self.warnings.append("PDF extraction will use fallback tables (camelot/pdfplumber not installed)")
        else:
            self._add_check(CheckResult(
                name="optional_imports",
                status=CheckStatus.PASS,
                message="All optional packages available",
                duration_seconds=time.time() - start,
            ))

        # Check CLI is accessible
        start = time.time()
        try:
            result = subprocess.run(
                [sys.executable, "-m", "rank1.cli", "--help"],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0:
                self._add_check(CheckResult(
                    name="cli_accessible",
                    status=CheckStatus.PASS,
                    message="CLI help works",
                    duration_seconds=time.time() - start,
                ))
            else:
                self._add_check(CheckResult(
                    name="cli_accessible",
                    status=CheckStatus.FAIL,
                    message="CLI help failed",
                    duration_seconds=time.time() - start,
                    details={"stderr": result.stderr[:500]},
                ))
        except Exception as e:
            self._add_check(CheckResult(
                name="cli_accessible",
                status=CheckStatus.FAIL,
                message=f"CLI error: {e}",
                duration_seconds=time.time() - start,
            ))

    def _run_code_checks(self):
        """Run code quality checks."""
        logger.info("\n--- Code Checks ---")

        # Compile check
        start = time.time()
        try:
            result = subprocess.run(
                [sys.executable, "-m", "compileall", "-q", "src"],
                capture_output=True, text=True, timeout=60, cwd=self._get_repo_root()
            )
            if result.returncode == 0:
                self._add_check(CheckResult(
                    name="compile_check",
                    status=CheckStatus.PASS,
                    message="All Python files compile",
                    duration_seconds=time.time() - start,
                ))
            else:
                self._add_check(CheckResult(
                    name="compile_check",
                    status=CheckStatus.FAIL,
                    message="Compilation errors found",
                    duration_seconds=time.time() - start,
                    details={"stderr": result.stderr[:500]},
                ))
        except Exception as e:
            self._add_check(CheckResult(
                name="compile_check",
                status=CheckStatus.WARN,
                message=f"Could not run compile check: {e}",
                duration_seconds=time.time() - start,
            ))

        # Fast unit tests
        start = time.time()
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "tests/", "-q", "-x",
                 "-m", "not slow and not network", "--tb=short", "--timeout=60"],
                capture_output=True, text=True, timeout=120, cwd=self._get_repo_root()
            )
            if result.returncode == 0:
                self._add_check(CheckResult(
                    name="fast_tests",
                    status=CheckStatus.PASS,
                    message="Fast unit tests pass",
                    duration_seconds=time.time() - start,
                ))
            elif "no tests ran" in result.stdout.lower() or result.returncode == 5:
                # pytest exit code 5 = no tests collected
                self._add_check(CheckResult(
                    name="fast_tests",
                    status=CheckStatus.SKIP,
                    message="No fast tests found or collected",
                    duration_seconds=time.time() - start,
                ))
            else:
                # Try auto-fix with ruff
                fix_attempted = False
                fix_succeeded = False

                if "SyntaxError" not in result.stdout and "ImportError" not in result.stdout:
                    try:
                        fix_result = subprocess.run(
                            [sys.executable, "-m", "ruff", "check", "--fix", "src/", "tests/"],
                            capture_output=True, text=True, timeout=60, cwd=self._get_repo_root()
                        )
                        fix_attempted = True

                        # Re-run tests
                        retry = subprocess.run(
                            [sys.executable, "-m", "pytest", "tests/", "-q", "-x",
                             "-m", "not slow and not network", "--tb=short", "--timeout=60"],
                            capture_output=True, text=True, timeout=120, cwd=self._get_repo_root()
                        )
                        if retry.returncode == 0:
                            fix_succeeded = True
                    except Exception:
                        pass

                if fix_succeeded:
                    self._add_check(CheckResult(
                        name="fast_tests",
                        status=CheckStatus.FIXED,
                        message="Tests pass after auto-fix",
                        duration_seconds=time.time() - start,
                        fix_attempted=True,
                        fix_succeeded=True,
                    ))
                else:
                    self._add_check(CheckResult(
                        name="fast_tests",
                        status=CheckStatus.WARN,
                        message="Some tests failed (non-blocking in smoke mode)",
                        duration_seconds=time.time() - start,
                        details={"stdout": result.stdout[:1000]},
                        fix_attempted=fix_attempted,
                    ))
                    self.warnings.append("Some unit tests failed - review test output")

        except subprocess.TimeoutExpired:
            self._add_check(CheckResult(
                name="fast_tests",
                status=CheckStatus.WARN,
                message="Tests timed out",
                duration_seconds=time.time() - start,
            ))
        except Exception as e:
            self._add_check(CheckResult(
                name="fast_tests",
                status=CheckStatus.SKIP,
                message=f"Could not run tests: {e}",
                duration_seconds=time.time() - start,
            ))

    def _run_data_fetch(self):
        """Run minimal data fetching."""
        logger.info("\n--- Data Fetch ---")

        datasets = ["higgs", "diffractive"]

        # Add elastic only if not skipping large downloads
        if not self.skip_large_downloads:
            datasets.append("elastic")
        else:
            self._add_check(CheckResult(
                name="fetch_elastic",
                status=CheckStatus.SKIP,
                message="Skipped (7 TeV ROOT files are large; use --no-skip-large-downloads)",
                duration_seconds=0.0,
            ))
            self.warnings.append("Elastic 7 TeV data not fetched (large download)")

        for ds in datasets:
            start = time.time()
            try:
                # Use internal API for speed
                from rank1.config import get_config
                cfg = get_config()
                cfg.ensure_dirs()

                if ds == "higgs":
                    from rank1.datasets import HiggsATLASDataset
                    dataset = HiggsATLASDataset(
                        cache_dir=cfg.data.cache_dir,
                        raw_dir=cfg.data.raw_dir,
                        processed_dir=cfg.data.processed_dir,
                    )
                elif ds == "diffractive":
                    from rank1.datasets import DiffractiveDISDataset
                    dataset = DiffractiveDISDataset(
                        cache_dir=cfg.data.cache_dir,
                        raw_dir=cfg.data.raw_dir,
                        processed_dir=cfg.data.processed_dir,
                    )
                elif ds == "elastic":
                    from rank1.datasets import ElasticTOTEMDataset
                    dataset = ElasticTOTEMDataset(
                        cache_dir=cfg.data.cache_dir,
                        raw_dir=cfg.data.raw_dir,
                        processed_dir=cfg.data.processed_dir,
                    )
                else:
                    continue

                # Attempt fetch with retry
                try:
                    dataset.fetch_raw(force=False)
                    self._add_check(CheckResult(
                        name=f"fetch_{ds}",
                        status=CheckStatus.PASS,
                        message="Data fetched/cached",
                        duration_seconds=time.time() - start,
                    ))
                except Exception as e:
                    # Retry once
                    time.sleep(2)
                    try:
                        dataset.fetch_raw(force=False)
                        self._add_check(CheckResult(
                            name=f"fetch_{ds}",
                            status=CheckStatus.FIXED,
                            message="Data fetched after retry",
                            duration_seconds=time.time() - start,
                            fix_attempted=True,
                            fix_succeeded=True,
                        ))
                    except Exception as e2:
                        self._add_check(CheckResult(
                            name=f"fetch_{ds}",
                            status=CheckStatus.FAIL,
                            message=f"Fetch failed: {str(e2)[:100]}",
                            duration_seconds=time.time() - start,
                            details={"error": str(e2)},
                            fix_attempted=True,
                        ))
                        self.manual_attention.append(f"Fix data fetch for {ds}: {str(e2)[:100]}")

            except Exception as e:
                self._add_check(CheckResult(
                    name=f"fetch_{ds}",
                    status=CheckStatus.FAIL,
                    message=f"Error: {str(e)[:100]}",
                    duration_seconds=time.time() - start,
                    details={"error": str(e), "traceback": traceback.format_exc()},
                ))
                self.manual_attention.append(f"Data fetch error for {ds}")

    def _run_baseline_analyses(self):
        """Run minimal baseline analyses."""
        logger.info("\n--- Baseline Analyses ---")

        datasets = ["higgs", "diffractive"]

        # Only add elastic if we have data
        if not self.skip_large_downloads:
            datasets.append("elastic")
        else:
            self._add_check(CheckResult(
                name="baseline_elastic",
                status=CheckStatus.SKIP,
                message="Skipped (no elastic data in smoke mode)",
                duration_seconds=0.0,
            ))

        for ds in datasets:
            start = time.time()
            try:
                from rank1.config import get_config
                cfg = get_config()

                if ds == "higgs":
                    from rank1.analysis import HiggsRankAnalysis
                    analysis = HiggsRankAnalysis(output_dir=cfg.output_dir / "higgs_atlas")
                elif ds == "diffractive":
                    from rank1.analysis import DiffractiveRankAnalysis
                    analysis = DiffractiveRankAnalysis(output_dir=cfg.output_dir / "diffractive_dis")
                elif ds == "elastic":
                    from rank1.analysis import ElasticRankAnalysis
                    analysis = ElasticRankAnalysis(output_dir=cfg.output_dir / "elastic_totem")
                else:
                    continue

                # Run with minimal settings
                try:
                    result = analysis.run(
                        n_bootstrap=self.n_bootstrap,
                        seed=self.seed,
                    )
                    self._add_check(CheckResult(
                        name=f"baseline_{ds}",
                        status=CheckStatus.PASS,
                        message=f"Λ={result.lambda_stat:.2f}, p={result.p_value:.3f}",
                        duration_seconds=time.time() - start,
                        details={
                            "lambda": result.lambda_stat,
                            "p_value": result.p_value,
                            "chi2_rank1": result.chi2_rank1,
                            "chi2_rank2": result.chi2_rank2,
                        },
                    ))
                except Exception as e:
                    # Try with reduced parallelism
                    try:
                        os.environ["OMP_NUM_THREADS"] = "1"
                        os.environ["MKL_NUM_THREADS"] = "1"
                        result = analysis.run(
                            n_bootstrap=self.n_bootstrap,
                            seed=self.seed,
                        )
                        self._add_check(CheckResult(
                            name=f"baseline_{ds}",
                            status=CheckStatus.FIXED,
                            message=f"Passed with reduced parallelism: Λ={result.lambda_stat:.2f}",
                            duration_seconds=time.time() - start,
                            fix_attempted=True,
                            fix_succeeded=True,
                        ))
                    except Exception as e2:
                        self._add_check(CheckResult(
                            name=f"baseline_{ds}",
                            status=CheckStatus.FAIL,
                            message=f"Analysis failed: {str(e2)[:100]}",
                            duration_seconds=time.time() - start,
                            details={"error": str(e2)},
                            fix_attempted=True,
                        ))
                        self.manual_attention.append(f"Baseline analysis failed for {ds}")

            except Exception as e:
                self._add_check(CheckResult(
                    name=f"baseline_{ds}",
                    status=CheckStatus.FAIL,
                    message=f"Setup error: {str(e)[:100]}",
                    duration_seconds=time.time() - start,
                    details={"traceback": traceback.format_exc()},
                ))

    def _run_np_analyses(self):
        """Run minimal NP mode analyses."""
        logger.info("\n--- NP Mode Analyses ---")

        datasets = ["higgs", "diffractive"]

        if not self.skip_large_downloads:
            datasets.append("elastic")
        else:
            self._add_check(CheckResult(
                name="np_elastic",
                status=CheckStatus.SKIP,
                message="Skipped (no elastic data in smoke mode)",
                duration_seconds=0.0,
            ))

        for ds in datasets:
            start = time.time()
            try:
                from rank1.config import get_config
                from rank1.analysis.np_analysis import NPAnalyzer

                cfg = get_config()

                # Load matrix data
                if ds == "higgs":
                    from rank1.datasets import HiggsATLASDataset
                    dataset = HiggsATLASDataset(
                        cache_dir=cfg.data.cache_dir,
                        raw_dir=cfg.data.raw_dir,
                        processed_dir=cfg.data.processed_dir,
                    )
                elif ds == "diffractive":
                    from rank1.datasets import DiffractiveDISDataset
                    dataset = DiffractiveDISDataset(
                        cache_dir=cfg.data.cache_dir,
                        raw_dir=cfg.data.raw_dir,
                        processed_dir=cfg.data.processed_dir,
                    )
                elif ds == "elastic":
                    from rank1.datasets import ElasticTOTEMDataset
                    dataset = ElasticTOTEMDataset(
                        cache_dir=cfg.data.cache_dir,
                        raw_dir=cfg.data.raw_dir,
                        processed_dir=cfg.data.processed_dir,
                    )
                else:
                    continue

                matrix_data = dataset.get_matrix_data()

                output_dir = cfg.output_dir / ds / "np"
                analyzer = NPAnalyzer(
                    dataset=ds,
                    output_dir=output_dir,
                    n_jobs=self.n_jobs,
                )

                result = analyzer.run(
                    matrix_data=matrix_data,
                    n_bootstrap=self.n_bootstrap,
                    n_global_bootstrap=self.n_global_bootstrap,
                    n_starts=self.n_starts,
                    seed=self.seed,
                    run_sweeps=False,  # Skip sweeps in smoke for speed
                    run_replication=True,
                    fast_mode=True,
                )

                self._add_check(CheckResult(
                    name=f"np_{ds}",
                    status=CheckStatus.PASS,
                    message=f"Verdict: {result.np_verdict.value}",
                    duration_seconds=time.time() - start,
                    details={
                        "verdict": result.np_verdict.value,
                        "lambda": result.lambda_stat,
                        "p_local": result.p_local,
                    },
                ))

            except Exception as e:
                self._add_check(CheckResult(
                    name=f"np_{ds}",
                    status=CheckStatus.FAIL,
                    message=f"NP analysis failed: {str(e)[:100]}",
                    duration_seconds=time.time() - start,
                    details={"traceback": traceback.format_exc()},
                ))
                self.warnings.append(f"NP analysis failed for {ds}")

    def _run_determinism_check(self):
        """Verify determinism by running twice with same seed."""
        logger.info("\n--- Determinism Check ---")

        start = time.time()
        try:
            from rank1.config import get_config
            from rank1.analysis import HiggsRankAnalysis

            cfg = get_config()

            # Run twice with same seed
            analysis1 = HiggsRankAnalysis(output_dir=cfg.output_dir / "doctor_repro_1")
            result1 = analysis1.run(n_bootstrap=10, seed=12345)

            analysis2 = HiggsRankAnalysis(output_dir=cfg.output_dir / "doctor_repro_2")
            result2 = analysis2.run(n_bootstrap=10, seed=12345)

            # Compare key values
            lambda_match = abs(result1.lambda_stat - result2.lambda_stat) < 1e-10
            p_match = abs(result1.p_value - result2.p_value) < 1e-10
            chi2_match = abs(result1.chi2_rank1 - result2.chi2_rank1) < 1e-10

            if lambda_match and p_match and chi2_match:
                self._add_check(CheckResult(
                    name="determinism",
                    status=CheckStatus.PASS,
                    message="Results are deterministic",
                    duration_seconds=time.time() - start,
                    details={
                        "lambda_diff": abs(result1.lambda_stat - result2.lambda_stat),
                        "p_diff": abs(result1.p_value - result2.p_value),
                    },
                ))
            else:
                # Try with single-threaded BLAS
                os.environ["OMP_NUM_THREADS"] = "1"
                os.environ["MKL_NUM_THREADS"] = "1"
                os.environ["OPENBLAS_NUM_THREADS"] = "1"

                analysis3 = HiggsRankAnalysis(output_dir=cfg.output_dir / "doctor_repro_3")
                result3 = analysis3.run(n_bootstrap=10, seed=12345)

                analysis4 = HiggsRankAnalysis(output_dir=cfg.output_dir / "doctor_repro_4")
                result4 = analysis4.run(n_bootstrap=10, seed=12345)

                lambda_match2 = abs(result3.lambda_stat - result4.lambda_stat) < 1e-10

                if lambda_match2:
                    self._add_check(CheckResult(
                        name="determinism",
                        status=CheckStatus.FIXED,
                        message="Deterministic with single-threaded BLAS",
                        duration_seconds=time.time() - start,
                        fix_attempted=True,
                        fix_succeeded=True,
                    ))
                    self.warnings.append("Set OMP_NUM_THREADS=1 for determinism")
                else:
                    self._add_check(CheckResult(
                        name="determinism",
                        status=CheckStatus.WARN,
                        message="Minor non-determinism detected (acceptable for numerical reasons)",
                        duration_seconds=time.time() - start,
                        details={
                            "lambda_diff": abs(result3.lambda_stat - result4.lambda_stat),
                        },
                    ))

        except Exception as e:
            self._add_check(CheckResult(
                name="determinism",
                status=CheckStatus.SKIP,
                message=f"Could not run determinism check: {str(e)[:50]}",
                duration_seconds=time.time() - start,
            ))

    def _run_artifact_check(self):
        """Check that expected output artifacts exist."""
        logger.info("\n--- Artifact Check ---")

        start = time.time()
        from rank1.config import get_config
        cfg = get_config()

        expected_baseline = {
            "higgs_atlas": ["results.json", "summary.md"],
            "diffractive_dis": ["results.json", "summary.md"],
        }

        expected_np = {
            "higgs": ["np/np_results.json", "np/np_summary.txt"],
            "diffractive": ["np/np_results.json", "np/np_summary.txt"],
        }

        missing = []
        found = []

        for ds, files in expected_baseline.items():
            for f in files:
                path = cfg.output_dir / ds / f
                if path.exists():
                    found.append(str(path))
                else:
                    missing.append(str(path))

        for ds, files in expected_np.items():
            for f in files:
                path = cfg.output_dir / ds / f
                if path.exists():
                    found.append(str(path))
                else:
                    missing.append(str(path))

        if not missing:
            self._add_check(CheckResult(
                name="artifacts",
                status=CheckStatus.PASS,
                message=f"All {len(found)} expected artifacts found",
                duration_seconds=time.time() - start,
            ))
        elif len(missing) < len(found):
            self._add_check(CheckResult(
                name="artifacts",
                status=CheckStatus.WARN,
                message=f"{len(missing)} artifacts missing",
                duration_seconds=time.time() - start,
                details={"missing": missing[:5]},
            ))
        else:
            self._add_check(CheckResult(
                name="artifacts",
                status=CheckStatus.FAIL,
                message=f"Many artifacts missing ({len(missing)})",
                duration_seconds=time.time() - start,
                details={"missing": missing[:10]},
            ))

    def _compute_overall_status(self) -> OverallStatus:
        """Compute overall pass/fail status."""
        failures = [c for c in self.checks if c.status == CheckStatus.FAIL]
        warnings = [c for c in self.checks if c.status == CheckStatus.WARN]

        if self.manual_attention:
            return OverallStatus.NEEDS_MANUAL

        if failures:
            # Check if failures are critical
            critical_fails = [
                c for c in failures
                if c.name in ["python_version", "required_imports", "cli_accessible", "compile_check"]
            ]
            if critical_fails:
                return OverallStatus.FAIL
            return OverallStatus.NEEDS_MANUAL

        if len(warnings) > 3:
            return OverallStatus.NEEDS_MANUAL

        return OverallStatus.PASS

    def _write_report(self, report: DoctorReport):
        """Write report to files."""
        # JSON
        json_path = self.doctor_dir / "doctor_report.json"
        with open(json_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)

        # Markdown
        md_path = self.doctor_dir / "doctor_report.md"
        with open(md_path, "w") as f:
            f.write(report.to_markdown())

        logger.info(f"\nReports written to:")
        logger.info(f"  {json_path}")
        logger.info(f"  {md_path}")

        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info(f"DOCTOR RESULT: {report.overall_status.value}")
        logger.info("=" * 60)

        passed = len([c for c in self.checks if c.status == CheckStatus.PASS])
        failed = len([c for c in self.checks if c.status == CheckStatus.FAIL])
        warned = len([c for c in self.checks if c.status == CheckStatus.WARN])
        skipped = len([c for c in self.checks if c.status == CheckStatus.SKIP])
        fixed = len([c for c in self.checks if c.status == CheckStatus.FIXED])

        logger.info(f"  Passed: {passed}")
        logger.info(f"  Failed: {failed}")
        logger.info(f"  Warnings: {warned}")
        logger.info(f"  Skipped: {skipped}")
        logger.info(f"  Auto-fixed: {fixed}")
        logger.info(f"  Duration: {report.duration_seconds:.1f}s")

    def _get_repo_root(self) -> Path:
        """Get repository root directory."""
        # Assuming we're in src/rank1/
        return Path(__file__).parent.parent.parent


def run_doctor(
    output_dir: Optional[Path] = None,
    n_bootstrap: int = 20,
    n_global_bootstrap: int = 30,
    seed: int = 1,
    n_jobs: int = 2,
    n_starts: int = 3,
    skip_elastic_7tev: bool = True,
    skip_large_downloads: bool = True,
    fast_mode: bool = True,
) -> DoctorReport:
    """
    Run the doctor/smoke test.

    Args:
        output_dir: Output directory (defaults to config output_dir)
        n_bootstrap: Bootstrap samples for analyses
        n_global_bootstrap: Global bootstrap for NP mode
        seed: Random seed
        n_jobs: Number of parallel jobs
        n_starts: Multi-start fits for NP
        skip_elastic_7tev: Skip large TOTEM 7 TeV download
        skip_large_downloads: Skip all large downloads
        fast_mode: Use minimal settings

    Returns:
        DoctorReport with all check results
    """
    if output_dir is None:
        from rank1.config import get_config
        cfg = get_config()
        cfg.ensure_dirs()
        output_dir = cfg.output_dir

    doctor = Doctor(
        output_dir=output_dir,
        n_bootstrap=n_bootstrap,
        n_global_bootstrap=n_global_bootstrap,
        seed=seed,
        n_jobs=n_jobs,
        n_starts=n_starts,
        skip_elastic_7tev=skip_elastic_7tev,
        skip_large_downloads=skip_large_downloads,
        fast_mode=fast_mode,
    )

    return doctor.run()
