"""
Base analysis class for rank-1 factorization tests.

Provides a uniform interface for running the complete analysis
workflow on any matrix dataset.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any
import json

import numpy as np
import pandas as pd

from rank1.datasets.base import MatrixDataset, MatrixData
from rank1.models.fit import LowRankFitter, FitResult
from rank1.models.bootstrap import BootstrapTester, BootstrapResult
from rank1.config import AnalysisConfig, get_config
from rank1.logging import get_logger

logger = get_logger()


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class AnalysisVerdict:
    """Verdict classifications for analysis results."""
    REJECTED = "REJECTED"          # Rank-1 hypothesis rejected (p < 0.05)
    NOT_REJECTED = "NOT_REJECTED"  # Rank-1 hypothesis not rejected (p >= 0.05)
    INCONCLUSIVE = "INCONCLUSIVE"  # Cannot draw conclusion (underconstrained, invalid fit, etc.)


@dataclass
class AnalysisResult:
    """Complete results from a rank-1 analysis."""

    # Dataset info
    dataset_name: str
    n_rows: int
    n_cols: int
    n_obs: int

    # Fit results
    chi2_rank1: float
    chi2_rank2: float
    ndof_rank1: int
    ndof_rank2: int

    # Test statistic
    lambda_stat: float

    # Bootstrap p-value
    p_value: float
    p_value_ci: tuple[float, float]
    n_bootstrap: int

    # Fit health
    rank1_converged: bool
    rank2_converged: bool
    is_stable: bool

    # Predicted matrices
    matrix_rank1: np.ndarray
    matrix_rank2: np.ndarray

    # Raw results
    fit_rank1: Optional[FitResult] = None
    fit_rank2: Optional[FitResult] = None
    bootstrap_result: Optional[BootstrapResult] = None

    # Cross-check results
    cross_checks: list[dict] = field(default_factory=list)

    # Metadata
    config: Optional[dict] = None
    seed: int = 42

    # Provenance - MANDATORY for valid results
    provenance: Optional[dict] = None

    # Data quality flags
    is_placeholder_data: bool = False

    @property
    def is_underconstrained(self) -> bool:
        """Check if fit is underconstrained (ndof <= 0 for either model)."""
        return self.ndof_rank1 <= 0 or self.ndof_rank2 <= 0

    @property
    def verdict(self) -> str:
        """
        Get the analysis verdict with proper gating.

        Returns INCONCLUSIVE if:
        - Either fit is underconstrained (ndof <= 0)
        - Either fit failed to converge
        - Data is placeholder/synthetic
        - Provenance is missing
        """
        # Gate 1: Provenance must exist
        if self.provenance is None:
            return AnalysisVerdict.INCONCLUSIVE

        # Gate 2: Cannot use placeholder data
        if self.is_placeholder_data:
            return AnalysisVerdict.INCONCLUSIVE

        # Gate 3: Both fits must converge
        if not self.rank1_converged or not self.rank2_converged:
            return AnalysisVerdict.INCONCLUSIVE

        # Gate 4: ndof must be positive for valid chi2 interpretation
        if self.ndof_rank1 <= 0 or self.ndof_rank2 <= 0:
            return AnalysisVerdict.INCONCLUSIVE

        # Gate 5: lambda_stat should be non-negative (chi2 improvement)
        # Small negative values can occur due to numerical issues
        if self.lambda_stat < -1.0:  # Allow small numerical errors
            return AnalysisVerdict.INCONCLUSIVE

        # All gates passed - return standard verdict
        if self.p_value < 0.05:
            return AnalysisVerdict.REJECTED
        return AnalysisVerdict.NOT_REJECTED

    @property
    def verdict_reason(self) -> str:
        """Explain why the current verdict was reached."""
        if self.provenance is None:
            return "No provenance tracking - cannot verify data integrity"
        if self.is_placeholder_data:
            return "Using placeholder/synthetic data - results not valid"
        if not self.rank1_converged:
            return "Rank-1 fit did not converge"
        if not self.rank2_converged:
            return "Rank-2 fit did not converge"
        if self.ndof_rank1 <= 0:
            return f"Rank-1 fit underconstrained (ndof={self.ndof_rank1})"
        if self.ndof_rank2 <= 0:
            return f"Rank-2 fit underconstrained (ndof={self.ndof_rank2})"
        if self.lambda_stat < -1.0:
            return f"Invalid test statistic (Λ={self.lambda_stat:.2f} < 0)"
        if self.p_value < 0.05:
            return f"p-value = {self.p_value:.4f} < 0.05"
        return f"p-value = {self.p_value:.4f} >= 0.05"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "dataset_name": self.dataset_name,
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "n_obs": self.n_obs,
            "chi2_rank1": self.chi2_rank1,
            "chi2_rank2": self.chi2_rank2,
            "chi2_ndof_rank1": self.chi2_rank1 / max(1, self.ndof_rank1) if self.ndof_rank1 > 0 else None,
            "chi2_ndof_rank2": self.chi2_rank2 / max(1, self.ndof_rank2) if self.ndof_rank2 > 0 else None,
            "ndof_rank1": self.ndof_rank1,
            "ndof_rank2": self.ndof_rank2,
            "lambda_stat": self.lambda_stat,
            "p_value": self.p_value,
            "p_value_ci_lower": self.p_value_ci[0],
            "p_value_ci_upper": self.p_value_ci[1],
            "n_bootstrap": self.n_bootstrap,
            "rank1_converged": self.rank1_converged,
            "rank2_converged": self.rank2_converged,
            "is_stable": self.is_stable,
            "cross_checks": self.cross_checks,
            "seed": self.seed,
            # Verdict with gating
            "verdict": self.verdict,
            "verdict_reason": self.verdict_reason,
            "is_underconstrained": self.is_underconstrained,
            "is_placeholder_data": self.is_placeholder_data,
            # Provenance - MANDATORY
            "provenance": self.provenance,
        }
        return result

    def save(self, path: Path) -> None:
        """Save results to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, cls=NumpyEncoder)

    @property
    def is_significant(self) -> bool:
        """Check if result is significant at p < 0.05 (only valid if verdict != INCONCLUSIVE)."""
        return self.verdict == AnalysisVerdict.REJECTED

    def summary_string(self) -> str:
        """Get human-readable summary."""
        lines = [
            f"=== {self.dataset_name} Rank-1 Test Results ===",
            f"Matrix size: {self.n_rows} x {self.n_cols} ({self.n_obs} observations)",
            "",
            "Fit Statistics:",
        ]

        # Rank-1 line with underconstrained warning
        if self.ndof_rank1 > 0:
            lines.append(f"  Rank-1: χ² = {self.chi2_rank1:.2f}, ndof = {self.ndof_rank1}, χ²/ndof = {self.chi2_rank1/self.ndof_rank1:.3f}")
        else:
            lines.append(f"  Rank-1: χ² = {self.chi2_rank1:.2f}, ndof = {self.ndof_rank1} [UNDERCONSTRAINED]")

        # Rank-2 line with underconstrained warning
        if self.ndof_rank2 > 0:
            lines.append(f"  Rank-2: χ² = {self.chi2_rank2:.2f}, ndof = {self.ndof_rank2}, χ²/ndof = {self.chi2_rank2/self.ndof_rank2:.3f}")
        else:
            lines.append(f"  Rank-2: χ² = {self.chi2_rank2:.2f}, ndof = {self.ndof_rank2} [UNDERCONSTRAINED]")

        lines.extend([
            "",
            f"Test Statistic: Λ = χ²(rank-1) - χ²(rank-2) = {self.lambda_stat:.2f}",
            "",
            f"Bootstrap p-value: {self.p_value:.4f} (95% CI: [{self.p_value_ci[0]:.4f}, {self.p_value_ci[1]:.4f}])",
            f"  (based on {self.n_bootstrap} bootstrap samples)",
            "",
        ])

        # Quality warnings
        warnings = []
        if self.is_placeholder_data:
            warnings.append("⚠ PLACEHOLDER DATA - Results are NOT valid for publication")
        if self.provenance is None:
            warnings.append("⚠ NO PROVENANCE - Data integrity cannot be verified")
        if self.is_underconstrained:
            warnings.append(f"⚠ UNDERCONSTRAINED FIT - ndof_rank1={self.ndof_rank1}, ndof_rank2={self.ndof_rank2}")
        if not self.rank1_converged:
            warnings.append("⚠ Rank-1 fit did not converge")
        if not self.rank2_converged:
            warnings.append("⚠ Rank-2 fit did not converge")

        if warnings:
            lines.extend(warnings)
            lines.append("")

        # Verdict with explanation
        lines.append(f"VERDICT: {self.verdict}")
        lines.append(f"  Reason: {self.verdict_reason}")

        return "\n".join(lines)


class BaseRankAnalysis(ABC):
    """
    Base class for rank-1 factorization analysis.

    Subclasses should implement dataset-specific diagnostics
    and cross-checks.
    """

    name: str = "base"
    description: str = "Base rank analysis"

    def __init__(
        self,
        dataset: MatrixDataset,
        config: Optional[AnalysisConfig] = None,
        output_dir: Optional[Path] = None,
    ):
        """
        Initialize analysis.

        Args:
            dataset: MatrixDataset to analyze
            config: Analysis configuration
            output_dir: Directory for outputs
        """
        self.dataset = dataset
        self.config = config or get_config().get_analysis_config(
            self.name, dataset.name
        )
        self.output_dir = output_dir or Path("outputs") / self.name

        self.fitter = LowRankFitter(
            regularization=self.config.regularization,
            n_starts=self.config.fit_health.n_starts,
        )

        self._matrix_data: Optional[MatrixData] = None
        self._result: Optional[AnalysisResult] = None

    def run(
        self,
        n_bootstrap: Optional[int] = None,
        seed: Optional[int] = None,
        save_outputs: bool = True,
    ) -> AnalysisResult:
        """
        Run the complete analysis.

        Args:
            n_bootstrap: Override number of bootstrap samples
            seed: Override random seed
            save_outputs: Save results and figures

        Returns:
            AnalysisResult with all statistics
        """
        if n_bootstrap is None:
            n_bootstrap = self.config.bootstrap.n_bootstrap
        if seed is None:
            seed = self.config.bootstrap.seed

        logger.info(f"Starting {self.name} analysis")

        # Load data
        self._matrix_data = self.dataset.get_matrix_data()

        # Extract observation vectors
        rows, cols, values, errors = self._matrix_data.to_vectors()
        n_rows = self._matrix_data.n_rows
        n_cols = self._matrix_data.n_cols

        # Run bootstrap test
        tester = BootstrapTester(
            n_bootstrap=n_bootstrap,
            seed=seed,
            use_parallel=self.config.bootstrap.use_parallel,
        )

        bootstrap_result = tester.test(
            rows, cols, values, errors,
            n_rows, n_cols,
            covariance=self._matrix_data.covariance if self.config.use_full_covariance else None,
        )

        # Run cross-checks
        cross_check_results = self._run_cross_checks()

        # Extract provenance from matrix data
        provenance_dict = None
        is_placeholder = False

        if self._matrix_data.provenance is not None:
            provenance_dict = self._matrix_data.provenance.to_dict()
            # Check if data is placeholder
            is_placeholder = (
                self._matrix_data.provenance.origin.value == "placeholder" or
                self._matrix_data.provenance.extra.get("is_placeholder", False)
            )

        # Also check metadata for placeholder flag
        if self._matrix_data.metadata.get("is_placeholder", False):
            is_placeholder = True

        # Build result object
        self._result = AnalysisResult(
            dataset_name=self.dataset.name,
            n_rows=n_rows,
            n_cols=n_cols,
            n_obs=len(values),
            chi2_rank1=bootstrap_result.chi2_rank1,
            chi2_rank2=bootstrap_result.chi2_rank2,
            ndof_rank1=bootstrap_result.fit_rank1.ndof,
            ndof_rank2=bootstrap_result.fit_rank2.ndof,
            lambda_stat=bootstrap_result.lambda_obs,
            p_value=bootstrap_result.p_value,
            p_value_ci=bootstrap_result.p_value_ci,
            n_bootstrap=n_bootstrap,
            rank1_converged=bootstrap_result.fit_rank1.success,
            rank2_converged=bootstrap_result.fit_rank2.success,
            is_stable=True,
            matrix_rank1=bootstrap_result.fit_rank1.predicted_matrix,
            matrix_rank2=bootstrap_result.fit_rank2.predicted_matrix,
            fit_rank1=bootstrap_result.fit_rank1,
            fit_rank2=bootstrap_result.fit_rank2,
            bootstrap_result=bootstrap_result,
            cross_checks=cross_check_results,
            config=self.config.model_dump() if self.config else None,
            seed=seed,
            provenance=provenance_dict,
            is_placeholder_data=is_placeholder,
        )

        # Save outputs
        if save_outputs:
            self._save_outputs()

        logger.info(self._result.summary_string())

        return self._result

    def _run_cross_checks(self) -> list[dict]:
        """Run all cross-checks and return results."""
        results = []

        # Dataset cross-checks
        for check in self.dataset.cross_checks():
            results.append({
                "name": check.name,
                "passed": check.passed,
                "message": check.message,
                "details": check.details,
            })

        # Add analysis-specific checks
        for check in self.additional_cross_checks():
            results.append(check)

        return results

    @abstractmethod
    def additional_cross_checks(self) -> list[dict]:
        """
        Run analysis-specific cross-checks.

        Override in subclasses to add dataset-specific validations.
        """
        pass

    def _save_outputs(self) -> None:
        """Save all outputs to disk."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save results JSON
        self._result.save(self.output_dir / "results.json")

        # Save config
        if self.config:
            config_path = self.output_dir / "config.json"
            with open(config_path, "w") as f:
                json.dump(self.config.model_dump(), f, indent=2)

        # Generate and save figures
        self.generate_figures()

        # Save summary markdown
        self._save_summary_markdown()

        logger.info(f"Outputs saved to {self.output_dir}")

    def _save_summary_markdown(self) -> None:
        """Generate summary markdown file."""
        md_path = self.output_dir / "summary.md"

        lines = [
            f"# {self.name.replace('_', ' ').title()} Rank-1 Test",
            "",
            "## Results",
            "",
            f"- **Dataset**: {self._result.dataset_name}",
            f"- **Matrix size**: {self._result.n_rows} × {self._result.n_cols}",
            f"- **Observations**: {self._result.n_obs}",
            "",
        ]

        # Quality warnings section
        if self._result.is_placeholder_data or self._result.provenance is None or self._result.is_underconstrained:
            lines.extend([
                "### Data Quality Warnings",
                "",
            ])
            if self._result.is_placeholder_data:
                lines.append("> **WARNING**: Using PLACEHOLDER DATA. Results are NOT valid for publication.")
            if self._result.provenance is None:
                lines.append("> **WARNING**: No provenance tracking. Data integrity cannot be verified.")
            if self._result.is_underconstrained:
                lines.append(f"> **WARNING**: Fit is underconstrained (ndof_rank1={self._result.ndof_rank1}, ndof_rank2={self._result.ndof_rank2}).")
            lines.append("")

        lines.extend([
            "### Fit Statistics",
            "",
            "| Model | χ² | ndof | χ²/ndof | Status |",
            "|-------|-----|------|---------|--------|",
        ])

        # Rank-1 row
        if self._result.ndof_rank1 > 0:
            lines.append(f"| Rank-1 | {self._result.chi2_rank1:.2f} | {self._result.ndof_rank1} | {self._result.chi2_rank1/self._result.ndof_rank1:.3f} | OK |")
        else:
            lines.append(f"| Rank-1 | {self._result.chi2_rank1:.2f} | {self._result.ndof_rank1} | N/A | **UNDERCONSTRAINED** |")

        # Rank-2 row
        if self._result.ndof_rank2 > 0:
            lines.append(f"| Rank-2 | {self._result.chi2_rank2:.2f} | {self._result.ndof_rank2} | {self._result.chi2_rank2/self._result.ndof_rank2:.3f} | OK |")
        else:
            lines.append(f"| Rank-2 | {self._result.chi2_rank2:.2f} | {self._result.ndof_rank2} | N/A | **UNDERCONSTRAINED** |")

        lines.extend([
            "",
            "### Hypothesis Test",
            "",
            f"- **Test statistic**: Λ = {self._result.lambda_stat:.2f}",
            f"- **p-value**: {self._result.p_value:.4f} (95% CI: [{self._result.p_value_ci[0]:.4f}, {self._result.p_value_ci[1]:.4f}])",
            f"- **Bootstrap samples**: {self._result.n_bootstrap}",
            "",
            "### Verdict",
            "",
            f"**{self._result.verdict}**: {self._result.verdict_reason}",
            "",
        ])

        lines.extend([
            "## Cross-checks",
            "",
        ])

        for check in self._result.cross_checks:
            status = "✓" if check["passed"] else "✗"
            lines.append(f"- [{status}] **{check['name']}**: {check['message']}")

        lines.extend([
            "",
            "## Provenance",
            "",
        ])
        if self._result.provenance:
            lines.append(f"- **Origin**: {self._result.provenance.get('origin', 'unknown')}")
            lines.append(f"- **Description**: {self._result.provenance.get('description', 'N/A')}")
            if self._result.provenance.get('sources'):
                lines.append(f"- **Sources**: {len(self._result.provenance['sources'])} source(s)")
        else:
            lines.append("*No provenance information available*")

        lines.extend([
            "",
            "## Figures",
            "",
            "![Residual heatmap](figures/residual_heatmap.png)",
            "",
            "![Bootstrap distribution](figures/bootstrap_distribution.png)",
            "",
        ])

        with open(md_path, "w") as f:
            f.write("\n".join(lines))

    @abstractmethod
    def generate_figures(self) -> None:
        """Generate analysis-specific figures."""
        pass

    def get_result(self) -> Optional[AnalysisResult]:
        """Get the analysis result (None if not run yet)."""
        return self._result
