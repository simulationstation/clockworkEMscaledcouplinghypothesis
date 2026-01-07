"""
Table generation utilities for rank-1 analysis reports.
"""

from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

from rank1.analysis.base import AnalysisResult
from rank1.logging import get_logger

logger = get_logger()


class TableGenerator:
    """Generate summary tables from analysis results."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_summary_table(
        self,
        results: list[AnalysisResult],
    ) -> pd.DataFrame:
        """Create summary table of all analyses."""
        records = []

        for r in results:
            records.append({
                "Dataset": r.dataset_name,
                "Matrix Size": f"{r.n_rows}×{r.n_cols}",
                "N_obs": r.n_obs,
                "χ²(R1)": f"{r.chi2_rank1:.2f}",
                "χ²(R2)": f"{r.chi2_rank2:.2f}",
                "Λ": f"{r.lambda_stat:.2f}",
                "p-value": f"{r.p_value:.4f}",
                "95% CI": f"[{r.p_value_ci[0]:.4f}, {r.p_value_ci[1]:.4f}]",
                "Significant": "Yes" if r.is_significant else "No",
            })

        return pd.DataFrame(records)

    def create_chi2_table(
        self,
        results: list[AnalysisResult],
    ) -> pd.DataFrame:
        """Create detailed chi-squared table."""
        records = []

        for r in results:
            chi2_ndof_r1 = r.chi2_rank1 / max(1, r.ndof_rank1)
            chi2_ndof_r2 = r.chi2_rank2 / max(1, r.ndof_rank2)

            records.append({
                "Dataset": r.dataset_name,
                "χ²_rank1": r.chi2_rank1,
                "ndof_rank1": r.ndof_rank1,
                "χ²/ndof_rank1": chi2_ndof_r1,
                "χ²_rank2": r.chi2_rank2,
                "ndof_rank2": r.ndof_rank2,
                "χ²/ndof_rank2": chi2_ndof_r2,
                "Δχ²": r.lambda_stat,
            })

        return pd.DataFrame(records)

    def create_cross_check_table(
        self,
        results: list[AnalysisResult],
    ) -> pd.DataFrame:
        """Create cross-check summary table."""
        records = []

        for r in results:
            for check in r.cross_checks:
                records.append({
                    "Dataset": r.dataset_name,
                    "Check": check["name"],
                    "Passed": "✓" if check["passed"] else "✗",
                    "Message": check["message"],
                })

        return pd.DataFrame(records)

    def save_table(
        self,
        df: pd.DataFrame,
        name: str,
        formats: list[str] = ["csv", "md"],
    ) -> list[Path]:
        """Save table in multiple formats."""
        paths = []

        for fmt in formats:
            path = self.output_dir / f"{name}.{fmt}"

            if fmt == "csv":
                df.to_csv(path, index=False)
            elif fmt == "md":
                with open(path, "w") as f:
                    f.write(df.to_markdown(index=False))
            elif fmt == "html":
                df.to_html(path, index=False)
            elif fmt == "latex":
                with open(path, "w") as f:
                    f.write(df.to_latex(index=False))

            paths.append(path)

        return paths

    def generate_all_tables(
        self,
        results: list[AnalysisResult],
    ) -> dict[str, pd.DataFrame]:
        """Generate and save all standard tables."""
        tables = {}

        # Summary table
        summary = self.create_summary_table(results)
        self.save_table(summary, "summary")
        tables["summary"] = summary

        # Chi2 table
        chi2 = self.create_chi2_table(results)
        self.save_table(chi2, "chi2_details")
        tables["chi2"] = chi2

        # Cross-check table
        checks = self.create_cross_check_table(results)
        self.save_table(checks, "cross_checks")
        tables["cross_checks"] = checks

        return tables
