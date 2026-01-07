"""
Markdown report generation for rank-1 analysis.

Includes full provenance audit trail for reproducibility.
"""

from pathlib import Path
from datetime import datetime
from typing import Any, Optional
import json

import pandas as pd

from rank1.analysis.base import AnalysisResult
from rank1.reporting.tables import TableGenerator
from rank1.reporting.figures import FigureGenerator
from rank1.provenance import DataProvenance
from rank1.logging import get_logger

logger = get_logger()


class ReportGenerator:
    """Generate comprehensive markdown reports from analysis results."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.table_gen = TableGenerator(self.output_dir / "tables")
        self.fig_gen = FigureGenerator(self.output_dir / "figures")

    def generate_full_report(
        self,
        results: list[AnalysisResult],
        title: str = "Rank-1 Factorization Analysis Report",
    ) -> Path:
        """
        Generate a full markdown report.

        Args:
            results: List of analysis results
            title: Report title

        Returns:
            Path to generated report
        """
        report_path = self.output_dir / "REPORT.md"

        # Generate tables
        tables = self.table_gen.generate_all_tables(results)

        # Generate summary figure
        summary_fig = self.fig_gen.create_summary_figure(results)
        self.fig_gen.save_figure(summary_fig, "summary_comparison")

        # Build report content
        lines = [
            f"# {title}",
            "",
            f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
            "",
            "## Executive Summary",
            "",
            self._generate_executive_summary(results),
            "",
            "## Summary Table",
            "",
            tables["summary"].to_markdown(index=False),
            "",
            "## Results by Dataset",
            "",
        ]

        # Add per-dataset sections
        for result in results:
            lines.extend(self._generate_dataset_section(result))

        # Add cross-checks section
        lines.extend([
            "## Cross-Check Summary",
            "",
            tables["cross_checks"].to_markdown(index=False),
            "",
        ])

        # Add provenance audit trail
        lines.extend(self._generate_provenance_section(results))

        # Add methodology section
        lines.extend(self._generate_methodology_section())

        # Add references
        lines.extend(self._generate_references(results))

        # Write report
        with open(report_path, "w") as f:
            f.write("\n".join(lines))

        # Also save provenance as JSON for programmatic access
        self._save_provenance_json(results)

        logger.info(f"Report generated: {report_path}")
        return report_path

    def _generate_executive_summary(self, results: list[AnalysisResult]) -> str:
        """Generate executive summary paragraph."""
        n_total = len(results)
        n_rejected = sum(1 for r in results if r.is_significant)
        n_passed = n_total - n_rejected

        if n_rejected == 0:
            conclusion = (
                f"All {n_total} datasets are consistent with rank-1 factorization "
                "at the 5% significance level."
            )
        elif n_rejected == n_total:
            conclusion = (
                f"All {n_total} datasets show significant deviations from rank-1 "
                "factorization at the 5% significance level."
            )
        else:
            conclusion = (
                f"Of {n_total} datasets analyzed, {n_passed} are consistent with "
                f"rank-1 factorization, while {n_rejected} show significant deviations "
                "at the 5% significance level."
            )

        return conclusion

    def _generate_dataset_section(self, result: AnalysisResult) -> list[str]:
        """Generate markdown section for a single dataset."""
        lines = [
            f"### {result.dataset_name.replace('_', ' ').title()}",
            "",
            f"- **Matrix size**: {result.n_rows} × {result.n_cols}",
            f"- **Observations**: {result.n_obs}",
            "",
            "| Metric | Rank-1 | Rank-2 |",
            "|--------|--------|--------|",
            f"| χ² | {result.chi2_rank1:.2f} | {result.chi2_rank2:.2f} |",
            f"| ndof | {result.ndof_rank1} | {result.ndof_rank2} |",
            f"| χ²/ndof | {result.chi2_rank1/max(1,result.ndof_rank1):.3f} | {result.chi2_rank2/max(1,result.ndof_rank2):.3f} |",
            "",
            f"**Test statistic**: Λ = {result.lambda_stat:.2f}",
            "",
            f"**p-value**: {result.p_value:.4f} (95% CI: [{result.p_value_ci[0]:.4f}, {result.p_value_ci[1]:.4f}])",
            "",
        ]

        if result.is_significant:
            lines.append("**Conclusion**: Rank-1 hypothesis **rejected** at 5% level.")
        else:
            lines.append("**Conclusion**: Rank-1 hypothesis **not rejected** at 5% level.")

        # Add figure references
        lines.extend([
            "",
            f"![Residuals]({result.dataset_name}/figures/residual_heatmap.png)",
            "",
            f"![Bootstrap]({result.dataset_name}/figures/bootstrap_distribution.png)",
            "",
        ])

        return lines

    def _generate_methodology_section(self) -> list[str]:
        """Generate methodology section."""
        return [
            "## Methodology",
            "",
            "### Rank-1 Factorization Model",
            "",
            "For a matrix M with observations at positions (i, j), the rank-1 model assumes:",
            "",
            "```",
            "M_ij = u_i × v_j",
            "```",
            "",
            "where u and v are vectors encoding the row and column factors respectively.",
            "",
            "### Test Statistic",
            "",
            "We use the likelihood ratio test statistic:",
            "",
            "```",
            "Λ = χ²(rank-1) - χ²(rank-2)",
            "```",
            "",
            "where χ² is computed as the weighted sum of squared residuals.",
            "",
            "### Parametric Bootstrap",
            "",
            "To compute p-values, we use parametric bootstrap:",
            "",
            "1. Fit the rank-1 model to observed data",
            "2. Generate B pseudo-datasets by adding Gaussian noise",
            "3. Refit both rank-1 and rank-2 models to each pseudo-dataset",
            "4. Compute Λ for each pseudo-dataset",
            "5. p-value = (k + 1) / (B + 1), where k is the count of Λ_boot ≥ Λ_obs",
            "",
            "### Gauge Fixing",
            "",
            "To avoid scale degeneracy, we fix ||v||₂ = 1 for rank-1 models.",
            "",
        ]

    def _generate_references(self, results: list[AnalysisResult]) -> list[str]:
        """Generate references section."""
        lines = [
            "## Data Sources",
            "",
        ]

        # Collect unique DOIs
        dois = set()
        urls = set()

        for r in results:
            if r.config and "source_dois" in r.config.get("metadata", {}):
                dois.update(r.config["metadata"]["source_dois"])

        lines.append("### DOIs")
        lines.append("")
        for doi in sorted(dois):
            lines.append(f"- [{doi}](https://doi.org/{doi})")

        lines.extend([
            "",
            "### URLs",
            "",
            "- [HEPData](https://www.hepdata.net/)",
            "- [CERN Open Data](http://opendata.cern.ch/)",
            "- [arXiv](https://arxiv.org/)",
            "",
        ])

        return lines

    def _generate_provenance_section(self, results: list[AnalysisResult]) -> list[str]:
        """
        Generate provenance audit trail section.

        This section provides full transparency on:
        - Data sources and their origins
        - File hashes for verification
        - Any fallbacks that occurred
        - Filters/selections applied
        """
        lines = [
            "## Data Provenance Audit Trail",
            "",
            "This section documents the origin and processing of all data used in this analysis.",
            "",
        ]

        has_warnings = False

        for result in results:
            lines.append(f"### {result.dataset_name.replace('_', ' ').title()}")
            lines.append("")

            # Check if provenance exists in metadata
            provenance = None
            if hasattr(result, 'matrix_data') and result.matrix_data:
                provenance = getattr(result.matrix_data, 'provenance', None)

            if provenance is None:
                # Try to get from config
                if result.config and 'provenance' in result.config:
                    provenance = DataProvenance.from_dict(result.config['provenance'])

            if provenance is None:
                lines.extend([
                    "**WARNING**: No provenance information available for this dataset.",
                    "",
                    "This may indicate:",
                    "- Data was loaded from cache without provenance tracking",
                    "- Older version of the pipeline was used",
                    "",
                ])
                has_warnings = True
                continue

            # Origin
            lines.append(f"**Data Origin**: `{provenance.origin.value}`")
            if provenance.origin_details:
                lines.append(f"  - {provenance.origin_details}")
            lines.append("")

            # Sources
            if provenance.sources:
                lines.append("**Data Sources**:")
                lines.append("")
                for src in provenance.sources:
                    src_line = f"- **{src.source_type}**: {src.identifier}"
                    if src.doi:
                        src_line += f" ([DOI:{src.doi}](https://doi.org/{src.doi}))"
                    lines.append(src_line)
                    if src.table_names:
                        lines.append(f"  - Tables: {', '.join(src.table_names)}")
                lines.append("")

            # File hashes
            if provenance.input_files:
                lines.append("**Input File Hashes** (SHA256):")
                lines.append("")
                lines.append("| File | Hash (first 16 chars) | Size |")
                lines.append("|------|----------------------|------|")
                for f in provenance.input_files[:10]:  # Limit to 10
                    fname = Path(f.path).name
                    lines.append(f"| {fname} | `{f.sha256[:16]}...` | {f.size_bytes:,} bytes |")
                if len(provenance.input_files) > 10:
                    lines.append(f"| ... | ({len(provenance.input_files) - 10} more files) | |")
                lines.append("")

            # Filters
            if provenance.selection_filters:
                lines.append("**Selection Filters Applied**:")
                lines.append("")
                for filt in provenance.selection_filters:
                    if filt.n_before and filt.n_after:
                        lines.append(
                            f"- **{filt.name}**: {filt.n_before} → {filt.n_after} observations"
                        )
                    else:
                        lines.append(f"- **{filt.name}**: {filt.description}")
                    if filt.parameters:
                        for k, v in filt.parameters.items():
                            lines.append(f"  - {k}: {v}")
                lines.append("")

            # Fallbacks
            if provenance.fallbacks:
                has_warnings = True
                lines.append("**FALLBACKS OCCURRED**:")
                lines.append("")
                lines.append("> ⚠️ The following fallbacks were triggered during data loading.")
                lines.append("> This may affect result validity. Verify data manually.")
                lines.append("")
                for fb in provenance.fallbacks:
                    lines.append(f"- **{fb.step}**: {fb.reason}")
                    lines.append(f"  - Fallback: {fb.fallback_to}")
                    if fb.original_source:
                        lines.append(f"  - Original: {fb.original_source}")
                lines.append("")

            # Timestamps
            if provenance.fetch_timestamp or provenance.build_timestamp:
                lines.append("**Timestamps**:")
                lines.append("")
                if provenance.fetch_timestamp:
                    lines.append(f"- Fetched: {provenance.fetch_timestamp}")
                if provenance.build_timestamp:
                    lines.append(f"- Built: {provenance.build_timestamp}")
                lines.append("")

            lines.append("---")
            lines.append("")

        # Summary warning
        if has_warnings:
            lines.insert(4, "> ⚠️ **Some datasets have provenance warnings.** See details below.")
            lines.insert(5, "")

        return lines

    def _save_provenance_json(self, results: list[AnalysisResult]) -> None:
        """Save complete provenance information as JSON."""
        provenance_path = self.output_dir / "provenance.json"

        provenance_data = {
            "report_generated": datetime.utcnow().isoformat() + "Z",
            "datasets": {},
        }

        for result in results:
            provenance = None
            if hasattr(result, 'matrix_data') and result.matrix_data:
                provenance = getattr(result.matrix_data, 'provenance', None)

            if provenance is None and result.config and 'provenance' in result.config:
                provenance = DataProvenance.from_dict(result.config['provenance'])

            if provenance:
                provenance_data["datasets"][result.dataset_name] = provenance.to_dict()
            else:
                provenance_data["datasets"][result.dataset_name] = {
                    "error": "No provenance available"
                }

        with open(provenance_path, "w") as f:
            json.dump(provenance_data, f, indent=2)

        logger.info(f"Provenance JSON saved: {provenance_path}")

    def generate_individual_summary(
        self,
        result: AnalysisResult,
        output_dir: Optional[Path] = None,
    ) -> Path:
        """Generate individual summary for one analysis."""
        if output_dir is None:
            output_dir = self.output_dir / result.dataset_name

        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = output_dir / "summary.md"

        lines = [
            f"# {result.dataset_name.replace('_', ' ').title()} Analysis",
            "",
            f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
            "",
            result.summary_string(),
            "",
            "## Figures",
            "",
            "![Residual Heatmap](figures/residual_heatmap.png)",
            "",
            "![Bootstrap Distribution](figures/bootstrap_distribution.png)",
            "",
        ]

        with open(summary_path, "w") as f:
            f.write("\n".join(lines))

        return summary_path
