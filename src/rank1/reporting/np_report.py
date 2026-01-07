"""
New Physics Sensitive (NP) mode report generation.

Generates comprehensive markdown reports for NP analysis including:
- Residual mode summary
- Localization metrics
- Stability across starts
- Stability across sweeps
- Global significance (look-elsewhere corrected)
- Replication checks
"""

from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import json

from rank1.logging import get_logger
from rank1.analysis.np_analysis import NPResult, NPVerdict

logger = get_logger()


class NPReportGenerator:
    """Generate comprehensive NP analysis reports."""

    def __init__(self, output_dir: Path):
        """
        Initialize report generator.

        Args:
            output_dir: Directory for report output
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_full_report(
        self,
        results: List[NPResult],
        title: str = "New Physics Sensitive Analysis Report",
    ) -> Path:
        """
        Generate full markdown report for multiple datasets.

        Args:
            results: List of NPResult from different datasets
            title: Report title

        Returns:
            Path to generated report
        """
        report_path = self.output_dir / "NP_REPORT.md"

        lines = [
            f"# {title}",
            "",
            f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
            "",
            "> **Note**: This analysis identifies structured residual modes that deviate",
            "> from rank-1 factorization. Results labeled as 'STRUCTURED_DEVIATION' are",
            "> candidates for further investigation, not claims of new physics.",
            "",
            "## Executive Summary",
            "",
            self._generate_executive_summary(results),
            "",
            "## Summary Table",
            "",
            self._generate_summary_table(results),
            "",
        ]

        # Per-dataset sections
        for result in results:
            lines.extend(self._generate_dataset_section(result))

        # Interpretation guide
        lines.extend(self._generate_interpretation_guide())

        # Methodology
        lines.extend(self._generate_methodology_section())

        # Write report
        with open(report_path, "w") as f:
            f.write("\n".join(lines))

        logger.info(f"NP report generated: {report_path}")
        return report_path

    def _generate_executive_summary(self, results: List[NPResult]) -> str:
        """Generate executive summary paragraph."""
        n_total = len(results)
        by_verdict = {}
        for r in results:
            v = r.np_verdict.value
            by_verdict[v] = by_verdict.get(v, 0) + 1

        n_structured = by_verdict.get("structured_deviation", 0)
        n_artifact = by_verdict.get("likely_artifact", 0)
        n_null = by_verdict.get("consistent_with_null", 0)
        n_inconclusive = by_verdict.get("inconclusive", 0)

        lines = [
            f"Analyzed {n_total} dataset(s) for structured deviations from rank-1 factorization:",
            "",
        ]

        if n_structured > 0:
            lines.append(f"- **{n_structured}** show structured deviations (candidates for investigation)")
        if n_artifact > 0:
            lines.append(f"- **{n_artifact}** show deviations likely due to artifacts or statistical fluctuation")
        if n_null > 0:
            lines.append(f"- **{n_null}** are consistent with the rank-1 null hypothesis")
        if n_inconclusive > 0:
            lines.append(f"- **{n_inconclusive}** have inconclusive results (fit issues or instability)")

        return "\n".join(lines)

    def _generate_summary_table(self, results: List[NPResult]) -> str:
        """Generate markdown summary table."""
        lines = [
            "| Dataset | Λ | p_local | p_global | Gini | Stability | Replication | Verdict |",
            "|---------|---|---------|----------|------|-----------|-------------|---------|",
        ]

        for r in results:
            gini = f"{r.localization_metrics.gini:.2f}" if r.localization_metrics else "N/A"
            stability = r.stability_metrics.stability_grade if r.stability_metrics else "N/A"
            repl = f"{r.replication_report.mean_replication_score:.2f}" if r.replication_report else "N/A"
            p_global = f"{r.global_significance.p_global:.4f}" if r.global_significance else "N/A"

            # Verdict formatting
            verdict = r.np_verdict.value.replace("_", " ").title()
            if r.np_verdict == NPVerdict.STRUCTURED_DEVIATION:
                verdict = f"**{verdict}**"
            elif r.np_verdict == NPVerdict.INCONCLUSIVE:
                verdict = f"*{verdict}*"

            lines.append(
                f"| {r.dataset} | {r.lambda_stat:.2f} | {r.p_local:.4f} | "
                f"{p_global} | {gini} | {stability} | {repl} | {verdict} |"
            )

        return "\n".join(lines)

    def _generate_dataset_section(self, result: NPResult) -> List[str]:
        """Generate detailed section for one dataset."""
        lines = [
            f"## {result.dataset.replace('_', ' ').title()} Analysis",
            "",
        ]

        # Verdict banner
        verdict_emoji = {
            NPVerdict.STRUCTURED_DEVIATION: "🔍",
            NPVerdict.LIKELY_ARTIFACT: "⚠️",
            NPVerdict.CONSISTENT_WITH_NULL: "✓",
            NPVerdict.INCONCLUSIVE: "❓",
        }
        emoji = verdict_emoji.get(result.np_verdict, "")
        lines.extend([
            f"### Verdict: {emoji} {result.np_verdict.value.replace('_', ' ').upper()}",
            "",
        ])

        # Reasons
        if result.np_reasons:
            for reason in result.np_reasons:
                lines.append(f"- {reason}")
            lines.append("")

        # Model comparison
        lines.extend([
            "### Model Comparison",
            "",
            f"- **Matrix size**: {result.n_rows} × {result.n_cols}",
            f"- **Observations**: {result.n_obs}",
            "",
            "| Model | χ² | ndof | χ²/ndof |",
            "|-------|-----|------|---------|",
            f"| Rank-1 (null) | {result.chi2_rank1:.2f} | {result.ndof_rank1} | "
            f"{result.chi2_rank1/max(1,result.ndof_rank1):.3f} |",
            f"| Rank-2 | {result.chi2_rank2:.2f} | {result.ndof_rank2} | "
            f"{result.chi2_rank2/max(1,result.ndof_rank2):.3f} |",
            "",
            f"**Test statistic**: Λ = {result.lambda_stat:.2f}",
            "",
            f"**Local p-value**: {result.p_local:.4f} "
            f"(95% CI: [{result.p_local_ci[0]:.4f}, {result.p_local_ci[1]:.4f}])",
            "",
        ])

        # Global significance
        if result.global_significance:
            gs = result.global_significance
            lines.extend([
                "### Global Significance (Look-Elsewhere Corrected)",
                "",
                f"- **Best preset**: {gs.best_preset}",
                f"- **T_obs (max Λ)**: {gs.T_obs:.2f}",
                f"- **Local p at best**: {gs.p_local_best:.4f}",
                f"- **Global p-value**: {gs.p_global:.4f}",
                f"- **Number of presets**: {gs.n_presets}",
                f"- **Bootstrap samples**: {gs.n_bootstrap}",
                "",
            ])

        # Residual mode
        if result.residual_mode:
            lines.extend([
                "### Residual Mode Summary",
                "",
            ])

            rm = result.residual_mode

            # v2 summary
            if rm.col_labels and len(rm.col_labels) <= 10:
                lines.append("**v2 (column/observable dependence)**:")
                lines.append("")
                lines.append("| Column | v2 |")
                lines.append("|--------|-----|")
                for i, label in enumerate(rm.col_labels):
                    lines.append(f"| {label} | {rm.v2[i]:.4f} |")
                lines.append("")
            else:
                lines.append(f"**v2**: {len(rm.v2)} elements, peak at index {result.localization_metrics.peak_index if result.localization_metrics else 'N/A'}")
                lines.append("")

            # u2 summary
            if rm.row_labels and len(rm.row_labels) <= 10:
                lines.append("**u2 (row/condition dependence)**:")
                lines.append("")
                lines.append("| Row | u2 |")
                lines.append("|-----|-----|")
                for i, label in enumerate(rm.row_labels):
                    lines.append(f"| {label} | {rm.u2[i]:.4f} |")
                lines.append("")
            else:
                lines.append(f"**u2**: {len(rm.u2)} elements")
                lines.append("")

        # Localization metrics
        if result.localization_metrics:
            loc = result.localization_metrics
            lines.extend([
                "### Localization Metrics",
                "",
                f"- **Peak index**: {loc.peak_index} (value: {loc.peak_value:.4f})",
                f"- **Gini coefficient**: {loc.gini:.3f} (0=uniform, 1=concentrated)",
                f"- **Normalized entropy**: {loc.normalized_entropy:.3f} (0=localized, 1=diffuse)",
                "",
            ])

            if loc.top_k_mass:
                lines.append("**Top-k mass** (fraction of ||v2||² in top k elements):")
                lines.append("")
                for k, mass in sorted(loc.top_k_mass.items()):
                    lines.append(f"- k={k}: {mass:.1%}")
                lines.append("")

            if loc.window_concentration:
                lines.append("**Window concentration** (max fraction in contiguous window):")
                lines.append("")
                for w, conc in sorted(loc.window_concentration.items()):
                    lines.append(f"- window={w}: {conc:.1%}")
                lines.append("")

        # Stability metrics
        if result.stability_metrics:
            stab = result.stability_metrics
            lines.extend([
                "### Stability Metrics",
                "",
                f"- **Grade**: {stab.stability_grade}",
                f"- **Number of starts**: {stab.n_starts}",
                f"- **v2 cosine similarity**: {stab.v2_cosine_mean:.3f} ± {stab.v2_cosine_std:.3f}",
                f"- **u2 cosine similarity**: {stab.u2_cosine_mean:.3f} ± {stab.u2_cosine_std:.3f}",
                "",
            ])

            if stab.n_bootstrap > 0:
                lines.extend([
                    f"- **Bootstrap v2 cosine**: {stab.v2_bootstrap_cosine_mean:.3f} ± {stab.v2_bootstrap_cosine_std:.3f}",
                    "",
                ])

        # Replication
        if result.replication_report:
            repl = result.replication_report
            lines.extend([
                "### Replication Checks",
                "",
                f"- **Number of comparisons**: {repl.n_comparisons}",
                f"- **Mean v2 cosine**: {repl.mean_v2_cosine:.3f}",
                f"- **Mean replication score**: {repl.mean_replication_score:.3f}",
                f"- **All replicate**: {'Yes' if repl.all_replicate else 'No'}",
                "",
            ])

            if repl.comparisons:
                lines.append("| Comparison | v2 Cosine | Score | Grade |")
                lines.append("|------------|-----------|-------|-------|")
                for comp in repl.comparisons[:10]:  # Limit to 10
                    lines.append(
                        f"| {comp.source_a} vs {comp.source_b} | "
                        f"{comp.v2_cosine:.3f} | {comp.replication_score:.3f} | "
                        f"{comp.replication_grade} |"
                    )
                lines.append("")

        # Sweep results
        if result.sweep_results:
            lines.extend([
                "### Sweep Results",
                "",
                "| Preset | Λ | p_local | Peak | Gini |",
                "|--------|---|---------|------|------|",
            ])

            for sr in result.sweep_results:
                peak = sr.v2_peak_index if sr.v2_peak_index is not None else "N/A"
                gini = f"{sr.v2_gini:.2f}" if sr.v2_gini is not None else "N/A"
                lines.append(
                    f"| {sr.preset.name} | {sr.lambda_stat:.2f} | "
                    f"{sr.p_local:.4f} | {peak} | {gini} |"
                )
            lines.append("")

        # Figure references
        lines.extend([
            "### Figures",
            "",
            f"![Residual Heatmap](figures/{result.dataset}_residual_heatmap.png)",
            "",
            f"![v2 Shape](figures/{result.dataset}_v2_shape.png)",
            "",
            f"![u2 Dependence](figures/{result.dataset}_u2_dependence.png)",
            "",
            "---",
            "",
        ])

        return lines

    def _generate_interpretation_guide(self) -> List[str]:
        """Generate interpretation guide section."""
        return [
            "## Interpretation Guide",
            "",
            "### Verdict Categories",
            "",
            "| Verdict | Meaning | Action |",
            "|---------|---------|--------|",
            "| STRUCTURED_DEVIATION | Localized, stable, replicating residual with global significance | Investigate further; check systematics |",
            "| LIKELY_ARTIFACT | Deviation detected but unstable, diffuse, or doesn't replicate | Likely statistical fluctuation or systematic |",
            "| CONSISTENT_WITH_NULL | No significant deviation from rank-1 | Data compatible with factorizable model |",
            "| INCONCLUSIVE | Fit issues or unstable results | Check data quality and fit diagnostics |",
            "",
            "### Key Metrics",
            "",
            "- **Λ (Lambda)**: Test statistic = χ²(rank-1) - χ²(rank-2). Larger = stronger deviation.",
            "- **p_local**: Bootstrap p-value at single preset. Raw significance before look-elsewhere.",
            "- **p_global**: Look-elsewhere corrected p-value across all sweeps. True global significance.",
            "- **Gini**: Concentration measure for v2 (0=uniform, 1=single-bin). Higher = more localized.",
            "- **Normalized entropy**: Diffuseness of v2 (0=delta, 1=uniform). Lower = more localized.",
            "- **Replication score**: Agreement of residual mode across conditions (0-1). Higher = more robust.",
            "",
            "### Caveats",
            "",
            "1. **Not claims of new physics**: These are statistical anomalies requiring follow-up.",
            "2. **Systematic uncertainties**: Not all systematics may be included; interpret with caution.",
            "3. **Look-elsewhere effect**: Multiple testing inflates false positives; use p_global.",
            "4. **Replication is key**: Isolated deviations are likely fluctuations.",
            "",
        ]

    def _generate_methodology_section(self) -> List[str]:
        """Generate methodology section."""
        return [
            "## Methodology",
            "",
            "### Rank-1 vs Rank-2 Test",
            "",
            "The analysis tests whether data follow a rank-1 factorizable model:",
            "",
            "```",
            "M_ij = u_i × v_j + ε_ij",
            "```",
            "",
            "vs a rank-2 extension:",
            "",
            "```",
            "M_ij = u1_i × v1_j + u2_i × v2_j + ε_ij",
            "```",
            "",
            "The test statistic Λ = χ²(rank-1) - χ²(rank-2) quantifies improvement from the extra mode.",
            "",
            "### Bootstrap P-Value",
            "",
            "Under the null (rank-1 true), we estimate the Λ distribution via parametric bootstrap:",
            "",
            "1. Generate pseudo-data from fitted rank-1 model",
            "2. Fit both rank-1 and rank-2 to pseudo-data",
            "3. Compute Λ for each bootstrap sample",
            "4. p = (k + 1) / (B + 1) where k = count(Λ_boot ≥ Λ_obs)",
            "",
            "### Look-Elsewhere Correction",
            "",
            "When testing multiple configurations (sweeps), we correct for look-elsewhere effect:",
            "",
            "1. Define family of presets S (pre-registered, not data-driven)",
            "2. Compute T_obs = max over s in S of Λ_s",
            "3. Bootstrap the null distribution of T = max Λ_s",
            "4. p_global = P(T ≥ T_obs | null)",
            "",
            "### Residual Mode Analysis",
            "",
            "The second mode (u2, v2) represents the structured residual:",
            "",
            "- **Sign convention**: v2 is oriented so max|v2| element is positive",
            "- **Normalization**: ||v2||₂ = 1, scale absorbed into u2",
            "- **Localization**: Gini and entropy quantify where v2 concentrates",
            "- **Stability**: Cosine similarity across multi-start fits",
            "- **Replication**: Agreement across independent data slices",
            "",
        ]

    def generate_individual_report(
        self,
        result: NPResult,
        output_dir: Optional[Path] = None,
    ) -> Path:
        """Generate individual report for one dataset."""
        if output_dir is None:
            output_dir = self.output_dir / result.dataset

        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "np_report.md"

        lines = [
            f"# NP Analysis: {result.dataset.replace('_', ' ').title()}",
            "",
            f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
            "",
        ]

        lines.extend(self._generate_dataset_section(result))
        lines.extend(self._generate_interpretation_guide())

        with open(report_path, "w") as f:
            f.write("\n".join(lines))

        logger.info(f"Individual NP report generated: {report_path}")
        return report_path
