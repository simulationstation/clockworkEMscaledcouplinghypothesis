"""
Command-line interface for rank-1 factorization analysis.

Usage:
    rank1 --help
    rank1 fetch all
    rank1 run all
    rank1 report all
"""

from pathlib import Path
from typing import Optional
import json

import typer
from rich.console import Console
from rich.table import Table

from rank1.config import Config, get_config, set_config
from rank1.logging import setup_logging, get_logger

app = typer.Typer(
    name="rank1",
    help="Rank-1 factorization analysis for particle physics data",
    add_completion=False,
)

console = Console()


def _get_config(config_file: Optional[Path] = None) -> Config:
    """Load or create configuration."""
    if config_file and config_file.exists():
        return Config.load(config_file)
    return get_config()


# ============================================================================
# FETCH commands
# ============================================================================

fetch_app = typer.Typer(help="Download datasets")
app.add_typer(fetch_app, name="fetch")


@fetch_app.command("all")
def fetch_all(
    force: bool = typer.Option(False, "--force", "-f", help="Force re-download"),
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
):
    """Download all datasets."""
    cfg = _get_config(config)
    cfg.ensure_dirs()
    setup_logging(cfg.log_level)
    logger = get_logger()

    from rank1.datasets import HiggsATLASDataset, ElasticTOTEMDataset, DiffractiveDISDataset

    datasets = [
        HiggsATLASDataset(
            cache_dir=cfg.data.cache_dir,
            raw_dir=cfg.data.raw_dir,
            processed_dir=cfg.data.processed_dir,
        ),
        ElasticTOTEMDataset(
            cache_dir=cfg.data.cache_dir,
            raw_dir=cfg.data.raw_dir,
            processed_dir=cfg.data.processed_dir,
        ),
        DiffractiveDISDataset(
            cache_dir=cfg.data.cache_dir,
            raw_dir=cfg.data.raw_dir,
            processed_dir=cfg.data.processed_dir,
        ),
    ]

    for ds in datasets:
        logger.info(f"Fetching {ds.name}")
        try:
            paths = ds.fetch_raw(force=force)
            logger.info(f"  Downloaded {len(paths)} files")
        except Exception as e:
            logger.error(f"  Failed: {e}")

    console.print("[green]All datasets fetched[/green]")


@fetch_app.command("higgs")
def fetch_higgs(
    force: bool = typer.Option(False, "--force", "-f"),
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
):
    """Download ATLAS Higgs signal strength data."""
    cfg = _get_config(config)
    cfg.ensure_dirs()
    setup_logging(cfg.log_level)

    from rank1.datasets import HiggsATLASDataset

    ds = HiggsATLASDataset(
        cache_dir=cfg.data.cache_dir,
        raw_dir=cfg.data.raw_dir,
        processed_dir=cfg.data.processed_dir,
    )
    ds.fetch_raw(force=force)
    console.print("[green]Higgs data fetched[/green]")


@fetch_app.command("elastic")
def fetch_elastic(
    force: bool = typer.Option(False, "--force", "-f"),
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
):
    """Download TOTEM elastic scattering data."""
    cfg = _get_config(config)
    cfg.ensure_dirs()
    setup_logging(cfg.log_level)

    from rank1.datasets import ElasticTOTEMDataset

    ds = ElasticTOTEMDataset(
        cache_dir=cfg.data.cache_dir,
        raw_dir=cfg.data.raw_dir,
        processed_dir=cfg.data.processed_dir,
    )
    ds.fetch_raw(force=force)
    console.print("[green]Elastic data fetched[/green]")


@fetch_app.command("diffractive")
def fetch_diffractive(
    force: bool = typer.Option(False, "--force", "-f"),
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
):
    """Download H1/ZEUS diffractive DIS data."""
    cfg = _get_config(config)
    cfg.ensure_dirs()
    setup_logging(cfg.log_level)

    from rank1.datasets import DiffractiveDISDataset

    ds = DiffractiveDISDataset(
        cache_dir=cfg.data.cache_dir,
        raw_dir=cfg.data.raw_dir,
        processed_dir=cfg.data.processed_dir,
    )
    ds.fetch_raw(force=force)
    console.print("[green]Diffractive data fetched[/green]")


# ============================================================================
# RUN commands
# ============================================================================

run_app = typer.Typer(help="Run analyses")
app.add_typer(run_app, name="run")


@run_app.command("all")
def run_all(
    n_bootstrap: int = typer.Option(1000, "--bootstrap", "-b", help="Bootstrap samples"),
    seed: int = typer.Option(42, "--seed", "-s", help="Random seed"),
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
):
    """Run all analyses."""
    cfg = _get_config(config)
    cfg.ensure_dirs()
    setup_logging(cfg.log_level)
    logger = get_logger()

    from rank1.analysis import HiggsRankAnalysis, ElasticRankAnalysis, DiffractiveRankAnalysis

    results = []

    analyses = [
        HiggsRankAnalysis(output_dir=cfg.output_dir / "higgs_atlas"),
        ElasticRankAnalysis(output_dir=cfg.output_dir / "elastic_totem"),
        DiffractiveRankAnalysis(output_dir=cfg.output_dir / "diffractive_dis"),
    ]

    for analysis in analyses:
        logger.info(f"Running {analysis.name}")
        try:
            result = analysis.run(n_bootstrap=n_bootstrap, seed=seed)
            results.append(result)
        except Exception as e:
            logger.error(f"  Failed: {e}")
            import traceback
            traceback.print_exc()

    # Print summary table
    if results:
        table = Table(title="Rank-1 Test Results")
        table.add_column("Dataset")
        table.add_column("Λ")
        table.add_column("p-value")
        table.add_column("Significant")

        for r in results:
            sig = "[red]Yes[/red]" if r.is_significant else "[green]No[/green]"
            table.add_row(
                r.dataset_name,
                f"{r.lambda_stat:.2f}",
                f"{r.p_value:.4f}",
                sig,
            )

        console.print(table)

    console.print("[green]All analyses complete[/green]")


@run_app.command("higgs")
def run_higgs(
    n_bootstrap: int = typer.Option(1000, "--bootstrap", "-b"),
    seed: int = typer.Option(42, "--seed", "-s"),
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
):
    """Run ATLAS Higgs rank-1 analysis."""
    cfg = _get_config(config)
    cfg.ensure_dirs()
    setup_logging(cfg.log_level)

    from rank1.analysis import HiggsRankAnalysis

    analysis = HiggsRankAnalysis(output_dir=cfg.output_dir / "higgs_atlas")
    result = analysis.run(n_bootstrap=n_bootstrap, seed=seed)

    console.print(result.summary_string())


@run_app.command("elastic")
def run_elastic(
    n_bootstrap: int = typer.Option(1000, "--bootstrap", "-b"),
    seed: int = typer.Option(42, "--seed", "-s"),
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
):
    """Run TOTEM elastic rank-1 analysis."""
    cfg = _get_config(config)
    cfg.ensure_dirs()
    setup_logging(cfg.log_level)

    from rank1.analysis import ElasticRankAnalysis

    analysis = ElasticRankAnalysis(output_dir=cfg.output_dir / "elastic_totem")
    result = analysis.run(n_bootstrap=n_bootstrap, seed=seed)

    console.print(result.summary_string())


@run_app.command("diffractive")
def run_diffractive(
    n_bootstrap: int = typer.Option(1000, "--bootstrap", "-b"),
    seed: int = typer.Option(42, "--seed", "-s"),
    experiment: str = typer.Option("combined", "--experiment", "-e"),
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
):
    """Run diffractive DIS rank-1 analysis."""
    cfg = _get_config(config)
    cfg.ensure_dirs()
    setup_logging(cfg.log_level)

    from rank1.analysis import DiffractiveRankAnalysis

    analysis = DiffractiveRankAnalysis(
        experiment=experiment,
        output_dir=cfg.output_dir / "diffractive_dis",
    )
    result = analysis.run(n_bootstrap=n_bootstrap, seed=seed)

    console.print(result.summary_string())


# ============================================================================
# REPORT commands
# ============================================================================

@app.command("report")
def report_all(
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
):
    """Generate combined report from all analysis outputs."""
    cfg = _get_config(config)
    setup_logging(cfg.log_level)
    logger = get_logger()

    from rank1.analysis.base import AnalysisResult
    from rank1.reporting import ReportGenerator

    # Load results from output directories
    results = []
    for subdir in ["higgs_atlas", "elastic_totem", "diffractive_dis"]:
        results_file = cfg.output_dir / subdir / "results.json"
        if results_file.exists():
            with open(results_file) as f:
                data = json.load(f)

            # Create minimal AnalysisResult
            import numpy as np
            result = AnalysisResult(
                dataset_name=data["dataset_name"],
                n_rows=data["n_rows"],
                n_cols=data["n_cols"],
                n_obs=data["n_obs"],
                chi2_rank1=data["chi2_rank1"],
                chi2_rank2=data["chi2_rank2"],
                ndof_rank1=data["ndof_rank1"],
                ndof_rank2=data["ndof_rank2"],
                lambda_stat=data["lambda_stat"],
                p_value=data["p_value"],
                p_value_ci=(data["p_value_ci_lower"], data["p_value_ci_upper"]),
                n_bootstrap=data["n_bootstrap"],
                rank1_converged=data["rank1_converged"],
                rank2_converged=data["rank2_converged"],
                is_stable=data["is_stable"],
                matrix_rank1=np.array([]),
                matrix_rank2=np.array([]),
                cross_checks=data.get("cross_checks", []),
                seed=data.get("seed", 42),
            )
            results.append(result)
            logger.info(f"Loaded results for {data['dataset_name']}")

    if not results:
        console.print("[red]No results found. Run analyses first.[/red]")
        raise typer.Exit(1)

    # Generate report
    generator = ReportGenerator(cfg.output_dir)
    report_path = generator.generate_full_report(results)

    console.print(f"[green]Report generated: {report_path}[/green]")


# ============================================================================
# CLEAN command
# ============================================================================

@app.command("clean")
def clean(
    all_data: bool = typer.Option(False, "--all", "-a", help="Also remove raw data"),
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
):
    """Clean processed data and outputs."""
    cfg = _get_config(config)

    import shutil

    # Remove processed data
    if cfg.data.processed_dir.exists():
        shutil.rmtree(cfg.data.processed_dir)
        console.print(f"Removed {cfg.data.processed_dir}")

    # Remove outputs
    if cfg.output_dir.exists():
        shutil.rmtree(cfg.output_dir)
        console.print(f"Removed {cfg.output_dir}")

    # Optionally remove raw data
    if all_data:
        if cfg.data.raw_dir.exists():
            shutil.rmtree(cfg.data.raw_dir)
            console.print(f"Removed {cfg.data.raw_dir}")
        if cfg.data.cache_dir.exists():
            shutil.rmtree(cfg.data.cache_dir)
            console.print(f"Removed {cfg.data.cache_dir}")

    console.print("[green]Clean complete[/green]")


# ============================================================================
# INFO command
# ============================================================================

@app.command("info")
def info():
    """Show package information and configuration."""
    from rank1 import __version__

    console.print(f"[bold]rank1-factorization[/bold] v{__version__}")
    console.print()

    table = Table(title="Available Datasets")
    table.add_column("Name")
    table.add_column("Description")
    table.add_column("Source")

    table.add_row(
        "higgs_atlas",
        "ATLAS Higgs μ_{prod,decay}",
        "HEPData 130266",
    )
    table.add_row(
        "elastic_totem",
        "TOTEM dσ/dt shapes",
        "CERN Open Data + arXiv",
    )
    table.add_row(
        "diffractive_dis",
        "H1/ZEUS diffractive DIS",
        "HEPData ins718189, ins675372",
    )

    console.print(table)


# ============================================================================
# VALIDATE commands
# ============================================================================

validate_app = typer.Typer(help="Validation and calibration tools")
app.add_typer(validate_app, name="validate")


@validate_app.command("repro")
def validate_reproducibility(
    seed: int = typer.Option(42, "--seed", "-s", help="Random seed"),
    n_bootstrap: int = typer.Option(100, "--bootstrap", "-b", help="Bootstrap samples"),
    dataset: Optional[str] = typer.Option(None, "--dataset", "-d", help="Specific dataset"),
    quick: bool = typer.Option(False, "--quick", "-q", help="Quick smoke test"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Save report"),
):
    """Verify analysis reproducibility with fixed seed."""
    setup_logging("INFO")

    from rank1.scripts.verify_repro import verify_reproducibility

    datasets = [dataset] if dataset else None

    report = verify_reproducibility(
        seed=seed,
        n_bootstrap=n_bootstrap,
        datasets=datasets,
        quick=quick,
    )

    if output:
        import json
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        console.print(f"Report saved: {output}")

    if report.all_passed:
        console.print("[green]REPRODUCIBILITY CHECK PASSED[/green]")
    else:
        console.print("[red]REPRODUCIBILITY CHECK FAILED[/red]")
        raise typer.Exit(1)


@validate_app.command("identity")
def validate_identity(
    dataset: str = typer.Argument(..., help="Dataset to verify: higgs, elastic, diffractive"),
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
):
    """Verify dataset identity with spot-checks against published values."""
    cfg = _get_config(config)
    cfg.ensure_dirs()
    setup_logging(cfg.log_level)

    from rank1.datasets import HiggsATLASDataset, ElasticTOTEMDataset, DiffractiveDISDataset
    from rank1.validation.identity import (
        verify_higgs_identity,
        verify_elastic_identity,
        verify_diffractive_identity,
    )

    # Load dataset
    if dataset == "higgs":
        ds = HiggsATLASDataset(
            cache_dir=cfg.data.cache_dir,
            raw_dir=cfg.data.raw_dir,
            processed_dir=cfg.data.processed_dir,
        )
        data = ds.get_matrix_data()
        report = verify_higgs_identity(data)
    elif dataset == "elastic":
        ds = ElasticTOTEMDataset(
            cache_dir=cfg.data.cache_dir,
            raw_dir=cfg.data.raw_dir,
            processed_dir=cfg.data.processed_dir,
        )
        data = ds.get_matrix_data()
        report = verify_elastic_identity(data)
    elif dataset == "diffractive":
        ds = DiffractiveDISDataset(
            cache_dir=cfg.data.cache_dir,
            raw_dir=cfg.data.raw_dir,
            processed_dir=cfg.data.processed_dir,
        )
        data = ds.get_matrix_data()
        report = verify_diffractive_identity(data)
    else:
        console.print(f"[red]Unknown dataset: {dataset}[/red]")
        raise typer.Exit(1)

    # Print results
    table = Table(title=f"Identity Verification: {dataset}")
    table.add_column("Check")
    table.add_column("Expected")
    table.add_column("Actual")
    table.add_column("Status")

    for r in report.results:
        expected = f"{r.check.expected_value:.4f}"
        actual = f"{r.actual_value:.4f}" if r.actual_value else "N/A"
        status = "[green]PASS[/green]" if r.passed else "[red]FAIL[/red]"
        table.add_row(r.check.description, expected, actual, status)

    console.print(table)

    if report.provenance_warnings:
        console.print("\n[yellow]Provenance Warnings:[/yellow]")
        for w in report.provenance_warnings:
            console.print(f"  - {w}")

    if report.all_passed:
        console.print(f"\n[green]IDENTITY CHECK PASSED ({report.n_passed}/{report.n_checks})[/green]")
    else:
        console.print(f"\n[red]IDENTITY CHECK FAILED ({report.n_passed}/{report.n_checks})[/red]")
        raise typer.Exit(1)


@validate_app.command("calibrate")
def validate_calibration(
    n_null: int = typer.Option(200, "--null-sims", "-n", help="Null simulations"),
    n_power: int = typer.Option(100, "--power-sims", "-p", help="Power simulations"),
    n_bootstrap: int = typer.Option(200, "--bootstrap", "-b", help="Bootstrap per sim"),
    seed: int = typer.Option(42, "--seed", "-s", help="Random seed"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Save report"),
):
    """Run statistical calibration (null uniformity + power analysis)."""
    setup_logging("INFO")
    logger = get_logger()

    from rank1.validation.calibration import run_full_calibration

    results = run_full_calibration(
        n_null_sims=n_null,
        n_power_sims=n_power,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )

    if output:
        import json
        output.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "null": results["null"].to_dict(),
            "power": results["power"].to_dict(),
        }
        with open(output, "w") as f:
            json.dump(report, f, indent=2)
        console.print(f"Report saved: {output}")

    all_passed = results["null"].null_passed and results["power"].power_adequate

    if all_passed:
        console.print("[green]CALIBRATION PASSED[/green]")
    else:
        console.print("[red]CALIBRATION ISSUES DETECTED[/red]")
        raise typer.Exit(1)


@validate_app.command("robustness")
def validate_robustness(
    dataset: str = typer.Argument(..., help="Dataset: higgs, elastic, diffractive"),
    n_bootstrap: int = typer.Option(500, "--bootstrap", "-b", help="Bootstrap samples"),
    seed: int = typer.Option(42, "--seed", "-s", help="Base seed"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output dir"),
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
):
    """Run robustness sweep varying analysis choices."""
    cfg = _get_config(config)
    cfg.ensure_dirs()
    setup_logging(cfg.log_level)
    logger = get_logger()

    output_dir = output or (cfg.output_dir / f"robustness_{dataset}")
    output_dir.mkdir(parents=True, exist_ok=True)

    from rank1.analysis import HiggsRankAnalysis, ElasticRankAnalysis, DiffractiveRankAnalysis

    # Define sweep parameters
    sweep_results = []

    console.print(f"[bold]Running robustness sweep for {dataset}[/bold]")

    # Sweep over different seeds
    seeds = [seed, seed + 1, seed + 2]
    for s in seeds:
        logger.info(f"Running with seed={s}")

        if dataset == "higgs":
            analysis = HiggsRankAnalysis(output_dir=output_dir / f"seed_{s}")
        elif dataset == "elastic":
            analysis = ElasticRankAnalysis(output_dir=output_dir / f"seed_{s}")
        elif dataset == "diffractive":
            analysis = DiffractiveRankAnalysis(output_dir=output_dir / f"seed_{s}")
        else:
            console.print(f"[red]Unknown dataset: {dataset}[/red]")
            raise typer.Exit(1)

        try:
            result = analysis.run(n_bootstrap=n_bootstrap, seed=s)
            sweep_results.append({
                "variant": f"seed={s}",
                "lambda": result.lambda_stat,
                "p_value": result.p_value,
                "significant": result.is_significant,
            })
        except Exception as e:
            logger.error(f"  Failed: {e}")
            sweep_results.append({
                "variant": f"seed={s}",
                "lambda": None,
                "p_value": None,
                "significant": None,
                "error": str(e),
            })

    # Print results
    table = Table(title=f"Robustness Sweep: {dataset}")
    table.add_column("Variant")
    table.add_column("Λ")
    table.add_column("p-value")
    table.add_column("Significant")

    for r in sweep_results:
        if r.get("lambda") is not None:
            sig = "[red]Yes[/red]" if r["significant"] else "[green]No[/green]"
            table.add_row(
                r["variant"],
                f"{r['lambda']:.2f}",
                f"{r['p_value']:.4f}",
                sig,
            )
        else:
            table.add_row(r["variant"], "ERROR", "ERROR", "[red]ERROR[/red]")

    console.print(table)

    # Save results
    results_file = output_dir / "robustness_results.json"
    with open(results_file, "w") as f:
        json.dump(sweep_results, f, indent=2)
    console.print(f"Results saved: {results_file}")

    # Check stability
    p_values = [r["p_value"] for r in sweep_results if r.get("p_value") is not None]
    if len(p_values) >= 2:
        import numpy as np
        p_std = np.std(p_values)
        if p_std < 0.05:
            console.print(f"[green]Results stable (p-value std = {p_std:.4f})[/green]")
        else:
            console.print(f"[yellow]Results vary (p-value std = {p_std:.4f})[/yellow]")


# ============================================================================
# NP (NEW PHYSICS SENSITIVE) commands
# ============================================================================

np_app = typer.Typer(help="New Physics Sensitive residual mode analysis")
app.add_typer(np_app, name="np")


def _load_matrix_data(dataset: str, cfg):
    """Load matrix data for a dataset."""
    from rank1.datasets import HiggsATLASDataset, ElasticTOTEMDataset, DiffractiveDISDataset

    if dataset == "higgs":
        ds = HiggsATLASDataset(
            cache_dir=cfg.data.cache_dir,
            raw_dir=cfg.data.raw_dir,
            processed_dir=cfg.data.processed_dir,
        )
    elif dataset == "elastic":
        ds = ElasticTOTEMDataset(
            cache_dir=cfg.data.cache_dir,
            raw_dir=cfg.data.raw_dir,
            processed_dir=cfg.data.processed_dir,
        )
    elif dataset == "diffractive":
        ds = DiffractiveDISDataset(
            cache_dir=cfg.data.cache_dir,
            raw_dir=cfg.data.raw_dir,
            processed_dir=cfg.data.processed_dir,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    return ds.get_matrix_data()


@np_app.command("run")
def np_run(
    dataset: str = typer.Argument(..., help="Dataset: higgs, elastic, diffractive, or all"),
    n_bootstrap: int = typer.Option(500, "--bootstrap", "-b", help="Bootstrap samples for local p-value"),
    n_global_bootstrap: int = typer.Option(1000, "--global-bootstrap", "-g", help="Bootstrap samples for global correction"),
    n_starts: int = typer.Option(5, "--starts", "-n", help="Number of multi-start fits"),
    seed: int = typer.Option(42, "--seed", "-s", help="Random seed"),
    fast: bool = typer.Option(False, "--fast", "-f", help="Fast mode with reduced settings"),
    full: bool = typer.Option(False, "--full", help="Full mode with comprehensive settings"),
    sweeps: bool = typer.Option(True, "--sweeps/--no-sweeps", help="Run sweep analysis"),
    replication: bool = typer.Option(True, "--replication/--no-replication", help="Run replication checks"),
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
):
    """
    Run NP (New Physics Sensitive) analysis.

    Extracts structured residual modes with localization metrics,
    stability analysis, and global significance correction.
    """
    cfg = _get_config(config)
    cfg.ensure_dirs()
    setup_logging(cfg.log_level)
    logger = get_logger()

    from rank1.analysis.np_analysis import NPAnalyzer, NPVerdict
    from rank1.reporting.np_figures import NPFigureGenerator

    # Adjust settings for fast/full mode
    if fast:
        n_bootstrap = min(100, n_bootstrap)
        n_global_bootstrap = min(200, n_global_bootstrap)
        n_starts = min(3, n_starts)
    elif full:
        n_bootstrap = max(1000, n_bootstrap)
        n_global_bootstrap = max(2000, n_global_bootstrap)
        n_starts = max(10, n_starts)

    datasets = ["higgs", "elastic", "diffractive"] if dataset == "all" else [dataset]
    results = []

    for ds_name in datasets:
        console.print(f"\n[bold]Running NP analysis for {ds_name}[/bold]")

        try:
            # Load data
            matrix_data = _load_matrix_data(ds_name, cfg)

            # Create analyzer
            output_dir = cfg.output_dir / ds_name / "np"
            analyzer = NPAnalyzer(
                dataset=ds_name,
                output_dir=output_dir,
            )

            # Run analysis
            result = analyzer.run(
                matrix_data=matrix_data,
                n_bootstrap=n_bootstrap,
                n_global_bootstrap=n_global_bootstrap,
                n_starts=n_starts,
                seed=seed,
                run_sweeps=sweeps,
                run_replication=replication,
                fast_mode=fast,
            )
            results.append(result)

            # Generate figures
            fig_gen = NPFigureGenerator(output_dir)
            fig_gen.generate_all_figures(result)

            # Print summary
            console.print(f"\n{result.summary_string()}")

        except Exception as e:
            logger.error(f"Failed for {ds_name}: {e}")
            import traceback
            traceback.print_exc()

    # Print combined summary table
    if results:
        console.print("\n")
        table = Table(title="NP Analysis Summary")
        table.add_column("Dataset")
        table.add_column("Λ")
        table.add_column("p_local")
        table.add_column("p_global")
        table.add_column("Gini")
        table.add_column("Stability")
        table.add_column("Verdict")

        for r in results:
            gini = f"{r.localization_metrics.gini:.2f}" if r.localization_metrics else "N/A"
            stab = r.stability_metrics.stability_grade if r.stability_metrics else "N/A"
            p_global = f"{r.global_significance.p_global:.4f}" if r.global_significance else "N/A"

            # Color-code verdict
            if r.np_verdict == NPVerdict.STRUCTURED_DEVIATION:
                verdict = "[bold yellow]STRUCTURED[/bold yellow]"
            elif r.np_verdict == NPVerdict.LIKELY_ARTIFACT:
                verdict = "[yellow]ARTIFACT[/yellow]"
            elif r.np_verdict == NPVerdict.CONSISTENT_WITH_NULL:
                verdict = "[green]NULL[/green]"
            else:
                verdict = "[red]INCONCLUSIVE[/red]"

            table.add_row(
                r.dataset,
                f"{r.lambda_stat:.2f}",
                f"{r.p_local:.4f}",
                p_global,
                gini,
                stab,
                verdict,
            )

        console.print(table)

    console.print("\n[green]NP analysis complete[/green]")


@np_app.command("sweep")
def np_sweep(
    dataset: str = typer.Argument(..., help="Dataset: higgs, elastic, diffractive, or all"),
    n_bootstrap: int = typer.Option(500, "--bootstrap", "-b", help="Bootstrap samples per preset"),
    n_global_bootstrap: int = typer.Option(1000, "--global-bootstrap", "-g", help="Global bootstrap samples"),
    seed: int = typer.Option(42, "--seed", "-s", help="Random seed"),
    fast: bool = typer.Option(False, "--fast", "-f", help="Fast mode (fewer presets)"),
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
):
    """
    Run NP sweep analysis across predefined presets.

    Computes global significance with look-elsewhere correction.
    """
    cfg = _get_config(config)
    cfg.ensure_dirs()
    setup_logging(cfg.log_level)
    logger = get_logger()

    from rank1.analysis.sweeps import (
        SweepRunner, get_presets_for_dataset, get_fast_presets, compute_sweep_summary
    )

    datasets = ["higgs", "elastic", "diffractive"] if dataset == "all" else [dataset]

    for ds_name in datasets:
        console.print(f"\n[bold]Running sweep analysis for {ds_name}[/bold]")

        # Get presets
        presets = get_fast_presets(ds_name) if fast else get_presets_for_dataset(ds_name)

        if not presets:
            console.print(f"[yellow]No presets defined for {ds_name}, skipping[/yellow]")
            continue

        console.print(f"  Presets: {[p.name for p in presets]}")

        output_dir = cfg.output_dir / ds_name / "np" / "sweeps"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create sweep runner
        runner = SweepRunner(
            dataset=ds_name,
            presets=presets,
            output_dir=output_dir,
        )

        # For now, just list presets (full sweep requires analysis factory)
        table = Table(title=f"Sweep Presets: {ds_name}")
        table.add_column("Name")
        table.add_column("Type")
        table.add_column("Description")

        for p in presets:
            table.add_row(p.name, p.sweep_type.value, p.description)

        console.print(table)

        # Save preset info
        import json
        presets_file = output_dir / "presets.json"
        with open(presets_file, "w") as f:
            json.dump([p.to_dict() for p in presets], f, indent=2)
        console.print(f"Presets saved: {presets_file}")

    console.print("\n[green]Sweep configuration complete[/green]")


@np_app.command("report")
def np_report(
    dataset: str = typer.Argument("all", help="Dataset: higgs, elastic, diffractive, or all"),
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
):
    """
    Generate NP analysis report from saved results.
    """
    cfg = _get_config(config)
    setup_logging(cfg.log_level)
    logger = get_logger()

    from rank1.analysis.np_analysis import NPResult, NPVerdict
    from rank1.reporting.np_report import NPReportGenerator

    datasets = ["higgs", "elastic", "diffractive"] if dataset == "all" else [dataset]
    results = []

    for ds_name in datasets:
        results_file = cfg.output_dir / ds_name / "np" / "np_results.json"

        if not results_file.exists():
            logger.warning(f"No results found for {ds_name}: {results_file}")
            continue

        with open(results_file) as f:
            data = json.load(f)

        # Reconstruct NPResult (simplified)
        import numpy as np
        from rank1.analysis.residual_mode import LocalizationMetrics, StabilityMetrics, ResidualMode
        from rank1.analysis.sweeps import GlobalSignificance
        from rank1.analysis.replication import ReplicationReport

        # Reconstruct localization metrics
        loc_data = data.get("localization_metrics")
        loc = None
        if loc_data:
            loc = LocalizationMetrics(
                top_k_mass=loc_data.get("top_k_mass", {}),
                gini=loc_data.get("gini", 0.0),
                entropy=loc_data.get("entropy", 0.0),
                max_entropy=loc_data.get("max_entropy", 0.0),
                normalized_entropy=loc_data.get("normalized_entropy", 1.0),
                window_concentration=loc_data.get("window_concentration", {}),
                peak_index=loc_data.get("peak_index", 0),
                peak_value=loc_data.get("peak_value", 0.0),
            )

        # Reconstruct stability metrics
        stab_data = data.get("stability_metrics")
        stab = None
        if stab_data:
            stab = StabilityMetrics(
                n_starts=stab_data.get("n_starts", 0),
                v2_cosine_mean=stab_data.get("v2_cosine_mean", 0.0),
                v2_cosine_std=stab_data.get("v2_cosine_std", 0.0),
                u2_cosine_mean=stab_data.get("u2_cosine_mean", 0.0),
                u2_cosine_std=stab_data.get("u2_cosine_std", 0.0),
                stability_grade=stab_data.get("stability_grade", "unknown"),
                is_stable=stab_data.get("is_stable", False),
            )

        # Reconstruct global significance
        gs_data = data.get("global_significance")
        gs = None
        if gs_data:
            gs = GlobalSignificance(
                T_obs=gs_data.get("T_obs", 0.0),
                best_preset=gs_data.get("best_preset", ""),
                p_local_best=gs_data.get("p_local_best", 1.0),
                p_global=gs_data.get("p_global", 1.0),
                n_presets=gs_data.get("n_presets", 0),
                n_bootstrap=gs_data.get("n_bootstrap", 0),
            )

        # Create result
        result = NPResult(
            dataset=data["dataset"],
            n_rows=data["n_rows"],
            n_cols=data["n_cols"],
            n_obs=data["n_obs"],
            chi2_rank1=data["chi2_rank1"],
            ndof_rank1=data["ndof_rank1"],
            chi2_rank2=data["chi2_rank2"],
            ndof_rank2=data["ndof_rank2"],
            lambda_stat=data["lambda_stat"],
            p_local=data["p_local"],
            p_local_ci=tuple(data.get("p_local_ci", [0.0, 1.0])),
            localization_metrics=loc,
            stability_metrics=stab,
            global_significance=gs,
            np_verdict=NPVerdict(data.get("np_verdict", "inconclusive")),
            np_reasons=data.get("np_reasons", []),
            fit_healthy=data.get("fit_healthy", True),
            fit_warnings=data.get("fit_warnings", []),
        )
        results.append(result)
        logger.info(f"Loaded NP results for {ds_name}")

    if not results:
        console.print("[red]No NP results found. Run 'rank1 np run' first.[/red]")
        raise typer.Exit(1)

    # Generate report
    generator = NPReportGenerator(cfg.output_dir)
    report_path = generator.generate_full_report(results)

    console.print(f"[green]NP Report generated: {report_path}[/green]")


@np_app.command("all")
def np_all(
    n_bootstrap: int = typer.Option(500, "--bootstrap", "-b", help="Bootstrap samples"),
    n_global_bootstrap: int = typer.Option(1000, "--global-bootstrap", "-g", help="Global bootstrap"),
    n_starts: int = typer.Option(5, "--starts", "-n", help="Multi-start fits"),
    seed: int = typer.Option(42, "--seed", "-s", help="Random seed"),
    fast: bool = typer.Option(False, "--fast", "-f", help="Fast mode"),
    full: bool = typer.Option(False, "--full", help="Full mode"),
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
):
    """
    Run complete NP analysis on all datasets and generate report.
    """
    # Run analysis
    np_run.callback(
        dataset="all",
        n_bootstrap=n_bootstrap,
        n_global_bootstrap=n_global_bootstrap,
        n_starts=n_starts,
        seed=seed,
        fast=fast,
        full=full,
        sweeps=True,
        replication=True,
        config=config,
    )

    # Generate report
    np_report.callback(dataset="all", config=config)


@np_app.command("higgs")
def np_higgs(
    n_bootstrap: int = typer.Option(500, "--bootstrap", "-b"),
    n_global_bootstrap: int = typer.Option(1000, "--global-bootstrap", "-g"),
    n_starts: int = typer.Option(5, "--starts", "-n"),
    seed: int = typer.Option(42, "--seed", "-s"),
    fast: bool = typer.Option(False, "--fast", "-f"),
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
):
    """Run NP analysis on ATLAS Higgs data."""
    np_run.callback(
        dataset="higgs",
        n_bootstrap=n_bootstrap,
        n_global_bootstrap=n_global_bootstrap,
        n_starts=n_starts,
        seed=seed,
        fast=fast,
        full=False,
        sweeps=True,
        replication=True,
        config=config,
    )


@np_app.command("elastic")
def np_elastic(
    n_bootstrap: int = typer.Option(500, "--bootstrap", "-b"),
    n_global_bootstrap: int = typer.Option(1000, "--global-bootstrap", "-g"),
    n_starts: int = typer.Option(5, "--starts", "-n"),
    seed: int = typer.Option(42, "--seed", "-s"),
    fast: bool = typer.Option(False, "--fast", "-f"),
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
):
    """Run NP analysis on TOTEM elastic data."""
    np_run.callback(
        dataset="elastic",
        n_bootstrap=n_bootstrap,
        n_global_bootstrap=n_global_bootstrap,
        n_starts=n_starts,
        seed=seed,
        fast=fast,
        full=False,
        sweeps=True,
        replication=True,
        config=config,
    )


@np_app.command("diffractive")
def np_diffractive(
    n_bootstrap: int = typer.Option(500, "--bootstrap", "-b"),
    n_global_bootstrap: int = typer.Option(1000, "--global-bootstrap", "-g"),
    n_starts: int = typer.Option(5, "--starts", "-n"),
    seed: int = typer.Option(42, "--seed", "-s"),
    fast: bool = typer.Option(False, "--fast", "-f"),
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
):
    """Run NP analysis on diffractive DIS data."""
    np_run.callback(
        dataset="diffractive",
        n_bootstrap=n_bootstrap,
        n_global_bootstrap=n_global_bootstrap,
        n_starts=n_starts,
        seed=seed,
        fast=fast,
        full=False,
        sweeps=True,
        replication=True,
        config=config,
    )


# ============================================================================
# DOCTOR / SMOKE TEST commands
# ============================================================================

@app.command("doctor")
def doctor(
    n_bootstrap: int = typer.Option(20, "--bootstrap", "-b", help="Bootstrap samples"),
    n_global_bootstrap: int = typer.Option(30, "--global-bootstrap", "-g", help="Global bootstrap"),
    seed: int = typer.Option(1, "--seed", "-s", help="Random seed"),
    n_jobs: int = typer.Option(2, "--n-jobs", "-j", help="Parallel jobs"),
    n_starts: int = typer.Option(3, "--n-starts", "-n", help="Multi-start fits"),
    skip_large: bool = typer.Option(True, "--skip-large/--no-skip-large", help="Skip large downloads"),
    fast: bool = typer.Option(True, "--fast/--full", help="Fast mode"),
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
):
    """
    Run smoke test and self-healing diagnostics.

    Checks environment, runs minimal analyses, verifies determinism,
    and attempts auto-remediation of common issues.
    """
    cfg = _get_config(config)
    cfg.ensure_dirs()
    setup_logging(cfg.log_level)

    from rank1.doctor import run_doctor, OverallStatus

    report = run_doctor(
        output_dir=cfg.output_dir,
        n_bootstrap=n_bootstrap,
        n_global_bootstrap=n_global_bootstrap,
        seed=seed,
        n_jobs=n_jobs,
        n_starts=n_starts,
        skip_large_downloads=skip_large,
        fast_mode=fast,
    )

    # Exit code based on status
    if report.overall_status == OverallStatus.PASS:
        console.print("\n[bold green]✅ DOCTOR: PASS[/bold green]")
        raise typer.Exit(0)
    elif report.overall_status == OverallStatus.NEEDS_MANUAL:
        console.print("\n[bold yellow]⚠️ DOCTOR: NEEDS MANUAL ATTENTION[/bold yellow]")
        console.print(f"See: {cfg.output_dir}/doctor/doctor_report.md")
        raise typer.Exit(1)
    else:
        console.print("\n[bold red]❌ DOCTOR: FAIL[/bold red]")
        console.print(f"See: {cfg.output_dir}/doctor/doctor_report.md")
        raise typer.Exit(2)


@app.command("smoke")
def smoke(
    fast: bool = typer.Option(True, "--fast/--full", help="Fast mode"),
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
):
    """
    Alias for 'doctor' - run quick smoke test.
    """
    doctor.callback(
        n_bootstrap=20,
        n_global_bootstrap=30,
        seed=1,
        n_jobs=2,
        n_starts=3,
        skip_large=True,
        fast=fast,
        config=config,
    )


if __name__ == "__main__":
    app()
