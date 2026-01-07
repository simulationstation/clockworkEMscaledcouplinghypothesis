#!/usr/bin/env python3
"""
Full reproducibility script for rank-1 factorization analysis.

This script runs the complete pipeline:
1. Fetch all data
2. Run all analyses
3. Generate report

Usage:
    python -m rank1.scripts.reproduce_all
    python -m rank1.scripts.reproduce_all --bootstrap 2000
"""

import argparse
import sys
from pathlib import Path

from rank1.config import Config, set_config
from rank1.logging import setup_logging, get_logger
from rank1.datasets import HiggsATLASDataset, ElasticTOTEMDataset, DiffractiveDISDataset
from rank1.analysis import HiggsRankAnalysis, ElasticRankAnalysis, DiffractiveRankAnalysis
from rank1.reporting import ReportGenerator


def main():
    parser = argparse.ArgumentParser(description="Reproduce all rank-1 analyses")
    parser.add_argument(
        "--bootstrap", "-b",
        type=int,
        default=1000,
        help="Number of bootstrap iterations (default: 1000)"
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("outputs"),
        help="Output directory (default: outputs)"
    )
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Skip data fetching (use cached data)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    args = parser.parse_args()

    # Setup
    config = Config(
        output_dir=args.output_dir,
        seed=args.seed,
        log_level="DEBUG" if args.verbose else "INFO",
    )
    config.ensure_dirs()
    set_config(config)

    setup_logging(config.log_level)
    logger = get_logger()

    logger.info("=" * 60)
    logger.info("Rank-1 Factorization Analysis - Full Reproduction")
    logger.info("=" * 60)
    logger.info(f"Bootstrap iterations: {args.bootstrap}")
    logger.info(f"Random seed: {args.seed}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("")

    # Initialize datasets
    datasets = {
        "higgs": HiggsATLASDataset(
            cache_dir=config.data.cache_dir,
            raw_dir=config.data.raw_dir,
            processed_dir=config.data.processed_dir,
        ),
        "elastic": ElasticTOTEMDataset(
            cache_dir=config.data.cache_dir,
            raw_dir=config.data.raw_dir,
            processed_dir=config.data.processed_dir,
        ),
        "diffractive": DiffractiveDISDataset(
            cache_dir=config.data.cache_dir,
            raw_dir=config.data.raw_dir,
            processed_dir=config.data.processed_dir,
        ),
    }

    # Fetch data
    if not args.skip_fetch:
        logger.info("Fetching data...")
        for name, dataset in datasets.items():
            logger.info(f"  Fetching {name}...")
            try:
                dataset.fetch_raw(force=False)
            except Exception as e:
                logger.warning(f"  Failed to fetch {name}: {e}")
        logger.info("")

    # Run analyses
    logger.info("Running analyses...")
    results = []

    analyses = [
        ("Higgs ATLAS", HiggsRankAnalysis(
            dataset=datasets["higgs"],
            output_dir=args.output_dir / "higgs_atlas",
        )),
        ("Elastic TOTEM", ElasticRankAnalysis(
            dataset=datasets["elastic"],
            output_dir=args.output_dir / "elastic_totem",
        )),
        ("Diffractive DIS", DiffractiveRankAnalysis(
            dataset=datasets["diffractive"],
            output_dir=args.output_dir / "diffractive_dis",
        )),
    ]

    for name, analysis in analyses:
        logger.info(f"  Running {name}...")
        try:
            result = analysis.run(
                n_bootstrap=args.bootstrap,
                seed=args.seed,
            )
            results.append(result)
            logger.info(f"    p-value = {result.p_value:.4f}")
        except Exception as e:
            logger.error(f"  Failed: {e}")
            import traceback
            traceback.print_exc()

    logger.info("")

    # Generate report
    if results:
        logger.info("Generating report...")
        generator = ReportGenerator(args.output_dir)
        report_path = generator.generate_full_report(results)
        logger.info(f"Report saved to: {report_path}")

    # Print summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    for result in results:
        sig = "REJECTED" if result.is_significant else "not rejected"
        logger.info(
            f"  {result.dataset_name}: "
            f"Λ = {result.lambda_stat:.2f}, "
            f"p = {result.p_value:.4f} ({sig})"
        )

    logger.info("")
    logger.info(f"Full outputs saved to: {args.output_dir}")

    return 0 if results else 1


if __name__ == "__main__":
    sys.exit(main())
