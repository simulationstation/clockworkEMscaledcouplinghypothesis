# Rank-1 Factorization Analysis for Particle Physics

[![CI](https://github.com/research/rank1-factorization/workflows/CI/badge.svg)](https://github.com/research/rank1-factorization/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A complete, reproducible pipeline for testing rank-1 factorization hypotheses on particle physics datasets.

## Overview

This package tests whether matrix-structured physics data is consistent with rank-1 factorization:

**M_ij ≈ u_i × v_j**

Three physics analyses are implemented:

1. **ATLAS Higgs μ_{prod,decay}**: Tests if Higgs signal strengths factorize as μ_prod × μ_decay
2. **TOTEM Elastic dσ/dt**: Tests if elastic scattering shapes are universal across √s
3. **H1/ZEUS Diffractive DIS**: Tests Regge factorization of diffractive structure functions

## Installation

### Quick Start

```bash
# Clone repository
git clone https://github.com/research/rank1-factorization.git
cd rank1-factorization

# Create virtual environment and install
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# Or use make
make install
```

### With All Dependencies (including PDF extraction)

```bash
pip install -e ".[all]"
```

### Development Installation

```bash
pip install -e ".[dev]"
```

## Usage

### Fetch All Data

```bash
rank1 fetch all
```

This downloads:
- ATLAS Higgs data from HEPData (record 130266)
- TOTEM 7 TeV data from CERN Open Data (~420 MB)
- TOTEM 8/13 TeV tables from arXiv PDFs (or uses curated manual tables)
- H1/ZEUS diffractive DIS data from HEPData

### Run All Analyses

```bash
rank1 run all
```

For more control over bootstrap iterations:

```bash
rank1 run all --bootstrap 1000 --seed 42
```

### Generate Combined Report

```bash
rank1 report all
```

Outputs are saved to `outputs/`:
- `outputs/REPORT.md` - Combined report
- `outputs/<dataset>/results.json` - Numerical results
- `outputs/<dataset>/figures/` - Diagnostic plots
- `outputs/<dataset>/summary.md` - Per-dataset summary

### Individual Analyses

```bash
# Fetch and run specific datasets
rank1 fetch higgs
rank1 run higgs --bootstrap 500

rank1 fetch elastic
rank1 run elastic

rank1 fetch diffractive
rank1 run diffractive --experiment combined
```

### Using Make

```bash
make fetch      # Download all data
make run        # Run all analyses
make report     # Generate report
make reproduce  # Full pipeline: install, fetch, run, report
make clean      # Remove outputs (keep raw data)
make clean-all  # Remove everything including data
```

## NP (New Physics Sensitive) Mode

The NP mode goes beyond simple pass/fail testing to:
- Extract structured residual modes (u2, v2) when rank-2 fits better
- Quantify where residuals localize (which bins/channels drive the deviation)
- Test stability across multi-start fits and sweep presets
- Compute global significance with look-elsewhere correction
- Check replication across independent data slices

### Quick Start

```bash
# Fast NP analysis (reduced settings for testing)
rank1 np run all --fast

# Full NP analysis (comprehensive)
rank1 np run all --full

# Generate NP report
rank1 np report all
```

### Using Make (Recommended)

```bash
make np-fast       # Fast NP analysis on all datasets
make np-full       # Full NP analysis
make np-report     # Generate NP_REPORT.md
make np-all        # Complete NP pipeline
```

### NP Mode Commands

```bash
# Run NP analysis on specific dataset
rank1 np higgs --bootstrap 500 --starts 5
rank1 np elastic
rank1 np diffractive

# Run with custom settings
rank1 np run all \
    --bootstrap 1000 \
    --global-bootstrap 2000 \
    --starts 10 \
    --seed 42

# Sweep analysis across presets
rank1 np sweep all --fast

# Generate report from saved results
rank1 np report all
```

### NP Mode Outputs

Outputs are saved to `outputs/<dataset>/np/`:
- `np_results.json` - Complete numerical results
- `np_summary.txt` - Human-readable summary
- `figures/` - Diagnostic plots:
  - Residual heatmap under rank-1
  - v2 shape (column/observable dependence)
  - u2 dependence (row/condition dependence)
  - Localization metrics panel
  - Sweep summary (if multiple presets)
  - Replication similarity matrix

Top-level report: `outputs/NP_REPORT.md`

### Interpreting NP Results

The NP mode produces tiered verdicts:

| Verdict | Meaning |
|---------|---------|
| **STRUCTURED_DEVIATION** | Localized, stable, replicating residual with global significance. Candidate for investigation. |
| **LIKELY_ARTIFACT** | Deviation detected but unstable, diffuse, or fails replication. Likely fluctuation or systematic. |
| **CONSISTENT_WITH_NULL** | No significant deviation from rank-1. Data compatible with factorizable model. |
| **INCONCLUSIVE** | Fit issues or unstable results. Check data quality and diagnostics. |

### Key NP Metrics

- **Λ (Lambda)**: Test statistic χ²(rank-1) - χ²(rank-2)
- **p_local**: Bootstrap p-value at single preset
- **p_global**: Look-elsewhere corrected p-value across all presets
- **Gini**: Concentration of residual mode (0=uniform, 1=concentrated)
- **Replication score**: Agreement across independent conditions (0-1)

## Smoke Test / Doctor

The `doctor` command runs comprehensive environment checks, minimal analyses,
determinism verification, and auto-remediation for common issues.

### Quick Smoke Test

```bash
# Fast smoke test (recommended for quick verification)
rank1 doctor

# Or using make
make doctor
make smoke  # alias
```

### What Doctor Checks

1. **Environment**: Python version, dependencies, CLI accessibility
2. **Code Quality**: Compile check, fast unit tests
3. **Data Fetch**: Downloads minimal datasets (skips large TOTEM 7 TeV by default)
4. **Baseline Analysis**: Runs Higgs + Diffractive with 20 bootstrap samples
5. **NP Analysis**: Runs NP mode with minimal settings
6. **Determinism**: Verifies same seed produces identical results
7. **Artifacts**: Checks expected output files exist

### Auto-Remediation

Doctor attempts to fix common issues automatically:
- Retries failed data fetches
- Reduces parallelism for BLAS issues
- Runs ruff auto-fix for formatting issues
- Sets single-threaded BLAS for determinism

### Doctor Options

```bash
# Full doctor with large downloads
rank1 doctor --no-skip-large --full

# Custom settings
rank1 doctor --bootstrap 50 --n-jobs 4 --seed 42
```

### Exit Codes

- `0`: PASS - all checks passed
- `1`: NEEDS_MANUAL_ATTENTION - some issues require manual intervention
- `2`: FAIL - critical failures detected

### Reports

Doctor generates reports in `outputs/doctor/`:
- `doctor_report.json` - machine-readable results
- `doctor_report.md` - human-readable markdown report

## Configuration

Default configuration can be overridden with a config file:

```bash
rank1 run all --config my_config.yaml
```

Example config:
```yaml
data:
  base_dir: data
  cache_dir: data/cache
  raw_dir: data/raw
  processed_dir: data/processed

output_dir: outputs
seed: 42
log_level: INFO
```

### Changing Bootstrap Iterations

Via CLI:
```bash
rank1 run all --bootstrap 2000
```

Via config:
```yaml
bootstrap:
  n_bootstrap: 2000
  seed: 42
  use_parallel: true
```

## Methodology

### Rank-1 Factorization Model

For a matrix M with observations at positions (i, j), the rank-1 model assumes:

```
M_ij = u_i × v_j
```

where u and v are vectors encoding the row and column factors.

### Test Statistic

We use the likelihood ratio test statistic:

```
Λ = χ²(rank-1) - χ²(rank-2)
```

Larger Λ indicates that rank-2 provides a significantly better fit.

### Parametric Bootstrap

P-values are computed via parametric bootstrap:

1. Fit rank-1 model to observed data
2. Generate B pseudo-datasets by adding Gaussian noise (from uncertainties)
3. Refit both rank-1 and rank-2 to each pseudo-dataset
4. Compute Λ for each pseudo-dataset
5. p-value = (k + 1) / (B + 1), where k = count(Λ_boot ≥ Λ_obs)

### Fit Health Checks

Each fit includes diagnostics:
- Convergence verification
- χ²/ndof reasonableness (not too good or too bad)
- Condition number check for numerical stability
- Multi-start stability analysis

### Cross-Checks

Each dataset includes automated validation:
- Data integrity verification (checksums, record IDs)
- Physical reasonableness of values
- Synthetic injection tests (verify rank-1 data is not rejected)
- Comparison with published values where available

## Data Sources

| Dataset | Source | Reference |
|---------|--------|-----------|
| ATLAS Higgs | [HEPData 130266](https://www.hepdata.net/record/130266) | DOI: 10.17182/hepdata.130266 |
| TOTEM 7 TeV | [CERN Open Data 84000](http://opendata.cern.ch/record/84000) | |
| TOTEM 8 TeV | [arXiv:1503.08111](https://arxiv.org/abs/1503.08111) | |
| TOTEM 13 TeV | [arXiv:1812.08283](https://arxiv.org/abs/1812.08283) | |
| H1 Diffractive | [HEPData ins718189](https://www.hepdata.net/record/ins718189) | DOI: 10.17182/hepdata.45891 |
| ZEUS Diffractive | [HEPData ins675372](https://www.hepdata.net/record/ins675372) | DOI: 10.17182/hepdata.11816 |

## Disk Usage

- TOTEM 7 TeV ROOT files: ~420 MB
- HEPData records: ~10 MB
- arXiv PDFs: ~5 MB each
- Processed data: ~5 MB
- Outputs (figures + JSON): ~50 MB

## Development

### Running Tests

```bash
# Fast tests (no network)
make test-fast

# All tests including slow ones
make test

# Network tests (requires internet)
make test-network
```

### Code Quality

```bash
make lint       # Run ruff linter
make format     # Format with black
make typecheck  # Run mypy
make ci         # Run all checks
```

## Troubleshooting

### PDF Extraction Fails

If camelot/pdfplumber fails to extract tables from arXiv PDFs:
1. The package will automatically fall back to curated manual tables
2. Manual tables are in `src/rank1/datasets/manual_tables/`
3. They have been verified against published values

To install PDF extraction dependencies:
```bash
pip install "rank1-factorization[pdf]"
# May require system packages:
# apt-get install ghostscript  # Ubuntu/Debian
# brew install ghostscript     # macOS
```

### Large Downloads

TOTEM 7 TeV data is ~420 MB. To skip it:
```bash
rank1 fetch higgs
rank1 fetch diffractive
# Only fetches smaller datasets
```

### SSL Errors

If you encounter SSL certificate errors:
```bash
pip install --upgrade certifi
```

Or set environment variable:
```bash
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
```

### Memory Issues

Bootstrap is parallelized by default. To reduce memory usage:
```bash
rank1 run all --config config_low_memory.yaml
```

With `config_low_memory.yaml`:
```yaml
parallel:
  n_jobs: 2  # Limit parallel workers
```

## Project Structure

```
.
├── pyproject.toml          # Package configuration
├── README.md               # This file
├── Makefile                # Task runner
├── src/rank1/
│   ├── cli.py              # Command-line interface
│   ├── config.py           # Configuration management
│   ├── utils/              # Utilities (HTTP, caching, stats)
│   ├── data_sources/       # Data acquisition clients
│   ├── datasets/           # Dataset implementations
│   ├── models/             # Low-rank models and fitting
│   ├── analysis/           # Analysis pipelines
│   └── reporting/          # Figure and report generation
├── tests/                  # Test suite
└── data/                   # Downloaded data (gitignored)
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

If you use this code, please cite the relevant data sources and:

```bibtex
@software{rank1_factorization,
  author = {Research Software Engineering},
  title = {Rank-1 Factorization Analysis for Particle Physics},
  year = {2024},
  url = {https://github.com/research/rank1-factorization}
}
```

## Future Enhancements

Potential extensions:
- Add CMS Higgs signal strength data
- Add ATLAS/ALFA elastic scattering data
- Support full covariance matrices where available
- Add asymptotic chi-squared distribution comparison
- Web-based interactive report viewer
