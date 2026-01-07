"""
ATLAS Higgs production × decay signal strength matrix dataset.

Source: HEPData record 130266
DOI: 10.17182/hepdata.130266

This dataset provides measurements of Higgs boson production and decay
signal strengths μ_{prod,decay} from ATLAS Run 2 data.

The rank-1 hypothesis corresponds to SM-like Higgs:
μ_{prod,decay} = μ_prod × μ_decay

Deviations from rank-1 would indicate BSM effects modifying
specific production-decay combinations.
"""

from pathlib import Path
from typing import Optional, Any
import json

import numpy as np
import pandas as pd

from rank1.datasets.base import (
    MatrixDataset,
    MatrixData,
    MatrixObservation,
    CrossCheckResult,
)
from rank1.data_sources.hepdata import HEPDataClient
from rank1.logging import get_logger
from rank1.provenance import DataProvenance, DataOrigin, create_provenance

logger = get_logger()


class HiggsATLASDataset(MatrixDataset):
    """
    ATLAS Higgs μ_{prod,decay} matrix dataset.

    Production modes (rows): ggF, VBF, WH, ZH, ttH, tH
    Decay channels (columns): γγ, ZZ*, WW*, ττ, bb, μμ
    """

    name = "higgs_atlas_mu"
    description = "ATLAS Higgs boson production × decay signal strengths (Run 2)"
    source_dois = ["10.17182/hepdata.130266"]
    source_urls = ["https://www.hepdata.net/record/130266"]

    HEPDATA_RECORD_ID = 130266

    # Expected production modes and decay channels
    PRODUCTION_MODES = ["ggF", "VBF", "WH", "ZH", "ttH", "tH"]
    DECAY_CHANNELS = ["gamgam", "ZZ", "WW", "tautau", "bb", "mumu"]

    # Display names for labels
    PROD_DISPLAY = {
        "ggF": "ggF",
        "VBF": "VBF",
        "WH": "WH",
        "ZH": "ZH",
        "ttH": "ttH",
        "tH": "tH",
    }
    DECAY_DISPLAY = {
        "gamgam": "γγ",
        "ZZ": "ZZ*",
        "WW": "WW*",
        "tautau": "ττ",
        "bb": "bb̄",
        "mumu": "μμ",
    }

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        raw_dir: Optional[Path] = None,
        processed_dir: Optional[Path] = None,
    ):
        super().__init__(cache_dir, raw_dir, processed_dir)
        self.hepdata = HEPDataClient(
            cache_dir=self.cache_dir,
            raw_dir=self.raw_dir / "hepdata",
        )
        self._record = None
        self._provenance: Optional[DataProvenance] = None

    def fetch_raw(self, force: bool = False) -> list[Path]:
        """Download HEPData record."""
        logger.info(f"Fetching HEPData record {self.HEPDATA_RECORD_ID}")

        # Initialize provenance
        self._provenance = create_provenance(
            DataOrigin.API,
            f"HEPData record {self.HEPDATA_RECORD_ID}"
        )
        self._provenance.set_fetch_timestamp()
        self._provenance.add_source(
            source_type="hepdata",
            identifier=str(self.HEPDATA_RECORD_ID),
            doi=self.source_dois[0],
            url=self.source_urls[0],
        )

        self._record = self.hepdata.download_record(
            self.HEPDATA_RECORD_ID, force=force
        )

        # Save record info and add file hash
        record_path = self.raw_dir / "hepdata" / f"record_{self.HEPDATA_RECORD_ID}.json"
        if record_path.exists():
            self._provenance.add_file_hash(record_path)

        return [record_path]

    def build_observations(self) -> MatrixData:
        """
        Build matrix observations from HEPData tables.

        Searches for tables containing signal strength measurements
        and organizes them into a production × decay matrix.
        """
        if self._record is None:
            self.fetch_raw()
            self._record = self.hepdata.download_record(self.HEPDATA_RECORD_ID)

        # Ensure provenance exists
        if self._provenance is None:
            self._provenance = create_provenance(
                DataOrigin.API,
                f"HEPData record {self.HEPDATA_RECORD_ID}"
            )
            self._provenance.add_source(
                source_type="hepdata",
                identifier=str(self.HEPDATA_RECORD_ID),
                doi=self.source_dois[0],
                url=self.source_urls[0],
            )

        self._provenance.set_build_timestamp()

        logger.info(f"Processing {len(self._record.tables)} tables from HEPData")

        # Find tables with signal strength data
        mu_tables = self._find_signal_strength_tables()

        if not mu_tables:
            logger.warning(
                "FALLBACK: No signal strength tables found in HEPData record %d. "
                "Using hardcoded fallback values. Results may not reflect actual data.",
                self.HEPDATA_RECORD_ID
            )
            self._provenance.record_fallback(
                step="build_observations",
                reason="No signal strength tables found in HEPData record",
                fallback_to="hardcoded_values",
                original_source=f"hepdata:{self.HEPDATA_RECORD_ID}",
            )
            self._provenance.origin = DataOrigin.FALLBACK
            return self._fallback_extraction()

        # Build matrix from tables
        observations = []
        prod_labels = []
        decay_labels = []

        # Track which cells we've seen
        cell_values = {}

        # Track table names used for provenance
        tables_used = []

        for table in mu_tables:
            logger.debug(f"Processing table: {table.name}")
            df = table.to_dataframe()

            # Try to extract production and decay mode from table name/keywords
            prod_mode = self._extract_production_mode(table)
            decay_mode = self._extract_decay_mode(table)

            if prod_mode and decay_mode:
                # Single measurement table
                value, error = self._extract_mu_from_table(df)
                if value is not None:
                    key = (prod_mode, decay_mode)
                    cell_values[key] = (value, error)
                    tables_used.append(table.name)
            else:
                # Matrix-style table - try to parse rows
                n_before = len(cell_values)
                self._parse_matrix_table(df, cell_values)
                if len(cell_values) > n_before:
                    tables_used.append(table.name)

        # Record which tables were used
        if self._provenance.sources:
            self._provenance.sources[0].table_names = tables_used

        # Build final matrix structure
        seen_prods = sorted(set(k[0] for k in cell_values.keys()))
        seen_decays = sorted(set(k[1] for k in cell_values.keys()))

        # Use canonical ordering where possible
        prod_labels = [p for p in self.PRODUCTION_MODES if p in seen_prods]
        prod_labels.extend([p for p in seen_prods if p not in prod_labels])

        decay_labels = [d for d in self.DECAY_CHANNELS if d in seen_decays]
        decay_labels.extend([d for d in seen_decays if d not in decay_labels])

        # Create observations
        prod_idx = {p: i for i, p in enumerate(prod_labels)}
        decay_idx = {d: i for i, d in enumerate(decay_labels)}

        for (prod, decay), (value, error) in cell_values.items():
            obs = MatrixObservation(
                row_idx=prod_idx[prod],
                col_idx=decay_idx[decay],
                row_label=self.PROD_DISPLAY.get(prod, prod),
                col_label=self.DECAY_DISPLAY.get(decay, decay),
                value=value,
                total_err=error,
            )
            observations.append(obs)

        # Try to get correlation matrix
        correlation = self._get_correlation_matrix(prod_labels, decay_labels)

        logger.info(
            f"Built Higgs matrix: {len(prod_labels)} production modes x "
            f"{len(decay_labels)} decay channels, {len(observations)} observations"
        )

        return MatrixData(
            name=self.name,
            description=self.description,
            row_labels=[self.PROD_DISPLAY.get(p, p) for p in prod_labels],
            col_labels=[self.DECAY_DISPLAY.get(d, d) for d in decay_labels],
            observations=observations,
            covariance=correlation,
            metadata={
                "source_doi": self.source_dois[0],
                "production_modes": prod_labels,
                "decay_channels": decay_labels,
            },
            provenance=self._provenance,
        )

    def _find_signal_strength_tables(self) -> list:
        """Find tables containing signal strength measurements."""
        mu_tables = []

        for table in self._record.tables:
            name_lower = table.name.lower()
            desc_lower = table.description.lower()

            # Look for signal strength indicators
            is_mu_table = (
                "signal strength" in desc_lower
                or "mu_" in name_lower
                or "μ" in table.name
                or ("prod" in desc_lower and "decay" in desc_lower)
                or "coupling" in desc_lower
            )

            if is_mu_table:
                mu_tables.append(table)

        logger.info(f"Found {len(mu_tables)} signal strength tables")
        return mu_tables

    def _extract_production_mode(self, table) -> Optional[str]:
        """Extract production mode from table metadata."""
        text = f"{table.name} {table.description}".lower()

        for mode in self.PRODUCTION_MODES:
            if mode.lower() in text:
                return mode

        # Check for alternative names
        alternatives = {
            "gluon fusion": "ggF",
            "vector boson fusion": "VBF",
            "associated production": "WH",
            "tth": "ttH",
        }
        for alt, mode in alternatives.items():
            if alt in text:
                return mode

        return None

    def _extract_decay_mode(self, table) -> Optional[str]:
        """Extract decay mode from table metadata."""
        text = f"{table.name} {table.description}".lower()

        mapping = {
            "gamma": "gamgam",
            "γγ": "gamgam",
            "diphoton": "gamgam",
            "zz": "ZZ",
            "4l": "ZZ",
            "ww": "WW",
            "tau": "tautau",
            "ττ": "tautau",
            "bb": "bb",
            "mu": "mumu",
            "μμ": "mumu",
        }

        for key, mode in mapping.items():
            if key in text:
                return mode

        return None

    def _extract_mu_from_table(self, df: pd.DataFrame) -> tuple[Optional[float], float]:
        """Extract signal strength value and error from DataFrame."""
        # Look for columns with mu values
        for col in df.columns:
            col_lower = str(col).lower()
            if "mu" in col_lower or "signal" in col_lower or "strength" in col_lower:
                try:
                    value = float(df[col].iloc[0])
                    # Find error column
                    err_col = None
                    for ec in df.columns:
                        if "err" in str(ec).lower() or "unc" in str(ec).lower():
                            err_col = ec
                            break

                    if err_col:
                        error = float(df[err_col].iloc[0])
                    else:
                        error = 0.1 * abs(value)  # Default 10% uncertainty

                    return (value, error)
                except (ValueError, IndexError):
                    continue

        return (None, 0.0)

    def _parse_matrix_table(self, df: pd.DataFrame, cell_values: dict) -> None:
        """Parse a matrix-style table into cell values."""
        # This is a fallback for complex tables
        # Try to identify rows as production modes and columns as decays
        for idx, row in df.iterrows():
            row_str = str(row.iloc[0]).strip() if len(row) > 0 else ""

            # Check if row label is a production mode
            prod = None
            for mode in self.PRODUCTION_MODES:
                if mode.lower() in row_str.lower():
                    prod = mode
                    break

            if prod:
                # Try to parse remaining columns as decay measurements
                for col in df.columns[1:]:
                    decay = self._column_to_decay(str(col))
                    if decay:
                        try:
                            value = float(row[col])
                            cell_values[(prod, decay)] = (value, 0.1 * abs(value))
                        except (ValueError, TypeError):
                            continue

    def _column_to_decay(self, col_name: str) -> Optional[str]:
        """Convert column name to decay mode."""
        col_lower = col_name.lower()
        mapping = {
            "gamma": "gamgam",
            "zz": "ZZ",
            "ww": "WW",
            "tau": "tautau",
            "bb": "bb",
            "mu": "mumu",
        }
        for key, mode in mapping.items():
            if key in col_lower:
                return mode
        return None

    def _get_correlation_matrix(
        self, prod_labels: list[str], decay_labels: list[str]
    ) -> Optional[np.ndarray]:
        """Try to fetch correlation matrix from HEPData."""
        # Look for correlation table
        for table in self._record.tables:
            if "corr" in table.name.lower() or "correlation" in table.description.lower():
                try:
                    df = table.to_dataframe()
                    # Parse correlation values
                    n = len(prod_labels) * len(decay_labels)
                    if len(df) >= n * n:
                        # Assume it's a flattened correlation matrix
                        corr_vals = df.select_dtypes(include=[np.number]).values.flatten()
                        if len(corr_vals) >= n * n:
                            return corr_vals[: n * n].reshape(n, n)
                except Exception as e:
                    logger.debug(f"Failed to parse correlation table: {e}")

        return None

    def _fallback_extraction(self) -> MatrixData:
        """
        Fallback extraction using known structure.

        Uses typical ATLAS signal strength values as placeholders
        when automatic parsing fails.

        WARNING: This uses hardcoded approximate values. Results
        should be verified against actual HEPData tables.
        """
        logger.warning(
            "FALLBACK ACTIVE: Using hardcoded signal strength values. "
            "These are approximate and may not reflect current published results."
        )

        # Representative values (approximately SM-like)
        # These should be replaced by actual extracted values
        fallback_data = {
            ("ggF", "gamgam"): (1.04, 0.10),
            ("ggF", "ZZ"): (1.01, 0.11),
            ("ggF", "WW"): (1.08, 0.12),
            ("ggF", "tautau"): (0.96, 0.15),
            ("VBF", "gamgam"): (1.13, 0.15),
            ("VBF", "ZZ"): (1.02, 0.20),
            ("VBF", "WW"): (0.95, 0.18),
            ("VBF", "tautau"): (1.05, 0.12),
            ("WH", "bb"): (0.98, 0.20),
            ("WH", "WW"): (1.10, 0.25),
            ("ZH", "bb"): (1.02, 0.18),
            ("ZH", "WW"): (0.90, 0.30),
            ("ttH", "gamgam"): (1.38, 0.25),
            ("ttH", "bb"): (0.85, 0.20),
            ("ttH", "WW"): (1.05, 0.30),
        }

        prod_labels = ["ggF", "VBF", "WH", "ZH", "ttH"]
        decay_labels = ["gamgam", "ZZ", "WW", "tautau", "bb"]

        prod_idx = {p: i for i, p in enumerate(prod_labels)}
        decay_idx = {d: i for i, d in enumerate(decay_labels)}

        observations = []
        for (prod, decay), (value, error) in fallback_data.items():
            if prod in prod_idx and decay in decay_idx:
                obs = MatrixObservation(
                    row_idx=prod_idx[prod],
                    col_idx=decay_idx[decay],
                    row_label=self.PROD_DISPLAY.get(prod, prod),
                    col_label=self.DECAY_DISPLAY.get(decay, decay),
                    value=value,
                    total_err=error,
                )
                observations.append(obs)

        # Create fallback provenance if needed
        if self._provenance is None:
            self._provenance = create_provenance(
                DataOrigin.FALLBACK,
                "Hardcoded fallback values (HEPData parsing failed)"
            )
            self._provenance.add_source(
                source_type="hardcoded",
                identifier="fallback_atlas_higgs_mu",
                doi=self.source_dois[0],
            )
            self._provenance.record_fallback(
                step="build_observations",
                reason="Complete fallback - no provenance from fetch",
                fallback_to="hardcoded_values",
            )

        self._provenance.extra["fallback_active"] = True
        self._provenance.extra["fallback_n_values"] = len(observations)

        return MatrixData(
            name=self.name,
            description=self.description + " (FALLBACK - verify values)",
            row_labels=[self.PROD_DISPLAY.get(p, p) for p in prod_labels],
            col_labels=[self.DECAY_DISPLAY.get(d, d) for d in decay_labels],
            observations=observations,
            metadata={
                "source_doi": self.source_dois[0],
                "extraction_method": "fallback",
                "warning": "FALLBACK VALUES - may not reflect actual published data",
            },
            provenance=self._provenance,
        )

    def cross_checks(self) -> list[CrossCheckResult]:
        """Run validation checks on the dataset."""
        results = []

        # Check 1: Verify record DOI/ID
        try:
            if self._record is None:
                self._record = self.hepdata.download_record(self.HEPDATA_RECORD_ID)

            doi_match = str(self.HEPDATA_RECORD_ID) in str(self._record.record_id)
            results.append(CrossCheckResult(
                name="record_id_verification",
                passed=doi_match,
                message=f"Record ID: {self._record.record_id}",
                details={"record_id": self._record.record_id},
            ))
        except Exception as e:
            results.append(CrossCheckResult(
                name="record_id_verification",
                passed=False,
                message=f"Failed to verify record: {e}",
            ))

        # Check 2: Matrix completeness
        try:
            data = self.get_matrix_data()
            fill_fraction = data.n_obs / (data.n_rows * data.n_cols)
            is_complete = fill_fraction > 0.5

            results.append(CrossCheckResult(
                name="matrix_completeness",
                passed=is_complete,
                message=f"Fill fraction: {fill_fraction:.2%}",
                details={"fill_fraction": fill_fraction, "n_obs": data.n_obs},
            ))
        except Exception as e:
            results.append(CrossCheckResult(
                name="matrix_completeness",
                passed=False,
                message=f"Failed to check completeness: {e}",
            ))

        # Check 3: Values are physically reasonable (μ should be O(1))
        try:
            data = self.get_matrix_data()
            values, _, _ = data.to_matrix()
            values_flat = values[~np.isnan(values)]

            reasonable = (
                np.all(values_flat > -1) and
                np.all(values_flat < 5) and
                np.mean(values_flat) > 0.5 and
                np.mean(values_flat) < 2.0
            )

            results.append(CrossCheckResult(
                name="physical_reasonability",
                passed=reasonable,
                message=f"Mean μ = {np.mean(values_flat):.3f}, range [{values_flat.min():.3f}, {values_flat.max():.3f}]",
                details={
                    "mean": float(np.mean(values_flat)),
                    "min": float(values_flat.min()),
                    "max": float(values_flat.max()),
                },
            ))
        except Exception as e:
            results.append(CrossCheckResult(
                name="physical_reasonability",
                passed=False,
                message=f"Failed check: {e}",
            ))

        # Check 4: Errors are reasonable (not too small or large)
        try:
            data = self.get_matrix_data()
            _, errors, _ = data.to_matrix()
            errors_flat = errors[~np.isnan(errors)]

            # Relative errors should typically be 5-50%
            rel_errors = errors_flat / np.abs(values_flat)
            reasonable_errors = (
                np.all(rel_errors > 0.01) and
                np.all(rel_errors < 1.0) and
                np.mean(rel_errors) > 0.05
            )

            results.append(CrossCheckResult(
                name="error_reasonability",
                passed=reasonable_errors,
                message=f"Mean relative error: {np.mean(rel_errors):.1%}",
                details={"mean_rel_error": float(np.mean(rel_errors))},
            ))
        except Exception as e:
            results.append(CrossCheckResult(
                name="error_reasonability",
                passed=False,
                message=f"Failed check: {e}",
            ))

        return results


def fetch_higgs_atlas(
    cache_dir: Optional[Path] = None,
    raw_dir: Optional[Path] = None,
    processed_dir: Optional[Path] = None,
    force: bool = False,
) -> MatrixData:
    """
    Convenience function to fetch ATLAS Higgs data.

    Returns:
        MatrixData object with signal strength measurements
    """
    dataset = HiggsATLASDataset(
        cache_dir=cache_dir,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
    )
    dataset.fetch_raw(force=force)
    return dataset.get_matrix_data(force_rebuild=force)
