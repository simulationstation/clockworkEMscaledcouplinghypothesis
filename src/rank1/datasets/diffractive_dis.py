"""
Diffractive DIS datasets from H1 and ZEUS experiments.

Sources:
- H1: HEPData record ins718189 (DOI: 10.17182/hepdata.45891.v1)
- ZEUS: HEPData record ins675372 (DOI: 10.17182/hepdata.11816.v1)

The rank-1 hypothesis tests Regge factorization:
σ_r^D(β, Q², x_P) ≈ f_P(x_P) × σ_r(β, Q²)

where f_P is the Pomeron flux and σ_r is the reduced cross section.
This tests whether the x_P dependence separates from the (β, Q²) dependence.
"""

from pathlib import Path
from typing import Optional, Any

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


class DiffractiveDISDataset(MatrixDataset):
    """
    H1 and ZEUS diffractive DIS dataset.

    Rows: (Q², β) combinations
    Columns: x_P bins

    Tests whether the diffractive reduced cross section factorizes
    as flux(x_P) × structure(β, Q²).
    """

    name = "diffractive_dis"
    description = "H1/ZEUS diffractive DIS: Regge factorization test"
    source_dois = [
        "10.17182/hepdata.45891.v1",  # H1
        "10.17182/hepdata.11816.v1",  # ZEUS
    ]
    source_urls = [
        "https://www.hepdata.net/record/ins718189",
        "https://www.hepdata.net/record/ins675372",
    ]

    # HEPData record IDs
    H1_INSPIRE_ID = "ins718189"
    ZEUS_INSPIRE_ID = "ins675372"

    # Kinematic selection criteria
    Q2_MIN = 4.0   # GeV²
    Q2_MAX = 100.0 # GeV²
    BETA_MIN = 0.01
    BETA_MAX = 0.9
    XP_MIN = 0.0003
    XP_MAX = 0.03

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        raw_dir: Optional[Path] = None,
        processed_dir: Optional[Path] = None,
        experiment: str = "combined",
    ):
        """
        Initialize diffractive DIS dataset.

        Args:
            cache_dir: HTTP cache directory
            raw_dir: Raw data directory
            processed_dir: Processed data directory
            experiment: 'H1', 'ZEUS', or 'combined'
        """
        super().__init__(cache_dir, raw_dir, processed_dir)

        self.experiment = experiment.upper() if experiment != "combined" else "combined"

        self.hepdata = HEPDataClient(
            cache_dir=self.cache_dir,
            raw_dir=self.raw_dir / "hepdata",
        )

        self._h1_record = None
        self._zeus_record = None
        self._combined_df = None
        self._provenance: Optional[DataProvenance] = None

    def fetch_raw(self, force: bool = False) -> list[Path]:
        """Download HEPData records for H1 and ZEUS."""
        paths = []

        # Initialize provenance
        self._provenance = create_provenance(
            DataOrigin.API,
            f"H1/ZEUS diffractive DIS data (experiment={self.experiment})"
        )
        self._provenance.set_fetch_timestamp()

        # H1 data
        if self.experiment in ["H1", "COMBINED", "combined"]:
            logger.info(f"Fetching H1 diffractive DIS data: {self.H1_INSPIRE_ID}")
            try:
                self._h1_record = self.hepdata.download_record_by_inspire(
                    self.H1_INSPIRE_ID, force=force
                )
                cache_path = self.raw_dir / "hepdata" / f"record_{self.H1_INSPIRE_ID}.json"
                paths.append(cache_path)

                self._provenance.add_source(
                    source_type="hepdata",
                    identifier=self.H1_INSPIRE_ID,
                    doi=self.source_dois[0],
                    url=self.source_urls[0],
                )
                if cache_path.exists():
                    self._provenance.add_file_hash(cache_path)

            except Exception as e:
                logger.warning(f"FETCH FAILED: H1 data from HEPData: {e}")
                self._provenance.record_fallback(
                    step="fetch_h1",
                    reason=str(e),
                    fallback_to="skip_experiment",
                    original_source=f"hepdata:{self.H1_INSPIRE_ID}",
                )

        # ZEUS data
        if self.experiment in ["ZEUS", "COMBINED", "combined"]:
            logger.info(f"Fetching ZEUS diffractive DIS data: {self.ZEUS_INSPIRE_ID}")
            try:
                self._zeus_record = self.hepdata.download_record_by_inspire(
                    self.ZEUS_INSPIRE_ID, force=force
                )
                cache_path = self.raw_dir / "hepdata" / f"record_{self.ZEUS_INSPIRE_ID}.json"
                paths.append(cache_path)

                self._provenance.add_source(
                    source_type="hepdata",
                    identifier=self.ZEUS_INSPIRE_ID,
                    doi=self.source_dois[1],
                    url=self.source_urls[1],
                )
                if cache_path.exists():
                    self._provenance.add_file_hash(cache_path)

            except Exception as e:
                logger.warning(f"FETCH FAILED: ZEUS data from HEPData: {e}")
                self._provenance.record_fallback(
                    step="fetch_zeus",
                    reason=str(e),
                    fallback_to="skip_experiment",
                    original_source=f"hepdata:{self.ZEUS_INSPIRE_ID}",
                )

        return paths

    def build_observations(self) -> MatrixData:
        """
        Build matrix observations from diffractive structure function data.

        Creates a matrix where:
        - Rows are (Q², β) combinations
        - Columns are x_P bins
        - Values are the diffractive reduced cross section
        """
        # Ensure provenance exists
        if self._provenance is None:
            self._provenance = create_provenance(
                DataOrigin.API,
                f"H1/ZEUS diffractive DIS data (experiment={self.experiment})"
            )

        self._provenance.set_build_timestamp()

        # Ensure data is loaded
        if self._h1_record is None and self.experiment in ["H1", "COMBINED", "combined"]:
            self._h1_record = self.hepdata.download_record_by_inspire(self.H1_INSPIRE_ID)
        if self._zeus_record is None and self.experiment in ["ZEUS", "COMBINED", "combined"]:
            self._zeus_record = self.hepdata.download_record_by_inspire(self.ZEUS_INSPIRE_ID)

        # Parse data into DataFrame
        all_data = []

        if self._h1_record:
            h1_df = self._parse_h1_data()
            if h1_df is not None:
                h1_df["experiment"] = "H1"
                all_data.append(h1_df)
                logger.info(f"Parsed {len(h1_df)} rows from H1 data")

        if self._zeus_record:
            zeus_df = self._parse_zeus_data()
            if zeus_df is not None:
                zeus_df["experiment"] = "ZEUS"
                all_data.append(zeus_df)
                logger.info(f"Parsed {len(zeus_df)} rows from ZEUS data")

        if not all_data:
            # FAIL LOUDLY instead of silently using placeholder
            error_msg = (
                "FATAL: No data could be parsed from HEPData records.\n"
                f"  - H1 record ({self.H1_INSPIRE_ID}): {'loaded' if self._h1_record else 'FAILED'}\n"
                f"  - ZEUS record ({self.ZEUS_INSPIRE_ID}): {'loaded' if self._zeus_record else 'FAILED'}\n"
                "Cannot proceed without at least one valid data source.\n"
                "Run 'rank1 doctor' to diagnose and fix data fetching issues."
            )
            logger.error(error_msg)
            self._provenance.record_fallback(
                step="build_observations",
                reason="No data parsed from any HEPData records - ANALYSIS ABORTED",
                fallback_to="abort",
            )
            raise ValueError(error_msg)

        # Combine data
        combined = pd.concat(all_data, ignore_index=True)
        self._combined_df = combined
        n_before_cuts = len(combined)

        # Apply kinematic cuts
        mask = (
            (combined["Q2"] >= self.Q2_MIN) &
            (combined["Q2"] <= self.Q2_MAX) &
            (combined["beta"] >= self.BETA_MIN) &
            (combined["beta"] <= self.BETA_MAX) &
            (combined["xP"] >= self.XP_MIN) &
            (combined["xP"] <= self.XP_MAX)
        )
        combined = combined[mask].copy()

        # Record the kinematic filter
        self._provenance.add_filter(
            name="kinematic_cuts",
            description="Apply kinematic selection cuts",
            parameters={
                "Q2_range": [self.Q2_MIN, self.Q2_MAX],
                "beta_range": [self.BETA_MIN, self.BETA_MAX],
                "xP_range": [self.XP_MIN, self.XP_MAX],
            },
            n_before=n_before_cuts,
            n_after=len(combined),
        )

        if len(combined) == 0:
            # FAIL LOUDLY instead of silently using placeholder
            error_msg = (
                f"FATAL: No data remaining after kinematic cuts.\n"
                f"  - Rows before cuts: {n_before_cuts}\n"
                f"  - Q² range: [{self.Q2_MIN}, {self.Q2_MAX}] GeV²\n"
                f"  - β range: [{self.BETA_MIN}, {self.BETA_MAX}]\n"
                f"  - x_P range: [{self.XP_MIN}, {self.XP_MAX}]\n"
                "The kinematic selection may be too restrictive for the available data.\n"
                "Run 'rank1 doctor' to diagnose data coverage issues."
            )
            logger.error(error_msg)
            self._provenance.record_fallback(
                step="kinematic_cuts",
                reason=f"All {n_before_cuts} rows filtered out - ANALYSIS ABORTED",
                fallback_to="abort",
            )
            raise ValueError(error_msg)

        # Create binned labels
        combined["row_key"] = combined.apply(
            lambda r: f"Q2={r['Q2']:.1f}_beta={r['beta']:.3f}",
            axis=1
        )
        combined["col_key"] = combined.apply(
            lambda r: f"xP={r['xP']:.5f}",
            axis=1
        )

        # Get unique row/column labels
        row_keys = sorted(combined["row_key"].unique())
        col_keys = sorted(combined["col_key"].unique())

        row_idx_map = {k: i for i, k in enumerate(row_keys)}
        col_idx_map = {k: i for i, k in enumerate(col_keys)}

        # Create observations
        observations = []
        for _, row in combined.iterrows():
            obs = MatrixObservation(
                row_idx=row_idx_map[row["row_key"]],
                col_idx=col_idx_map[row["col_key"]],
                row_label=row["row_key"],
                col_label=row["col_key"],
                value=row["value"],
                stat_err=row.get("stat_err", 0),
                sys_err=row.get("sys_err", 0),
                total_err=row.get("total_err", row.get("stat_err", 0.1 * abs(row["value"]))),
            )
            observations.append(obs)

        # Track experiments included
        experiments_used = combined["experiment"].unique().tolist()
        self._provenance.extra["experiments_included"] = experiments_used

        logger.info(
            f"Built diffractive DIS matrix: {len(row_keys)} (Q²,β) bins x "
            f"{len(col_keys)} x_P bins, {len(observations)} observations"
        )

        return MatrixData(
            name=self.name,
            description=self.description,
            row_labels=row_keys,
            col_labels=col_keys,
            observations=observations,
            metadata={
                "experiment": self.experiment,
                "experiments_included": experiments_used,
                "kinematic_cuts": {
                    "Q2_range": [self.Q2_MIN, self.Q2_MAX],
                    "beta_range": [self.BETA_MIN, self.BETA_MAX],
                    "xP_range": [self.XP_MIN, self.XP_MAX],
                },
            },
            provenance=self._provenance,
        )

    def _parse_h1_data(self) -> Optional[pd.DataFrame]:
        """Parse H1 HEPData tables into DataFrame."""
        if self._h1_record is None:
            return None

        all_rows = []

        for table in self._h1_record.tables:
            # Try all tables - let column parsing decide if usable
            # (HEPData descriptions are often empty or generic)
            try:
                # Fetch actual table data from HEPData API
                table_data = self.hepdata.get_table_data(
                    self._h1_record.record_id or self.H1_INSPIRE_ID,
                    table.name,
                    format="json"
                )
                df = self._hepdata_values_to_dataframe(table_data)
                if df is None or df.empty:
                    continue

                # Try to identify columns
                parsed = self._parse_table_columns(df, "H1")
                if parsed is not None:
                    all_rows.extend(parsed)

            except Exception as e:
                logger.debug(f"Failed to parse H1 table {table.name}: {e}")

        if not all_rows:
            return None

        return pd.DataFrame(all_rows)

    def _parse_zeus_data(self) -> Optional[pd.DataFrame]:
        """Parse ZEUS HEPData tables into DataFrame."""
        if self._zeus_record is None:
            return None

        all_rows = []

        for table in self._zeus_record.tables:
            # Try all tables - let column parsing decide if usable
            # (HEPData descriptions are often empty or generic)
            try:
                # Fetch actual table data from HEPData API
                table_data = self.hepdata.get_table_data(
                    self._zeus_record.record_id or self.ZEUS_INSPIRE_ID,
                    table.name,
                    format="json"
                )
                df = self._hepdata_values_to_dataframe(table_data)
                if df is None or df.empty:
                    continue

                parsed = self._parse_table_columns(df, "ZEUS")
                if parsed is not None:
                    all_rows.extend(parsed)

            except Exception as e:
                logger.debug(f"Failed to parse ZEUS table {table.name}: {e}")

        if not all_rows:
            return None

        return pd.DataFrame(all_rows)

    def _hepdata_values_to_dataframe(self, table_data: dict) -> Optional[pd.DataFrame]:
        """Convert HEPData table JSON to DataFrame."""
        import numpy as np

        values = table_data.get("values", [])
        if not values:
            return None

        # Build columns from headers
        headers = table_data.get("headers", [])
        x_headers = [h.get("name", f"x{i}") for i, h in enumerate(headers) if i == 0]
        y_headers = [h.get("name", f"y{i}") for i, h in enumerate(headers) if i > 0]

        rows = []
        for v in values:
            row = {}
            # Independent variables (x)
            for i, x_info in enumerate(v.get("x", [])):
                col_name = x_headers[i] if i < len(x_headers) else f"x{i}"
                row[col_name] = float(x_info.get("value", 0))

            # Dependent variables (y)
            for i, y_info in enumerate(v.get("y", [])):
                col_name = y_headers[i] if i < len(y_headers) else f"y{i}"
                row[col_name] = float(y_info.get("value", 0))

                # Add errors
                for err in y_info.get("errors", []):
                    err_label = err.get("label", "err")
                    if "symerror" in err:
                        row[f"{col_name}_{err_label}"] = float(err["symerror"])
                    elif "asymerror" in err:
                        asym = err["asymerror"]
                        plus = abs(float(asym.get("plus", 0)))
                        minus = abs(float(asym.get("minus", 0)))
                        row[f"{col_name}_{err_label}"] = (plus + minus) / 2

            rows.append(row)

        if not rows:
            return None

        return pd.DataFrame(rows)

    def _parse_table_columns(self, df: pd.DataFrame, exp: str) -> Optional[list[dict]]:
        """
        Parse table columns to extract Q², β, x_P and cross section.

        Returns list of dicts with standardized column names.
        """
        rows = []

        # Column name patterns (expanded for HEPData naming conventions)
        q2_patterns = ["q2", "q^2", "q**2", "name=q"]
        beta_patterns = ["beta", "β", "name=beta"]
        xp_patterns = ["xp", "x_p", "xpom", "x_pom", "xi", "x(name=pomeron)", "name=pomeron"]
        sigma_patterns = ["sigma", "f2d", "xsec", "cross", "sig(c=reduced"]

        # Find column mappings
        col_map = {}
        for col in df.columns:
            col_lower = str(col).lower()

            if any(p in col_lower for p in q2_patterns) and "q2" not in col_map:
                col_map["q2"] = col
            elif any(p in col_lower for p in beta_patterns) and "beta" not in col_map:
                col_map["beta"] = col
            elif any(p in col_lower for p in xp_patterns) and "xp" not in col_map:
                col_map["xp"] = col
            elif any(p in col_lower for p in sigma_patterns) and "value" not in col_map:
                if "err" not in col_lower and "unc" not in col_lower:
                    col_map["value"] = col

        # Check we have required columns
        required = ["q2", "beta", "xp", "value"]
        if not all(k in col_map for k in required):
            logger.debug(f"Missing columns: {set(required) - set(col_map.keys())}")
            return None

        # Find error columns
        value_col = col_map["value"]
        err_cols = {}
        for col in df.columns:
            col_lower = str(col).lower()
            if value_col.lower() in col_lower or any(p in col_lower for p in sigma_patterns):
                if "stat" in col_lower:
                    err_cols["stat_err"] = col
                elif "sys" in col_lower:
                    err_cols["sys_err"] = col
                elif "err" in col_lower or "unc" in col_lower:
                    err_cols["total_err"] = col

        # Extract rows
        for idx, row in df.iterrows():
            try:
                data = {
                    "Q2": float(row[col_map["q2"]]),
                    "beta": float(row[col_map["beta"]]),
                    "xP": float(row[col_map["xp"]]),
                    "value": float(row[col_map["value"]]),
                }

                # Add errors
                for err_name, err_col in err_cols.items():
                    try:
                        data[err_name] = float(row[err_col])
                    except (ValueError, TypeError):
                        pass

                # Compute total error if not present
                if "total_err" not in data:
                    stat = data.get("stat_err", 0)
                    sys = data.get("sys_err", 0)
                    if stat > 0 or sys > 0:
                        data["total_err"] = np.sqrt(stat**2 + sys**2)
                    else:
                        data["total_err"] = 0.1 * abs(data["value"])

                if data["value"] > 0:
                    rows.append(data)

            except (ValueError, TypeError) as e:
                continue

        return rows if rows else None

    def _create_placeholder_data(self) -> MatrixData:
        """Create placeholder data when real data unavailable."""
        logger.warning(
            "PLACEHOLDER ACTIVE: Creating synthetic diffractive DIS data. "
            "Results will NOT be valid for publication."
        )

        # Create synthetic data mimicking Regge-like behavior
        Q2_bins = [5.0, 10.0, 20.0, 40.0]
        beta_bins = [0.05, 0.1, 0.2, 0.4]
        xP_bins = [0.001, 0.003, 0.01, 0.02]

        observations = []
        row_labels = []
        col_labels = [f"xP={xp:.4f}" for xp in xP_bins]

        row_idx = 0
        for Q2 in Q2_bins:
            for beta in beta_bins:
                row_label = f"Q2={Q2:.1f}_beta={beta:.2f}"
                row_labels.append(row_label)

                for col_idx, xP in enumerate(xP_bins):
                    # Synthetic factorizable form: f(xP) * g(Q2, beta)
                    f_xP = xP**(-1.1)  # Pomeron-like flux
                    g_Q2_beta = (1 - beta)**2 / Q2  # Structure-like

                    value = 0.1 * f_xP * g_Q2_beta
                    error = 0.1 * value

                    obs = MatrixObservation(
                        row_idx=row_idx,
                        col_idx=col_idx,
                        row_label=row_label,
                        col_label=col_labels[col_idx],
                        value=value,
                        total_err=error,
                    )
                    observations.append(obs)

                row_idx += 1

        # Create placeholder provenance if needed
        if self._provenance is None:
            self._provenance = create_provenance(
                DataOrigin.PLACEHOLDER,
                "Synthetic diffractive DIS data (HEPData parsing failed)"
            )

        self._provenance.extra["is_placeholder"] = True
        self._provenance.extra["placeholder_n_observations"] = len(observations)

        return MatrixData(
            name=self.name,
            description=self.description + " (PLACEHOLDER - NOT VALID)",
            row_labels=row_labels,
            col_labels=col_labels,
            observations=observations,
            metadata={
                "is_placeholder": True,
                "warning": "PLACEHOLDER DATA - Results are NOT valid for publication",
            },
            provenance=self._provenance,
        )

    def cross_checks(self) -> list[CrossCheckResult]:
        """Run validation checks on the dataset."""
        results = []

        # Check 1: Records loaded (CRITICAL - at least 1 must load)
        n_records = sum([
            self._h1_record is not None,
            self._zeus_record is not None,
        ])
        # For combined mode, both should load; for single experiment, that one should load
        if self.experiment == "combined":
            required_records = 1  # At minimum, need one
            ideal_records = 2
        else:
            required_records = 1
            ideal_records = 1

        passed = n_records >= required_records
        results.append(CrossCheckResult(
            name="record_loading",
            passed=passed,
            message=f"Loaded {n_records}/{ideal_records} HEPData records" + (" - CRITICAL FAILURE" if not passed else ""),
            details={
                "h1_loaded": self._h1_record is not None,
                "zeus_loaded": self._zeus_record is not None,
                "required_records": required_records,
                "actual_records": n_records,
            },
        ))

        # Check 2: Data coverage
        try:
            data = self.get_matrix_data()
            results.append(CrossCheckResult(
                name="data_coverage",
                passed=data.n_obs >= 20,
                message=f"{data.n_obs} observations, {data.n_rows}x{data.n_cols} matrix",
                details={
                    "n_obs": data.n_obs,
                    "n_rows": data.n_rows,
                    "n_cols": data.n_cols,
                },
            ))
        except Exception as e:
            results.append(CrossCheckResult(
                name="data_coverage",
                passed=False,
                message=f"Failed to build matrix: {e}",
            ))

        # Check 3: Physical range
        if self._combined_df is not None:
            q2_range = [self._combined_df["Q2"].min(), self._combined_df["Q2"].max()]
            beta_range = [self._combined_df["beta"].min(), self._combined_df["beta"].max()]
            xp_range = [self._combined_df["xP"].min(), self._combined_df["xP"].max()]

            reasonable = (
                q2_range[0] > 0 and q2_range[1] < 1000 and
                beta_range[0] > 0 and beta_range[1] < 1 and
                xp_range[0] > 0 and xp_range[1] < 0.1
            )

            results.append(CrossCheckResult(
                name="physical_range",
                passed=reasonable,
                message=f"Q²: {q2_range}, β: {beta_range}, x_P: {xp_range}",
                details={
                    "Q2_range": q2_range,
                    "beta_range": beta_range,
                    "xP_range": xp_range,
                },
            ))
        else:
            results.append(CrossCheckResult(
                name="physical_range",
                passed=False,
                message="No combined data available",
            ))

        return results


def fetch_diffractive_dis(
    experiment: str = "combined",
    cache_dir: Optional[Path] = None,
    raw_dir: Optional[Path] = None,
    processed_dir: Optional[Path] = None,
    force: bool = False,
) -> MatrixData:
    """
    Convenience function to fetch diffractive DIS data.

    Args:
        experiment: 'H1', 'ZEUS', or 'combined'
        cache_dir: Cache directory
        raw_dir: Raw data directory
        processed_dir: Processed data directory
        force: Force re-download and rebuild

    Returns:
        MatrixData object with diffractive cross sections
    """
    dataset = DiffractiveDISDataset(
        cache_dir=cache_dir,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        experiment=experiment,
    )
    dataset.fetch_raw(force=force)
    return dataset.get_matrix_data(force_rebuild=force)
