"""
HEPData client for fetching high-energy physics data.

HEPData (https://www.hepdata.net) is the official repository for
HEP publication data. This client supports:
- Record metadata retrieval
- Table data download (JSON, YAML, CSV)
- Resource file downloads
- Correlation/covariance matrices when available
"""

import json
import yaml
import re
from pathlib import Path
from typing import Optional, Any
from dataclasses import dataclass

import pandas as pd
import numpy as np

from rank1.utils.http import HTTPClient
from rank1.logging import get_logger

logger = get_logger()

HEPDATA_BASE_URL = "https://www.hepdata.net"


@dataclass
class HEPDataTable:
    """A single table from a HEPData record."""

    name: str
    description: str
    keywords: dict[str, list[str]]
    independent_variables: list[dict[str, Any]]
    dependent_variables: list[dict[str, Any]]

    @property
    def n_rows(self) -> int:
        """Number of data points."""
        if self.independent_variables:
            return len(self.independent_variables[0].get("values", []))
        return 0

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert table to pandas DataFrame.

        Creates columns for each independent and dependent variable,
        with error columns for uncertainties.
        """
        data = {}

        # Independent variables (x-axes)
        for i, var in enumerate(self.independent_variables):
            name = var.get("header", {}).get("name", f"x{i}")
            values = var.get("values", [])

            # Handle binned vs point values
            vals = []
            lows = []
            highs = []

            for v in values:
                if "value" in v:
                    vals.append(v["value"])
                    lows.append(v.get("low", v["value"]))
                    highs.append(v.get("high", v["value"]))
                elif "low" in v and "high" in v:
                    vals.append((v["low"] + v["high"]) / 2)
                    lows.append(v["low"])
                    highs.append(v["high"])

            data[name] = vals
            if lows != vals or highs != vals:
                data[f"{name}_low"] = lows
                data[f"{name}_high"] = highs

        # Dependent variables (y-axes)
        for i, var in enumerate(self.dependent_variables):
            name = var.get("header", {}).get("name", f"y{i}")
            values = var.get("values", [])

            vals = []
            stat_errs = []
            sys_errs = []
            total_errs = []

            for v in values:
                vals.append(v.get("value"))

                # Parse errors
                errors = v.get("errors", [])
                stat = 0.0
                sys = 0.0

                for err in errors:
                    label = err.get("label", "").lower()
                    if "asymerror" in err:
                        # Asymmetric error - take average of abs values
                        asym = err["asymerror"]
                        err_val = (abs(asym.get("plus", 0)) + abs(asym.get("minus", 0))) / 2
                    elif "symerror" in err:
                        err_val = abs(err["symerror"])
                    else:
                        err_val = 0

                    if "stat" in label:
                        stat = err_val
                    elif "sys" in label or "syst" in label:
                        sys = err_val
                    else:
                        # Unknown error type - treat as systematic
                        sys = np.sqrt(sys**2 + err_val**2)

                stat_errs.append(stat)
                sys_errs.append(sys)
                total_errs.append(np.sqrt(stat**2 + sys**2))

            data[name] = vals
            data[f"{name}_stat_err"] = stat_errs
            data[f"{name}_sys_err"] = sys_errs
            data[f"{name}_total_err"] = total_errs

        return pd.DataFrame(data)


@dataclass
class HEPDataRecord:
    """Metadata and tables from a HEPData record."""

    record_id: int
    inspire_id: Optional[str]
    doi: Optional[str]
    title: str
    collaboration: Optional[str]
    tables: list[HEPDataTable]
    resources: list[dict[str, Any]]

    def get_table(self, name: str) -> Optional[HEPDataTable]:
        """Get table by name."""
        for table in self.tables:
            if table.name == name:
                return table
        return None

    def get_table_names(self) -> list[str]:
        """Get list of table names."""
        return [t.name for t in self.tables]


class HEPDataClient:
    """
    Client for accessing HEPData records and tables.

    Supports caching and multiple output formats.
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        raw_dir: Optional[Path] = None,
    ):
        """
        Initialize HEPData client.

        Args:
            cache_dir: Directory for HTTP cache
            raw_dir: Directory for downloaded raw files
        """
        self.cache_dir = cache_dir or Path("data/cache")
        self.raw_dir = raw_dir or Path("data/raw/hepdata")
        self.http = HTTPClient(cache_dir=self.cache_dir)

    def get_record_metadata(self, record_id: int) -> dict[str, Any]:
        """
        Fetch record metadata.

        Args:
            record_id: HEPData record ID

        Returns:
            Record metadata dict
        """
        url = f"{HEPDATA_BASE_URL}/record/{record_id}?format=json"
        logger.info(f"Fetching HEPData record {record_id}")

        response = self.http.get(url)
        return response.json()

    def get_record_by_inspire_id(self, inspire_id: str) -> dict[str, Any]:
        """
        Fetch record by INSPIRE ID (e.g., 'ins12345').

        Args:
            inspire_id: INSPIRE record ID

        Returns:
            Record metadata dict
        """
        url = f"{HEPDATA_BASE_URL}/record/{inspire_id}?format=json"
        logger.info(f"Fetching HEPData record for INSPIRE {inspire_id}")

        response = self.http.get(url)
        return response.json()

    def get_table_data(
        self,
        record_id: int,
        table_name: str,
        format: str = "json",
    ) -> Any:
        """
        Fetch a specific table from a record.

        Args:
            record_id: HEPData record ID
            table_name: Table name
            format: Output format (json, yaml, csv)

        Returns:
            Table data in requested format
        """
        # URL encode table name
        table_encoded = table_name.replace(" ", "%20")
        url = f"{HEPDATA_BASE_URL}/download/table/{record_id}/{table_encoded}/{format}"

        logger.debug(f"Fetching table {table_name} from record {record_id}")
        response = self.http.get(url)

        if format == "json":
            return response.json()
        elif format == "yaml":
            return yaml.safe_load(response.text)
        else:
            return response.text

    def download_record(
        self,
        record_id: int,
        force: bool = False,
    ) -> HEPDataRecord:
        """
        Download complete record with all tables.

        Args:
            record_id: HEPData record ID
            force: Force re-download

        Returns:
            HEPDataRecord object
        """
        # Check cache
        cache_file = self.raw_dir / f"record_{record_id}.json"

        if not force and cache_file.exists():
            logger.debug(f"Loading cached record {record_id}")
            with open(cache_file) as f:
                data = json.load(f)
        else:
            # Fetch metadata
            data = self.get_record_metadata(record_id)

            # Save to cache
            self.raw_dir.mkdir(parents=True, exist_ok=True)
            with open(cache_file, "w") as f:
                json.dump(data, f, indent=2)

        return self._parse_record(record_id, data)

    def download_record_by_inspire(
        self,
        inspire_id: str,
        force: bool = False,
    ) -> HEPDataRecord:
        """
        Download record by INSPIRE ID.

        Args:
            inspire_id: INSPIRE record ID (e.g., 'ins718189')
            force: Force re-download

        Returns:
            HEPDataRecord object
        """
        # Check cache
        cache_file = self.raw_dir / f"record_{inspire_id}.json"

        if not force and cache_file.exists():
            logger.debug(f"Loading cached record {inspire_id}")
            with open(cache_file) as f:
                data = json.load(f)
        else:
            # Fetch metadata
            data = self.get_record_by_inspire_id(inspire_id)

            # Save to cache
            self.raw_dir.mkdir(parents=True, exist_ok=True)
            with open(cache_file, "w") as f:
                json.dump(data, f, indent=2)

        # Extract numeric ID from response
        record_id = data.get("record_id", 0)

        return self._parse_record(record_id, data)

    def _parse_record(self, record_id: int, data: dict) -> HEPDataRecord:
        """Parse raw record data into HEPDataRecord."""
        tables = []

        # HEPData API returns tables in "data_tables" field, not "tables"
        table_list = data.get("data_tables", []) or data.get("tables", [])

        for table_data in table_list:
            table = HEPDataTable(
                name=table_data.get("name", ""),
                description=table_data.get("description", ""),
                keywords=table_data.get("keywords", {}),
                independent_variables=table_data.get("independent_variables", []),
                dependent_variables=table_data.get("dependent_variables", []),
            )
            tables.append(table)

        return HEPDataRecord(
            record_id=record_id,
            inspire_id=data.get("inspire_id"),
            doi=data.get("doi"),
            title=data.get("title", ""),
            collaboration=data.get("collaboration"),
            tables=tables,
            resources=data.get("resources", []),
        )

    def download_resource(
        self,
        record_id: int,
        resource_url: str,
        filename: Optional[str] = None,
        force: bool = False,
    ) -> Path:
        """
        Download a resource file from a record.

        Args:
            record_id: HEPData record ID
            resource_url: URL of the resource
            filename: Local filename (derived from URL if not specified)
            force: Force re-download

        Returns:
            Path to downloaded file
        """
        if filename is None:
            filename = resource_url.split("/")[-1]

        dest = self.raw_dir / str(record_id) / filename

        if not force and dest.exists():
            logger.debug(f"Using cached resource: {dest}")
            return dest

        return self.http.download(resource_url, dest, force=force)

    def get_correlation_matrix(
        self,
        record_id: int,
        table_name: str,
    ) -> Optional[np.ndarray]:
        """
        Try to find and parse correlation matrix for a table.

        Many HEPData records include correlation matrices as separate
        tables or resources.

        Args:
            record_id: HEPData record ID
            table_name: Table name to find correlation for

        Returns:
            Correlation matrix as numpy array, or None if not found
        """
        record = self.download_record(record_id)

        # Common naming patterns for correlation tables
        corr_patterns = [
            f"Correlation {table_name}",
            f"Correlation matrix {table_name}",
            f"{table_name} correlation",
            f"{table_name}_correlation",
            "Correlation matrix",
        ]

        for name in corr_patterns:
            table = record.get_table(name)
            if table:
                return self._parse_correlation_table(table)

        # Check resources for correlation files
        for resource in record.resources:
            if "correlation" in resource.get("description", "").lower():
                # Try to download and parse
                try:
                    path = self.download_resource(
                        record_id,
                        resource["location"],
                    )
                    return self._parse_correlation_file(path)
                except Exception as e:
                    logger.warning(f"Failed to parse correlation resource: {e}")

        return None

    def _parse_correlation_table(self, table: HEPDataTable) -> Optional[np.ndarray]:
        """Parse a correlation matrix table."""
        try:
            df = table.to_dataframe()

            # Try to find the correlation values column
            for col in df.columns:
                if "corr" in col.lower() or "rho" in col.lower():
                    values = df[col].values

                    # Infer matrix size
                    n = int(np.sqrt(len(values)))
                    if n * n == len(values):
                        return values.reshape(n, n)

            return None

        except Exception as e:
            logger.warning(f"Failed to parse correlation table: {e}")
            return None

    def _parse_correlation_file(self, path: Path) -> Optional[np.ndarray]:
        """Parse a correlation matrix file (various formats)."""
        try:
            if path.suffix in (".yaml", ".yml"):
                with open(path) as f:
                    data = yaml.safe_load(f)
                # Try various structures
                if isinstance(data, list):
                    return np.array(data)
                elif isinstance(data, dict) and "correlation" in data:
                    return np.array(data["correlation"])

            elif path.suffix == ".json":
                with open(path) as f:
                    data = json.load(f)
                if isinstance(data, list):
                    return np.array(data)
                elif isinstance(data, dict) and "correlation" in data:
                    return np.array(data["correlation"])

            elif path.suffix == ".csv":
                df = pd.read_csv(path, header=None)
                return df.values

            return None

        except Exception as e:
            logger.warning(f"Failed to parse correlation file {path}: {e}")
            return None

    def search_records(
        self,
        query: str,
        collaboration: Optional[str] = None,
        page: int = 1,
        size: int = 10,
    ) -> dict[str, Any]:
        """
        Search HEPData records.

        Args:
            query: Search query
            collaboration: Filter by collaboration
            page: Page number
            size: Results per page

        Returns:
            Search results dict
        """
        params = {
            "q": query,
            "page": page,
            "size": size,
            "format": "json",
        }
        if collaboration:
            params["collaboration"] = collaboration

        url = f"{HEPDATA_BASE_URL}/search"
        response = self.http.get(url, params=params)
        return response.json()


def fetch_hepdata_record(
    record_id: int,
    cache_dir: Optional[Path] = None,
    raw_dir: Optional[Path] = None,
) -> HEPDataRecord:
    """
    Convenience function to fetch a HEPData record.

    Args:
        record_id: HEPData record ID
        cache_dir: Cache directory
        raw_dir: Raw data directory

    Returns:
        HEPDataRecord object
    """
    client = HEPDataClient(cache_dir=cache_dir, raw_dir=raw_dir)
    return client.download_record(record_id)


def fetch_hepdata_by_inspire(
    inspire_id: str,
    cache_dir: Optional[Path] = None,
    raw_dir: Optional[Path] = None,
) -> HEPDataRecord:
    """
    Convenience function to fetch a HEPData record by INSPIRE ID.

    Args:
        inspire_id: INSPIRE record ID
        cache_dir: Cache directory
        raw_dir: Raw data directory

    Returns:
        HEPDataRecord object
    """
    client = HEPDataClient(cache_dir=cache_dir, raw_dir=raw_dir)
    return client.download_record_by_inspire(inspire_id)
