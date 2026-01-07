"""
Base classes for matrix datasets.

Provides a uniform interface for different physics datasets to be used
in rank-1 factorization tests.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any
import json

import numpy as np
import pandas as pd

from rank1.logging import get_logger
from rank1.provenance import DataProvenance, DataOrigin, create_provenance

logger = get_logger()


@dataclass
class MatrixObservation:
    """
    A single observation in the matrix.

    Represents one cell of the matrix with its coordinates,
    value, and uncertainties.
    """

    row_idx: int
    col_idx: int
    value: float
    stat_err: float = 0.0
    sys_err: float = 0.0
    total_err: float = 0.0

    row_label: str = ""
    col_label: str = ""

    def __post_init__(self):
        """Compute total error if not provided."""
        if self.total_err == 0.0 and (self.stat_err > 0 or self.sys_err > 0):
            self.total_err = np.sqrt(self.stat_err**2 + self.sys_err**2)

    @property
    def weight(self) -> float:
        """Weight for fitting (1/variance)."""
        if self.total_err > 0:
            return 1.0 / (self.total_err**2)
        return 1.0


@dataclass
class MatrixData:
    """
    Complete matrix data with observations and metadata.
    """

    name: str
    description: str

    row_labels: list[str]
    col_labels: list[str]

    observations: list[MatrixObservation]

    # Optional full covariance matrix for flattened observations
    covariance: Optional[np.ndarray] = None

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    # Provenance tracking - REQUIRED for auditability
    provenance: Optional[DataProvenance] = None

    @property
    def n_rows(self) -> int:
        return len(self.row_labels)

    @property
    def n_cols(self) -> int:
        return len(self.col_labels)

    @property
    def n_obs(self) -> int:
        return len(self.observations)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.n_rows, self.n_cols)

    def to_matrix(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert observations to matrix form.

        Returns:
            (values, uncertainties, mask) where mask[i,j] = 1 if observed
        """
        values = np.full((self.n_rows, self.n_cols), np.nan)
        errors = np.full((self.n_rows, self.n_cols), np.nan)
        mask = np.zeros((self.n_rows, self.n_cols))

        for obs in self.observations:
            values[obs.row_idx, obs.col_idx] = obs.value
            errors[obs.row_idx, obs.col_idx] = obs.total_err
            mask[obs.row_idx, obs.col_idx] = 1.0

        return values, errors, mask

    def to_vectors(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert observations to flattened vectors.

        Returns:
            (row_indices, col_indices, values, errors) as 1D arrays
        """
        rows = np.array([obs.row_idx for obs in self.observations], dtype=np.intp)
        cols = np.array([obs.col_idx for obs in self.observations], dtype=np.intp)
        values = np.array([obs.value for obs in self.observations], dtype=np.float64)
        errors = np.array([obs.total_err for obs in self.observations], dtype=np.float64)

        return rows, cols, values, errors

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        records = []
        for obs in self.observations:
            records.append({
                "row_idx": obs.row_idx,
                "col_idx": obs.col_idx,
                "row_label": obs.row_label,
                "col_label": obs.col_label,
                "value": obs.value,
                "stat_err": obs.stat_err,
                "sys_err": obs.sys_err,
                "total_err": obs.total_err,
            })
        return pd.DataFrame(records)

    def save(self, path: Path) -> None:
        """Save matrix data to file."""
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "name": self.name,
            "description": self.description,
            "row_labels": self.row_labels,
            "col_labels": self.col_labels,
            "observations": [
                {
                    "row_idx": o.row_idx,
                    "col_idx": o.col_idx,
                    "row_label": o.row_label,
                    "col_label": o.col_label,
                    "value": o.value,
                    "stat_err": o.stat_err,
                    "sys_err": o.sys_err,
                    "total_err": o.total_err,
                }
                for o in self.observations
            ],
            "metadata": self.metadata,
        }

        if self.covariance is not None:
            data["covariance"] = self.covariance.tolist()

        # Include provenance in saved data
        if self.provenance is not None:
            data["provenance"] = self.provenance.to_dict()

        if path.suffix == ".parquet":
            df = self.to_dataframe()
            df.to_parquet(path)

            # Save metadata separately
            meta_dict = {
                "name": self.name,
                "description": self.description,
                "row_labels": self.row_labels,
                "col_labels": self.col_labels,
                "metadata": self.metadata,
            }
            if self.provenance is not None:
                meta_dict["provenance"] = self.provenance.to_dict()

            meta_path = path.with_suffix(".meta.json")
            with open(meta_path, "w") as f:
                json.dump(meta_dict, f, indent=2)
        else:
            with open(path, "w") as f:
                json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "MatrixData":
        """Load matrix data from file."""
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
            meta_path = path.with_suffix(".meta.json")

            with open(meta_path) as f:
                meta = json.load(f)

            observations = [
                MatrixObservation(
                    row_idx=int(row["row_idx"]),
                    col_idx=int(row["col_idx"]),
                    row_label=row.get("row_label", ""),
                    col_label=row.get("col_label", ""),
                    value=float(row["value"]),
                    stat_err=float(row.get("stat_err", 0)),
                    sys_err=float(row.get("sys_err", 0)),
                    total_err=float(row.get("total_err", 0)),
                )
                for _, row in df.iterrows()
            ]

            # Load provenance if present
            provenance = None
            if "provenance" in meta:
                provenance = DataProvenance.from_dict(meta["provenance"])
            else:
                # Create provenance indicating loaded from cache
                provenance = create_provenance(
                    DataOrigin.CACHE,
                    f"Loaded from {path}"
                )

            return cls(
                name=meta["name"],
                description=meta["description"],
                row_labels=meta["row_labels"],
                col_labels=meta["col_labels"],
                observations=observations,
                metadata=meta.get("metadata", {}),
                provenance=provenance,
            )

        else:
            with open(path) as f:
                data = json.load(f)

            observations = [
                MatrixObservation(
                    row_idx=o["row_idx"],
                    col_idx=o["col_idx"],
                    row_label=o.get("row_label", ""),
                    col_label=o.get("col_label", ""),
                    value=o["value"],
                    stat_err=o.get("stat_err", 0),
                    sys_err=o.get("sys_err", 0),
                    total_err=o.get("total_err", 0),
                )
                for o in data["observations"]
            ]

            covariance = None
            if "covariance" in data:
                covariance = np.array(data["covariance"])

            # Load provenance if present
            provenance = None
            if "provenance" in data:
                provenance = DataProvenance.from_dict(data["provenance"])
            else:
                # Create provenance indicating loaded from cache
                provenance = create_provenance(
                    DataOrigin.CACHE,
                    f"Loaded from {path}"
                )

            return cls(
                name=data["name"],
                description=data["description"],
                row_labels=data["row_labels"],
                col_labels=data["col_labels"],
                observations=observations,
                covariance=covariance,
                metadata=data.get("metadata", {}),
                provenance=provenance,
            )


@dataclass
class CrossCheckResult:
    """Result of a cross-check validation."""

    name: str
    passed: bool
    message: str
    details: dict[str, Any] = field(default_factory=dict)


class MatrixDataset(ABC):
    """
    Abstract base class for matrix datasets.

    Subclasses must implement:
    - fetch_raw(): Download raw data files
    - build_observations(): Parse raw data into MatrixData
    - cross_checks(): Run dataset-specific validations
    """

    # Class-level metadata
    name: str = "base"
    description: str = "Base dataset class"
    source_dois: list[str] = []
    source_urls: list[str] = []

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        raw_dir: Optional[Path] = None,
        processed_dir: Optional[Path] = None,
    ):
        """
        Initialize dataset.

        Args:
            cache_dir: HTTP cache directory
            raw_dir: Raw data directory
            processed_dir: Processed data directory
        """
        self.cache_dir = cache_dir or Path("data/cache")
        self.raw_dir = raw_dir or Path("data/raw")
        self.processed_dir = processed_dir or Path("data/processed")

        # Ensure directories exist
        for d in [self.cache_dir, self.raw_dir, self.processed_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self._matrix_data: Optional[MatrixData] = None

    @abstractmethod
    def fetch_raw(self, force: bool = False) -> list[Path]:
        """
        Download raw data files.

        Args:
            force: Force re-download even if files exist

        Returns:
            List of paths to downloaded files
        """
        pass

    @abstractmethod
    def build_observations(self) -> MatrixData:
        """
        Parse raw data into matrix observations.

        Returns:
            MatrixData object with all observations
        """
        pass

    @abstractmethod
    def cross_checks(self) -> list[CrossCheckResult]:
        """
        Run dataset-specific validation checks.

        Returns:
            List of CrossCheckResult objects
        """
        pass

    def get_matrix_data(self, force_rebuild: bool = False) -> MatrixData:
        """
        Get matrix data, building from raw if needed.

        Args:
            force_rebuild: Force rebuilding from raw data

        Returns:
            MatrixData object
        """
        if self._matrix_data is not None and not force_rebuild:
            return self._matrix_data

        # Check for cached processed data
        processed_path = self.processed_dir / f"{self.name}.parquet"
        if not force_rebuild and processed_path.exists():
            logger.info(f"Loading processed data from {processed_path}")
            self._matrix_data = MatrixData.load(processed_path)
            return self._matrix_data

        # Build from raw data
        logger.info(f"Building {self.name} matrix data from raw files")
        self._matrix_data = self.build_observations()

        # Save processed data
        self._matrix_data.save(processed_path)
        logger.info(f"Saved processed data to {processed_path}")

        return self._matrix_data

    def validate(self) -> bool:
        """
        Run all cross-checks and return overall pass/fail.

        Returns:
            True if all checks pass
        """
        results = self.cross_checks()
        all_passed = all(r.passed for r in results)

        for r in results:
            status = "PASS" if r.passed else "FAIL"
            logger.info(f"  [{status}] {r.name}: {r.message}")

        return all_passed

    def summary(self) -> dict[str, Any]:
        """Get summary statistics for the dataset."""
        data = self.get_matrix_data()

        values, errors, mask = data.to_matrix()

        return {
            "name": self.name,
            "description": self.description,
            "n_rows": data.n_rows,
            "n_cols": data.n_cols,
            "n_observations": data.n_obs,
            "fill_fraction": mask.sum() / mask.size,
            "value_range": [float(np.nanmin(values)), float(np.nanmax(values))],
            "mean_rel_error": float(np.nanmean(errors / np.abs(values))),
            "has_covariance": data.covariance is not None,
            "source_dois": self.source_dois,
        }
