"""
Data provenance tracking for reproducibility and auditability.

Tracks:
- Data origin (API, manual table, placeholder, cache)
- Source identifiers (DOI, arXiv ID, HEPData record ID)
- Table names used from HEPData
- File hashes of raw inputs
- Selection filters applied
- Fallback reasons (if any)
- Timestamps
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional
import hashlib
import json


class DataOrigin(Enum):
    """Source of data."""
    API = "api"                    # Fetched from HEPData/CERN Open Data API
    MANUAL_TABLE = "manual_table"  # From curated manual tables
    ARXIV_PDF = "arxiv_pdf"        # Extracted from arXiv PDF
    ROOT_FILE = "root_file"        # Processed from ROOT files
    PLACEHOLDER = "placeholder"    # Synthetic placeholder data
    CACHE = "cache"                # Loaded from processed cache
    FALLBACK = "fallback"          # Used fallback extraction method


@dataclass
class SourceReference:
    """Reference to a data source."""
    source_type: str  # "hepdata", "cern_opendata", "arxiv", "manual"
    identifier: str   # Record ID, arXiv ID, etc.
    doi: Optional[str] = None
    url: Optional[str] = None
    table_names: list[str] = field(default_factory=list)
    version: Optional[str] = None


@dataclass
class FileHash:
    """Hash of a file used in processing."""
    path: str
    sha256: str
    size_bytes: int
    modified_time: Optional[str] = None


@dataclass
class SelectionFilter:
    """A filter applied to the data."""
    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)
    n_before: Optional[int] = None
    n_after: Optional[int] = None


@dataclass
class FallbackEvent:
    """Record of a fallback that occurred during data loading."""
    step: str
    reason: str
    fallback_to: str
    original_source: Optional[str] = None
    warning_logged: bool = True


@dataclass
class DataProvenance:
    """
    Complete provenance record for a dataset.

    This tracks everything needed to reproduce and audit
    the data loading process.
    """
    # Origin tracking
    origin: DataOrigin
    origin_details: str = ""

    # Source references
    sources: list[SourceReference] = field(default_factory=list)

    # File hashes
    input_files: list[FileHash] = field(default_factory=list)

    # Processing steps
    selection_filters: list[SelectionFilter] = field(default_factory=list)

    # Fallbacks
    fallbacks: list[FallbackEvent] = field(default_factory=list)

    # Timestamps
    fetch_timestamp: Optional[str] = None
    build_timestamp: Optional[str] = None

    # Software version
    package_version: str = "0.1.0"

    # Random seed used (if any)
    random_seed: Optional[int] = None

    # Additional metadata
    extra: dict[str, Any] = field(default_factory=dict)

    def add_source(
        self,
        source_type: str,
        identifier: str,
        doi: Optional[str] = None,
        url: Optional[str] = None,
        table_names: Optional[list[str]] = None,
    ) -> None:
        """Add a source reference."""
        self.sources.append(SourceReference(
            source_type=source_type,
            identifier=identifier,
            doi=doi,
            url=url,
            table_names=table_names or [],
        ))

    def add_file_hash(self, path: Path) -> None:
        """Compute and store hash of a file."""
        if not path.exists():
            return

        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)

        stat = path.stat()
        self.input_files.append(FileHash(
            path=str(path),
            sha256=sha256.hexdigest(),
            size_bytes=stat.st_size,
            modified_time=datetime.fromtimestamp(stat.st_mtime).isoformat(),
        ))

    def add_filter(
        self,
        name: str,
        description: str,
        parameters: Optional[dict[str, Any]] = None,
        n_before: Optional[int] = None,
        n_after: Optional[int] = None,
    ) -> None:
        """Record a selection filter applied to data."""
        self.selection_filters.append(SelectionFilter(
            name=name,
            description=description,
            parameters=parameters or {},
            n_before=n_before,
            n_after=n_after,
        ))

    def record_fallback(
        self,
        step: str,
        reason: str,
        fallback_to: str,
        original_source: Optional[str] = None,
    ) -> None:
        """Record that a fallback occurred."""
        self.fallbacks.append(FallbackEvent(
            step=step,
            reason=reason,
            fallback_to=fallback_to,
            original_source=original_source,
        ))

    def set_fetch_timestamp(self) -> None:
        """Set the fetch timestamp to now."""
        self.fetch_timestamp = datetime.utcnow().isoformat() + "Z"

    def set_build_timestamp(self) -> None:
        """Set the build timestamp to now."""
        self.build_timestamp = datetime.utcnow().isoformat() + "Z"

    @property
    def has_fallbacks(self) -> bool:
        """Check if any fallbacks occurred."""
        return len(self.fallbacks) > 0

    @property
    def is_placeholder(self) -> bool:
        """Check if data is placeholder/synthetic."""
        return self.origin == DataOrigin.PLACEHOLDER

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "origin": self.origin.value,
            "origin_details": self.origin_details,
            "sources": [
                {
                    "source_type": s.source_type,
                    "identifier": s.identifier,
                    "doi": s.doi,
                    "url": s.url,
                    "table_names": s.table_names,
                    "version": s.version,
                }
                for s in self.sources
            ],
            "input_files": [
                {
                    "path": f.path,
                    "sha256": f.sha256,
                    "size_bytes": f.size_bytes,
                    "modified_time": f.modified_time,
                }
                for f in self.input_files
            ],
            "selection_filters": [
                {
                    "name": f.name,
                    "description": f.description,
                    "parameters": f.parameters,
                    "n_before": f.n_before,
                    "n_after": f.n_after,
                }
                for f in self.selection_filters
            ],
            "fallbacks": [
                {
                    "step": f.step,
                    "reason": f.reason,
                    "fallback_to": f.fallback_to,
                    "original_source": f.original_source,
                    "warning_logged": f.warning_logged,
                }
                for f in self.fallbacks
            ],
            "fetch_timestamp": self.fetch_timestamp,
            "build_timestamp": self.build_timestamp,
            "package_version": self.package_version,
            "random_seed": self.random_seed,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DataProvenance":
        """Create from dictionary."""
        prov = cls(
            origin=DataOrigin(data.get("origin", "cache")),
            origin_details=data.get("origin_details", ""),
        )

        for s in data.get("sources", []):
            prov.sources.append(SourceReference(
                source_type=s["source_type"],
                identifier=s["identifier"],
                doi=s.get("doi"),
                url=s.get("url"),
                table_names=s.get("table_names", []),
                version=s.get("version"),
            ))

        for f in data.get("input_files", []):
            prov.input_files.append(FileHash(
                path=f["path"],
                sha256=f["sha256"],
                size_bytes=f["size_bytes"],
                modified_time=f.get("modified_time"),
            ))

        for f in data.get("selection_filters", []):
            prov.selection_filters.append(SelectionFilter(
                name=f["name"],
                description=f["description"],
                parameters=f.get("parameters", {}),
                n_before=f.get("n_before"),
                n_after=f.get("n_after"),
            ))

        for f in data.get("fallbacks", []):
            prov.fallbacks.append(FallbackEvent(
                step=f["step"],
                reason=f["reason"],
                fallback_to=f["fallback_to"],
                original_source=f.get("original_source"),
                warning_logged=f.get("warning_logged", True),
            ))

        prov.fetch_timestamp = data.get("fetch_timestamp")
        prov.build_timestamp = data.get("build_timestamp")
        prov.package_version = data.get("package_version", "0.1.0")
        prov.random_seed = data.get("random_seed")
        prov.extra = data.get("extra", {})

        return prov

    def save(self, path: Path) -> None:
        """Save provenance to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "DataProvenance":
        """Load provenance from JSON file."""
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Data Origin: {self.origin.value}",
            f"  Details: {self.origin_details}" if self.origin_details else "",
        ]

        if self.sources:
            lines.append("Sources:")
            for s in self.sources:
                ref = f"  - {s.source_type}: {s.identifier}"
                if s.doi:
                    ref += f" (DOI: {s.doi})"
                lines.append(ref)
                if s.table_names:
                    lines.append(f"    Tables: {', '.join(s.table_names)}")

        if self.input_files:
            lines.append(f"Input files: {len(self.input_files)}")
            for f in self.input_files[:3]:
                lines.append(f"  - {Path(f.path).name}: {f.sha256[:16]}...")
            if len(self.input_files) > 3:
                lines.append(f"  ... and {len(self.input_files) - 3} more")

        if self.selection_filters:
            lines.append(f"Filters applied: {len(self.selection_filters)}")
            for f in self.selection_filters:
                detail = f"  - {f.name}"
                if f.n_before and f.n_after:
                    detail += f" ({f.n_before} -> {f.n_after})"
                lines.append(detail)

        if self.fallbacks:
            lines.append(f"FALLBACKS: {len(self.fallbacks)}")
            for f in self.fallbacks:
                lines.append(f"  - {f.step}: {f.reason} -> {f.fallback_to}")

        if self.fetch_timestamp:
            lines.append(f"Fetched: {self.fetch_timestamp}")
        if self.build_timestamp:
            lines.append(f"Built: {self.build_timestamp}")

        return "\n".join(line for line in lines if line)


def create_provenance(
    origin: DataOrigin,
    origin_details: str = "",
) -> DataProvenance:
    """Factory function to create a new provenance record."""
    return DataProvenance(
        origin=origin,
        origin_details=origin_details,
    )
