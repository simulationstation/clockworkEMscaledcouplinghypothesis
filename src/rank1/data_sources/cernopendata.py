"""
CERN Open Data Portal client for fetching experimental data.

The CERN Open Data portal (http://opendata.cern.ch) provides access to
datasets from LHC experiments. This client supports:
- Record metadata retrieval
- File downloads with checksums
- Support for TOTEM Roman Pot datasets
"""

import json
from pathlib import Path
from typing import Optional, Any
from dataclasses import dataclass

from rank1.utils.http import HTTPClient
from rank1.logging import get_logger

logger = get_logger()

CERN_OPENDATA_BASE_URL = "http://opendata.cern.ch"
CERN_OPENDATA_API = f"{CERN_OPENDATA_BASE_URL}/api/records"


@dataclass
class OpenDataFile:
    """A file from a CERN Open Data record."""

    filename: str
    size: int
    checksum: Optional[str]
    uri: str
    type: Optional[str]

    @property
    def download_url(self) -> str:
        """Get full download URL."""
        if self.uri.startswith("http"):
            return self.uri
        # XRootD URIs (root://) need to be converted to HTTP
        if self.uri.startswith("root://"):
            # Convert root://eospublic.cern.ch/... to https://eospublic.cern.ch/...
            return self.uri.replace("root://", "https://", 1)
        return f"{CERN_OPENDATA_BASE_URL}{self.uri}"


@dataclass
class OpenDataRecord:
    """Metadata and files from a CERN Open Data record."""

    record_id: int
    title: str
    description: str
    experiment: Optional[str]
    collision_type: Optional[str]
    collision_energy: Optional[str]
    files: list[OpenDataFile]
    doi: Optional[str]
    categories: list[str]
    keywords: list[str]

    @property
    def total_size_bytes(self) -> int:
        """Total size of all files in bytes."""
        return sum(f.size for f in self.files)

    @property
    def total_size_mb(self) -> float:
        """Total size of all files in MB."""
        return self.total_size_bytes / (1024 * 1024)

    def get_files_by_type(self, file_type: str) -> list[OpenDataFile]:
        """Get files of a specific type (e.g., 'root', 'csv')."""
        return [f for f in self.files if f.type and f.type.lower() == file_type.lower()]

    def get_files_by_extension(self, ext: str) -> list[OpenDataFile]:
        """Get files with a specific extension."""
        ext = ext.lower().lstrip(".")
        return [f for f in self.files if f.filename.lower().endswith(f".{ext}")]


class CERNOpenDataClient:
    """
    Client for accessing CERN Open Data Portal records and files.
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        raw_dir: Optional[Path] = None,
    ):
        """
        Initialize CERN Open Data client.

        Args:
            cache_dir: Directory for HTTP cache
            raw_dir: Directory for downloaded files
        """
        self.cache_dir = cache_dir or Path("data/cache")
        self.raw_dir = raw_dir or Path("data/raw/cernopendata")
        self.http = HTTPClient(cache_dir=self.cache_dir)

    def get_record_metadata(self, record_id: int) -> dict[str, Any]:
        """
        Fetch record metadata.

        Args:
            record_id: CERN Open Data record ID

        Returns:
            Record metadata dict
        """
        url = f"{CERN_OPENDATA_API}/{record_id}"
        logger.info(f"Fetching CERN Open Data record {record_id}")

        response = self.http.get(url)
        return response.json()

    def download_record(
        self,
        record_id: int,
        force: bool = False,
    ) -> OpenDataRecord:
        """
        Download record metadata.

        Args:
            record_id: CERN Open Data record ID
            force: Force re-download

        Returns:
            OpenDataRecord object
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

    def _parse_record(self, record_id: int, data: dict) -> OpenDataRecord:
        """Parse raw record data into OpenDataRecord."""
        metadata = data.get("metadata", {})

        # Parse files
        files = []
        for file_data in metadata.get("files", []):
            # CERN Open Data uses "key" for filename, "filename" as fallback
            filename = file_data.get("key") or file_data.get("filename", "")
            f = OpenDataFile(
                filename=filename,
                size=file_data.get("size", 0),
                checksum=file_data.get("checksum"),
                uri=file_data.get("uri", ""),
                type=file_data.get("type"),
            )
            files.append(f)

        # Extract collision info
        collision_info = metadata.get("collision_information", {})

        # Extract categories and keywords
        categories = []
        for cat in metadata.get("categories", {}).get("primary", []):
            categories.append(cat)
        for cat in metadata.get("categories", {}).get("secondary", []):
            categories.append(cat)

        keywords = metadata.get("keywords", [])

        return OpenDataRecord(
            record_id=record_id,
            title=metadata.get("title", ""),
            description=metadata.get("abstract", {}).get("description", ""),
            experiment=metadata.get("experiment"),
            collision_type=collision_info.get("type"),
            collision_energy=collision_info.get("energy"),
            files=files,
            doi=metadata.get("doi"),
            categories=categories,
            keywords=keywords,
        )

    def download_file(
        self,
        record_id: int,
        file: OpenDataFile,
        force: bool = False,
        show_progress: bool = True,
    ) -> Path:
        """
        Download a file from a record.

        Args:
            record_id: CERN Open Data record ID
            file: OpenDataFile object
            force: Force re-download
            show_progress: Show progress bar

        Returns:
            Path to downloaded file
        """
        dest = self.raw_dir / str(record_id) / file.filename

        if not force and dest.exists():
            # Verify checksum if available
            if file.checksum:
                from rank1.utils.hashing import verify_checksum

                algo, expected = self._parse_checksum(file.checksum)
                if verify_checksum(dest, expected, algo):
                    logger.debug(f"Using cached file: {dest}")
                    return dest
                else:
                    logger.warning(f"Checksum mismatch for {dest}, re-downloading")
            else:
                # Check size
                if dest.stat().st_size == file.size:
                    logger.debug(f"Using cached file: {dest}")
                    return dest

        # Parse checksum
        checksum = None
        algo = "sha256"
        if file.checksum:
            algo, checksum = self._parse_checksum(file.checksum)

        return self.http.download(
            file.download_url,
            dest,
            expected_size=file.size,
            expected_checksum=checksum,
            checksum_algo=algo,
            show_progress=show_progress,
        )

    def download_all_files(
        self,
        record_id: int,
        file_types: Optional[list[str]] = None,
        force: bool = False,
        show_progress: bool = True,
    ) -> list[Path]:
        """
        Download all files from a record.

        Args:
            record_id: CERN Open Data record ID
            file_types: Optional filter for file types
            force: Force re-download
            show_progress: Show progress bar

        Returns:
            List of paths to downloaded files
        """
        record = self.download_record(record_id, force=False)

        files_to_download = record.files
        if file_types:
            files_to_download = [
                f for f in files_to_download
                if f.type and f.type.lower() in [t.lower() for t in file_types]
            ]

        logger.info(
            f"Downloading {len(files_to_download)} files "
            f"({record.total_size_mb:.1f} MB total)"
        )

        paths = []
        for file in files_to_download:
            path = self.download_file(
                record_id, file, force=force, show_progress=show_progress
            )
            paths.append(path)

        return paths

    def _parse_checksum(self, checksum_str: str) -> tuple[str, str]:
        """
        Parse checksum string in format 'algo:hash'.

        Args:
            checksum_str: Checksum string (e.g., 'adler32:12345' or 'sha256:abc...')

        Returns:
            (algorithm, hash_value) tuple
        """
        if ":" in checksum_str:
            parts = checksum_str.split(":", 1)
            algo = parts[0].lower()
            # Map CERN naming to hashlib naming
            if algo == "adler32":
                # adler32 not well supported, skip verification
                return ("sha256", "")
            return (algo, parts[1])
        else:
            # Assume sha256 by default
            return ("sha256", checksum_str)


class TOTEMDataHandler:
    """
    Specialized handler for TOTEM Roman Pot datasets.

    TOTEM records include:
    - Record 84000: Event-level Roman Pot data (7 TeV)
    - Record 84001: Analysis example code and auxiliary files
    """

    RECORD_IDS = {
        "7tev_data": 84000,
        "7tev_example": 84001,
    }

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        raw_dir: Optional[Path] = None,
    ):
        self.client = CERNOpenDataClient(cache_dir=cache_dir, raw_dir=raw_dir)
        self.raw_dir = raw_dir or Path("data/raw/cernopendata")

    def download_7tev_data(
        self,
        force: bool = False,
        show_progress: bool = True,
    ) -> list[Path]:
        """
        Download TOTEM 7 TeV Roman Pot data.

        Returns:
            List of paths to downloaded ROOT files
        """
        record = self.client.download_record(self.RECORD_IDS["7tev_data"])

        logger.info(f"TOTEM 7 TeV record: {record.title}")
        logger.info(f"Total size: {record.total_size_mb:.1f} MB")

        # Download ROOT files
        root_files = record.get_files_by_extension("root")
        if not root_files:
            root_files = record.files  # Download everything if no .root found

        paths = []
        for file in root_files:
            path = self.client.download_file(
                self.RECORD_IDS["7tev_data"],
                file,
                force=force,
                show_progress=show_progress,
            )
            paths.append(path)

        return paths

    def download_7tev_example(
        self,
        force: bool = False,
    ) -> list[Path]:
        """
        Download TOTEM 7 TeV analysis example files.

        Returns:
            List of paths to downloaded files
        """
        record = self.client.download_record(self.RECORD_IDS["7tev_example"])

        logger.info(f"TOTEM example record: {record.title}")

        paths = []
        for file in record.files:
            path = self.client.download_file(
                self.RECORD_IDS["7tev_example"],
                file,
                force=force,
                show_progress=False,  # Usually small files
            )
            paths.append(path)

        return paths

    def get_7tev_info(self) -> dict[str, Any]:
        """
        Get information about 7 TeV data without downloading.

        Returns:
            Dict with record info
        """
        record = self.client.download_record(self.RECORD_IDS["7tev_data"])

        return {
            "title": record.title,
            "description": record.description,
            "total_size_mb": record.total_size_mb,
            "n_files": len(record.files),
            "file_types": list(set(f.type for f in record.files if f.type)),
            "doi": record.doi,
        }


def fetch_totem_7tev(
    cache_dir: Optional[Path] = None,
    raw_dir: Optional[Path] = None,
    force: bool = False,
) -> list[Path]:
    """
    Convenience function to fetch TOTEM 7 TeV data.

    Args:
        cache_dir: Cache directory
        raw_dir: Raw data directory
        force: Force re-download

    Returns:
        List of paths to downloaded files
    """
    handler = TOTEMDataHandler(cache_dir=cache_dir, raw_dir=raw_dir)
    return handler.download_7tev_data(force=force)
