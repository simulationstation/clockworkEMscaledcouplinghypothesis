"""
Checksum utilities for data integrity verification.
"""

import hashlib
from pathlib import Path
from typing import Optional

from rank1.logging import get_logger

logger = get_logger()

# Supported hash algorithms
SUPPORTED_ALGORITHMS = {"sha256", "sha1", "md5", "sha512"}


def compute_checksum(
    path: Path,
    algorithm: str = "sha256",
    chunk_size: int = 65536,
) -> str:
    """
    Compute checksum of a file.

    Args:
        path: Path to file
        algorithm: Hash algorithm (sha256, sha1, md5, sha512)
        chunk_size: Read buffer size

    Returns:
        Hex-encoded hash string
    """
    if algorithm not in SUPPORTED_ALGORITHMS:
        raise ValueError(f"Unsupported algorithm: {algorithm}. Use one of {SUPPORTED_ALGORITHMS}")

    hasher = hashlib.new(algorithm)

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hasher.update(chunk)

    return hasher.hexdigest()


def verify_checksum(
    path: Path,
    expected: str,
    algorithm: str = "sha256",
) -> bool:
    """
    Verify file checksum matches expected value.

    Args:
        path: Path to file
        expected: Expected checksum value
        algorithm: Hash algorithm

    Returns:
        True if checksum matches
    """
    if not path.exists():
        logger.warning(f"File not found for checksum: {path}")
        return False

    actual = compute_checksum(path, algorithm)
    matches = actual.lower() == expected.lower()

    if not matches:
        logger.warning(f"Checksum mismatch for {path}")
        logger.warning(f"  Expected: {expected}")
        logger.warning(f"  Actual:   {actual}")

    return matches


def compute_checksum_string(data: bytes, algorithm: str = "sha256") -> str:
    """
    Compute checksum of bytes data.

    Args:
        data: Bytes to hash
        algorithm: Hash algorithm

    Returns:
        Hex-encoded hash string
    """
    hasher = hashlib.new(algorithm)
    hasher.update(data)
    return hasher.hexdigest()


def compute_content_hash(content: str, algorithm: str = "sha256") -> str:
    """
    Compute hash of string content.

    Args:
        content: String content
        algorithm: Hash algorithm

    Returns:
        Hex-encoded hash string
    """
    return compute_checksum_string(content.encode("utf-8"), algorithm)


class ChecksumRegistry:
    """
    Registry for expected checksums of known files.

    Used for validating downloaded data integrity.
    """

    def __init__(self):
        self._checksums: dict[str, dict[str, str]] = {}

    def register(
        self,
        file_id: str,
        checksum: str,
        algorithm: str = "sha256",
    ) -> None:
        """Register expected checksum for a file."""
        self._checksums[file_id] = {
            "checksum": checksum,
            "algorithm": algorithm,
        }

    def verify(self, file_id: str, path: Path) -> Optional[bool]:
        """
        Verify file against registered checksum.

        Returns:
            True if matches, False if mismatch, None if no checksum registered
        """
        if file_id not in self._checksums:
            logger.debug(f"No registered checksum for {file_id}")
            return None

        info = self._checksums[file_id]
        return verify_checksum(
            path,
            info["checksum"],
            info["algorithm"],
        )

    def get(self, file_id: str) -> Optional[tuple[str, str]]:
        """Get registered checksum info."""
        if file_id not in self._checksums:
            return None
        info = self._checksums[file_id]
        return (info["checksum"], info["algorithm"])


# Global checksum registry
checksum_registry = ChecksumRegistry()
