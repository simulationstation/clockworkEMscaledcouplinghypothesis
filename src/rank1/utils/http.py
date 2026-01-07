"""
HTTP client utilities with retries, caching, and progress tracking.
"""

import hashlib
import time
from pathlib import Path
from typing import Optional, Any
from urllib.parse import urlparse, urljoin

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm

from rank1.logging import get_logger

logger = get_logger()

# Default timeout for requests (connect, read)
DEFAULT_TIMEOUT = (10, 60)

# Retry configuration
DEFAULT_RETRIES = 3
DEFAULT_BACKOFF = 0.5


class HTTPClient:
    """
    HTTP client with automatic retries, caching, and progress tracking.
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        timeout: tuple[int, int] = DEFAULT_TIMEOUT,
        retries: int = DEFAULT_RETRIES,
        backoff_factor: float = DEFAULT_BACKOFF,
    ):
        self.cache_dir = cache_dir
        self.timeout = timeout

        # Set up session with retries
        self.session = requests.Session()

        retry_strategy = Retry(
            total=retries,
            backoff_factor=backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        # Set user agent
        self.session.headers.update({
            "User-Agent": "rank1-factorization/1.0 (research tool; +https://github.com/research/rank1)"
        })

    def get(
        self,
        url: str,
        params: Optional[dict] = None,
        headers: Optional[dict] = None,
        use_cache: bool = True,
        cache_key: Optional[str] = None,
    ) -> requests.Response:
        """
        GET request with optional caching.

        Args:
            url: URL to fetch
            params: Query parameters
            headers: Additional headers
            use_cache: Whether to use HTTP caching
            cache_key: Custom cache key (defaults to URL hash)

        Returns:
            Response object
        """
        if use_cache and self.cache_dir:
            cached = self._get_cached(url, params, cache_key)
            if cached:
                return cached

        response = self.session.get(
            url,
            params=params,
            headers=headers,
            timeout=self.timeout,
        )
        response.raise_for_status()

        if use_cache and self.cache_dir:
            self._cache_response(url, params, cache_key, response)

        return response

    def get_json(
        self,
        url: str,
        params: Optional[dict] = None,
        **kwargs: Any,
    ) -> dict:
        """GET request returning JSON."""
        response = self.get(url, params=params, **kwargs)
        return response.json()

    def download(
        self,
        url: str,
        dest: Path,
        expected_size: Optional[int] = None,
        expected_checksum: Optional[str] = None,
        checksum_algo: str = "sha256",
        show_progress: bool = True,
        force: bool = False,
    ) -> Path:
        """
        Download a file with progress bar and optional checksum verification.

        Args:
            url: URL to download
            dest: Destination path
            expected_size: Expected file size in bytes
            expected_checksum: Expected checksum string
            checksum_algo: Hash algorithm for checksum
            show_progress: Show progress bar
            force: Force re-download even if file exists

        Returns:
            Path to downloaded file
        """
        # Check if already downloaded and valid
        if not force and dest.exists():
            if expected_checksum:
                actual = _compute_file_hash(dest, checksum_algo)
                if actual == expected_checksum:
                    logger.debug(f"Using cached file: {dest}")
                    return dest
                else:
                    logger.warning(f"Checksum mismatch for {dest}, re-downloading")
            elif expected_size:
                if dest.stat().st_size == expected_size:
                    logger.debug(f"Using cached file: {dest}")
                    return dest
            else:
                logger.debug(f"Using cached file: {dest}")
                return dest

        # Create destination directory
        dest.parent.mkdir(parents=True, exist_ok=True)

        # Stream download with progress
        logger.info(f"Downloading: {url}")
        # CERN eospublic uses self-signed certificates
        verify_ssl = True
        if "eospublic.cern.ch" in url or "eostotem.cern.ch" in url:
            verify_ssl = False
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        response = self.session.get(url, stream=True, timeout=self.timeout, verify=verify_ssl)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0)) or expected_size or 0

        hasher = hashlib.new(checksum_algo) if expected_checksum else None

        with open(dest, "wb") as f:
            if show_progress and total_size > 0:
                pbar = tqdm(
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    desc=dest.name,
                )
            else:
                pbar = None

            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                if hasher:
                    hasher.update(chunk)
                if pbar:
                    pbar.update(len(chunk))

            if pbar:
                pbar.close()

        # Verify checksum
        if expected_checksum and hasher:
            actual = hasher.hexdigest()
            if actual != expected_checksum:
                dest.unlink()
                raise ValueError(
                    f"Checksum mismatch for {dest}: expected {expected_checksum}, got {actual}"
                )

        logger.info(f"Downloaded: {dest} ({dest.stat().st_size:,} bytes)")
        return dest

    def _cache_key(self, url: str, params: Optional[dict], custom_key: Optional[str]) -> str:
        """Generate cache key for a request."""
        if custom_key:
            return custom_key

        key_str = url
        if params:
            key_str += str(sorted(params.items()))

        return hashlib.sha256(key_str.encode()).hexdigest()[:16]

    def _get_cached(
        self,
        url: str,
        params: Optional[dict],
        custom_key: Optional[str],
    ) -> Optional[requests.Response]:
        """Try to get cached response."""
        if not self.cache_dir:
            return None

        cache_path = self.cache_dir / f"{self._cache_key(url, params, custom_key)}.cache"
        if not cache_path.exists():
            return None

        try:
            content = cache_path.read_bytes()
            response = requests.Response()
            response._content = content
            response.status_code = 200
            logger.debug(f"Cache hit: {url}")
            return response
        except Exception as e:
            logger.warning(f"Cache read failed: {e}")
            return None

    def _cache_response(
        self,
        url: str,
        params: Optional[dict],
        custom_key: Optional[str],
        response: requests.Response,
    ) -> None:
        """Cache a response."""
        if not self.cache_dir:
            return

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self.cache_dir / f"{self._cache_key(url, params, custom_key)}.cache"

        try:
            cache_path.write_bytes(response.content)
            logger.debug(f"Cached: {url}")
        except Exception as e:
            logger.warning(f"Cache write failed: {e}")


def _compute_file_hash(path: Path, algorithm: str = "sha256") -> str:
    """Compute hash of a file."""
    hasher = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def download_file(
    url: str,
    dest: Path,
    expected_checksum: Optional[str] = None,
    show_progress: bool = True,
    force: bool = False,
) -> Path:
    """
    Convenience function to download a file.

    Args:
        url: URL to download
        dest: Destination path
        expected_checksum: Expected SHA256 checksum
        show_progress: Show progress bar
        force: Force re-download

    Returns:
        Path to downloaded file
    """
    client = HTTPClient()
    return client.download(
        url,
        dest,
        expected_checksum=expected_checksum,
        show_progress=show_progress,
        force=force,
    )
