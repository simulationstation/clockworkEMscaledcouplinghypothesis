"""
Caching utilities with TTL support and disk persistence.
"""

import hashlib
import json
import pickle
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar, ParamSpec

from rank1.logging import get_logger

logger = get_logger()

P = ParamSpec("P")
T = TypeVar("T")


class Cache:
    """
    Disk-based cache with TTL support.

    Stores cached values as pickle files with metadata for expiration.
    """

    def __init__(
        self,
        cache_dir: Path,
        default_ttl_hours: int = 24 * 7,
    ):
        """
        Initialize cache.

        Args:
            cache_dir: Directory for cache files
            default_ttl_hours: Default time-to-live in hours
        """
        self.cache_dir = Path(cache_dir)
        self.default_ttl_hours = default_ttl_hours
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key_to_path(self, key: str) -> Path:
        """Convert cache key to file path."""
        # Use hash for filesystem-safe names
        key_hash = hashlib.sha256(key.encode()).hexdigest()[:32]
        return self.cache_dir / f"{key_hash}.pkl"

    def _meta_path(self, data_path: Path) -> Path:
        """Get metadata path for a data file."""
        return data_path.with_suffix(".meta")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get value from cache.

        Args:
            key: Cache key
            default: Default value if not found or expired

        Returns:
            Cached value or default
        """
        path = self._key_to_path(key)
        meta_path = self._meta_path(path)

        if not path.exists() or not meta_path.exists():
            return default

        # Check expiration
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)

            if time.time() > meta.get("expires", 0):
                logger.debug(f"Cache expired: {key}")
                self.delete(key)
                return default

            # Load and return data
            with open(path, "rb") as f:
                return pickle.load(f)

        except Exception as e:
            logger.warning(f"Cache read error for {key}: {e}")
            return default

    def set(
        self,
        key: str,
        value: Any,
        ttl_hours: Optional[int] = None,
    ) -> None:
        """
        Store value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl_hours: Time-to-live in hours (uses default if not specified)
        """
        if ttl_hours is None:
            ttl_hours = self.default_ttl_hours

        path = self._key_to_path(key)
        meta_path = self._meta_path(path)

        try:
            # Write data
            with open(path, "wb") as f:
                pickle.dump(value, f)

            # Write metadata
            meta = {
                "key": key,
                "created": time.time(),
                "expires": time.time() + (ttl_hours * 3600),
            }
            with open(meta_path, "w") as f:
                json.dump(meta, f)

            logger.debug(f"Cached: {key}")

        except Exception as e:
            logger.warning(f"Cache write error for {key}: {e}")

    def delete(self, key: str) -> bool:
        """
        Delete a cache entry.

        Args:
            key: Cache key

        Returns:
            True if deleted, False if not found
        """
        path = self._key_to_path(key)
        meta_path = self._meta_path(path)

        deleted = False
        if path.exists():
            path.unlink()
            deleted = True
        if meta_path.exists():
            meta_path.unlink()
            deleted = True

        return deleted

    def clear(self) -> int:
        """
        Clear all cache entries.

        Returns:
            Number of entries deleted
        """
        count = 0
        for path in self.cache_dir.glob("*.pkl"):
            path.unlink()
            count += 1
            meta = self._meta_path(path)
            if meta.exists():
                meta.unlink()

        logger.info(f"Cleared {count} cache entries")
        return count

    def cleanup_expired(self) -> int:
        """
        Remove expired cache entries.

        Returns:
            Number of entries removed
        """
        count = 0
        now = time.time()

        for meta_path in self.cache_dir.glob("*.meta"):
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)

                if now > meta.get("expires", 0):
                    data_path = meta_path.with_suffix(".pkl")
                    if data_path.exists():
                        data_path.unlink()
                    meta_path.unlink()
                    count += 1

            except Exception:
                pass

        if count > 0:
            logger.info(f"Cleaned up {count} expired cache entries")

        return count

    def contains(self, key: str) -> bool:
        """Check if key exists and is not expired."""
        return self.get(key) is not None


def cached(
    cache_dir: Optional[Path] = None,
    ttl_hours: int = 24 * 7,
    key_fn: Optional[Callable[..., str]] = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator to cache function results.

    Args:
        cache_dir: Cache directory (uses temp if not specified)
        ttl_hours: Time-to-live in hours
        key_fn: Function to generate cache key from arguments

    Returns:
        Decorator function
    """
    if cache_dir is None:
        cache_dir = Path("data/cache")

    cache = Cache(cache_dir, ttl_hours)

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            # Generate cache key
            if key_fn:
                key = key_fn(*args, **kwargs)
            else:
                key = f"{func.__module__}.{func.__name__}:{args}:{kwargs}"

            # Check cache
            result = cache.get(key)
            if result is not None:
                return result

            # Call function and cache result
            result = func(*args, **kwargs)
            cache.set(key, result, ttl_hours)
            return result

        return wrapper

    return decorator
