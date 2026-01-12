"""TTL-based caching for API clients."""

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class CacheEntry:
    """Single cache entry with TTL support."""
    data: Any
    timestamp: float
    ttl: int

    def is_valid(self) -> bool:
        """Check if cache entry is still valid."""
        return time.time() - self.timestamp < self.ttl


class TTLCache:
    """Simple TTL-based cache implementation."""

    def __init__(self, default_ttl: int = 300):
        """
        Initialize cache.

        Args:
            default_ttl: Default time-to-live in seconds (default 5 minutes)
        """
        self._cache: Dict[str, CacheEntry] = {}
        self.default_ttl = default_ttl

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache if valid.

        Args:
            key: Cache key

        Returns:
            Cached value if valid, None otherwise
        """
        if key in self._cache:
            entry = self._cache[key]
            if entry.is_valid():
                return entry.data
            else:
                # Remove expired entry
                del self._cache[key]
        return None

    def set(self, key: str, data: Any, ttl: Optional[int] = None) -> None:
        """
        Store value in cache.

        Args:
            key: Cache key
            data: Data to cache
            ttl: Optional custom TTL in seconds
        """
        self._cache[key] = CacheEntry(
            data=data,
            timestamp=time.time(),
            ttl=ttl if ttl is not None else self.default_ttl
        )

    def invalidate(self, key: str) -> bool:
        """
        Remove entry from cache.

        Args:
            key: Cache key

        Returns:
            True if entry was removed, False if not found
        """
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()

    def cleanup(self) -> int:
        """
        Remove all expired entries.

        Returns:
            Number of entries removed
        """
        expired_keys = [
            key for key, entry in self._cache.items()
            if not entry.is_valid()
        ]
        for key in expired_keys:
            del self._cache[key]
        return len(expired_keys)

    def __len__(self) -> int:
        """Return number of entries in cache."""
        return len(self._cache)

    def __contains__(self, key: str) -> bool:
        """Check if key exists and is valid."""
        return self.get(key) is not None
