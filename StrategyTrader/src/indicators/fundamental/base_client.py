"""Base API client with caching and rate limiting."""

import time
import logging
from abc import ABC
from typing import Any, Dict, Optional

import requests

from src.indicators.utils.cache import TTLCache


logger = logging.getLogger(__name__)


class APIError(Exception):
    """Base exception for API errors."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class RateLimitError(APIError):
    """Rate limit exceeded error."""
    pass


class AuthenticationError(APIError):
    """Authentication/authorization error."""
    pass


class BaseAPIClient(ABC):
    """
    Base class for API clients with caching and rate limiting.

    Features:
    - TTL-based caching to reduce API calls
    - Rate limiting to respect API limits
    - Automatic retries with exponential backoff
    - Error handling and logging
    """

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        cache_ttl: int = 300,
        rate_limit_per_min: int = 30,
        timeout: int = 30,
        max_retries: int = 3,
    ):
        """
        Initialize API client.

        Args:
            base_url: Base URL for the API
            api_key: Optional API key for authenticated requests
            cache_ttl: Cache time-to-live in seconds (default 5 minutes)
            rate_limit_per_min: Maximum requests per minute (default 30)
            timeout: Request timeout in seconds (default 30)
            max_retries: Maximum retries on failure (default 3)
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.cache_ttl = cache_ttl
        self.rate_limit_per_min = rate_limit_per_min
        self.timeout = timeout
        self.max_retries = max_retries

        self._cache = TTLCache(default_ttl=cache_ttl)
        self._last_request_time = 0.0
        self._min_request_interval = 60.0 / rate_limit_per_min

        self._session = requests.Session()
        self._setup_session()

    def _setup_session(self) -> None:
        """Configure session with default headers."""
        self._session.headers.update({
            "Accept": "application/json",
            "User-Agent": "StrategyTrader/1.0",
        })
        if self.api_key:
            self._add_auth_header()

    def _add_auth_header(self) -> None:
        """Add authentication header. Override in subclasses if needed."""
        self._session.headers["Authorization"] = f"Bearer {self.api_key}"

    def _rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            sleep_time = self._min_request_interval - elapsed
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f}s")
            time.sleep(sleep_time)
        self._last_request_time = time.time()

    def _get_cached(self, cache_key: str) -> Optional[Any]:
        """Get value from cache."""
        return self._cache.get(cache_key)

    def _set_cache(self, cache_key: str, data: Any, ttl: Optional[int] = None) -> None:
        """Store value in cache."""
        self._cache.set(cache_key, data, ttl)

    def _build_url(self, endpoint: str) -> str:
        """Build full URL from endpoint."""
        return f"{self.base_url}/{endpoint.lstrip('/')}"

    def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        use_cache: bool = True,
        cache_ttl: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Make HTTP request with caching and rate limiting.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            params: Query parameters
            data: Request body data
            headers: Additional headers
            use_cache: Whether to use caching (GET only)
            cache_ttl: Custom cache TTL

        Returns:
            JSON response as dictionary

        Raises:
            APIError: On API error
            RateLimitError: On rate limit exceeded
            AuthenticationError: On auth failure
        """
        url = self._build_url(endpoint)

        # Check cache for GET requests
        if method.upper() == "GET" and use_cache:
            cache_key = f"{url}:{str(params)}"
            cached = self._get_cached(cache_key)
            if cached is not None:
                logger.debug(f"Cache hit: {cache_key}")
                return cached

        # Apply rate limiting
        self._rate_limit()

        # Make request with retries
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self._session.request(
                    method=method,
                    url=url,
                    params=params,
                    json=data,
                    headers=headers,
                    timeout=self.timeout,
                )

                # Handle HTTP errors
                if response.status_code == 429:
                    raise RateLimitError("Rate limit exceeded", 429)
                elif response.status_code in (401, 403):
                    raise AuthenticationError(
                        f"Authentication failed: {response.text}",
                        response.status_code
                    )
                elif response.status_code >= 400:
                    raise APIError(
                        f"API error: {response.text}",
                        response.status_code
                    )

                result = response.json()

                # Cache successful GET responses
                if method.upper() == "GET" and use_cache:
                    self._set_cache(cache_key, result, cache_ttl)

                return result

            except requests.exceptions.RequestException as e:
                last_error = APIError(f"Request failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    # Exponential backoff
                    sleep_time = (2 ** attempt) * 0.5
                    logger.warning(
                        f"Request failed, retrying in {sleep_time}s: {e}"
                    )
                    time.sleep(sleep_time)

        raise last_error or APIError("Request failed after retries")

    def get(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        use_cache: bool = True,
        cache_ttl: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Make GET request."""
        return self._request(
            "GET", endpoint, params=params,
            use_cache=use_cache, cache_ttl=cache_ttl
        )

    def post(
        self,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Make POST request."""
        return self._request("POST", endpoint, params=params, data=data, use_cache=False)

    def clear_cache(self) -> None:
        """Clear all cached data."""
        self._cache.clear()

    def close(self) -> None:
        """Close the session."""
        self._session.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False
