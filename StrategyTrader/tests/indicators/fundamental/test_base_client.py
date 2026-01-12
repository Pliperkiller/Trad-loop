"""Tests for base API client."""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock

from src.indicators.fundamental.base_client import (
    BaseAPIClient,
    APIError,
    RateLimitError,
    AuthenticationError,
)
from src.indicators.utils.cache import TTLCache


class ConcreteClient(BaseAPIClient):
    """Concrete implementation for testing."""
    pass


class TestBaseAPIClient:
    """Tests for BaseAPIClient."""

    def test_initialization(self):
        """Client initializes with correct defaults."""
        client = ConcreteClient(
            base_url="https://api.example.com",
            cache_ttl=300,
            rate_limit_per_min=30,
        )

        assert client.base_url == "https://api.example.com"
        assert client.cache_ttl == 300
        assert client.rate_limit_per_min == 30
        assert client.api_key is None

    def test_initialization_with_api_key(self):
        """Client initializes with API key."""
        client = ConcreteClient(
            base_url="https://api.example.com",
            api_key="test-api-key",
        )

        assert client.api_key == "test-api-key"

    def test_build_url(self):
        """URL building is correct."""
        client = ConcreteClient(base_url="https://api.example.com/v1/")

        assert client._build_url("endpoint") == "https://api.example.com/v1/endpoint"
        assert client._build_url("/endpoint") == "https://api.example.com/v1/endpoint"

    def test_url_trailing_slash_handling(self):
        """Trailing slashes are handled correctly."""
        client = ConcreteClient(base_url="https://api.example.com/v1/")
        assert client.base_url == "https://api.example.com/v1"

    def test_caching(self):
        """Caching works correctly."""
        client = ConcreteClient(base_url="https://api.example.com", cache_ttl=300)

        # Set cache
        client._set_cache("test_key", {"data": "value"})

        # Get from cache
        cached = client._get_cached("test_key")
        assert cached == {"data": "value"}

    def test_cache_miss(self):
        """Cache miss returns None."""
        client = ConcreteClient(base_url="https://api.example.com")

        cached = client._get_cached("nonexistent_key")
        assert cached is None

    def test_clear_cache(self):
        """Cache clearing works."""
        client = ConcreteClient(base_url="https://api.example.com")

        client._set_cache("key1", "value1")
        client._set_cache("key2", "value2")

        client.clear_cache()

        assert client._get_cached("key1") is None
        assert client._get_cached("key2") is None

    def test_context_manager(self):
        """Context manager closes session."""
        with ConcreteClient(base_url="https://api.example.com") as client:
            assert client._session is not None

    @patch('requests.Session.request')
    def test_successful_request(self, mock_request):
        """Successful request returns data."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": "success"}
        mock_request.return_value = mock_response

        client = ConcreteClient(base_url="https://api.example.com")
        result = client.get("test_endpoint")

        assert result == {"result": "success"}

    @patch('requests.Session.request')
    def test_rate_limit_error(self, mock_request):
        """Rate limit error is raised correctly."""
        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.text = "Rate limit exceeded"
        mock_request.return_value = mock_response

        client = ConcreteClient(base_url="https://api.example.com")

        with pytest.raises(RateLimitError):
            client.get("test_endpoint")

    @patch('requests.Session.request')
    def test_authentication_error(self, mock_request):
        """Authentication error is raised correctly."""
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"
        mock_request.return_value = mock_response

        client = ConcreteClient(base_url="https://api.example.com")

        with pytest.raises(AuthenticationError):
            client.get("test_endpoint")

    @patch('requests.Session.request')
    def test_api_error(self, mock_request):
        """Generic API error is raised correctly."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_request.return_value = mock_response

        client = ConcreteClient(base_url="https://api.example.com")

        with pytest.raises(APIError):
            client.get("test_endpoint")

    @patch('requests.Session.request')
    def test_caching_get_requests(self, mock_request):
        """GET requests are cached."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": "test"}
        mock_request.return_value = mock_response

        client = ConcreteClient(base_url="https://api.example.com", rate_limit_per_min=1000)

        # First request
        result1 = client.get("test_endpoint")

        # Second request (should be cached)
        result2 = client.get("test_endpoint")

        # Request should only be made once
        assert mock_request.call_count == 1
        assert result1 == result2

    @patch('requests.Session.request')
    def test_no_caching_when_disabled(self, mock_request):
        """Caching can be disabled."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": "test"}
        mock_request.return_value = mock_response

        client = ConcreteClient(base_url="https://api.example.com", rate_limit_per_min=1000)

        # Two requests without caching
        client.get("test_endpoint", use_cache=False)
        client.get("test_endpoint", use_cache=False)

        # Both requests should be made
        assert mock_request.call_count == 2


class TestTTLCache:
    """Tests for TTL Cache."""

    def test_cache_set_get(self):
        """Basic set/get works."""
        cache = TTLCache(default_ttl=300)
        cache.set("key", "value")

        assert cache.get("key") == "value"

    def test_cache_expiration(self):
        """Expired entries return None."""
        cache = TTLCache(default_ttl=1)
        cache.set("key", "value")

        # Wait for expiration
        time.sleep(1.1)

        assert cache.get("key") is None

    def test_cache_custom_ttl(self):
        """Custom TTL per entry works."""
        cache = TTLCache(default_ttl=300)
        cache.set("key", "value", ttl=1)

        # Should still be valid
        assert cache.get("key") == "value"

        # Wait for expiration
        time.sleep(1.1)
        assert cache.get("key") is None

    def test_cache_invalidate(self):
        """Invalidation works."""
        cache = TTLCache(default_ttl=300)
        cache.set("key", "value")

        result = cache.invalidate("key")
        assert result is True
        assert cache.get("key") is None

    def test_cache_invalidate_nonexistent(self):
        """Invalidating nonexistent key returns False."""
        cache = TTLCache(default_ttl=300)

        result = cache.invalidate("nonexistent")
        assert result is False

    def test_cache_clear(self):
        """Clear removes all entries."""
        cache = TTLCache(default_ttl=300)
        cache.set("key1", "value1")
        cache.set("key2", "value2")

        cache.clear()

        assert cache.get("key1") is None
        assert cache.get("key2") is None
        assert len(cache) == 0

    def test_cache_len(self):
        """Length is correct."""
        cache = TTLCache(default_ttl=300)
        cache.set("key1", "value1")
        cache.set("key2", "value2")

        assert len(cache) == 2

    def test_cache_contains(self):
        """Contains check works."""
        cache = TTLCache(default_ttl=300)
        cache.set("key", "value")

        assert "key" in cache
        assert "nonexistent" not in cache

    def test_cache_cleanup(self):
        """Cleanup removes expired entries."""
        cache = TTLCache(default_ttl=1)
        cache.set("expired", "value")

        time.sleep(1.1)

        removed = cache.cleanup()
        assert removed == 1
        assert len(cache) == 0
