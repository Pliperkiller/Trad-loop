"""Tests for Santiment client structure."""

import pytest
from unittest.mock import patch, Mock
import pandas as pd

from src.indicators.fundamental.santiment import SantimentClient
from src.indicators.fundamental.base_client import AuthenticationError


class TestSantimentClient:
    """Tests for Santiment API client structure."""

    def test_initialization_without_api_key(self):
        """Client raises error without API key."""
        with pytest.raises(ValueError):
            SantimentClient(api_key="")

    def test_initialization_with_api_key(self):
        """Client initializes with API key."""
        client = SantimentClient(api_key="test-api-key")

        assert client.api_key == "test-api-key"
        assert "santiment" in client.base_url.lower()

    def test_has_required_methods(self):
        """Client has all required methods."""
        client = SantimentClient(api_key="test-api-key")

        assert hasattr(client, 'get_social_volume')
        assert hasattr(client, 'get_dev_activity')
        assert hasattr(client, 'get_sentiment')
        assert hasattr(client, 'get_social_dominance')

    @patch.object(SantimentClient, '_get_metric')
    def test_get_social_volume(self, mock_get_metric):
        """Get social volume metric."""
        mock_df = pd.DataFrame({
            "value": [1000.0, 1200.0]
        }, index=pd.to_datetime(["2021-01-01", "2021-01-02"]))
        mock_get_metric.return_value = mock_df

        client = SantimentClient(api_key="test-api-key")
        result = client.get_social_volume("bitcoin")

        assert isinstance(result, pd.DataFrame)

    @patch.object(SantimentClient, '_get_metric')
    def test_get_dev_activity(self, mock_get_metric):
        """Get development activity metric."""
        mock_df = pd.DataFrame({
            "value": [50.0, 55.0]
        }, index=pd.to_datetime(["2021-01-01", "2021-01-02"]))
        mock_get_metric.return_value = mock_df

        client = SantimentClient(api_key="test-api-key")
        result = client.get_dev_activity("ethereum")

        assert isinstance(result, pd.DataFrame)

    @patch.object(SantimentClient, '_get_metric')
    def test_get_sentiment(self, mock_get_metric):
        """Get sentiment score."""
        mock_df = pd.DataFrame({
            "value": [0.6, 0.7]
        }, index=pd.to_datetime(["2021-01-01", "2021-01-02"]))
        mock_get_metric.return_value = mock_df

        client = SantimentClient(api_key="test-api-key")
        result = client.get_sentiment("bitcoin")

        assert isinstance(result, pd.DataFrame)

    @patch.object(SantimentClient, '_get_metric')
    def test_get_social_dominance(self, mock_get_metric):
        """Get social dominance metric."""
        mock_df = pd.DataFrame({
            "value": [15.0, 18.0]
        }, index=pd.to_datetime(["2021-01-01", "2021-01-02"]))
        mock_get_metric.return_value = mock_df

        client = SantimentClient(api_key="test-api-key")
        result = client.get_social_dominance("bitcoin")

        assert isinstance(result, pd.DataFrame)

    def test_graphql_client(self):
        """Client uses GraphQL for queries."""
        client = SantimentClient(api_key="test-api-key")

        # Should have method for executing GraphQL queries
        assert hasattr(client, '_graphql_query')

    def test_get_supported_assets(self):
        """Get list of supported assets."""
        client = SantimentClient(api_key="test-api-key")
        result = client.get_supported_assets()

        assert isinstance(result, list)
        assert "bitcoin" in result
        assert "ethereum" in result

    @patch.object(SantimentClient, '_graphql_query')
    def test_get_trending_words(self, mock_query):
        """Get trending words."""
        mock_query.return_value = {
            "getTrendingWords": [
                {"word": "bitcoin", "score": 100},
                {"word": "eth", "score": 80},
            ]
        }

        client = SantimentClient(api_key="test-api-key")
        result = client.get_trending_words()

        assert isinstance(result, list)

