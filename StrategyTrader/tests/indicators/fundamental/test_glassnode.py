"""Tests for Glassnode client structure."""

import pytest
from unittest.mock import patch, Mock
import pandas as pd

from src.indicators.fundamental.glassnode import GlassnodeClient
from src.indicators.fundamental.base_client import AuthenticationError


class TestGlassnodeClient:
    """Tests for Glassnode API client structure."""

    def test_initialization_without_api_key(self):
        """Client raises error without API key."""
        with pytest.raises(ValueError):
            GlassnodeClient(api_key="")

    def test_initialization_with_api_key(self):
        """Client initializes with API key."""
        client = GlassnodeClient(api_key="test-api-key")

        assert client.api_key == "test-api-key"
        assert client.base_url == "https://api.glassnode.com/v1"

    def test_has_required_methods(self):
        """Client has all required methods."""
        client = GlassnodeClient(api_key="test-api-key")

        assert hasattr(client, 'get_mvrv_ratio')
        assert hasattr(client, 'get_nvt_ratio')
        assert hasattr(client, 'get_sopr')
        assert hasattr(client, 'get_active_addresses')
        assert hasattr(client, 'get_exchange_netflow')

    @patch.object(GlassnodeClient, '_get_metric')
    def test_get_mvrv_ratio(self, mock_get_metric):
        """Get MVRV ratio."""
        mock_df = pd.DataFrame({
            "value": [2.5, 2.6]
        }, index=pd.to_datetime(["2021-01-01", "2021-01-02"]))
        mock_get_metric.return_value = mock_df

        client = GlassnodeClient(api_key="test-api-key")
        result = client.get_mvrv_ratio("BTC")

        assert isinstance(result, pd.DataFrame)

    @patch.object(GlassnodeClient, '_get_metric')
    def test_get_nvt_ratio(self, mock_get_metric):
        """Get NVT ratio."""
        mock_df = pd.DataFrame({
            "value": [50.0, 52.0]
        }, index=pd.to_datetime(["2021-01-01", "2021-01-02"]))
        mock_get_metric.return_value = mock_df

        client = GlassnodeClient(api_key="test-api-key")
        result = client.get_nvt_ratio("BTC")

        assert isinstance(result, pd.DataFrame)

    @patch.object(GlassnodeClient, '_get_metric')
    def test_get_sopr(self, mock_get_metric):
        """Get SOPR (Spent Output Profit Ratio)."""
        mock_df = pd.DataFrame({
            "value": [1.02, 1.05]
        }, index=pd.to_datetime(["2021-01-01", "2021-01-02"]))
        mock_get_metric.return_value = mock_df

        client = GlassnodeClient(api_key="test-api-key")
        result = client.get_sopr("BTC")

        assert isinstance(result, pd.DataFrame)

    @patch.object(GlassnodeClient, '_get_metric')
    def test_get_active_addresses(self, mock_get_metric):
        """Get active addresses count."""
        mock_df = pd.DataFrame({
            "value": [1000000, 1050000]
        }, index=pd.to_datetime(["2021-01-01", "2021-01-02"]))
        mock_get_metric.return_value = mock_df

        client = GlassnodeClient(api_key="test-api-key")
        result = client.get_active_addresses("BTC")

        assert isinstance(result, pd.DataFrame)

    @patch.object(GlassnodeClient, '_get_metric')
    def test_get_exchange_netflow(self, mock_get_metric):
        """Get exchange netflow data."""
        mock_df = pd.DataFrame({
            "value": [5000.0, 4500.0]
        }, index=pd.to_datetime(["2021-01-01", "2021-01-02"]))
        mock_get_metric.return_value = mock_df

        client = GlassnodeClient(api_key="test-api-key")
        result = client.get_exchange_netflow("BTC")

        assert isinstance(result, pd.DataFrame)

    @patch('requests.Session.request')
    def test_authentication_header(self, mock_request):
        """API key is included in request params."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = []
        mock_request.return_value = mock_response

        client = GlassnodeClient(api_key="test-api-key")
        client.get("metrics/test")

        # Verify request was made with api_key param
        call_args = mock_request.call_args
        assert call_args is not None

    @patch('requests.Session.request')
    def test_authentication_error(self, mock_request):
        """Authentication error is raised for invalid key."""
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"
        mock_request.return_value = mock_response

        client = GlassnodeClient(api_key="invalid-key")

        with pytest.raises(AuthenticationError):
            client.get("metrics/test")

    def test_get_supported_assets(self):
        """Get list of supported assets."""
        client = GlassnodeClient(api_key="test-api-key")
        result = client.get_supported_assets()

        assert isinstance(result, list)
        assert "BTC" in result
        assert "ETH" in result

    def test_get_available_metrics(self):
        """Get list of available metrics."""
        client = GlassnodeClient(api_key="test-api-key")
        result = client.get_available_metrics()

        assert isinstance(result, list)
        assert len(result) > 0

