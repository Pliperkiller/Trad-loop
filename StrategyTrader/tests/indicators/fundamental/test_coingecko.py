"""Tests for CoinGecko client."""

import pytest
from unittest.mock import patch, Mock
import pandas as pd

from src.indicators.fundamental.coingecko import CoinGeckoClient
from src.indicators.fundamental.models import MarketData, GlobalMarketData


class TestCoinGeckoClient:
    """Tests for CoinGecko API client."""

    def test_initialization(self):
        """Client initializes correctly."""
        client = CoinGeckoClient()

        assert client.base_url == "https://api.coingecko.com/api/v3"
        assert client.rate_limit_per_min == 10

    def test_initialization_with_api_key(self):
        """Client initializes with Pro API key."""
        client = CoinGeckoClient(api_key="pro-api-key")

        assert client.api_key == "pro-api-key"
        assert "x-cg-pro-api-key" in client._session.headers

    @patch.object(CoinGeckoClient, 'get')
    def test_ping(self, mock_get):
        """Ping returns True when API is healthy."""
        mock_get.return_value = {"gecko_says": "(V3) To the Moon!"}

        client = CoinGeckoClient()
        result = client.ping()

        assert result is True

    @patch.object(CoinGeckoClient, 'get')
    def test_ping_failure(self, mock_get):
        """Ping returns False when API fails."""
        mock_get.side_effect = Exception("Connection error")

        client = CoinGeckoClient()
        result = client.ping()

        assert result is False

    @patch.object(CoinGeckoClient, 'get')
    def test_get_price(self, mock_get):
        """Get price returns correct value."""
        mock_get.return_value = {"bitcoin": {"usd": 50000.0}}

        client = CoinGeckoClient()
        result = client.get_price("bitcoin", "usd")

        assert result == 50000.0
        mock_get.assert_called_once()

    @patch.object(CoinGeckoClient, 'get')
    def test_get_price_not_found(self, mock_get):
        """Get price returns 0 for unknown coin."""
        mock_get.return_value = {}

        client = CoinGeckoClient()
        result = client.get_price("unknown", "usd")

        assert result == 0.0

    @patch.object(CoinGeckoClient, 'get')
    def test_get_prices_multiple(self, mock_get):
        """Get prices for multiple coins."""
        mock_get.return_value = {
            "bitcoin": {"usd": 50000.0, "usd_24h_change": 5.0},
            "ethereum": {"usd": 3000.0, "usd_24h_change": 3.0},
        }

        client = CoinGeckoClient()
        result = client.get_prices(["bitcoin", "ethereum"])

        assert "bitcoin" in result
        assert "ethereum" in result
        assert result["bitcoin"]["usd"] == 50000.0

    @patch.object(CoinGeckoClient, 'get')
    def test_get_market_data(self, mock_get):
        """Get market data returns MarketData object."""
        mock_get.return_value = {
            "symbol": "btc",
            "name": "Bitcoin",
            "market_data": {
                "current_price": {"usd": 50000.0},
                "market_cap": {"usd": 1000000000000.0},
                "total_volume": {"usd": 50000000000.0},
                "price_change_percentage_1h_in_currency": {"usd": 0.5},
                "price_change_percentage_24h": 5.0,
                "price_change_percentage_7d": 10.0,
                "price_change_percentage_30d": 15.0,
                "ath": {"usd": 69000.0},
                "ath_date": {"usd": "2021-11-10T14:24:11.849Z"},
                "atl": {"usd": 67.81},
                "atl_date": {"usd": "2013-07-06T00:00:00.000Z"},
                "circulating_supply": 19000000.0,
                "total_supply": 21000000.0,
                "max_supply": 21000000.0,
            }
        }

        client = CoinGeckoClient()
        result = client.get_market_data("bitcoin")

        assert isinstance(result, MarketData)
        assert result.symbol == "BTC"
        assert result.name == "Bitcoin"
        assert result.price == 50000.0
        assert result.market_cap == 1000000000000.0

    @patch.object(CoinGeckoClient, 'get')
    def test_get_market_chart(self, mock_get):
        """Get market chart returns DataFrame."""
        mock_get.return_value = {
            "prices": [[1609459200000, 29000], [1609545600000, 30000]],
            "market_caps": [[1609459200000, 540000000000], [1609545600000, 560000000000]],
            "total_volumes": [[1609459200000, 40000000000], [1609545600000, 45000000000]],
        }

        client = CoinGeckoClient()
        result = client.get_market_chart("bitcoin", days=30)

        assert isinstance(result, pd.DataFrame)
        assert "price" in result.columns
        assert "market_cap" in result.columns
        assert "volume" in result.columns
        assert len(result) == 2

    @patch.object(CoinGeckoClient, 'get')
    def test_get_global_data(self, mock_get):
        """Get global data returns GlobalMarketData."""
        mock_get.return_value = {
            "data": {
                "total_market_cap": {"usd": 2000000000000.0},
                "total_volume": {"usd": 100000000000.0},
                "market_cap_percentage": {"btc": 45.0, "eth": 18.0},
                "active_cryptocurrencies": 10000,
                "market_cap_change_percentage_24h_usd": 3.5,
            }
        }

        client = CoinGeckoClient()
        result = client.get_global_data()

        assert isinstance(result, GlobalMarketData)
        assert result.total_market_cap == 2000000000000.0
        assert result.btc_dominance == 45.0
        assert result.eth_dominance == 18.0

    @patch.object(CoinGeckoClient, 'get')
    def test_get_trending(self, mock_get):
        """Get trending coins returns list."""
        mock_get.return_value = {
            "coins": [
                {"item": {"id": "bitcoin", "name": "Bitcoin"}},
                {"item": {"id": "ethereum", "name": "Ethereum"}},
            ]
        }

        client = CoinGeckoClient()
        result = client.get_trending()

        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["id"] == "bitcoin"

    @patch.object(CoinGeckoClient, 'get')
    def test_search(self, mock_get):
        """Search returns results."""
        mock_get.return_value = {
            "coins": [{"id": "bitcoin", "name": "Bitcoin"}],
            "exchanges": [],
            "categories": [],
        }

        client = CoinGeckoClient()
        result = client.search("bitcoin")

        assert "coins" in result
        assert len(result["coins"]) == 1

    @patch.object(CoinGeckoClient, 'get')
    def test_get_coin_list(self, mock_get):
        """Get coin list returns list of coins."""
        mock_get.return_value = [
            {"id": "bitcoin", "symbol": "btc", "name": "Bitcoin"},
            {"id": "ethereum", "symbol": "eth", "name": "Ethereum"},
        ]

        client = CoinGeckoClient()
        result = client.get_coin_list()

        assert isinstance(result, list)
        assert len(result) == 2
