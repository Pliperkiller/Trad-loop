"""Tests for DefiLlama client."""

import pytest
from unittest.mock import patch, Mock
import pandas as pd

from src.indicators.fundamental.defillama import DefiLlamaClient
from src.indicators.fundamental.models import ProtocolData


class TestDefiLlamaClient:
    """Tests for DefiLlama API client."""

    def test_initialization(self):
        """Client initializes correctly."""
        client = DefiLlamaClient()

        assert client.base_url == "https://api.llama.fi"
        assert client.rate_limit_per_min == 60

    @patch.object(DefiLlamaClient, 'get')
    def test_get_protocol_tvl(self, mock_get):
        """Get protocol TVL returns correct value."""
        mock_get.return_value = {
            "name": "Aave",
            "tvl": 10000000000.0,
            "chainTvls": {"ethereum": 5000000000.0},
        }

        client = DefiLlamaClient()
        result = client.get_protocol_tvl("aave")

        assert result == 10000000000.0
        mock_get.assert_called_once()

    @patch.object(DefiLlamaClient, 'get')
    def test_get_protocol_tvl_not_found(self, mock_get):
        """Get protocol TVL returns 0 for unknown protocol."""
        mock_get.return_value = {}

        client = DefiLlamaClient()
        result = client.get_protocol_tvl("unknown")

        assert result == 0.0

    @patch.object(DefiLlamaClient, 'get')
    def test_get_protocol_data(self, mock_get):
        """Get protocol data returns ProtocolData object."""
        mock_get.return_value = {
            "name": "Aave",
            "symbol": "AAVE",
            "tvl": 10000000000.0,
            "chain": "Multi-Chain",
            "change_1d": 2.5,
            "change_7d": 5.0,
            "category": "Lending",
            "chains": ["Ethereum", "Polygon"],
        }

        client = DefiLlamaClient()
        result = client.get_protocol_data("aave")

        assert isinstance(result, ProtocolData)
        assert result.name == "Aave"
        assert result.tvl == 10000000000.0

    @patch.object(DefiLlamaClient, 'get')
    def test_get_chain_tvl(self, mock_get):
        """Get chain TVL returns correct value."""
        mock_get.return_value = [
            {"name": "Ethereum", "tvl": 50000000000.0},
            {"name": "BSC", "tvl": 10000000000.0},
        ]

        client = DefiLlamaClient()
        result = client.get_chain_tvl("ethereum")

        assert result == 50000000000.0

    @patch.object(DefiLlamaClient, 'get')
    def test_get_chain_tvl_not_found(self, mock_get):
        """Get chain TVL returns 0 for unknown chain."""
        mock_get.return_value = [
            {"name": "Ethereum", "tvl": 50000000000.0},
        ]

        client = DefiLlamaClient()
        result = client.get_chain_tvl("unknown")

        assert result == 0.0

    @patch.object(DefiLlamaClient, 'get')
    def test_get_tvl_history(self, mock_get):
        """Get global TVL history returns DataFrame."""
        mock_get.return_value = [
            {"date": 1609459200, "totalLiquidityUSD": 5000000000.0},
            {"date": 1609545600, "totalLiquidityUSD": 5500000000.0},
            {"date": 1609632000, "totalLiquidityUSD": 6000000000.0},
        ]

        client = DefiLlamaClient()
        result = client.get_tvl_history()

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3

    @patch.object(DefiLlamaClient, 'get')
    def test_get_top_protocols(self, mock_get):
        """Get top protocols returns list of ProtocolData."""
        mock_get.return_value = [
            {"name": "Lido", "symbol": "LDO", "tvl": 15000000000.0, "chains": ["Ethereum"]},
            {"name": "Aave", "symbol": "AAVE", "tvl": 10000000000.0, "chains": ["Ethereum"]},
            {"name": "MakerDAO", "symbol": "MKR", "tvl": 8000000000.0, "chains": ["Ethereum"]},
        ]

        client = DefiLlamaClient()
        result = client.get_top_protocols(n=3)

        assert isinstance(result, list)
        assert len(result) == 3
        assert isinstance(result[0], ProtocolData)
        assert result[0].name == "Lido"

    @patch.object(DefiLlamaClient, 'get')
    def test_get_all_chains_tvl(self, mock_get):
        """Get all chains TVL returns dict."""
        mock_get.return_value = [
            {"name": "Ethereum", "tvl": 50000000000.0},
            {"name": "BSC", "tvl": 10000000000.0},
            {"name": "Polygon", "tvl": 5000000000.0},
        ]

        client = DefiLlamaClient()
        result = client.get_all_chains_tvl()

        assert isinstance(result, dict)
        assert len(result) == 3
        assert "Ethereum" in result

    @patch.object(DefiLlamaClient, 'get')
    def test_get_stablecoins(self, mock_get):
        """Get stablecoins returns list."""
        mock_get.return_value = {
            "peggedAssets": [
                {"name": "Tether", "symbol": "USDT", "circulating": {"peggedUSD": 80000000000.0}},
                {"name": "USD Coin", "symbol": "USDC", "circulating": {"peggedUSD": 40000000000.0}},
            ]
        }

        client = DefiLlamaClient()
        result = client.get_stablecoins()

        assert isinstance(result, list)
        assert len(result) == 2

    @patch.object(DefiLlamaClient, 'get')
    def test_get_yields(self, mock_get):
        """Get yields returns list of pools."""
        mock_get.return_value = {
            "data": [
                {"pool": "pool1", "apy": 5.0, "tvlUsd": 1000000.0},
                {"pool": "pool2", "apy": 10.0, "tvlUsd": 500000.0},
            ]
        }

        client = DefiLlamaClient()
        result = client.get_yields()

        assert isinstance(result, list)
        assert len(result) == 2

    @patch.object(DefiLlamaClient, 'get')
    def test_get_global_tvl(self, mock_get):
        """Get global TVL returns sum of all chains."""
        mock_get.return_value = [
            {"name": "Ethereum", "tvl": 50000000000.0},
            {"name": "BSC", "tvl": 10000000000.0},
        ]

        client = DefiLlamaClient()
        result = client.get_global_tvl()

        assert result == 60000000000.0

