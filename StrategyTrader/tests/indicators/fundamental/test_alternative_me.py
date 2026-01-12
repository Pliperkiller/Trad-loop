"""Tests for Fear & Greed Index client."""

import pytest
from unittest.mock import patch, Mock
import pandas as pd

from src.indicators.fundamental.alternative_me import FearGreedClient
from src.indicators.fundamental.models import FearGreedIndex


class TestFearGreedClient:
    """Tests for Fear & Greed Index API client."""

    def test_initialization(self):
        """Client initializes correctly."""
        client = FearGreedClient()

        assert client.base_url == "https://api.alternative.me/fng"
        assert client.rate_limit_per_min == 30

    @patch.object(FearGreedClient, 'get')
    def test_get_current(self, mock_get):
        """Get current Fear & Greed index."""
        mock_get.return_value = {
            "data": [
                {
                    "value": "25",
                    "value_classification": "Extreme Fear",
                    "timestamp": "1609459200",
                }
            ]
        }

        client = FearGreedClient()
        result = client.get_current()

        assert isinstance(result, FearGreedIndex)
        assert result.value == 25
        assert result.classification == "Extreme Fear"

    @patch.object(FearGreedClient, 'get')
    def test_get_current_greed(self, mock_get):
        """Get current when market is greedy."""
        mock_get.return_value = {
            "data": [
                {
                    "value": "75",
                    "value_classification": "Greed",
                    "timestamp": "1609459200",
                }
            ]
        }

        client = FearGreedClient()
        result = client.get_current()

        assert result.value == 75
        assert result.classification == "Greed"

    @patch.object(FearGreedClient, 'get')
    def test_get_historical(self, mock_get):
        """Get historical Fear & Greed data."""
        mock_get.return_value = {
            "data": [
                {"value": "30", "value_classification": "Fear", "timestamp": "1609632000"},
                {"value": "25", "value_classification": "Extreme Fear", "timestamp": "1609545600"},
                {"value": "28", "value_classification": "Fear", "timestamp": "1609459200"},
            ]
        }

        client = FearGreedClient()
        result = client.get_historical(days=3)

        assert isinstance(result, pd.DataFrame)
        assert "value" in result.columns
        assert "classification" in result.columns
        assert len(result) == 3

    @patch.object(FearGreedClient, 'get')
    def test_is_extreme_fear(self, mock_get):
        """Check if market is in extreme fear."""
        mock_get.return_value = {
            "data": [
                {
                    "value": "15",
                    "value_classification": "Extreme Fear",
                    "timestamp": "1609459200",
                }
            ]
        }

        client = FearGreedClient()
        result = client.is_extreme_fear()

        assert result is True

    @patch.object(FearGreedClient, 'get')
    def test_is_not_extreme_fear(self, mock_get):
        """Check when market is not in extreme fear."""
        mock_get.return_value = {
            "data": [
                {
                    "value": "45",
                    "value_classification": "Fear",
                    "timestamp": "1609459200",
                }
            ]
        }

        client = FearGreedClient()
        result = client.is_extreme_fear()

        assert result is False

    @patch.object(FearGreedClient, 'get')
    def test_is_extreme_greed(self, mock_get):
        """Check if market is in extreme greed."""
        mock_get.return_value = {
            "data": [
                {
                    "value": "85",
                    "value_classification": "Extreme Greed",
                    "timestamp": "1609459200",
                }
            ]
        }

        client = FearGreedClient()
        result = client.is_extreme_greed()

        assert result is True

    @patch.object(FearGreedClient, 'get')
    def test_is_not_extreme_greed(self, mock_get):
        """Check when market is not in extreme greed."""
        mock_get.return_value = {
            "data": [
                {
                    "value": "60",
                    "value_classification": "Greed",
                    "timestamp": "1609459200",
                }
            ]
        }

        client = FearGreedClient()
        result = client.is_extreme_greed()

        assert result is False

    @patch.object(FearGreedClient, 'get')
    def test_classification_values(self, mock_get):
        """Test all classification values."""
        classifications = [
            (10, "Extreme Fear"),
            (30, "Fear"),
            (50, "Neutral"),
            (70, "Greed"),
            (90, "Extreme Greed"),
        ]

        client = FearGreedClient()

        for value, expected_class in classifications:
            mock_get.return_value = {
                "data": [
                    {
                        "value": str(value),
                        "value_classification": expected_class,
                        "timestamp": "1609459200",
                    }
                ]
            }

            result = client.get_current()
            assert result.classification == expected_class

    @patch.object(FearGreedClient, 'get')
    def test_get_average(self, mock_get):
        """Get average Fear & Greed over period."""
        mock_get.return_value = {
            "data": [
                {"value": "30", "value_classification": "Fear", "timestamp": "1609632000"},
                {"value": "40", "value_classification": "Fear", "timestamp": "1609545600"},
                {"value": "50", "value_classification": "Neutral", "timestamp": "1609459200"},
            ]
        }

        client = FearGreedClient()
        result = client.get_average(days=3)

        assert result == 40.0  # Average of 30, 40, 50

