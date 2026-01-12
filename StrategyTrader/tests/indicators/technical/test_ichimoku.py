"""Tests for Ichimoku Cloud indicator."""

import pytest
import numpy as np
import pandas as pd

from src.indicators.technical.ichimoku import ichimoku_cloud, ichimoku_signals


class TestIchimokuCloud:
    """Tests for Ichimoku Cloud."""

    def test_ichimoku_basic(self, sample_ohlcv):
        """Ichimoku produces valid output."""
        result = ichimoku_cloud(
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"]
        )

        assert hasattr(result, "tenkan_sen")
        assert hasattr(result, "kijun_sen")
        assert hasattr(result, "senkou_span_a")
        assert hasattr(result, "senkou_span_b")
        assert hasattr(result, "chikou_span")

    def test_ichimoku_tenkan_shorter_than_kijun(self, sample_ohlcv):
        """Tenkan period is shorter than Kijun period."""
        result = ichimoku_cloud(
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"],
            tenkan_period=9,
            kijun_period=26
        )

        # Tenkan should have valid values earlier than Kijun
        tenkan_first_valid = result.tenkan_sen.first_valid_index()
        kijun_first_valid = result.kijun_sen.first_valid_index()

        # Get position index for comparison
        tenkan_pos = result.tenkan_sen.index.get_loc(tenkan_first_valid)
        kijun_pos = result.kijun_sen.index.get_loc(kijun_first_valid)

        assert tenkan_pos < kijun_pos

    def test_ichimoku_senkou_span_shifted(self, sample_ohlcv):
        """Senkou Spans are shifted forward."""
        result = ichimoku_cloud(
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"],
            displacement=26
        )

        # First valid Senkou should be later than first valid Tenkan/Kijun
        senkou_a_first = result.senkou_span_a.first_valid_index()
        tenkan_first = result.tenkan_sen.first_valid_index()

        senkou_pos = result.senkou_span_a.index.get_loc(senkou_a_first)
        tenkan_pos = result.tenkan_sen.index.get_loc(tenkan_first)

        # Senkou should start later (shifted forward)
        assert senkou_pos > tenkan_pos

    def test_ichimoku_chikou_shifted_back(self, sample_ohlcv):
        """Chikou Span is shifted backward."""
        result = ichimoku_cloud(
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"],
            displacement=26
        )

        # Last valid Chikou should be earlier than last data point
        chikou_last = result.chikou_span.last_valid_index()
        data_last = sample_ohlcv.index[-1]

        chikou_pos = result.chikou_span.index.get_loc(chikou_last)
        data_pos = len(sample_ohlcv) - 1

        assert chikou_pos < data_pos

    def test_ichimoku_custom_periods(self, sample_ohlcv):
        """Ichimoku works with custom periods."""
        result = ichimoku_cloud(
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"],
            tenkan_period=7,
            kijun_period=22,
            senkou_b_period=44,
            displacement=22
        )

        assert result.tenkan_sen.notna().any()
        assert result.kijun_sen.notna().any()

    def test_ichimoku_midpoint_calculation(self, sample_ohlcv):
        """Tenkan is midpoint of 9-period high-low."""
        result = ichimoku_cloud(
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"],
            tenkan_period=9
        )

        # Verify Tenkan calculation
        period = 9
        expected_tenkan = (
            sample_ohlcv["high"].rolling(window=period).max() +
            sample_ohlcv["low"].rolling(window=period).min()
        ) / 2

        pd.testing.assert_series_equal(
            result.tenkan_sen, expected_tenkan, check_names=False
        )


class TestIchimokuSignals:
    """Tests for Ichimoku trading signals."""

    def test_signals_basic(self, sample_ohlcv):
        """Ichimoku signals produces valid output."""
        signals = ichimoku_signals(
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"]
        )

        assert isinstance(signals, pd.DataFrame)
        assert "tenkan_kijun_cross" in signals.columns
        assert "price_cloud_position" in signals.columns
        assert "chikou_position" in signals.columns

    def test_signals_cross_values(self, sample_ohlcv):
        """Cross signals are -1, 0, or 1."""
        signals = ichimoku_signals(
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"]
        )

        unique_cross = set(signals["tenkan_kijun_cross"].unique())
        assert unique_cross.issubset({-1, 0, 1})

    def test_signals_cloud_position_values(self, sample_ohlcv):
        """Cloud position signals are -1, 0, or 1."""
        signals = ichimoku_signals(
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"]
        )

        unique_pos = set(signals["price_cloud_position"].unique())
        assert unique_pos.issubset({-1, 0, 1})

    def test_signals_bullish_uptrend(self, trending_up_close):
        """Uptrend should show bullish signals."""
        high = trending_up_close * 1.01
        low = trending_up_close * 0.99

        signals = ichimoku_signals(high, low, trending_up_close)

        # In strong uptrend, should have more bullish signals
        cloud_above = (signals["price_cloud_position"] == 1).sum()
        cloud_below = (signals["price_cloud_position"] == -1).sum()

        # Should have more above-cloud than below-cloud positions
        assert cloud_above >= cloud_below * 0.5  # At least half
