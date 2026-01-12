"""Tests for volatility indicators."""

import pytest
import numpy as np
import pandas as pd

from src.indicators.technical.volatility import (
    atr,
    bollinger_bands,
    keltner_channels,
    donchian_channels,
)


class TestATR:
    """Tests for Average True Range."""

    def test_atr_basic(self, sample_ohlcv):
        """ATR produces valid output."""
        result = atr(
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"]
        )

        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_ohlcv)

    def test_atr_positive(self, sample_ohlcv):
        """ATR values are positive."""
        result = atr(
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"]
        )

        valid_values = result.dropna()
        assert (valid_values >= 0).all()

    def test_atr_higher_in_volatile_market(self, sample_ohlcv, high_volatility_close):
        """ATR is higher in volatile market."""
        # Normal market
        atr_normal = atr(
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"]
        )

        # Volatile market - create wider high-low range
        high_vol = high_volatility_close * 1.05
        low_vol = high_volatility_close * 0.95

        atr_volatile = atr(high_vol, low_vol, high_volatility_close)

        # Volatile should have higher ATR on average
        assert atr_volatile.dropna().mean() > atr_normal.dropna().mean()

    def test_atr_constant_range(self):
        """ATR equals constant range for constant range data."""
        n = 50
        high = pd.Series([110.0] * n)
        low = pd.Series([100.0] * n)
        close = pd.Series([105.0] * n)

        result = atr(high, low, close, period=14)

        # ATR should be approximately 10 (high - low)
        valid_atr = result.iloc[14:].dropna()
        assert (abs(valid_atr - 10.0) < 0.1).all()


class TestBollingerBands:
    """Tests for Bollinger Bands."""

    def test_bollinger_basic(self, sample_close):
        """Bollinger Bands produces valid output."""
        result = bollinger_bands(sample_close)

        assert hasattr(result, "middle")
        assert hasattr(result, "upper")
        assert hasattr(result, "lower")
        assert hasattr(result, "bandwidth")
        assert hasattr(result, "percent_b")

    def test_bollinger_band_ordering(self, sample_close):
        """Upper > Middle > Lower."""
        result = bollinger_bands(sample_close)

        valid_idx = ~result.middle.isna()
        assert (result.upper[valid_idx] >= result.middle[valid_idx]).all()
        assert (result.middle[valid_idx] >= result.lower[valid_idx]).all()

    def test_bollinger_middle_is_sma(self, sample_close):
        """Middle band equals SMA."""
        result = bollinger_bands(sample_close, period=20)

        expected_sma = sample_close.rolling(window=20).mean()
        pd.testing.assert_series_equal(
            result.middle, expected_sma, check_names=False
        )

    def test_bollinger_bandwidth_positive(self, sample_close):
        """Bandwidth is positive."""
        result = bollinger_bands(sample_close)

        valid_bw = result.bandwidth.dropna()
        assert (valid_bw >= 0).all()

    def test_bollinger_percent_b_range(self, sample_close):
        """Percent B is typically between 0 and 1 (can exceed)."""
        result = bollinger_bands(sample_close)

        valid_pb = result.percent_b.dropna()
        # Most values should be near 0-1 range
        in_range = ((valid_pb >= -0.5) & (valid_pb <= 1.5)).sum()
        assert in_range > len(valid_pb) * 0.8

    def test_bollinger_custom_std_dev(self, sample_close):
        """Custom std_dev affects band width."""
        result_2 = bollinger_bands(sample_close, std_dev=2.0)
        result_3 = bollinger_bands(sample_close, std_dev=3.0)

        # 3-std bands should be wider
        width_2 = (result_2.upper - result_2.lower).dropna().mean()
        width_3 = (result_3.upper - result_3.lower).dropna().mean()

        assert width_3 > width_2

    def test_bollinger_invalid_std_dev(self, sample_close):
        """Bollinger raises error for invalid std_dev."""
        with pytest.raises(ValueError):
            bollinger_bands(sample_close, std_dev=-1.0)


class TestKeltnerChannels:
    """Tests for Keltner Channels."""

    def test_keltner_basic(self, sample_ohlcv):
        """Keltner Channels produces valid output."""
        result = keltner_channels(
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"]
        )

        assert hasattr(result, "middle")
        assert hasattr(result, "upper")
        assert hasattr(result, "lower")

    def test_keltner_band_ordering(self, sample_ohlcv):
        """Upper > Middle > Lower."""
        result = keltner_channels(
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"]
        )

        # Use index where all bands are valid
        valid_idx = ~(result.middle.isna() | result.upper.isna() | result.lower.isna())
        assert (result.upper[valid_idx] >= result.middle[valid_idx]).all()
        assert (result.middle[valid_idx] >= result.lower[valid_idx]).all()

    def test_keltner_middle_is_ema(self, sample_ohlcv):
        """Middle band is EMA of close."""
        result = keltner_channels(
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"],
            period=20
        )

        expected_ema = sample_ohlcv["close"].ewm(span=20, adjust=False).mean()
        pd.testing.assert_series_equal(
            result.middle, expected_ema, check_names=False
        )

    def test_keltner_invalid_multiplier(self, sample_ohlcv):
        """Keltner raises error for invalid multiplier."""
        with pytest.raises(ValueError):
            keltner_channels(
                sample_ohlcv["high"],
                sample_ohlcv["low"],
                sample_ohlcv["close"],
                multiplier=-1.0
            )


class TestDonchianChannels:
    """Tests for Donchian Channels."""

    def test_donchian_basic(self, sample_ohlcv):
        """Donchian Channels produces valid output."""
        result = donchian_channels(
            sample_ohlcv["high"],
            sample_ohlcv["low"]
        )

        assert hasattr(result, "upper")
        assert hasattr(result, "middle")
        assert hasattr(result, "lower")

    def test_donchian_band_ordering(self, sample_ohlcv):
        """Upper > Middle > Lower."""
        result = donchian_channels(
            sample_ohlcv["high"],
            sample_ohlcv["low"]
        )

        valid_idx = ~result.middle.isna()
        assert (result.upper[valid_idx] >= result.middle[valid_idx]).all()
        assert (result.middle[valid_idx] >= result.lower[valid_idx]).all()

    def test_donchian_upper_is_highest_high(self, sample_ohlcv):
        """Upper band equals rolling max of high."""
        period = 20
        result = donchian_channels(
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            period=period
        )

        expected = sample_ohlcv["high"].rolling(window=period).max()
        pd.testing.assert_series_equal(result.upper, expected, check_names=False)

    def test_donchian_lower_is_lowest_low(self, sample_ohlcv):
        """Lower band equals rolling min of low."""
        period = 20
        result = donchian_channels(
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            period=period
        )

        expected = sample_ohlcv["low"].rolling(window=period).min()
        pd.testing.assert_series_equal(result.lower, expected, check_names=False)

    def test_donchian_middle_is_average(self, sample_ohlcv):
        """Middle band equals (upper + lower) / 2."""
        result = donchian_channels(
            sample_ohlcv["high"],
            sample_ohlcv["low"]
        )

        expected = (result.upper + result.lower) / 2
        pd.testing.assert_series_equal(result.middle, expected, check_names=False)
