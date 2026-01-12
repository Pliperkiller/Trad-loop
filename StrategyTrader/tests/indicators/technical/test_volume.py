"""Tests for volume indicators."""

import pytest
import numpy as np
import pandas as pd

from src.indicators.technical.volume import (
    vwap,
    obv,
    cmf,
    mfi,
)


class TestVWAP:
    """Tests for Volume Weighted Average Price."""

    def test_vwap_basic(self, sample_ohlcv):
        """VWAP produces valid output."""
        result = vwap(
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"],
            sample_ohlcv["volume"]
        )

        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_ohlcv)

    def test_vwap_within_range(self, sample_ohlcv):
        """VWAP is between low and high prices."""
        result = vwap(
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"],
            sample_ohlcv["volume"]
        )

        min_price = sample_ohlcv["low"].min()
        max_price = sample_ohlcv["high"].max()

        valid_vwap = result.dropna()
        assert (valid_vwap >= min_price * 0.9).all()
        assert (valid_vwap <= max_price * 1.1).all()

    def test_vwap_constant_volume(self, sample_ohlcv):
        """VWAP with constant volume is cumulative typical price average."""
        constant_vol = pd.Series([1000.0] * len(sample_ohlcv), index=sample_ohlcv.index)

        result = vwap(
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"],
            constant_vol
        )

        # Should be close to cumulative average of typical price
        typical_price = (sample_ohlcv["high"] + sample_ohlcv["low"] + sample_ohlcv["close"]) / 3
        expected = typical_price.expanding().mean()

        diff = abs(result - expected).dropna()
        assert (diff < 0.01).all()


class TestOBV:
    """Tests for On-Balance Volume."""

    def test_obv_basic(self, sample_ohlcv):
        """OBV produces valid output."""
        result = obv(sample_ohlcv["close"], sample_ohlcv["volume"])

        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_ohlcv)

    def test_obv_cumulative(self, sample_ohlcv):
        """OBV is cumulative."""
        result = obv(sample_ohlcv["close"], sample_ohlcv["volume"])

        # OBV should be cumulative sum of signed volumes
        # Check that it's monotonic when all ups or all downs
        close_up = sample_ohlcv["close"].iloc[:5]
        close_down = sample_ohlcv["close"].iloc[:5].iloc[::-1]
        vol = sample_ohlcv["volume"].iloc[:5]

        obv_up = obv(pd.Series([100, 101, 102, 103, 104]), vol)
        obv_down = obv(pd.Series([104, 103, 102, 101, 100]), vol)

        # Uptrend: OBV should increase
        assert obv_up.iloc[-1] > obv_up.iloc[1]
        # Downtrend: OBV should decrease
        assert obv_down.iloc[-1] < obv_down.iloc[1]

    def test_obv_zero_change(self):
        """OBV doesn't change when price is flat."""
        close = pd.Series([100.0] * 10)
        volume = pd.Series([1000.0] * 10)

        result = obv(close, volume)

        # Should be constant (after first value)
        unique_values = result.iloc[1:].unique()
        assert len(unique_values) == 1


class TestCMF:
    """Tests for Chaikin Money Flow."""

    def test_cmf_basic(self, sample_ohlcv):
        """CMF produces valid output."""
        result = cmf(
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"],
            sample_ohlcv["volume"]
        )

        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_ohlcv)

    def test_cmf_range(self, sample_ohlcv):
        """CMF values are between -1 and 1."""
        result = cmf(
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"],
            sample_ohlcv["volume"]
        )

        valid_values = result.dropna()
        assert (valid_values >= -1.0).all()
        assert (valid_values <= 1.0).all()

    def test_cmf_positive_in_uptrend(self, trending_up_close):
        """CMF is positive in uptrend (buying pressure)."""
        # In uptrend, close tends to be near high (buying pressure)
        high = trending_up_close * 1.02
        low = trending_up_close * 0.98
        # Close is near the high (70% of range from low)
        close = low + (high - low) * 0.7
        volume = pd.Series([1000.0] * len(trending_up_close), index=trending_up_close.index)

        result = cmf(high, low, close, volume)

        valid_values = result.iloc[20:].dropna()
        positive_pct = (valid_values > 0).sum() / len(valid_values)
        assert positive_pct > 0.4

    def test_cmf_high_volume_impact(self):
        """High volume periods have more impact on CMF."""
        high = pd.Series([101, 101, 101, 101, 101])
        low = pd.Series([99, 99, 99, 99, 99])
        close = pd.Series([100, 100, 100, 101, 101])  # Close near high last 2
        volume = pd.Series([100, 100, 100, 1000, 1000])  # High vol last 2

        result = cmf(high, low, close, volume, period=5)

        # CMF should be positive due to high volume near highs
        assert result.iloc[-1] > 0


class TestMFI:
    """Tests for Money Flow Index."""

    def test_mfi_basic(self, sample_ohlcv):
        """MFI produces valid output."""
        result = mfi(
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"],
            sample_ohlcv["volume"]
        )

        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_ohlcv)

    def test_mfi_range(self, sample_ohlcv):
        """MFI values are between 0 and 100."""
        result = mfi(
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"],
            sample_ohlcv["volume"]
        )

        valid_values = result.dropna()
        assert (valid_values >= 0).all()
        assert (valid_values <= 100).all()

    def test_mfi_high_in_uptrend(self, trending_up_close):
        """MFI is high in uptrend."""
        high = trending_up_close * 1.01
        low = trending_up_close * 0.99
        volume = pd.Series([1000.0] * len(trending_up_close), index=trending_up_close.index)

        result = mfi(high, low, trending_up_close, volume)

        valid_values = result.iloc[20:].dropna()
        assert valid_values.mean() > 50

    def test_mfi_similar_to_rsi_concept(self, sample_ohlcv):
        """MFI behaves similarly to RSI (volume-weighted)."""
        result = mfi(
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"],
            sample_ohlcv["volume"],
            period=14
        )

        # MFI should have similar characteristics to RSI
        valid_values = result.dropna()
        assert valid_values.std() > 5  # Should have meaningful variation
        assert 20 < valid_values.mean() < 80  # Should center around middle range
