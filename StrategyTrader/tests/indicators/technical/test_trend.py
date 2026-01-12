"""Tests for trend indicators."""

import pytest
import numpy as np
import pandas as pd

from src.indicators.technical.trend import (
    sma,
    ema,
    vwma,
    parabolic_sar,
    supertrend,
)


class TestSMA:
    """Tests for Simple Moving Average."""

    def test_sma_basic(self, sample_close):
        """SMA calculation is correct."""
        result = sma(sample_close, period=20)

        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_close)
        # First 19 values should be NaN
        assert result.iloc[:19].isna().all()
        # First valid SMA should be mean of first 20 values
        expected = sample_close.iloc[:20].mean()
        assert abs(result.iloc[19] - expected) < 0.01

    def test_sma_period_1(self, sample_close):
        """SMA with period 1 equals input."""
        result = sma(sample_close, period=1)
        pd.testing.assert_series_equal(result, sample_close, check_names=False)

    def test_sma_constant_input(self):
        """SMA of constant series equals constant."""
        data = pd.Series([100.0] * 50)
        result = sma(data, period=10)
        assert (result.iloc[9:] == 100.0).all()

    def test_sma_accepts_list(self):
        """SMA accepts list input."""
        data = [100, 101, 102, 103, 104]
        result = sma(data, period=3)
        assert isinstance(result, pd.Series)
        assert abs(result.iloc[2] - 101.0) < 0.01

    def test_sma_accepts_numpy(self):
        """SMA accepts numpy array input."""
        data = np.array([100, 101, 102, 103, 104])
        result = sma(data, period=3)
        assert isinstance(result, pd.Series)

    def test_sma_invalid_period(self, sample_close):
        """SMA raises error for invalid period."""
        with pytest.raises(ValueError):
            sma(sample_close, period=0)

    def test_sma_empty_input(self):
        """SMA raises error for empty input."""
        with pytest.raises(ValueError):
            sma(pd.Series([]), period=5)


class TestEMA:
    """Tests for Exponential Moving Average."""

    def test_ema_basic(self, sample_close):
        """EMA calculation produces valid output."""
        result = ema(sample_close, period=20)

        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_close)
        assert not result.isna().all()

    def test_ema_reacts_faster_than_sma(self, trending_up_close):
        """EMA reacts faster to price changes than SMA."""
        period = 20
        sma_result = sma(trending_up_close, period)
        ema_result = ema(trending_up_close, period)

        # In uptrend, EMA should be higher than SMA (closer to price)
        valid_idx = ~sma_result.isna()
        assert (ema_result[valid_idx] > sma_result[valid_idx]).sum() > len(ema_result[valid_idx]) / 2

    def test_ema_period_1(self, sample_close):
        """EMA with period 1 equals input."""
        result = ema(sample_close, period=1)
        pd.testing.assert_series_equal(result, sample_close, check_names=False)


class TestVWMA:
    """Tests for Volume Weighted Moving Average."""

    def test_vwma_basic(self, sample_ohlcv):
        """VWMA calculation produces valid output."""
        result = vwma(sample_ohlcv["close"], sample_ohlcv["volume"], period=20)

        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_ohlcv)
        assert result.iloc[19:].notna().all()

    def test_vwma_constant_volume(self, sample_close):
        """VWMA with constant volume equals SMA."""
        volume = pd.Series([1000.0] * len(sample_close), index=sample_close.index)
        vwma_result = vwma(sample_close, volume, period=20)
        sma_result = sma(sample_close, period=20)

        # Should be approximately equal
        valid_idx = ~sma_result.isna()
        diff = abs(vwma_result[valid_idx] - sma_result[valid_idx])
        assert (diff < 0.01).all()

    def test_vwma_high_volume_bias(self):
        """VWMA is biased towards high volume prices."""
        close = pd.Series([100, 100, 100, 200, 100])
        volume = pd.Series([1, 1, 1, 100, 1])  # High volume at price 200

        result = vwma(close, volume, period=5)
        # VWMA should be much closer to 200 than SMA
        assert result.iloc[4] > 150

    def test_vwma_length_mismatch(self):
        """VWMA raises error for length mismatch."""
        close = pd.Series([100, 101, 102])
        volume = pd.Series([1000, 2000])

        with pytest.raises(ValueError):
            vwma(close, volume, period=2)


class TestParabolicSAR:
    """Tests for Parabolic SAR."""

    def test_parabolic_sar_basic(self, sample_ohlcv):
        """Parabolic SAR produces valid output."""
        result = parabolic_sar(
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"]
        )

        assert hasattr(result, "sar")
        assert hasattr(result, "trend")
        assert len(result.sar) == len(sample_ohlcv)
        assert len(result.trend) == len(sample_ohlcv)

    def test_parabolic_sar_trend_values(self, sample_ohlcv):
        """Parabolic SAR trend is 1 or -1."""
        result = parabolic_sar(
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"]
        )

        unique_trends = set(result.trend.unique())
        assert unique_trends.issubset({1, -1})

    def test_parabolic_sar_uptrend(self, trending_up_close):
        """Parabolic SAR detects uptrend."""
        high = trending_up_close * 1.01
        low = trending_up_close * 0.99

        result = parabolic_sar(high, low, trending_up_close)

        # Should mostly be in uptrend
        uptrend_pct = (result.trend == 1).sum() / len(result.trend)
        assert uptrend_pct > 0.5

    def test_parabolic_sar_invalid_af(self, sample_ohlcv):
        """Parabolic SAR raises error for invalid AF."""
        with pytest.raises(ValueError):
            parabolic_sar(
                sample_ohlcv["high"],
                sample_ohlcv["low"],
                sample_ohlcv["close"],
                af_start=-0.02
            )


class TestSupertrend:
    """Tests for Supertrend indicator."""

    def test_supertrend_basic(self, sample_ohlcv):
        """Supertrend produces valid output."""
        result = supertrend(
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"]
        )

        assert hasattr(result, "supertrend")
        assert hasattr(result, "direction")
        assert len(result.supertrend) == len(sample_ohlcv)

    def test_supertrend_direction_values(self, sample_ohlcv):
        """Supertrend direction is 1 or -1 after warmup."""
        result = supertrend(
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"],
            period=10
        )

        # After warmup period, direction should be 1 or -1
        valid_dir = result.direction.iloc[10:].dropna()
        unique_dirs = set(valid_dir.unique())
        assert unique_dirs.issubset({1.0, -1.0})

    def test_supertrend_uptrend_detection(self, trending_up_close):
        """Supertrend detects strong uptrend."""
        high = trending_up_close * 1.01
        low = trending_up_close * 0.99

        result = supertrend(high, low, trending_up_close, period=10, multiplier=3.0)

        valid_dir = result.direction.iloc[10:].dropna()
        uptrend_pct = (valid_dir == 1).sum() / len(valid_dir)
        assert uptrend_pct > 0.5

    def test_supertrend_invalid_multiplier(self, sample_ohlcv):
        """Supertrend raises error for invalid multiplier."""
        with pytest.raises(ValueError):
            supertrend(
                sample_ohlcv["high"],
                sample_ohlcv["low"],
                sample_ohlcv["close"],
                multiplier=-1.0
            )
