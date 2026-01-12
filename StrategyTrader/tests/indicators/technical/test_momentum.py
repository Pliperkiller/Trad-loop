"""Tests for momentum indicators."""

import pytest
import numpy as np
import pandas as pd

from src.indicators.technical.momentum import (
    rsi,
    macd,
    stochastic,
    williams_r,
    adx,
)


class TestRSI:
    """Tests for Relative Strength Index."""

    def test_rsi_basic(self, sample_close):
        """RSI calculation produces valid output."""
        result = rsi(sample_close, period=14)

        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_close)

    def test_rsi_range(self, sample_close):
        """RSI values are between 0 and 100."""
        result = rsi(sample_close, period=14)

        valid_values = result.dropna()
        assert (valid_values >= 0).all()
        assert (valid_values <= 100).all()

    def test_rsi_overbought_uptrend(self, trending_up_close):
        """RSI is high in strong uptrend."""
        result = rsi(trending_up_close, period=14)

        # In strong uptrend, RSI should be high (>50)
        valid_values = result.iloc[20:].dropna()
        mean_rsi = valid_values.mean()
        assert mean_rsi > 50

    def test_rsi_oversold_downtrend(self, trending_down_close):
        """RSI is low in strong downtrend."""
        result = rsi(trending_down_close, period=14)

        valid_values = result.iloc[20:].dropna()
        mean_rsi = valid_values.mean()
        assert mean_rsi < 50

    def test_rsi_neutral_sideways(self, sideways_close):
        """RSI is around 50 in sideways market."""
        result = rsi(sideways_close, period=14)

        valid_values = result.iloc[20:].dropna()
        mean_rsi = valid_values.mean()
        assert 30 < mean_rsi < 70

    def test_rsi_all_up(self):
        """RSI is 100 when all moves are up."""
        data = pd.Series([100 + i for i in range(50)])
        result = rsi(data, period=14)

        # Last values should be 100 (or very close due to smoothing)
        assert result.iloc[-1] > 95


class TestMACD:
    """Tests for MACD."""

    def test_macd_basic(self, sample_close):
        """MACD calculation produces valid output."""
        result = macd(sample_close)

        assert hasattr(result, "macd_line")
        assert hasattr(result, "signal_line")
        assert hasattr(result, "histogram")

        assert len(result.macd_line) == len(sample_close)
        assert len(result.signal_line) == len(sample_close)
        assert len(result.histogram) == len(sample_close)

    def test_macd_histogram_calculation(self, sample_close):
        """MACD histogram equals macd_line - signal_line."""
        result = macd(sample_close)

        expected_histogram = result.macd_line - result.signal_line
        pd.testing.assert_series_equal(
            result.histogram, expected_histogram, check_names=False
        )

    def test_macd_uptrend_positive(self, trending_up_close):
        """MACD line is positive in uptrend."""
        result = macd(trending_up_close)

        # In uptrend, fast EMA > slow EMA, so MACD > 0
        valid_values = result.macd_line.iloc[30:].dropna()
        positive_pct = (valid_values > 0).sum() / len(valid_values)
        assert positive_pct > 0.5

    def test_macd_custom_periods(self, sample_close):
        """MACD works with custom periods."""
        result = macd(sample_close, fast_period=8, slow_period=21, signal_period=5)

        assert result.macd_line.notna().any()

    def test_macd_invalid_periods(self, sample_close):
        """MACD raises error when fast >= slow."""
        with pytest.raises(ValueError):
            macd(sample_close, fast_period=26, slow_period=12)


class TestStochastic:
    """Tests for Stochastic Oscillator."""

    def test_stochastic_basic(self, sample_ohlcv):
        """Stochastic calculation produces valid output."""
        result = stochastic(
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"]
        )

        assert hasattr(result, "k")
        assert hasattr(result, "d")
        assert len(result.k) == len(sample_ohlcv)
        assert len(result.d) == len(sample_ohlcv)

    def test_stochastic_range(self, sample_ohlcv):
        """Stochastic values are between 0 and 100."""
        result = stochastic(
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"]
        )

        valid_k = result.k.dropna()
        valid_d = result.d.dropna()

        assert (valid_k >= 0).all()
        assert (valid_k <= 100).all()
        assert (valid_d >= 0).all()
        assert (valid_d <= 100).all()

    def test_stochastic_overbought_uptrend(self, trending_up_close):
        """Stochastic is high in uptrend."""
        high = trending_up_close * 1.01
        low = trending_up_close * 0.99

        result = stochastic(high, low, trending_up_close)

        valid_k = result.k.iloc[20:].dropna()
        assert valid_k.mean() > 50

    def test_stochastic_d_smoother_than_k(self, sample_ohlcv):
        """%D is smoother than %K."""
        result = stochastic(
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"]
        )

        k_std = result.k.dropna().std()
        d_std = result.d.dropna().std()

        # %D should have lower volatility
        assert d_std <= k_std


class TestWilliamsR:
    """Tests for Williams %R."""

    def test_williams_r_basic(self, sample_ohlcv):
        """Williams %R produces valid output."""
        result = williams_r(
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"]
        )

        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_ohlcv)

    def test_williams_r_range(self, sample_ohlcv):
        """Williams %R values are between -100 and 0."""
        result = williams_r(
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"]
        )

        valid_values = result.dropna()
        assert (valid_values >= -100).all()
        assert (valid_values <= 0).all()

    def test_williams_r_inverse_of_stochastic(self, sample_ohlcv):
        """Williams %R = -100 + %K (approximately)."""
        wr = williams_r(
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"],
            period=14
        )

        stoch = stochastic(
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"],
            k_period=14,
            smooth_k=1,
            d_period=1
        )

        # Williams %R should be approximately -(100 - %K) = %K - 100
        expected = stoch.k - 100
        valid_idx = ~wr.isna() & ~expected.isna()

        diff = abs(wr[valid_idx] - expected[valid_idx])
        assert (diff < 1.0).all()


class TestADX:
    """Tests for Average Directional Index."""

    def test_adx_basic(self, sample_ohlcv):
        """ADX produces valid output."""
        result = adx(
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"]
        )

        assert hasattr(result, "adx")
        assert hasattr(result, "plus_di")
        assert hasattr(result, "minus_di")
        assert len(result.adx) == len(sample_ohlcv)

    def test_adx_positive_values(self, sample_ohlcv):
        """ADX values are positive."""
        result = adx(
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"]
        )

        valid_adx = result.adx.dropna()
        assert (valid_adx >= 0).all()

    def test_adx_high_in_trend(self, trending_up_close):
        """ADX is high in trending market."""
        high = trending_up_close * 1.01
        low = trending_up_close * 0.99

        result = adx(high, low, trending_up_close, period=14)

        valid_adx = result.adx.iloc[30:].dropna()
        # ADX > 25 indicates trending market
        assert valid_adx.mean() > 20

    def test_adx_plus_di_greater_in_uptrend(self, trending_up_close):
        """In uptrend, +DI > -DI."""
        high = trending_up_close * 1.01
        low = trending_up_close * 0.99

        result = adx(high, low, trending_up_close)

        # In strong uptrend, +DI should be greater than -DI
        valid_idx = ~result.plus_di.isna() & ~result.minus_di.isna()
        plus_greater = (result.plus_di[valid_idx] > result.minus_di[valid_idx]).sum()
        assert plus_greater > len(result.plus_di[valid_idx]) / 2
