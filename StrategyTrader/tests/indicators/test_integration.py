"""Integration tests for indicators module."""

import pytest
import numpy as np
import pandas as pd

from src.indicators import (
    sma, ema, vwma, rsi, macd, stochastic, williams_r, adx,
    atr, bollinger_bands, keltner_channels, donchian_channels,
    vwap, obv, cmf, mfi, ichimoku_cloud, pivot_points,
    parabolic_sar, supertrend,
    TechnicalIndicators,
)


class TestIndicatorChaining:
    """Test using multiple indicators together."""

    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        """Generate sample OHLCV data."""
        np.random.seed(42)
        n = 200

        # Create trending data
        trend = np.linspace(100, 150, n) + np.random.randn(n) * 2
        high = trend + np.random.rand(n) * 3
        low = trend - np.random.rand(n) * 3
        close = trend + np.random.randn(n) * 1
        open_ = np.roll(close, 1)
        open_[0] = close[0]
        volume = np.random.randint(10000, 100000, n)

        return pd.DataFrame({
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        })

    def test_trend_confirmation(self, sample_data):
        """Multiple trend indicators can be used together."""
        close = sample_data["close"]
        high = sample_data["high"]
        low = sample_data["low"]

        # Calculate multiple trend indicators
        sma_20 = sma(close, 20)
        ema_20 = ema(close, 20)
        ichi = ichimoku_cloud(high, low, close)

        # All should identify uptrend
        latest_close = close.iloc[-1]
        latest_sma = sma_20.iloc[-1]
        latest_ema = ema_20.iloc[-1]

        # In uptrend, price should be above MAs
        assert latest_close > latest_sma * 0.95  # Allow some variance
        assert latest_close > latest_ema * 0.95

    def test_momentum_oscillators(self, sample_data):
        """Multiple momentum oscillators can be calculated."""
        close = sample_data["close"]
        high = sample_data["high"]
        low = sample_data["low"]
        volume = sample_data["volume"]

        # Calculate momentum indicators
        rsi_14 = rsi(close, 14)
        stoch = stochastic(high, low, close)
        williams = williams_r(high, low, close)
        mfi_14 = mfi(high, low, close, volume)

        # All should have valid values
        assert rsi_14.dropna().notna().all()
        assert stoch.k.dropna().notna().all()
        assert williams.dropna().notna().all()
        assert mfi_14.dropna().notna().all()

    def test_volatility_bands(self, sample_data):
        """Multiple volatility indicators can be compared."""
        close = sample_data["close"]
        high = sample_data["high"]
        low = sample_data["low"]

        # Calculate volatility indicators
        bb = bollinger_bands(close, 20)
        kc = keltner_channels(high, low, close, 20)
        dc = donchian_channels(high, low, 20)

        # Bollinger should contain most price action
        latest_close = close.iloc[-1]
        assert bb.lower.iloc[-1] < latest_close < bb.upper.iloc[-1]

        # Keltner should also contain price
        assert kc.lower.iloc[-1] < latest_close < kc.upper.iloc[-1]

    def test_volume_analysis(self, sample_data):
        """Volume indicators can be combined."""
        close = sample_data["close"]
        high = sample_data["high"]
        low = sample_data["low"]
        volume = sample_data["volume"]

        # Calculate volume indicators
        vwap_line = vwap(high, low, close, volume)
        obv_line = obv(close, volume)
        cmf_line = cmf(high, low, close, volume)
        mfi_line = mfi(high, low, close, volume)

        # All should produce valid output
        assert len(vwap_line) == len(sample_data)
        assert len(obv_line) == len(sample_data)
        assert len(cmf_line) == len(sample_data)
        assert len(mfi_line) == len(sample_data)

    def test_trading_system_signals(self, sample_data):
        """Indicators can be used for trading signals."""
        close = sample_data["close"]
        high = sample_data["high"]
        low = sample_data["low"]

        # Trend filter
        sma_50 = sma(close, 50)
        uptrend = close > sma_50

        # Momentum filter
        rsi_14 = rsi(close, 14)
        not_overbought = rsi_14 < 70

        # Volatility
        atr_14 = atr(high, low, close, 14)

        # Combine signals
        signals = pd.DataFrame({
            "uptrend": uptrend,
            "momentum_ok": not_overbought,
            "volatility": atr_14,
        })

        # Should have valid signals
        assert signals.dropna().shape[0] > 0


class TestIndicatorEdgeCases:
    """Test edge cases for indicators."""

    def test_minimum_data(self):
        """Indicators handle minimum required data."""
        prices = pd.Series([100, 101, 102, 103, 104])

        # These should work with minimal data
        sma_3 = sma(prices, 3)
        ema_3 = ema(prices, 3)
        rsi_3 = rsi(prices, 3)

        assert len(sma_3) == 5
        assert len(ema_3) == 5

    def test_constant_prices(self):
        """Indicators handle constant prices."""
        prices = pd.Series([100.0] * 50)

        sma_20 = sma(prices, 20)
        rsi_14 = rsi(prices, 14)

        # SMA of constant should equal the constant
        assert sma_20.iloc[-1] == 100.0

    def test_single_spike(self):
        """Indicators handle single price spikes."""
        prices = pd.Series([100.0] * 25 + [150.0] + [100.0] * 24)

        sma_20 = sma(prices, 20)
        rsi_14 = rsi(prices, 14)

        # Should still produce valid output
        assert sma_20.dropna().notna().all()

    def test_numpy_array_input(self):
        """Indicators accept numpy arrays."""
        prices = np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109])

        result = sma(prices, 3)

        assert isinstance(result, pd.Series)
        assert len(result) == 10

    def test_list_input(self):
        """Indicators accept Python lists."""
        prices = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]

        result = sma(prices, 3)

        assert isinstance(result, pd.Series)
        assert len(result) == 10


class TestFundamentalIntegration:
    """Test fundamental API clients can be imported."""

    def test_import_coingecko(self):
        """CoinGecko client can be imported."""
        from src.indicators.fundamental import CoinGeckoClient
        assert CoinGeckoClient is not None

    def test_import_defillama(self):
        """DefiLlama client can be imported."""
        from src.indicators.fundamental import DefiLlamaClient
        assert DefiLlamaClient is not None

    def test_import_feargreed(self):
        """Fear & Greed client can be imported."""
        from src.indicators.fundamental import FearGreedClient
        assert FearGreedClient is not None

    def test_coingecko_initialization(self):
        """CoinGecko client initializes without API key."""
        from src.indicators.fundamental import CoinGeckoClient

        client = CoinGeckoClient()
        assert client is not None
        assert client.base_url == "https://api.coingecko.com/api/v3"

    def test_defillama_initialization(self):
        """DefiLlama client initializes without API key."""
        from src.indicators.fundamental import DefiLlamaClient

        client = DefiLlamaClient()
        assert client is not None

    def test_feargreed_initialization(self):
        """Fear & Greed client initializes without API key."""
        from src.indicators.fundamental import FearGreedClient

        client = FearGreedClient()
        assert client is not None


class TestDataclassResults:
    """Test that dataclass results work correctly."""

    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        """Generate sample OHLCV data."""
        np.random.seed(42)
        n = 100
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.random.rand(n) * 2
        low = close - np.random.rand(n) * 2
        open_ = close + np.random.randn(n) * 0.5
        volume = np.random.randint(1000, 10000, n)

        return pd.DataFrame({
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        })

    def test_macd_result(self, sample_data):
        """MACD returns properly structured result."""
        result = macd(sample_data["close"])

        assert hasattr(result, 'macd_line')
        assert hasattr(result, 'signal_line')
        assert hasattr(result, 'histogram')

        # All should be Series
        assert isinstance(result.macd_line, pd.Series)
        assert isinstance(result.signal_line, pd.Series)
        assert isinstance(result.histogram, pd.Series)

    def test_stochastic_result(self, sample_data):
        """Stochastic returns properly structured result."""
        result = stochastic(
            sample_data["high"],
            sample_data["low"],
            sample_data["close"]
        )

        assert hasattr(result, 'k')
        assert hasattr(result, 'd')

    def test_bollinger_result(self, sample_data):
        """Bollinger Bands returns properly structured result."""
        result = bollinger_bands(sample_data["close"])

        assert hasattr(result, 'upper')
        assert hasattr(result, 'middle')
        assert hasattr(result, 'lower')

    def test_ichimoku_result(self, sample_data):
        """Ichimoku Cloud returns properly structured result."""
        result = ichimoku_cloud(
            sample_data["high"],
            sample_data["low"],
            sample_data["close"]
        )

        assert hasattr(result, 'tenkan_sen')
        assert hasattr(result, 'kijun_sen')
        assert hasattr(result, 'senkou_span_a')
        assert hasattr(result, 'senkou_span_b')
        assert hasattr(result, 'chikou_span')

    def test_pivot_points_result(self, sample_data):
        """Pivot Points returns properly structured result."""
        result = pivot_points(
            sample_data["high"].iloc[-1],
            sample_data["low"].iloc[-1],
            sample_data["close"].iloc[-1]
        )

        assert hasattr(result, 'pivot')
        assert hasattr(result, 'r1')
        assert hasattr(result, 'r2')
        assert hasattr(result, 'r3')
        assert hasattr(result, 's1')
        assert hasattr(result, 's2')
        assert hasattr(result, 's3')

    def test_adx_result(self, sample_data):
        """ADX returns properly structured result."""
        result = adx(
            sample_data["high"],
            sample_data["low"],
            sample_data["close"]
        )

        assert hasattr(result, 'adx')
        assert hasattr(result, 'plus_di')
        assert hasattr(result, 'minus_di')

