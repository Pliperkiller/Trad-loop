"""Tests for backward compatibility with existing TechnicalIndicators usage."""

import pytest
import numpy as np
import pandas as pd

from src.indicators import TechnicalIndicators


class TestBackwardCompatibility:
    """Tests ensuring backward compatibility with existing code."""

    @pytest.fixture
    def sample_prices(self) -> pd.Series:
        """Sample price data."""
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
        return pd.Series(prices)

    @pytest.fixture
    def sample_ohlcv(self) -> pd.DataFrame:
        """Sample OHLCV data."""
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

    def test_sma_exists(self):
        """SMA is accessible as static method."""
        assert hasattr(TechnicalIndicators, 'sma')
        assert callable(TechnicalIndicators.sma)

    def test_ema_exists(self):
        """EMA is accessible as static method."""
        assert hasattr(TechnicalIndicators, 'ema')
        assert callable(TechnicalIndicators.ema)

    def test_rsi_exists(self):
        """RSI is accessible as static method."""
        assert hasattr(TechnicalIndicators, 'rsi')
        assert callable(TechnicalIndicators.rsi)

    def test_macd_exists(self):
        """MACD is accessible as static method."""
        assert hasattr(TechnicalIndicators, 'macd')
        assert callable(TechnicalIndicators.macd)

    def test_bollinger_bands_exists(self):
        """Bollinger Bands is accessible as static method."""
        assert hasattr(TechnicalIndicators, 'bollinger_bands')
        assert callable(TechnicalIndicators.bollinger_bands)

    def test_atr_exists(self):
        """ATR is accessible as static method."""
        assert hasattr(TechnicalIndicators, 'atr')
        assert callable(TechnicalIndicators.atr)

    def test_stochastic_exists(self):
        """Stochastic is accessible as static method."""
        assert hasattr(TechnicalIndicators, 'stochastic')
        assert callable(TechnicalIndicators.stochastic)

    def test_sma_works(self, sample_prices):
        """SMA produces expected output."""
        result = TechnicalIndicators.sma(sample_prices, period=20)

        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_prices)
        # First 19 values should be NaN
        assert result.iloc[:19].isna().all()
        # Rest should be valid
        assert result.iloc[19:].notna().all()

    def test_ema_works(self, sample_prices):
        """EMA produces expected output."""
        result = TechnicalIndicators.ema(sample_prices, period=20)

        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_prices)

    def test_rsi_works(self, sample_prices):
        """RSI produces expected output."""
        result = TechnicalIndicators.rsi(sample_prices, period=14)

        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_prices)
        # RSI should be between 0 and 100
        valid_values = result.dropna()
        assert (valid_values >= 0).all()
        assert (valid_values <= 100).all()

    def test_macd_works(self, sample_prices):
        """MACD produces expected output."""
        result = TechnicalIndicators.macd(sample_prices)

        # Should return a result with macd_line, signal_line, histogram
        assert hasattr(result, 'macd_line')
        assert hasattr(result, 'signal_line')
        assert hasattr(result, 'histogram')

    def test_bollinger_bands_works(self, sample_prices):
        """Bollinger Bands produces expected output."""
        result = TechnicalIndicators.bollinger_bands(sample_prices, period=20)

        # Should return result with upper, middle, lower
        assert hasattr(result, 'upper')
        assert hasattr(result, 'middle')
        assert hasattr(result, 'lower')

    def test_atr_works(self, sample_ohlcv):
        """ATR produces expected output."""
        result = TechnicalIndicators.atr(
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"],
            period=14
        )

        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_ohlcv)

    def test_stochastic_works(self, sample_ohlcv):
        """Stochastic produces expected output."""
        result = TechnicalIndicators.stochastic(
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"]
        )

        # Should return result with k and d
        assert hasattr(result, 'k')
        assert hasattr(result, 'd')

    # Test new indicators are also accessible
    def test_vwma_accessible(self):
        """VWMA is accessible."""
        assert hasattr(TechnicalIndicators, 'vwma')

    def test_williams_r_accessible(self):
        """Williams %R is accessible."""
        assert hasattr(TechnicalIndicators, 'williams_r')

    def test_adx_accessible(self):
        """ADX is accessible."""
        assert hasattr(TechnicalIndicators, 'adx')

    def test_vwap_accessible(self):
        """VWAP is accessible."""
        assert hasattr(TechnicalIndicators, 'vwap')

    def test_obv_accessible(self):
        """OBV is accessible."""
        assert hasattr(TechnicalIndicators, 'obv')

    def test_parabolic_sar_accessible(self):
        """Parabolic SAR is accessible."""
        assert hasattr(TechnicalIndicators, 'parabolic_sar')

    def test_supertrend_accessible(self):
        """Supertrend is accessible."""
        assert hasattr(TechnicalIndicators, 'supertrend')

    def test_ichimoku_cloud_accessible(self):
        """Ichimoku Cloud is accessible."""
        assert hasattr(TechnicalIndicators, 'ichimoku_cloud')

    def test_pivot_points_accessible(self):
        """Pivot Points is accessible."""
        assert hasattr(TechnicalIndicators, 'pivot_points')

    def test_keltner_channels_accessible(self):
        """Keltner Channels is accessible."""
        assert hasattr(TechnicalIndicators, 'keltner_channels')

    def test_donchian_channels_accessible(self):
        """Donchian Channels is accessible."""
        assert hasattr(TechnicalIndicators, 'donchian_channels')

    def test_cmf_accessible(self):
        """CMF is accessible."""
        assert hasattr(TechnicalIndicators, 'cmf')

    def test_mfi_accessible(self):
        """MFI is accessible."""
        assert hasattr(TechnicalIndicators, 'mfi')


class TestDirectImports:
    """Test that indicators can be imported directly."""

    def test_import_sma(self):
        """SMA can be imported directly."""
        from src.indicators import sma
        assert callable(sma)

    def test_import_ema(self):
        """EMA can be imported directly."""
        from src.indicators import ema
        assert callable(ema)

    def test_import_rsi(self):
        """RSI can be imported directly."""
        from src.indicators import rsi
        assert callable(rsi)

    def test_import_macd(self):
        """MACD can be imported directly."""
        from src.indicators import macd
        assert callable(macd)

    def test_import_ichimoku(self):
        """Ichimoku can be imported directly."""
        from src.indicators import ichimoku_cloud
        assert callable(ichimoku_cloud)

    def test_import_pivot_points(self):
        """Pivot Points can be imported directly."""
        from src.indicators import pivot_points
        assert callable(pivot_points)

    def test_import_vwap(self):
        """VWAP can be imported directly."""
        from src.indicators import vwap
        assert callable(vwap)

    def test_import_adx(self):
        """ADX can be imported directly."""
        from src.indicators import adx
        assert callable(adx)

