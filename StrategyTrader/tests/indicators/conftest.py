"""Fixtures for indicators tests."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


@pytest.fixture
def sample_close() -> pd.Series:
    """Sample close price data for testing."""
    np.random.seed(42)
    n = 100
    # Simulate a price series with trend and noise
    trend = np.linspace(100, 120, n)
    noise = np.random.normal(0, 2, n)
    prices = trend + noise

    dates = pd.date_range(start="2024-01-01", periods=n, freq="D")
    return pd.Series(prices, index=dates, name="close")


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """Sample OHLCV data for testing."""
    np.random.seed(42)
    n = 100

    # Generate close prices
    trend = np.linspace(100, 120, n)
    noise = np.random.normal(0, 2, n)
    close = trend + noise

    # Generate high/low around close
    high = close + np.abs(np.random.normal(0, 1, n))
    low = close - np.abs(np.random.normal(0, 1, n))
    open_ = low + (high - low) * np.random.random(n)

    # Generate volume
    volume = np.random.uniform(1000, 10000, n)

    dates = pd.date_range(start="2024-01-01", periods=n, freq="D")

    return pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }, index=dates)


@pytest.fixture
def trending_up_close() -> pd.Series:
    """Strong uptrend price data."""
    np.random.seed(42)
    n = 100
    prices = 100 + np.cumsum(np.abs(np.random.normal(0.5, 0.3, n)))
    dates = pd.date_range(start="2024-01-01", periods=n, freq="D")
    return pd.Series(prices, index=dates, name="close")


@pytest.fixture
def trending_down_close() -> pd.Series:
    """Strong downtrend price data."""
    np.random.seed(42)
    n = 100
    prices = 150 - np.cumsum(np.abs(np.random.normal(0.5, 0.3, n)))
    dates = pd.date_range(start="2024-01-01", periods=n, freq="D")
    return pd.Series(prices, index=dates, name="close")


@pytest.fixture
def sideways_close() -> pd.Series:
    """Sideways/ranging price data."""
    np.random.seed(42)
    n = 100
    prices = 100 + np.random.normal(0, 2, n)
    dates = pd.date_range(start="2024-01-01", periods=n, freq="D")
    return pd.Series(prices, index=dates, name="close")


@pytest.fixture
def high_volatility_close() -> pd.Series:
    """High volatility price data."""
    np.random.seed(42)
    n = 100
    prices = 100 + np.cumsum(np.random.normal(0, 5, n))
    dates = pd.date_range(start="2024-01-01", periods=n, freq="D")
    return pd.Series(prices, index=dates, name="close")


@pytest.fixture
def short_series() -> pd.Series:
    """Short price series for edge case testing."""
    return pd.Series([100, 101, 99, 102, 98], name="close")


@pytest.fixture
def sample_volume() -> pd.Series:
    """Sample volume data."""
    np.random.seed(42)
    n = 100
    volume = np.random.uniform(1000, 10000, n)
    dates = pd.date_range(start="2024-01-01", periods=n, freq="D")
    return pd.Series(volume, index=dates, name="volume")
