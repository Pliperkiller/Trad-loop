"""Fixtures for optimizer tests."""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta


@pytest.fixture
def sample_time_series():
    """Generate sample time series data for testing."""
    n = 500
    dates = pd.date_range(start='2020-01-01', periods=n, freq='D')
    np.random.seed(42)

    # Generate realistic OHLCV data
    close = 100 + np.cumsum(np.random.randn(n) * 2)
    high = close + np.abs(np.random.randn(n)) * 2
    low = close - np.abs(np.random.randn(n)) * 2
    open_ = close + np.random.randn(n) * 0.5
    volume = np.random.randint(1000, 10000, n)

    return pd.DataFrame({
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume,
    }, index=dates)


@pytest.fixture
def small_time_series():
    """Generate small time series for edge case testing."""
    n = 100
    dates = pd.date_range(start='2020-01-01', periods=n, freq='D')
    np.random.seed(42)

    close = 100 + np.cumsum(np.random.randn(n) * 1)

    return pd.DataFrame({
        'close': close,
        'volume': np.random.randint(1000, 5000, n),
    }, index=dates)


@pytest.fixture
def sample_returns():
    """Generate sample returns series."""
    np.random.seed(42)
    n = 252  # One year of daily returns
    returns = np.random.randn(n) * 0.02  # ~2% daily volatility
    return returns


@pytest.fixture
def sample_is_oos_sharpes():
    """Sample IS and OOS Sharpe ratios for testing."""
    return {
        'is_sharpes': [1.5, 1.2, 1.8, 1.0, 1.3, 1.6, 1.1, 1.4],
        'oos_sharpes': [0.8, 0.5, 0.9, 0.3, 0.6, 0.7, 0.4, 0.5],
    }


@pytest.fixture
def sample_split_results():
    """Sample split results for testing."""
    from src.optimizers.validation.results import SplitResult

    base_date = datetime(2020, 1, 1)

    return [
        SplitResult(
            split_idx=0,
            train_start=base_date,
            train_end=base_date + timedelta(days=100),
            test_start=base_date + timedelta(days=100),
            test_end=base_date + timedelta(days=120),
            train_rows=100,
            test_rows=20,
            best_params={'period': 20, 'threshold': 0.5},
            train_score=1.5,
            test_score=1.0,
            degradation_pct=33.3,
            train_metrics={'sharpe': 1.5, 'return': 0.15},
            test_metrics={'sharpe': 1.0, 'return': 0.08},
        ),
        SplitResult(
            split_idx=1,
            train_start=base_date,
            train_end=base_date + timedelta(days=120),
            test_start=base_date + timedelta(days=120),
            test_end=base_date + timedelta(days=140),
            train_rows=120,
            test_rows=20,
            best_params={'period': 22, 'threshold': 0.45},
            train_score=1.8,
            test_score=1.2,
            degradation_pct=33.3,
            train_metrics={'sharpe': 1.8, 'return': 0.18},
            test_metrics={'sharpe': 1.2, 'return': 0.10},
        ),
        SplitResult(
            split_idx=2,
            train_start=base_date,
            train_end=base_date + timedelta(days=140),
            test_start=base_date + timedelta(days=140),
            test_end=base_date + timedelta(days=160),
            train_rows=140,
            test_rows=20,
            best_params={'period': 18, 'threshold': 0.55},
            train_score=1.4,
            test_score=0.8,
            degradation_pct=42.9,
            train_metrics={'sharpe': 1.4, 'return': 0.12},
            test_metrics={'sharpe': 0.8, 'return': 0.05},
        ),
    ]


@pytest.fixture
def sample_params_list():
    """Sample list of parameter dictionaries across splits."""
    return [
        {'period': 20, 'threshold': 0.5, 'multiplier': 2.0},
        {'period': 22, 'threshold': 0.45, 'multiplier': 2.1},
        {'period': 18, 'threshold': 0.55, 'multiplier': 1.9},
        {'period': 21, 'threshold': 0.48, 'multiplier': 2.0},
        {'period': 19, 'threshold': 0.52, 'multiplier': 2.05},
    ]
