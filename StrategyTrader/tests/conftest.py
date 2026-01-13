"""
Fixtures compartidas para todos los tests
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


@pytest.fixture
def sample_ohlcv_data():
    """Genera datos OHLCV de ejemplo para testing"""
    np.random.seed(42)
    n_bars = 500

    dates = pd.date_range(start='2024-01-01', periods=n_bars, freq='1h')

    # Generar precios con tendencia y volatilidad
    base_price = 100
    returns = np.random.normal(0.0002, 0.02, n_bars)
    close_prices = base_price * np.cumprod(1 + returns)

    # Generar OHLC realista
    high_prices = close_prices * (1 + np.abs(np.random.normal(0, 0.01, n_bars)))
    low_prices = close_prices * (1 - np.abs(np.random.normal(0, 0.01, n_bars)))
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = base_price

    volume = np.random.uniform(1000, 10000, n_bars)

    data = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume
    }, index=dates)

    return data


@pytest.fixture
def trending_data():
    """Genera datos con tendencia alcista clara"""
    np.random.seed(123)
    n_bars = 300

    dates = pd.date_range(start='2024-01-01', periods=n_bars, freq='1h')

    # Tendencia alcista con algo de ruido
    trend = np.linspace(100, 150, n_bars)
    noise = np.random.normal(0, 2, n_bars)
    close_prices = trend + noise

    high_prices = close_prices * 1.01
    low_prices = close_prices * 0.99
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = 100

    volume = np.random.uniform(1000, 10000, n_bars)

    data = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume
    }, index=dates)

    return data


@pytest.fixture
def sample_strategy_config():
    """Configuracion de estrategia para testing"""
    from src.strategy import StrategyConfig

    return StrategyConfig(
        symbol='TEST/USDT',
        timeframe='1h',
        initial_capital=10000.0,
        risk_per_trade=2.0,
        max_positions=1,
        commission=0.1,
        slippage=0.05
    )


@pytest.fixture
def sample_trades():
    """Trades de ejemplo para testing de performance"""
    trades = [
        {
            'entry_time': datetime(2024, 1, 1, 10, 0),
            'exit_time': datetime(2024, 1, 1, 14, 0),
            'entry_price': 100.0,
            'exit_price': 105.0,
            'quantity': 10.0,
            'pnl': 50.0,
            'return_pct': 5.0,
            'reason': 'Take Profit'
        },
        {
            'entry_time': datetime(2024, 1, 2, 10, 0),
            'exit_time': datetime(2024, 1, 2, 14, 0),
            'entry_price': 105.0,
            'exit_price': 103.0,
            'quantity': 10.0,
            'pnl': -20.0,
            'return_pct': -1.9,
            'reason': 'Stop Loss'
        },
        {
            'entry_time': datetime(2024, 1, 3, 10, 0),
            'exit_time': datetime(2024, 1, 3, 18, 0),
            'entry_price': 103.0,
            'exit_price': 110.0,
            'quantity': 10.0,
            'pnl': 70.0,
            'return_pct': 6.8,
            'reason': 'Take Profit'
        },
        {
            'entry_time': datetime(2024, 1, 4, 10, 0),
            'exit_time': datetime(2024, 1, 4, 12, 0),
            'entry_price': 110.0,
            'exit_price': 108.0,
            'quantity': 10.0,
            'pnl': -20.0,
            'return_pct': -1.8,
            'reason': 'Stop Loss'
        },
        {
            'entry_time': datetime(2024, 1, 5, 10, 0),
            'exit_time': datetime(2024, 1, 5, 16, 0),
            'entry_price': 108.0,
            'exit_price': 115.0,
            'quantity': 10.0,
            'pnl': 70.0,
            'return_pct': 6.5,
            'reason': 'Take Profit'
        },
    ]
    return pd.DataFrame(trades)


@pytest.fixture
def sample_equity_curve():
    """Equity curve de ejemplo para testing"""
    np.random.seed(42)
    n_points = 100
    initial_capital = 10000

    # Generar equity curve con drawdowns
    returns = np.random.normal(0.002, 0.02, n_points)
    equity = initial_capital * np.cumprod(1 + returns)

    return list(equity)
