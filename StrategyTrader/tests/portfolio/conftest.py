"""
Fixtures compartidas para tests del modulo portfolio
"""

import pytest
import sys
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# Agregar src al path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from portfolio.models import (
    PortfolioConfig,
    PortfolioState,
    PortfolioPosition,
    AllocationMethod,
    RebalanceFrequency,
)


@pytest.fixture
def symbols():
    """Lista de simbolos para tests"""
    return ["BTC/USDT", "ETH/USDT", "SOL/USDT", "LINK/USDT"]


@pytest.fixture
def default_config(symbols):
    """Configuracion por defecto para tests"""
    return PortfolioConfig(
        initial_capital=10000.0,
        symbols=symbols,
        timeframe="1h",
        allocation_method=AllocationMethod.EQUAL_WEIGHT,
        rebalance_frequency=RebalanceFrequency.MONTHLY,
        commission=0.001,
        slippage=0.0005,
    )


@pytest.fixture
def risk_parity_config(symbols):
    """Configuracion con risk parity"""
    return PortfolioConfig(
        initial_capital=10000.0,
        symbols=symbols,
        allocation_method=AllocationMethod.RISK_PARITY,
        rebalance_frequency=RebalanceFrequency.WEEKLY,
    )


@pytest.fixture
def threshold_config(symbols):
    """Configuracion con rebalanceo por threshold"""
    return PortfolioConfig(
        initial_capital=10000.0,
        symbols=symbols,
        allocation_method=AllocationMethod.EQUAL_WEIGHT,
        rebalance_frequency=RebalanceFrequency.THRESHOLD,
        rebalance_threshold=0.05,
    )


@pytest.fixture
def sample_returns(symbols):
    """Retornos de ejemplo para tests"""
    np.random.seed(42)
    n_periods = 252  # 1 ano

    # Generar retornos correlacionados
    base = np.random.normal(0.0005, 0.02, n_periods)

    returns = {
        "BTC/USDT": base + np.random.normal(0, 0.01, n_periods),
        "ETH/USDT": base * 1.2 + np.random.normal(0, 0.015, n_periods),
        "SOL/USDT": base * 0.8 + np.random.normal(0, 0.025, n_periods),
        "LINK/USDT": np.random.normal(0.0003, 0.03, n_periods),  # Independiente
    }

    dates = pd.date_range(start="2024-01-01", periods=n_periods, freq="D")
    return pd.DataFrame(returns, index=dates)


@pytest.fixture
def sample_ohlcv_data(symbols):
    """Datos OHLCV de ejemplo para backtesting"""
    np.random.seed(42)
    n_periods = 500

    data = {}
    dates = pd.date_range(start="2024-01-01", periods=n_periods, freq="1h")

    base_prices = {
        "BTC/USDT": 50000.0,
        "ETH/USDT": 3000.0,
        "SOL/USDT": 100.0,
        "LINK/USDT": 15.0,
    }

    for symbol in symbols:
        base = base_prices.get(symbol, 100.0)
        returns = np.random.normal(0.0001, 0.02, n_periods)
        prices = base * np.exp(np.cumsum(returns))

        df = pd.DataFrame({
            "open": prices * (1 + np.random.uniform(-0.005, 0.005, n_periods)),
            "high": prices * (1 + np.random.uniform(0, 0.02, n_periods)),
            "low": prices * (1 - np.random.uniform(0, 0.02, n_periods)),
            "close": prices,
            "volume": np.random.uniform(1000, 10000, n_periods),
        }, index=dates)

        data[symbol] = df

    return data


@pytest.fixture
def sample_positions():
    """Posiciones de ejemplo"""
    return {
        "BTC/USDT": PortfolioPosition(
            symbol="BTC/USDT",
            quantity=0.1,
            entry_price=50000,
            current_price=51000,
        ),
        "ETH/USDT": PortfolioPosition(
            symbol="ETH/USDT",
            quantity=1.0,
            entry_price=3000,
            current_price=3100,
        ),
    }


@pytest.fixture
def sample_weights():
    """Pesos de ejemplo"""
    return {
        "BTC/USDT": 0.40,
        "ETH/USDT": 0.30,
        "SOL/USDT": 0.20,
        "LINK/USDT": 0.10,
    }


@pytest.fixture
def equal_weights(symbols):
    """Pesos iguales"""
    n = len(symbols)
    return {s: 1.0 / n for s in symbols}


@pytest.fixture
def sample_state(symbols, sample_positions):
    """Estado de portfolio de ejemplo"""
    state = PortfolioState(
        total_equity=10000,
        cash=5000,
        invested_value=5000,
        positions=sample_positions,
        target_weights={s: 0.25 for s in symbols},
    )
    state.update_weights()
    return state


@pytest.fixture
def benchmark_returns():
    """Retornos del benchmark (buy & hold BTC)"""
    np.random.seed(42)
    n_periods = 252
    returns = np.random.normal(0.0005, 0.02, n_periods)
    dates = pd.date_range(start="2024-01-01", periods=n_periods, freq="D")
    return pd.Series(returns, index=dates, name="benchmark")
