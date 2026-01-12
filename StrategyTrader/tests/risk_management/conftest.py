"""
Fixtures compartidas para tests del modulo risk_management
"""

import pytest
import sys
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np

# Agregar src al path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from risk_management.config import (
    RiskManagementConfig,
    PositionSizingConfig,
    ExposureLimitsConfig,
    DrawdownConfig,
    CorrelationConfig,
    VaRConfig,
)
from risk_management.models import SizingMethod
from risk_management.position_sizer import TradeHistory
from risk_management.risk_limits import Position


@pytest.fixture
def default_config():
    """Configuracion por defecto para tests"""
    return RiskManagementConfig(
        initial_capital=10000.0,
        max_open_positions=5,
    )


@pytest.fixture
def conservative_config():
    """Configuracion conservadora"""
    return RiskManagementConfig(
        initial_capital=10000.0,
        max_open_positions=3,
        position_sizing=PositionSizingConfig(
            method=SizingMethod.FIXED_FRACTIONAL,
            risk_per_trade=0.01,
            max_position_percent=0.10,
        ),
        exposure_limits=ExposureLimitsConfig(
            max_single_asset_exposure=0.15,
            max_total_exposure=0.50,
        ),
        drawdown=DrawdownConfig(
            warning_level=0.03,
            caution_level=0.05,
            danger_level=0.08,
            critical_level=0.10,
        ),
    )


@pytest.fixture
def aggressive_config():
    """Configuracion agresiva"""
    return RiskManagementConfig(
        initial_capital=10000.0,
        max_open_positions=10,
        position_sizing=PositionSizingConfig(
            method=SizingMethod.KELLY,
            risk_per_trade=0.03,
            max_position_percent=0.30,
            kelly_fraction=0.5,
        ),
        exposure_limits=ExposureLimitsConfig(
            max_single_asset_exposure=0.35,
            max_total_exposure=1.0,
        ),
        drawdown=DrawdownConfig(
            warning_level=0.10,
            caution_level=0.15,
            danger_level=0.20,
            critical_level=0.25,
        ),
    )


@pytest.fixture
def sample_trade_history():
    """Historial de trades de ejemplo"""
    returns = [
        0.02, 0.03, -0.01, 0.015, -0.02,
        0.025, 0.01, -0.015, 0.02, 0.03,
        -0.01, 0.02, 0.015, -0.005, 0.025,
        0.02, -0.01, 0.03, 0.015, -0.02,
        0.025, 0.02, -0.015, 0.01, 0.03,
        0.02, -0.01, 0.015, 0.025, -0.02,
    ]
    wins = sum(1 for r in returns if r > 0)
    losses = sum(1 for r in returns if r < 0)
    return TradeHistory(returns=returns, wins=wins, losses=losses)


@pytest.fixture
def sample_positions():
    """Posiciones de ejemplo"""
    return [
        Position(
            symbol="BTC/USDT",
            side="long",
            size=0.04,
            entry_price=50000,
            current_price=51000,
            sector="crypto_major",
        ),
        Position(
            symbol="ETH/USDT",
            side="long",
            size=0.5,
            entry_price=3000,
            current_price=3100,
            sector="crypto_major",
        ),
    ]


@pytest.fixture
def sample_returns_data():
    """Retornos de ejemplo para correlacion"""
    np.random.seed(42)
    base_returns = np.random.normal(0, 0.02, 60)

    return {
        "BTC/USDT": list(base_returns),
        "ETH/USDT": list(base_returns * 1.2 + np.random.normal(0, 0.005, 60)),  # Correlacionado
        "LINK/USDT": list(np.random.normal(0, 0.03, 60)),  # Independiente
        "XRP/USDT": list(-base_returns * 0.5 + np.random.normal(0, 0.01, 60)),  # Inverso
    }


@pytest.fixture
def var_returns():
    """Retornos para calculo de VaR"""
    np.random.seed(42)
    # Simular retornos con algunas perdidas extremas
    returns = list(np.random.normal(0.001, 0.02, 250))
    # Agregar algunas perdidas extremas
    returns[50] = -0.05
    returns[100] = -0.08
    returns[150] = -0.04
    returns[200] = -0.06
    return returns
