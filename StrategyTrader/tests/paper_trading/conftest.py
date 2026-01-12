"""
Fixtures compartidas para tests de paper trading
"""

import pytest
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Agregar src al path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from paper_trading.config import PaperTradingConfig
from paper_trading.models import (
    Order, OrderSide, OrderType, OrderStatus,
    PaperPosition, PositionSide, TradeRecord,
    RealtimeCandle, PaperTradingState
)


@pytest.fixture
def default_config():
    """Configuracion por defecto para tests"""
    return PaperTradingConfig(
        initial_balance=10000,
        symbols=["BTC/USDT"],
        commission_rate=0.001,
        max_position_size=0.25,
        max_positions=5,
        risk_per_trade=0.02,
    )


@pytest.fixture
def sample_order():
    """Orden de ejemplo"""
    return Order(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        type=OrderType.MARKET,
        quantity=0.1,
    )


@pytest.fixture
def sample_limit_order():
    """Orden limite de ejemplo"""
    return Order(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        type=OrderType.LIMIT,
        quantity=0.1,
        price=49000,
    )


@pytest.fixture
def sample_position():
    """Posicion de ejemplo"""
    return PaperPosition(
        symbol="BTC/USDT",
        side=PositionSide.LONG,
        entry_price=50000,
        quantity=0.1,
        stop_loss=49000,
        take_profit=52000,
    )


@pytest.fixture
def sample_trade():
    """Trade cerrado de ejemplo"""
    return TradeRecord(
        symbol="BTC/USDT",
        side=PositionSide.LONG,
        entry_price=50000,
        exit_price=51000,
        quantity=0.1,
        entry_time=datetime.now() - timedelta(hours=2),
        exit_time=datetime.now(),
        pnl=100,
        return_pct=2.0,
        commission=1.0,
        exit_reason="Take Profit"
    )


@pytest.fixture
def sample_candle():
    """Vela de ejemplo"""
    return RealtimeCandle(
        timestamp=datetime.now(),
        open=50000,
        high=50500,
        low=49800,
        close=50200,
        volume=100,
        symbol="BTC/USDT",
        timeframe="1m",
        is_closed=True,
    )


@pytest.fixture
def sample_candles():
    """Lista de velas de ejemplo"""
    candles = []
    base_price = 50000
    timestamp = datetime.now() - timedelta(hours=1)

    for i in range(60):
        # Variacion aleatoria
        change = np.random.uniform(-0.001, 0.001)
        base_price *= (1 + change)

        candle = RealtimeCandle(
            timestamp=timestamp + timedelta(minutes=i),
            open=base_price,
            high=base_price * 1.001,
            low=base_price * 0.999,
            close=base_price * (1 + np.random.uniform(-0.0005, 0.0005)),
            volume=np.random.uniform(50, 200),
            symbol="BTC/USDT",
            timeframe="1m",
            is_closed=True,
        )
        candles.append(candle)

    return candles


@pytest.fixture
def sample_trades_list():
    """Lista de trades de ejemplo"""
    trades = []
    base_time = datetime.now() - timedelta(days=30)

    # Crear 20 trades con resultados variados
    for i in range(20):
        # Alternar wins y losses
        is_win = i % 3 != 0  # 2/3 wins

        entry_price = 50000 + np.random.uniform(-1000, 1000)
        if is_win:
            exit_price = entry_price * 1.02  # 2% profit
            pnl = 100
        else:
            exit_price = entry_price * 0.98  # 2% loss
            pnl = -100

        trade = TradeRecord(
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=0.1,
            entry_time=base_time + timedelta(days=i),
            exit_time=base_time + timedelta(days=i, hours=2),
            pnl=pnl,
            return_pct=(pnl / (entry_price * 0.1)) * 100,
            commission=1.0,
            exit_reason="Signal" if is_win else "Stop Loss"
        )
        trades.append(trade)

    return trades
