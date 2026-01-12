"""
Tests para models.py
"""

import pytest
from datetime import datetime
import sys
from pathlib import Path

src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from paper_trading.models import (
    Order, OrderSide, OrderType, OrderStatus,
    OrderResult, PaperPosition, PositionSide,
    TradeRecord, RealtimeCandle, PaperTradingState
)


class TestOrder:
    """Tests para la clase Order"""

    def test_create_market_order(self, sample_order):
        """Test crear orden de mercado"""
        assert sample_order.symbol == "BTC/USDT"
        assert sample_order.side == OrderSide.BUY
        assert sample_order.type == OrderType.MARKET
        assert sample_order.quantity == 0.1
        assert sample_order.status == OrderStatus.PENDING
        assert sample_order.id is not None

    def test_create_limit_order(self, sample_limit_order):
        """Test crear orden limite"""
        assert sample_limit_order.price == 49000
        assert sample_limit_order.type == OrderType.LIMIT

    def test_order_is_active(self, sample_order):
        """Test verificar si orden esta activa"""
        assert sample_order.is_active() == True

        sample_order.status = OrderStatus.FILLED
        assert sample_order.is_active() == False

        sample_order.status = OrderStatus.CANCELLED
        assert sample_order.is_active() == False

    def test_order_is_filled(self, sample_order):
        """Test verificar si orden fue ejecutada"""
        assert sample_order.is_filled() == False

        sample_order.status = OrderStatus.FILLED
        assert sample_order.is_filled() == True

    def test_remaining_quantity(self, sample_order):
        """Test cantidad restante"""
        assert sample_order.remaining_quantity() == 0.1

        sample_order.filled_quantity = 0.05
        assert sample_order.remaining_quantity() == 0.05

    def test_order_to_dict(self, sample_order):
        """Test conversion a diccionario"""
        d = sample_order.to_dict()

        assert d["symbol"] == "BTC/USDT"
        assert d["side"] == "buy"
        assert d["type"] == "market"
        assert d["quantity"] == 0.1


class TestPaperPosition:
    """Tests para la clase PaperPosition"""

    def test_create_position(self, sample_position):
        """Test crear posicion"""
        assert sample_position.symbol == "BTC/USDT"
        assert sample_position.side == PositionSide.LONG
        assert sample_position.entry_price == 50000
        assert sample_position.quantity == 0.1
        assert sample_position.stop_loss == 49000
        assert sample_position.take_profit == 52000

    def test_update_unrealized_pnl_long(self, sample_position):
        """Test actualizar PnL para posicion long"""
        # Precio sube
        pnl = sample_position.update_unrealized_pnl(51000)
        assert pnl == 100  # (51000 - 50000) * 0.1

        # Precio baja
        pnl = sample_position.update_unrealized_pnl(49000)
        assert pnl == -100  # (49000 - 50000) * 0.1

    def test_update_unrealized_pnl_short(self):
        """Test actualizar PnL para posicion short"""
        position = PaperPosition(
            symbol="BTC/USDT",
            side=PositionSide.SHORT,
            entry_price=50000,
            quantity=0.1,
        )

        # Precio baja (ganancia para short)
        pnl = position.update_unrealized_pnl(49000)
        assert pnl == 100  # (50000 - 49000) * 0.1

        # Precio sube (perdida para short)
        pnl = position.update_unrealized_pnl(51000)
        assert pnl == -100

    def test_should_stop_loss_long(self, sample_position):
        """Test trigger de stop loss para long"""
        # Precio por encima del stop
        assert sample_position.should_stop_loss(50000) == False
        assert sample_position.should_stop_loss(49500) == False

        # Precio en o debajo del stop
        assert sample_position.should_stop_loss(49000) == True
        assert sample_position.should_stop_loss(48000) == True

    def test_should_take_profit_long(self, sample_position):
        """Test trigger de take profit para long"""
        # Precio por debajo del TP
        assert sample_position.should_take_profit(51000) == False

        # Precio en o arriba del TP
        assert sample_position.should_take_profit(52000) == True
        assert sample_position.should_take_profit(53000) == True

    def test_should_stop_loss_short(self):
        """Test trigger de stop loss para short"""
        position = PaperPosition(
            symbol="BTC/USDT",
            side=PositionSide.SHORT,
            entry_price=50000,
            quantity=0.1,
            stop_loss=51000,  # Stop arriba para short
        )

        assert position.should_stop_loss(50500) == False
        assert position.should_stop_loss(51000) == True
        assert position.should_stop_loss(52000) == True

    def test_position_to_dict(self, sample_position):
        """Test conversion a diccionario"""
        d = sample_position.to_dict()

        assert d["symbol"] == "BTC/USDT"
        assert d["side"] == "long"
        assert d["entry_price"] == 50000
        assert d["quantity"] == 0.1


class TestRealtimeCandle:
    """Tests para la clase RealtimeCandle"""

    def test_create_candle(self, sample_candle):
        """Test crear vela"""
        assert sample_candle.symbol == "BTC/USDT"
        assert sample_candle.open == 50000
        assert sample_candle.high == 50500
        assert sample_candle.low == 49800
        assert sample_candle.close == 50200
        assert sample_candle.is_closed == True

    def test_update_candle(self, sample_candle):
        """Test actualizar vela con tick"""
        sample_candle.is_closed = False

        # Tick hacia arriba
        sample_candle.update(50600, 10)
        assert sample_candle.close == 50600
        assert sample_candle.high == 50600
        assert sample_candle.low == 49800
        assert sample_candle.trades_count == 1

        # Tick hacia abajo
        sample_candle.update(49700, 5)
        assert sample_candle.close == 49700
        assert sample_candle.low == 49700
        assert sample_candle.high == 50600
        assert sample_candle.trades_count == 2

    def test_candle_to_dict(self, sample_candle):
        """Test conversion a diccionario"""
        d = sample_candle.to_dict()

        assert d["symbol"] == "BTC/USDT"
        assert d["open"] == 50000
        assert d["high"] == 50500
        assert d["low"] == 49800
        assert d["close"] == 50200


class TestPaperTradingState:
    """Tests para la clase PaperTradingState"""

    def test_create_state(self):
        """Test crear estado"""
        state = PaperTradingState()

        assert state.is_running == False
        assert state.balance == 0
        assert state.equity == 0
        assert state.total_trades == 0

    def test_win_rate_calculation(self):
        """Test calculo de win rate"""
        state = PaperTradingState()

        # Sin trades
        assert state.win_rate == 0

        # Con trades
        state.total_trades = 10
        state.winning_trades = 6
        assert state.win_rate == 60.0

    def test_state_to_dict(self):
        """Test conversion a diccionario"""
        state = PaperTradingState(
            is_running=True,
            symbol="BTC/USDT",
            balance=10000,
            equity=10500,
            total_trades=5,
            winning_trades=3,
        )

        d = state.to_dict()

        assert d["is_running"] == True
        assert d["symbol"] == "BTC/USDT"
        assert d["balance"] == 10000
        assert d["equity"] == 10500
        assert d["win_rate"] == 60.0


class TestTradeRecord:
    """Tests para la clase TradeRecord"""

    def test_create_trade(self, sample_trade):
        """Test crear registro de trade"""
        assert sample_trade.symbol == "BTC/USDT"
        assert sample_trade.side == PositionSide.LONG
        assert sample_trade.entry_price == 50000
        assert sample_trade.exit_price == 51000
        assert sample_trade.pnl == 100
        assert sample_trade.exit_reason == "Take Profit"

    def test_trade_to_dict(self, sample_trade):
        """Test conversion a diccionario"""
        d = sample_trade.to_dict()

        assert d["symbol"] == "BTC/USDT"
        assert d["side"] == "long"
        assert d["entry_price"] == 50000
        assert d["exit_price"] == 51000
        assert d["pnl"] == 100
