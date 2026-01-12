"""
Tests para position_manager.py
"""

import pytest
import sys
from pathlib import Path

src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from paper_trading.config import PaperTradingConfig
from paper_trading.position_manager import PositionManager
from paper_trading.models import PositionSide


class TestPositionManager:
    """Tests para PositionManager"""

    @pytest.fixture
    def manager(self, default_config):
        """Crea manager con configuracion por defecto"""
        return PositionManager(default_config)

    def test_create_manager(self, manager):
        """Test crear manager"""
        assert manager.balance == 10000
        assert manager.equity == 10000
        assert len(manager.positions) == 0
        assert len(manager.trade_history) == 0

    def test_open_long_position(self, manager):
        """Test abrir posicion long"""
        position = manager.open_position(
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            quantity=0.04,  # $2000 < max $2500
            entry_price=50000,
            stop_loss=49000,
            take_profit=52000,
        )

        assert position is not None
        assert position.symbol == "BTC/USDT"
        assert position.side == PositionSide.LONG
        assert position.quantity == 0.04
        assert position.entry_price == 50000
        assert len(manager.positions) == 1

        # Balance deberia reducirse
        assert manager.balance < 10000

    def test_open_short_position(self, manager):
        """Test abrir posicion short"""
        position = manager.open_position(
            symbol="BTC/USDT",
            side=PositionSide.SHORT,
            quantity=0.04,  # $2000 < max $2500
            entry_price=50000,
        )

        assert position is not None
        assert position.side == PositionSide.SHORT

    def test_open_position_insufficient_balance(self, manager):
        """Test no abrir posicion si no hay balance"""
        # Intentar abrir posicion muy grande
        position = manager.open_position(
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            quantity=1,
            entry_price=50000,  # 50000 > 10000 balance
        )

        assert position is None

    def test_open_position_max_positions(self, manager):
        """Test limite de posiciones"""
        # Abrir 5 posiciones (el maximo)
        for i in range(5):
            manager.open_position(
                symbol="BTC/USDT",
                side=PositionSide.LONG,
                quantity=0.01,
                entry_price=50000,
            )

        assert len(manager.positions) == 5

        # Intentar abrir otra
        position = manager.open_position(
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            quantity=0.01,
            entry_price=50000,
        )

        assert position is None
        assert len(manager.positions) == 5

    def test_close_position_profit(self, manager):
        """Test cerrar posicion con ganancia"""
        position = manager.open_position(
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            quantity=0.04,  # $2000 < max $2500
            entry_price=50000,
        )

        initial_balance = manager.balance

        # Cerrar con ganancia
        trade = manager.close_position(
            position.id,
            exit_price=51000,
            exit_reason="Take Profit"
        )

        assert trade is not None
        assert trade.pnl > 0
        assert trade.exit_reason == "Take Profit"
        assert len(manager.positions) == 0
        assert len(manager.trade_history) == 1
        assert manager.balance > initial_balance

    def test_close_position_loss(self, manager):
        """Test cerrar posicion con perdida"""
        position = manager.open_position(
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            quantity=0.04,  # $2000 < max $2500
            entry_price=50000,
        )

        # Cerrar con perdida
        trade = manager.close_position(
            position.id,
            exit_price=49000,
            exit_reason="Stop Loss"
        )

        assert trade is not None
        assert trade.pnl < 0
        assert trade.exit_reason == "Stop Loss"

    def test_update_prices_triggers_stop_loss(self, manager):
        """Test que stop loss se activa con update_prices"""
        position = manager.open_position(
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            quantity=0.04,  # $2000 < max $2500
            entry_price=50000,
            stop_loss=49000,
        )

        assert len(manager.positions) == 1

        # Actualizar precio a nivel de stop
        manager.update_prices({"BTC/USDT": 48500})

        # La posicion deberia haberse cerrado
        assert len(manager.positions) == 0
        assert len(manager.trade_history) == 1
        assert manager.trade_history[0].exit_reason == "Stop Loss"

    def test_update_prices_triggers_take_profit(self, manager):
        """Test que take profit se activa con update_prices"""
        position = manager.open_position(
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            quantity=0.04,  # $2000 < max $2500
            entry_price=50000,
            take_profit=52000,
        )

        # Actualizar precio a nivel de TP
        manager.update_prices({"BTC/USDT": 53000})

        assert len(manager.positions) == 0
        assert manager.trade_history[0].exit_reason == "Take Profit"

    def test_get_unrealized_pnl(self, manager):
        """Test obtener PnL no realizado"""
        manager.open_position(
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            quantity=0.04,  # $2000 < max $2500
            entry_price=50000,
        )

        # Precio sube
        manager.update_prices({"BTC/USDT": 51000})
        pnl = manager.get_unrealized_pnl()

        assert pnl == pytest.approx(40, rel=0.01)  # (51000-50000) * 0.04

    def test_get_realized_pnl(self, manager):
        """Test obtener PnL realizado"""
        position = manager.open_position(
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            quantity=0.04,  # $2000 < max $2500
            entry_price=50000,
        )

        manager.close_position(position.id, 51000, "Manual")

        pnl = manager.get_realized_pnl()
        assert pnl > 0

    def test_update_stop_loss(self, manager):
        """Test actualizar stop loss"""
        position = manager.open_position(
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            quantity=0.04,  # $2000 < max $2500
            entry_price=50000,
            stop_loss=49000,
        )

        result = manager.update_stop_loss(position.id, 49500)

        assert result == True
        assert manager.positions[0].stop_loss == 49500

    def test_update_take_profit(self, manager):
        """Test actualizar take profit"""
        position = manager.open_position(
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            quantity=0.04,  # $2000 < max $2500
            entry_price=50000,
            take_profit=52000,
        )

        result = manager.update_take_profit(position.id, 53000)

        assert result == True
        assert manager.positions[0].take_profit == 53000

    def test_close_all_positions(self, manager):
        """Test cerrar todas las posiciones"""
        # Abrir varias posiciones
        for _ in range(3):
            manager.open_position(
                symbol="BTC/USDT",
                side=PositionSide.LONG,
                quantity=0.01,  # Muy pequena para permitir varias
                entry_price=50000,
            )

        manager.update_prices({"BTC/USDT": 51000})
        manager.close_all_positions("Close All")

        assert len(manager.positions) == 0
        assert len(manager.trade_history) == 3

    def test_get_win_rate(self, manager):
        """Test calculo de win rate"""
        # Sin trades
        assert manager.get_win_rate() == 0

        # Abrir y cerrar con ganancia
        pos1 = manager.open_position(
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            quantity=0.01,  # Pequena
            entry_price=50000,
        )
        manager.close_position(pos1.id, 51000, "Win")

        # Abrir y cerrar con perdida
        pos2 = manager.open_position(
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            quantity=0.01,  # Pequena
            entry_price=50000,
        )
        manager.close_position(pos2.id, 49000, "Loss")

        # 1 win de 2 trades = 50%
        assert manager.get_win_rate() == 50.0

    def test_get_profit_factor(self, manager):
        """Test calculo de profit factor"""
        # Abrir y cerrar con ganancia
        pos1 = manager.open_position(
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            quantity=0.01,  # Pequena
            entry_price=50000,
        )
        manager.close_position(pos1.id, 52000, "Win")  # ~$20 profit

        # Abrir y cerrar con perdida
        pos2 = manager.open_position(
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            quantity=0.01,  # Pequena
            entry_price=50000,
        )
        manager.close_position(pos2.id, 49000, "Loss")  # ~$10 loss

        pf = manager.get_profit_factor()
        assert pf > 1  # Mas ganancias que perdidas

    def test_get_position_summary(self, manager):
        """Test obtener resumen de posiciones"""
        manager.open_position(
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            quantity=0.04,  # $2000 < max $2500
            entry_price=50000,
        )

        manager.update_prices({"BTC/USDT": 51000})
        summary = manager.get_position_summary()

        assert summary.total_positions == 1
        assert summary.long_positions == 1
        assert summary.short_positions == 0
        assert summary.unrealized_pnl > 0

    def test_reset(self, manager):
        """Test reiniciar manager"""
        manager.open_position(
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            quantity=0.04,  # $2000 < max $2500
            entry_price=50000,
        )

        manager.reset()

        assert manager.balance == 10000
        assert len(manager.positions) == 0
        assert len(manager.trade_history) == 0

    def test_get_statistics(self, manager):
        """Test obtener estadisticas"""
        stats = manager.get_statistics()

        assert "total_trades" in stats
        assert "win_rate" in stats
        assert "profit_factor" in stats
        assert "avg_win" in stats
        assert "avg_loss" in stats
