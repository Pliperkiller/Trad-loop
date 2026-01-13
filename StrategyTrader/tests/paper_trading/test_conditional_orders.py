"""
Tests para ordenes condicionales.

Cubre:
- If-Touched: Ordenes activadas por precio
- OCO (One-Cancels-Other)
- OTOCO (One-Triggers-OCO)
"""

import pytest
from datetime import datetime

import sys
from pathlib import Path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from paper_trading.orders.enums import (
    OrderSide, OrderStatus, TriggerDirection, TriggerType, CompositeOrderStatus
)
from paper_trading.orders.conditional_orders import IfTouchedOrder, OCOOrder, OTOCOOrder


class TestIfTouchedOrder:
    """Tests para IfTouchedOrder"""

    def test_if_touched_creation(self, btc_price):
        """Test creacion basica de If-Touched"""
        order = IfTouchedOrder(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=0.1,
            trigger_price=btc_price + 1000,
            trigger_direction=TriggerDirection.ABOVE
        )
        assert order.trigger_price == btc_price + 1000
        assert order.trigger_direction == TriggerDirection.ABOVE
        assert not order.triggered

    def test_if_touched_check_trigger_above(self, if_touched_order_above, btc_price):
        """Test trigger ABOVE"""
        # Precio por debajo - no trigger
        result = if_touched_order_above.check_trigger(btc_price)
        assert result == False
        
        # Precio alcanza trigger
        result = if_touched_order_above.check_trigger(btc_price + 1500)
        assert result == True

    def test_if_touched_check_trigger_below(self, if_touched_order_below, btc_price):
        """Test trigger BELOW"""
        # Precio por encima - no trigger
        result = if_touched_order_below.check_trigger(btc_price)
        assert result == False
        
        # Precio alcanza trigger
        result = if_touched_order_below.check_trigger(btc_price - 1500)
        assert result == True

    def test_if_touched_activate(self, if_touched_order_above, btc_price):
        """Test activacion"""
        trigger_price = btc_price + 1000
        if_touched_order_above.activate(trigger_price)
        
        assert if_touched_order_above.triggered == True
        assert if_touched_order_above.triggered_price == trigger_price
        assert if_touched_order_above.status == OrderStatus.TRIGGERED

    def test_if_touched_execute(self, if_touched_order_above, btc_price):
        """Test ejecucion despues de activacion"""
        if_touched_order_above.activate(btc_price + 1000)
        if_touched_order_above.execute(btc_price + 1050)
        
        assert if_touched_order_above.filled_price == btc_price + 1050
        assert if_touched_order_above.status == OrderStatus.FILLED

    def test_if_touched_with_limit_price(self, btc_price):
        """Test If-Touched con precio limite"""
        order = IfTouchedOrder(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=0.1,
            trigger_price=btc_price + 1000,
            trigger_direction=TriggerDirection.ABOVE,
            order_price=btc_price + 1050  # Limit despues de trigger
        )
        assert order.order_price == btc_price + 1050

    def test_if_touched_trigger_type(self, btc_price):
        """Test diferentes tipos de trigger"""
        order = IfTouchedOrder(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=0.1,
            trigger_price=btc_price,
            trigger_direction=TriggerDirection.ABOVE,
            trigger_type=TriggerType.MARK_PRICE
        )
        
        # Usar mark_price para evaluacion
        result = order.check_trigger(
            current_price=btc_price - 100,
            mark_price=btc_price + 100  # Mark esta arriba
        )
        assert result == True

    def test_if_touched_ccxt_params(self, if_touched_order_above):
        """Test parametros CCXT"""
        params = if_touched_order_above.to_ccxt_params("binance")
        
        assert "stopPrice" in params["params"]

    def test_if_touched_to_dict(self, if_touched_order_above):
        """Test serializacion"""
        d = if_touched_order_above.to_dict()
        
        assert d["type"] == "if_touched"
        assert "trigger_price" in d
        assert "trigger_direction" in d


class TestOCOOrder:
    """Tests para OCOOrder (One-Cancels-Other)"""

    def test_oco_creation(self, btc_price):
        """Test creacion basica de OCO"""
        oco = OCOOrder(
            symbol="BTC/USDT",
            quantity=0.1,
            side=OrderSide.SELL,
            stop_price=btc_price - 1000,
            limit_price=btc_price + 2000
        )
        assert oco.stop_price == btc_price - 1000
        assert oco.limit_price == btc_price + 2000
        assert oco.status == CompositeOrderStatus.ACTIVE

    def test_oco_check_triggers_long_stop(self, oco_order_long, btc_price):
        """Test trigger de stop para long"""
        # Precio no alcanza ninguno
        result = oco_order_long.check_triggers(btc_price)
        assert result is None
        
        # Precio alcanza stop
        result = oco_order_long.check_triggers(btc_price - 1500)
        assert result == "stop"

    def test_oco_check_triggers_long_limit(self, oco_order_long, btc_price):
        """Test trigger de limit para long"""
        result = oco_order_long.check_triggers(btc_price + 2500)
        assert result == "limit"

    def test_oco_check_triggers_short_stop(self, oco_order_short, btc_price):
        """Test trigger de stop para short"""
        result = oco_order_short.check_triggers(btc_price + 1500)
        assert result == "stop"

    def test_oco_check_triggers_short_limit(self, oco_order_short, btc_price):
        """Test trigger de limit para short"""
        result = oco_order_short.check_triggers(btc_price - 2500)
        assert result == "limit"

    def test_oco_execute_stop(self, oco_order_long, btc_price):
        """Test ejecucion de stop"""
        oco_order_long.execute_order("stop", btc_price - 1000)
        
        assert oco_order_long.executed_order == "stop"
        assert oco_order_long.cancelled_order == "limit"
        assert oco_order_long.status == CompositeOrderStatus.COMPLETED

    def test_oco_execute_limit(self, oco_order_long, btc_price):
        """Test ejecucion de limit"""
        oco_order_long.execute_order("limit", btc_price + 2000)
        
        assert oco_order_long.executed_order == "limit"
        assert oco_order_long.cancelled_order == "stop"

    def test_oco_get_executed_order_id(self, oco_order_long, btc_price):
        """Test obtener ID de orden ejecutada"""
        oco_order_long.execute_order("stop", btc_price - 1000)
        
        executed_id = oco_order_long.get_executed_order_id()
        cancelled_id = oco_order_long.get_cancelled_order_id()
        
        assert executed_id == oco_order_long.stop_order_id
        assert cancelled_id == oco_order_long.limit_order_id

    def test_oco_with_stop_limit_price(self, btc_price):
        """Test OCO con precio limite en stop"""
        oco = OCOOrder(
            symbol="BTC/USDT",
            quantity=0.1,
            side=OrderSide.SELL,
            stop_price=btc_price - 1000,
            stop_limit_price=btc_price - 1050,
            limit_price=btc_price + 2000
        )
        assert oco.stop_limit_price == btc_price - 1050

    def test_oco_ccxt_params_binance(self, oco_order_long):
        """Test parametros CCXT para Binance"""
        params = oco_order_long.to_ccxt_params("binance")
        
        assert "price" in params
        assert "stopPrice" in params

    def test_oco_ccxt_params_okx(self, oco_order_long):
        """Test parametros CCXT para OKX"""
        params = oco_order_long.to_ccxt_params("okx")
        
        assert params["ordType"] == "oco"
        assert "tpTriggerPx" in params
        assert "slTriggerPx" in params

    def test_oco_can_execute_on_exchange(self, oco_order_long):
        """Test verificacion de soporte de exchange"""
        assert oco_order_long.can_execute_on_exchange("binance") == True
        assert oco_order_long.can_execute_on_exchange("bybit") == False

    def test_oco_to_dict(self, oco_order_long):
        """Test serializacion"""
        d = oco_order_long.to_dict()
        
        assert d["type"] == "one_cancels_other"
        assert "stop_price" in d
        assert "limit_price" in d
        assert "stop_order_id" in d
        assert "limit_order_id" in d


class TestOTOCOOrder:
    """Tests para OTOCOOrder (One-Triggers-OCO)"""

    def test_otoco_creation(self, btc_price):
        """Test creacion basica de OTOCO"""
        otoco = OTOCOOrder(
            symbol="BTC/USDT",
            entry_side=OrderSide.BUY,
            entry_quantity=0.1,
            entry_price=btc_price,
            stop_loss_price=btc_price - 1000,
            take_profit_price=btc_price + 2000
        )
        assert otoco.entry_price == btc_price
        assert otoco.stop_loss_price == btc_price - 1000
        assert otoco.take_profit_price == btc_price + 2000
        assert otoco.status == CompositeOrderStatus.PENDING

    def test_otoco_validates_sl_for_long(self, btc_price):
        """Test validacion SL para long"""
        with pytest.raises(ValueError, match="Stop loss debe ser menor"):
            OTOCOOrder(
                symbol="BTC/USDT",
                entry_side=OrderSide.BUY,
                entry_quantity=0.1,
                entry_price=btc_price,
                stop_loss_price=btc_price + 500,  # Invalid
                take_profit_price=btc_price + 2000
            )

    def test_otoco_validates_tp_for_long(self, btc_price):
        """Test validacion TP para long"""
        with pytest.raises(ValueError, match="Take profit debe ser mayor"):
            OTOCOOrder(
                symbol="BTC/USDT",
                entry_side=OrderSide.BUY,
                entry_quantity=0.1,
                entry_price=btc_price,
                stop_loss_price=btc_price - 1000,
                take_profit_price=btc_price - 500  # Invalid
            )

    def test_otoco_check_entry_trigger_limit(self, otoco_order_long, btc_price):
        """Test trigger de entrada limit"""
        # Precio por encima - no trigger
        result = otoco_order_long.check_entry_trigger(btc_price + 100)
        assert result == False
        
        # Precio alcanza limit
        result = otoco_order_long.check_entry_trigger(btc_price - 100)
        assert result == True

    def test_otoco_execute_entry_activates_oco(self, otoco_order_long, btc_price):
        """Test que entry fill activa OCO"""
        otoco_order_long.execute_entry(btc_price, 0.1)
        
        assert otoco_order_long.entry_filled == True
        assert otoco_order_long.status == CompositeOrderStatus.ACTIVE
        assert otoco_order_long.oco is not None

    def test_otoco_oco_has_correct_params(self, otoco_order_long, btc_price):
        """Test que OCO tiene parametros correctos"""
        otoco_order_long.execute_entry(btc_price, 0.1)
        oco = otoco_order_long.oco
        
        assert oco.symbol == "BTC/USDT"
        assert oco.quantity == 0.1
        assert oco.side == OrderSide.SELL  # Opuesto al entry
        assert oco.stop_price == btc_price - 1000
        assert oco.limit_price == btc_price + 2000

    def test_otoco_check_exit_triggers_before_entry(self, otoco_order_long, btc_price):
        """Test que no hay exit triggers antes de entry"""
        result = otoco_order_long.check_exit_triggers(btc_price - 2000)
        assert result is None

    def test_otoco_check_exit_triggers_stop_loss(self, otoco_order_long, btc_price):
        """Test trigger de stop loss"""
        otoco_order_long.execute_entry(btc_price, 0.1)
        
        result = otoco_order_long.check_exit_triggers(btc_price - 1500)
        assert result == "stop_loss"

    def test_otoco_check_exit_triggers_take_profit(self, otoco_order_long, btc_price):
        """Test trigger de take profit"""
        otoco_order_long.execute_entry(btc_price, 0.1)
        
        result = otoco_order_long.check_exit_triggers(btc_price + 2500)
        assert result == "take_profit"

    def test_otoco_execute_exit_stop_loss(self, otoco_order_long, btc_price):
        """Test ejecucion de stop loss"""
        otoco_order_long.execute_entry(btc_price, 0.1)
        otoco_order_long.execute_exit("stop_loss", btc_price - 1000)
        
        assert otoco_order_long.exit_type == "stop_loss"
        assert otoco_order_long.exit_price == btc_price - 1000
        assert otoco_order_long.status == CompositeOrderStatus.COMPLETED

    def test_otoco_execute_exit_take_profit(self, otoco_order_long, btc_price):
        """Test ejecucion de take profit"""
        otoco_order_long.execute_entry(btc_price, 0.1)
        otoco_order_long.execute_exit("take_profit", btc_price + 2000)
        
        assert otoco_order_long.exit_type == "take_profit"

    def test_otoco_get_pnl_profit(self, otoco_order_long, btc_price):
        """Test calculo de PnL con ganancia"""
        otoco_order_long.execute_entry(btc_price, 0.1)
        otoco_order_long.execute_exit("take_profit", btc_price + 2000)
        
        pnl = otoco_order_long.get_pnl()
        expected = 2000 * 0.1  # $200 profit
        assert pnl == pytest.approx(expected, rel=0.01)

    def test_otoco_get_pnl_loss(self, otoco_order_long, btc_price):
        """Test calculo de PnL con perdida"""
        otoco_order_long.execute_entry(btc_price, 0.1)
        otoco_order_long.execute_exit("stop_loss", btc_price - 1000)
        
        pnl = otoco_order_long.get_pnl()
        expected = -1000 * 0.1  # $100 loss
        assert pnl == pytest.approx(expected, rel=0.01)

    def test_otoco_short_position(self, otoco_order_short, btc_price):
        """Test OTOCO para posicion short"""
        otoco_order_short.execute_entry(btc_price, 0.1)
        
        # OCO debe tener side BUY (para cerrar short)
        assert otoco_order_short.oco.side == OrderSide.BUY
        
        # SL sube, TP baja (inverso de long)
        result = otoco_order_short.check_exit_triggers(btc_price + 1500)
        assert result == "stop_loss"
        
        result = otoco_order_short.check_exit_triggers(btc_price - 2500)
        assert result == "take_profit"

    def test_otoco_to_dict(self, otoco_order_long):
        """Test serializacion"""
        d = otoco_order_long.to_dict()
        
        assert d["type"] == "one_triggers_oco"
        assert "entry_price" in d
        assert "stop_loss_price" in d
        assert "take_profit_price" in d
        assert "entry_filled" in d
        assert "pnl" in d


class TestCompositeOrderSimulator:
    """Tests para CompositeOrderSimulator"""

    def test_simulator_add_bracket(self, composite_simulator, bracket_order_long):
        """Test agregar bracket al simulador"""
        result = composite_simulator.add_bracket(bracket_order_long)
        
        assert result == bracket_order_long.id
        assert bracket_order_long.id in composite_simulator.brackets

    def test_simulator_add_oco(self, composite_simulator, oco_order_long):
        """Test agregar OCO al simulador"""
        result = composite_simulator.add_oco(oco_order_long)
        
        assert result == oco_order_long.id
        assert oco_order_long.id in composite_simulator.ocos

    def test_simulator_process_bracket(self, composite_simulator, bracket_order_long, btc_price):
        """Test procesamiento de bracket"""
        composite_simulator.add_bracket(bracket_order_long)
        
        # Procesar precio que llena entry
        actions = composite_simulator.process_price_update(
            "BTC/USDT", btc_price, datetime.now()
        )
        
        assert bracket_order_long.entry_filled

    def test_simulator_process_oco(self, composite_simulator, oco_order_long, btc_price):
        """Test procesamiento de OCO"""
        composite_simulator.add_oco(oco_order_long)
        
        # Procesar precio que activa stop
        actions = composite_simulator.process_price_update(
            "BTC/USDT", btc_price - 1500, datetime.now()
        )
        
        # Debe haber ejecutado stop y cancelado limit
        assert len(actions) >= 1
        assert oco_order_long.status == CompositeOrderStatus.COMPLETED

    def test_simulator_cancel_order(self, composite_simulator, bracket_order_long):
        """Test cancelar orden"""
        composite_simulator.add_bracket(bracket_order_long)
        
        result = composite_simulator.cancel_order(bracket_order_long.id)
        
        assert result == True
        assert bracket_order_long.id not in composite_simulator.brackets

    def test_simulator_statistics(self, composite_simulator, bracket_order_long, oco_order_long):
        """Test estadisticas del simulador"""
        composite_simulator.add_bracket(bracket_order_long)
        composite_simulator.add_oco(oco_order_long)
        
        stats = composite_simulator.get_statistics()
        
        assert stats["active_brackets"] == 1
        assert stats["active_ocos"] == 1
