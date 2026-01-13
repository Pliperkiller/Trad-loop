"""
Tests para ordenes de control de riesgo.

Cubre:
- Trailing Stop: activacion, actualizacion, trigger
- Bracket Order: creacion, entry fill, SL/TP execution
"""

import pytest
from datetime import datetime

import sys
from pathlib import Path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from paper_trading.orders.enums import OrderSide, OrderStatus, CompositeOrderStatus, OrderType
from paper_trading.orders.risk_control import TrailingStopOrder, BracketOrder


class TestTrailingStopOrder:
    """Tests para TrailingStopOrder"""

    def test_trailing_stop_creation_percent(self, btc_price):
        """Test creacion con porcentaje"""
        order = TrailingStopOrder(
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            quantity=0.1,
            trail_percent=0.02,
            initial_price=btc_price
        )
        assert order.trail_percent == 0.02
        assert order.trail_amount is None
        assert order.status == OrderStatus.PENDING

    def test_trailing_stop_creation_amount(self, btc_price):
        """Test creacion con cantidad fija"""
        order = TrailingStopOrder(
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            quantity=0.1,
            trail_amount=1000,
            initial_price=btc_price
        )
        assert order.trail_amount == 1000
        assert order.trail_percent is None

    def test_trailing_stop_requires_trail_param(self, btc_price):
        """Test que requiere trail_amount o trail_percent"""
        with pytest.raises(ValueError):
            TrailingStopOrder(
                symbol="BTC/USDT",
                side=OrderSide.SELL,
                quantity=0.1,
                initial_price=btc_price
            )

    def test_trailing_stop_initial_stop_price_long(self, btc_price):
        """Test precio stop inicial para long (2%)"""
        order = TrailingStopOrder(
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            quantity=0.1,
            trail_percent=0.02,
            initial_price=btc_price
        )
        expected_stop = btc_price * 0.98
        assert order.get_current_stop_price() == pytest.approx(expected_stop, rel=0.001)

    def test_trailing_stop_initial_stop_price_short(self, btc_price):
        """Test precio stop inicial para short (2%)"""
        order = TrailingStopOrder(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=0.1,
            trail_percent=0.02,
            initial_price=btc_price
        )
        expected_stop = btc_price * 1.02
        assert order.get_current_stop_price() == pytest.approx(expected_stop, rel=0.001)

    def test_trailing_stop_updates_on_price_increase_long(self, trailing_stop_long, btc_price):
        """Test que stop se actualiza cuando precio sube (long)"""
        initial_stop = trailing_stop_long.get_current_stop_price()
        
        # Precio sube 5%
        new_price = btc_price * 1.05
        action = trailing_stop_long.on_price_update(new_price, datetime.now())
        
        assert action is not None
        assert action.action_type == "modify"
        assert trailing_stop_long.get_current_stop_price() > initial_stop

    def test_trailing_stop_no_update_on_price_decrease_long(self, trailing_stop_long, btc_price):
        """Test que stop no cambia cuando precio baja (long)"""
        initial_stop = trailing_stop_long.get_current_stop_price()
        
        # Precio baja pero no toca stop
        new_price = btc_price * 0.99
        action = trailing_stop_long.on_price_update(new_price, datetime.now())
        
        # No hay accion de modificacion
        assert action is None or action.action_type != "modify"
        assert trailing_stop_long.get_current_stop_price() == initial_stop

    def test_trailing_stop_triggers_on_reversal(self, trailing_stop_long, btc_price):
        """Test que stop se ejecuta cuando precio cruza nivel"""
        # Primero sube
        trailing_stop_long.on_price_update(btc_price * 1.10, datetime.now())
        
        # Luego baja pasando el stop
        stop_price = trailing_stop_long.get_current_stop_price()
        action = trailing_stop_long.on_price_update(stop_price - 100, datetime.now())
        
        assert action is not None
        assert action.action_type == "execute"
        assert trailing_stop_long.status == OrderStatus.TRIGGERED

    def test_trailing_stop_with_activation_price(self, trailing_stop_with_activation, btc_price):
        """Test trailing stop con precio de activacion"""
        # No esta activado inicialmente
        assert not trailing_stop_with_activation.is_activated()
        
        # Precio por debajo de activacion - no se activa
        action = trailing_stop_with_activation.on_price_update(btc_price, datetime.now())
        assert trailing_stop_with_activation.is_activated() == False
        
        # Precio alcanza activacion
        activation = btc_price + 1000
        action = trailing_stop_with_activation.on_price_update(activation + 100, datetime.now())
        assert trailing_stop_with_activation.is_activated() == True

    def test_trailing_stop_short_updates_on_price_decrease(self, trailing_stop_short, btc_price):
        """Test que stop se actualiza cuando precio baja (short)"""
        initial_stop = trailing_stop_short.get_current_stop_price()
        
        # Precio baja 5%
        new_price = btc_price * 0.95
        action = trailing_stop_short.on_price_update(new_price, datetime.now())
        
        assert action is not None
        assert action.action_type == "modify"
        assert trailing_stop_short.get_current_stop_price() < initial_stop

    def test_trailing_stop_short_triggers(self, trailing_stop_short, btc_price):
        """Test que trailing stop short se ejecuta correctamente"""
        # Primero baja
        trailing_stop_short.on_price_update(btc_price * 0.90, datetime.now())
        
        # Luego sube pasando el stop
        stop_price = trailing_stop_short.get_current_stop_price()
        action = trailing_stop_short.on_price_update(stop_price + 100, datetime.now())
        
        assert action is not None
        assert action.action_type == "execute"

    def test_trailing_stop_to_dict(self, trailing_stop_long):
        """Test serializacion a diccionario"""
        d = trailing_stop_long.to_dict()
        
        assert "id" in d
        assert d["type"] == "trailing_stop"
        assert d["symbol"] == "BTC/USDT"
        assert "current_stop_price" in d
        assert "trail_percent" in d

    def test_trailing_stop_ccxt_params_binance(self, trailing_stop_long):
        """Test conversion a parametros CCXT para Binance"""
        params = trailing_stop_long.to_ccxt_params("binance")
        
        assert "trailingDelta" in params.get("params", {})
        assert params["params"]["trailingDelta"] == 200  # 2% = 200 bps


class TestBracketOrder:
    """Tests para BracketOrder"""

    def test_bracket_creation(self, btc_price):
        """Test creacion basica de bracket"""
        bracket = BracketOrder(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=0.1,
            entry_price=btc_price,
            stop_loss_price=btc_price - 1000,
            take_profit_price=btc_price + 2000
        )
        assert bracket.status == CompositeOrderStatus.PENDING
        assert not bracket.entry_filled
        assert bracket.stop_loss_price == btc_price - 1000
        assert bracket.take_profit_price == btc_price + 2000

    def test_bracket_validates_sl_for_long(self, btc_price):
        """Test que SL debe ser menor que entry para long"""
        with pytest.raises(ValueError, match="Stop loss debe ser menor"):
            BracketOrder(
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                quantity=0.1,
                entry_price=btc_price,
                stop_loss_price=btc_price + 500,  # Invalid: above entry
                take_profit_price=btc_price + 2000
            )

    def test_bracket_validates_tp_for_long(self, btc_price):
        """Test que TP debe ser mayor que entry para long"""
        with pytest.raises(ValueError, match="Take profit debe ser mayor"):
            BracketOrder(
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                quantity=0.1,
                entry_price=btc_price,
                stop_loss_price=btc_price - 1000,
                take_profit_price=btc_price - 500  # Invalid: below entry
            )

    def test_bracket_validates_sl_for_short(self, btc_price):
        """Test que SL debe ser mayor que entry para short"""
        with pytest.raises(ValueError, match="Stop loss debe ser mayor"):
            BracketOrder(
                symbol="BTC/USDT",
                side=OrderSide.SELL,
                quantity=0.1,
                entry_price=btc_price,
                stop_loss_price=btc_price - 500,  # Invalid: below entry
                take_profit_price=btc_price - 2000
            )

    def test_bracket_entry_fill_activates_sl_tp(self, bracket_order_long, btc_price):
        """Test que entry fill activa SL y TP"""
        action = bracket_order_long.on_child_filled(
            bracket_order_long.entry_order_id,
            fill_price=btc_price,
            fill_quantity=0.1
        )
        
        assert bracket_order_long.entry_filled
        assert bracket_order_long.status == CompositeOrderStatus.ACTIVE
        assert action.action_type == "activate_exits"

    def test_bracket_sl_triggers(self, bracket_order_long, btc_price):
        """Test que SL se ejecuta correctamente"""
        # Fill entry
        bracket_order_long.on_child_filled(
            bracket_order_long.entry_order_id, btc_price, 0.1
        )
        
        # Precio baja a SL
        sl_price = bracket_order_long.stop_loss_price
        action = bracket_order_long.on_price_update(sl_price - 100, datetime.now())
        
        assert action is not None
        assert action.action_type == "execute"
        assert action.order_id == bracket_order_long.stop_loss_order_id

    def test_bracket_tp_triggers(self, bracket_order_long, btc_price):
        """Test que TP se ejecuta correctamente"""
        # Fill entry
        bracket_order_long.on_child_filled(
            bracket_order_long.entry_order_id, btc_price, 0.1
        )
        
        # Precio sube a TP
        tp_price = bracket_order_long.take_profit_price
        action = bracket_order_long.on_price_update(tp_price + 100, datetime.now())
        
        assert action is not None
        assert action.action_type == "execute"
        assert action.order_id == bracket_order_long.take_profit_order_id

    def test_bracket_sl_cancels_tp(self, bracket_order_long, btc_price):
        """Test que SL fill cancela TP"""
        # Fill entry
        bracket_order_long.on_child_filled(
            bracket_order_long.entry_order_id, btc_price, 0.1
        )
        
        # SL fills
        action = bracket_order_long.on_child_filled(
            bracket_order_long.stop_loss_order_id, btc_price - 1000, 0.1
        )
        
        assert action is not None
        assert action.action_type == "cancel"
        assert action.order_id == bracket_order_long.take_profit_order_id
        assert bracket_order_long.status == CompositeOrderStatus.COMPLETED

    def test_bracket_tp_cancels_sl(self, bracket_order_long, btc_price):
        """Test que TP fill cancela SL"""
        # Fill entry
        bracket_order_long.on_child_filled(
            bracket_order_long.entry_order_id, btc_price, 0.1
        )
        
        # TP fills
        action = bracket_order_long.on_child_filled(
            bracket_order_long.take_profit_order_id, btc_price + 2000, 0.1
        )
        
        assert action is not None
        assert action.action_type == "cancel"
        assert action.order_id == bracket_order_long.stop_loss_order_id
        assert bracket_order_long.exit_reason == "take_profit"

    def test_bracket_cancel_cancels_all(self, bracket_order_long):
        """Test que cancelar bracket cancela todo"""
        action = bracket_order_long.cancel()
        
        assert action.action_type == "cancel_all"
        assert bracket_order_long.status == CompositeOrderStatus.CANCELLED

    def test_bracket_get_child_orders(self, bracket_order_long, btc_price):
        """Test obtener ordenes hijas"""
        orders = bracket_order_long.get_child_orders()
        
        # Solo entry antes de fill
        assert len(orders) == 1
        assert orders[0]["id"] == bracket_order_long.entry_order_id
        
        # Fill entry
        bracket_order_long.on_child_filled(
            bracket_order_long.entry_order_id, btc_price, 0.1
        )
        
        orders = bracket_order_long.get_child_orders()
        # Entry + SL + TP despues de fill
        assert len(orders) == 3

    def test_bracket_short_position(self, bracket_order_short, btc_price):
        """Test bracket para posicion short"""
        # Fill entry
        bracket_order_short.on_child_filled(
            bracket_order_short.entry_order_id, btc_price, 0.1
        )
        
        # Precio sube a SL (short)
        sl_price = bracket_order_short.stop_loss_price
        action = bracket_order_short.on_price_update(sl_price + 100, datetime.now())
        
        assert action is not None
        assert action.action_type == "execute"
        assert bracket_order_short.exit_reason == "stop_loss"

    def test_bracket_to_dict(self, bracket_order_long):
        """Test serializacion a diccionario"""
        d = bracket_order_long.to_dict()
        
        assert d["type"] == "bracket"
        assert "entry_order_id" in d
        assert "stop_loss_order_id" in d
        assert "take_profit_order_id" in d
        assert "entry_filled" in d

    def test_bracket_ccxt_params_bybit(self, bracket_order_long, btc_price):
        """Test conversion a parametros CCXT para Bybit"""
        params = bracket_order_long.to_ccxt_params("bybit")
        
        assert "takeProfit" in params
        assert "stopLoss" in params
        assert params["takeProfit"] == str(bracket_order_long.take_profit_price)


class TestTrailingStopSimulator:
    """Tests para TrailingStopSimulator"""

    def test_simulator_add_order(self, trailing_simulator, trailing_stop_long):
        """Test agregar orden al simulador"""
        result = trailing_simulator.add_order(trailing_stop_long)
        
        assert result == True
        assert trailing_stop_long.id in trailing_simulator.active_orders

    def test_simulator_process_price_update(self, trailing_simulator, trailing_stop_long, btc_price):
        """Test procesar actualizacion de precio"""
        trailing_simulator.add_order(trailing_stop_long)
        
        # Precio sube
        actions = trailing_simulator.process_price_update(
            "BTC/USDT", btc_price * 1.05, datetime.now()
        )
        
        assert len(actions) > 0
        assert any(a.action_type == "modify" for _, a in actions)

    def test_simulator_order_execution(self, trailing_simulator, trailing_stop_long, btc_price):
        """Test ejecucion de orden"""
        trailing_simulator.add_order(trailing_stop_long)
        
        # Primero sube
        trailing_simulator.process_price_update("BTC/USDT", btc_price * 1.10, datetime.now())
        
        # Luego baja hasta ejecutar
        stop = trailing_stop_long.get_current_stop_price()
        actions = trailing_simulator.process_price_update("BTC/USDT", stop - 500, datetime.now())
        
        assert any(a.action_type == "execute" for _, a in actions)
        assert trailing_stop_long.id not in trailing_simulator.active_orders
        assert len(trailing_simulator.executed_orders) == 1

    def test_simulator_cancel_order(self, trailing_simulator, trailing_stop_long):
        """Test cancelar orden"""
        trailing_simulator.add_order(trailing_stop_long)
        
        result = trailing_simulator.cancel_order(trailing_stop_long.id)
        
        assert result == True
        assert trailing_stop_long.id not in trailing_simulator.active_orders

    def test_simulator_statistics(self, trailing_simulator, trailing_stop_long):
        """Test estadisticas del simulador"""
        trailing_simulator.add_order(trailing_stop_long)
        
        stats = trailing_simulator.get_statistics()
        
        assert stats["active_orders"] == 1
        assert "BTC/USDT" in stats["symbols"]
