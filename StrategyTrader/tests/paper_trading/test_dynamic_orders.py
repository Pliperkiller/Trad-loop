"""
Tests para ordenes de gestion dinamica.

Cubre:
- FOK (Fill-or-Kill)
- IOC (Immediate-or-Cancel)
- Reduce-Only
- Post-Only
- Scale-In/Scale-Out
"""

import pytest
from datetime import datetime

import sys
from pathlib import Path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from paper_trading.orders.enums import OrderSide, OrderStatus, OrderType
from paper_trading.orders.dynamic_orders import (
    FOKOrder, IOCOrder, ReduceOnlyOrder, PostOnlyOrder, ScaleOrder
)


class TestFOKOrder:
    """Tests para FOKOrder (Fill-or-Kill)"""

    def test_fok_creation(self, btc_price):
        """Test creacion basica de FOK"""
        order = FOKOrder(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=1.0,
            price=btc_price
        )
        assert order.quantity == 1.0
        assert order.status == OrderStatus.PENDING
        assert not order.executed

    def test_fok_can_fill_with_liquidity(self, fok_order, btc_price):
        """Test verificacion de liquidez suficiente"""
        result = fok_order.can_fill(
            available_liquidity=2.0,  # Mas que quantity
            market_price=btc_price
        )
        assert result == True

    def test_fok_cannot_fill_insufficient_liquidity(self, fok_order, btc_price):
        """Test rechazo por liquidez insuficiente"""
        result = fok_order.can_fill(
            available_liquidity=0.5,  # Menos que quantity
            market_price=btc_price
        )
        assert result == False

    def test_fok_cannot_fill_price_moved(self, fok_order, btc_price):
        """Test rechazo por movimiento de precio"""
        # Precio de mercado subio por encima del limite
        result = fok_order.can_fill(
            available_liquidity=2.0,
            market_price=btc_price + 500
        )
        assert result == False

    def test_fok_execute(self, fok_order, btc_price):
        """Test ejecucion exitosa"""
        fok_order.execute(btc_price)
        
        assert fok_order.executed == True
        assert fok_order.executed_price == btc_price
        assert fok_order.status == OrderStatus.FILLED

    def test_fok_reject(self, fok_order):
        """Test rechazo de orden"""
        fok_order.reject("Insufficient liquidity")
        
        assert fok_order.executed == False
        assert fok_order.rejection_reason == "Insufficient liquidity"
        assert fok_order.status == OrderStatus.CANCELLED

    def test_fok_ccxt_params(self, fok_order):
        """Test parametros CCXT"""
        params = fok_order.to_ccxt_params("binance")
        
        assert params["params"]["timeInForce"] == "FOK"

    def test_fok_to_dict(self, fok_order):
        """Test serializacion"""
        d = fok_order.to_dict()
        
        assert d["type"] == "fill_or_kill"
        assert "executed" in d
        assert "rejection_reason" in d


class TestIOCOrder:
    """Tests para IOCOrder (Immediate-or-Cancel)"""

    def test_ioc_creation(self, btc_price):
        """Test creacion basica de IOC"""
        order = IOCOrder(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=1.0,
            price=btc_price
        )
        assert order.quantity == 1.0
        assert order.filled_quantity == 0.0

    def test_ioc_partial_fill(self, ioc_order, btc_price):
        """Test ejecucion parcial"""
        ioc_order.execute_partial(0.3, btc_price)
        
        assert ioc_order.filled_quantity == 0.3
        assert ioc_order.average_price == btc_price
        assert ioc_order.status == OrderStatus.PARTIALLY_FILLED

    def test_ioc_multiple_partial_fills(self, ioc_order, btc_price):
        """Test multiples fills parciales con precio promedio"""
        ioc_order.execute_partial(0.3, btc_price)
        ioc_order.execute_partial(0.2, btc_price + 100)
        
        # Precio promedio ponderado
        expected_avg = (btc_price * 0.3 + (btc_price + 100) * 0.2) / 0.5
        assert ioc_order.filled_quantity == 0.5
        assert ioc_order.average_price == pytest.approx(expected_avg, rel=0.01)

    def test_ioc_full_fill(self, ioc_order, btc_price):
        """Test fill completo"""
        ioc_order.execute_partial(1.0, btc_price)
        
        assert ioc_order.filled_quantity == 1.0
        assert ioc_order.status == OrderStatus.FILLED

    def test_ioc_finalize_partial(self, ioc_order, btc_price):
        """Test finalizacion con fill parcial"""
        ioc_order.execute_partial(0.3, btc_price)
        ioc_order.finalize()
        
        assert ioc_order.cancelled_quantity == 0.7
        assert ioc_order.status == OrderStatus.PARTIALLY_FILLED

    def test_ioc_finalize_no_fill(self, ioc_order):
        """Test finalizacion sin fills"""
        ioc_order.finalize()
        
        assert ioc_order.cancelled_quantity == 1.0
        assert ioc_order.status == OrderStatus.CANCELLED

    def test_ioc_ccxt_params(self, ioc_order):
        """Test parametros CCXT"""
        params = ioc_order.to_ccxt_params("binance")
        
        assert params["params"]["timeInForce"] == "IOC"

    def test_ioc_to_dict(self, ioc_order):
        """Test serializacion"""
        d = ioc_order.to_dict()
        
        assert d["type"] == "immediate_or_cancel"
        assert "filled_quantity" in d
        assert "cancelled_quantity" in d


class TestReduceOnlyOrder:
    """Tests para ReduceOnlyOrder"""

    def test_reduce_only_creation(self, btc_price):
        """Test creacion basica de Reduce-Only"""
        order = ReduceOnlyOrder(
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            quantity=0.5,
            position_quantity=1.0,
            price=btc_price
        )
        assert order.quantity == 0.5
        assert order.position_quantity == 1.0

    def test_reduce_only_validate_with_position(self, reduce_only_order):
        """Test validacion con posicion existente"""
        error = reduce_only_order.validate()
        assert error is None

    def test_reduce_only_validate_no_position(self, reduce_only_order_no_position):
        """Test validacion sin posicion"""
        error = reduce_only_order_no_position.validate()
        assert error is not None
        assert "No hay posicion" in error

    def test_reduce_only_validate_exceeds_position(self, btc_price):
        """Test validacion cuando excede posicion"""
        order = ReduceOnlyOrder(
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            quantity=2.0,  # Excede posicion
            position_quantity=1.0,
            price=btc_price
        )
        error = order.validate()
        assert error is not None
        assert "excede posicion" in error

    def test_reduce_only_adjust_to_position(self, btc_price):
        """Test ajuste automatico a tamano de posicion"""
        order = ReduceOnlyOrder(
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            quantity=2.0,
            position_quantity=1.0,
            price=btc_price
        )
        adjusted = order.adjust_to_position()
        assert adjusted == 1.0

    def test_reduce_only_execute(self, reduce_only_order, btc_price):
        """Test ejecucion"""
        reduce_only_order.execute(btc_price)
        
        assert reduce_only_order.filled_quantity == 0.5
        assert reduce_only_order.status == OrderStatus.FILLED

    def test_reduce_only_reject(self, reduce_only_order):
        """Test rechazo"""
        reduce_only_order.reject("No position to reduce")
        
        assert reduce_only_order.rejected == True
        assert reduce_only_order.status == OrderStatus.REJECTED

    def test_reduce_only_ccxt_params(self, reduce_only_order):
        """Test parametros CCXT"""
        params = reduce_only_order.to_ccxt_params("binance")
        
        assert params["params"]["reduceOnly"] == True

    def test_reduce_only_to_dict(self, reduce_only_order):
        """Test serializacion"""
        d = reduce_only_order.to_dict()
        
        assert d["type"] == "reduce_only"
        assert d["position_quantity"] == 1.0


class TestPostOnlyOrder:
    """Tests para PostOnlyOrder"""

    def test_post_only_creation(self, btc_price):
        """Test creacion basica de Post-Only"""
        order = PostOnlyOrder(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=1.0,
            price=btc_price - 100
        )
        assert order.quantity == 1.0
        assert order.price == btc_price - 100

    def test_post_only_would_cross_buy(self, post_only_order, btc_price):
        """Test deteccion de cruce para compra"""
        # Orden de compra cruza si precio >= ask
        order = PostOnlyOrder(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=1.0,
            price=btc_price + 100  # Por encima del ask
        )
        
        crosses = order.would_cross_spread(bid=btc_price - 5, ask=btc_price + 5)
        assert crosses == True

    def test_post_only_would_not_cross_buy(self, btc_price):
        """Test que no cruza para compra valida"""
        order = PostOnlyOrder(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=1.0,
            price=btc_price - 100  # Por debajo del ask
        )
        
        crosses = order.would_cross_spread(bid=btc_price - 5, ask=btc_price + 5)
        assert crosses == False

    def test_post_only_would_cross_sell(self, btc_price):
        """Test deteccion de cruce para venta"""
        order = PostOnlyOrder(
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            quantity=1.0,
            price=btc_price - 100  # Por debajo del bid
        )
        
        crosses = order.would_cross_spread(bid=btc_price - 5, ask=btc_price + 5)
        assert crosses == True

    def test_post_only_submit_success(self, btc_price):
        """Test envio exitoso"""
        order = PostOnlyOrder(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=1.0,
            price=btc_price - 100
        )
        
        result = order.submit(bid=btc_price - 5, ask=btc_price + 5)
        
        assert result == True
        assert order.status == OrderStatus.SUBMITTED

    def test_post_only_submit_cancelled(self, btc_price):
        """Test envio cancelado por cruce"""
        order = PostOnlyOrder(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=1.0,
            price=btc_price + 100  # Cruzaria spread
        )
        
        result = order.submit(bid=btc_price - 5, ask=btc_price + 5)
        
        assert result == False
        assert order.was_cancelled == True
        assert order.status == OrderStatus.CANCELLED

    def test_post_only_execute(self, post_only_order, btc_price):
        """Test ejecucion"""
        post_only_order.execute(btc_price - 100)
        
        assert post_only_order.filled_quantity == 1.0
        assert post_only_order.status == OrderStatus.FILLED

    def test_post_only_ccxt_params_binance(self, post_only_order):
        """Test parametros CCXT para Binance"""
        params = post_only_order.to_ccxt_params("binance")
        
        assert params["params"]["postOnly"] == True

    def test_post_only_ccxt_params_kraken(self, post_only_order):
        """Test parametros CCXT para Kraken"""
        params = post_only_order.to_ccxt_params("kraken")
        
        assert params["params"]["oflags"] == "post"

    def test_post_only_to_dict(self, post_only_order):
        """Test serializacion"""
        d = post_only_order.to_dict()
        
        assert d["type"] == "post_only"
        assert "was_cancelled" in d


class TestScaleOrder:
    """Tests para ScaleOrder (Scale-In/Scale-Out)"""

    def test_scale_in_creation(self, btc_price):
        """Test creacion basica de Scale-In"""
        order = ScaleOrder(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            total_quantity=1.0,
            levels=[btc_price, btc_price - 500, btc_price - 1000],
            quantities=[0.25, 0.25, 0.50],
            order_type=OrderType.SCALE_IN
        )
        assert order.total_quantity == 1.0
        assert len(order.levels) == 3

    def test_scale_validates_matching_lengths(self, btc_price):
        """Test que levels y quantities deben tener igual longitud"""
        with pytest.raises(ValueError, match="igual longitud"):
            ScaleOrder(
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                total_quantity=1.0,
                levels=[btc_price, btc_price - 500],
                quantities=[0.25, 0.25, 0.50],  # Mismatch
            )

    def test_scale_validates_quantities_sum(self, btc_price):
        """Test que quantities debe sumar 1.0"""
        with pytest.raises(ValueError, match="sumar 1.0"):
            ScaleOrder(
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                total_quantity=1.0,
                levels=[btc_price, btc_price - 500],
                quantities=[0.25, 0.50],  # Suma 0.75
            )

    def test_scale_quantity_for_level(self, scale_in_order):
        """Test calculo de cantidad por nivel"""
        qty_0 = scale_in_order.get_quantity_for_level(0)
        qty_1 = scale_in_order.get_quantity_for_level(1)
        qty_2 = scale_in_order.get_quantity_for_level(2)
        
        assert qty_0 == pytest.approx(0.25, rel=0.01)
        assert qty_1 == pytest.approx(0.25, rel=0.01)
        assert qty_2 == pytest.approx(0.50, rel=0.01)

    def test_scale_pending_levels(self, scale_in_order):
        """Test obtener niveles pendientes"""
        pending = scale_in_order.get_pending_levels()
        
        assert len(pending) == 3
        assert pending[0][0] == 0  # Indice
        assert pending[0][2] == pytest.approx(0.25, rel=0.01)  # Cantidad

    def test_scale_check_level_trigger_buy(self, scale_in_order, btc_price):
        """Test verificacion de trigger para compra"""
        # Nivel 0 deberia triggear cuando precio <= nivel
        should_trigger = scale_in_order.check_level_trigger(0, btc_price - 100)
        assert should_trigger == True
        
        should_not = scale_in_order.check_level_trigger(0, btc_price + 100)
        assert should_not == False

    def test_scale_check_level_trigger_sell(self, scale_out_order, btc_price):
        """Test verificacion de trigger para venta"""
        # Nivel 0 (btc_price + 500) deberia triggear cuando precio >= nivel
        level_price = btc_price + 500
        should_trigger = scale_out_order.check_level_trigger(0, level_price + 100)
        assert should_trigger == True

    def test_scale_execute_level(self, scale_in_order, btc_price):
        """Test ejecucion de un nivel"""
        scale_in_order.execute_level(0, btc_price)
        
        assert scale_in_order.level_statuses[0] == "filled"
        assert scale_in_order.filled_quantity == pytest.approx(0.25, rel=0.01)
        assert scale_in_order.average_price == btc_price

    def test_scale_execute_multiple_levels(self, scale_in_order, btc_price):
        """Test ejecucion de multiples niveles"""
        scale_in_order.execute_level(0, btc_price)
        scale_in_order.execute_level(1, btc_price - 500)
        
        # Promedio ponderado
        expected_avg = (btc_price * 0.25 + (btc_price - 500) * 0.25) / 0.5
        assert scale_in_order.filled_quantity == pytest.approx(0.5, rel=0.01)
        assert scale_in_order.average_price == pytest.approx(expected_avg, rel=0.01)
        assert scale_in_order.status == OrderStatus.PARTIALLY_FILLED

    def test_scale_complete_all_levels(self, scale_in_order, btc_price):
        """Test completar todos los niveles"""
        scale_in_order.execute_level(0, btc_price)
        scale_in_order.execute_level(1, btc_price - 500)
        scale_in_order.execute_level(2, btc_price - 1000)
        
        assert scale_in_order.filled_quantity == pytest.approx(1.0, rel=0.01)
        assert scale_in_order.status == OrderStatus.FILLED

    def test_scale_progress(self, scale_in_order, btc_price):
        """Test calculo de progreso"""
        assert scale_in_order.get_progress_pct() == 0.0
        
        scale_in_order.execute_level(0, btc_price)
        assert scale_in_order.get_progress_pct() == pytest.approx(33.33, rel=0.1)

    def test_scale_to_dict(self, scale_in_order):
        """Test serializacion"""
        d = scale_in_order.to_dict()
        
        assert d["type"] == "scale_in"
        assert "levels" in d
        assert "quantities" in d
        assert "level_statuses" in d
