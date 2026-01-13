"""
Tests para ordenes de ejecucion algoritmica.

Cubre:
- TWAP: Time-Weighted Average Price
- VWAP: Volume-Weighted Average Price
- Iceberg: Ordenes parcialmente visibles
- Hidden: Ordenes completamente ocultas
"""

import pytest
from datetime import datetime, timedelta

import sys
from pathlib import Path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from paper_trading.orders.enums import OrderSide, OrderStatus
from paper_trading.orders.execution_algos import TWAPOrder, VWAPOrder, IcebergOrder, HiddenOrder
from paper_trading.simulators.algo_simulator import (
    AlgoOrderSimulator, TWAPConfig, VWAPConfig, IcebergConfig
)


class TestTWAPOrder:
    """Tests para TWAPOrder"""

    def test_twap_creation(self, btc_price):
        """Test creacion basica de TWAP"""
        order = TWAPOrder(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            total_quantity=10.0,
            duration_seconds=3600,
            slice_count=60
        )
        assert order.total_quantity == 10.0
        assert order.slice_count == 60
        assert order.status == OrderStatus.PENDING

    def test_twap_default_slice_count(self):
        """Test que slice_count se calcula por defecto"""
        order = TWAPOrder(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            total_quantity=10.0,
            duration_seconds=3600  # 1 hora
        )
        # Default: 1 slice por minuto = 60 slices
        assert order.slice_count == 60

    def test_twap_slice_size(self, twap_order):
        """Test calculo de tamano de slice"""
        slice_size = twap_order.get_slice_size()
        expected = 10.0 / 60
        assert slice_size == pytest.approx(expected, rel=0.01)

    def test_twap_slice_interval(self, twap_order):
        """Test calculo de intervalo entre slices"""
        interval = twap_order.get_slice_interval()
        expected = 3600 / 60  # 60 segundos
        assert interval == pytest.approx(expected, rel=0.01)

    def test_twap_progress(self, twap_order):
        """Test calculo de progreso"""
        assert twap_order.get_progress_pct() == 0.0
        
        twap_order.filled_quantity = 5.0
        assert twap_order.get_progress_pct() == 50.0

    def test_twap_to_dict(self, twap_order):
        """Test serializacion"""
        d = twap_order.to_dict()
        
        assert d["type"] == "twap"
        assert d["total_quantity"] == 10.0
        assert "slice_count" in d


class TestVWAPOrder:
    """Tests para VWAPOrder"""

    def test_vwap_creation(self, btc_price):
        """Test creacion basica de VWAP"""
        order = VWAPOrder(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            total_quantity=10.0,
            duration_seconds=3600,
            max_participation=0.05
        )
        assert order.total_quantity == 10.0
        assert order.max_participation == 0.05

    def test_vwap_calculate_slice_for_volume(self, vwap_order):
        """Test calculo de slice basado en volumen"""
        market_volume = 1000
        slice_qty = vwap_order.calculate_slice_for_volume(market_volume)
        
        # Max: 5% de 1000 = 50
        assert slice_qty <= 50
        # No mas que remaining
        assert slice_qty <= vwap_order.get_remaining_quantity()

    def test_vwap_progress(self, vwap_order):
        """Test calculo de progreso"""
        assert vwap_order.get_progress_pct() == 0.0
        
        vwap_order.filled_quantity = 2.5
        assert vwap_order.get_progress_pct() == 25.0

    def test_vwap_to_dict(self, vwap_order):
        """Test serializacion"""
        d = vwap_order.to_dict()
        
        assert d["type"] == "vwap"
        assert "max_participation" in d


class TestIcebergOrder:
    """Tests para IcebergOrder"""

    def test_iceberg_creation(self, btc_price):
        """Test creacion basica de Iceberg"""
        order = IcebergOrder(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            total_quantity=100.0,
            display_quantity=10.0,
            price=btc_price
        )
        assert order.total_quantity == 100.0
        assert order.display_quantity == 10.0
        assert order.price == btc_price

    def test_iceberg_display_validation(self, btc_price):
        """Test que display no puede ser mayor que total"""
        with pytest.raises(ValueError):
            IcebergOrder(
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                total_quantity=10.0,
                display_quantity=20.0,  # Invalid
                price=btc_price
            )

    def test_iceberg_current_display(self, iceberg_order):
        """Test cantidad visible actual"""
        # Inicial
        assert iceberg_order.get_current_display() == 10.0
        
        # Despues de fills parciales
        iceberg_order.filled_quantity = 95.0
        assert iceberg_order.get_current_display() == 5.0  # Solo quedan 5

    def test_iceberg_progress(self, iceberg_order):
        """Test calculo de progreso"""
        assert iceberg_order.get_progress_pct() == 0.0
        
        iceberg_order.filled_quantity = 50.0
        assert iceberg_order.get_progress_pct() == 50.0

    def test_iceberg_ccxt_params_binance(self, iceberg_order):
        """Test parametros CCXT para Binance"""
        params = iceberg_order.to_ccxt_params("binance")
        
        assert "icebergQty" in params["params"]
        assert params["params"]["icebergQty"] == 10.0

    def test_iceberg_to_dict(self, iceberg_order):
        """Test serializacion"""
        d = iceberg_order.to_dict()
        
        assert d["type"] == "iceberg"
        assert d["display_quantity"] == 10.0
        assert "reloads" in d


class TestHiddenOrder:
    """Tests para HiddenOrder"""

    def test_hidden_creation(self, btc_price):
        """Test creacion basica de Hidden"""
        order = HiddenOrder(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=10.0,
            price=btc_price
        )
        assert order.quantity == 10.0
        assert order.price == btc_price

    def test_hidden_remaining(self, hidden_order):
        """Test cantidad restante"""
        assert hidden_order.get_remaining_quantity() == 10.0
        
        hidden_order.filled_quantity = 3.0
        assert hidden_order.get_remaining_quantity() == 7.0

    def test_hidden_is_complete(self, hidden_order):
        """Test verificacion de completado"""
        assert not hidden_order.is_complete()
        
        hidden_order.filled_quantity = 10.0
        assert hidden_order.is_complete()

    def test_hidden_to_dict(self, hidden_order):
        """Test serializacion"""
        d = hidden_order.to_dict()
        
        assert d["type"] == "hidden"
        assert d["quantity"] == 10.0


class TestAlgoOrderSimulator:
    """Tests para AlgoOrderSimulator"""

    def test_simulator_create_twap(self, algo_simulator, twap_config):
        """Test crear orden TWAP"""
        order_id = algo_simulator.create_twap_order(
            "BTC/USDT", OrderSide.BUY, twap_config
        )
        
        assert order_id is not None
        assert order_id in algo_simulator.active_orders

    def test_simulator_create_vwap(self, algo_simulator, vwap_config):
        """Test crear orden VWAP"""
        order_id = algo_simulator.create_vwap_order(
            "BTC/USDT", OrderSide.BUY, vwap_config
        )
        
        assert order_id is not None
        assert algo_simulator.active_orders[order_id].algo_type == "vwap"

    def test_simulator_create_iceberg(self, algo_simulator, iceberg_config):
        """Test crear orden Iceberg"""
        order_id = algo_simulator.create_iceberg_order(
            "BTC/USDT", OrderSide.BUY, iceberg_config
        )
        
        assert order_id is not None
        assert algo_simulator.active_orders[order_id].algo_type == "iceberg"

    def test_simulator_create_hidden(self, algo_simulator, btc_price):
        """Test crear orden Hidden"""
        order_id = algo_simulator.create_hidden_order(
            "BTC/USDT", OrderSide.BUY, 10.0, btc_price
        )
        
        assert order_id is not None
        assert algo_simulator.active_orders[order_id].algo_type == "hidden"

    def test_simulator_twap_execution(self, algo_simulator, twap_config, btc_price):
        """Test ejecucion de TWAP"""
        order_id = algo_simulator.create_twap_order(
            "BTC/USDT", OrderSide.BUY, twap_config
        )
        
        # Procesar varios ticks
        executions = []
        for i in range(5):
            timestamp = datetime.now() + timedelta(minutes=i)
            result = algo_simulator.process_tick(
                "BTC/USDT", btc_price, timestamp, market_volume=1000000
            )
            executions.extend(result)
        
        # Debe haber algunas ejecuciones
        assert len(executions) > 0

    def test_simulator_iceberg_execution(self, algo_simulator, iceberg_config, btc_price):
        """Test ejecucion de Iceberg"""
        order_id = algo_simulator.create_iceberg_order(
            "BTC/USDT", OrderSide.BUY, iceberg_config
        )
        
        # Procesar tick con precio <= limit
        executions = algo_simulator.process_tick(
            "BTC/USDT", btc_price, datetime.now(),
            bid=btc_price - 5, ask=btc_price  # ask = price del iceberg
        )
        
        # Debe ejecutar display_quantity
        if executions:
            _, qty, _ = executions[0]
            assert qty <= iceberg_config.display_quantity

    def test_simulator_cancel_order(self, algo_simulator, twap_config):
        """Test cancelar orden"""
        order_id = algo_simulator.create_twap_order(
            "BTC/USDT", OrderSide.BUY, twap_config
        )
        
        result = algo_simulator.cancel_order(order_id)
        
        assert result == True
        assert order_id not in algo_simulator.active_orders

    def test_simulator_order_status(self, algo_simulator, twap_config):
        """Test obtener estado de orden"""
        order_id = algo_simulator.create_twap_order(
            "BTC/USDT", OrderSide.BUY, twap_config
        )
        
        status = algo_simulator.get_order_status(order_id)
        
        assert status is not None
        assert status["id"] == order_id
        assert status["algo_type"] == "twap"
        assert status["progress_pct"] == 0.0

    def test_simulator_statistics(self, algo_simulator, twap_config, vwap_config):
        """Test estadisticas del simulador"""
        algo_simulator.create_twap_order("BTC/USDT", OrderSide.BUY, twap_config)
        algo_simulator.create_vwap_order("ETH/USDT", OrderSide.SELL, vwap_config)
        
        stats = algo_simulator.get_statistics()
        
        assert stats["active_orders"] == 2
        assert stats["by_type"]["twap"] == 1
        assert stats["by_type"]["vwap"] == 1
