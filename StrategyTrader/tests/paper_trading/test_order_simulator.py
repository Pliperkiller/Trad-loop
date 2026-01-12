"""
Tests para order_simulator.py
"""

import pytest
import asyncio
import sys
from pathlib import Path

src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from paper_trading.config import PaperTradingConfig, SlippageModel
from paper_trading.order_simulator import OrderSimulator
from paper_trading.models import Order, OrderSide, OrderType, OrderStatus


class TestOrderSimulator:
    """Tests para OrderSimulator"""

    @pytest.fixture
    def simulator(self, default_config):
        """Crea simulador con configuracion por defecto"""
        return OrderSimulator(default_config)

    @pytest.fixture
    def simulator_no_slippage(self):
        """Crea simulador sin slippage"""
        config = PaperTradingConfig(
            slippage_model=SlippageModel.NONE,
            commission_rate=0.001,
        )
        return OrderSimulator(config)

    def test_create_simulator(self, simulator):
        """Test crear simulador"""
        assert simulator.config is not None
        assert len(simulator.pending_orders) == 0
        assert len(simulator.order_history) == 0

    def test_update_market_state(self, simulator):
        """Test actualizar estado de mercado"""
        simulator.update_market_state("BTC/USDT", 50000)

        state = simulator.get_market_state("BTC/USDT")
        assert state is not None
        assert state.current_price == 50000
        assert state.bid_price < 50000
        assert state.ask_price > 50000

    @pytest.mark.asyncio
    async def test_submit_market_order(self, simulator_no_slippage):
        """Test enviar orden de mercado"""
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            type=OrderType.MARKET,
            quantity=0.04,  # $2000 < max $2500
        )

        result = await simulator_no_slippage.submit_order(order, market_price=50000)

        assert result.success == True
        assert result.order.status == OrderStatus.FILLED
        assert result.executed_quantity == 0.04
        assert result.executed_price > 0
        assert result.commission > 0

    @pytest.mark.asyncio
    async def test_submit_market_order_sell(self, simulator_no_slippage):
        """Test enviar orden de venta de mercado"""
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            type=OrderType.MARKET,
            quantity=0.04,  # $2000 < max $2500
        )

        result = await simulator_no_slippage.submit_order(order, market_price=50000)

        assert result.success == True
        assert result.order.status == OrderStatus.FILLED

    @pytest.mark.asyncio
    async def test_submit_limit_order_immediate_fill(self, simulator_no_slippage):
        """Test orden limite que se ejecuta inmediatamente"""
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            type=OrderType.LIMIT,
            quantity=0.04,  # $2000 < max $2500
            price=51000,  # Por encima del ask
        )

        result = await simulator_no_slippage.submit_order(order, market_price=50000)

        assert result.success == True
        assert result.order.status == OrderStatus.FILLED
        assert result.executed_price == 51000  # Ejecuta al precio limite

    @pytest.mark.asyncio
    async def test_submit_limit_order_pending(self, simulator_no_slippage):
        """Test orden limite que queda pendiente"""
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            type=OrderType.LIMIT,
            quantity=0.04,  # $2000 < max $2500
            price=48000,  # Por debajo del mercado
        )

        result = await simulator_no_slippage.submit_order(order, market_price=50000)

        assert result.success == True
        assert result.order.status == OrderStatus.SUBMITTED
        assert order.id in simulator_no_slippage.pending_orders

    @pytest.mark.asyncio
    async def test_reject_order_no_market_data(self, simulator):
        """Test rechazo por falta de datos de mercado"""
        order = Order(
            symbol="XYZ/USDT",  # No hay datos
            side=OrderSide.BUY,
            type=OrderType.MARKET,
            quantity=0.1,
        )

        result = await simulator.submit_order(order)  # Sin market_price

        assert result.success == False
        assert result.order.status == OrderStatus.REJECTED
        assert "mercado" in result.message.lower()

    @pytest.mark.asyncio
    async def test_reject_order_invalid_quantity(self, simulator):
        """Test rechazo por cantidad invalida"""
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            type=OrderType.MARKET,
            quantity=-0.1,  # Cantidad negativa
        )

        result = await simulator.submit_order(order, market_price=50000)

        assert result.success == False
        assert result.order.status == OrderStatus.REJECTED

    @pytest.mark.asyncio
    async def test_slippage_applied(self, simulator):
        """Test que se aplica slippage"""
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            type=OrderType.MARKET,
            quantity=0.04,  # $2000 < max $2500
        )

        simulator.update_market_state("BTC/USDT", 50000, volatility=0.05)
        result = await simulator.submit_order(order)

        # Con slippage el precio deberia ser mayor que el ask
        assert result.slippage >= 0

    @pytest.mark.asyncio
    async def test_check_pending_orders(self, simulator_no_slippage):
        """Test verificar ordenes pendientes"""
        # Crear orden limite pendiente
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            type=OrderType.LIMIT,
            quantity=0.04,  # $2000 < max $2500
            price=49000,
        )

        await simulator_no_slippage.submit_order(order, market_price=50000)
        assert order.id in simulator_no_slippage.pending_orders

        # El precio baja por debajo del limite - la orden se deberia ejecutar
        await simulator_no_slippage.check_pending_orders("BTC/USDT", 48500)

        # La orden deberia haberse ejecutado
        assert order.id not in simulator_no_slippage.pending_orders

    def test_cancel_order(self, simulator_no_slippage):
        """Test cancelar orden"""
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            type=OrderType.LIMIT,
            quantity=0.04,  # $2000 < max $2500
            price=49000,
        )

        # Simular que esta pendiente
        order.status = OrderStatus.SUBMITTED
        simulator_no_slippage.pending_orders[order.id] = order

        result = simulator_no_slippage.cancel_order(order.id)

        assert result == True
        assert order.id not in simulator_no_slippage.pending_orders
        assert order.status == OrderStatus.CANCELLED

    def test_cancel_nonexistent_order(self, simulator):
        """Test cancelar orden que no existe"""
        result = simulator.cancel_order("fake_id")
        assert result == False

    def test_cancel_all_orders(self, simulator_no_slippage):
        """Test cancelar todas las ordenes"""
        # Crear varias ordenes pendientes
        for i in range(5):
            order = Order(
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                type=OrderType.LIMIT,
                quantity=0.04,  # $2000 < max $2500
                price=49000 - i * 100,
            )
            order.status = OrderStatus.SUBMITTED
            simulator_no_slippage.pending_orders[order.id] = order

        assert len(simulator_no_slippage.pending_orders) == 5

        cancelled = simulator_no_slippage.cancel_all_orders()

        assert cancelled == 5
        assert len(simulator_no_slippage.pending_orders) == 0

    def test_get_order_history(self, simulator):
        """Test obtener historial de ordenes"""
        history = simulator.get_order_history()
        assert isinstance(history, list)

    @pytest.mark.asyncio
    async def test_order_callback(self, simulator_no_slippage):
        """Test callback cuando se ejecuta orden"""
        filled_orders = []

        def on_fill(result):
            filled_orders.append(result)

        simulator_no_slippage.on_order_filled = on_fill

        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            type=OrderType.MARKET,
            quantity=0.04,  # $2000 < max $2500
        )

        await simulator_no_slippage.submit_order(order, market_price=50000)

        assert len(filled_orders) == 1
        assert filled_orders[0].success == True
