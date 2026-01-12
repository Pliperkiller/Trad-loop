"""
Tests para UnifiedExecutor.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import sys
from pathlib import Path

src_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_path))

from broker_bridge.core.enums import BrokerType, AssetClass, OrderType, OrderSide, OrderStatus
from broker_bridge.core.models import BrokerOrder, BrokerPosition, ExecutionReport
from broker_bridge.core.exceptions import BrokerNotRegisteredError, BrokerNotConnectedError
from broker_bridge.execution.unified_executor import UnifiedExecutor
from broker_bridge.execution.symbol_router import SymbolRouter


class TestUnifiedExecutor:
    """Tests para UnifiedExecutor"""

    @pytest.fixture
    def mock_ccxt_broker(self):
        """Mock de CCXTBroker"""
        broker = AsyncMock()
        broker.broker_type = BrokerType.CCXT
        broker.broker_id = "binance"
        broker.is_connected = True
        broker.connect = AsyncMock(return_value=True)
        broker.disconnect = AsyncMock()
        broker.get_capabilities = MagicMock(return_value=MagicMock(
            supports_order_type=MagicMock(return_value=True)
        ))
        broker.submit_order = AsyncMock(return_value=ExecutionReport(
            order_id="ccxt-123",
            status=OrderStatus.OPEN,
            filled_quantity=0,
            remaining_quantity=0.1,
            average_price=0,
            commission=0,
            timestamp=MagicMock()
        ))
        broker.cancel_order = AsyncMock(return_value=True)
        broker.get_positions = AsyncMock(return_value=[])
        broker.get_balance = AsyncMock(return_value={"USDT": 10000})
        broker.get_ticker = AsyncMock(return_value={"last": 50000})
        broker.get_orderbook = AsyncMock(return_value={"bids": [], "asks": []})
        broker.get_account_info = AsyncMock()
        return broker

    @pytest.fixture
    def mock_ibkr_broker(self):
        """Mock de IBKRBroker"""
        broker = AsyncMock()
        broker.broker_type = BrokerType.IBKR
        broker.broker_id = "ibkr"
        broker.is_connected = True
        broker.connect = AsyncMock(return_value=True)
        broker.disconnect = AsyncMock()
        broker.get_capabilities = MagicMock(return_value=MagicMock(
            supports_order_type=MagicMock(return_value=True)
        ))
        broker.submit_order = AsyncMock(return_value=ExecutionReport(
            order_id="ibkr-456",
            status=OrderStatus.OPEN,
            filled_quantity=0,
            remaining_quantity=10,
            average_price=0,
            commission=0,
            timestamp=MagicMock()
        ))
        broker.cancel_order = AsyncMock(return_value=True)
        broker.get_positions = AsyncMock(return_value=[])
        broker.get_balance = AsyncMock(return_value={"USD": 100000})
        broker.get_ticker = AsyncMock(return_value={"last": 150})
        broker.get_orderbook = AsyncMock(return_value={"bids": [], "asks": []})
        broker.get_account_info = AsyncMock()
        return broker

    @pytest.fixture
    def executor(self, mock_ccxt_broker, mock_ibkr_broker):
        """Executor con brokers registrados"""
        exec = UnifiedExecutor()
        exec.register_broker(mock_ccxt_broker)
        exec.register_broker(mock_ibkr_broker)
        return exec

    # ==================== Registration Tests ====================

    def test_register_broker(self, mock_ccxt_broker):
        """Test registrar broker"""
        executor = UnifiedExecutor()
        executor.register_broker(mock_ccxt_broker)

        assert BrokerType.CCXT in executor.get_registered_brokers()

    def test_unregister_broker(self, executor):
        """Test desregistrar broker"""
        executor.unregister_broker(BrokerType.CCXT)

        assert BrokerType.CCXT not in executor.get_registered_brokers()

    def test_get_broker(self, executor, mock_ccxt_broker):
        """Test obtener broker"""
        broker = executor.get_broker(BrokerType.CCXT)

        assert broker == mock_ccxt_broker

    def test_get_nonexistent_broker(self, executor):
        """Test obtener broker no registrado"""
        broker = executor.get_broker(BrokerType.PAPER)

        assert broker is None

    # ==================== Connection Tests ====================

    @pytest.mark.asyncio
    async def test_connect_all(self, executor, mock_ccxt_broker, mock_ibkr_broker):
        """Test conectar todos los brokers"""
        results = await executor.connect_all()

        assert results[BrokerType.CCXT] == True
        assert results[BrokerType.IBKR] == True
        mock_ccxt_broker.connect.assert_called_once()
        mock_ibkr_broker.connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect_all(self, executor, mock_ccxt_broker, mock_ibkr_broker):
        """Test desconectar todos los brokers"""
        await executor.disconnect_all()

        mock_ccxt_broker.disconnect.assert_called_once()
        mock_ibkr_broker.disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_single_broker(self, executor, mock_ccxt_broker):
        """Test conectar un broker especifico"""
        result = await executor.connect_broker(BrokerType.CCXT)

        assert result == True
        mock_ccxt_broker.connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_unregistered_broker(self, executor):
        """Test conectar broker no registrado"""
        with pytest.raises(BrokerNotRegisteredError):
            await executor.connect_broker(BrokerType.PAPER)

    # ==================== Order Execution Tests ====================

    @pytest.mark.asyncio
    async def test_submit_crypto_order_auto_route(self, executor, mock_ccxt_broker):
        """Test enviar orden crypto con ruteo automatico"""
        order = BrokerOrder(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.1,
            price=50000,
        )

        report = await executor.submit_order(order)

        assert report.order_id == "ccxt-123"
        mock_ccxt_broker.submit_order.assert_called()

    @pytest.mark.asyncio
    async def test_submit_stock_order_auto_route(self, executor, mock_ibkr_broker):
        """Test enviar orden de acciones con ruteo automatico"""
        order = BrokerOrder(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=10,
            price=150,
        )

        report = await executor.submit_order(order)

        assert report.order_id == "ibkr-456"
        mock_ibkr_broker.submit_order.assert_called()

    @pytest.mark.asyncio
    async def test_submit_order_explicit_broker(self, executor, mock_ccxt_broker):
        """Test enviar orden a broker especifico"""
        order = BrokerOrder(
            symbol="TEST",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1,
        )

        report = await executor.submit_order(order, broker_type=BrokerType.CCXT)

        mock_ccxt_broker.submit_order.assert_called()

    @pytest.mark.asyncio
    async def test_submit_order_unregistered_broker(self, executor):
        """Test enviar orden a broker no registrado"""
        order = BrokerOrder(
            symbol="TEST",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1,
        )

        with pytest.raises(BrokerNotRegisteredError):
            await executor.submit_order(order, broker_type=BrokerType.PAPER)

    @pytest.mark.asyncio
    async def test_cancel_order(self, executor, mock_ccxt_broker):
        """Test cancelar orden"""
        order = BrokerOrder(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.1,
            price=50000,
        )
        report = await executor.submit_order(order)

        result = await executor.cancel_order(report.order_id)

        # Should try to cancel in fallback first, then broker
        assert result == True

    # ==================== Position Tests ====================

    @pytest.mark.asyncio
    async def test_get_all_positions(self, executor, mock_ccxt_broker, mock_ibkr_broker):
        """Test obtener posiciones de todos los brokers"""
        positions = await executor.get_all_positions()

        assert BrokerType.CCXT in positions
        assert BrokerType.IBKR in positions

    @pytest.mark.asyncio
    async def test_get_positions_single_broker(self, executor, mock_ccxt_broker):
        """Test obtener posiciones de un broker"""
        positions = await executor.get_positions(BrokerType.CCXT)

        assert isinstance(positions, list)
        mock_ccxt_broker.get_positions.assert_called()

    # ==================== Balance Tests ====================

    @pytest.mark.asyncio
    async def test_get_all_balances(self, executor):
        """Test obtener balances de todos los brokers"""
        balances = await executor.get_all_balances()

        assert BrokerType.CCXT in balances
        assert BrokerType.IBKR in balances
        assert balances[BrokerType.CCXT]["USDT"] == 10000
        assert balances[BrokerType.IBKR]["USD"] == 100000

    @pytest.mark.asyncio
    async def test_get_balance_single_broker(self, executor, mock_ccxt_broker):
        """Test obtener balance de un broker"""
        balance = await executor.get_balance(BrokerType.CCXT)

        assert balance["USDT"] == 10000

    # ==================== Market Data Tests ====================

    @pytest.mark.asyncio
    async def test_get_ticker_crypto(self, executor, mock_ccxt_broker):
        """Test obtener ticker crypto"""
        ticker = await executor.get_ticker("BTC/USDT")

        assert ticker["last"] == 50000
        mock_ccxt_broker.get_ticker.assert_called_with("BTC/USDT")

    @pytest.mark.asyncio
    async def test_get_ticker_stock(self, executor, mock_ibkr_broker):
        """Test obtener ticker de acciones"""
        ticker = await executor.get_ticker("AAPL")

        assert ticker["last"] == 150
        mock_ibkr_broker.get_ticker.assert_called_with("AAPL")

    @pytest.mark.asyncio
    async def test_get_orderbook(self, executor, mock_ccxt_broker):
        """Test obtener order book"""
        orderbook = await executor.get_orderbook("BTC/USDT")

        assert "bids" in orderbook
        assert "asks" in orderbook

    # ==================== Utility Tests ====================

    def test_get_router(self, executor):
        """Test obtener router"""
        router = executor.get_router()

        assert isinstance(router, SymbolRouter)

    def test_get_execution_history(self, executor):
        """Test obtener historial de ejecuciones"""
        history = executor.get_execution_history()

        assert isinstance(history, list)

    # ==================== Context Manager Tests ====================

    @pytest.mark.asyncio
    async def test_context_manager(self, mock_ccxt_broker, mock_ibkr_broker):
        """Test uso como context manager"""
        executor = UnifiedExecutor()
        executor.register_broker(mock_ccxt_broker)
        executor.register_broker(mock_ibkr_broker)

        async with executor:
            mock_ccxt_broker.connect.assert_called()
            mock_ibkr_broker.connect.assert_called()

        mock_ccxt_broker.disconnect.assert_called()
        mock_ibkr_broker.disconnect.assert_called()
