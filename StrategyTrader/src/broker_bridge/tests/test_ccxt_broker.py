"""
Tests para CCXTBroker.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

import sys
from pathlib import Path
src_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_path))

from broker_bridge.core.enums import OrderType, OrderSide, OrderStatus
from broker_bridge.core.models import BrokerOrder
from broker_bridge.adapters.ccxt.ccxt_broker import CCXTBroker
from broker_bridge.adapters.ccxt.ccxt_capabilities import (
    get_exchange_capabilities, EXCHANGE_CAPABILITIES
)
from broker_bridge.adapters.ccxt.ccxt_order_mapper import CCXTOrderMapper


class TestCCXTCapabilities:
    """Tests para capacidades de exchanges"""

    def test_get_binance_capabilities(self):
        """Test capacidades de Binance"""
        caps = get_exchange_capabilities("binance")

        assert caps.supports_trailing_stop == True
        assert caps.supports_oco == True
        assert caps.supports_iceberg == True
        assert caps.supports_bracket == False

    def test_get_bybit_capabilities(self):
        """Test capacidades de Bybit"""
        caps = get_exchange_capabilities("bybit")

        assert caps.supports_trailing_stop == True
        assert caps.supports_bracket == True
        assert caps.supports_oco == False

    def test_get_okx_capabilities(self):
        """Test capacidades de OKX"""
        caps = get_exchange_capabilities("okx")

        assert caps.supports_twap == True
        assert caps.supports_vwap == True
        assert caps.supports_oco == True

    def test_unknown_exchange_uses_defaults(self):
        """Test que exchange desconocido usa defaults"""
        caps = get_exchange_capabilities("unknown_exchange")

        assert caps.supports_trailing_stop == False
        assert caps.supports_market == True
        assert caps.supports_limit == True

    def test_exchange_capabilities_dict(self):
        """Test que el diccionario de capacidades existe"""
        assert "binance" in EXCHANGE_CAPABILITIES
        assert "bybit" in EXCHANGE_CAPABILITIES
        assert "okx" in EXCHANGE_CAPABILITIES


class TestCCXTOrderMapper:
    """Tests para CCXTOrderMapper"""

    def test_to_ccxt_order_type(self):
        """Test conversion de tipos de orden"""
        mapper = CCXTOrderMapper("binance")

        assert mapper.to_ccxt_order_type(OrderType.MARKET) == "market"
        assert mapper.to_ccxt_order_type(OrderType.LIMIT) == "limit"
        assert mapper.to_ccxt_order_type(OrderType.STOP_LOSS) == "stop_loss"
        assert mapper.to_ccxt_order_type(OrderType.TRAILING_STOP) == "trailing_stop_market"

    def test_to_ccxt_side(self):
        """Test conversion de lado"""
        mapper = CCXTOrderMapper("binance")

        assert mapper.to_ccxt_side(OrderSide.BUY) == "buy"
        assert mapper.to_ccxt_side(OrderSide.SELL) == "sell"

    def test_trailing_params_binance(self, sample_trailing_stop_order):
        """Test parametros de trailing stop para Binance"""
        mapper = CCXTOrderMapper("binance")
        params = mapper.to_ccxt_params(sample_trailing_stop_order)

        assert "trailingDelta" in params
        assert params["trailingDelta"] == 200  # 2% = 200 bps

    def test_trailing_params_bybit(self, sample_trailing_stop_order):
        """Test parametros de trailing stop para Bybit"""
        mapper = CCXTOrderMapper("bybit")
        params = mapper.to_ccxt_params(sample_trailing_stop_order)

        assert "trailingStop" in params
        assert "triggerDirection" in params

    def test_reduce_only_params(self, sample_limit_order):
        """Test parametros reduce only"""
        sample_limit_order.reduce_only = True
        mapper = CCXTOrderMapper("binanceusdm")
        params = mapper.to_ccxt_params(sample_limit_order)

        assert params.get("reduceOnly") == True

    def test_post_only_params_binance(self, sample_limit_order):
        """Test parametros post only para Binance"""
        sample_limit_order.post_only = True
        mapper = CCXTOrderMapper("binance")
        params = mapper.to_ccxt_params(sample_limit_order)

        assert params.get("timeInForce") == "GTX"

    def test_from_ccxt_order(self):
        """Test conversion de orden CCXT a BrokerOrder"""
        mapper = CCXTOrderMapper("binance")
        ccxt_order = {
            "id": "123",
            "symbol": "BTC/USDT",
            "side": "buy",
            "type": "limit",
            "amount": 0.1,
            "price": 50000,
            "filled": 0,
            "remaining": 0.1,
            "status": "open",
        }

        order = mapper.from_ccxt_order(ccxt_order)

        assert order.id == "123"
        assert order.symbol == "BTC/USDT"
        assert order.side == OrderSide.BUY
        assert order.quantity == 0.1
        assert order.status == OrderStatus.OPEN

    def test_to_execution_report(self):
        """Test creacion de ExecutionReport"""
        mapper = CCXTOrderMapper("binance")
        ccxt_result = {
            "id": "123",
            "status": "closed",
            "filled": 0.1,
            "remaining": 0,
            "average": 50100,
            "fee": {"cost": 0.05},
        }

        report = mapper.to_execution_report(ccxt_result)

        assert report.order_id == "123"
        assert report.status == OrderStatus.FILLED
        assert report.filled_quantity == 0.1
        assert report.commission == 0.05


class TestCCXTBroker:
    """Tests para CCXTBroker"""

    @pytest.fixture
    def ccxt_broker(self, mock_ccxt_exchange):
        """CCXTBroker con mock"""
        with patch('broker_bridge.adapters.ccxt.ccxt_broker.ccxt') as mock_ccxt:
            mock_ccxt.binance = MagicMock(return_value=mock_ccxt_exchange)
            broker = CCXTBroker("binance", "test_key", "test_secret")
            broker._exchange = mock_ccxt_exchange
            broker._connected = True
            return broker

    @pytest.mark.asyncio
    async def test_connect_success(self, mock_ccxt_exchange):
        """Test conexion exitosa"""
        with patch('broker_bridge.adapters.ccxt.ccxt_broker.ccxt') as mock_ccxt:
            mock_ccxt.binance = MagicMock(return_value=mock_ccxt_exchange)
            broker = CCXTBroker("binance", "test_key", "test_secret")

            result = await broker.connect()

            assert result == True
            assert broker.is_connected == True

    @pytest.mark.asyncio
    async def test_disconnect(self, ccxt_broker, mock_ccxt_exchange):
        """Test desconexion"""
        await ccxt_broker.disconnect()

        assert ccxt_broker.is_connected == False
        mock_ccxt_exchange.close.assert_called_once()

    def test_get_capabilities(self, ccxt_broker):
        """Test obtener capacidades"""
        caps = ccxt_broker.get_capabilities()

        assert caps.broker_type.value == "ccxt"
        assert caps.exchange_id == "binance"
        assert caps.supports_trailing_stop == True

    @pytest.mark.asyncio
    async def test_submit_market_order(self, ccxt_broker, sample_market_order, mock_ccxt_exchange):
        """Test enviar orden de mercado"""
        report = await ccxt_broker.submit_order(sample_market_order)

        assert report.order_id == "order-123"
        assert report.status == OrderStatus.OPEN
        mock_ccxt_exchange.create_order.assert_called_once()

    @pytest.mark.asyncio
    async def test_submit_limit_order(self, ccxt_broker, sample_limit_order, mock_ccxt_exchange):
        """Test enviar orden limite"""
        report = await ccxt_broker.submit_order(sample_limit_order)

        assert report.order_id == "order-123"
        mock_ccxt_exchange.create_order.assert_called_once()
        call_args = mock_ccxt_exchange.create_order.call_args
        assert call_args[1]["type"] == "limit"
        assert call_args[1]["price"] == 50000

    @pytest.mark.asyncio
    async def test_cancel_order_success(self, ccxt_broker, mock_ccxt_exchange):
        """Test cancelar orden exitoso"""
        result = await ccxt_broker.cancel_order("order-123", "BTC/USDT")

        assert result == True
        mock_ccxt_exchange.cancel_order.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_balance(self, ccxt_broker, mock_ccxt_exchange):
        """Test obtener balance"""
        balance = await ccxt_broker.get_balance()

        assert "USDT" in balance
        assert "BTC" in balance
        assert balance["USDT"] == 10000

    @pytest.mark.asyncio
    async def test_get_ticker(self, ccxt_broker, mock_ccxt_exchange):
        """Test obtener ticker"""
        ticker = await ccxt_broker.get_ticker("BTC/USDT")

        assert ticker["bid"] == 49990
        assert ticker["ask"] == 50010
        assert ticker["last"] == 50000

    @pytest.mark.asyncio
    async def test_get_orderbook(self, ccxt_broker, mock_ccxt_exchange):
        """Test obtener order book"""
        orderbook = await ccxt_broker.get_orderbook("BTC/USDT")

        assert "bids" in orderbook
        assert "asks" in orderbook
        assert len(orderbook["bids"]) == 2

    @pytest.mark.asyncio
    async def test_get_open_orders(self, ccxt_broker, mock_ccxt_exchange):
        """Test obtener ordenes abiertas"""
        orders = await ccxt_broker.get_open_orders("BTC/USDT")

        assert isinstance(orders, list)
        mock_ccxt_exchange.fetch_open_orders.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_positions_empty(self, ccxt_broker, mock_ccxt_exchange):
        """Test obtener posiciones vacias"""
        positions = await ccxt_broker.get_positions()

        assert positions == []

    @pytest.mark.asyncio
    async def test_context_manager(self, mock_ccxt_exchange):
        """Test uso como context manager"""
        with patch('broker_bridge.adapters.ccxt.ccxt_broker.ccxt') as mock_ccxt:
            mock_ccxt.binance = MagicMock(return_value=mock_ccxt_exchange)

            async with CCXTBroker("binance", "key", "secret") as broker:
                assert broker.is_connected == True

            mock_ccxt_exchange.close.assert_called()
