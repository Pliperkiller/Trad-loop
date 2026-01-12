"""
Fixtures compartidos para tests de broker_bridge.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_path))

from broker_bridge.core.enums import (
    BrokerType, AssetClass, OrderStatus, OrderSide, OrderType, TimeInForce
)
from broker_bridge.core.models import (
    BrokerCapabilities, BrokerOrder, BrokerPosition, ExecutionReport, AccountInfo
)


# ============================================================================
# Mock CCXT Exchange
# ============================================================================

@pytest.fixture
def mock_ccxt_exchange():
    """Mock de exchange CCXT"""
    exchange = AsyncMock()

    # Markets
    exchange.load_markets = AsyncMock(return_value={
        "BTC/USDT": {"symbol": "BTC/USDT", "base": "BTC", "quote": "USDT"},
        "ETH/USDT": {"symbol": "ETH/USDT", "base": "ETH", "quote": "USDT"},
    })

    # Orders
    exchange.create_order = AsyncMock(return_value={
        "id": "order-123",
        "status": "open",
        "filled": 0,
        "remaining": 1.0,
        "average": None,
        "fee": {"cost": 0.1, "currency": "USDT"}
    })
    exchange.cancel_order = AsyncMock(return_value=True)
    exchange.fetch_order = AsyncMock(return_value={
        "id": "order-123",
        "status": "open",
        "symbol": "BTC/USDT",
        "side": "buy",
        "type": "limit",
        "amount": 1.0,
        "price": 50000,
        "filled": 0,
        "remaining": 1.0,
    })
    exchange.fetch_open_orders = AsyncMock(return_value=[])
    exchange.fetch_closed_orders = AsyncMock(return_value=[])

    # Balance
    exchange.fetch_balance = AsyncMock(return_value={
        "USDT": {"free": 10000, "used": 0, "total": 10000},
        "BTC": {"free": 0.5, "used": 0, "total": 0.5},
        "total": {"USDT": 10000, "BTC": 0.5},
        "free": {"USDT": 10000, "BTC": 0.5},
        "used": {"USDT": 0, "BTC": 0},
    })

    # Positions (futures)
    exchange.fetch_positions = AsyncMock(return_value=[])

    # Market data
    exchange.fetch_ticker = AsyncMock(return_value={
        "symbol": "BTC/USDT",
        "bid": 49990,
        "ask": 50010,
        "last": 50000,
        "baseVolume": 1000,
        "high": 51000,
        "low": 49000,
        "percentage": 2.5,
    })
    exchange.fetch_order_book = AsyncMock(return_value={
        "bids": [[49990, 1.0], [49980, 2.0]],
        "asks": [[50010, 1.0], [50020, 2.0]],
    })

    # Close
    exchange.close = AsyncMock()

    return exchange


# ============================================================================
# Mock IB
# ============================================================================

@pytest.fixture
def mock_ib():
    """Mock de Interactive Brokers"""
    ib = MagicMock()

    # Connection
    ib.isConnected.return_value = True
    ib.connectAsync = AsyncMock(return_value=None)
    ib.disconnect = MagicMock()

    # Contracts
    mock_contract = MagicMock()
    mock_contract.symbol = "AAPL"
    ib.qualifyContractsAsync = AsyncMock(return_value=[mock_contract])

    # Orders
    mock_trade = MagicMock()
    mock_trade.order.orderId = 12345
    mock_trade.order.orderRef = "client-123"
    mock_trade.order.action = "BUY"
    mock_trade.order.totalQuantity = 10
    mock_trade.orderStatus.status = "Submitted"
    mock_trade.orderStatus.filled = 0
    mock_trade.orderStatus.remaining = 10
    mock_trade.orderStatus.avgFillPrice = 0
    mock_trade.fills = []
    mock_trade.contract = mock_contract

    ib.placeOrder = MagicMock(return_value=mock_trade)
    ib.cancelOrder = MagicMock()
    ib.openTrades = MagicMock(return_value=[mock_trade])

    # Positions
    mock_position = MagicMock()
    mock_position.contract = mock_contract
    mock_position.position = 100
    mock_position.avgCost = 150.0
    ib.positions = MagicMock(return_value=[mock_position])

    # Account
    mock_account_value = MagicMock()
    mock_account_value.tag = "CashBalance"
    mock_account_value.currency = "USD"
    mock_account_value.value = "100000"
    mock_account_value.account = "DU123456"
    ib.accountValues = MagicMock(return_value=[mock_account_value])
    ib.accountSummary = MagicMock(return_value=[])

    # Market data
    mock_ticker = MagicMock()
    mock_ticker.bid = 150.5
    mock_ticker.ask = 150.6
    mock_ticker.last = 150.55
    mock_ticker.close = 149.0
    mock_ticker.high = 152.0
    mock_ticker.low = 148.0
    mock_ticker.volume = 1000000
    mock_ticker.domBids = []
    mock_ticker.domAsks = []
    ib.reqMktData = MagicMock(return_value=mock_ticker)
    ib.reqMktDepth = MagicMock()
    ib.ticker = MagicMock(return_value=mock_ticker)

    return ib


# ============================================================================
# Sample Orders
# ============================================================================

@pytest.fixture
def sample_market_order():
    """Orden de mercado de ejemplo"""
    return BrokerOrder(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=0.1,
    )


@pytest.fixture
def sample_limit_order():
    """Orden limite de ejemplo"""
    return BrokerOrder(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=0.1,
        price=50000,
    )


@pytest.fixture
def sample_trailing_stop_order():
    """Orden trailing stop de ejemplo"""
    return BrokerOrder(
        symbol="BTC/USDT",
        side=OrderSide.SELL,
        order_type=OrderType.TRAILING_STOP,
        quantity=0.1,
        trail_percent=0.02,
    )


@pytest.fixture
def sample_bracket_order():
    """Orden bracket de ejemplo"""
    return BrokerOrder(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        order_type=OrderType.BRACKET,
        quantity=0.1,
        price=50000,
        stop_loss=49000,
        take_profit=52000,
    )


@pytest.fixture
def sample_stock_order():
    """Orden de acciones de ejemplo"""
    return BrokerOrder(
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=10,
        price=150.0,
    )


# ============================================================================
# Sample Positions
# ============================================================================

@pytest.fixture
def sample_long_position():
    """Posicion long de ejemplo"""
    from broker_bridge.core.enums import PositionSide
    return BrokerPosition(
        symbol="BTC/USDT",
        side=PositionSide.LONG,
        quantity=0.5,
        entry_price=48000,
        current_price=50000,
        unrealized_pnl=1000,
    )


@pytest.fixture
def sample_short_position():
    """Posicion short de ejemplo"""
    from broker_bridge.core.enums import PositionSide
    return BrokerPosition(
        symbol="ETH/USDT",
        side=PositionSide.SHORT,
        quantity=2.0,
        entry_price=3500,
        current_price=3400,
        unrealized_pnl=200,
    )


# ============================================================================
# Capabilities
# ============================================================================

@pytest.fixture
def binance_capabilities():
    """Capacidades de Binance"""
    return BrokerCapabilities(
        broker_type=BrokerType.CCXT,
        exchange_id="binance",
        supports_trailing_stop=True,
        supports_oco=True,
        supports_iceberg=True,
        supports_fok=True,
        supports_ioc=True,
        asset_classes=[AssetClass.CRYPTO],
    )


@pytest.fixture
def ibkr_capabilities():
    """Capacidades de IBKR"""
    return BrokerCapabilities(
        broker_type=BrokerType.IBKR,
        exchange_id="ibkr",
        supports_trailing_stop=True,
        supports_bracket=True,
        supports_oco=True,
        supports_fok=True,
        supports_ioc=True,
        supports_gtd=True,
        supports_day=True,
        asset_classes=[
            AssetClass.STOCK,
            AssetClass.INDEX,
            AssetClass.FOREX,
            AssetClass.FUTURES,
            AssetClass.OPTIONS,
        ],
    )


# ============================================================================
# Execution Reports
# ============================================================================

@pytest.fixture
def sample_execution_report():
    """Reporte de ejecucion de ejemplo"""
    return ExecutionReport(
        order_id="order-123",
        status=OrderStatus.FILLED,
        filled_quantity=0.1,
        remaining_quantity=0,
        average_price=50000,
        commission=0.05,
        timestamp=datetime.now(),
    )


# ============================================================================
# Account Info
# ============================================================================

@pytest.fixture
def sample_account_info():
    """Informacion de cuenta de ejemplo"""
    return AccountInfo(
        broker_type=BrokerType.CCXT,
        account_id="binance-main",
        total_balance=10000,
        available_balance=8000,
        margin_used=2000,
        balances={"USDT": 8000, "BTC": 0.1},
    )
