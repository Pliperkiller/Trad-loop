"""
Broker Bridge - Arquitectura Multi-Broker Generica

Proporciona una interfaz unificada para operar con multiples brokers:
- CCXT: 100+ exchanges de criptomonedas
- IBKR: Interactive Brokers para mercados tradicionales

Uso basico:
    from broker_bridge import UnifiedExecutor, CCXTBroker, IBKRBroker, BrokerOrder

    executor = UnifiedExecutor()
    executor.register_broker(CCXTBroker("binance", api_key="...", api_secret="..."))
    executor.register_broker(IBKRBroker(host="127.0.0.1", port=7497))

    await executor.connect_all()

    # Crypto -> CCXT automaticamente
    await executor.submit_order(BrokerOrder(symbol="BTC/USDT", ...))

    # Stock -> IBKR automaticamente
    await executor.submit_order(BrokerOrder(symbol="AAPL", ...))
"""

# Core
from .core.enums import (
    BrokerType,
    AssetClass,
    ConnectionStatus,
    OrderStatus,
    OrderSide,
    PositionSide,
    OrderType,
    TimeInForce,
)

from .core.models import (
    BrokerCapabilities,
    BrokerOrder,
    BrokerPosition,
    ExecutionReport,
    AccountInfo,
)

from .core.interfaces import IBrokerAdapter

from .core.exceptions import (
    BrokerError,
    BrokerConnectionError,
    AuthenticationError,
    OrderError,
    InsufficientFundsError,
    OrderNotFoundError,
    OrderRejectedError,
    UnsupportedOrderTypeError,
    InvalidOrderError,
    RateLimitError,
    PositionError,
    PositionNotFoundError,
    MarketClosedError,
    SymbolNotFoundError,
    ContractQualificationError,
    BrokerNotConnectedError,
    BrokerNotRegisteredError,
)

# Adapters
from .adapters.ccxt import CCXTBroker
from .adapters.ibkr import IBKRBroker

# Execution
from .execution.symbol_router import SymbolRouter
from .execution.unified_executor import UnifiedExecutor
from .execution.fallback_simulator import FallbackSimulator

__version__ = "0.1.0"

__all__ = [
    # Version
    "__version__",
    # Enums
    "BrokerType",
    "AssetClass",
    "ConnectionStatus",
    "OrderStatus",
    "OrderSide",
    "PositionSide",
    "OrderType",
    "TimeInForce",
    # Models
    "BrokerCapabilities",
    "BrokerOrder",
    "BrokerPosition",
    "ExecutionReport",
    "AccountInfo",
    # Interfaces
    "IBrokerAdapter",
    # Exceptions
    "BrokerError",
    "BrokerConnectionError",
    "AuthenticationError",
    "OrderError",
    "InsufficientFundsError",
    "OrderNotFoundError",
    "OrderRejectedError",
    "UnsupportedOrderTypeError",
    "InvalidOrderError",
    "RateLimitError",
    "PositionError",
    "PositionNotFoundError",
    "MarketClosedError",
    "SymbolNotFoundError",
    "ContractQualificationError",
    "BrokerNotConnectedError",
    "BrokerNotRegisteredError",
    # Adapters
    "CCXTBroker",
    "IBKRBroker",
    # Execution
    "SymbolRouter",
    "UnifiedExecutor",
    "FallbackSimulator",
]
