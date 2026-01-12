"""
Core module for broker_bridge.

Contiene interfaces, modelos, enums y excepciones.
"""

from .enums import (
    BrokerType,
    AssetClass,
    ConnectionStatus,
    OrderStatus,
    OrderSide,
    PositionSide,
    OrderType,
    TimeInForce,
)

from .models import (
    BrokerCapabilities,
    BrokerOrder,
    BrokerPosition,
    ExecutionReport,
    AccountInfo,
)

from .interfaces import IBrokerAdapter

from .exceptions import (
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
    TimeoutError,
    BrokerNotConnectedError,
    BrokerNotRegisteredError,
)

__all__ = [
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
    "TimeoutError",
    "BrokerNotConnectedError",
    "BrokerNotRegisteredError",
]
