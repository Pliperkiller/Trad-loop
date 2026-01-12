"""
Enums para el modulo broker_bridge.

Define tipos de broker, clases de activos, estados de conexion
y otros enums compartidos entre adaptadores.
"""

from enum import Enum


class BrokerType(Enum):
    """Tipos de broker soportados"""
    CCXT = "ccxt"       # Crypto exchanges via CCXT
    IBKR = "ibkr"       # Interactive Brokers
    PAPER = "paper"     # Paper trading (simulacion)


class AssetClass(Enum):
    """Clases de activos"""
    CRYPTO = "crypto"
    STOCK = "stock"
    INDEX = "index"
    FOREX = "forex"
    FUTURES = "futures"
    OPTIONS = "options"


class ConnectionStatus(Enum):
    """Estado de conexion del broker"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


class OrderStatus(Enum):
    """Estado de una orden"""
    PENDING = "pending"
    OPEN = "open"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class OrderSide(Enum):
    """Lado de la orden"""
    BUY = "buy"
    SELL = "sell"


class PositionSide(Enum):
    """Lado de la posicion"""
    LONG = "long"
    SHORT = "short"
    BOTH = "both"  # Para modo hedge


class OrderType(Enum):
    """Tipos de orden"""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    STOP_LIMIT = "stop_limit"
    TAKE_PROFIT = "take_profit"
    TRAILING_STOP = "trailing_stop"
    BRACKET = "bracket"
    TWAP = "twap"
    VWAP = "vwap"
    ICEBERG = "iceberg"
    HIDDEN = "hidden"
    OCO = "oco"
    OTOCO = "otoco"


class TimeInForce(Enum):
    """Vigencia de la orden"""
    GTC = "good_till_cancel"
    IOC = "immediate_or_cancel"
    FOK = "fill_or_kill"
    GTD = "good_till_date"
    DAY = "day"
