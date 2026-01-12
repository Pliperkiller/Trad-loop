"""
Modelos de datos para el modulo broker_bridge.

Define las estructuras de datos unificadas para ordenes,
posiciones, capacidades y reportes de ejecucion.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any
import uuid

from .enums import (
    BrokerType, AssetClass, OrderStatus, OrderSide,
    PositionSide, OrderType, TimeInForce
)


@dataclass
class BrokerCapabilities:
    """
    Capacidades soportadas por un broker/exchange.

    Permite determinar que tipos de ordenes y features
    estan disponibles en cada broker.
    """
    broker_type: BrokerType
    exchange_id: str

    # Order types basicos
    supports_market: bool = True
    supports_limit: bool = True
    supports_stop_loss: bool = True
    supports_stop_limit: bool = True
    supports_take_profit: bool = True

    # Order types avanzados
    supports_trailing_stop: bool = False
    supports_bracket: bool = False
    supports_oco: bool = False
    supports_otoco: bool = False
    supports_twap: bool = False
    supports_vwap: bool = False
    supports_iceberg: bool = False
    supports_hidden: bool = False

    # Time in force
    supports_gtc: bool = True
    supports_fok: bool = False
    supports_ioc: bool = False
    supports_gtd: bool = False
    supports_day: bool = False

    # Features adicionales
    supports_reduce_only: bool = False
    supports_post_only: bool = False
    supports_hedging: bool = False
    supports_leverage: bool = False
    supports_margin: bool = False

    # Asset classes soportadas
    asset_classes: List[AssetClass] = field(default_factory=list)

    # Limites
    max_orders_per_second: float = 10.0
    min_order_size: Dict[str, float] = field(default_factory=dict)
    max_leverage: float = 1.0

    def supports_order_type(self, order_type: OrderType) -> bool:
        """Verificar si un tipo de orden esta soportado"""
        mapping = {
            OrderType.MARKET: self.supports_market,
            OrderType.LIMIT: self.supports_limit,
            OrderType.STOP_LOSS: self.supports_stop_loss,
            OrderType.STOP_LIMIT: self.supports_stop_limit,
            OrderType.TAKE_PROFIT: self.supports_take_profit,
            OrderType.TRAILING_STOP: self.supports_trailing_stop,
            OrderType.BRACKET: self.supports_bracket,
            OrderType.OCO: self.supports_oco,
            OrderType.OTOCO: self.supports_otoco,
            OrderType.TWAP: self.supports_twap,
            OrderType.VWAP: self.supports_vwap,
            OrderType.ICEBERG: self.supports_iceberg,
            OrderType.HIDDEN: self.supports_hidden,
        }
        return mapping.get(order_type, False)

    def supports_time_in_force(self, tif: TimeInForce) -> bool:
        """Verificar si un time in force esta soportado"""
        mapping = {
            TimeInForce.GTC: self.supports_gtc,
            TimeInForce.FOK: self.supports_fok,
            TimeInForce.IOC: self.supports_ioc,
            TimeInForce.GTD: self.supports_gtd,
            TimeInForce.DAY: self.supports_day,
        }
        return mapping.get(tif, False)


@dataclass
class BrokerOrder:
    """
    Orden unificada para cualquier broker.

    Representa una orden de trading con todos los parametros
    necesarios para cualquier tipo de orden y broker.
    """
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float

    # Identificadores
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    client_order_id: Optional[str] = None

    # Precios
    price: Optional[float] = None
    stop_price: Optional[float] = None

    # Trailing stop params
    trail_percent: Optional[float] = None
    trail_amount: Optional[float] = None
    activation_price: Optional[float] = None

    # Bracket/OCO params
    take_profit: Optional[float] = None
    stop_loss: Optional[float] = None

    # Iceberg params
    display_quantity: Optional[float] = None

    # Algo params
    duration_seconds: Optional[int] = None
    slice_count: Optional[int] = None
    max_participation: Optional[float] = None

    # Time in force
    time_in_force: TimeInForce = TimeInForce.GTC
    expire_time: Optional[datetime] = None

    # Flags
    reduce_only: bool = False
    post_only: bool = False

    # Leverage (para futuros)
    leverage: Optional[float] = None

    # Estado
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    average_price: float = 0.0

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None

    @property
    def remaining_quantity(self) -> float:
        """Cantidad restante por llenar"""
        return self.quantity - self.filled_quantity

    @property
    def is_filled(self) -> bool:
        """Verificar si la orden esta completamente llena"""
        return self.filled_quantity >= self.quantity

    @property
    def is_active(self) -> bool:
        """Verificar si la orden esta activa"""
        return self.status in [OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED]

    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario"""
        return {
            "id": self.id,
            "client_order_id": self.client_order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "order_type": self.order_type.value,
            "quantity": self.quantity,
            "price": self.price,
            "stop_price": self.stop_price,
            "trail_percent": self.trail_percent,
            "trail_amount": self.trail_amount,
            "take_profit": self.take_profit,
            "stop_loss": self.stop_loss,
            "time_in_force": self.time_in_force.value,
            "reduce_only": self.reduce_only,
            "post_only": self.post_only,
            "status": self.status.value,
            "filled_quantity": self.filled_quantity,
            "remaining_quantity": self.remaining_quantity,
            "average_price": self.average_price,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class BrokerPosition:
    """
    Posicion unificada.

    Representa una posicion abierta en cualquier broker.
    """
    symbol: str
    side: PositionSide
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float

    # Opcionales
    realized_pnl: float = 0.0
    leverage: float = 1.0
    margin: float = 0.0
    liquidation_price: Optional[float] = None

    # Metadata
    opened_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    @property
    def notional_value(self) -> float:
        """Valor nocional de la posicion"""
        return self.quantity * self.current_price

    @property
    def pnl_percent(self) -> float:
        """PnL en porcentaje"""
        if self.entry_price == 0:
            return 0.0
        if self.side == PositionSide.LONG:
            return ((self.current_price - self.entry_price) / self.entry_price) * 100
        else:
            return ((self.entry_price - self.current_price) / self.entry_price) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario"""
        return {
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "current_price": self.current_price,
            "unrealized_pnl": self.unrealized_pnl,
            "realized_pnl": self.realized_pnl,
            "leverage": self.leverage,
            "liquidation_price": self.liquidation_price,
            "pnl_percent": self.pnl_percent,
            "notional_value": self.notional_value,
        }


@dataclass
class ExecutionReport:
    """
    Reporte de ejecucion de orden.

    Contiene el resultado de enviar, modificar o cancelar una orden.
    """
    order_id: str
    status: OrderStatus
    filled_quantity: float
    remaining_quantity: float
    average_price: float
    commission: float
    timestamp: datetime

    # Opcionales
    client_order_id: Optional[str] = None
    trade_id: Optional[str] = None

    # Error info
    error_code: Optional[str] = None
    error_message: Optional[str] = None

    # Referencia a orden original
    original_order: Optional[BrokerOrder] = None

    @property
    def is_success(self) -> bool:
        """Verificar si la ejecucion fue exitosa"""
        return self.error_code is None and self.error_message is None

    @property
    def is_filled(self) -> bool:
        """Verificar si la orden fue llenada completamente"""
        return self.status == OrderStatus.FILLED

    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario"""
        return {
            "order_id": self.order_id,
            "client_order_id": self.client_order_id,
            "trade_id": self.trade_id,
            "status": self.status.value,
            "filled_quantity": self.filled_quantity,
            "remaining_quantity": self.remaining_quantity,
            "average_price": self.average_price,
            "commission": self.commission,
            "timestamp": self.timestamp.isoformat(),
            "is_success": self.is_success,
            "error_code": self.error_code,
            "error_message": self.error_message,
        }


@dataclass
class AccountInfo:
    """
    Informacion de cuenta.

    Contiene balances y estado general de la cuenta.
    """
    broker_type: BrokerType
    account_id: str

    # Balances
    total_balance: float = 0.0
    available_balance: float = 0.0
    margin_used: float = 0.0

    # Balances por activo
    balances: Dict[str, float] = field(default_factory=dict)

    # Posiciones
    total_unrealized_pnl: float = 0.0
    total_realized_pnl: float = 0.0

    # Limites
    max_leverage: float = 1.0
    current_leverage: float = 1.0

    # Estado
    is_active: bool = True
    last_updated: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario"""
        return {
            "broker_type": self.broker_type.value,
            "account_id": self.account_id,
            "total_balance": self.total_balance,
            "available_balance": self.available_balance,
            "margin_used": self.margin_used,
            "balances": self.balances,
            "total_unrealized_pnl": self.total_unrealized_pnl,
            "total_realized_pnl": self.total_realized_pnl,
            "is_active": self.is_active,
            "last_updated": self.last_updated.isoformat(),
        }
