"""
Modelos de datos para Paper Trading

Define las estructuras de datos fundamentales:
- Order: Representa una orden de trading
- OrderResult: Resultado de ejecucion de orden
- PaperTradingState: Estado actual del sistema
- RealtimeCandle: Vela con datos en tiempo real
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, TYPE_CHECKING
import uuid

if TYPE_CHECKING:
    from .orders.base import AdvancedOrderParams


class OrderType(Enum):
    """Tipos de orden soportados"""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Lado de la orden (compra/venta)"""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Estados posibles de una orden"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class PositionSide(Enum):
    """Tipo de posicion"""
    LONG = "long"
    SHORT = "short"


@dataclass
class Order:
    """
    Representa una orden de trading.

    Attributes:
        id: Identificador unico de la orden
        symbol: Par de trading (ej: BTC/USDT)
        side: Lado de la orden (BUY/SELL)
        type: Tipo de orden (MARKET, LIMIT, etc.)
        quantity: Cantidad a operar
        price: Precio limite (None para ordenes MARKET)
        stop_price: Precio de activacion para ordenes STOP
        created_at: Timestamp de creacion
        updated_at: Timestamp de ultima actualizacion
        status: Estado actual de la orden
        filled_quantity: Cantidad ejecutada
        filled_price: Precio promedio de ejecucion
        commission: Comision total pagada
        client_order_id: ID opcional del cliente
    """
    symbol: str
    side: OrderSide
    type: OrderType
    quantity: float
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    price: Optional[float] = None
    stop_price: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    filled_price: float = 0.0
    commission: float = 0.0
    client_order_id: Optional[str] = None
    advanced_params: Optional[AdvancedOrderParams] = None

    def is_active(self) -> bool:
        """Verifica si la orden esta activa"""
        return self.status in (
            OrderStatus.PENDING,
            OrderStatus.SUBMITTED,
            OrderStatus.PARTIALLY_FILLED
        )

    def is_filled(self) -> bool:
        """Verifica si la orden fue ejecutada completamente"""
        return self.status == OrderStatus.FILLED

    def remaining_quantity(self) -> float:
        """Cantidad pendiente de ejecutar"""
        return self.quantity - self.filled_quantity

    def to_dict(self) -> Dict[str, Any]:
        """Convierte la orden a diccionario"""
        result = {
            "id": self.id,
            "symbol": self.symbol,
            "side": self.side.value,
            "type": self.type.value,
            "quantity": self.quantity,
            "price": self.price,
            "stop_price": self.stop_price,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "status": self.status.value,
            "filled_quantity": self.filled_quantity,
            "filled_price": self.filled_price,
            "commission": self.commission,
        }
        if self.advanced_params:
            result["advanced_params"] = self.advanced_params.to_dict()
        return result


@dataclass
class OrderResult:
    """
    Resultado de la ejecucion de una orden.

    Attributes:
        order: Orden ejecutada
        success: Si la ejecucion fue exitosa
        message: Mensaje descriptivo
        executed_price: Precio real de ejecucion
        executed_quantity: Cantidad ejecutada
        slippage: Slippage aplicado
        commission: Comision cobrada
        timestamp: Momento de ejecucion
        latency_ms: Latencia simulada en ms
    """
    order: Order
    success: bool
    message: str
    executed_price: float
    executed_quantity: float
    slippage: float
    commission: float
    timestamp: datetime = field(default_factory=datetime.now)
    latency_ms: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convierte el resultado a diccionario"""
        return {
            "order_id": self.order.id,
            "success": self.success,
            "message": self.message,
            "executed_price": self.executed_price,
            "executed_quantity": self.executed_quantity,
            "slippage": self.slippage,
            "commission": self.commission,
            "timestamp": self.timestamp.isoformat(),
            "latency_ms": self.latency_ms,
        }


@dataclass
class PaperPosition:
    """
    Representa una posicion abierta en paper trading.

    Attributes:
        id: Identificador unico de la posicion
        symbol: Par de trading
        side: Tipo de posicion (LONG/SHORT)
        entry_price: Precio de entrada
        quantity: Cantidad de la posicion
        entry_time: Momento de apertura
        stop_loss: Precio de stop loss
        take_profit: Precio de take profit
        unrealized_pnl: PnL no realizado actual
        commission_paid: Comision pagada al abrir
    """
    symbol: str
    side: PositionSide
    entry_price: float
    quantity: float
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    entry_time: datetime = field(default_factory=datetime.now)
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    unrealized_pnl: float = 0.0
    commission_paid: float = 0.0

    def update_unrealized_pnl(self, current_price: float) -> float:
        """Actualiza y retorna el PnL no realizado"""
        if self.side == PositionSide.LONG:
            self.unrealized_pnl = (current_price - self.entry_price) * self.quantity
        else:
            self.unrealized_pnl = (self.entry_price - current_price) * self.quantity
        return self.unrealized_pnl

    def should_stop_loss(self, current_price: float) -> bool:
        """Verifica si se debe activar el stop loss"""
        if self.stop_loss is None:
            return False
        if self.side == PositionSide.LONG:
            return current_price <= self.stop_loss
        return current_price >= self.stop_loss

    def should_take_profit(self, current_price: float) -> bool:
        """Verifica si se debe activar el take profit"""
        if self.take_profit is None:
            return False
        if self.side == PositionSide.LONG:
            return current_price >= self.take_profit
        return current_price <= self.take_profit

    def to_dict(self) -> Dict[str, Any]:
        """Convierte la posicion a diccionario"""
        return {
            "id": self.id,
            "symbol": self.symbol,
            "side": self.side.value,
            "entry_price": self.entry_price,
            "quantity": self.quantity,
            "entry_time": self.entry_time.isoformat(),
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "unrealized_pnl": self.unrealized_pnl,
            "commission_paid": self.commission_paid,
        }


@dataclass
class PaperTradingState:
    """
    Estado actual del sistema de paper trading.

    Attributes:
        is_running: Si el sistema esta activo
        is_paused: Si el sistema esta pausado
        symbol: Par de trading actual
        strategy_name: Nombre de la estrategia
        start_time: Momento de inicio
        current_price: Precio actual del activo
        balance: Balance disponible
        equity: Equity total (balance + posiciones)
        open_positions: Numero de posiciones abiertas
        total_trades: Numero total de trades cerrados
        realized_pnl: PnL realizado total
        unrealized_pnl: PnL no realizado total
        win_rate: Tasa de aciertos actual
        last_update: Ultima actualizacion
    """
    is_running: bool = False
    is_paused: bool = False
    symbol: str = ""
    strategy_name: str = ""
    start_time: Optional[datetime] = None
    current_price: float = 0.0
    balance: float = 0.0
    equity: float = 0.0
    open_positions: int = 0
    total_trades: int = 0
    winning_trades: int = 0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    last_update: datetime = field(default_factory=datetime.now)

    @property
    def win_rate(self) -> float:
        """Calcula la tasa de aciertos"""
        if self.total_trades == 0:
            return 0.0
        return (self.winning_trades / self.total_trades) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convierte el estado a diccionario"""
        return {
            "is_running": self.is_running,
            "is_paused": self.is_paused,
            "symbol": self.symbol,
            "strategy_name": self.strategy_name,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "current_price": self.current_price,
            "balance": self.balance,
            "equity": self.equity,
            "open_positions": self.open_positions,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "win_rate": self.win_rate,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "last_update": self.last_update.isoformat(),
        }


@dataclass
class RealtimeCandle:
    """
    Vela con datos de tiempo real.

    Extiende la estructura OHLCV con metadatos adicionales
    para procesamiento en tiempo real.

    Attributes:
        timestamp: Momento de inicio de la vela
        open: Precio de apertura
        high: Precio maximo
        low: Precio minimo
        close: Precio de cierre actual
        volume: Volumen acumulado
        symbol: Par de trading
        timeframe: Temporalidad de la vela
        is_closed: Si la vela esta cerrada
        trades_count: Numero de trades en la vela
    """
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str
    timeframe: str = "1m"
    is_closed: bool = False
    trades_count: int = 0

    def update(self, price: float, volume: float = 0.0):
        """Actualiza la vela con un nuevo tick"""
        self.close = price
        self.high = max(self.high, price)
        self.low = min(self.low, price)
        self.volume += volume
        self.trades_count += 1

    def to_dict(self) -> Dict[str, Any]:
        """Convierte la vela a diccionario"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "is_closed": self.is_closed,
            "trades_count": self.trades_count,
        }


@dataclass
class TradeRecord:
    """
    Registro de un trade cerrado en paper trading.

    Attributes:
        id: Identificador unico
        symbol: Par de trading
        side: Tipo de posicion
        entry_price: Precio de entrada
        exit_price: Precio de salida
        quantity: Cantidad operada
        entry_time: Momento de entrada
        exit_time: Momento de salida
        pnl: Ganancia/perdida neta
        return_pct: Retorno porcentual
        commission: Comision total pagada
        exit_reason: Razon de cierre
    """
    symbol: str
    side: PositionSide
    entry_price: float
    exit_price: float
    quantity: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    return_pct: float
    commission: float
    exit_reason: str
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def to_dict(self) -> Dict[str, Any]:
        """Convierte el registro a diccionario"""
        return {
            "id": self.id,
            "symbol": self.symbol,
            "side": self.side.value,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "quantity": self.quantity,
            "entry_time": self.entry_time.isoformat(),
            "exit_time": self.exit_time.isoformat(),
            "pnl": self.pnl,
            "return_pct": self.return_pct,
            "commission": self.commission,
            "exit_reason": self.exit_reason,
        }
