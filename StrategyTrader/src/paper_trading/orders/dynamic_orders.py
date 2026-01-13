"""
Ordenes de Gestion Dinamica.

Implementa ordenes con comportamiento especial de ejecucion:
- FOK (Fill or Kill): Todo o nada, inmediato
- IOC (Immediate or Cancel): Parcial permitido, cancela resto
- Reduce-Only: Solo puede reducir posicion
- Post-Only: Solo maker, cancela si cruza spread
- Scale-In/Scale-Out: Entrada/salida gradual
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import uuid

from .enums import OrderType, OrderSide, OrderStatus, TimeInForce


@dataclass
class FOKOrder:
    """
    Orden Fill-or-Kill.

    Debe ejecutarse completamente de inmediato o se cancela.
    No permite ejecuciones parciales.

    Attributes:
        symbol: Par de trading
        side: BUY o SELL
        quantity: Cantidad requerida
        price: Precio limite (None para market)
        order_type: MARKET o LIMIT
    """
    symbol: str
    side: OrderSide
    quantity: float
    price: Optional[float] = None
    order_type: OrderType = OrderType.LIMIT
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)

    # Resultado
    executed: bool = False
    executed_price: float = 0.0
    rejection_reason: str = ""

    def can_fill(self, available_liquidity: float, market_price: float) -> bool:
        """
        Verifica si hay suficiente liquidez para fill completo.

        Args:
            available_liquidity: Liquidez disponible al precio
            market_price: Precio actual de mercado

        Returns:
            True si puede ejecutarse completamente
        """
        if available_liquidity < self.quantity:
            return False

        if self.price is not None:
            if self.side == OrderSide.BUY and market_price > self.price:
                return False
            if self.side == OrderSide.SELL and market_price < self.price:
                return False

        return True

    def execute(self, fill_price: float):
        """Marca la orden como ejecutada"""
        self.executed = True
        self.executed_price = fill_price
        self.status = OrderStatus.FILLED

    def reject(self, reason: str = "Insufficient liquidity"):
        """Marca la orden como rechazada/cancelada"""
        self.executed = False
        self.rejection_reason = reason
        self.status = OrderStatus.CANCELLED

    def to_ccxt_params(self, exchange_id: str) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "side": self.side.value,
            "type": "limit" if self.price else "market",
            "amount": self.quantity,
            "price": self.price,
            "params": {
                "timeInForce": "FOK",
            }
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": OrderType.FOK.value,
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": self.quantity,
            "price": self.price,
            "executed": self.executed,
            "executed_price": self.executed_price,
            "rejection_reason": self.rejection_reason,
            "status": self.status.value,
        }


@dataclass
class IOCOrder:
    """
    Orden Immediate-or-Cancel.

    Ejecuta lo que pueda inmediatamente y cancela el resto.
    Permite ejecuciones parciales.

    Attributes:
        symbol: Par de trading
        side: BUY o SELL
        quantity: Cantidad solicitada
        price: Precio limite (None para market)
    """
    symbol: str
    side: OrderSide
    quantity: float
    price: Optional[float] = None
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)

    # Resultado
    filled_quantity: float = 0.0
    cancelled_quantity: float = 0.0
    average_price: float = 0.0

    def execute_partial(self, fill_quantity: float, fill_price: float):
        """
        Registra ejecucion parcial.

        Args:
            fill_quantity: Cantidad ejecutada
            fill_price: Precio de ejecucion
        """
        if self.filled_quantity > 0:
            # Actualizar precio promedio
            total_value = (self.average_price * self.filled_quantity) + (fill_price * fill_quantity)
            self.filled_quantity += fill_quantity
            self.average_price = total_value / self.filled_quantity
        else:
            self.filled_quantity = fill_quantity
            self.average_price = fill_price

        self.cancelled_quantity = self.quantity - self.filled_quantity

        if self.filled_quantity >= self.quantity:
            self.status = OrderStatus.FILLED
        elif self.filled_quantity > 0:
            self.status = OrderStatus.PARTIALLY_FILLED
        else:
            self.status = OrderStatus.CANCELLED

    def finalize(self):
        """Finaliza la orden cancelando cantidad no ejecutada"""
        self.cancelled_quantity = self.quantity - self.filled_quantity
        if self.filled_quantity == 0:
            self.status = OrderStatus.CANCELLED
        elif self.filled_quantity < self.quantity:
            self.status = OrderStatus.PARTIALLY_FILLED

    def to_ccxt_params(self, exchange_id: str) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "side": self.side.value,
            "type": "limit" if self.price else "market",
            "amount": self.quantity,
            "price": self.price,
            "params": {
                "timeInForce": "IOC",
            }
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": OrderType.IOC.value,
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": self.quantity,
            "price": self.price,
            "filled_quantity": self.filled_quantity,
            "cancelled_quantity": self.cancelled_quantity,
            "average_price": self.average_price,
            "status": self.status.value,
        }


@dataclass
class ReduceOnlyOrder:
    """
    Orden Reduce-Only.

    Solo puede cerrar o reducir una posicion existente.
    No puede abrir nueva posicion ni aumentar exposicion.

    Attributes:
        symbol: Par de trading
        side: BUY (para cerrar short) o SELL (para cerrar long)
        quantity: Cantidad a reducir
        price: Precio limite (None para market)
        position_quantity: Cantidad actual de la posicion
    """
    symbol: str
    side: OrderSide
    quantity: float
    position_quantity: float  # Posicion actual a reducir
    price: Optional[float] = None
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)

    # Resultado
    filled_quantity: float = 0.0
    filled_price: float = 0.0
    rejected: bool = False
    rejection_reason: str = ""

    def validate(self) -> Optional[str]:
        """
        Valida que la orden puede reducir la posicion.

        Returns:
            Mensaje de error o None si es valida
        """
        if self.position_quantity <= 0:
            return "No hay posicion para reducir"

        if self.quantity > self.position_quantity:
            return f"Cantidad ({self.quantity}) excede posicion ({self.position_quantity})"

        return None

    def adjust_to_position(self) -> float:
        """
        Ajusta cantidad para no exceder posicion.

        Returns:
            Cantidad ajustada
        """
        return min(self.quantity, self.position_quantity)

    def execute(self, fill_price: float, fill_quantity: Optional[float] = None):
        """Ejecuta la orden"""
        self.filled_quantity = fill_quantity or self.quantity
        self.filled_price = fill_price
        self.status = OrderStatus.FILLED

    def reject(self, reason: str):
        """Rechaza la orden"""
        self.rejected = True
        self.rejection_reason = reason
        self.status = OrderStatus.REJECTED

    def to_ccxt_params(self, exchange_id: str) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "side": self.side.value,
            "type": "limit" if self.price else "market",
            "amount": self.quantity,
            "price": self.price,
            "params": {
                "reduceOnly": True,
            }
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": OrderType.REDUCE_ONLY.value,
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": self.quantity,
            "position_quantity": self.position_quantity,
            "price": self.price,
            "filled_quantity": self.filled_quantity,
            "filled_price": self.filled_price,
            "rejected": self.rejected,
            "rejection_reason": self.rejection_reason,
            "status": self.status.value,
        }


@dataclass
class PostOnlyOrder:
    """
    Orden Post-Only (Maker Only).

    Garantiza que la orden agregue liquidez al order book.
    Si cruzaria el spread (tomaria liquidez), se cancela.

    Attributes:
        symbol: Par de trading
        side: BUY o SELL
        quantity: Cantidad
        price: Precio de la orden
    """
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    time_in_force: TimeInForce = TimeInForce.GTX  # Good-till-crossing

    # Resultado
    filled_quantity: float = 0.0
    filled_price: float = 0.0
    was_cancelled: bool = False
    cancellation_reason: str = ""

    def would_cross_spread(self, bid: float, ask: float) -> bool:
        """
        Verifica si la orden cruzaria el spread.

        Args:
            bid: Mejor precio de compra actual
            ask: Mejor precio de venta actual

        Returns:
            True si cruzaria (seria taker)
        """
        if self.side == OrderSide.BUY:
            # Orden de compra cruza si precio >= ask
            return self.price >= ask
        else:
            # Orden de venta cruza si precio <= bid
            return self.price <= bid

    def submit(self, bid: float, ask: float) -> bool:
        """
        Intenta enviar la orden.

        Returns:
            True si se envio, False si se cancelo por cruzar
        """
        if self.would_cross_spread(bid, ask):
            self.was_cancelled = True
            self.cancellation_reason = "Would cross spread"
            self.status = OrderStatus.CANCELLED
            return False

        self.status = OrderStatus.SUBMITTED
        return True

    def execute(self, fill_price: float, fill_quantity: Optional[float] = None):
        """Ejecuta la orden como maker"""
        self.filled_quantity = fill_quantity or self.quantity
        self.filled_price = fill_price
        self.status = OrderStatus.FILLED

    def to_ccxt_params(self, exchange_id: str) -> Dict[str, Any]:
        exchange = exchange_id.lower()
        params = {}

        if exchange in ("binance", "bybit"):
            params["postOnly"] = True
        elif exchange == "kraken":
            params["oflags"] = "post"
        elif exchange == "coinbase":
            params["post_only"] = True

        return {
            "symbol": self.symbol,
            "side": self.side.value,
            "type": "limit",
            "amount": self.quantity,
            "price": self.price,
            "params": params,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": OrderType.POST_ONLY.value,
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": self.quantity,
            "price": self.price,
            "filled_quantity": self.filled_quantity,
            "filled_price": self.filled_price,
            "was_cancelled": self.was_cancelled,
            "cancellation_reason": self.cancellation_reason,
            "status": self.status.value,
        }


@dataclass
class ScaleOrder:
    """
    Orden Scale-In o Scale-Out.

    Ejecuta gradualmente en multiples niveles de precio.

    Attributes:
        symbol: Par de trading
        side: BUY (scale-in long) o SELL (scale-out)
        total_quantity: Cantidad total
        levels: Lista de precios para cada nivel
        quantities: Proporcion de cantidad en cada nivel (suma = 1.0)
        order_type: SCALE_IN o SCALE_OUT
    """
    symbol: str
    side: OrderSide
    total_quantity: float
    levels: List[float]
    quantities: List[float]  # Proporciones (0.25, 0.25, 0.50)
    order_type: OrderType = OrderType.SCALE_IN
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)

    # Estado por nivel
    level_statuses: List[str] = field(default_factory=list)
    level_filled: List[float] = field(default_factory=list)
    level_prices: List[float] = field(default_factory=list)

    # Totales
    filled_quantity: float = 0.0
    average_price: float = 0.0

    def __post_init__(self):
        if len(self.levels) != len(self.quantities):
            raise ValueError("levels y quantities deben tener igual longitud")

        if abs(sum(self.quantities) - 1.0) > 0.001:
            raise ValueError("quantities debe sumar 1.0")

        # Inicializar estados por nivel
        self.level_statuses = ["pending"] * len(self.levels)
        self.level_filled = [0.0] * len(self.levels)
        self.level_prices = [0.0] * len(self.levels)

    def get_quantity_for_level(self, level_index: int) -> float:
        """Obtiene cantidad para un nivel especifico"""
        return self.total_quantity * self.quantities[level_index]

    def get_pending_levels(self) -> List[Tuple[int, float, float]]:
        """
        Obtiene niveles pendientes de ejecucion.

        Returns:
            Lista de (indice, precio, cantidad)
        """
        result = []
        for i, status in enumerate(self.level_statuses):
            if status == "pending":
                qty = self.get_quantity_for_level(i)
                result.append((i, self.levels[i], qty))
        return result

    def check_level_trigger(
        self,
        level_index: int,
        current_price: float
    ) -> bool:
        """
        Verifica si un nivel debe ejecutarse.

        Args:
            level_index: Indice del nivel
            current_price: Precio actual

        Returns:
            True si el nivel debe ejecutarse
        """
        if self.level_statuses[level_index] != "pending":
            return False

        level_price = self.levels[level_index]

        if self.side == OrderSide.BUY:
            # Scale-in compra: trigger cuando precio <= nivel
            return current_price <= level_price
        else:
            # Scale-out venta: trigger cuando precio >= nivel
            return current_price >= level_price

    def execute_level(self, level_index: int, fill_price: float):
        """
        Ejecuta un nivel.

        Args:
            level_index: Indice del nivel
            fill_price: Precio de ejecucion
        """
        qty = self.get_quantity_for_level(level_index)

        self.level_statuses[level_index] = "filled"
        self.level_filled[level_index] = qty
        self.level_prices[level_index] = fill_price

        # Actualizar totales
        if self.filled_quantity > 0:
            total_value = (self.average_price * self.filled_quantity) + (fill_price * qty)
            self.filled_quantity += qty
            self.average_price = total_value / self.filled_quantity
        else:
            self.filled_quantity = qty
            self.average_price = fill_price

        # Verificar si todos los niveles estan completos
        if all(s == "filled" for s in self.level_statuses):
            self.status = OrderStatus.FILLED
        else:
            self.status = OrderStatus.PARTIALLY_FILLED

    def get_progress_pct(self) -> float:
        """Porcentaje de niveles completados"""
        filled = sum(1 for s in self.level_statuses if s == "filled")
        return (filled / len(self.levels)) * 100

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.order_type.value,
            "symbol": self.symbol,
            "side": self.side.value,
            "total_quantity": self.total_quantity,
            "levels": self.levels,
            "quantities": self.quantities,
            "level_statuses": self.level_statuses,
            "level_filled": self.level_filled,
            "level_prices": self.level_prices,
            "filled_quantity": self.filled_quantity,
            "average_price": self.average_price,
            "progress_pct": self.get_progress_pct(),
            "status": self.status.value,
        }
