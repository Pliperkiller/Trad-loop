"""
Clases base y parametros para ordenes avanzadas.

Define estructuras de datos fundamentales para todos los tipos de ordenes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..models import Order

from .enums import (
    OrderType,
    OrderSide,
    OrderStatus,
    TimeInForce,
    TriggerType,
    TriggerDirection,
    CompositeOrderStatus,
)


@dataclass
class AdvancedOrderParams:
    """
    Parametros extendidos para ordenes avanzadas.

    Agrupa todos los parametros opcionales que pueden aplicar
    a diferentes tipos de ordenes avanzadas.

    Attributes:
        trail_amount: Distancia fija de trailing (en precio)
        trail_percent: Distancia porcentual de trailing (0.02 = 2%)
        activation_price: Precio para activar trailing stop
        callback_rate: Tasa de callback para trailing (Binance)

        duration_seconds: Duracion total para ordenes algoritmicas
        slice_count: Numero de slices para TWAP/VWAP
        slice_interval_seconds: Intervalo entre slices
        size_variation: Variacion aleatoria de tamano (0.1 = 10%)
        max_participation: Participacion maxima en volumen (VWAP)

        display_quantity: Cantidad visible para iceberg
        hidden: Si la orden es completamente oculta

        scale_levels: Niveles de precio para scaling
        scale_quantities: Cantidades en cada nivel (proporciones)

        time_in_force: Vigencia de la orden
        expire_time: Tiempo de expiracion para GTD

        trigger_price: Precio de activacion para condicionales
        trigger_type: Tipo de precio para trigger
        trigger_direction: Direccion del trigger

        reduce_only: Solo permite reducir posicion
        post_only: Solo permite ser maker

        stop_loss_price: Precio de stop loss (para bracket)
        take_profit_price: Precio de take profit (para bracket)
        stop_loss_limit_price: Precio limite del stop (stop-limit)
        take_profit_limit_price: Precio limite del TP (TP-limit)
    """
    # Trailing stop params
    trail_amount: Optional[float] = None
    trail_percent: Optional[float] = None
    activation_price: Optional[float] = None
    callback_rate: Optional[float] = None

    # Algorithmic execution params
    duration_seconds: Optional[int] = None
    slice_count: Optional[int] = None
    slice_interval_seconds: Optional[int] = None
    size_variation: Optional[float] = None
    max_participation: float = 0.05  # 5% max del volumen

    # Iceberg params
    display_quantity: Optional[float] = None
    hidden: bool = False

    # Scaling params
    scale_levels: Optional[List[float]] = None
    scale_quantities: Optional[List[float]] = None

    # Time constraints
    time_in_force: TimeInForce = TimeInForce.GTC
    expire_time: Optional[datetime] = None

    # Conditional params
    trigger_price: Optional[float] = None
    trigger_type: TriggerType = TriggerType.LAST_PRICE
    trigger_direction: Optional[TriggerDirection] = None

    # Execution flags
    reduce_only: bool = False
    post_only: bool = False

    # Bracket/OCO params
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    stop_loss_limit_price: Optional[float] = None
    take_profit_limit_price: Optional[float] = None

    def validate(self, order_type: OrderType) -> Optional[str]:
        """
        Valida parametros segun el tipo de orden.

        Args:
            order_type: Tipo de orden a validar

        Returns:
            Mensaje de error o None si es valido
        """
        if order_type == OrderType.TRAILING_STOP:
            if self.trail_amount is None and self.trail_percent is None:
                return "Trailing stop requiere trail_amount o trail_percent"
            if self.trail_amount and self.trail_amount <= 0:
                return "trail_amount debe ser positivo"
            if self.trail_percent and (self.trail_percent <= 0 or self.trail_percent >= 1):
                return "trail_percent debe estar entre 0 y 1"

        elif order_type == OrderType.TWAP:
            if not self.duration_seconds or self.duration_seconds <= 0:
                return "TWAP requiere duration_seconds positivo"
            if self.slice_count and self.slice_count < 2:
                return "TWAP requiere al menos 2 slices"

        elif order_type == OrderType.VWAP:
            if not self.duration_seconds or self.duration_seconds <= 0:
                return "VWAP requiere duration_seconds positivo"
            if self.max_participation <= 0 or self.max_participation > 1:
                return "max_participation debe estar entre 0 y 1"

        elif order_type == OrderType.ICEBERG:
            if not self.display_quantity or self.display_quantity <= 0:
                return "Iceberg requiere display_quantity positivo"

        elif order_type in (OrderType.SCALE_IN, OrderType.SCALE_OUT):
            if not self.scale_levels or not self.scale_quantities:
                return "Scaling requiere scale_levels y scale_quantities"
            if len(self.scale_levels) != len(self.scale_quantities):
                return "scale_levels y scale_quantities deben tener igual longitud"
            if abs(sum(self.scale_quantities) - 1.0) > 0.001:
                return "scale_quantities debe sumar 1.0"

        elif order_type == OrderType.BRACKET:
            if not self.stop_loss_price:
                return "Bracket requiere stop_loss_price"
            if not self.take_profit_price:
                return "Bracket requiere take_profit_price"

        elif order_type == OrderType.IF_TOUCHED:
            if not self.trigger_price:
                return "If-touched requiere trigger_price"

        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario para serializacion"""
        return {
            "trail_amount": self.trail_amount,
            "trail_percent": self.trail_percent,
            "activation_price": self.activation_price,
            "duration_seconds": self.duration_seconds,
            "slice_count": self.slice_count,
            "display_quantity": self.display_quantity,
            "scale_levels": self.scale_levels,
            "scale_quantities": self.scale_quantities,
            "time_in_force": self.time_in_force.value,
            "expire_time": self.expire_time.isoformat() if self.expire_time else None,
            "trigger_price": self.trigger_price,
            "reduce_only": self.reduce_only,
            "post_only": self.post_only,
            "stop_loss_price": self.stop_loss_price,
            "take_profit_price": self.take_profit_price,
        }


@dataclass
class OrderAction:
    """
    Accion a tomar en una orden.

    Resultado de procesar una actualizacion de precio u otro evento.
    """
    action_type: str  # "execute", "modify", "cancel", "none"
    order_id: str
    new_price: Optional[float] = None
    new_stop_price: Optional[float] = None
    execution_price: Optional[float] = None
    message: str = ""


class AdvancedOrderInterface(ABC):
    """
    Interfaz base para todas las ordenes avanzadas.

    Define el contrato que deben implementar todos los tipos
    de ordenes avanzadas para integrarse con el simulador.
    """

    @abstractmethod
    def get_id(self) -> str:
        """Retorna el ID unico de la orden"""
        pass

    @abstractmethod
    def get_type(self) -> OrderType:
        """Retorna el tipo de orden"""
        pass

    @abstractmethod
    def get_child_orders(self) -> List['Order']:
        """
        Retorna ordenes hijas para ordenes compuestas.

        Para ordenes simples retorna lista vacia.
        Para Bracket retorna [entry, sl, tp].
        Para OCO retorna [order_a, order_b].
        """
        pass

    @abstractmethod
    def on_price_update(
        self,
        current_price: float,
        timestamp: datetime,
        bid: Optional[float] = None,
        ask: Optional[float] = None
    ) -> Optional[OrderAction]:
        """
        Procesa actualizacion de precio.

        Llamado en cada tick para ordenes que necesitan
        monitorear precio (trailing, condicionales).

        Args:
            current_price: Precio actual del mercado
            timestamp: Momento de la actualizacion
            bid: Precio bid (opcional)
            ask: Precio ask (opcional)

        Returns:
            Accion a tomar o None si no hay accion
        """
        pass

    @abstractmethod
    def on_child_filled(
        self,
        child_order_id: str,
        fill_price: float,
        fill_quantity: float
    ) -> Optional[OrderAction]:
        """
        Procesa ejecucion de orden hija.

        Para ordenes compuestas, determina que hacer cuando
        una de las ordenes hijas se ejecuta.

        Args:
            child_order_id: ID de la orden hija ejecutada
            fill_price: Precio de ejecucion
            fill_quantity: Cantidad ejecutada

        Returns:
            Accion a tomar (ej: cancelar otra orden en OCO)
        """
        pass

    @abstractmethod
    def is_complete(self) -> bool:
        """
        Verifica si la orden esta completamente terminada.

        Incluye estados: FILLED, CANCELLED, REJECTED, EXPIRED.
        """
        pass

    @abstractmethod
    def can_execute_on_exchange(self, exchange_id: str) -> bool:
        """
        Verifica si el exchange soporta este tipo de orden nativamente.

        Args:
            exchange_id: Identificador del exchange (ej: "binance")

        Returns:
            True si el exchange soporta la orden nativamente
        """
        pass

    @abstractmethod
    def to_ccxt_params(self, exchange_id: str) -> Dict[str, Any]:
        """
        Convierte la orden a parametros CCXT.

        Args:
            exchange_id: Identificador del exchange

        Returns:
            Diccionario de parametros para ccxt.create_order()
        """
        pass


@dataclass
class TrailingStopState:
    """
    Estado interno de un trailing stop.

    Mantiene el seguimiento del precio para calcular
    cuando debe ejecutarse el stop.
    """
    order_id: str
    side: OrderSide
    trail_amount: Optional[float]
    trail_percent: Optional[float]
    high_water_mark: float  # Precio mas alto visto (para long)
    low_water_mark: float   # Precio mas bajo visto (para short)
    current_stop_price: float
    is_activated: bool = False
    activation_price: Optional[float] = None

    def update_for_long(self, current_price: float) -> Optional[float]:
        """
        Actualiza trailing para posicion long.

        Returns:
            Nuevo stop price si cambio, None si no
        """
        if self.activation_price and current_price < self.activation_price:
            return None

        self.is_activated = True

        if current_price > self.high_water_mark:
            self.high_water_mark = current_price
            # Calcular nuevo stop
            if self.trail_percent is not None:
                self.current_stop_price = current_price * (1 - self.trail_percent)
            elif self.trail_amount is not None:
                self.current_stop_price = current_price - self.trail_amount
            return self.current_stop_price
        return None

    def update_for_short(self, current_price: float) -> Optional[float]:
        """
        Actualiza trailing para posicion short.

        Returns:
            Nuevo stop price si cambio, None si no
        """
        if self.activation_price and current_price > self.activation_price:
            return None

        self.is_activated = True

        if current_price < self.low_water_mark:
            self.low_water_mark = current_price
            # Calcular nuevo stop
            if self.trail_percent is not None:
                self.current_stop_price = current_price * (1 + self.trail_percent)
            elif self.trail_amount is not None:
                self.current_stop_price = current_price + self.trail_amount
            return self.current_stop_price
        return None

    def should_trigger(self, current_price: float, side: OrderSide) -> bool:
        """Verifica si el trailing stop debe ejecutarse"""
        if not self.is_activated:
            return False

        if side == OrderSide.SELL:  # Long position, stop es venta
            return current_price <= self.current_stop_price
        else:  # Short position, stop es compra
            return current_price >= self.current_stop_price


@dataclass
class AlgoOrderState:
    """
    Estado interno de una orden algoritmica (TWAP/VWAP/Iceberg).

    Mantiene seguimiento del progreso de ejecucion.
    """
    order_id: str
    total_quantity: float
    filled_quantity: float = 0.0
    remaining_quantity: float = 0.0
    slices_executed: int = 0
    slices_total: int = 0
    next_execution_time: Optional[datetime] = None
    average_price: float = 0.0
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None

    def __post_init__(self):
        self.remaining_quantity = self.total_quantity

    def record_fill(self, quantity: float, price: float):
        """Registra una ejecucion parcial"""
        # Actualizar precio promedio ponderado
        if self.filled_quantity > 0:
            total_value = (self.average_price * self.filled_quantity) + (price * quantity)
            self.filled_quantity += quantity
            self.average_price = total_value / self.filled_quantity
        else:
            self.filled_quantity = quantity
            self.average_price = price

        self.remaining_quantity = self.total_quantity - self.filled_quantity
        self.slices_executed += 1

    def is_complete(self) -> bool:
        """Verifica si la orden algoritmica esta completa"""
        return self.remaining_quantity <= 0 or (
            self.slices_total > 0 and self.slices_executed >= self.slices_total
        )

    def get_progress_pct(self) -> float:
        """Retorna porcentaje de progreso"""
        if self.total_quantity <= 0:
            return 100.0
        return (self.filled_quantity / self.total_quantity) * 100


@dataclass
class CompositeOrderState:
    """
    Estado interno de una orden compuesta (Bracket, OCO, OTOCO).

    Mantiene seguimiento de todas las ordenes hijas.
    """
    parent_id: str
    child_order_ids: List[str]
    status: CompositeOrderStatus = CompositeOrderStatus.PENDING
    trigger_filled: bool = False
    active_child_ids: List[str] = field(default_factory=list)
    filled_child_ids: List[str] = field(default_factory=list)
    cancelled_child_ids: List[str] = field(default_factory=list)

    def activate(self):
        """Activa la orden compuesta"""
        self.status = CompositeOrderStatus.ACTIVE
        self.active_child_ids = self.child_order_ids.copy()

    def on_child_filled(self, child_id: str) -> List[str]:
        """
        Procesa ejecucion de orden hija.

        Returns:
            Lista de IDs de ordenes a cancelar
        """
        if child_id in self.active_child_ids:
            self.active_child_ids.remove(child_id)
        self.filled_child_ids.append(child_id)

        # Retornar ordenes restantes a cancelar (para OCO)
        return self.active_child_ids.copy()

    def on_child_cancelled(self, child_id: str):
        """Procesa cancelacion de orden hija"""
        if child_id in self.active_child_ids:
            self.active_child_ids.remove(child_id)
        self.cancelled_child_ids.append(child_id)

    def is_complete(self) -> bool:
        """Verifica si la orden compuesta esta completa"""
        return len(self.active_child_ids) == 0
