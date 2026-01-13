"""
Ordenes Condicionales.

Implementa ordenes que se activan basadas en eventos:
- If-Touched: Se activa cuando precio toca un nivel
- OCO (One-Cancels-Other): Dos ordenes vinculadas
- OTOCO (One-Triggers-OCO): Trigger que activa un OCO
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import uuid

from .enums import (
    OrderType,
    OrderSide,
    OrderStatus,
    TimeInForce,
    TriggerType,
    TriggerDirection,
    CompositeOrderStatus,
)


# Soporte por exchange
OCO_SUPPORT = {
    "binance": True,
    "bybit": False,
    "okx": True,
    "kraken": False,
    "kucoin": True,
}


@dataclass
class IfTouchedOrder:
    """
    Orden If-Touched.

    Una orden que se activa cuando el precio toca un nivel especifico.
    Una vez activada, se convierte en orden de mercado o limite.

    Attributes:
        symbol: Par de trading
        side: BUY o SELL (de la orden resultante)
        quantity: Cantidad
        trigger_price: Precio que activa la orden
        trigger_direction: ABOVE o BELOW
        order_price: Precio limite despues de trigger (None = market)
        trigger_type: Tipo de precio para trigger (last, mark, index)

    Example:
        # Comprar cuando precio sube a 52000 (breakout)
        order = IfTouchedOrder(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=0.1,
            trigger_price=52000,
            trigger_direction=TriggerDirection.ABOVE
        )
    """
    symbol: str
    side: OrderSide
    quantity: float
    trigger_price: float
    trigger_direction: TriggerDirection
    order_price: Optional[float] = None  # None = market order on trigger
    trigger_type: TriggerType = TriggerType.LAST_PRICE
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    time_in_force: TimeInForce = TimeInForce.GTC

    # Estado
    triggered: bool = False
    triggered_at: Optional[datetime] = None
    triggered_price: float = 0.0
    filled_quantity: float = 0.0
    filled_price: float = 0.0

    def check_trigger(
        self,
        current_price: float,
        mark_price: Optional[float] = None,
        index_price: Optional[float] = None
    ) -> bool:
        """
        Verifica si la orden debe activarse.

        Args:
            current_price: Ultimo precio
            mark_price: Precio de marca (futuros)
            index_price: Precio del indice

        Returns:
            True si se activo el trigger
        """
        if self.triggered:
            return False

        # Seleccionar precio segun tipo de trigger
        check_price = current_price
        if self.trigger_type == TriggerType.MARK_PRICE and mark_price:
            check_price = mark_price
        elif self.trigger_type == TriggerType.INDEX_PRICE and index_price:
            check_price = index_price

        # Verificar condicion
        if self.trigger_direction == TriggerDirection.ABOVE:
            return check_price >= self.trigger_price
        else:
            return check_price <= self.trigger_price

    def activate(self, trigger_price: float, timestamp: Optional[datetime] = None):
        """Activa la orden"""
        self.triggered = True
        self.triggered_at = timestamp or datetime.now()
        self.triggered_price = trigger_price
        self.status = OrderStatus.TRIGGERED

    def execute(self, fill_price: float, fill_quantity: Optional[float] = None):
        """Ejecuta la orden activada"""
        self.filled_price = fill_price
        self.filled_quantity = fill_quantity or self.quantity
        self.status = OrderStatus.FILLED

    def to_ccxt_params(self, exchange_id: str) -> Dict[str, Any]:
        """Convierte a parametros CCXT (stop order)"""
        order_type = "stop" if self.order_price is None else "stop_limit"

        return {
            "symbol": self.symbol,
            "side": self.side.value,
            "type": order_type,
            "amount": self.quantity,
            "price": self.order_price,
            "params": {
                "stopPrice": self.trigger_price,
                "triggerType": self.trigger_type.value,
            }
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": OrderType.IF_TOUCHED.value,
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": self.quantity,
            "trigger_price": self.trigger_price,
            "trigger_direction": self.trigger_direction.value,
            "order_price": self.order_price,
            "trigger_type": self.trigger_type.value,
            "triggered": self.triggered,
            "triggered_at": self.triggered_at.isoformat() if self.triggered_at else None,
            "triggered_price": self.triggered_price,
            "filled_quantity": self.filled_quantity,
            "filled_price": self.filled_price,
            "status": self.status.value,
        }


@dataclass
class OCOOrder:
    """
    Orden OCO (One-Cancels-Other).

    Dos ordenes vinculadas donde la ejecucion de una
    cancela automaticamente la otra.

    Uso tipico: Stop Loss + Take Profit para una posicion.

    Attributes:
        symbol: Par de trading
        quantity: Cantidad (igual para ambas ordenes)
        side: Lado de las ordenes (mismo para ambas)
        stop_price: Precio del stop loss
        stop_limit_price: Precio limite del stop (opcional)
        limit_price: Precio del take profit (limit order)

    Example:
        # OCO para posicion long: SL en 48000, TP en 55000
        oco = OCOOrder(
            symbol="BTC/USDT",
            quantity=0.1,
            side=OrderSide.SELL,
            stop_price=48000,
            limit_price=55000
        )
    """
    symbol: str
    quantity: float
    side: OrderSide
    stop_price: float
    limit_price: float
    stop_limit_price: Optional[float] = None
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    stop_order_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    limit_order_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    status: CompositeOrderStatus = CompositeOrderStatus.ACTIVE
    created_at: datetime = field(default_factory=datetime.now)

    # Estado
    executed_order: Optional[str] = None  # "stop" o "limit"
    cancelled_order: Optional[str] = None
    fill_price: float = 0.0
    fill_quantity: float = 0.0

    def check_triggers(self, current_price: float) -> Optional[str]:
        """
        Verifica si alguna orden debe ejecutarse.

        Args:
            current_price: Precio actual

        Returns:
            "stop" si stop debe ejecutarse
            "limit" si limit debe ejecutarse
            None si ninguna
        """
        if self.status != CompositeOrderStatus.ACTIVE:
            return None

        if self.side == OrderSide.SELL:
            # Para cerrar long: stop si baja, limit si sube
            if current_price <= self.stop_price:
                return "stop"
            if current_price >= self.limit_price:
                return "limit"
        else:
            # Para cerrar short: stop si sube, limit si baja
            if current_price >= self.stop_price:
                return "stop"
            if current_price <= self.limit_price:
                return "limit"

        return None

    def execute_order(
        self,
        order_type: str,
        fill_price: float,
        fill_quantity: Optional[float] = None
    ):
        """
        Ejecuta una de las ordenes y cancela la otra.

        Args:
            order_type: "stop" o "limit"
            fill_price: Precio de ejecucion
            fill_quantity: Cantidad ejecutada
        """
        self.executed_order = order_type
        self.cancelled_order = "limit" if order_type == "stop" else "stop"
        self.fill_price = fill_price
        self.fill_quantity = fill_quantity or self.quantity
        self.status = CompositeOrderStatus.COMPLETED

    def get_executed_order_id(self) -> Optional[str]:
        """Retorna ID de la orden ejecutada"""
        if self.executed_order == "stop":
            return self.stop_order_id
        elif self.executed_order == "limit":
            return self.limit_order_id
        return None

    def get_cancelled_order_id(self) -> Optional[str]:
        """Retorna ID de la orden cancelada"""
        if self.cancelled_order == "stop":
            return self.stop_order_id
        elif self.cancelled_order == "limit":
            return self.limit_order_id
        return None

    def can_execute_on_exchange(self, exchange_id: str) -> bool:
        return OCO_SUPPORT.get(exchange_id.lower(), False)

    def to_ccxt_params(self, exchange_id: str) -> Dict[str, Any]:
        """Convierte a parametros CCXT para exchanges con OCO nativo"""
        exchange = exchange_id.lower()

        if exchange == "binance":
            return {
                "symbol": self.symbol,
                "side": self.side.value,
                "quantity": self.quantity,
                "price": self.limit_price,
                "stopPrice": self.stop_price,
                "stopLimitPrice": self.stop_limit_price or self.stop_price,
                "stopLimitTimeInForce": "GTC",
            }
        elif exchange == "okx":
            return {
                "instId": self.symbol.replace("/", "-"),
                "side": self.side.value.lower(),
                "ordType": "oco",
                "sz": str(self.quantity),
                "tpTriggerPx": str(self.limit_price),
                "tpOrdPx": str(self.limit_price),
                "slTriggerPx": str(self.stop_price),
                "slOrdPx": str(self.stop_limit_price or self.stop_price),
            }

        return {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": OrderType.OCO.value,
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": self.quantity,
            "stop_price": self.stop_price,
            "stop_limit_price": self.stop_limit_price,
            "limit_price": self.limit_price,
            "stop_order_id": self.stop_order_id,
            "limit_order_id": self.limit_order_id,
            "executed_order": self.executed_order,
            "cancelled_order": self.cancelled_order,
            "fill_price": self.fill_price,
            "fill_quantity": self.fill_quantity,
            "status": self.status.value,
        }


@dataclass
class OTOCOOrder:
    """
    Orden OTOCO (One-Triggers-OCO).

    Una orden de entrada que al ejecutarse activa automaticamente
    un par de ordenes OCO (stop loss + take profit).

    Attributes:
        symbol: Par de trading
        entry_side: BUY o SELL para la entrada
        entry_quantity: Cantidad de entrada
        entry_price: Precio de entrada (None = market)
        entry_type: MARKET o LIMIT
        stop_loss_price: Precio de stop loss
        take_profit_price: Precio de take profit
        stop_loss_limit_price: Precio limite del SL (opcional)
        take_profit_limit_price: Precio limite del TP (opcional)

    Example:
        # Entrar long en 50000, SL en 49000, TP en 52000
        otoco = OTOCOOrder(
            symbol="BTC/USDT",
            entry_side=OrderSide.BUY,
            entry_quantity=0.1,
            entry_price=50000,
            stop_loss_price=49000,
            take_profit_price=52000
        )
    """
    symbol: str
    entry_side: OrderSide
    entry_quantity: float
    stop_loss_price: float
    take_profit_price: float
    entry_price: Optional[float] = None  # None = market
    entry_type: OrderType = OrderType.LIMIT
    stop_loss_limit_price: Optional[float] = None
    take_profit_limit_price: Optional[float] = None
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    entry_order_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    status: CompositeOrderStatus = CompositeOrderStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)

    # Estado de entrada
    entry_filled: bool = False
    entry_fill_price: float = 0.0
    entry_fill_quantity: float = 0.0

    # OCO interno (creado despues de entry fill)
    oco: Optional[OCOOrder] = None

    # Estado final
    exit_type: Optional[str] = None  # "stop_loss" o "take_profit"
    exit_price: float = 0.0

    def __post_init__(self):
        """Valida precios"""
        if self.entry_side == OrderSide.BUY:
            # Long: SL < entry < TP
            if self.entry_price:
                if self.stop_loss_price >= self.entry_price:
                    raise ValueError("Stop loss debe ser menor que entrada para long")
                if self.take_profit_price <= self.entry_price:
                    raise ValueError("Take profit debe ser mayor que entrada para long")
        else:
            # Short: TP < entry < SL
            if self.entry_price:
                if self.stop_loss_price <= self.entry_price:
                    raise ValueError("Stop loss debe ser mayor que entrada para short")
                if self.take_profit_price >= self.entry_price:
                    raise ValueError("Take profit debe ser menor que entrada para short")

    def check_entry_trigger(
        self,
        current_price: float,
        bid: Optional[float] = None,
        ask: Optional[float] = None
    ) -> bool:
        """
        Verifica si la entrada debe ejecutarse.

        Returns:
            True si entry debe ejecutarse
        """
        if self.entry_filled:
            return False

        if self.entry_type == OrderType.MARKET:
            return True

        if self.entry_side == OrderSide.BUY:
            check_price = ask or current_price
            return check_price <= self.entry_price
        else:
            check_price = bid or current_price
            return check_price >= self.entry_price

    def execute_entry(
        self,
        fill_price: float,
        fill_quantity: Optional[float] = None
    ):
        """Ejecuta la entrada y activa el OCO"""
        self.entry_filled = True
        self.entry_fill_price = fill_price
        self.entry_fill_quantity = fill_quantity or self.entry_quantity
        self.status = CompositeOrderStatus.ACTIVE

        # Crear OCO para la salida
        exit_side = OrderSide.SELL if self.entry_side == OrderSide.BUY else OrderSide.BUY

        self.oco = OCOOrder(
            symbol=self.symbol,
            quantity=self.entry_fill_quantity,
            side=exit_side,
            stop_price=self.stop_loss_price,
            limit_price=self.take_profit_price,
            stop_limit_price=self.stop_loss_limit_price,
        )

    def check_exit_triggers(self, current_price: float) -> Optional[str]:
        """
        Verifica si SL o TP debe ejecutarse.

        Returns:
            "stop_loss" o "take_profit" o None
        """
        if not self.entry_filled or not self.oco:
            return None

        trigger = self.oco.check_triggers(current_price)
        if trigger == "stop":
            return "stop_loss"
        elif trigger == "limit":
            return "take_profit"
        return None

    def execute_exit(self, exit_type: str, fill_price: float):
        """Ejecuta la salida (SL o TP)"""
        self.exit_type = exit_type
        self.exit_price = fill_price
        self.status = CompositeOrderStatus.COMPLETED

        if self.oco:
            order_type = "stop" if exit_type == "stop_loss" else "limit"
            self.oco.execute_order(order_type, fill_price, self.entry_fill_quantity)

    def get_pnl(self) -> float:
        """Calcula PnL de la operacion"""
        if not self.entry_filled or self.exit_price == 0:
            return 0.0

        if self.entry_side == OrderSide.BUY:
            return (self.exit_price - self.entry_fill_price) * self.entry_fill_quantity
        else:
            return (self.entry_fill_price - self.exit_price) * self.entry_fill_quantity

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": OrderType.OTOCO.value,
            "symbol": self.symbol,
            "entry_side": self.entry_side.value,
            "entry_quantity": self.entry_quantity,
            "entry_price": self.entry_price,
            "entry_type": self.entry_type.value,
            "stop_loss_price": self.stop_loss_price,
            "take_profit_price": self.take_profit_price,
            "entry_order_id": self.entry_order_id,
            "entry_filled": self.entry_filled,
            "entry_fill_price": self.entry_fill_price,
            "entry_fill_quantity": self.entry_fill_quantity,
            "oco": self.oco.to_dict() if self.oco else None,
            "exit_type": self.exit_type,
            "exit_price": self.exit_price,
            "pnl": self.get_pnl(),
            "status": self.status.value,
        }
