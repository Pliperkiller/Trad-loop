"""
Ordenes de control de riesgo.

Implementa ordenes avanzadas para gestion profesional de riesgo:
- Trailing Stop: Stop loss dinamico que sigue al precio
- Bracket Order: Entrada atomica con stop loss y take profit
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
import uuid

from .enums import (
    OrderType,
    OrderSide,
    OrderStatus,
    TimeInForce,
    CompositeOrderStatus,
)
from .base import (
    AdvancedOrderParams,
    AdvancedOrderInterface,
    OrderAction,
    TrailingStopState,
    CompositeOrderState,
)


# Mapeo de soporte nativo por exchange
TRAILING_STOP_SUPPORT = {
    "binance": True,
    "bybit": True,
    "okx": True,
    "kraken": True,
    "bitget": True,
    "kucoin": True,
}

BRACKET_ORDER_SUPPORT = {
    "bybit": True,
    "okx": True,
    "binance": False,  # No soporta bracket atomico
    "kraken": False,
}


@dataclass
class TrailingStopOrder(AdvancedOrderInterface):
    """
    Orden de trailing stop.

    Un stop loss dinamico que sigue al precio cuando este se mueve
    a favor de la posicion. Cuando el precio revierte, el stop
    se mantiene fijo y se ejecuta si el precio lo cruza.

    Attributes:
        symbol: Par de trading (ej: BTC/USDT)
        side: Lado de la orden (SELL para long, BUY para short)
        quantity: Cantidad a vender/comprar
        trail_amount: Distancia fija del trailing en precio
        trail_percent: Distancia porcentual del trailing (0.02 = 2%)
        activation_price: Precio para activar el trailing (opcional)
        initial_price: Precio inicial del mercado al crear la orden

    Example:
        # Trailing stop del 2% para posicion long
        order = TrailingStopOrder(
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            quantity=0.1,
            trail_percent=0.02,
            initial_price=50000
        )
    """
    symbol: str
    side: OrderSide
    quantity: float
    initial_price: float
    trail_amount: Optional[float] = None
    trail_percent: Optional[float] = None
    activation_price: Optional[float] = None
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    time_in_force: TimeInForce = TimeInForce.GTC

    # Estado interno del trailing
    _state: Optional[TrailingStopState] = field(default=None, repr=False)

    def __post_init__(self):
        """Inicializa el estado del trailing"""
        if self.trail_amount is None and self.trail_percent is None:
            raise ValueError("Debe especificar trail_amount o trail_percent")

        # Inicializar estado
        if self.side == OrderSide.SELL:  # Long position
            initial_stop = self._calculate_initial_stop_long()
        else:  # Short position
            initial_stop = self._calculate_initial_stop_short()

        self._state = TrailingStopState(
            order_id=self.id,
            side=self.side,
            trail_amount=self.trail_amount,
            trail_percent=self.trail_percent,
            high_water_mark=self.initial_price,
            low_water_mark=self.initial_price,
            current_stop_price=initial_stop,
            is_activated=self.activation_price is None,
            activation_price=self.activation_price,
        )

    def _calculate_initial_stop_long(self) -> float:
        """Calcula stop inicial para posicion long"""
        if self.trail_percent is not None:
            return self.initial_price * (1 - self.trail_percent)
        if self.trail_amount is not None:
            return self.initial_price - self.trail_amount
        raise ValueError("trail_percent o trail_amount debe estar definido")

    def _calculate_initial_stop_short(self) -> float:
        """Calcula stop inicial para posicion short"""
        if self.trail_percent is not None:
            return self.initial_price * (1 + self.trail_percent)
        if self.trail_amount is not None:
            return self.initial_price + self.trail_amount
        raise ValueError("trail_percent o trail_amount debe estar definido")

    def get_id(self) -> str:
        return self.id

    def get_type(self) -> OrderType:
        return OrderType.TRAILING_STOP

    def get_child_orders(self) -> List:
        """Trailing stop no tiene ordenes hijas"""
        return []

    def get_current_stop_price(self) -> float:
        """Retorna el precio actual del stop"""
        return self._state.current_stop_price

    def is_activated(self) -> bool:
        """Verifica si el trailing esta activado"""
        return self._state.is_activated

    def on_price_update(
        self,
        current_price: float,
        timestamp: datetime,
        bid: Optional[float] = None,
        ask: Optional[float] = None
    ) -> Optional[OrderAction]:
        """
        Procesa actualizacion de precio.

        Actualiza el stop si el precio se mueve a favor,
        o ejecuta si el precio cruza el stop.

        Returns:
            OrderAction si debe ejecutarse, None si no
        """
        if self.status != OrderStatus.PENDING and self.status != OrderStatus.SUBMITTED:
            return None

        if self.side == OrderSide.SELL:  # Long position
            # Actualizar trailing
            new_stop = self._state.update_for_long(current_price)

            # Verificar trigger
            if self._state.should_trigger(current_price, self.side):
                self.status = OrderStatus.TRIGGERED
                return OrderAction(
                    action_type="execute",
                    order_id=self.id,
                    execution_price=current_price,
                    message=f"Trailing stop triggered at {current_price}"
                )

            if new_stop:
                return OrderAction(
                    action_type="modify",
                    order_id=self.id,
                    new_stop_price=new_stop,
                    message=f"Trailing stop updated to {new_stop}"
                )
        else:  # Short position
            # Actualizar trailing
            new_stop = self._state.update_for_short(current_price)

            # Verificar trigger
            if self._state.should_trigger(current_price, self.side):
                self.status = OrderStatus.TRIGGERED
                return OrderAction(
                    action_type="execute",
                    order_id=self.id,
                    execution_price=current_price,
                    message=f"Trailing stop triggered at {current_price}"
                )

            if new_stop:
                return OrderAction(
                    action_type="modify",
                    order_id=self.id,
                    new_stop_price=new_stop,
                    message=f"Trailing stop updated to {new_stop}"
                )

        return None

    def on_child_filled(
        self,
        child_order_id: str,
        fill_price: float,
        fill_quantity: float
    ) -> Optional[OrderAction]:
        """Trailing stop no tiene ordenes hijas"""
        return None

    def is_complete(self) -> bool:
        return self.status in (
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED
        )

    def can_execute_on_exchange(self, exchange_id: str) -> bool:
        return TRAILING_STOP_SUPPORT.get(exchange_id.lower(), False)

    def to_ccxt_params(self, exchange_id: str) -> Dict[str, Any]:
        """
        Convierte a parametros CCXT.

        Los parametros varian segun el exchange:
        - Binance: trailingDelta (en basis points)
        - Bybit: trailing_stop (en precio)
        """
        base_params = {
            "symbol": self.symbol,
            "side": self.side.value,
            "amount": self.quantity,
            "type": "TRAILING_STOP_MARKET",
        }

        params = {}
        exchange = exchange_id.lower()

        if exchange == "binance":
            # Binance usa trailingDelta en basis points (1 bp = 0.01%)
            if self.trail_percent:
                params["trailingDelta"] = int(self.trail_percent * 10000)
            else:
                # Convertir trail_amount a porcentaje aproximado
                params["trailingDelta"] = int(
                    (self.trail_amount / self.initial_price) * 10000
                )
            if self.activation_price:
                params["activationPrice"] = self.activation_price

        elif exchange == "bybit":
            if self.trail_amount:
                params["trailing_stop"] = str(self.trail_amount)
            else:
                params["trailing_stop"] = str(
                    self.initial_price * self.trail_percent
                )

        elif exchange == "okx":
            params["callbackRatio"] = str(
                self.trail_percent * 100 if self.trail_percent else
                (self.trail_amount / self.initial_price) * 100
            )

        base_params["params"] = params
        return base_params

    def to_dict(self) -> Dict[str, Any]:
        """Serializa la orden a diccionario"""
        return {
            "id": self.id,
            "type": OrderType.TRAILING_STOP.value,
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": self.quantity,
            "trail_amount": self.trail_amount,
            "trail_percent": self.trail_percent,
            "activation_price": self.activation_price,
            "current_stop_price": self._state.current_stop_price if self._state else None,
            "is_activated": self._state.is_activated if self._state else False,
            "high_water_mark": self._state.high_water_mark if self._state else None,
            "low_water_mark": self._state.low_water_mark if self._state else None,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass
class BracketOrder(AdvancedOrderInterface):
    """
    Orden bracket (entrada + stop loss + take profit).

    Crea tres ordenes atomicamente:
    1. Orden de entrada (limit o market)
    2. Stop loss (activado despues de entry fill)
    3. Take profit (activado despues de entry fill)

    Cuando SL o TP se ejecuta, la otra orden se cancela automaticamente.

    Attributes:
        symbol: Par de trading
        side: Lado de la entrada (BUY para long, SELL para short)
        quantity: Cantidad de la posicion
        entry_price: Precio de entrada (None para market)
        entry_type: Tipo de orden de entrada (MARKET o LIMIT)
        stop_loss_price: Precio del stop loss
        take_profit_price: Precio del take profit
        stop_loss_limit_price: Precio limite del SL (para stop-limit)
        take_profit_limit_price: Precio limite del TP (para TP-limit)

    Example:
        # Bracket order para long: entrada en 50000, SL en 49000, TP en 52000
        bracket = BracketOrder(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=0.1,
            entry_price=50000,
            stop_loss_price=49000,
            take_profit_price=52000
        )
    """
    symbol: str
    side: OrderSide
    quantity: float
    stop_loss_price: float
    take_profit_price: float
    entry_price: Optional[float] = None
    entry_type: OrderType = OrderType.LIMIT
    stop_loss_limit_price: Optional[float] = None
    take_profit_limit_price: Optional[float] = None
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    status: CompositeOrderStatus = CompositeOrderStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    time_in_force: TimeInForce = TimeInForce.GTC

    # IDs de ordenes hijas
    entry_order_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    stop_loss_order_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    take_profit_order_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    # Estado
    entry_filled: bool = False
    entry_fill_price: float = 0.0
    entry_fill_quantity: float = 0.0
    exit_order_id: Optional[str] = None
    exit_reason: Optional[str] = None

    # Estado interno
    _state: Optional[CompositeOrderState] = field(default=None, repr=False)

    def __post_init__(self):
        """Valida y prepara la orden bracket"""
        # Validar precios
        if self.side == OrderSide.BUY:  # Long entry
            if self.entry_price and self.stop_loss_price >= self.entry_price:
                raise ValueError("Stop loss debe ser menor que precio de entrada para long")
            if self.entry_price and self.take_profit_price <= self.entry_price:
                raise ValueError("Take profit debe ser mayor que precio de entrada para long")
        else:  # Short entry
            if self.entry_price and self.stop_loss_price <= self.entry_price:
                raise ValueError("Stop loss debe ser mayor que precio de entrada para short")
            if self.entry_price and self.take_profit_price >= self.entry_price:
                raise ValueError("Take profit debe ser menor que precio de entrada para short")

        # Inicializar estado
        self._state = CompositeOrderState(
            parent_id=self.id,
            child_order_ids=[
                self.entry_order_id,
                self.stop_loss_order_id,
                self.take_profit_order_id
            ],
            status=CompositeOrderStatus.PENDING,
        )

    def get_id(self) -> str:
        return self.id

    def get_type(self) -> OrderType:
        return OrderType.BRACKET

    def get_child_orders(self) -> List[Dict[str, Any]]:
        """
        Retorna las ordenes hijas como diccionarios.

        Solo retorna SL y TP si la entrada ya se ejecuto.
        """
        orders = []

        # Orden de entrada
        entry_order = {
            "id": self.entry_order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "type": self.entry_type.value,
            "quantity": self.quantity,
            "price": self.entry_price,
            "status": "filled" if self.entry_filled else "pending",
        }
        orders.append(entry_order)

        # SL y TP solo si entry esta filled
        if self.entry_filled:
            # Stop Loss
            sl_side = OrderSide.SELL if self.side == OrderSide.BUY else OrderSide.BUY
            sl_order = {
                "id": self.stop_loss_order_id,
                "symbol": self.symbol,
                "side": sl_side.value,
                "type": OrderType.STOP_LOSS.value,
                "quantity": self.entry_fill_quantity,
                "stop_price": self.stop_loss_price,
                "limit_price": self.stop_loss_limit_price,
                "status": "cancelled" if self.exit_order_id == self.take_profit_order_id else "pending",
            }
            orders.append(sl_order)

            # Take Profit
            tp_order = {
                "id": self.take_profit_order_id,
                "symbol": self.symbol,
                "side": sl_side.value,
                "type": OrderType.TAKE_PROFIT.value,
                "quantity": self.entry_fill_quantity,
                "stop_price": self.take_profit_price,
                "limit_price": self.take_profit_limit_price,
                "status": "cancelled" if self.exit_order_id == self.stop_loss_order_id else "pending",
            }
            orders.append(tp_order)

        return orders

    def on_price_update(
        self,
        current_price: float,
        timestamp: datetime,
        bid: Optional[float] = None,
        ask: Optional[float] = None
    ) -> Optional[OrderAction]:
        """
        Verifica si SL o TP deben ejecutarse.

        Solo procesa si la entrada ya se ejecuto.
        """
        if not self.entry_filled:
            return None

        if self.status == CompositeOrderStatus.COMPLETED:
            return None

        # Verificar Stop Loss
        if self.side == OrderSide.BUY:  # Long position
            if current_price <= self.stop_loss_price:
                self.exit_order_id = self.stop_loss_order_id
                self.exit_reason = "stop_loss"
                return OrderAction(
                    action_type="execute",
                    order_id=self.stop_loss_order_id,
                    execution_price=current_price,
                    message="Stop loss triggered"
                )
            elif current_price >= self.take_profit_price:
                self.exit_order_id = self.take_profit_order_id
                self.exit_reason = "take_profit"
                return OrderAction(
                    action_type="execute",
                    order_id=self.take_profit_order_id,
                    execution_price=current_price,
                    message="Take profit triggered"
                )
        else:  # Short position
            if current_price >= self.stop_loss_price:
                self.exit_order_id = self.stop_loss_order_id
                self.exit_reason = "stop_loss"
                return OrderAction(
                    action_type="execute",
                    order_id=self.stop_loss_order_id,
                    execution_price=current_price,
                    message="Stop loss triggered"
                )
            elif current_price <= self.take_profit_price:
                self.exit_order_id = self.take_profit_order_id
                self.exit_reason = "take_profit"
                return OrderAction(
                    action_type="execute",
                    order_id=self.take_profit_order_id,
                    execution_price=current_price,
                    message="Take profit triggered"
                )

        return None

    def on_child_filled(
        self,
        child_order_id: str,
        fill_price: float,
        fill_quantity: float
    ) -> Optional[OrderAction]:
        """
        Procesa ejecucion de orden hija.

        - Si entry se ejecuta: activa SL y TP
        - Si SL se ejecuta: cancela TP
        - Si TP se ejecuta: cancela SL
        """
        if child_order_id == self.entry_order_id:
            # Entry filled - activar SL y TP
            self.entry_filled = True
            self.entry_fill_price = fill_price
            self.entry_fill_quantity = fill_quantity
            self.status = CompositeOrderStatus.ACTIVE
            self._state.trigger_filled = True
            self._state.activate()
            return OrderAction(
                action_type="activate_exits",
                order_id=self.id,
                message="Entry filled, SL and TP activated"
            )

        elif child_order_id == self.stop_loss_order_id:
            # SL filled - cancelar TP
            self.exit_order_id = self.stop_loss_order_id
            self.exit_reason = "stop_loss"
            self.status = CompositeOrderStatus.COMPLETED
            return OrderAction(
                action_type="cancel",
                order_id=self.take_profit_order_id,
                message="Stop loss filled, cancelling take profit"
            )

        elif child_order_id == self.take_profit_order_id:
            # TP filled - cancelar SL
            self.exit_order_id = self.take_profit_order_id
            self.exit_reason = "take_profit"
            self.status = CompositeOrderStatus.COMPLETED
            return OrderAction(
                action_type="cancel",
                order_id=self.stop_loss_order_id,
                message="Take profit filled, cancelling stop loss"
            )

        return None

    def cancel(self) -> OrderAction:
        """Cancela todo el bracket"""
        self.status = CompositeOrderStatus.CANCELLED
        return OrderAction(
            action_type="cancel_all",
            order_id=self.id,
            message="Bracket order cancelled"
        )

    def is_complete(self) -> bool:
        return self.status in (
            CompositeOrderStatus.COMPLETED,
            CompositeOrderStatus.CANCELLED
        )

    def can_execute_on_exchange(self, exchange_id: str) -> bool:
        return BRACKET_ORDER_SUPPORT.get(exchange_id.lower(), False)

    def to_ccxt_params(self, exchange_id: str) -> Dict[str, Any]:
        """
        Convierte a parametros CCXT.

        Bybit soporta bracket nativo.
        Para otros exchanges, se deben enviar ordenes separadas.
        """
        exchange = exchange_id.lower()

        if exchange == "bybit":
            params = {
                "symbol": self.symbol,
                "side": self.side.value.capitalize(),
                "type": "Limit" if self.entry_price else "Market",
                "qty": str(self.quantity),
                "price": str(self.entry_price) if self.entry_price else None,
                "takeProfit": str(self.take_profit_price),
                "stopLoss": str(self.stop_loss_price),
            }
            if self.take_profit_limit_price:
                params["tpLimitPrice"] = str(self.take_profit_limit_price)
            if self.stop_loss_limit_price:
                params["slLimitPrice"] = str(self.stop_loss_limit_price)
            return params

        # Para otros exchanges, retornar solo la entrada
        # SL y TP se enviaran despues de entry fill
        return {
            "symbol": self.symbol,
            "side": self.side.value,
            "type": self.entry_type.value,
            "amount": self.quantity,
            "price": self.entry_price,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serializa el bracket a diccionario"""
        return {
            "id": self.id,
            "type": OrderType.BRACKET.value,
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "entry_type": self.entry_type.value,
            "stop_loss_price": self.stop_loss_price,
            "take_profit_price": self.take_profit_price,
            "stop_loss_limit_price": self.stop_loss_limit_price,
            "take_profit_limit_price": self.take_profit_limit_price,
            "entry_order_id": self.entry_order_id,
            "stop_loss_order_id": self.stop_loss_order_id,
            "take_profit_order_id": self.take_profit_order_id,
            "entry_filled": self.entry_filled,
            "entry_fill_price": self.entry_fill_price,
            "entry_fill_quantity": self.entry_fill_quantity,
            "exit_order_id": self.exit_order_id,
            "exit_reason": self.exit_reason,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
        }
