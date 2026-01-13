"""
Simulador de Ordenes Compuestas.

Maneja la simulacion de ordenes que contienen multiples ordenes hijas:
- Bracket Order (Entry + SL + TP)
- OCO (One-Cancels-Other)
- OTOCO (One-Triggers-OCO)
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import uuid

from ..orders.enums import OrderSide, OrderStatus, CompositeOrderStatus
from ..orders.base import CompositeOrderState, OrderAction
from ..orders.risk_control import BracketOrder
from ..orders.conditional_orders import OCOOrder as ConditionalOCOOrder, OTOCOOrder as ConditionalOTOCOOrder


logger = logging.getLogger(__name__)


@dataclass
class OCOOrder:
    """
    Orden OCO (One-Cancels-Other).

    Dos ordenes vinculadas donde la ejecucion de una
    cancela automaticamente la otra.

    Tipicamente usado para stop loss + take profit.

    Attributes:
        symbol: Par de trading
        quantity: Cantidad de ambas ordenes
        order_a_type: Tipo de primera orden
        order_a_price: Precio de primera orden
        order_a_stop_price: Stop price de primera orden (si aplica)
        order_b_type: Tipo de segunda orden
        order_b_price: Precio de segunda orden
        order_b_stop_price: Stop price de segunda orden (si aplica)
    """
    symbol: str
    quantity: float
    side: OrderSide
    order_a_price: float
    order_a_stop_price: Optional[float]
    order_b_price: float
    order_b_stop_price: Optional[float]
    order_a_type: str = "stop_loss"
    order_b_type: str = "take_profit"
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    order_a_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    order_b_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    status: CompositeOrderStatus = CompositeOrderStatus.ACTIVE
    created_at: datetime = field(default_factory=datetime.now)
    filled_order_id: Optional[str] = None
    fill_price: float = 0.0

    def check_triggers(
        self,
        current_price: float,
        position_side: str = "long"
    ) -> Optional[Tuple[str, str, float]]:
        """
        Verifica si alguna orden debe ejecutarse.

        Args:
            current_price: Precio actual
            position_side: "long" o "short"

        Returns:
            Tupla (order_id, order_type, trigger_price) o None
        """
        if self.status != CompositeOrderStatus.ACTIVE:
            return None

        if position_side == "long":
            # Para long: SL si precio <= stop, TP si precio >= target
            if self.order_a_stop_price and current_price <= self.order_a_stop_price:
                return (self.order_a_id, self.order_a_type, self.order_a_stop_price)
            if self.order_b_stop_price and current_price >= self.order_b_stop_price:
                return (self.order_b_id, self.order_b_type, self.order_b_stop_price)
        else:
            # Para short: SL si precio >= stop, TP si precio <= target
            if self.order_a_stop_price and current_price >= self.order_a_stop_price:
                return (self.order_a_id, self.order_a_type, self.order_a_stop_price)
            if self.order_b_stop_price and current_price <= self.order_b_stop_price:
                return (self.order_b_id, self.order_b_type, self.order_b_stop_price)

        return None

    def on_fill(self, filled_order_id: str, fill_price: float):
        """Procesa ejecucion de una de las ordenes"""
        self.filled_order_id = filled_order_id
        self.fill_price = fill_price
        self.status = CompositeOrderStatus.COMPLETED

    def get_cancelled_order_id(self) -> Optional[str]:
        """Retorna ID de la orden que debe cancelarse"""
        if self.filled_order_id == self.order_a_id:
            return self.order_b_id
        elif self.filled_order_id == self.order_b_id:
            return self.order_a_id
        return None


@dataclass
class OTOCOOrder:
    """
    Orden OTOCO (One-Triggers-OCO).

    Una orden trigger que al ejecutarse activa una orden OCO.

    Ejemplo: Orden limit de entrada que al llenarse activa SL + TP.

    Attributes:
        symbol: Par de trading
        trigger_side: Lado del trigger (BUY/SELL)
        trigger_price: Precio del trigger (limit order)
        trigger_quantity: Cantidad del trigger
        oco_sl_price: Precio stop loss del OCO
        oco_tp_price: Precio take profit del OCO
    """
    symbol: str
    trigger_side: OrderSide
    trigger_price: float
    trigger_quantity: float
    oco_sl_price: float
    oco_tp_price: float
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    trigger_order_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    status: CompositeOrderStatus = CompositeOrderStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    trigger_filled: bool = False
    trigger_fill_price: float = 0.0
    trigger_fill_quantity: float = 0.0
    oco: Optional[OCOOrder] = None

    def check_trigger(self, current_price: float, bid: float, ask: float) -> bool:
        """
        Verifica si el trigger debe activarse.

        Returns:
            True si el trigger se activo
        """
        if self.trigger_filled:
            return False

        if self.trigger_side == OrderSide.BUY:
            # Compra: trigger si ask <= precio
            return ask <= self.trigger_price
        else:
            # Venta: trigger si bid >= precio
            return bid >= self.trigger_price

    def activate_oco(self, fill_price: float, fill_quantity: float):
        """Activa el OCO despues de trigger fill"""
        self.trigger_filled = True
        self.trigger_fill_price = fill_price
        self.trigger_fill_quantity = fill_quantity
        self.status = CompositeOrderStatus.ACTIVE

        # Determinar lado del OCO (opuesto al trigger)
        oco_side = OrderSide.SELL if self.trigger_side == OrderSide.BUY else OrderSide.BUY

        self.oco = OCOOrder(
            symbol=self.symbol,
            quantity=fill_quantity,
            side=oco_side,
            order_a_price=self.oco_sl_price,
            order_a_stop_price=self.oco_sl_price,
            order_b_price=self.oco_tp_price,
            order_b_stop_price=self.oco_tp_price,
        )


class CompositeOrderSimulator:
    """
    Simulador de ordenes compuestas.

    Maneja Bracket, OCO y OTOCO orders.

    Example:
        simulator = CompositeOrderSimulator()

        # Crear bracket order
        bracket = BracketOrder(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=0.1,
            entry_price=50000,
            stop_loss_price=49000,
            take_profit_price=52000
        )
        simulator.add_bracket(bracket)

        # Procesar precio
        actions = simulator.process_price_update("BTC/USDT", 50000)
    """

    def __init__(self):
        self.brackets: Dict[str, BracketOrder] = {}
        self.ocos: Dict[str, OCOOrder] = {}
        self.otocos: Dict[str, OTOCOOrder] = {}
        self.completed_orders: List[Any] = []

    def add_bracket(self, bracket: BracketOrder) -> str:
        """
        Agrega un bracket order.

        Returns:
            ID del bracket
        """
        self.brackets[bracket.id] = bracket
        logger.info(
            f"Bracket {bracket.id} added: entry={bracket.entry_price}, "
            f"SL={bracket.stop_loss_price}, TP={bracket.take_profit_price}"
        )
        return bracket.id

    def add_oco(self, oco) -> str:
        """
        Agrega una orden OCO.

        Acepta tanto OCOOrder interno como ConditionalOCOOrder.

        Returns:
            ID del OCO
        """
        self.ocos[oco.id] = oco
        # Compatibilidad con ambas clases OCO
        if hasattr(oco, 'stop_price'):
            logger.info(f"OCO {oco.id} added: SL={oco.stop_price}, TP={oco.limit_price}")
        else:
            logger.info(f"OCO {oco.id} added: A={oco.order_a_stop_price}, B={oco.order_b_stop_price}")
        return oco.id

    def add_otoco(self, otoco: OTOCOOrder) -> str:
        """
        Agrega una orden OTOCO.

        Returns:
            ID del OTOCO
        """
        self.otocos[otoco.id] = otoco
        logger.info(
            f"OTOCO {otoco.id} added: trigger={otoco.trigger_price}, "
            f"SL={otoco.oco_sl_price}, TP={otoco.oco_tp_price}"
        )
        return otoco.id

    def process_price_update(
        self,
        symbol: str,
        current_price: float,
        timestamp: Optional[datetime] = None,
        bid: Optional[float] = None,
        ask: Optional[float] = None,
    ) -> List[OrderAction]:
        """
        Procesa actualizacion de precio para todas las ordenes.

        Args:
            symbol: Par de trading
            current_price: Precio actual
            timestamp: Momento de la actualizacion
            bid: Precio bid
            ask: Precio ask

        Returns:
            Lista de acciones a ejecutar
        """
        if timestamp is None:
            timestamp = datetime.now()

        bid = bid or current_price * 0.9999
        ask = ask or current_price * 1.0001

        actions: List[OrderAction] = []

        # Procesar brackets
        actions.extend(self._process_brackets(symbol, current_price, timestamp))

        # Procesar OCOs
        actions.extend(self._process_ocos(symbol, current_price))

        # Procesar OTOCOs
        actions.extend(self._process_otocos(symbol, current_price, bid, ask))

        return actions

    def _process_brackets(
        self,
        symbol: str,
        current_price: float,
        timestamp: datetime
    ) -> List[OrderAction]:
        """Procesa brackets para un simbolo"""
        actions = []
        completed = []

        for bracket_id, bracket in self.brackets.items():
            if bracket.symbol != symbol:
                continue

            # Verificar entrada si no esta filled
            if not bracket.entry_filled:
                # Simular fill de entrada si precio alcanza
                if bracket.entry_price:
                    if bracket.side == OrderSide.BUY and current_price <= bracket.entry_price:
                        # Entry filled
                        action = bracket.on_child_filled(
                            bracket.entry_order_id,
                            bracket.entry_price,
                            bracket.quantity
                        )
                        if action:
                            actions.append(action)
                            logger.info(f"Bracket {bracket_id} entry filled at {bracket.entry_price}")
                    elif bracket.side == OrderSide.SELL and current_price >= bracket.entry_price:
                        action = bracket.on_child_filled(
                            bracket.entry_order_id,
                            bracket.entry_price,
                            bracket.quantity
                        )
                        if action:
                            actions.append(action)
                else:
                    # Market entry - fill inmediato
                    action = bracket.on_child_filled(
                        bracket.entry_order_id,
                        current_price,
                        bracket.quantity
                    )
                    if action:
                        actions.append(action)
                continue

            # Verificar SL/TP
            action = bracket.on_price_update(current_price, timestamp)
            if action:
                actions.append(action)

                if action.action_type == "execute":
                    # Procesar fill del SL o TP
                    fill_action = bracket.on_child_filled(
                        action.order_id,
                        current_price,
                        bracket.entry_fill_quantity
                    )
                    if fill_action:
                        actions.append(fill_action)

            if bracket.is_complete():
                completed.append(bracket_id)

        # Mover completados
        for bracket_id in completed:
            bracket = self.brackets.pop(bracket_id)
            self.completed_orders.append(bracket)
            logger.info(f"Bracket {bracket_id} completed: {bracket.exit_reason}")

        return actions

    def _process_ocos(
        self,
        symbol: str,
        current_price: float
    ) -> List[OrderAction]:
        """Procesa OCOs para un simbolo"""
        actions = []
        completed = []

        for oco_id, oco in self.ocos.items():
            if oco.symbol != symbol:
                continue

            # Manejar OCOOrder de conditional_orders.py
            if hasattr(oco, 'stop_price') and hasattr(oco, 'limit_price'):
                # Clase de conditional_orders.py
                trigger = oco.check_triggers(current_price)
                if trigger:
                    # trigger es "stop" o "limit"
                    if trigger == "stop":
                        order_id = oco.stop_order_id  # type: ignore
                        trigger_price = oco.stop_price  # type: ignore
                    else:
                        order_id = oco.limit_order_id  # type: ignore
                        trigger_price = oco.limit_price  # type: ignore

                    actions.append(OrderAction(
                        action_type="execute",
                        order_id=order_id,
                        execution_price=current_price,
                        message=f"OCO {trigger} triggered at {trigger_price}"
                    ))

                    oco.execute_order(trigger, current_price)  # type: ignore

                    cancelled_id = oco.get_cancelled_order_id()
                    if cancelled_id:
                        actions.append(OrderAction(
                            action_type="cancel",
                            order_id=cancelled_id,
                            message="OCO counterpart cancelled"
                        ))

                    completed.append(oco_id)
                    logger.info(f"OCO {oco_id} triggered: {trigger} at {current_price:.2f}")
            else:
                # Clase interna OCOOrder
                position_side = "long" if oco.side == OrderSide.SELL else "short"
                trigger = oco.check_triggers(current_price, position_side)
                if trigger:
                    order_id, order_type, trigger_price = trigger

                    actions.append(OrderAction(
                        action_type="execute",
                        order_id=order_id,
                        execution_price=current_price,
                        message=f"OCO {order_type} triggered at {trigger_price}"
                    ))

                    oco.on_fill(order_id, current_price)

                    # Cancelar la otra orden
                    cancelled_id = oco.get_cancelled_order_id()
                    if cancelled_id:
                        actions.append(OrderAction(
                            action_type="cancel",
                            order_id=cancelled_id,
                            message="OCO counterpart cancelled"
                        ))

                    completed.append(oco_id)
                    logger.info(
                        f"OCO {oco_id} triggered: {order_type} at {current_price:.2f}"
                    )

        # Mover completados
        for oco_id in completed:
            oco = self.ocos.pop(oco_id)
            self.completed_orders.append(oco)

        return actions

    def _process_otocos(
        self,
        symbol: str,
        current_price: float,
        bid: float,
        ask: float
    ) -> List[OrderAction]:
        """Procesa OTOCOs para un simbolo"""
        actions = []
        to_convert = []
        completed = []

        for otoco_id, otoco in self.otocos.items():
            if otoco.symbol != symbol:
                continue

            if not otoco.trigger_filled:
                # Verificar trigger
                if otoco.check_trigger(current_price, bid, ask):
                    fill_price = ask if otoco.trigger_side == OrderSide.BUY else bid
                    otoco.activate_oco(fill_price, otoco.trigger_quantity)

                    actions.append(OrderAction(
                        action_type="execute",
                        order_id=otoco.trigger_order_id,
                        execution_price=fill_price,
                        message=f"OTOCO trigger filled at {fill_price}"
                    ))

                    # Agregar OCO activo
                    to_convert.append((otoco_id, otoco.oco))

                    logger.info(f"OTOCO {otoco_id} triggered, OCO activated")
            else:
                # Procesar OCO interno
                if otoco.oco:
                    position_side = "long" if otoco.trigger_side == OrderSide.BUY else "short"
                    trigger = otoco.oco.check_triggers(current_price, position_side)

                    if trigger:
                        order_id, order_type, trigger_price = trigger
                        otoco.oco.on_fill(order_id, current_price)
                        otoco.status = CompositeOrderStatus.COMPLETED

                        actions.append(OrderAction(
                            action_type="execute",
                            order_id=order_id,
                            execution_price=current_price,
                            message=f"OTOCO {order_type} triggered"
                        ))

                        cancelled_id = otoco.oco.get_cancelled_order_id()
                        if cancelled_id:
                            actions.append(OrderAction(
                                action_type="cancel",
                                order_id=cancelled_id,
                                message="OTOCO OCO counterpart cancelled"
                            ))

                        completed.append(otoco_id)

        # Mover completados
        for otoco_id in completed:
            otoco = self.otocos.pop(otoco_id)
            self.completed_orders.append(otoco)

        return actions

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancela una orden por ID.

        Busca en brackets, OCOs y OTOCOs.

        Returns:
            True si se encontro y cancelo
        """
        # Buscar en brackets
        if order_id in self.brackets:
            bracket = self.brackets.pop(order_id)
            bracket.status = CompositeOrderStatus.CANCELLED
            self.completed_orders.append(bracket)
            logger.info(f"Bracket {order_id} cancelled")
            return True

        # Buscar en OCOs
        if order_id in self.ocos:
            oco = self.ocos.pop(order_id)
            oco.status = CompositeOrderStatus.CANCELLED
            self.completed_orders.append(oco)
            logger.info(f"OCO {order_id} cancelled")
            return True

        # Buscar en OTOCOs
        if order_id in self.otocos:
            otoco = self.otocos.pop(order_id)
            otoco.status = CompositeOrderStatus.CANCELLED
            self.completed_orders.append(otoco)
            logger.info(f"OTOCO {order_id} cancelled")
            return True

        return False

    def get_order_status(self, order_id: str) -> Optional[Dict]:
        """Obtiene estado de una orden"""
        # Buscar en activas
        if order_id in self.brackets:
            b = self.brackets[order_id]
            return {
                "type": "bracket",
                "id": b.id,
                "status": b.status.value,
                "entry_filled": b.entry_filled,
                "exit_reason": b.exit_reason,
            }

        if order_id in self.ocos:
            o = self.ocos[order_id]
            result: Dict[str, Any] = {
                "type": "oco",
                "id": o.id,
                "status": o.status.value,
            }
            # Handle both OCO types
            if hasattr(o, 'filled_order_id'):
                result["filled_order_id"] = o.filled_order_id
            elif hasattr(o, 'executed_order'):
                result["executed_order"] = o.executed_order  # type: ignore
            return result

        if order_id in self.otocos:
            t = self.otocos[order_id]
            return {
                "type": "otoco",
                "id": t.id,
                "status": t.status.value,
                "trigger_filled": t.trigger_filled,
            }

        return None

    def get_statistics(self) -> Dict:
        """Obtiene estadisticas del simulador"""
        return {
            "active_brackets": len(self.brackets),
            "active_ocos": len(self.ocos),
            "active_otocos": len(self.otocos),
            "completed_orders": len(self.completed_orders),
        }
