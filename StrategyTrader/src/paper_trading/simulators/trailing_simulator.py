"""
Simulador de Trailing Stops.

Maneja la logica de simulacion para ordenes trailing stop,
incluyendo actualizacion de precios y ejecucion.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from ..orders.enums import OrderSide, OrderStatus
from ..orders.base import TrailingStopState, OrderAction
from ..orders.risk_control import TrailingStopOrder


logger = logging.getLogger(__name__)


class TrailingStopSimulator:
    """
    Simulador especializado para trailing stops.

    Mantiene seguimiento de multiples trailing stops y
    procesa actualizaciones de precio para determinar
    cuando deben ejecutarse.

    Attributes:
        active_orders: Diccionario de ordenes activas por ID
        executed_orders: Lista de ordenes ejecutadas
        on_execution: Callback cuando una orden se ejecuta
    """

    def __init__(self):
        self.active_orders: Dict[str, TrailingStopOrder] = {}
        self.executed_orders: List[TrailingStopOrder] = []
        self._execution_callbacks: List = []

    def add_order(self, order: TrailingStopOrder) -> bool:
        """
        Agrega una orden trailing stop al simulador.

        Args:
            order: Orden trailing stop a monitorear

        Returns:
            True si se agrego exitosamente
        """
        if order.id in self.active_orders:
            logger.warning(f"Orden {order.id} ya existe en el simulador")
            return False

        order.status = OrderStatus.SUBMITTED
        self.active_orders[order.id] = order
        logger.info(
            f"Trailing stop {order.id} agregado: "
            f"side={order.side.value}, stop={order.get_current_stop_price():.2f}"
        )
        return True

    def remove_order(self, order_id: str) -> Optional[TrailingStopOrder]:
        """
        Remueve una orden del simulador.

        Args:
            order_id: ID de la orden a remover

        Returns:
            Orden removida o None si no existia
        """
        return self.active_orders.pop(order_id, None)

    def process_price_update(
        self,
        symbol: str,
        current_price: float,
        timestamp: Optional[datetime] = None,
        bid: Optional[float] = None,
        ask: Optional[float] = None
    ) -> List[Tuple[TrailingStopOrder, OrderAction]]:
        """
        Procesa actualizacion de precio para todas las ordenes.

        Args:
            symbol: Par de trading
            current_price: Precio actual
            timestamp: Momento de la actualizacion
            bid: Precio bid (opcional)
            ask: Precio ask (opcional)

        Returns:
            Lista de tuplas (orden, accion) para ordenes que requieren accion
        """
        if timestamp is None:
            timestamp = datetime.now()

        actions: List[Tuple[TrailingStopOrder, OrderAction]] = []
        orders_to_execute: List[str] = []

        for order_id, order in self.active_orders.items():
            if order.symbol != symbol:
                continue

            action = order.on_price_update(
                current_price=current_price,
                timestamp=timestamp,
                bid=bid,
                ask=ask
            )

            if action:
                actions.append((order, action))

                if action.action_type == "execute":
                    orders_to_execute.append(order_id)
                    logger.info(
                        f"Trailing stop {order_id} triggered at {current_price:.2f} "
                        f"(stop was {order.get_current_stop_price():.2f})"
                    )
                elif action.action_type == "modify":
                    logger.debug(
                        f"Trailing stop {order_id} updated: "
                        f"new stop = {action.new_stop_price:.2f}"
                    )

        # Mover ordenes ejecutadas
        for order_id in orders_to_execute:
            order = self.active_orders.pop(order_id)
            order.status = OrderStatus.FILLED
            self.executed_orders.append(order)

        return actions

    def get_order(self, order_id: str) -> Optional[TrailingStopOrder]:
        """Obtiene una orden por ID"""
        return self.active_orders.get(order_id)

    def get_active_orders(self, symbol: Optional[str] = None) -> List[TrailingStopOrder]:
        """
        Obtiene ordenes activas.

        Args:
            symbol: Filtrar por simbolo (opcional)

        Returns:
            Lista de ordenes activas
        """
        orders = list(self.active_orders.values())
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        return orders

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancela una orden trailing stop.

        Args:
            order_id: ID de la orden a cancelar

        Returns:
            True si se cancelo, False si no existia
        """
        order = self.active_orders.pop(order_id, None)
        if order:
            order.status = OrderStatus.CANCELLED
            logger.info(f"Trailing stop {order_id} cancelado")
            return True
        return False

    def cancel_all(self, symbol: Optional[str] = None) -> int:
        """
        Cancela todas las ordenes.

        Args:
            symbol: Si se especifica, solo cancela ordenes de ese simbolo

        Returns:
            Numero de ordenes canceladas
        """
        cancelled = 0
        for order_id in list(self.active_orders.keys()):
            order = self.active_orders[order_id]
            if symbol is None or order.symbol == symbol:
                self.cancel_order(order_id)
                cancelled += 1
        return cancelled

    def get_statistics(self) -> Dict:
        """
        Obtiene estadisticas del simulador.

        Returns:
            Diccionario con estadisticas
        """
        return {
            "active_orders": len(self.active_orders),
            "executed_orders": len(self.executed_orders),
            "symbols": list(set(o.symbol for o in self.active_orders.values())),
        }

    def simulate_price_sequence(
        self,
        symbol: str,
        prices: List[float],
        start_time: Optional[datetime] = None,
        interval_seconds: int = 60
    ) -> List[Tuple[float, List[OrderAction]]]:
        """
        Simula una secuencia de precios.

        Util para testing y backtesting de trailing stops.

        Args:
            symbol: Par de trading
            prices: Lista de precios a simular
            start_time: Tiempo inicial
            interval_seconds: Intervalo entre precios

        Returns:
            Lista de tuplas (precio, acciones) para cada precio
        """
        if start_time is None:
            start_time = datetime.now()

        results = []
        current_time = start_time

        for price in prices:
            actions = self.process_price_update(
                symbol=symbol,
                current_price=price,
                timestamp=current_time
            )
            # Extraer solo las acciones
            action_list = [action for _, action in actions]
            results.append((price, action_list))

            from datetime import timedelta
            current_time = current_time + timedelta(seconds=interval_seconds)

        return results
