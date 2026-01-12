"""
Simulador de fallback para ordenes no soportadas.

Cuando un broker no soporta un tipo de orden de forma nativa,
este modulo simula la orden localmente usando los simuladores
del modulo paper_trading.
"""

from typing import Optional, Dict, Any
from datetime import datetime

from ..core.enums import OrderType, OrderStatus
from ..core.models import BrokerOrder, BrokerCapabilities, ExecutionReport
from ..core.interfaces import IBrokerAdapter
from ..core.exceptions import UnsupportedOrderTypeError


class FallbackSimulator:
    """
    Simulador de fallback para ordenes no soportadas.

    Ejecuta ordenes usando simulacion local cuando el broker
    no las soporta nativamente.
    """

    def __init__(self, broker: IBrokerAdapter):
        """
        Inicializar simulador.

        Args:
            broker: Broker a usar para ordenes soportadas
        """
        self._broker = broker

        # Cache de ordenes en simulacion
        self._simulated_orders: Dict[str, BrokerOrder] = {}

        # Intentar importar simuladores de paper_trading
        self._trailing_simulator = None
        self._algo_simulator = None
        self._composite_simulator = None

        try:
            from paper_trading.simulators.trailing_simulator import TrailingStopSimulator
            self._trailing_simulator = TrailingStopSimulator()
        except ImportError:
            pass

        try:
            from paper_trading.simulators.algo_simulator import AlgoOrderSimulator
            self._algo_simulator = AlgoOrderSimulator()
        except ImportError:
            pass

        try:
            from paper_trading.simulators.composite_simulator import CompositeOrderSimulator
            self._composite_simulator = CompositeOrderSimulator()
        except ImportError:
            pass

    async def execute_with_fallback(
        self,
        order: BrokerOrder,
        capabilities: BrokerCapabilities
    ) -> ExecutionReport:
        """
        Ejecutar orden con fallback a simulacion si es necesario.

        Args:
            order: Orden a ejecutar
            capabilities: Capacidades del broker

        Returns:
            ExecutionReport con el resultado
        """
        # Verificar si el tipo es soportado nativamente
        if self._is_natively_supported(order.order_type, capabilities):
            return await self._broker.submit_order(order)

        # Fallback a simulacion
        return await self._simulate_order(order, capabilities)

    def _is_natively_supported(
        self,
        order_type: OrderType,
        capabilities: BrokerCapabilities
    ) -> bool:
        """
        Verificar si el tipo de orden es soportado nativamente.

        Args:
            order_type: Tipo de orden
            capabilities: Capacidades del broker

        Returns:
            True si es soportado
        """
        # Los tipos basicos siempre estan soportados
        basic_types = {
            OrderType.MARKET,
            OrderType.LIMIT,
            OrderType.STOP_LOSS,
            OrderType.STOP_LIMIT,
            OrderType.TAKE_PROFIT
        }
        if order_type in basic_types:
            return True

        # Verificar tipos avanzados
        return capabilities.supports_order_type(order_type)

    async def _simulate_order(
        self,
        order: BrokerOrder,
        capabilities: BrokerCapabilities
    ) -> ExecutionReport:
        """
        Simular orden localmente.

        Args:
            order: Orden a simular
            capabilities: Capacidades del broker

        Returns:
            ExecutionReport con el resultado
        """
        if order.order_type == OrderType.TRAILING_STOP:
            return await self._simulate_trailing_stop(order)

        elif order.order_type in [OrderType.TWAP, OrderType.VWAP]:
            return await self._simulate_algo_order(order)

        elif order.order_type in [OrderType.BRACKET, OrderType.OCO, OrderType.OTOCO]:
            return await self._simulate_composite_order(order)

        # Tipo no soportado
        raise UnsupportedOrderTypeError(
            order.order_type.value,
            self._broker.broker_id,
            order.id
        )

    async def _simulate_trailing_stop(self, order: BrokerOrder) -> ExecutionReport:
        """Simular trailing stop"""
        if not self._trailing_simulator:
            raise UnsupportedOrderTypeError(
                "trailing_stop",
                self._broker.broker_id,
                order.id
            )

        # Obtener precio actual
        ticker = await self._broker.get_ticker(order.symbol)
        current_price = ticker.get("last", 0)

        # Crear trailing stop en simulador
        try:
            from paper_trading.orders.risk_control import TrailingStopOrder
            from paper_trading.orders.enums import OrderSide as PTOrderSide

            pt_side = PTOrderSide.SELL if order.side.value == "sell" else PTOrderSide.BUY

            trailing = TrailingStopOrder(
                symbol=order.symbol,
                side=pt_side,
                quantity=order.quantity,
                trail_percent=order.trail_percent,
                trail_amount=order.trail_amount,
                activation_price=order.activation_price,
                initial_price=current_price
            )

            self._trailing_simulator.add_order(trailing)
            self._simulated_orders[trailing.id] = order
            order.id = trailing.id

            return ExecutionReport(
                order_id=trailing.id,
                client_order_id=order.client_order_id,
                status=OrderStatus.OPEN,
                filled_quantity=0,
                remaining_quantity=order.quantity,
                average_price=0,
                commission=0,
                timestamp=datetime.now(),
                original_order=order,
            )

        except ImportError:
            raise UnsupportedOrderTypeError(
                "trailing_stop",
                self._broker.broker_id,
                order.id
            )

    async def _simulate_algo_order(self, order: BrokerOrder) -> ExecutionReport:
        """Simular orden algoritmica (TWAP/VWAP)"""
        if not self._algo_simulator:
            raise UnsupportedOrderTypeError(
                order.order_type.value,
                self._broker.broker_id,
                order.id
            )

        try:
            from paper_trading.simulators.algo_simulator import TWAPConfig, VWAPConfig
            from paper_trading.orders.enums import OrderSide as PTOrderSide

            pt_side = PTOrderSide.SELL if order.side.value == "sell" else PTOrderSide.BUY

            if order.order_type == OrderType.TWAP:
                config = TWAPConfig(
                    total_quantity=order.quantity,
                    duration_seconds=order.duration_seconds or 3600,
                    slice_count=order.slice_count or 60,
                )
                order_id = self._algo_simulator.create_twap_order(
                    order.symbol, pt_side, config
                )
            else:  # VWAP
                config = VWAPConfig(
                    total_quantity=order.quantity,
                    duration_seconds=order.duration_seconds or 3600,
                    max_participation=order.max_participation or 0.05,
                )
                order_id = self._algo_simulator.create_vwap_order(
                    order.symbol, pt_side, config
                )

            self._simulated_orders[order_id] = order
            order.id = order_id

            return ExecutionReport(
                order_id=order_id,
                client_order_id=order.client_order_id,
                status=OrderStatus.OPEN,
                filled_quantity=0,
                remaining_quantity=order.quantity,
                average_price=0,
                commission=0,
                timestamp=datetime.now(),
                original_order=order,
            )

        except ImportError:
            raise UnsupportedOrderTypeError(
                order.order_type.value,
                self._broker.broker_id,
                order.id
            )

    async def _simulate_composite_order(self, order: BrokerOrder) -> ExecutionReport:
        """Simular orden compuesta (Bracket, OCO, OTOCO)"""
        if not self._composite_simulator:
            raise UnsupportedOrderTypeError(
                order.order_type.value,
                self._broker.broker_id,
                order.id
            )

        try:
            from paper_trading.orders.enums import OrderSide as PTOrderSide

            pt_side = PTOrderSide.SELL if order.side.value == "sell" else PTOrderSide.BUY

            # Obtener precio actual
            ticker = await self._broker.get_ticker(order.symbol)
            current_price = ticker.get("last", 0)

            if order.order_type == OrderType.BRACKET:
                from paper_trading.orders.risk_control import BracketOrder

                bracket = BracketOrder(
                    symbol=order.symbol,
                    side=pt_side,
                    quantity=order.quantity,
                    entry_price=order.price or current_price,
                    stop_loss_price=order.stop_loss,
                    take_profit_price=order.take_profit
                )

                self._composite_simulator.add_bracket_order(bracket)
                self._simulated_orders[bracket.id] = order
                order.id = bracket.id

            elif order.order_type == OrderType.OCO:
                from paper_trading.orders.conditional_orders import OCOOrder

                oco = OCOOrder(
                    symbol=order.symbol,
                    quantity=order.quantity,
                    side=pt_side,
                    stop_price=order.stop_loss,
                    limit_price=order.take_profit
                )

                self._composite_simulator.add_oco_order(oco)
                self._simulated_orders[oco.id] = order
                order.id = oco.id

            return ExecutionReport(
                order_id=order.id,
                client_order_id=order.client_order_id,
                status=OrderStatus.OPEN,
                filled_quantity=0,
                remaining_quantity=order.quantity,
                average_price=0,
                commission=0,
                timestamp=datetime.now(),
                original_order=order,
            )

        except ImportError:
            raise UnsupportedOrderTypeError(
                order.order_type.value,
                self._broker.broker_id,
                order.id
            )

    async def process_price_update(
        self,
        symbol: str,
        price: float,
        timestamp: Optional[datetime] = None
    ) -> list:
        """
        Procesar actualizacion de precio para ordenes simuladas.

        Args:
            symbol: Simbolo
            price: Nuevo precio
            timestamp: Timestamp

        Returns:
            Lista de ejecuciones
        """
        timestamp = timestamp or datetime.now()
        executions = []

        # Procesar trailing stops
        if self._trailing_simulator:
            actions = self._trailing_simulator.process_price_update(
                symbol, price, timestamp
            )
            for order_id, action in actions:
                if action.action_type == "execute":
                    original = self._simulated_orders.get(order_id)
                    if original:
                        report = ExecutionReport(
                            order_id=order_id,
                            status=OrderStatus.FILLED,
                            filled_quantity=original.quantity,
                            remaining_quantity=0,
                            average_price=price,
                            commission=0,
                            timestamp=timestamp,
                            original_order=original,
                        )
                        executions.append(report)

        # Procesar ordenes algoritmicas
        if self._algo_simulator:
            algo_execs = self._algo_simulator.process_tick(
                symbol, price, timestamp
            )
            for order_id, qty, exec_price in algo_execs:
                original = self._simulated_orders.get(order_id)
                if original:
                    report = ExecutionReport(
                        order_id=order_id,
                        status=OrderStatus.PARTIALLY_FILLED,
                        filled_quantity=qty,
                        remaining_quantity=original.quantity - qty,
                        average_price=exec_price,
                        commission=0,
                        timestamp=timestamp,
                        original_order=original,
                    )
                    executions.append(report)

        return executions

    def get_simulated_orders(self) -> Dict[str, BrokerOrder]:
        """Obtener ordenes en simulacion"""
        return dict(self._simulated_orders)

    def cancel_simulated_order(self, order_id: str) -> bool:
        """
        Cancelar orden simulada.

        Args:
            order_id: ID de la orden

        Returns:
            True si se cancelo
        """
        if order_id in self._simulated_orders:
            # Cancelar en simuladores
            if self._trailing_simulator:
                self._trailing_simulator.cancel_order(order_id)
            if self._algo_simulator:
                self._algo_simulator.cancel_order(order_id)
            if self._composite_simulator:
                self._composite_simulator.cancel_order(order_id)

            del self._simulated_orders[order_id]
            return True

        return False
