"""
Order Simulator

Simula la ejecucion de ordenes de trading con realismo.
Incluye modelos de slippage, comisiones y latencia.
"""

import asyncio
import logging
import random
from datetime import datetime
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass

from .models import (
    Order,
    OrderResult,
    OrderType,
    OrderSide,
    OrderStatus,
)
from .config import PaperTradingConfig, SlippageModel


logger = logging.getLogger(__name__)


@dataclass
class MarketState:
    """Estado actual del mercado para simulacion"""
    current_price: float
    bid_price: float
    ask_price: float
    spread: float
    volatility: float  # ATR o similar
    volume_24h: float


class OrderSimulator:
    """
    Simula la ejecucion de ordenes de trading.

    Caracteristicas:
    - Slippage basado en volatilidad o fijo
    - Comisiones configurables
    - Latencia simulada
    - Validacion de ordenes
    - Ejecuciones parciales (opcional)

    Example:
        simulator = OrderSimulator(config)

        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            type=OrderType.MARKET,
            quantity=0.1
        )

        result = await simulator.submit_order(order, market_price=50000)

    Attributes:
        config: Configuracion de paper trading
        pending_orders: Ordenes pendientes de ejecucion
        order_history: Historial de ordenes
    """

    def __init__(self, config: PaperTradingConfig):
        """
        Inicializa el simulador.

        Args:
            config: Configuracion de paper trading
        """
        self.config = config
        self.pending_orders: Dict[str, Order] = {}
        self.order_history: List[OrderResult] = []
        self._market_states: Dict[str, MarketState] = {}

        # Callbacks
        self.on_order_filled: Optional[Callable[[OrderResult], None]] = None
        self.on_order_rejected: Optional[Callable[[OrderResult], None]] = None

    def update_market_state(
        self,
        symbol: str,
        price: float,
        volatility: float = 0.0,
        spread: float = 0.0001,
        volume_24h: float = 1000000
    ):
        """
        Actualiza el estado del mercado para un simbolo.

        Args:
            symbol: Par de trading
            price: Precio actual
            volatility: Volatilidad (ATR normalizado)
            spread: Spread bid-ask
            volume_24h: Volumen 24h
        """
        half_spread = price * spread / 2
        self._market_states[symbol] = MarketState(
            current_price=price,
            bid_price=price - half_spread,
            ask_price=price + half_spread,
            spread=spread,
            volatility=volatility if volatility > 0 else 0.01,
            volume_24h=volume_24h
        )

    def get_market_state(self, symbol: str) -> Optional[MarketState]:
        """Obtiene el estado del mercado para un simbolo"""
        return self._market_states.get(symbol)

    async def submit_order(
        self,
        order: Order,
        market_price: Optional[float] = None
    ) -> OrderResult:
        """
        Envia una orden para ejecucion.

        Args:
            order: Orden a ejecutar
            market_price: Precio de mercado actual (opcional si ya se actualizo)

        Returns:
            Resultado de la ejecucion
        """
        # Actualizar estado del mercado si se proporciona precio
        if market_price:
            self.update_market_state(order.symbol, market_price)

        # Obtener estado del mercado
        market = self._market_states.get(order.symbol)
        if not market:
            return self._reject_order(
                order,
                "No hay datos de mercado disponibles"
            )

        # Simular latencia
        if self.config.latency_ms > 0:
            await asyncio.sleep(self.config.latency_ms / 1000)

        # Simular rechazo aleatorio
        if self.config.simulate_rejects:
            if random.random() < self.config.reject_probability:
                return self._reject_order(order, "Orden rechazada (simulado)")

        # Validar orden
        validation_error = self._validate_order(order, market)
        if validation_error:
            return self._reject_order(order, validation_error)

        # Ejecutar segun tipo
        if order.type == OrderType.MARKET:
            return await self._execute_market_order(order, market)
        elif order.type == OrderType.LIMIT:
            return await self._execute_limit_order(order, market)
        elif order.type == OrderType.STOP_LOSS:
            return await self._execute_stop_order(order, market)
        elif order.type == OrderType.TAKE_PROFIT:
            return await self._execute_stop_order(order, market)
        else:
            return self._reject_order(order, f"Tipo de orden no soportado: {order.type}")

    def _validate_order(self, order: Order, market: MarketState) -> Optional[str]:
        """
        Valida una orden antes de ejecutar.

        Returns:
            Mensaje de error o None si es valida
        """
        if order.quantity <= 0:
            return "Cantidad debe ser positiva"

        if order.type == OrderType.LIMIT and not order.price:
            return "Orden LIMIT requiere precio"

        if order.type in (OrderType.STOP_LOSS, OrderType.STOP_LIMIT) and not order.stop_price:
            return "Orden STOP requiere stop_price"

        # Validar tamano maximo
        order_value = order.quantity * market.current_price
        max_value = self.config.get_max_position_value()
        if order_value > max_value:
            return f"Orden excede tamano maximo ({order_value:.2f} > {max_value:.2f})"

        return None

    async def _execute_market_order(
        self,
        order: Order,
        market: MarketState
    ) -> OrderResult:
        """Ejecuta una orden de mercado"""
        # Determinar precio base
        if order.side == OrderSide.BUY:
            base_price = market.ask_price
        else:
            base_price = market.bid_price

        # Calcular slippage
        slippage = self._calculate_slippage(order, market)

        # Aplicar slippage
        if order.side == OrderSide.BUY:
            executed_price = base_price * (1 + slippage)
        else:
            executed_price = base_price * (1 - slippage)

        # Calcular comision
        order_value = order.quantity * executed_price
        commission = self.config.get_commission(order_value, is_maker=False)

        # Actualizar orden
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.filled_price = executed_price
        order.commission = commission
        order.updated_at = datetime.now()

        result = OrderResult(
            order=order,
            success=True,
            message="Orden ejecutada exitosamente",
            executed_price=executed_price,
            executed_quantity=order.quantity,
            slippage=slippage,
            commission=commission,
            latency_ms=self.config.latency_ms
        )

        self.order_history.append(result)

        if self.on_order_filled:
            self.on_order_filled(result)

        logger.info(
            f"Orden {order.id} ejecutada: {order.side.value} "
            f"{order.quantity} @ {executed_price:.2f} "
            f"(slippage: {slippage*100:.3f}%, commission: {commission:.2f})"
        )

        return result

    async def _execute_limit_order(
        self,
        order: Order,
        market: MarketState
    ) -> OrderResult:
        """Ejecuta una orden limite"""
        # Verificar si se puede ejecutar inmediatamente
        can_fill = False

        if order.side == OrderSide.BUY:
            # Orden de compra se ejecuta si precio <= ask
            can_fill = order.price >= market.ask_price
        else:
            # Orden de venta se ejecuta si precio >= bid
            can_fill = order.price <= market.bid_price

        if can_fill:
            # Ejecutar al precio limite (mejor para el trader)
            executed_price = order.price

            # Calcular comision (maker fee para limit)
            order_value = order.quantity * executed_price
            commission = self.config.get_commission(order_value, is_maker=True)

            order.status = OrderStatus.FILLED
            order.filled_quantity = order.quantity
            order.filled_price = executed_price
            order.commission = commission
            order.updated_at = datetime.now()

            result = OrderResult(
                order=order,
                success=True,
                message="Orden limite ejecutada",
                executed_price=executed_price,
                executed_quantity=order.quantity,
                slippage=0.0,  # Sin slippage en limit fills
                commission=commission,
                latency_ms=self.config.latency_ms
            )

            self.order_history.append(result)

            if self.on_order_filled:
                self.on_order_filled(result)

            return result

        else:
            # Agregar a ordenes pendientes
            order.status = OrderStatus.SUBMITTED
            order.updated_at = datetime.now()
            self.pending_orders[order.id] = order

            logger.info(f"Orden limite {order.id} en espera @ {order.price}")

            return OrderResult(
                order=order,
                success=True,
                message="Orden limite en espera",
                executed_price=0,
                executed_quantity=0,
                slippage=0,
                commission=0,
                latency_ms=self.config.latency_ms
            )

    async def _execute_stop_order(
        self,
        order: Order,
        market: MarketState
    ) -> OrderResult:
        """Ejecuta una orden stop"""
        # Verificar si stop fue alcanzado
        stop_triggered = False

        if order.type == OrderType.STOP_LOSS:
            if order.side == OrderSide.SELL:
                # Stop loss de posicion long: se activa si precio <= stop
                stop_triggered = market.current_price <= order.stop_price
            else:
                # Stop loss de posicion short: se activa si precio >= stop
                stop_triggered = market.current_price >= order.stop_price

        elif order.type == OrderType.TAKE_PROFIT:
            if order.side == OrderSide.SELL:
                # Take profit de posicion long: se activa si precio >= stop
                stop_triggered = market.current_price >= order.stop_price
            else:
                # Take profit de posicion short: se activa si precio <= stop
                stop_triggered = market.current_price <= order.stop_price

        if stop_triggered:
            # Convertir a orden de mercado y ejecutar
            order.type = OrderType.MARKET
            return await self._execute_market_order(order, market)
        else:
            # Agregar a ordenes pendientes
            order.status = OrderStatus.SUBMITTED
            order.updated_at = datetime.now()
            self.pending_orders[order.id] = order

            logger.info(f"Orden stop {order.id} en espera @ {order.stop_price}")

            return OrderResult(
                order=order,
                success=True,
                message="Orden stop en espera",
                executed_price=0,
                executed_quantity=0,
                slippage=0,
                commission=0,
                latency_ms=self.config.latency_ms
            )

    def _calculate_slippage(self, order: Order, market: MarketState) -> float:
        """
        Calcula el slippage para una orden.

        Returns:
            Slippage como decimal (0.001 = 0.1%)
        """
        if self.config.slippage_model == SlippageModel.NONE:
            return 0.0

        elif self.config.slippage_model == SlippageModel.FIXED:
            return self.config.fixed_slippage

        elif self.config.slippage_model == SlippageModel.VOLATILITY:
            # Slippage basado en volatilidad
            # Mayor volatilidad = mayor slippage
            base_slippage = self.config.fixed_slippage
            volatility_factor = market.volatility * 2  # Escalar volatilidad

            # Agregar componente aleatorio
            random_factor = random.uniform(0.5, 1.5)

            slippage = base_slippage * (1 + volatility_factor) * random_factor

            # Limitar al maximo
            return min(slippage, self.config.max_slippage)

        elif self.config.slippage_model == SlippageModel.ORDERBOOK:
            # Slippage basado en impacto de mercado
            # Ordenes mas grandes = mayor slippage
            order_value = order.quantity * market.current_price
            market_impact = order_value / market.volume_24h

            # Factor de impacto (simplificado)
            slippage = self.config.fixed_slippage + (market_impact * 0.1)

            return min(slippage, self.config.max_slippage)

        return self.config.fixed_slippage

    def _reject_order(self, order: Order, reason: str) -> OrderResult:
        """Rechaza una orden"""
        order.status = OrderStatus.REJECTED
        order.updated_at = datetime.now()

        result = OrderResult(
            order=order,
            success=False,
            message=reason,
            executed_price=0,
            executed_quantity=0,
            slippage=0,
            commission=0
        )

        self.order_history.append(result)

        if self.on_order_rejected:
            self.on_order_rejected(result)

        logger.warning(f"Orden {order.id} rechazada: {reason}")

        return result

    async def check_pending_orders(self, symbol: str, current_price: float):
        """
        Verifica ordenes pendientes contra precio actual.

        Args:
            symbol: Par de trading
            current_price: Precio actual
        """
        self.update_market_state(symbol, current_price)
        market = self._market_states[symbol]

        orders_to_check = [
            order for order in self.pending_orders.values()
            if order.symbol == symbol
        ]

        for order in orders_to_check:
            if order.type == OrderType.LIMIT:
                # Verificar si limite fue alcanzado
                should_fill = False

                if order.side == OrderSide.BUY:
                    should_fill = current_price <= order.price
                else:
                    should_fill = current_price >= order.price

                if should_fill:
                    del self.pending_orders[order.id]
                    await self._execute_limit_order(order, market)

            elif order.type in (OrderType.STOP_LOSS, OrderType.TAKE_PROFIT):
                # Verificar si stop fue alcanzado
                del self.pending_orders[order.id]
                await self._execute_stop_order(order, market)

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancela una orden pendiente.

        Args:
            order_id: ID de la orden a cancelar

        Returns:
            True si se cancelo, False si no existia
        """
        if order_id in self.pending_orders:
            order = self.pending_orders[order_id]
            order.status = OrderStatus.CANCELLED
            order.updated_at = datetime.now()
            del self.pending_orders[order_id]

            logger.info(f"Orden {order_id} cancelada")
            return True

        return False

    def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        """
        Cancela todas las ordenes pendientes.

        Args:
            symbol: Si se especifica, solo cancela ordenes de ese simbolo

        Returns:
            Numero de ordenes canceladas
        """
        cancelled = 0

        for order_id, order in list(self.pending_orders.items()):
            if symbol is None or order.symbol == symbol:
                self.cancel_order(order_id)
                cancelled += 1

        return cancelled

    def get_pending_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """
        Obtiene ordenes pendientes.

        Args:
            symbol: Filtrar por simbolo (opcional)

        Returns:
            Lista de ordenes pendientes
        """
        orders = list(self.pending_orders.values())

        if symbol:
            orders = [o for o in orders if o.symbol == symbol]

        return orders

    def get_order_history(
        self,
        symbol: Optional[str] = None,
        limit: int = 100
    ) -> List[OrderResult]:
        """
        Obtiene historial de ordenes.

        Args:
            symbol: Filtrar por simbolo (opcional)
            limit: Numero maximo de resultados

        Returns:
            Lista de resultados de ordenes
        """
        history = self.order_history

        if symbol:
            history = [r for r in history if r.order.symbol == symbol]

        return history[-limit:]
