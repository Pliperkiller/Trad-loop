"""
Order Simulator

Simula la ejecucion de ordenes de trading con realismo.
Incluye modelos de slippage, comisiones y latencia.

Soporta ordenes avanzadas:
- Risk Control: Trailing Stop, Bracket Order
- Execution Algorithms: TWAP, VWAP, Iceberg, Hidden
- Dynamic: FOK, IOC, Reduce-Only, Post-Only, Scale In/Out
- Conditional: If-Touched, OCO, OTOCO
"""

import asyncio
import logging
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any, Union

from .models import (
    Order,
    OrderResult,
    OrderType,
    OrderSide,
    OrderStatus,
)
from .config import PaperTradingConfig, SlippageModel

# Advanced order imports
from .orders.enums import OrderType as AdvancedOrderType, TimeInForce
from .orders.base import AdvancedOrderParams
from .orders.risk_control import TrailingStopOrder, BracketOrder
from .orders.execution_algos import TWAPOrder, VWAPOrder, IcebergOrder, HiddenOrder
from .orders.dynamic_orders import FOKOrder, IOCOrder, ReduceOnlyOrder, PostOnlyOrder, ScaleOrder
from .orders.conditional_orders import IfTouchedOrder, OCOOrder, OTOCOOrder

# Specialized simulators
from .simulators.trailing_simulator import TrailingStopSimulator
from .simulators.algo_simulator import AlgoOrderSimulator, TWAPConfig, VWAPConfig, IcebergConfig
from .simulators.composite_simulator import CompositeOrderSimulator


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
    - Ordenes avanzadas (trailing stop, bracket, TWAP, VWAP, etc.)

    Example:
        simulator = OrderSimulator(config)

        # Orden basica
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            type=OrderType.MARKET,
            quantity=0.1
        )
        result = await simulator.submit_order(order, market_price=50000)

        # Trailing stop
        trailing = await simulator.create_trailing_stop(
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            quantity=0.1,
            trail_percent=0.02,
            initial_price=50000
        )

        # Bracket order
        bracket = await simulator.create_bracket_order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=0.1,
            entry_price=50000,
            stop_loss_price=49000,
            take_profit_price=52000
        )

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

        # Specialized simulators for advanced orders
        self._trailing_simulator = TrailingStopSimulator()
        self._algo_simulator = AlgoOrderSimulator()
        self._composite_simulator = CompositeOrderSimulator()

        # Advanced order storage
        self._trailing_orders: Dict[str, TrailingStopOrder] = {}
        self._bracket_orders: Dict[str, BracketOrder] = {}
        self._twap_orders: Dict[str, TWAPOrder] = {}
        self._vwap_orders: Dict[str, VWAPOrder] = {}
        self._iceberg_orders: Dict[str, IcebergOrder] = {}
        self._hidden_orders: Dict[str, HiddenOrder] = {}
        self._fok_orders: Dict[str, FOKOrder] = {}
        self._ioc_orders: Dict[str, IOCOrder] = {}
        self._reduce_only_orders: Dict[str, ReduceOnlyOrder] = {}
        self._post_only_orders: Dict[str, PostOnlyOrder] = {}
        self._scale_orders: Dict[str, ScaleOrder] = {}
        self._if_touched_orders: Dict[str, IfTouchedOrder] = {}
        self._oco_orders: Dict[str, OCOOrder] = {}
        self._otoco_orders: Dict[str, OTOCOOrder] = {}

        # Position quantities for reduce-only validation
        self._positions: Dict[str, float] = {}

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

    # =========================================================================
    # Position Management (for reduce-only validation)
    # =========================================================================

    def update_position(self, symbol: str, quantity: float):
        """
        Actualiza la posicion para un simbolo.

        Args:
            symbol: Par de trading
            quantity: Cantidad (positivo = long, negativo = short, 0 = sin posicion)
        """
        self._positions[symbol] = quantity

    def get_position(self, symbol: str) -> float:
        """Obtiene la posicion actual para un simbolo"""
        return self._positions.get(symbol, 0.0)

    # =========================================================================
    # Trailing Stop Orders
    # =========================================================================

    async def create_trailing_stop(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        trail_percent: Optional[float] = None,
        trail_amount: Optional[float] = None,
        initial_price: Optional[float] = None,
        activation_price: Optional[float] = None
    ) -> TrailingStopOrder:
        """
        Crea una orden trailing stop.

        Args:
            symbol: Par de trading
            side: SELL para proteger long, BUY para proteger short
            quantity: Cantidad
            trail_percent: Porcentaje de trail (ej: 0.02 = 2%)
            trail_amount: Cantidad fija de trail en precio
            initial_price: Precio inicial (usa mercado si no se especifica)
            activation_price: Precio para activar el trailing (opcional)

        Returns:
            Orden trailing stop creada
        """
        # Obtener precio inicial del mercado si no se especifica
        if initial_price is None:
            market = self._market_states.get(symbol)
            if market:
                initial_price = market.current_price
            else:
                raise ValueError(f"No hay datos de mercado para {symbol}")

        order = TrailingStopOrder(
            symbol=symbol,
            side=side,
            quantity=quantity,
            trail_percent=trail_percent,
            trail_amount=trail_amount,
            initial_price=initial_price,
            activation_price=activation_price
        )

        self._trailing_orders[order.id] = order
        self._trailing_simulator.add_order(order)

        logger.info(
            f"Trailing stop {order.id} creado: {side.value} {quantity} @ "
            f"stop={order.get_current_stop_price():.2f}"
        )

        return order

    # =========================================================================
    # Bracket Orders
    # =========================================================================

    async def create_bracket_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        entry_price: Optional[float],
        stop_loss_price: float,
        take_profit_price: float,
        entry_type: OrderType = OrderType.LIMIT
    ) -> BracketOrder:
        """
        Crea una orden bracket (entry + SL + TP).

        Args:
            symbol: Par de trading
            side: BUY para long, SELL para short
            quantity: Cantidad
            entry_price: Precio de entrada (None = market)
            stop_loss_price: Precio de stop loss
            take_profit_price: Precio de take profit
            entry_type: Tipo de orden de entrada

        Returns:
            Orden bracket creada
        """
        from .orders.enums import OrderType as AdvOrderType

        bracket = BracketOrder(
            symbol=symbol,
            side=side,
            quantity=quantity,
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            entry_type=AdvOrderType.MARKET if entry_price is None else AdvOrderType.LIMIT
        )

        self._bracket_orders[bracket.id] = bracket
        self._composite_simulator.add_bracket(bracket)

        logger.info(
            f"Bracket {bracket.id} creado: {side.value} {quantity} @ "
            f"entry={entry_price}, SL={stop_loss_price}, TP={take_profit_price}"
        )

        return bracket

    # =========================================================================
    # TWAP Orders
    # =========================================================================

    async def create_twap_order(
        self,
        symbol: str,
        side: OrderSide,
        total_quantity: float,
        duration_seconds: int,
        slice_count: Optional[int] = None,
        size_variation: float = 0.0
    ) -> str:
        """
        Crea una orden TWAP (Time-Weighted Average Price).

        Args:
            symbol: Par de trading
            side: BUY o SELL
            total_quantity: Cantidad total a ejecutar
            duration_seconds: Duracion en segundos
            slice_count: Numero de slices (default: 1 por minuto)
            size_variation: Variacion aleatoria del tamano (0-1)

        Returns:
            ID de la orden TWAP
        """
        config = TWAPConfig(
            total_quantity=total_quantity,
            duration_seconds=duration_seconds,
            slice_count=slice_count or (duration_seconds // 60),
            size_variation=size_variation
        )

        order_id = self._algo_simulator.create_twap_order(symbol, side, config)

        logger.info(
            f"TWAP {order_id} creado: {side.value} {total_quantity} en "
            f"{duration_seconds}s ({config.slice_count} slices)"
        )

        return order_id

    # =========================================================================
    # VWAP Orders
    # =========================================================================

    async def create_vwap_order(
        self,
        symbol: str,
        side: OrderSide,
        total_quantity: float,
        duration_seconds: int,
        max_participation: float = 0.05
    ) -> str:
        """
        Crea una orden VWAP (Volume-Weighted Average Price).

        Args:
            symbol: Par de trading
            side: BUY o SELL
            total_quantity: Cantidad total a ejecutar
            duration_seconds: Duracion en segundos
            max_participation: Participacion maxima del volumen (default 5%)

        Returns:
            ID de la orden VWAP
        """
        config = VWAPConfig(
            total_quantity=total_quantity,
            duration_seconds=duration_seconds,
            max_participation=max_participation
        )

        order_id = self._algo_simulator.create_vwap_order(symbol, side, config)

        logger.info(
            f"VWAP {order_id} creado: {side.value} {total_quantity} en "
            f"{duration_seconds}s (max part: {max_participation*100}%)"
        )

        return order_id

    # =========================================================================
    # Iceberg Orders
    # =========================================================================

    async def create_iceberg_order(
        self,
        symbol: str,
        side: OrderSide,
        total_quantity: float,
        display_quantity: float,
        price: float
    ) -> str:
        """
        Crea una orden Iceberg.

        Args:
            symbol: Par de trading
            side: BUY o SELL
            total_quantity: Cantidad total (oculta)
            display_quantity: Cantidad visible en order book
            price: Precio limite

        Returns:
            ID de la orden Iceberg
        """
        config = IcebergConfig(
            total_quantity=total_quantity,
            display_quantity=display_quantity,
            price=price
        )

        order_id = self._algo_simulator.create_iceberg_order(symbol, side, config)

        logger.info(
            f"Iceberg {order_id} creado: {side.value} {total_quantity} "
            f"(visible: {display_quantity}) @ {price}"
        )

        return order_id

    # =========================================================================
    # Hidden Orders
    # =========================================================================

    async def create_hidden_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        price: float
    ) -> str:
        """
        Crea una orden oculta (no visible en order book).

        Args:
            symbol: Par de trading
            side: BUY o SELL
            quantity: Cantidad
            price: Precio limite

        Returns:
            ID de la orden Hidden
        """
        order_id = self._algo_simulator.create_hidden_order(symbol, side, quantity, price)

        logger.info(f"Hidden {order_id} creado: {side.value} {quantity} @ {price}")

        return order_id

    # =========================================================================
    # FOK (Fill-or-Kill) Orders
    # =========================================================================

    async def submit_fok_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        price: float,
        available_liquidity: float
    ) -> OrderResult:
        """
        Envia una orden Fill-or-Kill.

        La orden se ejecuta completamente o se cancela.

        Args:
            symbol: Par de trading
            side: BUY o SELL
            quantity: Cantidad
            price: Precio limite
            available_liquidity: Liquidez disponible en el mercado

        Returns:
            Resultado de la orden
        """
        from .orders.enums import OrderSide as AdvOrderSide

        fok = FOKOrder(
            symbol=symbol,
            side=AdvOrderSide.BUY if side == OrderSide.BUY else AdvOrderSide.SELL,
            quantity=quantity,
            price=price
        )

        market = self._market_states.get(symbol)
        if not market:
            return self._create_advanced_result(fok, False, "No hay datos de mercado")

        current_price = market.ask_price if side == OrderSide.BUY else market.bid_price

        if fok.can_fill(available_liquidity, current_price):
            fok.execute(current_price)
            commission = self.config.get_commission(quantity * current_price, is_maker=False)

            logger.info(f"FOK {fok.id} ejecutada: {quantity} @ {current_price}")
            return self._create_advanced_result(
                fok, True, "FOK ejecutada completamente",
                executed_price=current_price,
                executed_quantity=quantity,
                commission=commission
            )
        else:
            fok.reject("Liquidez insuficiente o precio movido")
            logger.info(f"FOK {fok.id} rechazada")
            return self._create_advanced_result(fok, False, "FOK rechazada - liquidez insuficiente")

    # =========================================================================
    # IOC (Immediate-or-Cancel) Orders
    # =========================================================================

    async def submit_ioc_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        price: float,
        available_liquidity: float
    ) -> OrderResult:
        """
        Envia una orden Immediate-or-Cancel.

        Ejecuta lo que pueda inmediatamente y cancela el resto.

        Args:
            symbol: Par de trading
            side: BUY o SELL
            quantity: Cantidad
            price: Precio limite
            available_liquidity: Liquidez disponible

        Returns:
            Resultado de la orden
        """
        from .orders.enums import OrderSide as AdvOrderSide

        ioc = IOCOrder(
            symbol=symbol,
            side=AdvOrderSide.BUY if side == OrderSide.BUY else AdvOrderSide.SELL,
            quantity=quantity,
            price=price
        )

        market = self._market_states.get(symbol)
        if not market:
            return self._create_advanced_result(ioc, False, "No hay datos de mercado")

        current_price = market.ask_price if side == OrderSide.BUY else market.bid_price

        # Calcular cantidad que se puede llenar
        fill_qty = min(quantity, available_liquidity)

        # Verificar precio
        if side == OrderSide.BUY and current_price > price:
            fill_qty = 0
        elif side == OrderSide.SELL and current_price < price:
            fill_qty = 0

        if fill_qty > 0:
            ioc.partial_fill(fill_qty, current_price)
            ioc.finalize()
            commission = self.config.get_commission(fill_qty * current_price, is_maker=False)

            logger.info(f"IOC {ioc.id} parcialmente ejecutada: {fill_qty}/{quantity}")
            return self._create_advanced_result(
                ioc, True,
                f"IOC ejecutada parcialmente ({fill_qty}/{quantity})",
                executed_price=current_price,
                executed_quantity=fill_qty,
                commission=commission
            )
        else:
            ioc.finalize()
            logger.info(f"IOC {ioc.id} cancelada - sin fills")
            return self._create_advanced_result(ioc, False, "IOC cancelada - sin ejecucion")

    # =========================================================================
    # Reduce-Only Orders
    # =========================================================================

    async def submit_reduce_only_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        price: Optional[float] = None
    ) -> OrderResult:
        """
        Envia una orden reduce-only.

        Solo puede reducir una posicion existente.

        Args:
            symbol: Par de trading
            side: BUY (para cerrar short) o SELL (para cerrar long)
            quantity: Cantidad
            price: Precio limite (None = market)

        Returns:
            Resultado de la orden
        """
        from .orders.enums import OrderSide as AdvOrderSide

        position_qty = abs(self.get_position(symbol))

        reduce_only = ReduceOnlyOrder(
            symbol=symbol,
            side=AdvOrderSide.BUY if side == OrderSide.BUY else AdvOrderSide.SELL,
            quantity=quantity,
            position_quantity=position_qty,
            price=price
        )

        # Validar
        is_valid, error = reduce_only.validate()
        if not is_valid:
            reduce_only.reject(error or "Validacion fallida")
            logger.warning(f"Reduce-only {reduce_only.id} rechazada: {error}")
            return self._create_advanced_result(reduce_only, False, error or "Validacion fallida")

        # Ajustar cantidad si excede posicion
        reduce_only.adjust_to_position()

        market = self._market_states.get(symbol)
        if not market:
            return self._create_advanced_result(reduce_only, False, "No hay datos de mercado")

        exec_price = market.ask_price if side == OrderSide.BUY else market.bid_price
        exec_qty = reduce_only.quantity

        reduce_only.execute(exec_price, exec_qty)
        commission = self.config.get_commission(exec_qty * exec_price, is_maker=False)

        # Actualizar posicion
        if side == OrderSide.SELL:
            self._positions[symbol] = max(0, self._positions.get(symbol, 0) - exec_qty)
        else:
            self._positions[symbol] = min(0, self._positions.get(symbol, 0) + exec_qty)

        logger.info(f"Reduce-only {reduce_only.id} ejecutada: {exec_qty} @ {exec_price}")
        return self._create_advanced_result(
            reduce_only, True, "Reduce-only ejecutada",
            executed_price=exec_price,
            executed_quantity=exec_qty,
            commission=commission
        )

    # =========================================================================
    # Post-Only Orders
    # =========================================================================

    async def submit_post_only_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        price: float
    ) -> OrderResult:
        """
        Envia una orden post-only (solo maker).

        Se cancela si cruzaria el spread inmediatamente.

        Args:
            symbol: Par de trading
            side: BUY o SELL
            quantity: Cantidad
            price: Precio limite

        Returns:
            Resultado de la orden
        """
        from .orders.enums import OrderSide as AdvOrderSide

        post_only = PostOnlyOrder(
            symbol=symbol,
            side=AdvOrderSide.BUY if side == OrderSide.BUY else AdvOrderSide.SELL,
            quantity=quantity,
            price=price
        )

        market = self._market_states.get(symbol)
        if not market:
            return self._create_advanced_result(post_only, False, "No hay datos de mercado")

        best_bid = market.bid_price
        best_ask = market.ask_price

        # Verificar si cruzaria
        if post_only.would_cross_spread(best_bid, best_ask):
            post_only.submit(best_bid, best_ask)  # Esto cancelara la orden
            logger.info(f"Post-only {post_only.id} cancelada - cruzaria spread")
            return self._create_advanced_result(
                post_only, False, "Post-only cancelada - cruzaria spread"
            )

        # Agregar a ordenes pendientes
        post_only.submit(best_bid, best_ask)
        self._post_only_orders[post_only.id] = post_only

        logger.info(f"Post-only {post_only.id} en espera @ {price}")
        return self._create_advanced_result(
            post_only, True, "Post-only en espera",
            executed_price=0, executed_quantity=0
        )

    # =========================================================================
    # Scale In/Out Orders
    # =========================================================================

    async def create_scale_order(
        self,
        symbol: str,
        side: OrderSide,
        total_quantity: float,
        levels: List[float],
        quantities: List[float],
        order_type: str = "scale_in"
    ) -> ScaleOrder:
        """
        Crea una orden de scaling (scale-in o scale-out).

        Args:
            symbol: Par de trading
            side: BUY o SELL
            total_quantity: Cantidad total
            levels: Lista de niveles de precio
            quantities: Lista de porcentajes por nivel (deben sumar 1.0)
            order_type: "scale_in" o "scale_out"

        Returns:
            Orden scale creada
        """
        from .orders.enums import OrderSide as AdvOrderSide, OrderType as AdvOrderType

        scale = ScaleOrder(
            symbol=symbol,
            side=AdvOrderSide.BUY if side == OrderSide.BUY else AdvOrderSide.SELL,
            total_quantity=total_quantity,
            levels=levels,
            quantities=quantities,
            order_type=AdvOrderType.SCALE_IN if order_type == "scale_in" else AdvOrderType.SCALE_OUT
        )

        self._scale_orders[scale.id] = scale

        logger.info(
            f"Scale {scale.id} creado: {side.value} {total_quantity} en "
            f"{len(levels)} niveles"
        )

        return scale

    # =========================================================================
    # If-Touched Orders
    # =========================================================================

    async def create_if_touched_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        trigger_price: float,
        trigger_direction: str,
        order_price: Optional[float] = None
    ) -> IfTouchedOrder:
        """
        Crea una orden if-touched.

        Se activa cuando el precio toca el nivel de trigger.

        Args:
            symbol: Par de trading
            side: BUY o SELL
            quantity: Cantidad
            trigger_price: Precio de activacion
            trigger_direction: "above" o "below"
            order_price: Precio limite despues de trigger (None = market)

        Returns:
            Orden if-touched creada
        """
        from .orders.enums import OrderSide as AdvOrderSide, TriggerDirection

        if_touched = IfTouchedOrder(
            symbol=symbol,
            side=AdvOrderSide.BUY if side == OrderSide.BUY else AdvOrderSide.SELL,
            quantity=quantity,
            trigger_price=trigger_price,
            trigger_direction=TriggerDirection.ABOVE if trigger_direction == "above" else TriggerDirection.BELOW,
            order_price=order_price
        )

        self._if_touched_orders[if_touched.id] = if_touched

        logger.info(
            f"If-touched {if_touched.id} creado: trigger @ {trigger_price} "
            f"({trigger_direction})"
        )

        return if_touched

    # =========================================================================
    # OCO Orders
    # =========================================================================

    async def create_oco_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        stop_price: float,
        limit_price: float,
        stop_limit_price: Optional[float] = None
    ) -> OCOOrder:
        """
        Crea una orden OCO (One-Cancels-Other).

        Dos ordenes vinculadas: cuando una ejecuta, la otra se cancela.

        Args:
            symbol: Par de trading
            side: BUY o SELL
            quantity: Cantidad
            stop_price: Precio del stop
            limit_price: Precio del limit (take profit)
            stop_limit_price: Precio limite del stop (opcional)

        Returns:
            Orden OCO creada
        """
        from .orders.enums import OrderSide as AdvOrderSide

        oco = OCOOrder(
            symbol=symbol,
            side=AdvOrderSide.BUY if side == OrderSide.BUY else AdvOrderSide.SELL,
            quantity=quantity,
            stop_price=stop_price,
            limit_price=limit_price,
            stop_limit_price=stop_limit_price
        )

        self._oco_orders[oco.id] = oco
        self._composite_simulator.add_oco(oco)

        logger.info(
            f"OCO {oco.id} creado: SL @ {stop_price}, TP @ {limit_price}"
        )

        return oco

    # =========================================================================
    # OTOCO Orders
    # =========================================================================

    async def create_otoco_order(
        self,
        symbol: str,
        entry_side: OrderSide,
        entry_quantity: float,
        entry_price: Optional[float],
        stop_loss_price: float,
        take_profit_price: float
    ) -> OTOCOOrder:
        """
        Crea una orden OTOCO (One-Triggers-OCO).

        Una orden de entrada que al ejecutarse activa un OCO (SL + TP).

        Args:
            symbol: Par de trading
            entry_side: BUY o SELL
            entry_quantity: Cantidad de entrada
            entry_price: Precio de entrada (None = market)
            stop_loss_price: Precio de stop loss
            take_profit_price: Precio de take profit

        Returns:
            Orden OTOCO creada
        """
        from .orders.enums import OrderSide as AdvOrderSide

        otoco = OTOCOOrder(
            symbol=symbol,
            entry_side=AdvOrderSide.BUY if entry_side == OrderSide.BUY else AdvOrderSide.SELL,
            entry_quantity=entry_quantity,
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price
        )

        self._otoco_orders[otoco.id] = otoco

        logger.info(
            f"OTOCO {otoco.id} creado: entry @ {entry_price}, "
            f"SL @ {stop_loss_price}, TP @ {take_profit_price}"
        )

        return otoco

    # =========================================================================
    # Price Update Processing
    # =========================================================================

    async def process_price_update(
        self,
        symbol: str,
        price: float,
        timestamp: Optional[datetime] = None,
        bid: Optional[float] = None,
        ask: Optional[float] = None,
        volume: Optional[float] = None
    ) -> List[OrderResult]:
        """
        Procesa actualizacion de precio para todas las ordenes activas.

        Verifica trailing stops, brackets, algoritmos, condicionales, etc.

        Args:
            symbol: Par de trading
            price: Precio actual
            timestamp: Momento de la actualizacion
            bid: Precio bid
            ask: Precio ask
            volume: Volumen del tick

        Returns:
            Lista de resultados de ordenes ejecutadas
        """
        if timestamp is None:
            timestamp = datetime.now()

        self.update_market_state(symbol, price)
        market = self._market_states[symbol]

        results: List[OrderResult] = []

        # 1. Procesar ordenes pendientes basicas
        await self.check_pending_orders(symbol, price)

        # 2. Procesar trailing stops
        trailing_actions = self._trailing_simulator.process_price_update(
            symbol, price, timestamp
        )
        for order_id, action in trailing_actions:
            if action.action_type == "execute":
                order = self._trailing_orders.get(order_id)
                if order:
                    result = await self._execute_trailing_stop(order, market)
                    results.append(result)

        # 3. Procesar brackets y OCOs
        composite_actions = self._composite_simulator.process_price_update(
            symbol, price, timestamp, bid, ask
        )
        for action in composite_actions:
            if action.action_type == "execute":
                result = self._create_action_result(action, market)
                results.append(result)

        # 4. Procesar algoritmos (TWAP, VWAP, Iceberg)
        algo_executions = self._algo_simulator.process_tick(
            symbol, price, timestamp, market_volume=volume or 0
        )
        for order_id, qty, exec_price in algo_executions:
            commission = self.config.get_commission(qty * exec_price, is_maker=False)
            result = OrderResult(
                order=self._create_dummy_order(symbol, qty),
                success=True,
                message=f"Algoritmo {order_id} slice ejecutado",
                executed_price=exec_price,
                executed_quantity=qty,
                slippage=0,
                commission=commission
            )
            results.append(result)

        # 5. Procesar if-touched
        for order_id, if_touched in list(self._if_touched_orders.items()):
            if if_touched.symbol != symbol:
                continue
            if if_touched.check_trigger(price):
                if_touched.activate(price, timestamp)
                # Ejecutar como market o limit
                exec_result = await self._execute_if_touched(if_touched, market)
                results.append(exec_result)
                del self._if_touched_orders[order_id]

        # 6. Procesar scale orders
        for order_id, scale in list(self._scale_orders.items()):
            if scale.symbol != symbol:
                continue
            triggered_level = scale.check_level_trigger(price)
            if triggered_level is not None:
                qty = scale.get_quantity_for_level(triggered_level)
                scale.execute_level(triggered_level, price)
                commission = self.config.get_commission(qty * price, is_maker=False)
                result = OrderResult(
                    order=self._create_dummy_order(symbol, qty),
                    success=True,
                    message=f"Scale nivel {triggered_level} ejecutado",
                    executed_price=price,
                    executed_quantity=qty,
                    slippage=0,
                    commission=commission
                )
                results.append(result)

                if scale.is_complete():
                    del self._scale_orders[order_id]

        # 7. Procesar OTOCO
        for order_id, otoco in list(self._otoco_orders.items()):
            if otoco.symbol != symbol:
                continue

            if not otoco.entry_filled:
                # Verificar entry
                if otoco.check_entry_trigger(price, bid, ask):
                    fill_price = ask if otoco.entry_side.value == "buy" else bid
                    fill_price = fill_price or price
                    otoco.execute_entry(fill_price)
                    logger.info(f"OTOCO {order_id} entry ejecutado @ {fill_price}")
            else:
                # Verificar exit
                exit_type = otoco.check_exit_triggers(price)
                if exit_type:
                    otoco.execute_exit(exit_type, price)
                    commission = self.config.get_commission(
                        otoco.entry_fill_quantity * price, is_maker=False
                    )
                    result = OrderResult(
                        order=self._create_dummy_order(symbol, otoco.entry_fill_quantity),
                        success=True,
                        message=f"OTOCO {exit_type} ejecutado",
                        executed_price=price,
                        executed_quantity=otoco.entry_fill_quantity,
                        slippage=0,
                        commission=commission
                    )
                    results.append(result)
                    del self._otoco_orders[order_id]

        return results

    # =========================================================================
    # Helper Methods
    # =========================================================================

    async def _execute_trailing_stop(
        self,
        order: TrailingStopOrder,
        market: MarketState
    ) -> OrderResult:
        """Ejecuta un trailing stop como market order"""
        exec_price = market.bid_price if order.side.value == "sell" else market.ask_price
        slippage = self._calculate_slippage(
            self._create_dummy_order(order.symbol, order.quantity),
            market
        )

        if order.side.value == "sell":
            exec_price *= (1 - slippage)
        else:
            exec_price *= (1 + slippage)

        commission = self.config.get_commission(order.quantity * exec_price, is_maker=False)

        logger.info(
            f"Trailing stop {order.id} ejecutado @ {exec_price:.2f} "
            f"(stop was {order.get_current_stop_price():.2f})"
        )

        return OrderResult(
            order=self._create_dummy_order(order.symbol, order.quantity),
            success=True,
            message="Trailing stop ejecutado",
            executed_price=exec_price,
            executed_quantity=order.quantity,
            slippage=slippage,
            commission=commission
        )

    async def _execute_if_touched(
        self,
        order: IfTouchedOrder,
        market: MarketState
    ) -> OrderResult:
        """Ejecuta una orden if-touched"""
        if order.order_price:
            # Ejecutar como limit
            exec_price = order.order_price
        else:
            # Ejecutar como market
            exec_price = market.ask_price if order.side.value == "buy" else market.bid_price

        order.execute(exec_price, order.quantity)
        commission = self.config.get_commission(order.quantity * exec_price, is_maker=False)

        return OrderResult(
            order=self._create_dummy_order(order.symbol, order.quantity),
            success=True,
            message="If-touched ejecutada",
            executed_price=exec_price,
            executed_quantity=order.quantity,
            slippage=0,
            commission=commission
        )

    def _create_dummy_order(self, symbol: str, quantity: float) -> Order:
        """Crea una orden dummy para resultados"""
        return Order(
            symbol=symbol,
            side=OrderSide.BUY,
            type=OrderType.MARKET,
            quantity=quantity
        )

    def _create_advanced_result(
        self,
        order: Any,
        success: bool,
        message: str,
        executed_price: float = 0,
        executed_quantity: float = 0,
        commission: float = 0
    ) -> OrderResult:
        """Crea un resultado de orden para ordenes avanzadas"""
        return OrderResult(
            order=self._create_dummy_order(order.symbol, order.quantity),
            success=success,
            message=message,
            executed_price=executed_price,
            executed_quantity=executed_quantity,
            slippage=0,
            commission=commission
        )

    def _create_action_result(
        self,
        action: Any,
        market: MarketState
    ) -> OrderResult:
        """Crea un resultado de orden desde una accion"""
        return OrderResult(
            order=self._create_dummy_order("", 0),
            success=True,
            message=action.message if hasattr(action, 'message') else "Accion ejecutada",
            executed_price=action.execution_price if hasattr(action, 'execution_price') else market.current_price,
            executed_quantity=0,
            slippage=0,
            commission=0
        )

    # =========================================================================
    # Advanced Order Management
    # =========================================================================

    def cancel_advanced_order(self, order_id: str) -> bool:
        """
        Cancela una orden avanzada por ID.

        Args:
            order_id: ID de la orden a cancelar

        Returns:
            True si se cancelo, False si no se encontro
        """
        # Check each advanced order type
        if order_id in self._trailing_orders:
            self._trailing_simulator.cancel_order(order_id)
            del self._trailing_orders[order_id]
            return True

        if order_id in self._bracket_orders:
            self._composite_simulator.cancel_order(order_id)
            del self._bracket_orders[order_id]
            return True

        if order_id in self._oco_orders:
            self._composite_simulator.cancel_order(order_id)
            del self._oco_orders[order_id]
            return True

        if order_id in self._scale_orders:
            del self._scale_orders[order_id]
            return True

        if order_id in self._if_touched_orders:
            del self._if_touched_orders[order_id]
            return True

        if order_id in self._otoco_orders:
            del self._otoco_orders[order_id]
            return True

        # Algo orders
        if self._algo_simulator.cancel_order(order_id):
            return True

        return False

    def get_advanced_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        """
        Obtiene el estado de una orden avanzada.

        Args:
            order_id: ID de la orden

        Returns:
            Diccionario con estado o None si no existe
        """
        if order_id in self._trailing_orders:
            order = self._trailing_orders[order_id]
            return {
                "type": "trailing_stop",
                "id": order.id,
                "status": order.status.value,
                "current_stop": order.get_current_stop_price(),
            }

        if order_id in self._bracket_orders:
            order = self._bracket_orders[order_id]
            return order.to_dict()

        if order_id in self._oco_orders:
            order = self._oco_orders[order_id]
            return order.to_dict()

        if order_id in self._scale_orders:
            order = self._scale_orders[order_id]
            return order.to_dict()

        if order_id in self._otoco_orders:
            order = self._otoco_orders[order_id]
            return order.to_dict()

        # Check algo simulator
        return self._algo_simulator.get_order_status(order_id)

    def get_all_advanced_orders(self, symbol: Optional[str] = None) -> Dict[str, List[Any]]:
        """
        Obtiene todas las ordenes avanzadas activas.

        Args:
            symbol: Filtrar por simbolo (opcional)

        Returns:
            Diccionario con listas de ordenes por tipo
        """
        def filter_by_symbol(orders: Dict) -> List:
            if symbol is None:
                return list(orders.values())
            return [o for o in orders.values() if o.symbol == symbol]

        return {
            "trailing_stops": filter_by_symbol(self._trailing_orders),
            "brackets": filter_by_symbol(self._bracket_orders),
            "ocos": filter_by_symbol(self._oco_orders),
            "otocos": filter_by_symbol(self._otoco_orders),
            "scales": filter_by_symbol(self._scale_orders),
            "if_touched": filter_by_symbol(self._if_touched_orders),
        }

    def get_statistics(self) -> Dict[str, Any]:
        """
        Obtiene estadisticas del simulador.

        Returns:
            Diccionario con estadisticas
        """
        trailing_stats = self._trailing_simulator.get_statistics()
        algo_stats = self._algo_simulator.get_statistics()
        composite_stats = self._composite_simulator.get_statistics()

        return {
            "pending_orders": len(self.pending_orders),
            "order_history": len(self.order_history),
            "trailing_stops": {
                "active": len(self._trailing_orders),
                **trailing_stats
            },
            "brackets": {
                "active": len(self._bracket_orders),
                **composite_stats
            },
            "algo_orders": algo_stats,
            "oco_orders": len(self._oco_orders),
            "otoco_orders": len(self._otoco_orders),
            "scale_orders": len(self._scale_orders),
            "if_touched_orders": len(self._if_touched_orders),
        }
