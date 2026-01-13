"""
Simulador de Ordenes Algoritmicas.

Maneja la simulacion de ordenes de ejecucion algoritmica:
- TWAP (Time-Weighted Average Price)
- VWAP (Volume-Weighted Average Price)
- Iceberg
- Hidden
"""

from __future__ import annotations

import logging
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from ..orders.enums import OrderSide, OrderStatus
from ..orders.base import AlgoOrderState, OrderAction


logger = logging.getLogger(__name__)


@dataclass
class TWAPConfig:
    """Configuracion para orden TWAP"""
    total_quantity: float
    duration_seconds: int
    slice_count: Optional[int] = None
    size_variation: float = 0.0  # 0.1 = 10% variacion
    min_slice_size: float = 0.0

    def __post_init__(self):
        if self.slice_count is None:
            # Default: 1 slice por minuto
            self.slice_count = max(1, self.duration_seconds // 60)


@dataclass
class VWAPConfig:
    """Configuracion para orden VWAP"""
    total_quantity: float
    duration_seconds: int
    max_participation: float = 0.05  # Max 5% del volumen
    volume_profile: Optional[Dict[int, float]] = None  # hora -> % volumen

    def get_volume_weight(self, hour: int) -> float:
        """Obtiene peso de volumen para una hora"""
        if self.volume_profile:
            return self.volume_profile.get(hour, 1.0 / 24)
        # Perfil por defecto: mas volumen en apertura y cierre
        default_profile = {
            9: 0.08, 10: 0.07, 11: 0.05, 12: 0.04,
            13: 0.04, 14: 0.05, 15: 0.06, 16: 0.08,
            # Crypto (24h): distribucion mas uniforme
            0: 0.04, 1: 0.03, 2: 0.03, 3: 0.03,
            4: 0.03, 5: 0.04, 6: 0.04, 7: 0.05,
            8: 0.06, 17: 0.05, 18: 0.04, 19: 0.04,
            20: 0.04, 21: 0.04, 22: 0.04, 23: 0.04,
        }
        return default_profile.get(hour, 0.04)


@dataclass
class IcebergConfig:
    """Configuracion para orden Iceberg"""
    total_quantity: float
    display_quantity: float
    price: float
    price_variation: float = 0.0  # Variacion de precio entre recargas


@dataclass
class AlgoOrder:
    """
    Representa una orden algoritmica activa.

    Attributes:
        id: ID unico de la orden
        symbol: Par de trading
        side: BUY o SELL
        algo_type: Tipo de algoritmo (twap, vwap, iceberg, hidden)
        state: Estado actual de ejecucion
        config: Configuracion del algoritmo
    """
    id: str
    symbol: str
    side: OrderSide
    algo_type: str
    state: AlgoOrderState
    config: TWAPConfig | VWAPConfig | IcebergConfig
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)


class AlgoOrderSimulator:
    """
    Simulador de ordenes algoritmicas.

    Maneja la ejecucion simulada de TWAP, VWAP, Iceberg y Hidden orders.

    Example:
        simulator = AlgoOrderSimulator()

        # Crear orden TWAP
        config = TWAPConfig(total_quantity=10.0, duration_seconds=3600)
        order_id = simulator.create_twap_order("BTC/USDT", OrderSide.BUY, config)

        # Procesar tick
        slices = simulator.process_tick("BTC/USDT", current_price=50000, volume=1000)
    """

    def __init__(self):
        self.active_orders: Dict[str, AlgoOrder] = {}
        self.completed_orders: List[AlgoOrder] = []
        self._order_counter: int = 0

    def _generate_id(self) -> str:
        """Genera ID unico para orden"""
        self._order_counter += 1
        return f"algo_{self._order_counter:06d}"

    def create_twap_order(
        self,
        symbol: str,
        side: OrderSide,
        config: TWAPConfig,
        start_time: Optional[datetime] = None
    ) -> str:
        """
        Crea una orden TWAP.

        Args:
            symbol: Par de trading
            side: BUY o SELL
            config: Configuracion TWAP
            start_time: Tiempo de inicio (default: ahora)

        Returns:
            ID de la orden creada
        """
        order_id = self._generate_id()
        start = start_time or datetime.now()

        slice_interval = config.duration_seconds / config.slice_count
        slice_size = config.total_quantity / config.slice_count

        state = AlgoOrderState(
            order_id=order_id,
            total_quantity=config.total_quantity,
            slices_total=config.slice_count,
            start_time=start,
            end_time=start + timedelta(seconds=config.duration_seconds),
            next_execution_time=start,
        )

        order = AlgoOrder(
            id=order_id,
            symbol=symbol,
            side=side,
            algo_type="twap",
            state=state,
            config=config,
            status=OrderStatus.SUBMITTED,
        )

        self.active_orders[order_id] = order
        logger.info(
            f"TWAP order {order_id} created: {config.total_quantity} {symbol} "
            f"over {config.duration_seconds}s in {config.slice_count} slices"
        )
        return order_id

    def create_vwap_order(
        self,
        symbol: str,
        side: OrderSide,
        config: VWAPConfig,
        start_time: Optional[datetime] = None
    ) -> str:
        """
        Crea una orden VWAP.

        Args:
            symbol: Par de trading
            side: BUY o SELL
            config: Configuracion VWAP
            start_time: Tiempo de inicio

        Returns:
            ID de la orden creada
        """
        order_id = self._generate_id()
        start = start_time or datetime.now()

        state = AlgoOrderState(
            order_id=order_id,
            total_quantity=config.total_quantity,
            start_time=start,
            end_time=start + timedelta(seconds=config.duration_seconds),
            next_execution_time=start,
        )

        order = AlgoOrder(
            id=order_id,
            symbol=symbol,
            side=side,
            algo_type="vwap",
            state=state,
            config=config,
            status=OrderStatus.SUBMITTED,
        )

        self.active_orders[order_id] = order
        logger.info(
            f"VWAP order {order_id} created: {config.total_quantity} {symbol} "
            f"over {config.duration_seconds}s, max participation {config.max_participation*100}%"
        )
        return order_id

    def create_iceberg_order(
        self,
        symbol: str,
        side: OrderSide,
        config: IcebergConfig,
    ) -> str:
        """
        Crea una orden Iceberg.

        Args:
            symbol: Par de trading
            side: BUY o SELL
            config: Configuracion Iceberg

        Returns:
            ID de la orden creada
        """
        order_id = self._generate_id()

        state = AlgoOrderState(
            order_id=order_id,
            total_quantity=config.total_quantity,
            start_time=datetime.now(),
        )

        order = AlgoOrder(
            id=order_id,
            symbol=symbol,
            side=side,
            algo_type="iceberg",
            state=state,
            config=config,
            status=OrderStatus.SUBMITTED,
        )

        self.active_orders[order_id] = order
        logger.info(
            f"Iceberg order {order_id} created: {config.total_quantity} {symbol} "
            f"showing {config.display_quantity} at {config.price}"
        )
        return order_id

    def create_hidden_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        price: float,
    ) -> str:
        """
        Crea una orden Hidden.

        Args:
            symbol: Par de trading
            side: BUY o SELL
            quantity: Cantidad total
            price: Precio de la orden

        Returns:
            ID de la orden creada
        """
        order_id = self._generate_id()

        config = IcebergConfig(
            total_quantity=quantity,
            display_quantity=0,  # Completamente oculta
            price=price,
        )

        state = AlgoOrderState(
            order_id=order_id,
            total_quantity=quantity,
            start_time=datetime.now(),
        )

        order = AlgoOrder(
            id=order_id,
            symbol=symbol,
            side=side,
            algo_type="hidden",
            state=state,
            config=config,
            status=OrderStatus.SUBMITTED,
        )

        self.active_orders[order_id] = order
        logger.info(f"Hidden order {order_id} created: {quantity} {symbol} at {price}")
        return order_id

    def process_tick(
        self,
        symbol: str,
        current_price: float,
        timestamp: Optional[datetime] = None,
        market_volume: float = 0.0,
        bid: Optional[float] = None,
        ask: Optional[float] = None,
    ) -> List[Tuple[str, float, float]]:
        """
        Procesa un tick de mercado.

        Args:
            symbol: Par de trading
            current_price: Precio actual
            timestamp: Momento del tick
            market_volume: Volumen de mercado en el periodo
            bid: Precio bid
            ask: Precio ask

        Returns:
            Lista de (order_id, executed_quantity, executed_price) para slices ejecutados
        """
        if timestamp is None:
            timestamp = datetime.now()

        executions: List[Tuple[str, float, float]] = []
        completed_orders: List[str] = []

        for order_id, order in self.active_orders.items():
            if order.symbol != symbol:
                continue

            execution = None

            if order.algo_type == "twap":
                execution = self._process_twap_tick(order, current_price, timestamp)
            elif order.algo_type == "vwap":
                execution = self._process_vwap_tick(
                    order, current_price, timestamp, market_volume
                )
            elif order.algo_type == "iceberg":
                execution = self._process_iceberg_tick(order, current_price, bid, ask)
            elif order.algo_type == "hidden":
                execution = self._process_hidden_tick(order, current_price, bid, ask)

            if execution:
                qty, price = execution
                executions.append((order_id, qty, price))
                order.state.record_fill(qty, price)

                if order.state.is_complete():
                    completed_orders.append(order_id)
                    order.status = OrderStatus.FILLED

        # Mover ordenes completadas
        for order_id in completed_orders:
            order = self.active_orders.pop(order_id)
            self.completed_orders.append(order)
            logger.info(
                f"Algo order {order_id} completed: avg price {order.state.average_price:.2f}"
            )

        return executions

    def _process_twap_tick(
        self,
        order: AlgoOrder,
        current_price: float,
        timestamp: datetime
    ) -> Optional[Tuple[float, float]]:
        """Procesa tick para orden TWAP"""
        config: TWAPConfig = order.config
        state = order.state

        # Verificar si es momento de ejecutar
        if state.next_execution_time and timestamp < state.next_execution_time:
            return None

        # Verificar si ya terminamos
        if state.is_complete():
            return None

        # Calcular cantidad para este slice
        base_slice = config.total_quantity / config.slice_count

        # Aplicar variacion aleatoria
        if config.size_variation > 0:
            variation = random.uniform(-config.size_variation, config.size_variation)
            slice_qty = base_slice * (1 + variation)
        else:
            slice_qty = base_slice

        # No exceder cantidad restante
        slice_qty = min(slice_qty, state.remaining_quantity)

        if slice_qty < config.min_slice_size:
            return None

        # Calcular siguiente ejecucion
        interval = config.duration_seconds / config.slice_count
        state.next_execution_time = timestamp + timedelta(seconds=interval)

        # Precio con pequeno slippage simulado
        slippage = random.uniform(-0.0001, 0.0003)  # -0.01% a +0.03%
        exec_price = current_price * (1 + slippage)

        logger.debug(
            f"TWAP {order.id} slice {state.slices_executed + 1}/{config.slice_count}: "
            f"{slice_qty:.4f} @ {exec_price:.2f}"
        )

        return (slice_qty, exec_price)

    def _process_vwap_tick(
        self,
        order: AlgoOrder,
        current_price: float,
        timestamp: datetime,
        market_volume: float
    ) -> Optional[Tuple[float, float]]:
        """Procesa tick para orden VWAP"""
        config: VWAPConfig = order.config
        state = order.state

        if state.is_complete():
            return None

        # Obtener peso de volumen para la hora actual
        hour = timestamp.hour
        volume_weight = config.get_volume_weight(hour)

        # Calcular cantidad objetivo basada en volumen
        target_qty = state.remaining_quantity * volume_weight

        # Limitar por participacion maxima
        if market_volume > 0:
            max_qty = market_volume * config.max_participation
            target_qty = min(target_qty, max_qty)

        if target_qty <= 0:
            return None

        # No exceder cantidad restante
        slice_qty = min(target_qty, state.remaining_quantity)

        # Precio VWAP simulado (cercano al precio actual)
        slippage = random.uniform(-0.0002, 0.0002)
        exec_price = current_price * (1 + slippage)

        logger.debug(
            f"VWAP {order.id} execution: {slice_qty:.4f} @ {exec_price:.2f} "
            f"(volume weight: {volume_weight:.2%})"
        )

        return (slice_qty, exec_price)

    def _process_iceberg_tick(
        self,
        order: AlgoOrder,
        current_price: float,
        bid: Optional[float],
        ask: Optional[float]
    ) -> Optional[Tuple[float, float]]:
        """Procesa tick para orden Iceberg"""
        config: IcebergConfig = order.config
        state = order.state

        if state.is_complete():
            return None

        # Verificar si el precio permite ejecucion
        can_execute = False

        if order.side == OrderSide.BUY:
            # Compra: ejecuta si ask <= precio limite
            check_price = ask if ask else current_price
            can_execute = check_price <= config.price
        else:
            # Venta: ejecuta si bid >= precio limite
            check_price = bid if bid else current_price
            can_execute = check_price >= config.price

        if not can_execute:
            return None

        # Ejecutar cantidad visible
        slice_qty = min(config.display_quantity, state.remaining_quantity)

        # Aplicar variacion de precio si configurada
        if config.price_variation > 0:
            variation = random.uniform(0, config.price_variation)
            exec_price = config.price * (1 - variation if order.side == OrderSide.BUY else 1 + variation)
        else:
            exec_price = config.price

        logger.debug(
            f"Iceberg {order.id} slice: {slice_qty:.4f} @ {exec_price:.2f} "
            f"(remaining: {state.remaining_quantity - slice_qty:.4f})"
        )

        return (slice_qty, exec_price)

    def _process_hidden_tick(
        self,
        order: AlgoOrder,
        current_price: float,
        bid: Optional[float],
        ask: Optional[float]
    ) -> Optional[Tuple[float, float]]:
        """Procesa tick para orden Hidden"""
        # Hidden es como iceberg pero ejecuta todo cuando cruzan
        config: IcebergConfig = order.config
        state = order.state

        if state.is_complete():
            return None

        # Verificar si alguien cruza nuestro precio
        crossed = False

        if order.side == OrderSide.BUY:
            check_price = ask if ask else current_price
            crossed = check_price <= config.price
        else:
            check_price = bid if bid else current_price
            crossed = check_price >= config.price

        if not crossed:
            return None

        # Ejecutar toda la cantidad restante
        slice_qty = state.remaining_quantity
        exec_price = config.price

        logger.debug(f"Hidden {order.id} executed: {slice_qty:.4f} @ {exec_price:.2f}")

        return (slice_qty, exec_price)

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancela una orden algoritmica.

        Returns:
            True si se cancelo, False si no existia
        """
        order = self.active_orders.pop(order_id, None)
        if order:
            order.status = OrderStatus.CANCELLED
            self.completed_orders.append(order)
            logger.info(
                f"Algo order {order_id} cancelled: "
                f"filled {order.state.filled_quantity}/{order.state.total_quantity}"
            )
            return True
        return False

    def get_order_status(self, order_id: str) -> Optional[Dict]:
        """
        Obtiene estado de una orden.

        Returns:
            Diccionario con estado o None si no existe
        """
        order = self.active_orders.get(order_id)
        if not order:
            # Buscar en completadas
            for o in self.completed_orders:
                if o.id == order_id:
                    order = o
                    break

        if not order:
            return None

        return {
            "id": order.id,
            "symbol": order.symbol,
            "side": order.side.value,
            "algo_type": order.algo_type,
            "status": order.status.value,
            "total_quantity": order.state.total_quantity,
            "filled_quantity": order.state.filled_quantity,
            "remaining_quantity": order.state.remaining_quantity,
            "average_price": order.state.average_price,
            "progress_pct": order.state.get_progress_pct(),
            "slices_executed": order.state.slices_executed,
            "created_at": order.created_at.isoformat(),
        }

    def get_statistics(self) -> Dict:
        """Obtiene estadisticas del simulador"""
        return {
            "active_orders": len(self.active_orders),
            "completed_orders": len(self.completed_orders),
            "by_type": {
                "twap": len([o for o in self.active_orders.values() if o.algo_type == "twap"]),
                "vwap": len([o for o in self.active_orders.values() if o.algo_type == "vwap"]),
                "iceberg": len([o for o in self.active_orders.values() if o.algo_type == "iceberg"]),
                "hidden": len([o for o in self.active_orders.values() if o.algo_type == "hidden"]),
            }
        }
