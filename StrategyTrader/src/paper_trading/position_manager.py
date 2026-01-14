"""
Position Manager

Gestiona posiciones abiertas en paper trading.
Tracking en tiempo real de PnL, stop loss y take profit.
"""

import logging
import threading
from contextlib import contextmanager
from datetime import datetime
from typing import Dict, List, Optional, Callable, Generator
from dataclasses import dataclass

from .models import (
    PaperPosition,
    PositionSide,
    TradeRecord,
    Order,
    OrderResult,
    OrderSide,
    OrderType,
)
from .config import PaperTradingConfig
from .order_simulator import OrderSimulator


logger = logging.getLogger(__name__)


@dataclass
class PositionSummary:
    """Resumen de todas las posiciones"""
    total_positions: int
    long_positions: int
    short_positions: int
    total_exposure: float
    unrealized_pnl: float
    realized_pnl: float
    margin_used: float


class PositionManager:
    """
    Gestor de posiciones para Paper Trading.

    Responsabilidades:
    - Abrir y cerrar posiciones
    - Tracking de PnL no realizado
    - Gestion de Stop Loss y Take Profit
    - Historial de trades cerrados

    Example:
        manager = PositionManager(config, order_simulator)

        # Abrir posicion
        position = manager.open_position(
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            quantity=0.1,
            entry_price=50000,
            stop_loss=49000,
            take_profit=52000
        )

        # Actualizar precios
        manager.update_prices({"BTC/USDT": 51000})

        # Obtener PnL
        pnl = manager.get_unrealized_pnl()

    Attributes:
        config: Configuracion de paper trading
        positions: Posiciones abiertas
        trade_history: Historial de trades cerrados
    """

    def __init__(
        self,
        config: PaperTradingConfig,
        order_simulator: Optional[OrderSimulator] = None
    ):
        """
        Inicializa el gestor de posiciones.

        Args:
            config: Configuracion de paper trading
            order_simulator: Simulador de ordenes (opcional)
        """
        self.config = config
        self.order_simulator = order_simulator

        # Lock para thread safety (RLock permite llamadas reentrantes)
        self._lock = threading.RLock()

        self._balance = config.initial_balance
        self._positions: Dict[str, PaperPosition] = {}
        self._trade_history: List[TradeRecord] = []
        self._current_prices: Dict[str, float] = {}

        # Callbacks
        self.on_position_opened: Optional[Callable[[PaperPosition], None]] = None
        self.on_position_closed: Optional[Callable[[TradeRecord], None]] = None
        self.on_stop_loss_triggered: Optional[Callable[[PaperPosition], None]] = None
        self.on_take_profit_triggered: Optional[Callable[[PaperPosition], None]] = None
        self.on_margin_call: Optional[Callable[[float], None]] = None

    @contextmanager
    def _thread_safe(self) -> Generator[None, None, None]:
        """Context manager para operaciones thread-safe."""
        self._lock.acquire()
        try:
            yield
        finally:
            self._lock.release()

    @property
    def balance(self) -> float:
        """Balance disponible (thread-safe)"""
        with self._thread_safe():
            return self._balance

    @property
    def equity(self) -> float:
        """Equity total (balance + PnL no realizado) (thread-safe)"""
        with self._thread_safe():
            return self._balance + self._get_unrealized_pnl_unsafe()

    @property
    def positions(self) -> List[PaperPosition]:
        """Lista de posiciones abiertas (thread-safe)"""
        with self._thread_safe():
            return list(self._positions.values())

    @property
    def trade_history(self) -> List[TradeRecord]:
        """Historial de trades cerrados (thread-safe)"""
        with self._thread_safe():
            return self._trade_history.copy()

    def open_position(
        self,
        symbol: str,
        side: PositionSide,
        quantity: float,
        entry_price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Optional[PaperPosition]:
        """
        Abre una nueva posicion (thread-safe).

        Args:
            symbol: Par de trading
            side: Tipo de posicion (LONG/SHORT)
            quantity: Cantidad a operar
            entry_price: Precio de entrada
            stop_loss: Precio de stop loss (opcional)
            take_profit: Precio de take profit (opcional)

        Returns:
            Posicion creada o None si no se pudo abrir
        """
        position = None
        callback_to_call = None

        with self._thread_safe():
            # Validar numero maximo de posiciones
            if len(self._positions) >= self.config.max_positions:
                logger.warning(
                    f"No se puede abrir posicion: limite de {self.config.max_positions} alcanzado"
                )
                return None

            # Validar balance
            position_value = quantity * entry_price
            if position_value > self._balance:
                logger.warning(
                    f"Balance insuficiente: {self._balance:.2f} < {position_value:.2f}"
                )
                return None

            # Validar tamano maximo
            max_position_value = self.config.get_max_position_value()
            if position_value > max_position_value:
                logger.warning(
                    f"Posicion excede maximo: {position_value:.2f} > {max_position_value:.2f}"
                )
                return None

            # Aplicar stop loss y take profit por defecto si no se especifican
            if stop_loss is None and self.config.use_stop_loss:
                if side == PositionSide.LONG:
                    stop_loss = entry_price * (1 - self.config.default_stop_loss_pct)
                else:
                    stop_loss = entry_price * (1 + self.config.default_stop_loss_pct)

            if take_profit is None and self.config.use_take_profit:
                if side == PositionSide.LONG:
                    take_profit = entry_price * (1 + self.config.default_take_profit_pct)
                else:
                    take_profit = entry_price * (1 - self.config.default_take_profit_pct)

            # Calcular comision de entrada
            commission = self.config.get_commission(position_value)

            # Crear posicion
            position = PaperPosition(
                symbol=symbol,
                side=side,
                entry_price=entry_price,
                quantity=quantity,
                stop_loss=stop_loss,
                take_profit=take_profit,
                commission_paid=commission
            )

            # Actualizar balance
            self._balance -= position_value + commission
            self._positions[position.id] = position
            self._current_prices[symbol] = entry_price

            callback_to_call = self.on_position_opened

        # Ejecutar callback fuera del lock para evitar deadlocks
        if callback_to_call and position:
            callback_to_call(position)

        logger.info(
            f"Posicion abierta: {position.id} - {side.value} {quantity} {symbol} @ {entry_price:.2f}"
        )

        return position

    def close_position(
        self,
        position_id: str,
        exit_price: float,
        exit_reason: str = "Manual"
    ) -> Optional[TradeRecord]:
        """
        Cierra una posicion existente (thread-safe).

        Args:
            position_id: ID de la posicion a cerrar
            exit_price: Precio de salida
            exit_reason: Razon del cierre

        Returns:
            Registro del trade o None si no existe
        """
        trade = None
        callback_to_call = None

        with self._thread_safe():
            if position_id not in self._positions:
                logger.warning(f"Posicion {position_id} no encontrada")
                return None

            position = self._positions[position_id]

            # Calcular PnL
            if position.side == PositionSide.LONG:
                gross_pnl = (exit_price - position.entry_price) * position.quantity
            else:
                gross_pnl = (position.entry_price - exit_price) * position.quantity

            # Calcular comision de salida
            exit_value = position.quantity * exit_price
            exit_commission = self.config.get_commission(exit_value)
            total_commission = position.commission_paid + exit_commission

            # PnL neto
            net_pnl = gross_pnl - total_commission

            # Calcular retorno porcentual con protección contra división por cero
            position_value = position.quantity * position.entry_price
            if position_value > 0:
                return_pct = (net_pnl / position_value) * 100
            else:
                return_pct = 0.0
                logger.warning(f"close_position: position_value es cero para {position_id}")

            # Crear registro de trade
            trade = TradeRecord(
                symbol=position.symbol,
                side=position.side,
                entry_price=position.entry_price,
                exit_price=exit_price,
                quantity=position.quantity,
                entry_time=position.entry_time,
                exit_time=datetime.now(),
                pnl=net_pnl,
                return_pct=return_pct,
                commission=total_commission,
                exit_reason=exit_reason
            )

            # Actualizar balance
            self._balance += exit_value - exit_commission

            # Remover posicion y guardar historial
            del self._positions[position_id]
            self._trade_history.append(trade)

            callback_to_call = self.on_position_closed

        # Ejecutar callback fuera del lock para evitar deadlocks
        if callback_to_call and trade:
            callback_to_call(trade)

        logger.info(
            f"Posicion cerrada: {position_id} - {exit_reason} @ {exit_price:.2f} "
            f"(PnL: {net_pnl:+.2f}, {return_pct:+.2f}%)"
        )

        return trade

    def update_prices(self, prices: Dict[str, float]):
        """
        Actualiza precios y verifica stop loss / take profit (thread-safe).

        Args:
            prices: Diccionario {symbol: precio}
        """
        positions_to_close: List[tuple] = []  # (position_id, exit_price, reason, callback)

        with self._thread_safe():
            self._current_prices.update(prices)

            for position in list(self._positions.values()):
                if position.symbol not in prices:
                    continue

                current_price = prices[position.symbol]

                # Actualizar PnL no realizado
                position.update_unrealized_pnl(current_price)

                # Verificar stop loss
                if position.should_stop_loss(current_price):
                    exit_price = position.stop_loss if position.stop_loss is not None else current_price
                    positions_to_close.append((
                        position.id,
                        exit_price,
                        "Stop Loss",
                        self.on_stop_loss_triggered,
                        position
                    ))
                    continue

                # Verificar take profit
                if position.should_take_profit(current_price):
                    exit_price = position.take_profit if position.take_profit is not None else current_price
                    positions_to_close.append((
                        position.id,
                        exit_price,
                        "Take Profit",
                        self.on_take_profit_triggered,
                        position
                    ))
                    continue

        # Ejecutar cierres fuera del lock principal (close_position tiene su propio lock)
        for pos_id, exit_price, reason, callback, position in positions_to_close:
            if callback:
                callback(position)
            self.close_position(pos_id, exit_price, reason)

        # Verificar margin call
        self._check_margin_call()

    def update_single_price(self, symbol: str, price: float):
        """
        Actualiza precio de un solo simbolo.

        Args:
            symbol: Par de trading
            price: Precio actual
        """
        self.update_prices({symbol: price})

    def _check_margin_call(self):
        """Verifica si hay margin call por drawdown excesivo"""
        if self.equity <= 0:
            logger.error("MARGIN CALL: Equity <= 0")
            if self.on_margin_call:
                self.on_margin_call(self.equity)

            # Cerrar todas las posiciones
            self.close_all_positions("Margin Call")
            return

        # Verificar drawdown maximo
        drawdown = 1 - (self.equity / self.config.initial_balance)
        if drawdown >= self.config.max_drawdown_pct:
            logger.warning(f"Alerta de drawdown: {drawdown*100:.2f}%")
            if self.on_margin_call:
                self.on_margin_call(self.equity)

    def _get_unrealized_pnl_unsafe(self, symbol: Optional[str] = None) -> float:
        """
        Obtiene PnL no realizado (versión sin lock para uso interno).

        Args:
            symbol: Filtrar por simbolo (opcional)

        Returns:
            PnL no realizado total o por simbolo
        """
        total_pnl = 0.0

        for position in self._positions.values():
            if symbol is None or position.symbol == symbol:
                if position.symbol in self._current_prices:
                    position.update_unrealized_pnl(
                        self._current_prices[position.symbol]
                    )
                total_pnl += position.unrealized_pnl

        return total_pnl

    def get_unrealized_pnl(self, symbol: Optional[str] = None) -> float:
        """
        Obtiene PnL no realizado (thread-safe).

        Args:
            symbol: Filtrar por simbolo (opcional)

        Returns:
            PnL no realizado total o por simbolo
        """
        with self._thread_safe():
            return self._get_unrealized_pnl_unsafe(symbol)

    def get_realized_pnl(self) -> float:
        """Obtiene PnL realizado total (thread-safe)"""
        with self._thread_safe():
            return sum(trade.pnl for trade in self._trade_history)

    def get_total_pnl(self) -> float:
        """Obtiene PnL total (realizado + no realizado) (thread-safe)"""
        with self._thread_safe():
            realized = sum(trade.pnl for trade in self._trade_history)
            unrealized = self._get_unrealized_pnl_unsafe()
            return realized + unrealized

    def get_position(self, position_id: str) -> Optional[PaperPosition]:
        """Obtiene una posicion por ID (thread-safe)"""
        with self._thread_safe():
            return self._positions.get(position_id)

    def get_positions_by_symbol(self, symbol: str) -> List[PaperPosition]:
        """Obtiene posiciones de un simbolo (thread-safe)"""
        with self._thread_safe():
            return [p for p in self._positions.values() if p.symbol == symbol]

    def get_position_summary(self) -> PositionSummary:
        """Obtiene resumen de posiciones (thread-safe)"""
        with self._thread_safe():
            long_positions = sum(
                1 for p in self._positions.values() if p.side == PositionSide.LONG
            )
            short_positions = sum(
                1 for p in self._positions.values() if p.side == PositionSide.SHORT
            )

            total_exposure = sum(
                p.quantity * self._current_prices.get(p.symbol, p.entry_price)
                for p in self._positions.values()
            )

            unrealized = self._get_unrealized_pnl_unsafe()
            realized = sum(trade.pnl for trade in self._trade_history)

            return PositionSummary(
                total_positions=len(self._positions),
                long_positions=long_positions,
                short_positions=short_positions,
                total_exposure=total_exposure,
                unrealized_pnl=unrealized,
                realized_pnl=realized,
                margin_used=total_exposure
            )

    def close_all_positions(self, reason: str = "Manual Close All"):
        """
        Cierra todas las posiciones abiertas (thread-safe).

        Args:
            reason: Razon del cierre
        """
        # Obtener lista de posiciones a cerrar dentro del lock
        positions_to_close = []
        with self._thread_safe():
            for position in list(self._positions.values()):
                current_price = self._current_prices.get(
                    position.symbol,
                    position.entry_price
                )
                positions_to_close.append((position.id, current_price))

        # Cerrar fuera del lock (close_position tiene su propio lock)
        for pos_id, price in positions_to_close:
            self.close_position(pos_id, price, reason)

    def update_stop_loss(
        self,
        position_id: str,
        new_stop_loss: float
    ) -> bool:
        """
        Actualiza el stop loss de una posicion (thread-safe).

        Args:
            position_id: ID de la posicion
            new_stop_loss: Nuevo precio de stop loss

        Returns:
            True si se actualizo, False si no existe
        """
        with self._thread_safe():
            if position_id not in self._positions:
                return False

            position = self._positions[position_id]
            old_sl = position.stop_loss
            position.stop_loss = new_stop_loss

        logger.info(
            f"Stop loss actualizado: {position_id} - {old_sl} -> {new_stop_loss}"
        )

        return True

    def update_take_profit(
        self,
        position_id: str,
        new_take_profit: float
    ) -> bool:
        """
        Actualiza el take profit de una posicion (thread-safe).

        Args:
            position_id: ID de la posicion
            new_take_profit: Nuevo precio de take profit

        Returns:
            True si se actualizo, False si no existe
        """
        with self._thread_safe():
            if position_id not in self._positions:
                return False

            position = self._positions[position_id]
            old_tp = position.take_profit
            position.take_profit = new_take_profit

        logger.info(
            f"Take profit actualizado: {position_id} - {old_tp} -> {new_take_profit}"
        )

        return True

    def get_win_rate(self) -> float:
        """Calcula la tasa de aciertos (thread-safe)"""
        with self._thread_safe():
            if not self._trade_history:
                return 0.0

            winning_trades = sum(1 for t in self._trade_history if t.pnl > 0)
            return (winning_trades / len(self._trade_history)) * 100

    def get_profit_factor(self) -> float:
        """Calcula el profit factor (thread-safe)"""
        with self._thread_safe():
            gross_profit = sum(t.pnl for t in self._trade_history if t.pnl > 0)
            gross_loss = abs(sum(t.pnl for t in self._trade_history if t.pnl < 0))

            if gross_loss == 0:
                return float('inf') if gross_profit > 0 else 0.0

            return gross_profit / gross_loss

    def reset(self):
        """Reinicia el gestor de posiciones (thread-safe)"""
        with self._thread_safe():
            self._balance = self.config.initial_balance
            self._positions.clear()
            self._trade_history.clear()
            self._current_prices.clear()

        logger.info("Position manager reiniciado")

    def get_statistics(self) -> Dict:
        """Obtiene estadisticas completas (thread-safe)"""
        with self._thread_safe():
            if not self._trade_history:
                return {
                    "total_trades": 0,
                    "winning_trades": 0,
                    "losing_trades": 0,
                    "win_rate": 0.0,
                    "profit_factor": 0.0,
                    "total_pnl": 0.0,
                    "avg_win": 0.0,
                    "avg_loss": 0.0,
                    "largest_win": 0.0,
                    "largest_loss": 0.0,
                }

            winning = [t for t in self._trade_history if t.pnl > 0]
            losing = [t for t in self._trade_history if t.pnl < 0]

            # Calcular métricas dentro del lock
            total_trades = len(self._trade_history)
            win_rate = (len(winning) / total_trades * 100) if total_trades > 0 else 0.0

            gross_profit = sum(t.pnl for t in winning)
            gross_loss = abs(sum(t.pnl for t in losing))
            profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (float('inf') if gross_profit > 0 else 0.0)

            realized_pnl = sum(trade.pnl for trade in self._trade_history)

            return {
                "total_trades": total_trades,
                "winning_trades": len(winning),
                "losing_trades": len(losing),
                "win_rate": win_rate,
                "profit_factor": profit_factor,
                "total_pnl": realized_pnl,
                "avg_win": sum(t.pnl for t in winning) / len(winning) if winning else 0.0,
                "avg_loss": sum(t.pnl for t in losing) / len(losing) if losing else 0.0,
                "largest_win": max((t.pnl for t in winning), default=0.0),
                "largest_loss": min((t.pnl for t in losing), default=0.0),
            }
