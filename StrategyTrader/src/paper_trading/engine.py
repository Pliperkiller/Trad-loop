"""
Paper Trading Engine

Motor principal que orquesta todos los componentes del sistema
de paper trading: feed de datos, estrategias, ordenes y posiciones.
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional, Callable, Dict, Any, Type
from abc import ABC, abstractmethod

from .models import (
    PaperTradingState,
    RealtimeCandle,
    Order,
    OrderSide,
    OrderType,
    PositionSide,
    TradeRecord,
)
from .config import PaperTradingConfig
from .realtime_feed import RealtimeFeedManager, MockFeedManager
from .order_simulator import OrderSimulator
from .position_manager import PositionManager
from .performance_tracker import RealtimePerformanceTracker


logger = logging.getLogger(__name__)


class RealtimeStrategy(ABC):
    """
    Clase base abstracta para estrategias en tiempo real.

    Las estrategias deben implementar on_candle() para procesar
    velas y generar senales de trading.

    Example:
        class MyStrategy(RealtimeStrategy):
            def on_candle(self, candle):
                if should_buy(candle):
                    self.buy(candle.close)
                elif should_sell(candle):
                    self.sell(candle.close)
    """

    def __init__(self, engine: "PaperTradingEngine"):
        self.engine = engine
        self.name = self.__class__.__name__

    @abstractmethod
    def on_candle(self, candle: RealtimeCandle):
        """
        Procesa una nueva vela y genera senales.

        Args:
            candle: Vela de mercado
        """
        pass

    def on_tick(self, symbol: str, price: float):
        """
        Procesa un tick de precio (opcional).

        Args:
            symbol: Par de trading
            price: Precio actual
        """
        pass

    def on_start(self):
        """Llamado cuando inicia el paper trading"""
        pass

    def on_stop(self):
        """Llamado cuando se detiene el paper trading"""
        pass

    def buy(
        self,
        price: float,
        quantity: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ):
        """
        Ejecuta una compra (posicion LONG).

        Args:
            price: Precio de entrada
            quantity: Cantidad (usa calculo automatico si None)
            stop_loss: Precio de stop loss
            take_profit: Precio de take profit
        """
        self.engine._execute_signal(
            side=PositionSide.LONG,
            price=price,
            quantity=quantity,
            stop_loss=stop_loss,
            take_profit=take_profit
        )

    def sell(
        self,
        price: float,
        quantity: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ):
        """
        Ejecuta una venta (cierra LONG o abre SHORT).

        Args:
            price: Precio de salida/entrada
            quantity: Cantidad
            stop_loss: Precio de stop loss (para shorts)
            take_profit: Precio de take profit (para shorts)
        """
        # Si hay posiciones long abiertas, cerrarlas
        positions = self.engine.position_manager.get_positions_by_symbol(
            self.engine.current_symbol
        )
        long_positions = [p for p in positions if p.side == PositionSide.LONG]

        if long_positions:
            for pos in long_positions:
                self.engine.position_manager.close_position(pos.id, price, "Signal Sell")
        else:
            # Abrir short si no hay longs
            self.engine._execute_signal(
                side=PositionSide.SHORT,
                price=price,
                quantity=quantity,
                stop_loss=stop_loss,
                take_profit=take_profit
            )

    def close_all(self, price: float):
        """
        Cierra todas las posiciones.

        Args:
            price: Precio de cierre
        """
        self.engine.position_manager.close_all_positions("Strategy Close All")


class PaperTradingEngine:
    """
    Motor principal de Paper Trading.

    Coordina todos los componentes:
    - Feed de datos en tiempo real
    - Estrategia de trading
    - Simulador de ordenes
    - Gestor de posiciones
    - Tracker de performance

    Example:
        config = PaperTradingConfig(initial_balance=10000)
        engine = PaperTradingEngine(config)

        class MyStrategy(RealtimeStrategy):
            def on_candle(self, candle):
                # Logica de trading
                pass

        engine.set_strategy(MyStrategy)
        await engine.start("BTC/USDT")

    Attributes:
        config: Configuracion de paper trading
        state: Estado actual del sistema
        position_manager: Gestor de posiciones
        order_simulator: Simulador de ordenes
        performance_tracker: Tracker de metricas
    """

    def __init__(
        self,
        config: Optional[PaperTradingConfig] = None,
        use_mock_feed: bool = False
    ):
        """
        Inicializa el motor.

        Args:
            config: Configuracion (usa default si None)
            use_mock_feed: Usar feed simulado en lugar de real
        """
        self.config = config or PaperTradingConfig()

        # Validar configuracion
        errors = self.config.validate()
        if errors:
            raise ValueError(f"Configuracion invalida: {errors}")

        # Inicializar componentes
        self.order_simulator = OrderSimulator(self.config)
        self.position_manager = PositionManager(self.config, self.order_simulator)
        self.performance_tracker = RealtimePerformanceTracker(self.config)

        # Feed de datos
        if use_mock_feed:
            self.feed_manager = MockFeedManager()
        else:
            self.feed_manager = RealtimeFeedManager(
                exchange=self.config.exchange,
                testnet=self.config.testnet
            )

        # Estado
        self._state = PaperTradingState()
        self._strategy: Optional[RealtimeStrategy] = None
        self._running = False
        self._paused = False
        self._task: Optional[asyncio.Task] = None
        self.current_symbol: str = ""

        # Callbacks
        self.on_trade: Optional[Callable[[TradeRecord], None]] = None
        self.on_candle: Optional[Callable[[RealtimeCandle], None]] = None
        self.on_state_update: Optional[Callable[[PaperTradingState], None]] = None

        # Configurar callbacks internos
        self._setup_callbacks()

    def _setup_callbacks(self):
        """Configura callbacks internos entre componentes"""
        # Cuando se cierra una posicion, agregar trade al tracker
        def on_position_closed(trade: TradeRecord):
            self.performance_tracker.add_trade(trade)
            if self.on_trade:
                self.on_trade(trade)

        self.position_manager.on_position_closed = on_position_closed

    def set_strategy(
        self,
        strategy_class: Type[RealtimeStrategy],
        **kwargs
    ):
        """
        Configura la estrategia a usar.

        Args:
            strategy_class: Clase de la estrategia
            **kwargs: Argumentos adicionales para la estrategia
        """
        self._strategy = strategy_class(self, **kwargs)
        self._state.strategy_name = self._strategy.name

        logger.info(f"Estrategia configurada: {self._strategy.name}")

    @property
    def state(self) -> PaperTradingState:
        """Estado actual del sistema"""
        return self._state

    @property
    def is_running(self) -> bool:
        """Verifica si esta corriendo"""
        return self._running

    @property
    def is_paused(self) -> bool:
        """Verifica si esta pausado"""
        return self._paused

    async def start(self, symbol: str, timeframe: str = "1m"):
        """
        Inicia el paper trading.

        Args:
            symbol: Par de trading (ej: BTC/USDT)
            timeframe: Temporalidad de velas
        """
        if self._running:
            logger.warning("Paper trading ya esta corriendo")
            return

        if not self._strategy:
            raise ValueError("Debe configurar una estrategia primero con set_strategy()")

        self.current_symbol = symbol
        self._running = True
        self._paused = False

        # Actualizar estado
        self._state.is_running = True
        self._state.symbol = symbol
        self._state.start_time = datetime.now()

        # Suscribirse a velas
        self.feed_manager.subscribe_candles(
            symbol=symbol,
            timeframe=timeframe,
            callback=self._on_candle_received
        )

        # Notificar a la estrategia
        self._strategy.on_start()

        logger.info(f"Paper trading iniciado: {symbol} @ {timeframe}")

        # Iniciar feed
        try:
            await self.feed_manager.start()
        except Exception as e:
            logger.error(f"Error en feed: {e}")
            self._running = False
            self._state.is_running = False
            raise

    def start_async(self, symbol: str, timeframe: str = "1m") -> asyncio.Task:
        """
        Inicia paper trading en background.

        Args:
            symbol: Par de trading
            timeframe: Temporalidad

        Returns:
            Task de asyncio
        """
        self._task = asyncio.create_task(self.start(symbol, timeframe))
        return self._task

    async def stop(self):
        """Detiene el paper trading"""
        if not self._running:
            return

        self._running = False
        self._state.is_running = False

        # Notificar a la estrategia
        if self._strategy:
            self._strategy.on_stop()

        # Detener feed
        await self.feed_manager.stop()

        # Cancelar task si existe
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        logger.info("Paper trading detenido")

    def pause(self):
        """Pausa el paper trading (no procesa senales)"""
        self._paused = True
        self._state.is_paused = True
        logger.info("Paper trading pausado")

    def resume(self):
        """Reanuda el paper trading"""
        self._paused = False
        self._state.is_paused = False
        logger.info("Paper trading reanudado")

    def _on_candle_received(self, candle: RealtimeCandle):
        """Callback cuando se recibe una vela"""
        if self._paused:
            return

        # Actualizar precio en componentes
        self.order_simulator.update_market_state(candle.symbol, candle.close)
        self.position_manager.update_single_price(candle.symbol, candle.close)

        # Actualizar equity en tracker
        self.performance_tracker.update_equity(self.position_manager.equity)

        # Actualizar estado
        self._update_state(candle)

        # Callback externo
        if self.on_candle:
            self.on_candle(candle)

        # Procesar con estrategia (solo velas cerradas por defecto)
        if self._strategy and candle.is_closed:
            try:
                self._strategy.on_candle(candle)
            except Exception as e:
                logger.error(f"Error en estrategia: {e}")

    def _update_state(self, candle: RealtimeCandle):
        """Actualiza el estado interno"""
        self._state.current_price = candle.close
        self._state.balance = self.position_manager.balance
        self._state.equity = self.position_manager.equity
        self._state.open_positions = len(self.position_manager.positions)
        self._state.total_trades = len(self.position_manager.trade_history)
        self._state.winning_trades = len([
            t for t in self.position_manager.trade_history if t.pnl > 0
        ])
        self._state.realized_pnl = self.position_manager.get_realized_pnl()
        self._state.unrealized_pnl = self.position_manager.get_unrealized_pnl()
        self._state.last_update = datetime.now()

        if self.on_state_update:
            self.on_state_update(self._state)

    def _execute_signal(
        self,
        side: PositionSide,
        price: float,
        quantity: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ):
        """Ejecuta una senal de trading"""
        if quantity is None:
            # Calcular cantidad basada en riesgo
            quantity = self._calculate_position_size(price, stop_loss)

        if quantity <= 0:
            logger.warning("Cantidad calculada es 0, no se abre posicion")
            return

        position = self.position_manager.open_position(
            symbol=self.current_symbol,
            side=side,
            quantity=quantity,
            entry_price=price,
            stop_loss=stop_loss,
            take_profit=take_profit
        )

        if position:
            logger.info(
                f"Posicion abierta: {side.value} {quantity:.6f} @ {price:.2f}"
            )

    def _calculate_position_size(
        self,
        price: float,
        stop_loss: Optional[float]
    ) -> float:
        """Calcula tamano de posicion basado en riesgo"""
        # Riesgo maximo por trade
        risk_amount = self.position_manager.balance * self.config.risk_per_trade

        if stop_loss:
            # Tamano basado en distancia al stop loss
            risk_per_unit = abs(price - stop_loss)
            if risk_per_unit > 0:
                quantity = risk_amount / risk_per_unit
            else:
                quantity = risk_amount / price
        else:
            # Tamano basado en % del balance
            quantity = risk_amount / price

        # Limitar al maximo permitido
        max_quantity = (
            self.position_manager.balance * self.config.max_position_size
        ) / price

        return min(quantity, max_quantity)

    def get_performance_report(self) -> Dict[str, Any]:
        """
        Obtiene reporte completo de performance.

        Returns:
            Diccionario con metricas y estadisticas
        """
        return {
            "state": self._state.to_dict(),
            "metrics": self.performance_tracker.get_metrics(),
            "position_stats": self.position_manager.get_statistics(),
            "verdict": self.performance_tracker.get_verdict(),
        }

    def print_report(self):
        """Imprime reporte de performance"""
        self.performance_tracker.print_report()

    def reset(self):
        """Reinicia todo el sistema"""
        if self._running:
            raise RuntimeError("No se puede reiniciar mientras esta corriendo")

        self.order_simulator = OrderSimulator(self.config)
        self.position_manager = PositionManager(self.config, self.order_simulator)
        self.performance_tracker = RealtimePerformanceTracker(self.config)

        self._state = PaperTradingState()
        self._setup_callbacks()

        logger.info("Paper trading engine reiniciado")


# Estrategia de ejemplo simple
class SimpleMovingAverageStrategy(RealtimeStrategy):
    """
    Estrategia simple de cruce de medias moviles.

    Compra cuando EMA rapida cruza por encima de EMA lenta.
    Vende cuando EMA rapida cruza por debajo de EMA lenta.
    """

    def __init__(
        self,
        engine: PaperTradingEngine,
        fast_period: int = 10,
        slow_period: int = 30
    ):
        super().__init__(engine)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self._prices: list = []
        self._prev_fast_ema: Optional[float] = None
        self._prev_slow_ema: Optional[float] = None

    def _calculate_ema(self, prices: list, period: int) -> float:
        """Calcula EMA simple"""
        if len(prices) < period:
            return sum(prices) / len(prices)

        multiplier = 2 / (period + 1)
        ema = prices[0]
        for price in prices[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        return ema

    def on_candle(self, candle: RealtimeCandle):
        """Procesa vela y genera senales"""
        self._prices.append(candle.close)

        # Mantener solo los ultimos N precios
        max_prices = max(self.fast_period, self.slow_period) * 2
        if len(self._prices) > max_prices:
            self._prices = self._prices[-max_prices:]

        # Necesitamos suficientes precios
        if len(self._prices) < self.slow_period:
            return

        # Calcular EMAs
        fast_ema = self._calculate_ema(self._prices, self.fast_period)
        slow_ema = self._calculate_ema(self._prices, self.slow_period)

        # Detectar cruces
        if self._prev_fast_ema and self._prev_slow_ema:
            # Cruce alcista
            if self._prev_fast_ema <= self._prev_slow_ema and fast_ema > slow_ema:
                stop_loss = candle.close * 0.98  # 2% stop loss
                take_profit = candle.close * 1.04  # 4% take profit
                self.buy(candle.close, stop_loss=stop_loss, take_profit=take_profit)

            # Cruce bajista
            elif self._prev_fast_ema >= self._prev_slow_ema and fast_ema < slow_ema:
                self.sell(candle.close)

        self._prev_fast_ema = fast_ema
        self._prev_slow_ema = slow_ema
