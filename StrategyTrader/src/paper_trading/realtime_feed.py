"""
Realtime Feed Manager

Gestor de feeds de datos en tiempo real.
Abstrae la conexion a diferentes exchanges y proporciona
una interfaz unificada para recibir datos de mercado.

Patron Observer para notificar a multiples suscriptores.
"""

import asyncio
import logging
from datetime import datetime
from typing import Callable, Dict, List, Optional, Set
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

from .models import RealtimeCandle
from .websocket_handlers.binance_ws import BinanceWebSocketHandler


logger = logging.getLogger(__name__)


class FeedStatus(Enum):
    """Estado del feed"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"


@dataclass
class FeedSubscription:
    """Representa una suscripcion a un feed"""
    symbol: str
    timeframe: str
    callback: Callable[[RealtimeCandle], None]
    subscription_id: str


class RealtimeFeedManager:
    """
    Gestor central de feeds de datos en tiempo real.

    Maneja conexiones a multiples exchanges y proporciona
    una interfaz unificada para suscribirse a datos de mercado.

    Example:
        manager = RealtimeFeedManager(exchange="binance")

        # Suscribirse a velas
        manager.subscribe_candles(
            symbol="BTC/USDT",
            timeframe="1m",
            callback=my_callback
        )

        # Iniciar
        await manager.start()

    Attributes:
        exchange: Exchange actual
        status: Estado de conexion
    """

    SUPPORTED_EXCHANGES = ["binance", "kraken"]

    def __init__(
        self,
        exchange: str = "binance",
        testnet: bool = False,
        reconnect_attempts: int = 5,
        reconnect_delay: float = 5.0
    ):
        """
        Inicializa el gestor de feeds.

        Args:
            exchange: Exchange a usar (binance, kraken)
            testnet: Usar testnet del exchange
            reconnect_attempts: Intentos de reconexion
            reconnect_delay: Delay entre reconexiones en segundos
        """
        self.exchange = exchange.lower()
        self.testnet = testnet
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_delay = reconnect_delay

        self._status = FeedStatus.DISCONNECTED
        self._handler = self._create_handler()
        self._subscriptions: Dict[str, List[FeedSubscription]] = {}
        self._candle_callbacks: List[Callable[[RealtimeCandle], None]] = []
        self._price_callbacks: List[Callable[[str, float], None]] = []
        self._current_prices: Dict[str, float] = {}
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._subscription_counter = 0

        # Configurar callbacks del handler
        self._setup_handler_callbacks()

    def _create_handler(self):
        """Crea el handler apropiado para el exchange"""
        if self.exchange == "binance":
            return BinanceWebSocketHandler(testnet=self.testnet)
        elif self.exchange == "kraken":
            # TODO: Implementar KrakenWebSocketHandler
            logger.warning("Kraken no implementado, usando Binance")
            return BinanceWebSocketHandler(testnet=self.testnet)
        else:
            raise ValueError(f"Exchange no soportado: {self.exchange}")

    def _setup_handler_callbacks(self):
        """Configura callbacks del handler interno"""
        self._handler.on_candle = self._on_candle_received
        self._handler.on_trade = self._on_trade_received
        self._handler.on_ticker = self._on_ticker_received
        self._handler.on_connect = self._on_connected
        self._handler.on_disconnect = self._on_disconnected
        self._handler.on_error = self._on_error

    @property
    def status(self) -> FeedStatus:
        """Estado actual del feed"""
        return self._status

    @property
    def is_connected(self) -> bool:
        """Verifica si esta conectado"""
        return self._status == FeedStatus.CONNECTED

    def subscribe_candles(
        self,
        symbol: str,
        timeframe: str = "1m",
        callback: Optional[Callable[[RealtimeCandle], None]] = None
    ) -> str:
        """
        Suscribe a velas de un simbolo.

        Args:
            symbol: Par de trading (ej: BTC/USDT)
            timeframe: Temporalidad (1m, 5m, 15m, 1h, etc.)
            callback: Funcion a llamar con cada vela

        Returns:
            ID de suscripcion
        """
        self._subscription_counter += 1
        sub_id = f"candle_{self._subscription_counter}"

        # Agregar stream al handler
        self._handler.add_stream(symbol, "kline", timeframe)

        # Registrar suscripcion
        key = f"{symbol}_{timeframe}"
        if key not in self._subscriptions:
            self._subscriptions[key] = []

        if callback:
            subscription = FeedSubscription(
                symbol=symbol,
                timeframe=timeframe,
                callback=callback,
                subscription_id=sub_id
            )
            self._subscriptions[key].append(subscription)

        logger.info(f"Suscrito a {symbol} {timeframe} (ID: {sub_id})")
        return sub_id

    def subscribe_trades(
        self,
        symbol: str,
        callback: Callable[[Dict], None]
    ) -> str:
        """
        Suscribe a trades de un simbolo.

        Args:
            symbol: Par de trading
            callback: Funcion a llamar con cada trade

        Returns:
            ID de suscripcion
        """
        self._subscription_counter += 1
        sub_id = f"trade_{self._subscription_counter}"

        self._handler.add_stream(symbol, "trade")
        self._handler.on_trade = callback

        logger.info(f"Suscrito a trades de {symbol} (ID: {sub_id})")
        return sub_id

    def subscribe_ticker(
        self,
        symbol: str,
        callback: Callable[[Dict], None]
    ) -> str:
        """
        Suscribe al ticker de un simbolo.

        Args:
            symbol: Par de trading
            callback: Funcion a llamar con cada actualizacion

        Returns:
            ID de suscripcion
        """
        self._subscription_counter += 1
        sub_id = f"ticker_{self._subscription_counter}"

        self._handler.add_stream(symbol, "ticker")
        self._handler.on_ticker = callback

        logger.info(f"Suscrito a ticker de {symbol} (ID: {sub_id})")
        return sub_id

    def unsubscribe(self, subscription_id: str):
        """
        Cancela una suscripcion.

        Args:
            subscription_id: ID de la suscripcion a cancelar
        """
        for key, subs in self._subscriptions.items():
            self._subscriptions[key] = [
                s for s in subs if s.subscription_id != subscription_id
            ]
        logger.info(f"Suscripcion {subscription_id} cancelada")

    def on_candle(self, callback: Callable[[RealtimeCandle], None]):
        """
        Registra un callback global para todas las velas.

        Args:
            callback: Funcion a llamar con cada vela
        """
        self._candle_callbacks.append(callback)

    def on_price_update(self, callback: Callable[[str, float], None]):
        """
        Registra un callback para actualizaciones de precio.

        Args:
            callback: Funcion(symbol, price) a llamar
        """
        self._price_callbacks.append(callback)

    async def start(self):
        """
        Inicia la conexion y recepcion de datos.

        Este metodo es bloqueante. Usar start_async() para no bloquear.
        """
        if self._running:
            logger.warning("Feed ya esta corriendo")
            return

        self._running = True
        self._status = FeedStatus.CONNECTING

        attempt = 0
        while self._running and attempt < self.reconnect_attempts:
            try:
                await self._handler.connect()
                self._status = FeedStatus.CONNECTED
                await self._handler.start()

            except Exception as e:
                attempt += 1
                logger.error(f"Error en feed (intento {attempt}): {e}")
                self._status = FeedStatus.RECONNECTING

                if attempt < self.reconnect_attempts:
                    await asyncio.sleep(self.reconnect_delay)
                else:
                    self._status = FeedStatus.ERROR
                    raise

    def start_async(self) -> asyncio.Task:
        """
        Inicia el feed en background.

        Returns:
            Task de asyncio
        """
        self._task = asyncio.create_task(self.start())
        return self._task

    async def stop(self):
        """Detiene el feed"""
        self._running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        await self._handler.stop()
        self._status = FeedStatus.DISCONNECTED
        logger.info("Feed detenido")

    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Obtiene el precio actual de un simbolo.

        Args:
            symbol: Par de trading

        Returns:
            Precio actual o None si no disponible
        """
        # Primero intentar del handler
        price = self._handler.get_current_price(symbol)
        if price:
            return price

        # Luego de cache local
        return self._current_prices.get(symbol)

    def get_current_candle(
        self,
        symbol: str,
        timeframe: str = "1m"
    ) -> Optional[RealtimeCandle]:
        """
        Obtiene la vela actual de un simbolo.

        Args:
            symbol: Par de trading
            timeframe: Temporalidad

        Returns:
            Vela actual o None
        """
        return self._handler.get_current_candle(symbol, timeframe)

    def _on_candle_received(self, candle: RealtimeCandle):
        """Callback interno cuando se recibe una vela"""
        # Actualizar precio
        self._current_prices[candle.symbol] = candle.close

        # Notificar callbacks globales
        for callback in self._candle_callbacks:
            try:
                callback(candle)
            except Exception as e:
                logger.error(f"Error en callback de vela: {e}")

        # Notificar suscriptores especificos
        key = f"{candle.symbol}_{candle.timeframe}"
        if key in self._subscriptions:
            for sub in self._subscriptions[key]:
                try:
                    sub.callback(candle)
                except Exception as e:
                    logger.error(f"Error en suscriptor {sub.subscription_id}: {e}")

        # Notificar callbacks de precio
        for callback in self._price_callbacks:
            try:
                callback(candle.symbol, candle.close)
            except Exception as e:
                logger.error(f"Error en callback de precio: {e}")

    def _on_trade_received(self, trade: Dict):
        """Callback interno cuando se recibe un trade"""
        symbol = trade.get("symbol", "")
        price = trade.get("price", 0)
        self._current_prices[symbol] = price

    def _on_ticker_received(self, ticker: Dict):
        """Callback interno cuando se recibe ticker"""
        symbol = ticker.get("symbol", "")
        price = ticker.get("price", 0)
        self._current_prices[symbol] = price

    def _on_connected(self):
        """Callback cuando se conecta"""
        self._status = FeedStatus.CONNECTED
        logger.info("Feed conectado")

    def _on_disconnected(self):
        """Callback cuando se desconecta"""
        if self._running:
            self._status = FeedStatus.RECONNECTING
        else:
            self._status = FeedStatus.DISCONNECTED
        logger.warning("Feed desconectado")

    def _on_error(self, error: Exception):
        """Callback cuando hay error"""
        logger.error(f"Error en feed: {error}")
        self._status = FeedStatus.ERROR


class MockFeedManager(RealtimeFeedManager):
    """
    Feed manager para pruebas.

    Genera datos simulados en lugar de conectar a un exchange real.
    """

    def __init__(self, **kwargs):
        # No llamar al padre para evitar crear handler real
        self.exchange = "mock"
        self._status = FeedStatus.DISCONNECTED
        self._subscriptions: Dict[str, List[FeedSubscription]] = {}
        self._candle_callbacks: List[Callable[[RealtimeCandle], None]] = []
        self._price_callbacks: List[Callable[[str, float], None]] = []
        self._current_prices: Dict[str, float] = {}
        self._running = False
        self._subscription_counter = 0
        self._mock_price = 50000.0  # Precio inicial BTC
        self._mock_symbols: Set[str] = set()

    def subscribe_candles(
        self,
        symbol: str,
        timeframe: str = "1m",
        callback: Optional[Callable[[RealtimeCandle], None]] = None
    ) -> str:
        """Suscribe a velas mock"""
        self._subscription_counter += 1
        sub_id = f"mock_candle_{self._subscription_counter}"
        self._mock_symbols.add(symbol)

        key = f"{symbol}_{timeframe}"
        if key not in self._subscriptions:
            self._subscriptions[key] = []

        if callback:
            subscription = FeedSubscription(
                symbol=symbol,
                timeframe=timeframe,
                callback=callback,
                subscription_id=sub_id
            )
            self._subscriptions[key].append(subscription)

        return sub_id

    async def start(self):
        """Inicia generacion de datos mock"""
        import random

        self._running = True
        self._status = FeedStatus.CONNECTED

        while self._running:
            for symbol in self._mock_symbols:
                # Generar variacion aleatoria
                change = random.uniform(-0.001, 0.001)
                self._mock_price *= (1 + change)

                candle = RealtimeCandle(
                    timestamp=datetime.now(),
                    open=self._mock_price,
                    high=self._mock_price * 1.001,
                    low=self._mock_price * 0.999,
                    close=self._mock_price,
                    volume=random.uniform(100, 1000),
                    symbol=symbol,
                    timeframe="1m",
                    is_closed=False
                )

                self._current_prices[symbol] = self._mock_price

                # Notificar callbacks
                for callback in self._candle_callbacks:
                    callback(candle)

                key = f"{symbol}_1m"
                if key in self._subscriptions:
                    for sub in self._subscriptions[key]:
                        sub.callback(candle)

            await asyncio.sleep(1)  # Generar cada segundo

    async def stop(self):
        """Detiene generacion"""
        self._running = False
        self._status = FeedStatus.DISCONNECTED
