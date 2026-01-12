"""
Binance WebSocket Handler

Maneja la conexion WebSocket a Binance para recibir
datos de mercado en tiempo real.

Streams soportados:
- Trade stream: Trades individuales
- Kline stream: Velas OHLCV
- Ticker stream: Precios actuales
- Depth stream: Order book
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Callable, Optional, Dict, Any, List
from dataclasses import dataclass

try:
    import websockets
    from websockets.client import WebSocketClientProtocol
except ImportError:
    websockets = None
    WebSocketClientProtocol = None

from ..models import RealtimeCandle


logger = logging.getLogger(__name__)


@dataclass
class StreamConfig:
    """Configuracion de un stream WebSocket"""
    stream_type: str  # trade, kline, ticker, depth
    symbol: str
    interval: Optional[str] = None  # Para klines: 1m, 5m, 1h, etc.


class BinanceWebSocketHandler:
    """
    Handler para WebSocket de Binance.

    Soporta conexion a streams de mercado en tiempo real.

    Example:
        handler = BinanceWebSocketHandler()
        await handler.connect("BTC/USDT", "kline", "1m")
        handler.on_candle = my_candle_callback
        await handler.start()

    Attributes:
        base_url: URL base del WebSocket
        is_connected: Estado de conexion
        is_testnet: Si usa testnet
    """

    # URLs de WebSocket
    MAINNET_URL = "wss://stream.binance.com:9443/ws"
    TESTNET_URL = "wss://testnet.binance.vision/ws"
    FUTURES_URL = "wss://fstream.binance.com/ws"

    def __init__(self, testnet: bool = False, futures: bool = False):
        """
        Inicializa el handler.

        Args:
            testnet: Usar testnet en lugar de mainnet
            futures: Usar futures en lugar de spot
        """
        if websockets is None:
            raise ImportError(
                "websockets package is required. "
                "Install with: pip install websockets"
            )

        self.testnet = testnet
        self.futures = futures

        if futures:
            self.base_url = self.FUTURES_URL
        elif testnet:
            self.base_url = self.TESTNET_URL
        else:
            self.base_url = self.MAINNET_URL

        self._ws: Optional[WebSocketClientProtocol] = None
        self._running = False
        self._streams: List[StreamConfig] = []
        self._current_candles: Dict[str, RealtimeCandle] = {}

        # Callbacks
        self.on_candle: Optional[Callable[[RealtimeCandle], None]] = None
        self.on_trade: Optional[Callable[[Dict], None]] = None
        self.on_ticker: Optional[Callable[[Dict], None]] = None
        self.on_depth: Optional[Callable[[Dict], None]] = None
        self.on_error: Optional[Callable[[Exception], None]] = None
        self.on_connect: Optional[Callable[[], None]] = None
        self.on_disconnect: Optional[Callable[[], None]] = None

    @property
    def is_connected(self) -> bool:
        """Verifica si esta conectado"""
        return self._ws is not None and self._ws.open

    def _normalize_symbol(self, symbol: str) -> str:
        """
        Normaliza el formato del simbolo para Binance.

        Args:
            symbol: Simbolo en formato estandar (BTC/USDT)

        Returns:
            Simbolo en formato Binance (btcusdt)
        """
        return symbol.replace("/", "").lower()

    def _build_stream_name(self, config: StreamConfig) -> str:
        """
        Construye el nombre del stream para Binance.

        Args:
            config: Configuracion del stream

        Returns:
            Nombre del stream (ej: btcusdt@kline_1m)
        """
        symbol = self._normalize_symbol(config.symbol)

        if config.stream_type == "trade":
            return f"{symbol}@trade"
        elif config.stream_type == "kline":
            interval = config.interval or "1m"
            return f"{symbol}@kline_{interval}"
        elif config.stream_type == "ticker":
            return f"{symbol}@ticker"
        elif config.stream_type == "depth":
            return f"{symbol}@depth20@100ms"
        elif config.stream_type == "aggTrade":
            return f"{symbol}@aggTrade"
        else:
            return f"{symbol}@{config.stream_type}"

    def add_stream(
        self,
        symbol: str,
        stream_type: str = "kline",
        interval: str = "1m"
    ):
        """
        Agrega un stream a monitorear.

        Args:
            symbol: Par de trading (ej: BTC/USDT)
            stream_type: Tipo de stream (kline, trade, ticker, depth)
            interval: Intervalo para klines (1m, 5m, 15m, 1h, etc.)
        """
        config = StreamConfig(
            stream_type=stream_type,
            symbol=symbol,
            interval=interval
        )
        self._streams.append(config)
        logger.info(f"Stream agregado: {symbol} - {stream_type}")

    def _build_combined_url(self) -> str:
        """Construye URL con multiples streams"""
        if not self._streams:
            raise ValueError("No hay streams configurados")

        if len(self._streams) == 1:
            stream_name = self._build_stream_name(self._streams[0])
            return f"{self.base_url}/{stream_name}"
        else:
            # Usar combined stream
            stream_names = [
                self._build_stream_name(s) for s in self._streams
            ]
            combined = "/".join(stream_names)
            base = self.base_url.replace("/ws", "/stream")
            return f"{base}?streams={combined}"

    async def connect(self):
        """Establece conexion WebSocket"""
        if self.is_connected:
            logger.warning("Ya conectado")
            return

        url = self._build_combined_url()
        logger.info(f"Conectando a: {url}")

        try:
            self._ws = await websockets.connect(
                url,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=10
            )
            self._running = True

            if self.on_connect:
                self.on_connect()

            logger.info("Conexion WebSocket establecida")

        except Exception as e:
            logger.error(f"Error conectando: {e}")
            if self.on_error:
                self.on_error(e)
            raise

    async def disconnect(self):
        """Cierra la conexion WebSocket"""
        self._running = False

        if self._ws:
            await self._ws.close()
            self._ws = None

        if self.on_disconnect:
            self.on_disconnect()

        logger.info("Desconectado de WebSocket")

    async def start(self):
        """
        Inicia el loop de recepcion de mensajes.

        Este metodo bloquea hasta que se llame a stop().
        """
        if not self.is_connected:
            await self.connect()

        logger.info("Iniciando recepcion de datos...")

        try:
            async for message in self._ws:
                if not self._running:
                    break

                await self._handle_message(message)

        except websockets.exceptions.ConnectionClosed as e:
            logger.warning(f"Conexion cerrada: {e}")
            if self.on_disconnect:
                self.on_disconnect()

        except Exception as e:
            logger.error(f"Error en loop: {e}")
            if self.on_error:
                self.on_error(e)

    async def stop(self):
        """Detiene la recepcion de datos"""
        self._running = False
        await self.disconnect()

    async def _handle_message(self, message: str):
        """
        Procesa un mensaje recibido.

        Args:
            message: Mensaje JSON del WebSocket
        """
        try:
            data = json.loads(message)

            # Combined streams tienen estructura diferente
            if "stream" in data:
                stream_name = data["stream"]
                payload = data["data"]
            else:
                payload = data
                stream_name = None

            # Determinar tipo de mensaje
            event_type = payload.get("e", "")

            if event_type == "kline":
                await self._handle_kline(payload)
            elif event_type == "trade":
                await self._handle_trade(payload)
            elif event_type == "24hrTicker":
                await self._handle_ticker(payload)
            elif event_type == "depthUpdate":
                await self._handle_depth(payload)
            elif event_type == "aggTrade":
                await self._handle_trade(payload)

        except json.JSONDecodeError as e:
            logger.error(f"Error parseando JSON: {e}")
        except Exception as e:
            logger.error(f"Error procesando mensaje: {e}")

    async def _handle_kline(self, data: Dict[str, Any]):
        """Procesa mensaje de kline/vela"""
        kline = data.get("k", {})
        symbol = data.get("s", "")
        interval = kline.get("i", "1m")

        # Crear o actualizar vela
        key = f"{symbol}_{interval}"

        candle = RealtimeCandle(
            timestamp=datetime.fromtimestamp(kline["t"] / 1000),
            open=float(kline["o"]),
            high=float(kline["h"]),
            low=float(kline["l"]),
            close=float(kline["c"]),
            volume=float(kline["v"]),
            symbol=self._format_symbol(symbol),
            timeframe=interval,
            is_closed=kline.get("x", False),
            trades_count=kline.get("n", 0)
        )

        self._current_candles[key] = candle

        # Notificar callback
        if self.on_candle:
            self.on_candle(candle)

    async def _handle_trade(self, data: Dict[str, Any]):
        """Procesa mensaje de trade"""
        trade_data = {
            "symbol": self._format_symbol(data.get("s", "")),
            "price": float(data.get("p", 0)),
            "quantity": float(data.get("q", 0)),
            "timestamp": datetime.fromtimestamp(data.get("T", 0) / 1000),
            "is_buyer_maker": data.get("m", False),
            "trade_id": data.get("t", 0)
        }

        if self.on_trade:
            self.on_trade(trade_data)

    async def _handle_ticker(self, data: Dict[str, Any]):
        """Procesa mensaje de ticker"""
        ticker_data = {
            "symbol": self._format_symbol(data.get("s", "")),
            "price": float(data.get("c", 0)),
            "price_change": float(data.get("p", 0)),
            "price_change_pct": float(data.get("P", 0)),
            "high_24h": float(data.get("h", 0)),
            "low_24h": float(data.get("l", 0)),
            "volume_24h": float(data.get("v", 0)),
            "quote_volume_24h": float(data.get("q", 0)),
        }

        if self.on_ticker:
            self.on_ticker(ticker_data)

    async def _handle_depth(self, data: Dict[str, Any]):
        """Procesa mensaje de order book"""
        depth_data = {
            "symbol": self._format_symbol(data.get("s", "")),
            "bids": [(float(p), float(q)) for p, q in data.get("b", [])],
            "asks": [(float(p), float(q)) for p, q in data.get("a", [])],
            "last_update_id": data.get("u", 0),
        }

        if self.on_depth:
            self.on_depth(depth_data)

    def _format_symbol(self, symbol: str) -> str:
        """
        Convierte simbolo de Binance a formato estandar.

        Args:
            symbol: BTCUSDT

        Returns:
            BTC/USDT
        """
        # Lista de quotes comunes
        quotes = ["USDT", "BUSD", "BTC", "ETH", "BNB", "USD"]

        for quote in quotes:
            if symbol.endswith(quote):
                base = symbol[:-len(quote)]
                return f"{base}/{quote}"

        return symbol

    def get_current_candle(self, symbol: str, timeframe: str = "1m") -> Optional[RealtimeCandle]:
        """
        Obtiene la vela actual para un simbolo.

        Args:
            symbol: Par de trading
            timeframe: Temporalidad

        Returns:
            Vela actual o None
        """
        key = f"{self._normalize_symbol(symbol).upper()}_{timeframe}"
        return self._current_candles.get(key)

    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Obtiene el precio actual para un simbolo.

        Args:
            symbol: Par de trading

        Returns:
            Precio actual o None
        """
        # Buscar en cualquier timeframe
        symbol_normalized = self._normalize_symbol(symbol).upper()

        for key, candle in self._current_candles.items():
            if key.startswith(symbol_normalized):
                return candle.close

        return None


class BinanceFuturesWebSocketHandler(BinanceWebSocketHandler):
    """Handler especializado para Binance Futures"""

    def __init__(self, testnet: bool = False):
        super().__init__(testnet=testnet, futures=True)

    async def _handle_kline(self, data: Dict[str, Any]):
        """Procesa kline de futuros (incluye mark price)"""
        await super()._handle_kline(data)

        # Futures tiene datos adicionales
        kline = data.get("k", {})
        if "mp" in kline:  # Mark price
            logger.debug(f"Mark price: {kline['mp']}")
