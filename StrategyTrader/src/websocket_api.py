"""
WebSocket API para datos en tiempo real de Trad-loop

Proporciona endpoints WebSocket para:
- Velas OHLCV en tiempo real
- Progreso de backtests
- Updates de paper trading

Uso:
    Los endpoints WebSocket se registran automaticamente al importar este modulo
    en api.py con: from src.websocket_api import register_websocket_routes
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, Set, Optional, Any
from dataclasses import dataclass, asdict

from fastapi import WebSocket, WebSocketDisconnect

# Importar ccxt para datos en tiempo real
try:
    import ccxt.async_support as ccxt_async
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False
    print("Warning: ccxt async not available for WebSocket")


@dataclass
class CandleUpdate:
    """Representa una actualizacion de vela"""
    type: str  # 'candle'
    symbol: str
    timeframe: str
    exchange: str
    data: Dict[str, Any]
    is_closed: bool  # True si la vela esta cerrada, False si es update de vela actual


class ConnectionManager:
    """Gestiona conexiones WebSocket activas por canal"""

    def __init__(self):
        # Mapea canal -> set de conexiones
        # Canal format: "candles:{exchange}:{symbol}:{timeframe}"
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        # Mapea canal -> task de streaming
        self.streaming_tasks: Dict[str, asyncio.Task] = {}
        # Lock para operaciones thread-safe
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, channel: str) -> None:
        """Conecta un cliente a un canal"""
        await websocket.accept()
        async with self._lock:
            if channel not in self.active_connections:
                self.active_connections[channel] = set()
            self.active_connections[channel].add(websocket)

    async def disconnect(self, websocket: WebSocket, channel: str) -> None:
        """Desconecta un cliente de un canal"""
        async with self._lock:
            if channel in self.active_connections:
                self.active_connections[channel].discard(websocket)
                # Si no quedan conexiones, detener streaming
                if not self.active_connections[channel]:
                    del self.active_connections[channel]
                    if channel in self.streaming_tasks:
                        self.streaming_tasks[channel].cancel()
                        del self.streaming_tasks[channel]

    async def broadcast(self, channel: str, message: dict) -> None:
        """Envia mensaje a todos los clientes de un canal"""
        if channel not in self.active_connections:
            return

        disconnected = set()
        for connection in self.active_connections[channel]:
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.add(connection)

        # Limpiar conexiones muertas
        for conn in disconnected:
            await self.disconnect(conn, channel)

    def get_connection_count(self, channel: str) -> int:
        """Retorna numero de conexiones en un canal"""
        return len(self.active_connections.get(channel, set()))

    def get_all_channels(self) -> list:
        """Retorna lista de canales activos"""
        return list(self.active_connections.keys())


# Instancia global del manager
manager = ConnectionManager()


class ExchangeStreamer:
    """Maneja streaming de datos desde exchanges via CCXT"""

    def __init__(self):
        self.exchanges: Dict[str, Any] = {}

    async def get_exchange(self, exchange_id: str):
        """Obtiene o crea instancia de exchange"""
        if exchange_id not in self.exchanges:
            if not CCXT_AVAILABLE:
                raise RuntimeError("CCXT async not available")

            exchange_class = getattr(ccxt_async, exchange_id, None)
            if not exchange_class:
                raise ValueError(f"Exchange {exchange_id} not supported")

            self.exchanges[exchange_id] = exchange_class({
                'enableRateLimit': True,
            })

        return self.exchanges[exchange_id]

    async def close_exchange(self, exchange_id: str):
        """Cierra conexion de exchange"""
        if exchange_id in self.exchanges:
            await self.exchanges[exchange_id].close()
            del self.exchanges[exchange_id]

    async def stream_candles(
        self,
        channel: str,
        exchange_id: str,
        symbol: str,
        timeframe: str
    ):
        """
        Stream de velas en tiempo real.

        Usa polling de CCXT ya que no todos los exchanges soportan WebSocket.
        Para exchanges con WebSocket nativo, se podria optimizar.
        """
        try:
            exchange = await self.get_exchange(exchange_id)

            # Intervalo de polling basado en timeframe
            poll_intervals = {
                '1m': 5,    # Cada 5 segundos
                '5m': 15,   # Cada 15 segundos
                '15m': 30,  # Cada 30 segundos
                '30m': 60,  # Cada minuto
                '1h': 60,   # Cada minuto
                '4h': 120,  # Cada 2 minutos
                '1d': 300,  # Cada 5 minutos
            }
            interval = poll_intervals.get(timeframe, 30)

            last_candle_time = None

            while True:
                try:
                    # Obtener ultima vela
                    ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, limit=2)

                    if ohlcv and len(ohlcv) > 0:
                        # La ultima vela puede estar incompleta
                        current_candle = ohlcv[-1]
                        previous_candle = ohlcv[-2] if len(ohlcv) > 1 else None

                        candle_time = current_candle[0]

                        # Detectar si la vela cambio (nueva vela)
                        is_new_candle = last_candle_time is not None and candle_time != last_candle_time

                        # Si hay nueva vela, enviar la anterior como cerrada
                        if is_new_candle and previous_candle:
                            closed_update = CandleUpdate(
                                type='candle',
                                symbol=symbol,
                                timeframe=timeframe,
                                exchange=exchange_id,
                                data={
                                    'time': previous_candle[0] // 1000,  # Unix seconds
                                    'open': previous_candle[1],
                                    'high': previous_candle[2],
                                    'low': previous_candle[3],
                                    'close': previous_candle[4],
                                    'volume': previous_candle[5],
                                },
                                is_closed=True
                            )
                            await manager.broadcast(channel, asdict(closed_update))

                        # Enviar vela actual (puede estar incompleta)
                        update = CandleUpdate(
                            type='candle',
                            symbol=symbol,
                            timeframe=timeframe,
                            exchange=exchange_id,
                            data={
                                'time': current_candle[0] // 1000,  # Unix seconds
                                'open': current_candle[1],
                                'high': current_candle[2],
                                'low': current_candle[3],
                                'close': current_candle[4],
                                'volume': current_candle[5],
                            },
                            is_closed=False
                        )
                        await manager.broadcast(channel, asdict(update))

                        last_candle_time = candle_time

                except Exception as e:
                    # Enviar error pero continuar
                    error_msg = {
                        'type': 'error',
                        'message': str(e),
                        'channel': channel
                    }
                    await manager.broadcast(channel, error_msg)

                # Esperar antes del siguiente poll
                await asyncio.sleep(interval)

        except asyncio.CancelledError:
            # Streaming cancelado, limpiar
            pass
        except Exception as e:
            error_msg = {
                'type': 'error',
                'message': f"Stream error: {str(e)}",
                'channel': channel
            }
            await manager.broadcast(channel, error_msg)


# Instancia global del streamer
streamer = ExchangeStreamer()


def register_websocket_routes(app):
    """
    Registra las rutas WebSocket en la aplicacion FastAPI.

    Llamar desde api.py:
        from src.websocket_api import register_websocket_routes
        register_websocket_routes(app)
    """

    @app.websocket("/ws/candles/{exchange}/{symbol:path}/{timeframe}")
    async def websocket_candles(
        websocket: WebSocket,
        exchange: str,
        symbol: str,
        timeframe: str
    ):
        """
        WebSocket para recibir velas en tiempo real.

        Conectar a: ws://localhost:8000/ws/candles/binance/BTC/USDT/1m

        Mensajes recibidos:
        {
            "type": "candle",
            "symbol": "BTC/USDT",
            "timeframe": "1m",
            "exchange": "binance",
            "data": {
                "time": 1234567890,
                "open": 50000.0,
                "high": 50100.0,
                "low": 49900.0,
                "close": 50050.0,
                "volume": 123.45
            },
            "is_closed": false
        }
        """
        # Decodificar symbol (puede venir URL-encoded)
        symbol_decoded = symbol.replace('%2F', '/')

        channel = f"candles:{exchange}:{symbol_decoded}:{timeframe}"

        await manager.connect(websocket, channel)

        # Iniciar streaming si es la primera conexion al canal
        if channel not in manager.streaming_tasks:
            task = asyncio.create_task(
                streamer.stream_candles(channel, exchange, symbol_decoded, timeframe)
            )
            manager.streaming_tasks[channel] = task

        # Enviar mensaje de bienvenida
        welcome = {
            'type': 'connected',
            'channel': channel,
            'exchange': exchange,
            'symbol': symbol_decoded,
            'timeframe': timeframe,
            'message': f'Subscribed to {symbol_decoded} {timeframe} on {exchange}'
        }
        await websocket.send_json(welcome)

        try:
            # Mantener conexion abierta, escuchar mensajes del cliente
            while True:
                data = await websocket.receive_text()

                # Procesar comandos del cliente si es necesario
                try:
                    msg = json.loads(data)
                    if msg.get('type') == 'ping':
                        await websocket.send_json({'type': 'pong'})
                except json.JSONDecodeError:
                    pass

        except WebSocketDisconnect:
            await manager.disconnect(websocket, channel)

    @app.websocket("/ws/status")
    async def websocket_status(websocket: WebSocket):
        """
        WebSocket para monitorear estado del servidor.

        Retorna informacion sobre canales activos y conexiones.
        """
        await websocket.accept()

        try:
            while True:
                # Enviar status cada 5 segundos
                status = {
                    'type': 'status',
                    'timestamp': datetime.now().isoformat(),
                    'channels': manager.get_all_channels(),
                    'connections': {
                        ch: manager.get_connection_count(ch)
                        for ch in manager.get_all_channels()
                    }
                }
                await websocket.send_json(status)
                await asyncio.sleep(5)

        except WebSocketDisconnect:
            pass

    @app.get("/api/v1/ws/channels")
    async def get_active_channels():
        """Lista canales WebSocket activos"""
        return {
            'channels': manager.get_all_channels(),
            'connections': {
                ch: manager.get_connection_count(ch)
                for ch in manager.get_all_channels()
            }
        }

    return app
