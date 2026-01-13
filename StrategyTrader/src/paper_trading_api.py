"""
API de Paper Trading para Trad-loop

Proporciona endpoints REST y WebSocket para:
- Iniciar/detener sesiones de paper trading
- Consultar estado y posiciones
- Recibir updates en tiempo real

Uso:
    from src.paper_trading_api import register_paper_trading_routes
    register_paper_trading_routes(app)
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Type
from dataclasses import dataclass, field, asdict

from fastapi import HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from .paper_trading.engine import PaperTradingEngine, RealtimeStrategy, SimpleMovingAverageStrategy
from .paper_trading.config import PaperTradingConfig
from .paper_trading.models import PaperTradingState, TradeRecord, RealtimeCandle

logger = logging.getLogger(__name__)


# ============ Modelos Pydantic ============

class PaperTradingSessionRequest(BaseModel):
    """Request para iniciar una sesion de paper trading"""
    strategy_id: str
    exchange: str
    symbol: str
    timeframe: str = "1m"
    initial_balance: float = 10000.0
    risk_per_trade: float = 0.02
    max_position_size: float = 0.1
    commission: float = 0.001
    parameters: Dict[str, Any] = {}


class SessionResponse(BaseModel):
    """Response con info de una sesion"""
    session_id: str
    status: str
    message: str


class SessionStateResponse(BaseModel):
    """Response con estado completo de una sesion"""
    session_id: str
    state: Dict[str, Any]
    positions: List[Dict[str, Any]]
    recent_trades: List[Dict[str, Any]]
    performance: Dict[str, Any]


# ============ Gestor de Sesiones ============

@dataclass
class PaperTradingSession:
    """Representa una sesion activa de paper trading"""
    id: str
    engine: PaperTradingEngine
    strategy_id: str
    symbol: str
    timeframe: str
    created_at: datetime = field(default_factory=datetime.now)
    websocket_clients: List[WebSocket] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "strategy_id": self.strategy_id,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "created_at": self.created_at.isoformat(),
            "is_running": self.engine.is_running,
            "is_paused": self.engine.is_paused,
            "state": self.engine.state.to_dict() if self.engine.state else None,
        }


class PaperTradingSessionManager:
    """Gestor de sesiones de paper trading"""

    def __init__(self):
        self._sessions: Dict[str, PaperTradingSession] = {}
        self._strategies: Dict[str, Type[RealtimeStrategy]] = {}
        self._counter = 0

        # Registrar estrategias por defecto
        self._register_default_strategies()

    def _register_default_strategies(self):
        """Registra estrategias disponibles"""
        self._strategies["ma_crossover"] = SimpleMovingAverageStrategy
        # Agregar mas estrategias aqui...

    def register_strategy(self, strategy_id: str, strategy_class: Type[RealtimeStrategy]):
        """Registra una estrategia para usar en paper trading"""
        self._strategies[strategy_id] = strategy_class

    def get_available_strategies(self) -> List[Dict[str, Any]]:
        """Lista estrategias disponibles"""
        return [
            {
                "id": "ma_crossover",
                "name": "Moving Average Crossover",
                "description": "Estrategia basada en cruce de medias moviles",
                "parameters": [
                    {"name": "fast_period", "type": "int", "default": 10, "min": 2, "max": 50},
                    {"name": "slow_period", "type": "int", "default": 30, "min": 10, "max": 200},
                ]
            }
        ]

    def create_session(
        self,
        strategy_id: str,
        exchange: str,
        symbol: str,
        timeframe: str,
        initial_balance: float,
        risk_per_trade: float,
        max_position_size: float,
        commission: float,
        parameters: Dict[str, Any]
    ) -> PaperTradingSession:
        """Crea una nueva sesion de paper trading"""
        if strategy_id not in self._strategies:
            raise ValueError(f"Strategy '{strategy_id}' not found")

        # Generar ID de sesion
        self._counter += 1
        session_id = f"pt_{self._counter:04d}"

        # Crear configuracion
        config = PaperTradingConfig(
            exchange=exchange,
            initial_balance=initial_balance,
            risk_per_trade=risk_per_trade,
            max_position_size=max_position_size,
            commission_rate=commission,
        )

        # Crear engine
        engine = PaperTradingEngine(config, use_mock_feed=False)

        # Configurar estrategia
        strategy_class = self._strategies[strategy_id]
        engine.set_strategy(strategy_class, **parameters)

        # Crear sesion
        session = PaperTradingSession(
            id=session_id,
            engine=engine,
            strategy_id=strategy_id,
            symbol=symbol,
            timeframe=timeframe,
        )

        self._sessions[session_id] = session

        return session

    def get_session(self, session_id: str) -> Optional[PaperTradingSession]:
        """Obtiene una sesion por ID"""
        return self._sessions.get(session_id)

    def get_all_sessions(self) -> List[PaperTradingSession]:
        """Lista todas las sesiones"""
        return list(self._sessions.values())

    def get_active_sessions(self) -> List[PaperTradingSession]:
        """Lista sesiones activas"""
        return [s for s in self._sessions.values() if s.engine.is_running]

    async def start_session(self, session_id: str) -> bool:
        """Inicia una sesion"""
        session = self.get_session(session_id)
        if not session:
            return False

        if session.engine.is_running:
            return True

        # Configurar callbacks para WebSocket
        def on_state_update(state: PaperTradingState):
            asyncio.create_task(self._broadcast_state(session))

        def on_trade(trade: TradeRecord):
            asyncio.create_task(self._broadcast_trade(session, trade))

        def on_candle(candle: RealtimeCandle):
            asyncio.create_task(self._broadcast_candle(session, candle))

        session.engine.on_state_update = on_state_update
        session.engine.on_trade = on_trade
        session.engine.on_candle = on_candle

        # Iniciar en background
        session.engine.start_async(session.symbol, session.timeframe)

        return True

    async def stop_session(self, session_id: str) -> bool:
        """Detiene una sesion"""
        session = self.get_session(session_id)
        if not session:
            return False

        await session.engine.stop()
        return True

    def pause_session(self, session_id: str) -> bool:
        """Pausa una sesion"""
        session = self.get_session(session_id)
        if not session:
            return False

        session.engine.pause()
        return True

    def resume_session(self, session_id: str) -> bool:
        """Reanuda una sesion pausada"""
        session = self.get_session(session_id)
        if not session:
            return False

        session.engine.resume()
        return True

    async def delete_session(self, session_id: str) -> bool:
        """Elimina una sesion"""
        session = self.get_session(session_id)
        if not session:
            return False

        # Detener si esta corriendo
        if session.engine.is_running:
            await session.engine.stop()

        # Cerrar WebSocket clients
        for ws in session.websocket_clients:
            try:
                await ws.close()
            except Exception:
                pass

        del self._sessions[session_id]
        return True

    def add_websocket_client(self, session_id: str, websocket: WebSocket):
        """Agrega un cliente WebSocket a una sesion"""
        session = self.get_session(session_id)
        if session:
            session.websocket_clients.append(websocket)

    def remove_websocket_client(self, session_id: str, websocket: WebSocket):
        """Remueve un cliente WebSocket"""
        session = self.get_session(session_id)
        if session and websocket in session.websocket_clients:
            session.websocket_clients.remove(websocket)

    async def _broadcast_state(self, session: PaperTradingSession):
        """Envia estado a todos los clientes WebSocket"""
        if not session.websocket_clients:
            return

        message = {
            "type": "state",
            "data": session.engine.state.to_dict()
        }

        await self._broadcast(session, message)

    async def _broadcast_trade(self, session: PaperTradingSession, trade: TradeRecord):
        """Envia trade a todos los clientes WebSocket"""
        if not session.websocket_clients:
            return

        message = {
            "type": "trade",
            "data": trade.to_dict()
        }

        await self._broadcast(session, message)

    async def _broadcast_candle(self, session: PaperTradingSession, candle: RealtimeCandle):
        """Envia vela a todos los clientes WebSocket"""
        if not session.websocket_clients:
            return

        message = {
            "type": "candle",
            "data": candle.to_dict()
        }

        await self._broadcast(session, message)

    async def _broadcast(self, session: PaperTradingSession, message: Dict[str, Any]):
        """Broadcast mensaje a todos los clientes"""
        import json

        disconnected = []
        for ws in session.websocket_clients:
            try:
                await ws.send_json(message)
            except Exception:
                disconnected.append(ws)

        # Limpiar clientes desconectados
        for ws in disconnected:
            session.websocket_clients.remove(ws)


# Instancia global del gestor de sesiones
_session_manager = PaperTradingSessionManager()


def get_session_manager() -> PaperTradingSessionManager:
    """Obtiene el gestor de sesiones"""
    return _session_manager


# ============ Registro de Rutas ============

def register_paper_trading_routes(app):
    """
    Registra las rutas de paper trading en la app FastAPI.

    Uso:
        from src.paper_trading_api import register_paper_trading_routes
        register_paper_trading_routes(app)
    """

    manager = get_session_manager()

    # ============ REST Endpoints ============

    @app.get("/api/v1/paper-trading/strategies")
    async def list_paper_trading_strategies() -> List[Dict[str, Any]]:
        """Lista estrategias disponibles para paper trading"""
        return manager.get_available_strategies()

    @app.post("/api/v1/paper-trading/start")
    async def start_paper_trading(request: PaperTradingSessionRequest) -> SessionResponse:
        """
        Inicia una nueva sesion de paper trading.

        La sesion se ejecuta en background. Use WebSocket o GET endpoints
        para obtener actualizaciones.
        """
        try:
            # Convertir risk_per_trade de porcentaje a decimal si es > 1
            # Ej: 2 -> 0.02 (2%), pero 0.02 se mantiene como 0.02
            risk_per_trade = request.risk_per_trade
            if risk_per_trade > 1:
                risk_per_trade = risk_per_trade / 100.0

            session = manager.create_session(
                strategy_id=request.strategy_id,
                exchange=request.exchange,
                symbol=request.symbol,
                timeframe=request.timeframe,
                initial_balance=request.initial_balance,
                risk_per_trade=risk_per_trade,
                max_position_size=request.max_position_size,
                commission=request.commission,
                parameters=request.parameters
            )

            # Iniciar sesion
            await manager.start_session(session.id)

            return SessionResponse(
                session_id=session.id,
                status="running",
                message=f"Paper trading started for {request.symbol}"
            )

        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Error starting paper trading: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/v1/paper-trading/sessions")
    async def list_paper_trading_sessions() -> List[Dict[str, Any]]:
        """Lista todas las sesiones de paper trading"""
        sessions = manager.get_all_sessions()
        return [s.to_dict() for s in sessions]

    @app.get("/api/v1/paper-trading/{session_id}")
    async def get_paper_trading_session(session_id: str) -> SessionStateResponse:
        """Obtiene estado completo de una sesion"""
        session = manager.get_session(session_id)

        if not session:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

        engine = session.engine

        # Obtener posiciones
        positions = []
        if hasattr(engine, 'position_manager') and engine.position_manager:
            positions = [p.to_dict() for p in engine.position_manager.positions]

        # Obtener trades recientes
        trades = []
        if hasattr(engine, 'position_manager') and engine.position_manager:
            trades = [t.to_dict() for t in engine.position_manager.trade_history[-20:]]

        # Obtener performance
        performance = {}
        if hasattr(engine, 'get_performance_report'):
            try:
                performance = engine.get_performance_report()
            except Exception:
                pass

        return SessionStateResponse(
            session_id=session_id,
            state=session.engine.state.to_dict(),
            positions=positions,
            recent_trades=trades,
            performance=performance
        )

    @app.post("/api/v1/paper-trading/{session_id}/stop")
    async def stop_paper_trading_session(session_id: str) -> Dict[str, Any]:
        """Detiene una sesion de paper trading"""
        session = manager.get_session(session_id)

        if not session:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

        success = await manager.stop_session(session_id)

        return {
            "session_id": session_id,
            "stopped": success,
            "final_state": session.engine.state.to_dict()
        }

    @app.post("/api/v1/paper-trading/{session_id}/pause")
    async def pause_paper_trading_session(session_id: str) -> Dict[str, Any]:
        """Pausa una sesion (no procesa senales pero mantiene posiciones)"""
        success = manager.pause_session(session_id)

        if not success:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

        return {"session_id": session_id, "paused": True}

    @app.post("/api/v1/paper-trading/{session_id}/resume")
    async def resume_paper_trading_session(session_id: str) -> Dict[str, Any]:
        """Reanuda una sesion pausada"""
        success = manager.resume_session(session_id)

        if not success:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

        return {"session_id": session_id, "resumed": True}

    @app.delete("/api/v1/paper-trading/{session_id}")
    async def delete_paper_trading_session(session_id: str) -> Dict[str, Any]:
        """Elimina una sesion (la detiene primero si esta corriendo)"""
        success = await manager.delete_session(session_id)

        if not success:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

        return {"session_id": session_id, "deleted": True}

    @app.get("/api/v1/paper-trading/{session_id}/trades")
    async def get_paper_trading_trades(session_id: str) -> List[Dict[str, Any]]:
        """Obtiene historial de trades de una sesion"""
        session = manager.get_session(session_id)

        if not session:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

        trades = []
        if hasattr(session.engine, 'position_manager') and session.engine.position_manager:
            trades = [t.to_dict() for t in session.engine.position_manager.trade_history]

        return trades

    @app.get("/api/v1/paper-trading/{session_id}/positions")
    async def get_paper_trading_positions(session_id: str) -> List[Dict[str, Any]]:
        """Obtiene posiciones abiertas de una sesion"""
        session = manager.get_session(session_id)

        if not session:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

        positions = []
        if hasattr(session.engine, 'position_manager') and session.engine.position_manager:
            positions = [p.to_dict() for p in session.engine.position_manager.positions]

        return positions

    # ============ WebSocket Endpoint ============

    @app.websocket("/ws/paper-trading/{session_id}")
    async def websocket_paper_trading(websocket: WebSocket, session_id: str):
        """
        WebSocket para recibir updates en tiempo real de una sesion.

        Mensajes enviados:
        - {"type": "state", "data": {...}}  - Estado actualizado
        - {"type": "trade", "data": {...}}  - Nuevo trade ejecutado
        - {"type": "candle", "data": {...}} - Nueva vela recibida
        """
        session = manager.get_session(session_id)

        if not session:
            await websocket.close(code=4004, reason="Session not found")
            return

        await websocket.accept()
        manager.add_websocket_client(session_id, websocket)

        # Enviar estado inicial
        try:
            await websocket.send_json({
                "type": "connected",
                "data": {
                    "session_id": session_id,
                    "state": session.engine.state.to_dict()
                }
            })
        except Exception:
            pass

        try:
            while True:
                # Mantener conexion abierta, escuchar mensajes del cliente
                data = await websocket.receive_text()

                # Procesar comandos del cliente si es necesario
                import json
                try:
                    message = json.loads(data)
                    cmd = message.get("command")

                    if cmd == "pause":
                        manager.pause_session(session_id)
                        await websocket.send_json({"type": "paused"})

                    elif cmd == "resume":
                        manager.resume_session(session_id)
                        await websocket.send_json({"type": "resumed"})

                    elif cmd == "get_state":
                        await websocket.send_json({
                            "type": "state",
                            "data": session.engine.state.to_dict()
                        })

                except json.JSONDecodeError:
                    pass

        except WebSocketDisconnect:
            pass
        finally:
            manager.remove_websocket_client(session_id, websocket)

    return app
