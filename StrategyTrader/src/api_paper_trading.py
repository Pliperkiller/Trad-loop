"""
API REST para Paper Trading

Endpoints para controlar el sistema de paper trading:
- Iniciar/detener sesiones
- Obtener estado y metricas
- WebSocket para actualizaciones en tiempo real
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from paper_trading import PaperTradingConfig
from paper_trading.engine import PaperTradingEngine, SimpleMovingAverageStrategy
from paper_trading.models import PaperTradingState, PositionSide


logger = logging.getLogger(__name__)


# Instancia global del engine
_engine: Optional[PaperTradingEngine] = None
_connected_clients: List[WebSocket] = []


# Modelos Pydantic
class StartPaperTradingRequest(BaseModel):
    """Request para iniciar paper trading"""
    symbol: str = Field(..., description="Par de trading (ej: BTC/USDT)")
    timeframe: str = Field(default="1m", description="Temporalidad de velas")
    initial_balance: float = Field(default=10000, description="Capital inicial")
    commission_rate: float = Field(default=0.001, description="Tasa de comision")
    max_position_size: float = Field(default=0.25, description="Tamano max de posicion")
    risk_per_trade: float = Field(default=0.02, description="Riesgo por trade")
    use_mock_feed: bool = Field(default=False, description="Usar feed simulado")


class StopPaperTradingRequest(BaseModel):
    """Request para detener paper trading"""
    close_positions: bool = Field(default=True, description="Cerrar posiciones abiertas")


class PositionResponse(BaseModel):
    """Respuesta con informacion de posicion"""
    id: str
    symbol: str
    side: str
    entry_price: float
    quantity: float
    entry_time: datetime
    stop_loss: Optional[float]
    take_profit: Optional[float]
    unrealized_pnl: float


class TradeResponse(BaseModel):
    """Respuesta con informacion de trade"""
    id: str
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    quantity: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    return_pct: float
    exit_reason: str


class MetricsResponse(BaseModel):
    """Respuesta con metricas de performance"""
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate_pct: float
    profit_factor: float
    total_trades: int
    current_equity: float


class StateResponse(BaseModel):
    """Respuesta con estado del sistema"""
    is_running: bool
    is_paused: bool
    symbol: str
    strategy_name: str
    current_price: float
    balance: float
    equity: float
    open_positions: int
    total_trades: int
    realized_pnl: float
    unrealized_pnl: float
    win_rate: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager para la aplicacion"""
    logger.info("Paper Trading API iniciando...")
    yield
    # Cleanup
    global _engine
    if _engine and _engine.is_running:
        await _engine.stop()
    logger.info("Paper Trading API detenida")


# Crear aplicacion
app = FastAPI(
    title="Paper Trading API",
    description="API para controlar el sistema de paper trading",
    version="1.0.0",
    lifespan=lifespan
)


# Endpoints
@app.post("/api/v1/paper-trading/start", response_model=StateResponse)
async def start_paper_trading(
    request: StartPaperTradingRequest,
    background_tasks: BackgroundTasks
):
    """
    Inicia una sesion de paper trading.

    Crea un nuevo engine con la configuracion especificada
    y comienza a recibir datos de mercado.
    """
    global _engine

    if _engine and _engine.is_running:
        raise HTTPException(
            status_code=400,
            detail="Paper trading ya esta corriendo. Detener primero."
        )

    # Crear configuracion
    config = PaperTradingConfig(
        initial_balance=request.initial_balance,
        symbols=[request.symbol],
        commission_rate=request.commission_rate,
        max_position_size=request.max_position_size,
        risk_per_trade=request.risk_per_trade,
    )

    # Crear engine
    _engine = PaperTradingEngine(config, use_mock_feed=request.use_mock_feed)

    # Configurar estrategia (por ahora usar la simple)
    _engine.set_strategy(SimpleMovingAverageStrategy)

    # Configurar callback para WebSocket
    _engine.on_state_update = _broadcast_state

    # Iniciar en background
    background_tasks.add_task(
        _run_paper_trading,
        request.symbol,
        request.timeframe
    )

    # Esperar un momento para que inicie
    await asyncio.sleep(0.5)

    return _state_to_response(_engine.state)


async def _run_paper_trading(symbol: str, timeframe: str):
    """Ejecuta paper trading en background"""
    global _engine
    try:
        await _engine.start(symbol, timeframe)
    except Exception as e:
        logger.error(f"Error en paper trading: {e}")


@app.post("/api/v1/paper-trading/stop")
async def stop_paper_trading(request: StopPaperTradingRequest = None):
    """
    Detiene la sesion de paper trading.
    """
    global _engine

    if not _engine or not _engine.is_running:
        raise HTTPException(
            status_code=400,
            detail="Paper trading no esta corriendo"
        )

    if request and request.close_positions:
        current_price = _engine.state.current_price
        _engine.position_manager.close_all_positions("API Stop")

    await _engine.stop()

    return {"message": "Paper trading detenido", "success": True}


@app.post("/api/v1/paper-trading/pause")
async def pause_paper_trading():
    """Pausa el paper trading (no procesa senales)"""
    global _engine

    if not _engine or not _engine.is_running:
        raise HTTPException(status_code=400, detail="Paper trading no esta corriendo")

    _engine.pause()
    return {"message": "Paper trading pausado", "is_paused": True}


@app.post("/api/v1/paper-trading/resume")
async def resume_paper_trading():
    """Reanuda el paper trading"""
    global _engine

    if not _engine:
        raise HTTPException(status_code=400, detail="Paper trading no iniciado")

    _engine.resume()
    return {"message": "Paper trading reanudado", "is_paused": False}


@app.get("/api/v1/paper-trading/status", response_model=StateResponse)
async def get_status():
    """Obtiene el estado actual del paper trading"""
    global _engine

    if not _engine:
        return StateResponse(
            is_running=False,
            is_paused=False,
            symbol="",
            strategy_name="",
            current_price=0,
            balance=0,
            equity=0,
            open_positions=0,
            total_trades=0,
            realized_pnl=0,
            unrealized_pnl=0,
            win_rate=0
        )

    return _state_to_response(_engine.state)


@app.get("/api/v1/paper-trading/positions", response_model=List[PositionResponse])
async def get_positions():
    """Obtiene las posiciones abiertas"""
    global _engine

    if not _engine:
        return []

    positions = _engine.position_manager.positions
    return [
        PositionResponse(
            id=p.id,
            symbol=p.symbol,
            side=p.side.value,
            entry_price=p.entry_price,
            quantity=p.quantity,
            entry_time=p.entry_time,
            stop_loss=p.stop_loss,
            take_profit=p.take_profit,
            unrealized_pnl=p.unrealized_pnl
        )
        for p in positions
    ]


@app.get("/api/v1/paper-trading/trades", response_model=List[TradeResponse])
async def get_trades(limit: int = 100):
    """Obtiene el historial de trades cerrados"""
    global _engine

    if not _engine:
        return []

    trades = _engine.position_manager.trade_history[-limit:]
    return [
        TradeResponse(
            id=t.id,
            symbol=t.symbol,
            side=t.side.value,
            entry_price=t.entry_price,
            exit_price=t.exit_price,
            quantity=t.quantity,
            entry_time=t.entry_time,
            exit_time=t.exit_time,
            pnl=t.pnl,
            return_pct=t.return_pct,
            exit_reason=t.exit_reason
        )
        for t in trades
    ]


@app.get("/api/v1/paper-trading/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Obtiene las metricas de performance"""
    global _engine

    if not _engine:
        raise HTTPException(status_code=400, detail="Paper trading no iniciado")

    metrics = _engine.performance_tracker.get_metrics()

    return MetricsResponse(
        total_return_pct=metrics.get("total_return_pct", 0),
        sharpe_ratio=metrics.get("sharpe_ratio", 0),
        max_drawdown_pct=metrics.get("max_drawdown_pct", 0),
        win_rate_pct=metrics.get("win_rate_pct", 0),
        profit_factor=metrics.get("profit_factor", 0),
        total_trades=metrics.get("total_trades", 0),
        current_equity=metrics.get("current_equity", 0),
    )


@app.get("/api/v1/paper-trading/equity-curve")
async def get_equity_curve():
    """Obtiene la curva de equity"""
    global _engine

    if not _engine:
        return {"timestamps": [], "values": []}

    df = _engine.performance_tracker.get_equity_curve()
    return {
        "timestamps": df["timestamp"].dt.isoformat().tolist(),
        "values": df["equity"].tolist()
    }


@app.get("/api/v1/paper-trading/report")
async def get_report():
    """Obtiene reporte completo de performance"""
    global _engine

    if not _engine:
        raise HTTPException(status_code=400, detail="Paper trading no iniciado")

    return _engine.get_performance_report()


# WebSocket para actualizaciones en tiempo real
@app.websocket("/ws/paper-trading")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket para recibir actualizaciones en tiempo real.

    Envia el estado del sistema cada vez que hay cambios.
    """
    await websocket.accept()
    _connected_clients.append(websocket)

    try:
        # Enviar estado inicial
        if _engine:
            await websocket.send_json(_engine.state.to_dict())

        # Mantener conexion abierta
        while True:
            # Esperar mensaje del cliente (keepalive)
            data = await websocket.receive_text()

            if data == "ping":
                await websocket.send_text("pong")
            elif data == "status":
                if _engine:
                    await websocket.send_json(_engine.state.to_dict())

    except WebSocketDisconnect:
        _connected_clients.remove(websocket)
    except Exception as e:
        logger.error(f"Error en WebSocket: {e}")
        if websocket in _connected_clients:
            _connected_clients.remove(websocket)


async def _broadcast_state(state: PaperTradingState):
    """Envia estado a todos los clientes conectados"""
    for client in _connected_clients.copy():
        try:
            await client.send_json(state.to_dict())
        except Exception:
            _connected_clients.remove(client)


def _state_to_response(state: PaperTradingState) -> StateResponse:
    """Convierte estado interno a respuesta API"""
    return StateResponse(
        is_running=state.is_running,
        is_paused=state.is_paused,
        symbol=state.symbol,
        strategy_name=state.strategy_name,
        current_price=state.current_price,
        balance=state.balance,
        equity=state.equity,
        open_positions=state.open_positions,
        total_trades=state.total_trades,
        realized_pnl=state.realized_pnl,
        unrealized_pnl=state.unrealized_pnl,
        win_rate=state.win_rate
    )


# Ejecutar con: uvicorn api_paper_trading:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
