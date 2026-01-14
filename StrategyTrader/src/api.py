"""
API REST para exponer trades de Trad-loop a fyGraphr
Ejecutar: uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
"""

import sys
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Agregar el directorio raiz de Trad-loop al path para importar DataExtractor
# api.py -> src/ -> StrategyTrader/ -> Trad-loop/
TRAD_LOOP_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(TRAD_LOOP_ROOT))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

# Importar componentes de DataExtractor
try:
    from DataExtractor.src.application.services import DataExtractionService
    from DataExtractor.src.infrastructure.exchanges.ccxt_adapter import (
        CCXTAdapter,
        list_available_exchanges,
        get_exchanges_by_category
    )
    from DataExtractor.src.domain import MarketConfig, MarketType, Timeframe
    DATA_EXTRACTOR_AVAILABLE = True
except ImportError as e:
    logger.warning(f"DataExtractor not available: {e}")
    DATA_EXTRACTOR_AVAILABLE = False

app = FastAPI(
    title="Trad-loop API",
    description="API para acceder a trades y métricas de estrategias de trading",
    version="1.0.0"
)

# CORS para permitir acceso desde fyGraphr (cualquier origen en desarrollo)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permitir cualquier origen para acceso desde red local
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============ Servicios de DataExtractor ============

_data_service = None
_ccxt_instances: Dict[str, Any] = {}

if DATA_EXTRACTOR_AVAILABLE:
    _data_service = DataExtractionService()

def get_ccxt_adapter(exchange_id: str):
    """Cache de adaptadores CCXT para evitar reconexiones."""
    if not DATA_EXTRACTOR_AVAILABLE:
        raise RuntimeError("DataExtractor no disponible")
    if exchange_id not in _ccxt_instances:
        _ccxt_instances[exchange_id] = CCXTAdapter(exchange_id)
    return _ccxt_instances[exchange_id]


# ============ Modelos Pydantic ============

class Trade(BaseModel):
    id: str
    entry_time: str  # ISO format
    exit_time: str   # ISO format
    entry_price: float
    exit_price: float
    quantity: float
    position_type: str  # 'LONG' | 'SHORT'
    pnl: float
    return_pct: float
    reason: str


class TradesResponse(BaseModel):
    trades: List[Trade]
    symbol: str
    strategy_name: str
    total_trades: int


class PerformanceMetrics(BaseModel):
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    total_return_pct: float
    max_drawdown_pct: float
    final_capital: float
    avg_win: float
    avg_loss: float
    sharpe_ratio: float


class StrategyInfo(BaseModel):
    id: str
    name: str
    symbol: str
    timeframe: str
    total_trades: int
    is_active: bool


class StrategiesResponse(BaseModel):
    strategies: List[StrategyInfo]


# ============ Almacenamiento en memoria ============

_strategy_registry: Dict[str, Any] = {}


def register_strategy(strategy_id: str, strategy_instance: Any) -> str:
    """
    Registrar una estrategia para exponer sus trades via API.

    Uso:
        from src.api import register_strategy
        from src.strategy import MovingAverageCrossoverStrategy

        strategy = MovingAverageCrossoverStrategy(config)
        strategy.load_data(data)
        strategy.backtest()

        register_strategy("mi-estrategia", strategy)

    Returns:
        El ID de la estrategia registrada
    """
    _strategy_registry[strategy_id] = strategy_instance
    return strategy_id


def unregister_strategy(strategy_id: str) -> bool:
    """Desregistrar una estrategia"""
    if strategy_id in _strategy_registry:
        del _strategy_registry[strategy_id]
        return True
    return False


def clear_all_strategies():
    """Limpiar todas las estrategias registradas"""
    _strategy_registry.clear()


# ============ Endpoints de Datos (DataExtractor) ============

@app.get("/api/v1/exchanges")
async def get_exchanges():
    """
    Lista todos los exchanges disponibles.

    Returns:
        - exchanges: Lista completa de 100+ exchanges soportados por CCXT
        - by_category: Exchanges agrupados por categoria (tier_1, tier_2, futures, dex)
        - configured: Exchanges con configuracion especifica (Binance, Kraken)
    """
    if not DATA_EXTRACTOR_AVAILABLE:
        raise HTTPException(status_code=503, detail="DataExtractor no disponible")

    return {
        "exchanges": list_available_exchanges(),
        "by_category": get_exchanges_by_category(),
        "configured": _data_service.get_available_exchanges()
    }


@app.get("/api/v1/exchanges/{exchange_id}/symbols")
async def get_symbols(exchange_id: str):
    """
    Lista simbolos disponibles en un exchange.

    Args:
        exchange_id: ID del exchange (ej: binance, kraken, bybit)

    Returns:
        Lista de pares de trading disponibles (ej: BTC/USDT, ETH/USD)
    """
    if not DATA_EXTRACTOR_AVAILABLE:
        raise HTTPException(status_code=503, detail="DataExtractor no disponible")

    try:
        adapter = get_ccxt_adapter(exchange_id)
        symbols = adapter.get_supported_symbols()
        return {
            "exchange": exchange_id,
            "count": len(symbols),
            "symbols": symbols
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al obtener simbolos: {str(e)}")


@app.get("/api/v1/exchanges/{exchange_id}/symbols/catalog")
async def get_symbols_catalog(exchange_id: str):
    """
    Obtiene un catalogo organizado de simbolos del exchange.

    Agrupa los simbolos por moneda cotizada (USDT, BTC, ETH, etc.)
    e incluye una lista de simbolos populares al inicio.

    Returns:
        - exchange: ID del exchange
        - total: Total de simbolos
        - popular: Lista de simbolos populares (BTC, ETH, etc. contra USDT)
        - by_quote: Simbolos agrupados por moneda cotizada
    """
    if not DATA_EXTRACTOR_AVAILABLE:
        raise HTTPException(status_code=503, detail="DataExtractor no disponible")

    try:
        adapter = get_ccxt_adapter(exchange_id)
        symbols = adapter.get_supported_symbols()

        # Simbolos populares (pares comunes contra USDT)
        popular_bases = ['BTC', 'ETH', 'BNB', 'SOL', 'XRP', 'DOGE', 'ADA', 'AVAX', 'DOT', 'LINK',
                         'MATIC', 'UNI', 'ATOM', 'LTC', 'ETC', 'XLM', 'ALGO', 'NEAR', 'FTM', 'SAND']
        popular_quotes = ['USDT', 'USDC', 'USD', 'BUSD']

        popular = []
        for base in popular_bases:
            for quote in popular_quotes:
                pair = f"{base}/{quote}"
                if pair in symbols:
                    popular.append(pair)
                    break  # Solo agregar una vez por base

        # Agrupar por moneda cotizada (quote currency)
        by_quote: Dict[str, List[str]] = {}
        for symbol in symbols:
            if '/' in symbol:
                parts = symbol.split('/')
                quote = parts[1].split(':')[0]  # Manejar futuros como BTC/USDT:USDT
                if quote not in by_quote:
                    by_quote[quote] = []
                by_quote[quote].append(symbol)

        # Ordenar por cantidad de simbolos (mas populares primero)
        sorted_quotes = sorted(by_quote.keys(), key=lambda q: len(by_quote[q]), reverse=True)
        by_quote_sorted = {q: sorted(by_quote[q]) for q in sorted_quotes}

        return {
            "exchange": exchange_id,
            "total": len(symbols),
            "popular": popular,
            "by_quote": by_quote_sorted,
            "quote_currencies": sorted_quotes
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al obtener catalogo: {str(e)}")


@app.get("/api/v1/exchanges/{exchange_id}/info")
async def get_exchange_info(exchange_id: str):
    """
    Obtiene informacion detallada de un exchange.

    Returns:
        - exchange_id: ID del exchange
        - name: Nombre del exchange
        - has_ohlcv: Si soporta datos OHLCV
        - supported_timeframes: Lista de temporalidades disponibles
        - rate_limit: Limite de peticiones (ms entre requests)
    """
    if not DATA_EXTRACTOR_AVAILABLE:
        raise HTTPException(status_code=503, detail="DataExtractor no disponible")

    try:
        adapter = get_ccxt_adapter(exchange_id)
        return adapter.get_exchange_info()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al obtener info: {str(e)}")


class OHLCVResponse(BaseModel):
    """Respuesta del endpoint OHLCV"""
    symbol: str
    timeframe: str
    exchange: str
    count: int
    warmup_count: int = 0  # Número de velas de warmup al inicio (para cálculo de indicadores)
    data: List[Dict[str, Any]]


@app.get("/api/v1/ohlcv", response_model=OHLCVResponse)
async def get_ohlcv(
    exchange: str = Query(..., description="ID del exchange (ej: binance)"),
    symbol: str = Query(..., description="Par de trading (ej: BTC/USDT)"),
    timeframe: str = Query(..., description="Temporalidad (ej: 1m, 5m, 1h, 1d)"),
    start: str = Query(..., description="Fecha inicio (ISO format: 2024-01-01T00:00:00)"),
    end: str = Query(..., description="Fecha fin (ISO format: 2024-01-31T23:59:59)"),
    warmup_candles: int = Query(100, description="Velas adicionales para warmup de indicadores (default: 100)"),
):
    """
    Obtiene datos OHLCV (velas) de un exchange.

    Los datos se obtienen directamente del exchange via CCXT.
    Para grandes rangos de fechas, considerar usar el endpoint de extraccion async.

    El parametro warmup_candles permite obtener velas adicionales antes del rango solicitado
    para que los indicadores tecnicos (EMA, RSI, MACD, etc.) tengan suficientes datos
    historicos para calcular valores precisos desde el inicio del rango.

    Returns:
        - symbol: Par de trading
        - timeframe: Temporalidad
        - exchange: Exchange de origen
        - count: Numero de velas
        - data: Lista de velas [{timestamp, open, high, low, close, volume}]
    """
    if not DATA_EXTRACTOR_AVAILABLE:
        raise HTTPException(status_code=503, detail="DataExtractor no disponible")

    try:
        adapter = get_ccxt_adapter(exchange)

        # Mapear string a enum de Timeframe
        timeframe_map = {
            "1m": Timeframe.ONE_MINUTE,
            "5m": Timeframe.FIVE_MINUTES,
            "15m": Timeframe.FIFTEEN_MINUTES,
            "30m": Timeframe.THIRTY_MINUTES,
            "1h": Timeframe.ONE_HOUR,
            "2h": Timeframe.TWO_HOURS,
            "4h": Timeframe.FOUR_HOURS,
            "6h": Timeframe.SIX_HOURS,
            "12h": Timeframe.TWELVE_HOURS,
            "1d": Timeframe.ONE_DAY,
            "3d": Timeframe.THREE_DAYS,
            "1w": Timeframe.ONE_WEEK,
            "1M": Timeframe.ONE_MONTH,
        }

        # Mapear timeframe a minutos para calcular warmup
        timeframe_minutes = {
            "1m": 1,
            "5m": 5,
            "15m": 15,
            "30m": 30,
            "1h": 60,
            "2h": 120,
            "4h": 240,
            "6h": 360,
            "12h": 720,
            "1d": 1440,
            "3d": 4320,
            "1w": 10080,
            "1M": 43200,
        }

        if timeframe not in timeframe_map:
            raise HTTPException(
                status_code=400,
                detail=f"Timeframe '{timeframe}' no valido. Opciones: {list(timeframe_map.keys())}"
            )

        tf_enum = timeframe_map[timeframe]

        # Parsear fechas
        try:
            start_dt = datetime.fromisoformat(start.replace('Z', '+00:00'))
        except ValueError:
            raise HTTPException(status_code=400, detail="Formato de fecha inicio invalido. Use ISO format.")

        try:
            end_dt = datetime.fromisoformat(end.replace('Z', '+00:00'))
        except ValueError:
            raise HTTPException(status_code=400, detail="Formato de fecha fin invalido. Use ISO format.")

        if start_dt >= end_dt:
            raise HTTPException(status_code=400, detail="La fecha inicio debe ser anterior a la fecha fin")

        # Calcular fecha inicio extendida para warmup de indicadores
        if warmup_candles > 0:
            warmup_minutes = warmup_candles * timeframe_minutes[timeframe]
            extended_start_dt = start_dt - timedelta(minutes=warmup_minutes)
        else:
            extended_start_dt = start_dt

        # Obtener datos OHLCV (con warmup incluido)
        df = adapter.fetch_ohlcv(symbol, tf_enum, extended_start_dt, end_dt)

        # Calcular cuántas velas son de warmup (antes del start_dt solicitado)
        actual_warmup_count = 0
        if warmup_candles > 0 and 'timestamp' in df.columns and len(df) > 0:
            try:
                # Asegurar comparación sin timezone (naive)
                start_naive = start_dt.replace(tzinfo=None) if start_dt.tzinfo else start_dt
                # Contar velas con timestamp < start_dt
                timestamps = df['timestamp']
                if hasattr(timestamps.iloc[0], 'tzinfo') and timestamps.iloc[0].tzinfo:
                    timestamps = timestamps.dt.tz_localize(None)
                warmup_mask = timestamps < start_naive
                actual_warmup_count = int(warmup_mask.sum())
            except Exception as e:
                # Si falla el cálculo de warmup, continuar sin warmup
                logger.warning(f"Could not calculate warmup count: {e}")
                actual_warmup_count = 0

        # Convertir timestamp a string ISO para JSON
        data = df.to_dict(orient='records')
        for row in data:
            if 'timestamp' in row and hasattr(row['timestamp'], 'isoformat'):
                row['timestamp'] = row['timestamp'].isoformat()

        return OHLCVResponse(
            symbol=symbol,
            timeframe=timeframe,
            exchange=exchange,
            count=len(data),
            warmup_count=actual_warmup_count,
            data=data
        )

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al obtener datos OHLCV: {str(e)}")


# ============ Endpoints Principales ============

@app.get("/")
async def root():
    """Health check"""
    return {
        "service": "Trad-loop API",
        "status": "running",
        "version": "1.0.0",
        "features": {
            "data_extractor": DATA_EXTRACTOR_AVAILABLE,
            "websocket": WEBSOCKET_AVAILABLE if 'WEBSOCKET_AVAILABLE' in globals() else False,
            "backtest": BACKTEST_AVAILABLE if 'BACKTEST_AVAILABLE' in globals() else False,
        }
    }


@app.get("/api/v1/strategies", response_model=StrategiesResponse)
async def list_strategies():
    """Listar todas las estrategias registradas"""
    strategies = []

    for strategy_id, strategy in _strategy_registry.items():
        strategies.append(StrategyInfo(
            id=strategy_id,
            name=strategy.__class__.__name__,
            symbol=strategy.config.symbol,
            timeframe=strategy.config.timeframe,
            total_trades=len(strategy.closed_trades),
            is_active=len(strategy.positions) > 0
        ))

    return StrategiesResponse(strategies=strategies)


@app.get("/api/v1/trades/{strategy_id}", response_model=TradesResponse)
async def get_trades(
    strategy_id: str,
    start_time: Optional[str] = Query(None, description="Filtrar desde (ISO format)"),
    end_time: Optional[str] = Query(None, description="Filtrar hasta (ISO format)"),
    position_type: Optional[str] = Query(None, description="Filtrar por tipo: LONG o SHORT"),
    profitable_only: Optional[bool] = Query(None, description="Solo trades rentables")
):
    """
    Obtener trades de una estrategia.

    Soporta filtros opcionales por:
    - Rango de tiempo (start_time, end_time)
    - Tipo de posición (LONG/SHORT)
    - Solo trades rentables
    """
    if strategy_id not in _strategy_registry:
        raise HTTPException(
            status_code=404,
            detail=f"Estrategia '{strategy_id}' no encontrada. Estrategias disponibles: {list(_strategy_registry.keys())}"
        )

    strategy = _strategy_registry[strategy_id]
    trades = strategy.closed_trades.copy()

    # Aplicar filtros
    if start_time:
        try:
            start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
            trades = [t for t in trades if t['entry_time'] >= start_dt]
        except ValueError:
            raise HTTPException(status_code=400, detail="start_time debe estar en formato ISO")

    if end_time:
        try:
            end_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
            trades = [t for t in trades if t['exit_time'] <= end_dt]
        except ValueError:
            raise HTTPException(status_code=400, detail="end_time debe estar en formato ISO")

    if position_type:
        position_type_upper = position_type.upper()
        if position_type_upper not in ['LONG', 'SHORT']:
            raise HTTPException(status_code=400, detail="position_type debe ser LONG o SHORT")
        trades = [t for t in trades if t.get('position_type') == position_type_upper]

    if profitable_only:
        trades = [t for t in trades if t['pnl'] > 0]

    # Convertir a formato de respuesta
    trades_response = []
    for i, t in enumerate(trades):
        # Convertir datetime a ISO string
        entry_time = t['entry_time']
        exit_time = t['exit_time']

        if hasattr(entry_time, 'isoformat'):
            entry_time = entry_time.isoformat()
        else:
            entry_time = str(entry_time)

        if hasattr(exit_time, 'isoformat'):
            exit_time = exit_time.isoformat()
        else:
            exit_time = str(exit_time)

        trades_response.append(Trade(
            id=f"{strategy_id}-trade-{i}",
            entry_time=entry_time,
            exit_time=exit_time,
            entry_price=t['entry_price'],
            exit_price=t['exit_price'],
            quantity=t['quantity'],
            position_type=t.get('position_type', 'LONG'),  # Default LONG para compatibilidad
            pnl=t['pnl'],
            return_pct=t['return_pct'],
            reason=t['reason']
        ))

    return TradesResponse(
        trades=trades_response,
        symbol=strategy.config.symbol,
        strategy_name=strategy.__class__.__name__,
        total_trades=len(trades_response)
    )


@app.get("/api/v1/performance/{strategy_id}", response_model=PerformanceMetrics)
async def get_performance(strategy_id: str):
    """Obtener métricas de rendimiento de una estrategia"""
    if strategy_id not in _strategy_registry:
        raise HTTPException(
            status_code=404,
            detail=f"Estrategia '{strategy_id}' no encontrada"
        )

    strategy = _strategy_registry[strategy_id]
    metrics = strategy.get_performance_metrics()

    if not metrics:
        raise HTTPException(
            status_code=404,
            detail="No hay datos de performance disponibles. Ejecute backtest() primero."
        )

    return PerformanceMetrics(
        total_trades=metrics.get('total_trades', 0),
        winning_trades=metrics.get('winning_trades', 0),
        losing_trades=metrics.get('losing_trades', 0),
        win_rate=metrics.get('win_rate', 0),
        profit_factor=metrics.get('profit_factor', 0),
        total_return_pct=metrics.get('total_return_pct', 0),
        max_drawdown_pct=metrics.get('max_drawdown_pct', 0),
        final_capital=metrics.get('final_capital', 0),
        avg_win=metrics.get('avg_win', 0),
        avg_loss=metrics.get('avg_loss', 0),
        sharpe_ratio=metrics.get('sharpe_ratio', 0)
    )


@app.get("/api/v1/equity/{strategy_id}")
async def get_equity_curve(strategy_id: str):
    """Obtener la curva de equity de una estrategia"""
    if strategy_id not in _strategy_registry:
        raise HTTPException(
            status_code=404,
            detail=f"Estrategia '{strategy_id}' no encontrada"
        )

    strategy = _strategy_registry[strategy_id]

    return {
        "strategy_id": strategy_id,
        "initial_capital": strategy.config.initial_capital,
        "equity_curve": strategy.equity_curve,
        "final_capital": strategy.equity_curve[-1] if strategy.equity_curve else strategy.config.initial_capital
    }


# ============ WebSocket Routes ============

try:
    from .websocket_api import register_websocket_routes
    register_websocket_routes(app)
    WEBSOCKET_AVAILABLE = True
except ImportError as e:
    logger.warning(f"WebSocket routes not available: {e}")
    WEBSOCKET_AVAILABLE = False


# ============ Backtest & Optimization Routes ============

try:
    from .backtest_api import register_backtest_routes
    register_backtest_routes(app)
    BACKTEST_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Backtest routes not available: {e}")
    BACKTEST_AVAILABLE = False


# ============ Paper Trading Routes ============

try:
    from .paper_trading_api import register_paper_trading_routes
    register_paper_trading_routes(app)
    PAPER_TRADING_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Paper trading routes not available: {e}")
    PAPER_TRADING_AVAILABLE = False


# ============ Función para ejecutar el servidor ============

def run_api(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """
    Iniciar el servidor API.

    Uso:
        from src.api import run_api
        run_api(port=8000)
    """
    import uvicorn
    uvicorn.run("src.api:app", host=host, port=port, reload=reload)


# ============ Ejemplo de uso ============

if __name__ == "__main__":
    # Ejemplo: ejecutar directamente con python -m src.api
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    logger.info("Iniciando Trad-loop API en http://localhost:8000")
    logger.info("Documentación disponible en http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
