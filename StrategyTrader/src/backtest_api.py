"""
API de Backtest y Optimizacion para Trad-loop

Proporciona endpoints para:
- Listar estrategias disponibles
- Ejecutar backtests
- Ejecutar optimizaciones
- Consultar progreso y resultados

Uso:
    from src.backtest_api import register_backtest_routes
    register_backtest_routes(app)
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass

from fastapi import HTTPException
from pydantic import BaseModel

# Asegurar que podemos importar los modulos de Trad-loop
TRAD_LOOP_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(TRAD_LOOP_ROOT))

from .job_manager import get_job_manager, JobType, JobStatus

# Intentar importar componentes de StrategyTrader
try:
    from .strategy import TradingStrategy, StrategyConfig
    STRATEGY_AVAILABLE = True
except ImportError:
    STRATEGY_AVAILABLE = False
    print("Warning: Strategy module not available")

try:
    from .optimizer import StrategyOptimizer
    OPTIMIZER_AVAILABLE = True
except ImportError:
    OPTIMIZER_AVAILABLE = False
    print("Warning: Optimizer module not available")

# Intentar importar DataExtractor para datos OHLCV
try:
    from DataExtractor.src.infrastructure.exchanges.ccxt_adapter import CCXTAdapter
    from DataExtractor.src.domain import Timeframe
    DATA_AVAILABLE = True
except ImportError:
    DATA_AVAILABLE = False
    print("Warning: DataExtractor not available for backtest")


# ============ Modelos Pydantic ============

class StrategyParameter(BaseModel):
    """Definicion de un parametro de estrategia"""
    name: str
    type: str  # 'int', 'float', 'bool', 'str'
    default: Any
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    step: Optional[float] = None
    description: str = ""


class StrategyDefinition(BaseModel):
    """Definicion de una estrategia disponible"""
    id: str
    name: str
    description: str
    parameters: List[StrategyParameter]


class BacktestRequest(BaseModel):
    """Request para ejecutar un backtest"""
    strategy_id: str
    exchange: str
    symbol: str
    timeframe: str
    start_date: str  # ISO format
    end_date: str    # ISO format
    initial_capital: float = 10000.0
    commission: float = 0.1
    slippage: float = 0.05
    parameters: Dict[str, Any] = {}


class BacktestJobResponse(BaseModel):
    """Response al crear un job de backtest"""
    job_id: str
    status: str
    message: str


class OptimizationRequest(BaseModel):
    """Request para ejecutar una optimizacion"""
    strategy_id: str
    exchange: str
    symbol: str
    timeframe: str
    start_date: str
    end_date: str
    initial_capital: float = 10000.0
    method: str = "grid"  # grid, random, bayesian, genetic
    objective_metric: str = "sharpe_ratio"
    parameters: List[Dict[str, Any]]  # [{name, min, max, step}]
    max_iterations: int = 100


class JobStatusResponse(BaseModel):
    """Response con estado de un job"""
    job_id: str
    type: str
    status: str
    progress: Dict[str, Any]
    result: Optional[Dict[str, Any]] = None
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


# ============ Registro de Estrategias ============

# Estrategias disponibles para backtest/optimizacion
_available_strategies: Dict[str, Dict[str, Any]] = {}


def register_strategy_class(
    strategy_id: str,
    strategy_class: type,
    name: str,
    description: str,
    parameters: List[Dict[str, Any]]
):
    """
    Registra una clase de estrategia para que este disponible via API.

    Args:
        strategy_id: ID unico de la estrategia
        strategy_class: Clase de la estrategia (debe heredar de TradingStrategy)
        name: Nombre para mostrar
        description: Descripcion de la estrategia
        parameters: Lista de parametros configurables
    """
    _available_strategies[strategy_id] = {
        "class": strategy_class,
        "name": name,
        "description": description,
        "parameters": parameters
    }


def get_strategy_class(strategy_id: str):
    """Obtiene la clase de estrategia por ID"""
    if strategy_id not in _available_strategies:
        return None
    return _available_strategies[strategy_id]["class"]


# ============ Funciones de Ejecucion ============

def fetch_ohlcv_data(exchange: str, symbol: str, timeframe: str, start: str, end: str):
    """Obtiene datos OHLCV para backtest"""
    if not DATA_AVAILABLE:
        raise RuntimeError("DataExtractor not available")

    # Mapear timeframe string a enum
    timeframe_map = {
        "1m": Timeframe.ONE_MINUTE,
        "5m": Timeframe.FIVE_MINUTES,
        "15m": Timeframe.FIFTEEN_MINUTES,
        "30m": Timeframe.THIRTY_MINUTES,
        "1h": Timeframe.ONE_HOUR,
        "2h": Timeframe.TWO_HOURS,
        "4h": Timeframe.FOUR_HOURS,
        "1d": Timeframe.ONE_DAY,
        "1w": Timeframe.ONE_WEEK,
    }

    tf_enum = timeframe_map.get(timeframe)
    if not tf_enum:
        raise ValueError(f"Timeframe {timeframe} not supported")

    start_dt = datetime.fromisoformat(start.replace('Z', '+00:00'))
    end_dt = datetime.fromisoformat(end.replace('Z', '+00:00'))

    adapter = CCXTAdapter(exchange)
    df = adapter.fetch_ohlcv(symbol, tf_enum, start_dt, end_dt)

    # Usar timestamp como índice para que el backtest tenga los tiempos correctos
    if 'timestamp' in df.columns:
        df = df.set_index('timestamp')

    return df


def run_backtest(params: Dict[str, Any], progress_callback: Callable) -> Dict[str, Any]:
    """
    Ejecuta un backtest.

    Esta funcion se ejecuta en un thread pool.
    """
    progress_callback(0, 100, "Iniciando backtest...")

    strategy_id = params["strategy_id"]
    strategy_class = get_strategy_class(strategy_id)

    if not strategy_class:
        raise ValueError(f"Strategy {strategy_id} not found")

    progress_callback(10, 100, "Obteniendo datos OHLCV...")

    # Obtener datos
    df = fetch_ohlcv_data(
        params["exchange"],
        params["symbol"],
        params["timeframe"],
        params["start_date"],
        params["end_date"]
    )

    if df.empty:
        raise ValueError("No data available for the specified range")

    progress_callback(30, 100, "Configurando estrategia...")

    # Crear configuracion
    config = StrategyConfig(
        symbol=params["symbol"],
        timeframe=params["timeframe"],
        initial_capital=params["initial_capital"],
        commission=params["commission"],
        slippage=params["slippage"],
        risk_per_trade=2.0,
        max_positions=1,
    )

    # Crear instancia de estrategia con parametros
    strategy_params = params.get("parameters", {})
    strategy = strategy_class(config, **strategy_params)

    progress_callback(40, 100, "Cargando datos...")

    # Cargar datos
    strategy.load_data(df)

    progress_callback(50, 100, "Ejecutando backtest...")

    # Ejecutar backtest
    strategy.backtest()

    progress_callback(90, 100, "Calculando metricas...")

    # Obtener resultados
    metrics = strategy.get_performance_metrics()
    trades = strategy.closed_trades
    equity_curve = strategy.equity_curve

    # Convertir trades a formato serializable
    trades_data = []
    for i, t in enumerate(trades):
        entry_time = t['entry_time']
        exit_time = t['exit_time']

        if hasattr(entry_time, 'isoformat'):
            entry_time = entry_time.isoformat()
        if hasattr(exit_time, 'isoformat'):
            exit_time = exit_time.isoformat()

        trades_data.append({
            "id": i,
            "entry_time": entry_time,
            "exit_time": exit_time,
            "entry_price": t['entry_price'],
            "exit_price": t['exit_price'],
            "quantity": t['quantity'],
            "position_type": t.get('position_type', 'LONG'),
            "pnl": t['pnl'],
            "return_pct": t['return_pct'],
            "reason": t['reason'],
        })

    progress_callback(100, 100, "Backtest completado")

    return {
        "metrics": metrics,
        "trades": trades_data,
        "equity_curve": equity_curve,
        "total_trades": len(trades),
        "data_points": len(df),
    }


def run_optimization(params: Dict[str, Any], progress_callback: Callable) -> Dict[str, Any]:
    """
    Ejecuta una optimizacion de parametros.

    Esta funcion se ejecuta en un thread pool.
    """
    if not OPTIMIZER_AVAILABLE:
        raise RuntimeError("Optimizer not available")

    progress_callback(0, 100, "Iniciando optimizacion...")

    strategy_id = params["strategy_id"]
    strategy_class = get_strategy_class(strategy_id)

    if not strategy_class:
        raise ValueError(f"Strategy {strategy_id} not found")

    progress_callback(10, 100, "Obteniendo datos OHLCV...")

    # Obtener datos
    df = fetch_ohlcv_data(
        params["exchange"],
        params["symbol"],
        params["timeframe"],
        params["start_date"],
        params["end_date"]
    )

    if df.empty:
        raise ValueError("No data available for the specified range")

    progress_callback(20, 100, "Configurando optimizador...")

    # Crear configuracion base
    config_template = {
        "symbol": params["symbol"],
        "timeframe": params["timeframe"],
        "initial_capital": params["initial_capital"],
        "commission": 0.1,
        "slippage": 0.05,
        "risk_per_trade": 2.0,
        "max_positions": 1,
    }

    # Crear optimizador
    optimizer = StrategyOptimizer(
        strategy_class=strategy_class,
        data=df,
        config_template=config_template,
        objective_metric=params["objective_metric"]
    )

    # Agregar parametros a optimizar
    for param_def in params["parameters"]:
        optimizer.add_parameter(
            name=param_def["name"],
            param_type=param_def.get("type", "int"),
            low=param_def.get("min", 5),
            high=param_def.get("max", 50),
            step=param_def.get("step", 1)
        )

    progress_callback(30, 100, f"Ejecutando optimizacion ({params['method']})...")

    # Ejecutar optimizacion segun metodo
    method = params["method"]
    max_iter = params["max_iterations"]

    # Crear callback de progreso para el optimizador
    iteration_count = [0]

    def opt_progress(current, total):
        iteration_count[0] = current
        pct = 30 + int((current / max(total, 1)) * 60)
        progress_callback(pct, 100, f"Iteracion {current}/{total}")

    if method == "grid":
        result = optimizer.grid_search()
    elif method == "random":
        result = optimizer.random_search(n_iter=max_iter)
    elif method == "bayesian":
        result = optimizer.bayesian_optimization(n_calls=max_iter)
    elif method == "genetic":
        result = optimizer.genetic_algorithm(
            population_size=min(20, max_iter // 5),
            max_generations=max_iter // 20
        )
    else:
        raise ValueError(f"Unknown optimization method: {method}")

    progress_callback(95, 100, "Procesando resultados...")

    # Extraer resultados
    best_params = result.best_params if hasattr(result, 'best_params') else {}
    best_score = result.best_score if hasattr(result, 'best_score') else 0
    all_results_raw = result.all_results if hasattr(result, 'all_results') else []

    # Convertir DataFrame a lista de diccionarios si es necesario
    import pandas as pd
    import math

    def sanitize_value(v):
        """Reemplaza inf/-inf/nan con None para JSON"""
        if isinstance(v, float):
            if math.isnan(v) or math.isinf(v):
                return None
        return v

    def sanitize_dict(d):
        """Sanitiza todos los valores de un diccionario"""
        return {k: sanitize_value(v) for k, v in d.items()}

    if isinstance(all_results_raw, pd.DataFrame):
        # Reemplazar inf/nan antes de convertir
        all_results_raw = all_results_raw.replace([float('inf'), float('-inf')], None)
        all_results = all_results_raw.head(100).to_dict('records')
    elif isinstance(all_results_raw, list):
        all_results = [sanitize_dict(r) if isinstance(r, dict) else r for r in all_results_raw[:100]]
    else:
        all_results = []

    # Sanitizar best_score
    best_score = sanitize_value(best_score) or 0

    progress_callback(100, 100, "Optimizacion completada")

    return {
        "best_params": best_params,
        "best_score": best_score,
        "method": method,
        "iterations": iteration_count[0],
        "all_results": all_results,
        "objective_metric": params["objective_metric"],
    }


# ============ Registro de Rutas ============

def register_backtest_routes(app):
    """
    Registra las rutas de backtest y optimizacion en la app FastAPI.

    Llamar desde api.py:
        from src.backtest_api import register_backtest_routes
        register_backtest_routes(app)
    """

    job_manager = get_job_manager()

    # Registrar estrategia de ejemplo si esta disponible
    if STRATEGY_AVAILABLE:
        try:
            from .strategy import MovingAverageCrossoverStrategy

            register_strategy_class(
                strategy_id="ma_crossover",
                strategy_class=MovingAverageCrossoverStrategy,
                name="Moving Average Crossover",
                description="Estrategia basada en cruce de medias moviles",
                parameters=[
                    {
                        "name": "fast_period",
                        "type": "int",
                        "default": 10,
                        "min_value": 5,
                        "max_value": 50,
                        "step": 1,
                        "description": "Periodo de la media rapida"
                    },
                    {
                        "name": "slow_period",
                        "type": "int",
                        "default": 30,
                        "min_value": 20,
                        "max_value": 200,
                        "step": 5,
                        "description": "Periodo de la media lenta"
                    },
                ]
            )
        except ImportError:
            pass

    @app.get("/api/v1/strategies/available")
    async def list_available_strategies() -> List[StrategyDefinition]:
        """Lista todas las estrategias disponibles para backtest"""
        strategies = []
        for strategy_id, info in _available_strategies.items():
            strategies.append(StrategyDefinition(
                id=strategy_id,
                name=info["name"],
                description=info["description"],
                parameters=[StrategyParameter(**p) for p in info["parameters"]]
            ))
        return strategies

    @app.post("/api/v1/backtest/run")
    async def start_backtest(request: BacktestRequest) -> BacktestJobResponse:
        """
        Inicia un backtest asincrono.

        El backtest se ejecuta en segundo plano. Use GET /api/v1/backtest/{job_id}
        para consultar el progreso y resultados.
        """
        if not STRATEGY_AVAILABLE:
            raise HTTPException(status_code=503, detail="Strategy module not available")

        if request.strategy_id not in _available_strategies:
            raise HTTPException(
                status_code=400,
                detail=f"Strategy '{request.strategy_id}' not found. Available: {list(_available_strategies.keys())}"
            )

        # Crear job
        job_id = await job_manager.create_job(
            job_type=JobType.BACKTEST,
            params=request.model_dump(),
            executor=run_backtest,
            run_sync=True
        )

        return BacktestJobResponse(
            job_id=job_id,
            status="queued",
            message=f"Backtest job created. Use GET /api/v1/backtest/{job_id} to check status."
        )

    @app.get("/api/v1/backtest/{job_id}")
    async def get_backtest_status(job_id: str) -> JobStatusResponse:
        """Obtiene el estado y resultados de un backtest"""
        job = job_manager.get_job(job_id)

        if not job:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

        if job.type != JobType.BACKTEST:
            raise HTTPException(status_code=400, detail=f"Job {job_id} is not a backtest")

        job_dict = job.to_dict()

        return JobStatusResponse(
            job_id=job_dict["id"],
            type=job_dict["type"],
            status=job_dict["status"],
            progress=job_dict["progress"],
            result=job_dict["result"],
            created_at=job_dict["created_at"],
            started_at=job_dict["started_at"],
            completed_at=job_dict["completed_at"],
        )

    @app.post("/api/v1/backtest/{job_id}/cancel")
    async def cancel_backtest(job_id: str) -> Dict[str, Any]:
        """Cancela un backtest en ejecucion"""
        job = job_manager.get_job(job_id)

        if not job:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

        success = await job_manager.cancel_job(job_id)

        return {
            "job_id": job_id,
            "cancelled": success,
            "message": "Job cancelled" if success else "Job could not be cancelled (may have already completed)"
        }

    @app.get("/api/v1/backtest")
    async def list_backtests() -> List[JobStatusResponse]:
        """Lista todos los jobs de backtest"""
        jobs = job_manager.get_all_jobs(JobType.BACKTEST)

        return [
            JobStatusResponse(
                job_id=j.to_dict()["id"],
                type=j.to_dict()["type"],
                status=j.to_dict()["status"],
                progress=j.to_dict()["progress"],
                result=j.to_dict()["result"],
                created_at=j.to_dict()["created_at"],
                started_at=j.to_dict()["started_at"],
                completed_at=j.to_dict()["completed_at"],
            )
            for j in jobs[:20]  # Limitar a ultimos 20
        ]

    # ============ Endpoints de Optimizacion ============

    @app.post("/api/v1/optimize/run")
    async def start_optimization(request: OptimizationRequest) -> BacktestJobResponse:
        """
        Inicia una optimizacion asincrona.

        Metodos disponibles: grid, random, bayesian, genetic
        """
        if not OPTIMIZER_AVAILABLE:
            raise HTTPException(status_code=503, detail="Optimizer not available")

        if request.strategy_id not in _available_strategies:
            raise HTTPException(
                status_code=400,
                detail=f"Strategy '{request.strategy_id}' not found"
            )

        if request.method not in ["grid", "random", "bayesian", "genetic"]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid method '{request.method}'. Available: grid, random, bayesian, genetic"
            )

        # Crear job
        job_id = await job_manager.create_job(
            job_type=JobType.OPTIMIZATION,
            params=request.model_dump(),
            executor=run_optimization,
            run_sync=True
        )

        return BacktestJobResponse(
            job_id=job_id,
            status="queued",
            message=f"Optimization job created. Use GET /api/v1/optimize/{job_id} to check status."
        )

    @app.get("/api/v1/optimize/{job_id}")
    async def get_optimization_status(job_id: str) -> JobStatusResponse:
        """Obtiene el estado y resultados de una optimizacion"""
        job = job_manager.get_job(job_id)

        if not job:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

        if job.type != JobType.OPTIMIZATION:
            raise HTTPException(status_code=400, detail=f"Job {job_id} is not an optimization")

        job_dict = job.to_dict()

        return JobStatusResponse(
            job_id=job_dict["id"],
            type=job_dict["type"],
            status=job_dict["status"],
            progress=job_dict["progress"],
            result=job_dict["result"],
            created_at=job_dict["created_at"],
            started_at=job_dict["started_at"],
            completed_at=job_dict["completed_at"],
        )

    @app.post("/api/v1/optimize/{job_id}/cancel")
    async def cancel_optimization(job_id: str) -> Dict[str, Any]:
        """Cancela una optimizacion en ejecucion"""
        job = job_manager.get_job(job_id)

        if not job:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

        success = await job_manager.cancel_job(job_id)

        return {
            "job_id": job_id,
            "cancelled": success,
        }

    @app.get("/api/v1/optimize")
    async def list_optimizations() -> List[JobStatusResponse]:
        """Lista todos los jobs de optimizacion"""
        jobs = job_manager.get_all_jobs(JobType.OPTIMIZATION)

        return [
            JobStatusResponse(
                job_id=j.to_dict()["id"],
                type=j.to_dict()["type"],
                status=j.to_dict()["status"],
                progress=j.to_dict()["progress"],
                result=j.to_dict()["result"],
                created_at=j.to_dict()["created_at"],
                started_at=j.to_dict()["started_at"],
                completed_at=j.to_dict()["completed_at"],
            )
            for j in jobs[:20]
        ]

    # ============ Endpoint de Jobs General ============

    @app.get("/api/v1/jobs")
    async def list_all_jobs() -> List[JobStatusResponse]:
        """Lista todos los jobs (backtests y optimizaciones)"""
        jobs = job_manager.get_all_jobs()

        return [
            JobStatusResponse(
                job_id=j.to_dict()["id"],
                type=j.to_dict()["type"],
                status=j.to_dict()["status"],
                progress=j.to_dict()["progress"],
                result=j.to_dict()["result"],
                created_at=j.to_dict()["created_at"],
                started_at=j.to_dict()["started_at"],
                completed_at=j.to_dict()["completed_at"],
            )
            for j in jobs[:50]
        ]

    return app
