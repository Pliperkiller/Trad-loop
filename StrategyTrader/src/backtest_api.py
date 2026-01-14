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
import time
import hashlib
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from collections import OrderedDict
import threading

from fastapi import HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Asegurar que podemos importar los modulos de Trad-loop
TRAD_LOOP_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(TRAD_LOOP_ROOT))

from .job_manager import get_job_manager, JobType, JobStatus

# Intentar importar componentes de StrategyTrader
try:
    from .strategy import TradingStrategy, StrategyConfig
    from .strategy import (
        MovingAverageCrossoverStrategy,
        TrendFollowingEMAStrategy,
        MeanReversionLinearRegressionStrategy,
    )
    STRATEGY_AVAILABLE = True
except ImportError:
    STRATEGY_AVAILABLE = False
    logger.warning("Strategy module not available")

try:
    from .optimizer import StrategyOptimizer
    OPTIMIZER_AVAILABLE = True
except ImportError:
    OPTIMIZER_AVAILABLE = False
    logger.warning("Optimizer module not available")

# Intentar importar DataExtractor para datos OHLCV
try:
    from DataExtractor.src.infrastructure.exchanges.ccxt_adapter import CCXTAdapter
    from DataExtractor.src.domain import Timeframe
    DATA_AVAILABLE = True
except ImportError:
    DATA_AVAILABLE = False
    logger.warning("DataExtractor not available for backtest")


# ============ Cache de OHLCV ============


def _estimate_dataframe_memory(data: Any) -> int:
    """Estima el uso de memoria de un DataFrame en bytes."""
    try:
        if hasattr(data, 'memory_usage'):
            # pandas DataFrame
            return int(data.memory_usage(deep=True).sum())
        elif hasattr(data, 'nbytes'):
            # numpy array
            return int(data.nbytes)
        else:
            # Estimación genérica basada en sys.getsizeof
            import sys
            return sys.getsizeof(data)
    except Exception:
        return 0


class OHLCVCache:
    """
    Cache LRU con TTL y límite de memoria para datos OHLCV.

    Evita descargas redundantes cuando se ejecutan múltiples
    backtests/optimizaciones con los mismos datos.

    Features:
    - Límite por número de entradas (max_size)
    - Límite por memoria total (max_memory_mb)
    - TTL configurable
    - Evicción LRU cuando se exceden límites
    """

    def __init__(
        self,
        max_size: int = 20,
        ttl_seconds: int = 300,
        max_memory_mb: int = 500
    ):
        """
        Args:
            max_size: Número máximo de datasets en cache
            ttl_seconds: Tiempo de vida en segundos (default: 5 minutos)
            max_memory_mb: Límite de memoria en MB (default: 500 MB)
        """
        self._cache: OrderedDict[str, Tuple[Any, float, int]] = OrderedDict()
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._max_memory_bytes = max_memory_mb * 1024 * 1024
        self._current_memory = 0
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def _make_key(self, exchange: str, symbol: str, timeframe: str,
                  start: str, end: str) -> str:
        """Genera una clave única para el dataset"""
        key_str = f"{exchange}:{symbol}:{timeframe}:{start}:{end}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def _evict_if_needed(self, needed_bytes: int = 0) -> None:
        """Evicta entradas hasta tener espacio suficiente (debe llamarse con lock)."""
        # Evictar por tamaño
        while len(self._cache) >= self._max_size and self._cache:
            _, (_, _, mem_size) = self._cache.popitem(last=False)
            self._current_memory -= mem_size

        # Evictar por memoria
        while (self._current_memory + needed_bytes > self._max_memory_bytes
               and self._cache):
            _, (_, _, mem_size) = self._cache.popitem(last=False)
            self._current_memory -= mem_size
            logger.debug(f"Cache eviction: freed {mem_size / 1024 / 1024:.2f} MB")

    def get(self, exchange: str, symbol: str, timeframe: str,
            start: str, end: str, copy: bool = True) -> Optional[Any]:
        """
        Obtiene datos del cache si existen y no han expirado.

        Args:
            exchange: ID del exchange
            symbol: Par de trading
            timeframe: Timeframe
            start: Fecha inicio
            end: Fecha fin
            copy: Si True, retorna copia del DataFrame (default: True)
                  Si False, retorna referencia directa (más rápido, pero no modificar)

        Returns:
            DataFrame con datos OHLCV o None si no está en cache
        """
        key = self._make_key(exchange, symbol, timeframe, start, end)

        with self._lock:
            if key in self._cache:
                data, timestamp, mem_size = self._cache[key]

                # Verificar TTL
                if time.time() - timestamp < self._ttl:
                    # Mover al final (LRU)
                    self._cache.move_to_end(key)
                    self._hits += 1
                    return data.copy() if copy else data
                else:
                    # Expirado, eliminar
                    self._current_memory -= mem_size
                    del self._cache[key]
                    logger.debug(f"Cache expired: {key[:8]}...")

            self._misses += 1
            return None

    def set(self, exchange: str, symbol: str, timeframe: str,
            start: str, end: str, data: Any) -> None:
        """
        Almacena datos en el cache.
        """
        key = self._make_key(exchange, symbol, timeframe, start, end)
        mem_size = _estimate_dataframe_memory(data)

        # No cachear si el dataset es más grande que el límite total
        if mem_size > self._max_memory_bytes:
            logger.warning(
                f"Dataset too large for cache: {mem_size / 1024 / 1024:.2f} MB "
                f"> {self._max_memory_bytes / 1024 / 1024:.2f} MB limit"
            )
            return

        with self._lock:
            # Si ya existe, actualizar (liberar memoria antigua primero)
            if key in self._cache:
                _, _, old_mem_size = self._cache[key]
                self._current_memory -= old_mem_size
                self._cache.move_to_end(key)
                self._cache[key] = (data.copy(), time.time(), mem_size)
                self._current_memory += mem_size
                return

            # Evictar si es necesario
            self._evict_if_needed(mem_size)

            # Agregar nuevo
            self._cache[key] = (data.copy(), time.time(), mem_size)
            self._current_memory += mem_size

            logger.debug(
                f"Cache set: {key[:8]}... ({mem_size / 1024 / 1024:.2f} MB, "
                f"total: {self._current_memory / 1024 / 1024:.2f} MB)"
            )

    def clear(self) -> None:
        """Limpia todo el cache"""
        with self._lock:
            self._cache.clear()
            self._current_memory = 0
            self._hits = 0
            self._misses = 0
            logger.info("Cache cleared")

    def stats(self) -> Dict[str, Any]:
        """Retorna estadísticas del cache"""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = (self._hits / total * 100) if total > 0 else 0
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "memory_used_mb": round(self._current_memory / 1024 / 1024, 2),
                "max_memory_mb": self._max_memory_bytes / 1024 / 1024,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": f"{hit_rate:.1f}%",
                "ttl_seconds": self._ttl,
            }


# Instancia global del cache con límite de memoria
_ohlcv_cache = OHLCVCache(max_size=20, ttl_seconds=300, max_memory_mb=500)


def get_ohlcv_cache() -> OHLCVCache:
    """Obtiene la instancia del cache de OHLCV"""
    return _ohlcv_cache


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
    train_split: float = 0.7  # Porcentaje de datos para entrenamiento (0.0 - 1.0)


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

def fetch_ohlcv_data(exchange: str, symbol: str, timeframe: str, start: str, end: str, warmup_candles: int = 100):
    """
    Obtiene datos OHLCV para backtest.

    Utiliza cache interno para evitar descargas redundantes.

    Args:
        warmup_candles: Numero de velas adicionales a obtener antes del rango para
                        warmup de indicadores (EMA, RSI, MACD, etc.). Default: 100
    """
    if not DATA_AVAILABLE:
        raise RuntimeError("DataExtractor not available")

    # Mapear timeframe a minutos para calcular warmup
    timeframe_minutes = {
        "1m": 1,
        "5m": 5,
        "15m": 15,
        "30m": 30,
        "1h": 60,
        "2h": 120,
        "4h": 240,
        "1d": 1440,
        "1w": 10080,
    }

    start_dt = datetime.fromisoformat(start.replace('Z', '+00:00'))
    end_dt = datetime.fromisoformat(end.replace('Z', '+00:00'))

    # Calcular fecha inicio extendida para warmup de indicadores
    if warmup_candles > 0 and timeframe in timeframe_minutes:
        warmup_minutes = warmup_candles * timeframe_minutes[timeframe]
        extended_start_dt = start_dt - timedelta(minutes=warmup_minutes)
    else:
        extended_start_dt = start_dt

    # Crear key para cache con el rango extendido
    cache_start = extended_start_dt.isoformat()
    cache_end = end_dt.isoformat()

    # Intentar obtener del cache
    cache = get_ohlcv_cache()
    cached_df = cache.get(exchange, symbol, timeframe, cache_start, cache_end)

    if cached_df is not None:
        logger.debug(f"[CACHE HIT] {exchange}:{symbol}:{timeframe}")
        return cached_df

    logger.debug(f"[CACHE MISS] {exchange}:{symbol}:{timeframe} - Descargando...")

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

    adapter = CCXTAdapter(exchange)
    df = adapter.fetch_ohlcv(symbol, tf_enum, extended_start_dt, end_dt)

    # Usar timestamp como índice para que el backtest tenga los tiempos correctos
    if 'timestamp' in df.columns:
        df = df.set_index('timestamp')

    # Guardar en cache
    cache.set(exchange, symbol, timeframe, cache_start, cache_end, df)

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

    # Aplicar train/test split
    train_split = params.get("train_split", 0.7)
    if train_split < 1.0:
        split_idx = int(len(df) * train_split)
        # Usar .copy() para evitar problemas con vistas
        # y .reset_index(drop=False) para mantener timestamp como columna si es necesario
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        progress_callback(15, 100, f"Split: {len(train_df)} train / {len(test_df)} test")
    else:
        train_df = df.copy()
        test_df = None

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

    # Crear optimizador (usar datos de entrenamiento)
    optimizer = StrategyOptimizer(
        strategy_class=strategy_class,
        data=train_df,
        config_template=config_template,
        objective_metric=params["objective_metric"]
    )

    # Obtener definicion de estrategia para lookup de tipos de parametros
    strategy_def = _available_strategies.get(strategy_id, {})
    strategy_params = strategy_def.get("parameters", [])
    # Usar 'or "int"' para manejar casos donde type existe pero es None
    param_type_lookup = {p["name"]: p.get("type") or "int" for p in strategy_params}

    # Agregar parametros a optimizar
    logger.info(f"[OPTIMIZER] Configurando {len(params['parameters'])} parámetros")
    for param_def in params["parameters"]:
        param_name = param_def["name"]
        # Buscar tipo en la definicion de estrategia, luego en el request, default a int
        param_type = param_type_lookup.get(param_name) or param_def.get("type") or "int"
        low = param_def.get("min", 5)
        high = param_def.get("max", 50)
        step = param_def.get("step", 1)
        logger.debug(f"  Parámetro {param_name}: type={param_type}, min={low}, max={high}, step={step}")
        optimizer.add_parameter(
            name=param_name,
            param_type=param_type,
            low=low,
            high=high,
            step=step
        )

    # Detectar cores disponibles
    import os
    available_cores = os.cpu_count() or 4
    effective_cores = max(1, available_cores - 1)  # Dejar 1 para el sistema

    progress_callback(30, 100, f"Ejecutando optimizacion ({params['method']}) con {effective_cores} cores...")
    logger.info(f"[OPTIMIZER] Método: {params['method']}, Cores: {effective_cores}/{available_cores}")

    # Ejecutar optimizacion segun metodo
    method = params["method"]
    max_iter = params["max_iterations"]

    # Crear callback de progreso para el optimizador
    iteration_count = [0]

    def opt_progress(current, total):
        iteration_count[0] = current
        pct = 30 + int((current / max(total, 1)) * 60)
        progress_callback(pct, 100, f"Iteracion {current}/{total}")

    # Usar todos los cores disponibles (-1 = auto)
    n_jobs = -1

    if method == "grid":
        result = optimizer.grid_search(progress_callback=opt_progress)
    elif method == "random":
        result = optimizer.random_search(n_iter=max_iter, progress_callback=opt_progress, n_jobs=n_jobs)
    elif method == "bayesian":
        result = optimizer.bayesian_optimization(n_calls=max_iter, progress_callback=opt_progress, n_jobs=n_jobs)
    elif method == "genetic":
        result = optimizer.genetic_algorithm(
            population_size=min(20, max_iter // 5),
            max_generations=max_iter // 20,
            progress_callback=opt_progress,
            n_jobs=n_jobs
        )
    else:
        raise ValueError(f"Unknown optimization method: {method}")

    progress_callback(91, 100, "Procesando resultados...")

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

    # ============ VALIDACIÓN EN TEST SET ============
    test_metrics = None
    train_metrics = None

    if test_df is not None and len(test_df) > 0 and best_params:
        progress_callback(93, 100, "Validando en datos de test...")

        try:
            # Backtest en TRAIN con mejores parámetros
            logger.debug(f"train_df shape: {train_df.shape}, index type: {type(train_df.index)}")
            logger.debug(f"test_df shape: {test_df.shape}, index type: {type(test_df.index)}")
            logger.debug(f"best_params: {best_params}")

            train_config = StrategyConfig(
                symbol=params["symbol"],
                timeframe=params["timeframe"],
                initial_capital=params["initial_capital"],
                commission=0.1,
                slippage=0.05,
                risk_per_trade=2.0,
                max_positions=1,
            )
            train_strategy = strategy_class(train_config, **best_params)
            logger.debug("Loading train data...")
            train_strategy.load_data(train_df)
            logger.debug("Running train backtest...")
            train_strategy.backtest()
            logger.debug("Getting train metrics...")
            train_metrics_raw = train_strategy.get_performance_metrics()
            train_metrics = sanitize_dict(train_metrics_raw)
            logger.debug(f"Train metrics: {train_metrics}")

            progress_callback(96, 100, "Ejecutando backtest en test...")

            # Backtest en TEST con mejores parámetros
            test_config = StrategyConfig(
                symbol=params["symbol"],
                timeframe=params["timeframe"],
                initial_capital=params["initial_capital"],
                commission=0.1,
                slippage=0.05,
                risk_per_trade=2.0,
                max_positions=1,
            )
            test_strategy = strategy_class(test_config, **best_params)
            logger.debug("Loading test data...")
            test_strategy.load_data(test_df)
            logger.debug("Running test backtest...")
            test_strategy.backtest()
            logger.debug("Getting test metrics...")
            test_metrics_raw = test_strategy.get_performance_metrics()
            test_metrics = sanitize_dict(test_metrics_raw)
            logger.debug(f"Test metrics: {test_metrics}")

        except Exception as e:
            import traceback
            logger.error(f"Error en validación test: {e}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            test_metrics = {"error": str(e)}
            train_metrics = {"error": str(e)}

    progress_callback(100, 100, "Optimizacion completada")

    # Obtener iteraciones del resultado o del contador
    iterations = iteration_count[0]
    if hasattr(result, 'iterations') and result.iterations > 0:
        iterations = result.iterations
    elif len(all_results) > 0:
        iterations = len(all_results)

    return {
        "best_params": best_params,
        "best_score": best_score,
        "method": method,
        "iterations": iterations,
        "all_results": all_results,
        "objective_metric": params["objective_metric"],
        "train_split": train_split,
        "train_samples": len(train_df),
        "test_samples": len(test_df) if test_df is not None else 0,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
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

    # Registrar estrategias disponibles
    if STRATEGY_AVAILABLE:
        try:
            register_strategy_class(
                strategy_id="ma_crossover",
                strategy_class=MovingAverageCrossoverStrategy,
                name="Moving Average Crossover",
                description="Estrategia basada en cruce de medias moviles con filtro RSI",
                parameters=[
                    {
                        "name": "fast_period",
                        "type": "int",
                        "default": 10,
                        "min_value": 5,
                        "max_value": 50,
                        "step": 1,
                        "description": "Periodo de la EMA rapida"
                    },
                    {
                        "name": "slow_period",
                        "type": "int",
                        "default": 30,
                        "min_value": 20,
                        "max_value": 200,
                        "step": 5,
                        "description": "Periodo de la EMA lenta"
                    },
                    {
                        "name": "rsi_period",
                        "type": "int",
                        "default": 14,
                        "min_value": 5,
                        "max_value": 50,
                        "step": 1,
                        "description": "Periodo del RSI"
                    },
                    {
                        "name": "rsi_lower_bound",
                        "type": "int",
                        "default": 30,
                        "min_value": 10,
                        "max_value": 50,
                        "step": 5,
                        "description": "Limite inferior RSI para BUY"
                    },
                    {
                        "name": "rsi_upper_bound",
                        "type": "int",
                        "default": 70,
                        "min_value": 50,
                        "max_value": 90,
                        "step": 5,
                        "description": "Limite superior RSI para BUY"
                    },
                    {
                        "name": "rsi_sell_threshold",
                        "type": "int",
                        "default": 80,
                        "min_value": 70,
                        "max_value": 95,
                        "step": 5,
                        "description": "Umbral RSI para SELL forzado"
                    },
                ]
            )
        except ImportError:
            pass

        # Registrar estrategia Trend Following EMA
        try:
            register_strategy_class(
                strategy_id="trend_following_ema",
                strategy_class=TrendFollowingEMAStrategy,
                name="Trend Following EMA",
                description="Estrategia de seguimiento de tendencia con Triple EMA (21/55/200) y filtro ADX. Soporta entradas por cruce o pullback.",
                parameters=[
                    {
                        "name": "ema_fast",
                        "type": "int",
                        "default": 21,
                        "min_value": 5,
                        "max_value": 50,
                        "step": 1,
                        "description": "Periodo de la EMA rapida (tendencia corta)"
                    },
                    {
                        "name": "ema_medium",
                        "type": "int",
                        "default": 55,
                        "min_value": 20,
                        "max_value": 100,
                        "step": 5,
                        "description": "Periodo de la EMA media (tendencia media)"
                    },
                    {
                        "name": "ema_slow",
                        "type": "int",
                        "default": 200,
                        "min_value": 100,
                        "max_value": 300,
                        "step": 10,
                        "description": "Periodo de la EMA lenta (tendencia principal)"
                    },
                    {
                        "name": "adx_period",
                        "type": "int",
                        "default": 14,
                        "min_value": 7,
                        "max_value": 28,
                        "step": 1,
                        "description": "Periodo del ADX"
                    },
                    {
                        "name": "adx_threshold",
                        "type": "int",
                        "default": 25,
                        "min_value": 15,
                        "max_value": 40,
                        "step": 5,
                        "description": "Umbral minimo de ADX para confirmar tendencia"
                    },
                    {
                        "name": "atr_period",
                        "type": "int",
                        "default": 14,
                        "min_value": 7,
                        "max_value": 28,
                        "step": 1,
                        "description": "Periodo del ATR para calculo de stop loss"
                    },
                    {
                        "name": "max_distance_pct",
                        "type": "float",
                        "default": 5.0,
                        "min_value": 1.0,
                        "max_value": 10.0,
                        "step": 0.5,
                        "description": "Distancia maxima (%) del precio a EMA para evitar sobre-extension"
                    },
                    {
                        "name": "atr_sl_multiplier",
                        "type": "float",
                        "default": 0.5,
                        "min_value": 0.1,
                        "max_value": 2.0,
                        "step": 0.1,
                        "description": "Multiplicador ATR para stop loss"
                    },
                    {
                        "name": "tp1_ratio",
                        "type": "float",
                        "default": 1.5,
                        "min_value": 1.0,
                        "max_value": 3.0,
                        "step": 0.5,
                        "description": "Ratio riesgo:beneficio para Take Profit 1"
                    },
                    {
                        "name": "tp2_ratio",
                        "type": "float",
                        "default": 2.5,
                        "min_value": 1.5,
                        "max_value": 5.0,
                        "step": 0.5,
                        "description": "Ratio riesgo:beneficio para Take Profit 2"
                    },
                ]
            )
        except ImportError:
            pass

        # Registrar estrategia Mean Reversion Linear Regression
        try:
            register_strategy_class(
                strategy_id="mean_reversion_lr",
                strategy_class=MeanReversionLinearRegressionStrategy,
                name="Mean Reversion Linear Regression",
                description="Estrategia de reversion a la media usando regresion lineal. Opera cuando el precio se desvia significativamente de la linea de regresion en mercados laterales (R² bajo).",
                parameters=[
                    {
                        "name": "lr_period",
                        "type": "int",
                        "default": 20,
                        "min_value": 10,
                        "max_value": 50,
                        "step": 5,
                        "description": "Periodos para calcular regresion lineal"
                    },
                    {
                        "name": "entry_zscore",
                        "type": "float",
                        "default": 2.0,
                        "min_value": 1.5,
                        "max_value": 3.0,
                        "step": 0.25,
                        "description": "Z-score minimo para entrar (desviacion de la linea)"
                    },
                    {
                        "name": "exit_zscore",
                        "type": "float",
                        "default": 0.5,
                        "min_value": 0.0,
                        "max_value": 1.0,
                        "step": 0.1,
                        "description": "Z-score para salir (precio volvio a la linea)"
                    },
                    {
                        "name": "max_r2",
                        "type": "float",
                        "default": 0.4,
                        "min_value": 0.2,
                        "max_value": 0.6,
                        "step": 0.05,
                        "description": "Maximo R² permitido (filtra mercados en tendencia)"
                    },
                    {
                        "name": "max_slope_pct",
                        "type": "float",
                        "default": 0.3,
                        "min_value": 0.1,
                        "max_value": 0.5,
                        "step": 0.05,
                        "description": "Maxima pendiente normalizada (%) permitida"
                    },
                    {
                        "name": "atr_period",
                        "type": "int",
                        "default": 14,
                        "min_value": 7,
                        "max_value": 28,
                        "step": 1,
                        "description": "Periodo del ATR para stop loss"
                    },
                    {
                        "name": "atr_sl_multiplier",
                        "type": "float",
                        "default": 0.5,
                        "min_value": 0.1,
                        "max_value": 2.0,
                        "step": 0.1,
                        "description": "Multiplicador ATR para stop loss"
                    },
                    {
                        "name": "tp1_ratio",
                        "type": "float",
                        "default": 1.5,
                        "min_value": 1.0,
                        "max_value": 3.0,
                        "step": 0.5,
                        "description": "Ratio riesgo:beneficio para Take Profit"
                    },
                    {
                        "name": "require_candle_confirmation",
                        "type": "bool",
                        "default": False,
                        "description": "Requerir vela de confirmacion para entrar (puede reducir senales)"
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

    # ============ Endpoints de Cache ============

    @app.get("/api/v1/cache/stats")
    async def get_cache_stats():
        """Obtiene estadísticas del cache de OHLCV"""
        cache = get_ohlcv_cache()
        return cache.stats()

    @app.post("/api/v1/cache/clear")
    async def clear_cache():
        """Limpia el cache de OHLCV"""
        cache = get_ohlcv_cache()
        cache.clear()
        return {"message": "Cache cleared", "stats": cache.stats()}

    return app
