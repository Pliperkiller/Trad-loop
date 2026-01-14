"""
Configuración centralizada para StrategyTrader.

Este módulo contiene todas las constantes y valores por defecto
que anteriormente estaban hardcodeados en diferentes módulos.

Uso:
    from src.config.settings import (
        CACHE_SETTINGS,
        OPTIMIZER_DEFAULTS,
        TIMEFRAME_MINUTES,
    )
"""

import os
from dataclasses import dataclass
from typing import Dict, Any, Optional


# ============ Cache Settings ============

@dataclass(frozen=True)
class CacheSettings:
    """Configuración del cache OHLCV."""
    max_size: int = 20
    ttl_seconds: int = 300
    max_memory_mb: int = 500


CACHE_SETTINGS = CacheSettings()


# ============ Optimizer Defaults ============

@dataclass(frozen=True)
class OptimizerDefaults:
    """Valores por defecto para optimizadores."""
    # Bayesian
    bayesian_n_calls: int = 50
    bayesian_n_initial_points: int = 10

    # Genetic
    genetic_population_size: int = 20
    genetic_max_generations: int = 50
    genetic_mutation_min: float = 0.5
    genetic_mutation_max: float = 1.0
    genetic_recombination: float = 0.7

    # Random Search
    random_n_iterations: int = 100

    # Grid Search
    grid_max_combinations: int = 10000

    # Walk Forward
    walk_forward_n_splits: int = 5
    walk_forward_test_size: float = 0.2

    # General
    default_n_jobs: int = -1  # -1 = usar todos los cores


OPTIMIZER_DEFAULTS = OptimizerDefaults()


# ============ Timeframe Mappings ============

# Mapeo de string a minutos
TIMEFRAME_MINUTES: Dict[str, int] = {
    "1m": 1,
    "3m": 3,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "2h": 120,
    "4h": 240,
    "6h": 360,
    "8h": 480,
    "12h": 720,
    "1d": 1440,
    "3d": 4320,
    "1w": 10080,
}


def get_timeframe_minutes(timeframe: str) -> int:
    """
    Obtiene los minutos para un timeframe dado.

    Args:
        timeframe: String del timeframe (ej: "1h", "4h", "1d")

    Returns:
        Minutos del timeframe

    Raises:
        ValueError: Si el timeframe no es válido
    """
    if timeframe not in TIMEFRAME_MINUTES:
        raise ValueError(f"Timeframe inválido: {timeframe}. Válidos: {list(TIMEFRAME_MINUTES.keys())}")
    return TIMEFRAME_MINUTES[timeframe]


# ============ API Defaults ============

@dataclass(frozen=True)
class APIDefaults:
    """Valores por defecto para la API."""
    host: str = "0.0.0.0"
    port: int = 8000
    warmup_candles: int = 100
    max_candles_per_request: int = 10000


API_DEFAULTS = APIDefaults()


# ============ Paper Trading Defaults ============

@dataclass(frozen=True)
class PaperTradingDefaults:
    """Valores por defecto para paper trading."""
    initial_balance: float = 10000.0
    max_positions: int = 5
    default_stop_loss_pct: float = 0.02  # 2%
    default_take_profit_pct: float = 0.04  # 4%
    commission_pct: float = 0.1  # 0.1%
    slippage_pct: float = 0.05  # 0.05%
    max_drawdown_pct: float = 0.25  # 25%


PAPER_TRADING_DEFAULTS = PaperTradingDefaults()


# ============ Strategy Defaults ============

@dataclass(frozen=True)
class StrategyDefaults:
    """Valores por defecto para estrategias."""
    risk_per_trade: float = 2.0  # 2%
    max_positions: int = 1
    commission: float = 0.1  # 0.1%
    slippage: float = 0.05  # 0.05%


STRATEGY_DEFAULTS = StrategyDefaults()


# ============ Environment Variables ============

def get_env_or_default(key: str, default: Any, cast_type: type = str) -> Any:
    """
    Obtiene un valor de variable de entorno o usa el default.

    Args:
        key: Nombre de la variable de entorno
        default: Valor por defecto
        cast_type: Tipo al que convertir el valor

    Returns:
        Valor de la variable de entorno o el default
    """
    value = os.environ.get(key)
    if value is None:
        return default
    try:
        if cast_type == bool:
            return value.lower() in ('true', '1', 'yes')
        return cast_type(value)
    except (ValueError, TypeError):
        return default


# Configuración desde variables de entorno
ENV_CONFIG = {
    "debug": get_env_or_default("STRATEGYTRADER_DEBUG", False, bool),
    "log_level": get_env_or_default("STRATEGYTRADER_LOG_LEVEL", "INFO", str),
    "cache_max_memory_mb": get_env_or_default("STRATEGYTRADER_CACHE_MB", 500, int),
}
