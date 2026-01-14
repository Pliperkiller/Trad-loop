"""
Módulo de configuración para StrategyTrader.

Exporta todas las configuraciones centralizadas.
"""

from .settings import (
    # Cache
    CACHE_SETTINGS,
    CacheSettings,

    # Optimizer
    OPTIMIZER_DEFAULTS,
    OptimizerDefaults,

    # Timeframes
    TIMEFRAME_MINUTES,
    get_timeframe_minutes,

    # API
    API_DEFAULTS,
    APIDefaults,

    # Paper Trading
    PAPER_TRADING_DEFAULTS,
    PaperTradingDefaults,

    # Strategy
    STRATEGY_DEFAULTS,
    StrategyDefaults,

    # Environment
    ENV_CONFIG,
    get_env_or_default,
)

__all__ = [
    "CACHE_SETTINGS",
    "CacheSettings",
    "OPTIMIZER_DEFAULTS",
    "OptimizerDefaults",
    "TIMEFRAME_MINUTES",
    "get_timeframe_minutes",
    "API_DEFAULTS",
    "APIDefaults",
    "PAPER_TRADING_DEFAULTS",
    "PaperTradingDefaults",
    "STRATEGY_DEFAULTS",
    "StrategyDefaults",
    "ENV_CONFIG",
    "get_env_or_default",
]
