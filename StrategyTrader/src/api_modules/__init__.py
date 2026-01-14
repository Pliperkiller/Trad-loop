"""
Módulo API para StrategyTrader.

Submódulos:
- strategy_registry: Registro de estrategias
- data_service: Servicios de datos (OHLCV, exchanges)
"""

from .strategy_registry import (
    StrategyRegistry,
    get_registry,
    register_strategy,
    unregister_strategy,
    clear_all_strategies,
    get_strategy,
)

from .data_service import (
    DataService,
    get_data_service,
    TIMEFRAME_MAP,
    TIMEFRAME_MINUTES,
)

__all__ = [
    # Strategy Registry
    "StrategyRegistry",
    "get_registry",
    "register_strategy",
    "unregister_strategy",
    "clear_all_strategies",
    "get_strategy",

    # Data Service
    "DataService",
    "get_data_service",
    "TIMEFRAME_MAP",
    "TIMEFRAME_MINUTES",
]
