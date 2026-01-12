"""
Execution module for broker_bridge.

Contiene el executor unificado, router de simbolos y
simulador de fallback.
"""

from .symbol_router import SymbolRouter
from .unified_executor import UnifiedExecutor
from .fallback_simulator import FallbackSimulator

__all__ = [
    "SymbolRouter",
    "UnifiedExecutor",
    "FallbackSimulator",
]
