"""
Paper Trading Module

Sistema de simulacion de trading en tiempo real.
Permite validar estrategias con datos de mercado en vivo
sin arriesgar capital real.

Componentes principales:
- PaperTradingEngine: Motor principal de simulacion
- RealtimeFeedManager: Gestor de feeds WebSocket
- OrderSimulator: Simulador de ordenes
- PositionManager: Gestor de posiciones
- RealtimePerformanceTracker: Metricas en tiempo real
"""

from .models import (
    OrderType,
    OrderSide,
    OrderStatus,
    PositionSide,
    Order,
    OrderResult,
    PaperPosition,
    PaperTradingState,
    RealtimeCandle,
    TradeRecord,
)
from .config import PaperTradingConfig, SlippageModel, CommissionModel

__all__ = [
    # Enums
    "OrderType",
    "OrderSide",
    "OrderStatus",
    "PositionSide",
    "SlippageModel",
    "CommissionModel",
    # Dataclasses
    "Order",
    "OrderResult",
    "PaperPosition",
    "PaperTradingState",
    "RealtimeCandle",
    "TradeRecord",
    # Config
    "PaperTradingConfig",
]
