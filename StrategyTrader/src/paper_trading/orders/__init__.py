"""
Paquete de ordenes avanzadas para paper trading.

Este modulo proporciona soporte para ordenes de trading profesional:
- Control de riesgo: Trailing Stop, Bracket Order
- Algoritmos de ejecucion: TWAP, VWAP, Iceberg, Hidden
- Gestion dinamica: FOK, IOC, Reduce-Only, Post-Only, Scale-In/Out
- Condicionales: If-Touched, OCO, OTOCO
"""

from .enums import (
    OrderType,
    OrderSide,
    OrderStatus,
    PositionSide,
    TimeInForce,
    TriggerType,
    TriggerDirection,
    CompositeOrderStatus,
)

from .base import (
    AdvancedOrderParams,
    OrderAction,
    AdvancedOrderInterface,
    TrailingStopState,
    AlgoOrderState,
    CompositeOrderState,
)

from .risk_control import (
    TrailingStopOrder,
    BracketOrder,
)

from .execution_algos import (
    TWAPOrder,
    VWAPOrder,
    IcebergOrder,
    HiddenOrder,
)

from .dynamic_orders import (
    FOKOrder,
    IOCOrder,
    ReduceOnlyOrder,
    PostOnlyOrder,
    ScaleOrder,
)

from .conditional_orders import (
    IfTouchedOrder,
    OCOOrder,
    OTOCOOrder,
)

__all__ = [
    # Enums
    "OrderType",
    "OrderSide",
    "OrderStatus",
    "PositionSide",
    "TimeInForce",
    "TriggerType",
    "TriggerDirection",
    "CompositeOrderStatus",
    # Base classes
    "AdvancedOrderParams",
    "OrderAction",
    "AdvancedOrderInterface",
    "TrailingStopState",
    "AlgoOrderState",
    "CompositeOrderState",
    # Risk Control
    "TrailingStopOrder",
    "BracketOrder",
    # Execution Algos
    "TWAPOrder",
    "VWAPOrder",
    "IcebergOrder",
    "HiddenOrder",
    # Dynamic Orders
    "FOKOrder",
    "IOCOrder",
    "ReduceOnlyOrder",
    "PostOnlyOrder",
    "ScaleOrder",
    # Conditional Orders
    "IfTouchedOrder",
    "OCOOrder",
    "OTOCOOrder",
]
