"""
Interfaces y Protocolos para StrategyTrader.

Este módulo define interfaces abstractas usando typing.Protocol para:
- Permitir dependency injection
- Facilitar testing con mocks
- Mejorar extensibilidad sin modificar clases existentes

Las interfaces son opcionales y el código existente sigue funcionando sin ellas.

Ejemplo de uso:
    from src.interfaces import IDataValidator, DefaultDataValidator

    # Usar implementación por defecto
    validator = DefaultDataValidator(strict=True)
    result = validator.validate(df)

    # O crear implementación custom
    class MyValidator:
        def validate(self, data):
            ...
        def sanitize(self, data):
            ...
"""

from .protocols import (
    # Core interfaces
    IDataValidator,
    IMetricsCalculator,
    IPositionSizer,
    ISignalGenerator,
    IRiskManager,

    # Data provider interfaces
    IDataProvider,
    IOHLCVFetcher,

    # Trading interfaces
    IOrderExecutor,
    IPositionManager,

    # Optimization interfaces
    IOptimizer,
    IObjectiveFunction,

    # Data types
    ValidationResult,
    Signal,
    PositionInfo,
    TradeResult,
    PerformanceMetrics,
)

from .implementations import (
    # Default implementations
    DefaultDataValidator,
    DefaultMetricsCalculator,
    DefaultRiskManager,

    # Position sizers
    FixedFractionalSizer,
    KellySizer,
    VolatilitySizer,
)

from .container import (
    Container,
    get_container,
    set_container,
    reset_container,
    resolve,
)

__all__ = [
    # Interfaces
    "IDataValidator",
    "IMetricsCalculator",
    "IPositionSizer",
    "ISignalGenerator",
    "IRiskManager",
    "IDataProvider",
    "IOHLCVFetcher",
    "IOrderExecutor",
    "IPositionManager",
    "IOptimizer",
    "IObjectiveFunction",

    # Data types
    "ValidationResult",
    "Signal",
    "PositionInfo",
    "TradeResult",
    "PerformanceMetrics",

    # Default implementations
    "DefaultDataValidator",
    "DefaultMetricsCalculator",
    "DefaultRiskManager",
    "FixedFractionalSizer",
    "KellySizer",
    "VolatilitySizer",

    # Container
    "Container",
    "get_container",
    "set_container",
    "reset_container",
    "resolve",
]
