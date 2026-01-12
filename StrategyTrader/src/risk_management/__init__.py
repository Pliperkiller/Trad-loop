"""
Risk Management Module

Sistema avanzado de gestion de riesgo que incluye:
- Position Sizing (Kelly, Optimal-f, Fixed Fractional, ATR-based)
- Limites de exposicion por activo/sector
- Proteccion contra drawdown
- Gestion de correlacion entre posiciones
- Calculo de Value at Risk (VaR)
"""

from .models import (
    SizingMethod,
    RiskLevel,
    DrawdownAction,
    PositionSizeResult,
    RiskMetrics,
    RiskLimit,
    TradeRiskAssessment,
    CorrelationData,
    DrawdownState,
)

from .config import (
    PositionSizingConfig,
    ExposureLimitsConfig,
    DrawdownConfig,
    CorrelationConfig,
    VaRConfig,
    RiskManagementConfig,
    CONSERVATIVE_RISK_CONFIG,
    MODERATE_RISK_CONFIG,
    AGGRESSIVE_RISK_CONFIG,
)

from .position_sizer import (
    PositionSizer,
    PositionSizerFactory,
    TradeHistory,
)

from .risk_limits import (
    Position,
    ExposureLimitManager,
    DrawdownProtection,
    VaRCalculator,
    RiskLimitChecker,
)

from .correlation_manager import (
    CorrelationManager,
    AssetReturns,
)

from .risk_manager import (
    RiskManager,
    RiskManagerFactory,
    RiskManagerState,
)

__all__ = [
    # Enums
    "SizingMethod",
    "RiskLevel",
    "DrawdownAction",
    # Models
    "PositionSizeResult",
    "RiskMetrics",
    "RiskLimit",
    "TradeRiskAssessment",
    "CorrelationData",
    "DrawdownState",
    "Position",
    "AssetReturns",
    "TradeHistory",
    # Config
    "PositionSizingConfig",
    "ExposureLimitsConfig",
    "DrawdownConfig",
    "CorrelationConfig",
    "VaRConfig",
    "RiskManagementConfig",
    "CONSERVATIVE_RISK_CONFIG",
    "MODERATE_RISK_CONFIG",
    "AGGRESSIVE_RISK_CONFIG",
    # Components
    "PositionSizer",
    "PositionSizerFactory",
    "ExposureLimitManager",
    "DrawdownProtection",
    "VaRCalculator",
    "RiskLimitChecker",
    "CorrelationManager",
    # Main
    "RiskManager",
    "RiskManagerFactory",
    "RiskManagerState",
]
