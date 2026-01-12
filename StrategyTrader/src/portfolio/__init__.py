"""
Modulo de gestion de portfolio multi-activo.

Proporciona herramientas para:
- Asset allocation (Equal Weight, Risk Parity, Mean-Variance, etc.)
- Rebalanceo automatico (calendario o threshold)
- Backtesting multi-activo
- Metricas de portfolio (diversification ratio, contribution to return)
"""

from .models import (
    AllocationMethod,
    RebalanceFrequency,
    RebalanceReason,
    PortfolioPosition,
    AssetAllocation,
    RebalanceTrade,
    PortfolioState,
    PortfolioConfig,
    PortfolioTradeRecord,
    RebalanceEvent,
    PortfolioMetrics,
    PortfolioResult,
    PortfolioSplitResult,
    PortfolioWalkForwardResult,
)
from .allocator import PortfolioAllocator, AllocationResult, PortfolioAllocatorFactory
from .rebalancer import PortfolioRebalancer, RebalanceDecision, RebalancerFactory
from .backtester import PortfolioBacktester
from .metrics import PortfolioMetricsCalculator, DiversificationMetrics, CorrelationMetrics, ContributionMetrics
from .portfolio_manager import PortfolioManager, PortfolioManagerFactory

__all__ = [
    # Enums
    "AllocationMethod",
    "RebalanceFrequency",
    "RebalanceReason",
    # Models
    "PortfolioPosition",
    "AssetAllocation",
    "RebalanceTrade",
    "PortfolioState",
    "PortfolioConfig",
    "PortfolioTradeRecord",
    "RebalanceEvent",
    "PortfolioMetrics",
    "PortfolioResult",
    "PortfolioSplitResult",
    "PortfolioWalkForwardResult",
    # Allocator
    "PortfolioAllocator",
    "AllocationResult",
    "PortfolioAllocatorFactory",
    # Rebalancer
    "PortfolioRebalancer",
    "RebalanceDecision",
    "RebalancerFactory",
    # Backtester
    "PortfolioBacktester",
    # Metrics
    "PortfolioMetricsCalculator",
    "DiversificationMetrics",
    "CorrelationMetrics",
    "ContributionMetrics",
    # Manager
    "PortfolioManager",
    "PortfolioManagerFactory",
]
