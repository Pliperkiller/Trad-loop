"""
Strategy Trader - Sistema de Trading Algorítmico
"""

__version__ = "1.0.0"
__author__ = "Strategy Trader Team"

from .strategy import (
    TradingStrategy,
    TechnicalIndicators,
    StrategyConfig,
    TradeSignal,
    Position
)

from .performance import (
    PerformanceAnalyzer,
    PerformanceVisualizer
)

from .optimizer import (
    StrategyOptimizer,
    ParameterSpace,
    OptimizationResult,
    OptimizationVisualizer
)

__all__ = [
    'TradingStrategy',
    'TechnicalIndicators',
    'StrategyConfig',
    'TradeSignal',
    'Position',
    'PerformanceAnalyzer',
    'PerformanceVisualizer',
    'StrategyOptimizer',
    'ParameterSpace',
    'OptimizationResult',
    'OptimizationVisualizer'
]
