"""
Modulo de optimizers para estrategias de trading.

Incluye:
- Metodos de optimizacion: grid, random, bayesian, genetic
- Walk-forward analysis para validacion IS/OOS
- Modulo de validacion con splitters temporales
- Analisis de estabilidad de parametros y deteccion de overfitting
"""

from .optimization_types import ParameterSpace, OptimizationResult, WalkForwardResult
from .grid_search import grid_search
from .random_search import random_search
from .bayesian import bayesian_optimization
from .genetic import genetic_algorithm
from .walk_forward import (
    walk_forward_optimization,
    create_walk_forward_splitter,
    convert_to_validation_result,
)

# Validation submodule
from .validation import (
    # Splitters
    BaseSplitter,
    SplitInfo,
    TimeSeriesSplit,
    RollingWindowSplit,
    ExpandingWindowSplit,
    TrainValTestSplit,
    WalkForwardSplit,
    create_splitter,
    # Results
    SplitResult,
    ParameterStability,
    CrossValidationResult,
    TrainValTestResult,
    calculate_robustness_score,
    # Cross-Validation
    TimeSeriesCV,
    WalkForwardCV,
    train_val_test_evaluate,
    # Purged K-Fold
    PurgedKFold,
    PurgedSplitInfo,
    CombinatorialPurgedCV,
    calculate_optimal_purge_gap,
    calculate_optimal_embargo,
)

# Analysis submodule
from .analysis import (
    # Parameter Stability
    ParameterStabilityAnalyzer,
    analyze_parameter_drift,
    get_consensus_parameters,
    # Overfitting Detection
    OverfittingDetector,
    deflated_sharpe_ratio,
    probability_of_backtest_overfitting,
    performance_degradation_test,
    # Visualization
    ValidationVisualizer,
    plot_equity_curves,
    plot_parameter_evolution,
    plot_split_timeline,
)

__all__ = [
    # Types
    'ParameterSpace',
    'OptimizationResult',
    'WalkForwardResult',
    # Optimization methods
    'grid_search',
    'random_search',
    'bayesian_optimization',
    'genetic_algorithm',
    'walk_forward_optimization',
    'create_walk_forward_splitter',
    'convert_to_validation_result',
    # Splitters
    'BaseSplitter',
    'SplitInfo',
    'TimeSeriesSplit',
    'RollingWindowSplit',
    'ExpandingWindowSplit',
    'TrainValTestSplit',
    'WalkForwardSplit',
    'create_splitter',
    # Results
    'SplitResult',
    'ParameterStability',
    'CrossValidationResult',
    'TrainValTestResult',
    'calculate_robustness_score',
    # Cross-Validation
    'TimeSeriesCV',
    'WalkForwardCV',
    'train_val_test_evaluate',
    # Purged K-Fold
    'PurgedKFold',
    'PurgedSplitInfo',
    'CombinatorialPurgedCV',
    'calculate_optimal_purge_gap',
    'calculate_optimal_embargo',
    # Analysis
    'ParameterStabilityAnalyzer',
    'analyze_parameter_drift',
    'get_consensus_parameters',
    'OverfittingDetector',
    'deflated_sharpe_ratio',
    'probability_of_backtest_overfitting',
    'performance_degradation_test',
    'ValidationVisualizer',
    'plot_equity_curves',
    'plot_parameter_evolution',
    'plot_split_timeline',
]
