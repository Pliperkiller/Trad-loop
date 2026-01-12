"""Analysis module for IS/OOS validation.

Provides tools for analyzing parameter stability, detecting overfitting,
and visualizing validation results.
"""

from .parameter_stability import (
    ParameterStabilityAnalyzer,
    analyze_parameter_drift,
    get_consensus_parameters,
)

from .overfitting_detection import (
    OverfittingDetector,
    deflated_sharpe_ratio,
    probability_of_backtest_overfitting,
    performance_degradation_test,
)

from .visualization import (
    ValidationVisualizer,
    plot_equity_curves,
    plot_parameter_evolution,
    plot_split_timeline,
)


__all__ = [
    # Parameter Stability
    "ParameterStabilityAnalyzer",
    "analyze_parameter_drift",
    "get_consensus_parameters",
    # Overfitting Detection
    "OverfittingDetector",
    "deflated_sharpe_ratio",
    "probability_of_backtest_overfitting",
    "performance_degradation_test",
    # Visualization
    "ValidationVisualizer",
    "plot_equity_curves",
    "plot_parameter_evolution",
    "plot_split_timeline",
]
