"""Validation module for IS/OOS time series analysis.

This module provides various strategies for splitting time series data
and cross-validation methods that respect temporal order.
"""

from .splitters import (
    BaseSplitter,
    SplitInfo,
    TimeSeriesSplit,
    RollingWindowSplit,
    ExpandingWindowSplit,
    TrainValTestSplit,
    WalkForwardSplit,
    create_splitter,
)

from .results import (
    SplitResult,
    ParameterStability,
    CrossValidationResult,
    TrainValTestResult,
    calculate_robustness_score,
)

from .time_series_cv import (
    TimeSeriesCV,
    WalkForwardCV,
    train_val_test_evaluate,
)

from .purged_kfold import (
    PurgedKFold,
    PurgedSplitInfo,
    CombinatorialPurgedCV,
    calculate_optimal_purge_gap,
    calculate_optimal_embargo,
)


__all__ = [
    # Splitters
    "BaseSplitter",
    "SplitInfo",
    "TimeSeriesSplit",
    "RollingWindowSplit",
    "ExpandingWindowSplit",
    "TrainValTestSplit",
    "WalkForwardSplit",
    "create_splitter",
    # Results
    "SplitResult",
    "ParameterStability",
    "CrossValidationResult",
    "TrainValTestResult",
    "calculate_robustness_score",
    # Cross-Validation
    "TimeSeriesCV",
    "WalkForwardCV",
    "train_val_test_evaluate",
    # Purged K-Fold
    "PurgedKFold",
    "PurgedSplitInfo",
    "CombinatorialPurgedCV",
    "calculate_optimal_purge_gap",
    "calculate_optimal_embargo",
]
