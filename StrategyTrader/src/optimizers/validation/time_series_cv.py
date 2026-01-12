"""Time Series Cross-Validation for strategy optimization.

Provides cross-validation methods that respect temporal order
and integrate with the strategy optimizer.
"""

import time
from datetime import datetime
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd

from .splitters import BaseSplitter, TimeSeriesSplit, WalkForwardSplit
from .results import (
    SplitResult,
    ParameterStability,
    CrossValidationResult,
    TrainValTestResult,
    calculate_robustness_score,
)

if TYPE_CHECKING:
    from ..base import StrategyOptimizer


class TimeSeriesCV:
    """Time Series Cross-Validation for strategy optimization.

    Performs cross-validation while respecting temporal order.
    Integrates with StrategyOptimizer for parameter optimization
    in each fold.

    Example:
        >>> cv = TimeSeriesCV(n_splits=5, gap=10)
        >>> result = cv.cross_validate(optimizer, method='bayesian', n_iter=50)
        >>> print(f"Robustness: {result.robustness_score:.2f}")
    """

    def __init__(
        self,
        n_splits: int = 5,
        gap: int = 0,
        test_size: Optional[int] = None,
        max_train_size: Optional[int] = None,
        splitter: Optional[BaseSplitter] = None,
    ):
        """Initialize TimeSeriesCV.

        Args:
            n_splits: Number of cross-validation folds.
            gap: Number of samples to exclude between train and test.
            test_size: Size of test set per fold.
            max_train_size: Maximum training size (for rolling window).
            splitter: Custom splitter to use. If None, uses TimeSeriesSplit.
        """
        self.n_splits = n_splits
        self.gap = gap
        self.test_size = test_size
        self.max_train_size = max_train_size

        if splitter is not None:
            self.splitter = splitter
        else:
            self.splitter = TimeSeriesSplit(
                n_splits=n_splits,
                test_size=test_size,
                gap=gap,
                max_train_size=max_train_size,
            )

    def cross_validate(
        self,
        optimizer: "StrategyOptimizer",
        optimization_method: str = "random",
        n_iter: int = 50,
        scoring: str = "sharpe_ratio",
        verbose: bool = True,
        track_params: bool = True,
    ) -> CrossValidationResult:
        """Run cross-validation with parameter optimization.

        Args:
            optimizer: Strategy optimizer instance with parameter grid.
            optimization_method: Optimization method ('grid', 'random', 'bayesian', 'genetic').
            n_iter: Number of iterations for random/bayesian optimization.
            scoring: Scoring metric to optimize.
            verbose: Print progress information.
            track_params: Track parameter stability across folds.

        Returns:
            CrossValidationResult with aggregated metrics and per-split results.
        """
        start_time = time.time()
        splits_results: List[SplitResult] = []
        all_params: List[Dict[str, Any]] = []
        combined_oos_equity: List[float] = []

        data = optimizer.data
        if data is None:
            raise ValueError("Optimizer must have data loaded")

        split_infos = self.splitter.get_split_info(data)

        for split_idx, (train_idx, test_idx) in enumerate(self.splitter.split(data)):
            if verbose:
                print(f"\n{'='*50}")
                print(f"Fold {split_idx + 1}/{self.n_splits}")
                print(f"{'='*50}")

            # Split data
            train_data = data.iloc[train_idx].copy()
            test_data = data.iloc[test_idx].copy()

            if verbose:
                print(f"Train: {len(train_data)} samples")
                print(f"Test:  {len(test_data)} samples")

            # Optimize on training data
            optimizer_copy = self._clone_optimizer(optimizer)
            optimizer_copy.data = train_data

            opt_result = optimizer_copy.optimize(
                method=optimization_method,
                n_iter=n_iter,
                verbose=verbose,
            )

            best_params = opt_result.best_params
            train_score = opt_result.best_score
            all_params.append(best_params)

            # Evaluate on test data
            test_metrics = optimizer_copy.evaluate_params(best_params, test_data)
            test_score = test_metrics.get(scoring, 0.0)

            # Calculate degradation
            if train_score != 0:
                degradation_pct = ((train_score - test_score) / abs(train_score)) * 100
            else:
                degradation_pct = 0.0

            # Get dates
            split_info = split_infos[split_idx]

            split_result = SplitResult(
                split_idx=split_idx,
                train_start=split_info.train_start_date or datetime.now(),
                train_end=split_info.train_end_date or datetime.now(),
                test_start=split_info.test_start_date or datetime.now(),
                test_end=split_info.test_end_date or datetime.now(),
                train_rows=len(train_idx),
                test_rows=len(test_idx),
                best_params=best_params,
                train_score=train_score,
                test_score=test_score,
                degradation_pct=degradation_pct,
                train_metrics={"score": train_score},
                test_metrics=test_metrics,
            )
            splits_results.append(split_result)

            # Collect OOS equity if available
            if "equity_curve" in test_metrics:
                combined_oos_equity.extend(test_metrics["equity_curve"])

            if verbose:
                print(f"Best params: {best_params}")
                print(f"Train score: {train_score:.4f}")
                print(f"Test score:  {test_score:.4f}")
                print(f"Degradation: {degradation_pct:.1f}%")

        # Calculate parameter stability
        param_stability: Dict[str, ParameterStability] = {}
        if track_params and all_params:
            param_names = all_params[0].keys()
            for param_name in param_names:
                values = [p.get(param_name) for p in all_params]
                param_stability[param_name] = ParameterStability.from_values(
                    parameter=param_name,
                    values=values,
                )

        # Calculate aggregated metrics
        robustness = calculate_robustness_score(splits_results)
        aggregated_metrics = self._aggregate_metrics(splits_results)

        # Calculate overfitting probability
        overfitting_prob = self._estimate_overfitting_probability(splits_results)

        total_time = time.time() - start_time

        result = CrossValidationResult(
            splits=splits_results,
            aggregated_metrics=aggregated_metrics,
            robustness_score=robustness,
            parameter_stability=param_stability,
            overfitting_probability=overfitting_prob,
            combined_oos_equity=combined_oos_equity,
            optimization_time=total_time,
            method="time_series_cv",
        )

        if verbose:
            print("\n")
            result.print_summary()

        return result

    def _clone_optimizer(self, optimizer: "StrategyOptimizer") -> "StrategyOptimizer":
        """Create a copy of the optimizer for each fold.

        Args:
            optimizer: Original optimizer.

        Returns:
            Cloned optimizer with same configuration.
        """
        # Import here to avoid circular imports
        from ..base import StrategyOptimizer

        cloned = StrategyOptimizer(
            strategy_class=optimizer.strategy_class,
            param_grid=optimizer.param_grid.copy(),
        )
        return cloned

    def _aggregate_metrics(self, splits: List[SplitResult]) -> Dict[str, float]:
        """Aggregate metrics across all splits.

        Args:
            splits: List of split results.

        Returns:
            Dictionary of aggregated metrics.
        """
        if not splits:
            return {}

        metrics: Dict[str, List[float]] = {}

        # Collect all test metrics
        for split in splits:
            for key, value in split.test_metrics.items():
                if isinstance(value, (int, float)):
                    if key not in metrics:
                        metrics[key] = []
                    metrics[key].append(float(value))

        # Aggregate with mean and std
        aggregated = {}
        for key, values in metrics.items():
            aggregated[f"{key}_mean"] = float(np.mean(values))
            aggregated[f"{key}_std"] = float(np.std(values))
            aggregated[f"{key}_min"] = float(np.min(values))
            aggregated[f"{key}_max"] = float(np.max(values))

        return aggregated

    def _estimate_overfitting_probability(
        self, splits: List[SplitResult]
    ) -> float:
        """Estimate probability of overfitting based on IS/OOS comparison.

        Args:
            splits: List of split results.

        Returns:
            Estimated overfitting probability (0-1).
        """
        if len(splits) < 2:
            return 0.0

        # Count splits where test score is significantly worse than train
        degraded_count = 0
        for split in splits:
            # Consider significant degradation as > 30%
            if split.degradation_pct > 30:
                degraded_count += 1

        return degraded_count / len(splits)

    def get_split_info(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Get information about how the data will be split.

        Args:
            data: DataFrame to be split.

        Returns:
            List of dictionaries with split information.
        """
        split_infos = self.splitter.get_split_info(data)
        return [
            {
                "split_idx": info.split_idx,
                "train_start": info.train_start_date,
                "train_end": info.train_end_date,
                "test_start": info.test_start_date,
                "test_end": info.test_end_date,
                "train_size": info.train_size,
                "test_size": info.test_size,
            }
            for info in split_infos
        ]


class WalkForwardCV(TimeSeriesCV):
    """Walk-Forward Cross-Validation with non-overlapping test periods.

    Extends TimeSeriesCV with walk-forward specific functionality.
    Test periods are strictly sequential and non-overlapping.

    Example:
        >>> wf = WalkForwardCV(n_splits=5, train_pct=0.6, anchored=True)
        >>> result = wf.cross_validate(optimizer)
    """

    def __init__(
        self,
        n_splits: int = 5,
        train_pct: float = 0.6,
        anchored: bool = True,
        gap: int = 0,
        min_train_size: Optional[int] = None,
        min_test_size: Optional[int] = None,
    ):
        """Initialize WalkForwardCV.

        Args:
            n_splits: Number of walk-forward periods.
            train_pct: Initial percentage of data for training.
            anchored: If True, use expanding window. If False, rolling window.
            gap: Number of samples between train and test.
            min_train_size: Minimum required training samples.
            min_test_size: Minimum required test samples.
        """
        self.train_pct = train_pct
        self.anchored = anchored
        self.min_train_size = min_train_size
        self.min_test_size = min_test_size

        splitter = WalkForwardSplit(
            n_splits=n_splits,
            train_pct=train_pct,
            anchored=anchored,
            gap=gap,
            min_train_size=min_train_size,
            min_test_size=min_test_size,
        )

        super().__init__(
            n_splits=n_splits,
            gap=gap,
            splitter=splitter,
        )

    def cross_validate(
        self,
        optimizer: "StrategyOptimizer",
        optimization_method: str = "random",
        n_iter: int = 50,
        scoring: str = "sharpe_ratio",
        verbose: bool = True,
        track_params: bool = True,
    ) -> CrossValidationResult:
        """Run walk-forward cross-validation.

        Overrides parent to set method name correctly.
        """
        result = super().cross_validate(
            optimizer=optimizer,
            optimization_method=optimization_method,
            n_iter=n_iter,
            scoring=scoring,
            verbose=verbose,
            track_params=track_params,
        )

        # Update method name
        result.method = f"walk_forward_{'anchored' if self.anchored else 'rolling'}"

        return result


def train_val_test_evaluate(
    optimizer: "StrategyOptimizer",
    train_pct: float = 0.6,
    val_pct: float = 0.2,
    test_pct: float = 0.2,
    gap: int = 0,
    optimization_method: str = "random",
    n_iter: int = 50,
    scoring: str = "sharpe_ratio",
    overfitting_threshold: float = 0.5,
    verbose: bool = True,
) -> TrainValTestResult:
    """Perform train/validation/test evaluation.

    Uses validation set for parameter selection and test set
    for final unbiased evaluation.

    Args:
        optimizer: Strategy optimizer instance.
        train_pct: Percentage of data for training.
        val_pct: Percentage of data for validation.
        test_pct: Percentage of data for testing.
        gap: Gap between splits.
        optimization_method: Optimization method for training.
        n_iter: Iterations for optimization.
        scoring: Scoring metric.
        overfitting_threshold: Degradation threshold for overfitting detection.
        verbose: Print progress.

    Returns:
        TrainValTestResult with scores and overfitting assessment.
    """
    from .splitters import TrainValTestSplit

    data = optimizer.data
    if data is None:
        raise ValueError("Optimizer must have data loaded")

    splitter = TrainValTestSplit(
        train_pct=train_pct,
        val_pct=val_pct,
        test_pct=test_pct,
        gap=gap,
    )

    train_data, val_data, test_data = splitter.split(data)

    if verbose:
        print("=" * 50)
        print("TRAIN/VALIDATION/TEST EVALUATION")
        print("=" * 50)
        print(f"Train: {len(train_data)} samples ({train_pct:.0%})")
        print(f"Val:   {len(val_data)} samples ({val_pct:.0%})")
        print(f"Test:  {len(test_data)} samples ({test_pct:.0%})")
        print()

    # Step 1: Optimize on training data
    if verbose:
        print("Step 1: Optimizing on training data...")

    optimizer.data = train_data
    train_result = optimizer.optimize(
        method=optimization_method,
        n_iter=n_iter,
        verbose=verbose,
    )
    best_params = train_result.best_params
    train_score = train_result.best_score

    # Step 2: Evaluate on validation data
    if verbose:
        print("\nStep 2: Evaluating on validation data...")

    val_metrics = optimizer.evaluate_params(best_params, val_data)
    val_score = val_metrics.get(scoring, 0.0)

    # Check for overfitting on validation
    if train_score != 0:
        train_val_deg = ((train_score - val_score) / abs(train_score)) * 100
    else:
        train_val_deg = 0.0

    # Step 3: Final evaluation on test data
    if verbose:
        print("\nStep 3: Final evaluation on test data...")

    test_metrics = optimizer.evaluate_params(best_params, test_data)
    test_score = test_metrics.get(scoring, 0.0)

    # Check degradation from validation to test
    if val_score != 0:
        val_test_deg = ((val_score - test_score) / abs(val_score)) * 100
    else:
        val_test_deg = 0.0

    # Detect overfitting
    overfitting_detected = (
        train_val_deg > overfitting_threshold * 100 or
        val_test_deg > overfitting_threshold * 100
    )

    result = TrainValTestResult(
        train_result={"best_params": best_params, "best_score": train_score},
        val_result=val_metrics,
        test_result=test_metrics,
        final_params=best_params,
        overfitting_detected=overfitting_detected,
        train_score=train_score,
        val_score=val_score,
        test_score=test_score,
        train_val_degradation=train_val_deg,
        val_test_degradation=val_test_deg,
    )

    if verbose:
        print()
        result.print_summary()

    # Restore original data
    optimizer.data = data

    return result
