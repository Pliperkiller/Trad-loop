"""Parameter stability analysis across validation splits.

Analyzes how strategy parameters change across different time periods
to identify potential overfitting or regime changes.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..validation.results import ParameterStability, SplitResult


@dataclass
class ParameterDrift:
    """Analysis of parameter drift over splits."""

    parameter: str
    values: List[Any]
    trend: str  # 'increasing', 'decreasing', 'stable', 'volatile'
    drift_magnitude: float
    correlation_with_split: float
    is_significant: bool


@dataclass
class ConsensusParameters:
    """Consensus parameters from multiple splits."""

    parameters: Dict[str, Any]
    confidence: Dict[str, float]
    method: str
    n_splits_used: int


class ParameterStabilityAnalyzer:
    """Analyzes parameter stability across cross-validation splits.

    Identifies which parameters are stable and which vary significantly,
    helping to identify potential overfitting or regime-dependent parameters.

    Example:
        >>> analyzer = ParameterStabilityAnalyzer(stability_threshold=0.3)
        >>> stability = analyzer.analyze(cv_result.splits)
        >>> unstable = analyzer.get_unstable_parameters()
        >>> print(f"Unstable parameters: {unstable}")
    """

    def __init__(
        self,
        stability_threshold: float = 0.3,
        trend_threshold: float = 0.5,
    ):
        """Initialize ParameterStabilityAnalyzer.

        Args:
            stability_threshold: Coefficient of variation threshold for stability.
                               Parameters with CV > threshold are considered unstable.
            trend_threshold: Correlation threshold for detecting trends.
        """
        if not (0 < stability_threshold <= 1):
            raise ValueError("stability_threshold must be between 0 and 1")
        if not (0 <= trend_threshold <= 1):
            raise ValueError("trend_threshold must be between 0 and 1")

        self.stability_threshold = stability_threshold
        self.trend_threshold = trend_threshold
        self._results: Dict[str, ParameterStability] = {}
        self._drift_analysis: Dict[str, ParameterDrift] = {}

    def analyze(
        self,
        splits: List[SplitResult],
    ) -> Dict[str, ParameterStability]:
        """Analyze parameter stability across splits.

        Args:
            splits: List of split results with best_params.

        Returns:
            Dictionary mapping parameter names to ParameterStability.
        """
        if not splits:
            return {}

        # Extract all parameters
        all_params = [s.best_params for s in splits]
        if not all_params:
            return {}

        param_names = set()
        for params in all_params:
            param_names.update(params.keys())

        results: Dict[str, ParameterStability] = {}
        for param_name in param_names:
            values = [p.get(param_name) for p in all_params]
            results[param_name] = ParameterStability.from_values(
                parameter=param_name,
                values=values,
                threshold=self.stability_threshold,
            )

        self._results = results
        return results

    def analyze_drift(
        self,
        splits: List[SplitResult],
    ) -> Dict[str, ParameterDrift]:
        """Analyze parameter drift over time.

        Args:
            splits: List of split results.

        Returns:
            Dictionary of ParameterDrift for each parameter.
        """
        if not splits or len(splits) < 3:
            return {}

        all_params = [s.best_params for s in splits]
        param_names = set()
        for params in all_params:
            param_names.update(params.keys())

        results: Dict[str, ParameterDrift] = {}

        for param_name in param_names:
            values = [p.get(param_name) for p in all_params]

            # Try to convert to numeric
            try:
                numeric_values = [float(v) for v in values if v is not None]
                if len(numeric_values) < 3:
                    continue

                # Calculate correlation with split index
                split_indices = list(range(len(numeric_values)))
                corr = float(np.corrcoef(split_indices, numeric_values)[0, 1])

                # Determine trend
                if abs(corr) < self.trend_threshold:
                    # Check for volatility
                    cv = float(np.std(numeric_values) / abs(np.mean(numeric_values))) if np.mean(numeric_values) != 0 else 0
                    trend = "volatile" if cv > self.stability_threshold else "stable"
                elif corr > 0:
                    trend = "increasing"
                else:
                    trend = "decreasing"

                # Calculate drift magnitude
                drift_magnitude = abs(numeric_values[-1] - numeric_values[0])
                if numeric_values[0] != 0:
                    drift_magnitude = drift_magnitude / abs(numeric_values[0])

                # Is drift significant?
                is_significant = abs(corr) > self.trend_threshold

                results[param_name] = ParameterDrift(
                    parameter=param_name,
                    values=values,
                    trend=trend,
                    drift_magnitude=drift_magnitude,
                    correlation_with_split=corr,
                    is_significant=is_significant,
                )

            except (TypeError, ValueError):
                # Non-numeric parameter - analyze categorically
                unique_values = len(set(str(v) for v in values if v is not None))
                is_significant = unique_values > 1

                results[param_name] = ParameterDrift(
                    parameter=param_name,
                    values=values,
                    trend="categorical",
                    drift_magnitude=unique_values / len(values),
                    correlation_with_split=0.0,
                    is_significant=is_significant,
                )

        self._drift_analysis = results
        return results

    def get_unstable_parameters(self) -> List[str]:
        """Get list of unstable parameters.

        Returns:
            List of parameter names that are not stable.
        """
        return [
            name for name, stability in self._results.items()
            if not stability.is_stable
        ]

    def get_stable_parameters(self) -> List[str]:
        """Get list of stable parameters.

        Returns:
            List of parameter names that are stable.
        """
        return [
            name for name, stability in self._results.items()
            if stability.is_stable
        ]

    def get_drifting_parameters(self) -> List[str]:
        """Get parameters with significant drift.

        Returns:
            List of parameter names with significant trend.
        """
        return [
            name for name, drift in self._drift_analysis.items()
            if drift.is_significant and drift.trend in ("increasing", "decreasing")
        ]

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of parameter stability analysis.

        Returns:
            Dictionary with summary statistics.
        """
        n_params = len(self._results)
        n_stable = len(self.get_stable_parameters())
        n_drifting = len(self.get_drifting_parameters())

        return {
            "total_parameters": n_params,
            "stable_parameters": n_stable,
            "unstable_parameters": n_params - n_stable,
            "drifting_parameters": n_drifting,
            "stability_ratio": n_stable / n_params if n_params > 0 else 0,
            "stable_param_names": self.get_stable_parameters(),
            "unstable_param_names": self.get_unstable_parameters(),
            "drifting_param_names": self.get_drifting_parameters(),
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Export stability results to DataFrame.

        Returns:
            DataFrame with stability metrics for each parameter.
        """
        if not self._results:
            return pd.DataFrame()

        records = []
        for name, stability in self._results.items():
            record = {
                "parameter": name,
                "mean": stability.mean,
                "std": stability.std,
                "cv": stability.coefficient_of_variation,
                "is_stable": stability.is_stable,
                "values": stability.values_per_split,
            }

            # Add drift info if available
            if name in self._drift_analysis:
                drift = self._drift_analysis[name]
                record["trend"] = drift.trend
                record["drift_magnitude"] = drift.drift_magnitude
                record["correlation"] = drift.correlation_with_split
                record["significant_drift"] = drift.is_significant

            records.append(record)

        return pd.DataFrame(records)


def analyze_parameter_drift(
    params_per_split: List[Dict[str, Any]],
    trend_threshold: float = 0.5,
) -> Dict[str, ParameterDrift]:
    """Analyze parameter drift from a list of parameter dictionaries.

    Args:
        params_per_split: List of parameter dictionaries from each split.
        trend_threshold: Threshold for significant trend detection.

    Returns:
        Dictionary mapping parameter names to ParameterDrift.
    """
    if len(params_per_split) < 3:
        return {}

    param_names = set()
    for params in params_per_split:
        param_names.update(params.keys())

    results: Dict[str, ParameterDrift] = {}

    for param_name in param_names:
        values = [p.get(param_name) for p in params_per_split]

        try:
            numeric_values = [float(v) for v in values if v is not None]
            if len(numeric_values) < 3:
                continue

            split_indices = list(range(len(numeric_values)))
            corr = float(np.corrcoef(split_indices, numeric_values)[0, 1])
            mean_val = float(np.mean(numeric_values))
            std_val = float(np.std(numeric_values))

            if abs(corr) < trend_threshold:
                cv = std_val / abs(mean_val) if mean_val != 0 else 0
                trend = "volatile" if cv > 0.3 else "stable"
            elif corr > 0:
                trend = "increasing"
            else:
                trend = "decreasing"

            drift_magnitude = abs(numeric_values[-1] - numeric_values[0])
            if numeric_values[0] != 0:
                drift_magnitude = drift_magnitude / abs(numeric_values[0])

            results[param_name] = ParameterDrift(
                parameter=param_name,
                values=values,
                trend=trend,
                drift_magnitude=drift_magnitude,
                correlation_with_split=corr,
                is_significant=abs(corr) > trend_threshold,
            )

        except (TypeError, ValueError):
            unique_values = len(set(str(v) for v in values if v is not None))
            results[param_name] = ParameterDrift(
                parameter=param_name,
                values=values,
                trend="categorical",
                drift_magnitude=unique_values / len(values),
                correlation_with_split=0.0,
                is_significant=unique_values > 1,
            )

    return results


def get_consensus_parameters(
    params_per_split: List[Dict[str, Any]],
    method: str = "median",
    min_agreement: float = 0.5,
) -> ConsensusParameters:
    """Calculate consensus parameters from multiple splits.

    Args:
        params_per_split: List of parameter dictionaries.
        method: Aggregation method ('median', 'mean', 'mode', 'last').
        min_agreement: Minimum agreement ratio for confidence.

    Returns:
        ConsensusParameters with aggregated values and confidence.
    """
    if not params_per_split:
        return ConsensusParameters(
            parameters={},
            confidence={},
            method=method,
            n_splits_used=0,
        )

    param_names = set()
    for params in params_per_split:
        param_names.update(params.keys())

    consensus: Dict[str, Any] = {}
    confidence: Dict[str, float] = {}

    for param_name in param_names:
        values = [p.get(param_name) for p in params_per_split if param_name in p]
        if not values:
            continue

        # Try numeric aggregation
        try:
            numeric_values = [float(v) for v in values if v is not None]
            if not numeric_values:
                continue

            if method == "median":
                consensus[param_name] = float(np.median(numeric_values))
            elif method == "mean":
                consensus[param_name] = float(np.mean(numeric_values))
            elif method == "last":
                consensus[param_name] = numeric_values[-1]
            else:  # mode
                from scipy import stats
                mode_result = stats.mode(numeric_values, keepdims=False)
                consensus[param_name] = float(mode_result.mode)

            # Calculate confidence as inverse of CV
            std_val = float(np.std(numeric_values))
            mean_val = float(np.mean(numeric_values))
            cv = std_val / abs(mean_val) if mean_val != 0 else float('inf')
            confidence[param_name] = max(0.0, min(1.0, 1.0 - cv))

        except (TypeError, ValueError):
            # Non-numeric: use mode
            from collections import Counter
            counter = Counter(str(v) for v in values if v is not None)
            if counter:
                most_common = counter.most_common(1)[0]
                consensus[param_name] = most_common[0]
                confidence[param_name] = most_common[1] / len(values)

    return ConsensusParameters(
        parameters=consensus,
        confidence=confidence,
        method=method,
        n_splits_used=len(params_per_split),
    )
