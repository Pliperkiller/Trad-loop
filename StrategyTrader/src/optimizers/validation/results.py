"""Result dataclasses for validation operations.

Contains structured result types for cross-validation,
walk-forward analysis, and train/val/test splits.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class SplitResult:
    """Results from a single train/test split."""

    split_idx: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_rows: int
    test_rows: int
    best_params: Dict[str, Any]
    train_score: float
    test_score: float
    degradation_pct: float
    train_metrics: Dict[str, float] = field(default_factory=dict)
    test_metrics: Dict[str, float] = field(default_factory=dict)
    optimization_time: float = 0.0

    @property
    def is_overfitting(self) -> bool:
        """Check if this split shows signs of overfitting."""
        return self.degradation_pct > 50.0

    @property
    def train_test_ratio(self) -> float:
        """Ratio of test score to train score."""
        if self.train_score == 0:
            return 0.0
        return self.test_score / self.train_score

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "split_idx": self.split_idx,
            "train_start": self.train_start.isoformat() if self.train_start else None,
            "train_end": self.train_end.isoformat() if self.train_end else None,
            "test_start": self.test_start.isoformat() if self.test_start else None,
            "test_end": self.test_end.isoformat() if self.test_end else None,
            "train_rows": self.train_rows,
            "test_rows": self.test_rows,
            "best_params": self.best_params,
            "train_score": self.train_score,
            "test_score": self.test_score,
            "degradation_pct": self.degradation_pct,
            "train_metrics": self.train_metrics,
            "test_metrics": self.test_metrics,
            "optimization_time": self.optimization_time,
        }


@dataclass
class ParameterStability:
    """Stability analysis for a single parameter across splits."""

    parameter: str
    values_per_split: List[Any]
    mean: float
    std: float
    coefficient_of_variation: float
    is_stable: bool
    stability_threshold: float = 0.3

    @classmethod
    def from_values(
        cls,
        parameter: str,
        values: List[Any],
        threshold: float = 0.3,
    ) -> "ParameterStability":
        """Create ParameterStability from a list of values.

        Args:
            parameter: Parameter name.
            values: Values across splits.
            threshold: CV threshold for stability.

        Returns:
            ParameterStability instance.
        """
        # Handle non-numeric parameters
        try:
            numeric_values = [float(v) for v in values]
            mean = np.mean(numeric_values)
            std = np.std(numeric_values)
            cv = std / abs(mean) if mean != 0 else float('inf')
        except (TypeError, ValueError):
            # Non-numeric: stability based on uniqueness
            unique_values = len(set(str(v) for v in values))
            mean = 0.0
            std = 0.0
            cv = unique_values / len(values) if values else float('inf')

        is_stable = bool(cv < threshold)

        return cls(
            parameter=parameter,
            values_per_split=values,
            mean=float(mean),
            std=float(std),
            coefficient_of_variation=float(cv),
            is_stable=is_stable,
            stability_threshold=threshold,
        )


@dataclass
class CrossValidationResult:
    """Results from cross-validation or walk-forward analysis."""

    splits: List[SplitResult]
    aggregated_metrics: Dict[str, float]
    robustness_score: float
    parameter_stability: Dict[str, ParameterStability] = field(default_factory=dict)
    overfitting_probability: float = 0.0
    combined_oos_equity: List[float] = field(default_factory=list)
    optimization_time: float = 0.0
    method: str = "walk_forward"

    @property
    def n_splits(self) -> int:
        """Number of splits."""
        return len(self.splits)

    @property
    def mean_train_score(self) -> float:
        """Mean training score across splits."""
        if not self.splits:
            return 0.0
        return float(np.mean([s.train_score for s in self.splits]))

    @property
    def mean_test_score(self) -> float:
        """Mean test score across splits."""
        if not self.splits:
            return 0.0
        return float(np.mean([s.test_score for s in self.splits]))

    @property
    def mean_degradation(self) -> float:
        """Mean degradation percentage across splits."""
        if not self.splits:
            return 0.0
        return float(np.mean([s.degradation_pct for s in self.splits]))

    @property
    def positive_oos_ratio(self) -> float:
        """Ratio of splits with positive OOS performance."""
        if not self.splits:
            return 0.0
        positive = sum(1 for s in self.splits if s.test_score > 0)
        return positive / len(self.splits)

    @property
    def consistency_score(self) -> float:
        """Consistency of performance across splits (0-1)."""
        if not self.splits:
            return 0.0
        test_scores = [s.test_score for s in self.splits]
        std = float(np.std(test_scores))
        if std == 0:
            return 1.0
        mean = float(np.mean(test_scores))
        cv = std / abs(mean) if mean != 0 else float('inf')
        return float(max(0, 1 - cv))

    @property
    def unstable_parameters(self) -> List[str]:
        """List of parameters that are not stable across splits."""
        return [
            name for name, stability in self.parameter_stability.items()
            if not stability.is_stable
        ]

    def to_dataframe(self) -> pd.DataFrame:
        """Export split results to DataFrame."""
        records = []
        for split in self.splits:
            record = {
                "split": split.split_idx,
                "train_start": split.train_start,
                "train_end": split.train_end,
                "test_start": split.test_start,
                "test_end": split.test_end,
                "train_rows": split.train_rows,
                "test_rows": split.test_rows,
                "train_score": split.train_score,
                "test_score": split.test_score,
                "degradation_pct": split.degradation_pct,
            }
            # Add metrics
            for key, value in split.train_metrics.items():
                record[f"train_{key}"] = value
            for key, value in split.test_metrics.items():
                record[f"test_{key}"] = value
            # Add parameters
            for key, value in split.best_params.items():
                record[f"param_{key}"] = value
            records.append(record)

        return pd.DataFrame(records)

    def print_summary(self) -> None:
        """Print a formatted summary of results."""
        print("=" * 60)
        print("CROSS-VALIDATION RESULTS")
        print("=" * 60)
        print(f"Method: {self.method}")
        print(f"Number of splits: {self.n_splits}")
        print(f"Total optimization time: {self.optimization_time:.1f}s")
        print()
        print("PERFORMANCE METRICS")
        print("-" * 40)
        print(f"Mean Train Score:     {self.mean_train_score:.4f}")
        print(f"Mean Test Score:      {self.mean_test_score:.4f}")
        print(f"Mean Degradation:     {self.mean_degradation:.1f}%")
        print(f"Positive OOS Ratio:   {self.positive_oos_ratio:.1%}")
        print(f"Consistency Score:    {self.consistency_score:.2f}")
        print()
        print("ROBUSTNESS ASSESSMENT")
        print("-" * 40)
        print(f"Robustness Score:     {self.robustness_score:.2f}")
        print(f"Overfitting Prob:     {self.overfitting_probability:.1%}")

        if self.unstable_parameters:
            print(f"Unstable Parameters:  {', '.join(self.unstable_parameters)}")
        else:
            print("Unstable Parameters:  None")

        print()
        print("SPLIT DETAILS")
        print("-" * 40)
        for split in self.splits:
            print(f"  Split {split.split_idx}: "
                  f"Train={split.train_score:.4f}, "
                  f"Test={split.test_score:.4f}, "
                  f"Deg={split.degradation_pct:.1f}%")
        print("=" * 60)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "method": self.method,
            "n_splits": self.n_splits,
            "robustness_score": self.robustness_score,
            "overfitting_probability": self.overfitting_probability,
            "optimization_time": self.optimization_time,
            "mean_train_score": self.mean_train_score,
            "mean_test_score": self.mean_test_score,
            "mean_degradation": self.mean_degradation,
            "positive_oos_ratio": self.positive_oos_ratio,
            "consistency_score": self.consistency_score,
            "aggregated_metrics": self.aggregated_metrics,
            "unstable_parameters": self.unstable_parameters,
            "splits": [s.to_dict() for s in self.splits],
        }


@dataclass
class TrainValTestResult:
    """Results from train/validation/test split evaluation."""

    train_result: Dict[str, Any]
    val_result: Dict[str, float]
    test_result: Dict[str, float]
    final_params: Dict[str, Any]
    overfitting_detected: bool
    train_score: float
    val_score: float
    test_score: float
    train_val_degradation: float = 0.0
    val_test_degradation: float = 0.0

    @property
    def is_valid(self) -> bool:
        """Check if results show valid generalization."""
        # Valid if test performance is reasonably close to validation
        if self.val_score == 0:
            return False
        ratio = self.test_score / self.val_score
        return 0.5 <= ratio <= 1.5 and not self.overfitting_detected

    @property
    def total_degradation(self) -> float:
        """Total degradation from train to test."""
        if self.train_score == 0:
            return 0.0
        return ((self.train_score - self.test_score) / self.train_score) * 100

    def print_summary(self) -> None:
        """Print a formatted summary."""
        print("=" * 50)
        print("TRAIN/VALIDATION/TEST RESULTS")
        print("=" * 50)
        print(f"Train Score:          {self.train_score:.4f}")
        print(f"Validation Score:     {self.val_score:.4f}")
        print(f"Test Score:           {self.test_score:.4f}")
        print("-" * 50)
        print(f"Train→Val Degradation:  {self.train_val_degradation:.1f}%")
        print(f"Val→Test Degradation:   {self.val_test_degradation:.1f}%")
        print(f"Total Degradation:      {self.total_degradation:.1f}%")
        print("-" * 50)
        print(f"Overfitting Detected:   {'Yes' if self.overfitting_detected else 'No'}")
        print(f"Valid Generalization:   {'Yes' if self.is_valid else 'No'}")
        print("=" * 50)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "train_result": self.train_result,
            "val_result": self.val_result,
            "test_result": self.test_result,
            "final_params": self.final_params,
            "overfitting_detected": self.overfitting_detected,
            "train_score": self.train_score,
            "val_score": self.val_score,
            "test_score": self.test_score,
            "train_val_degradation": self.train_val_degradation,
            "val_test_degradation": self.val_test_degradation,
            "total_degradation": self.total_degradation,
            "is_valid": self.is_valid,
        }


def calculate_robustness_score(
    splits: List[SplitResult],
    positive_weight: float = 0.4,
    consistency_weight: float = 0.3,
    degradation_weight: float = 0.3,
) -> float:
    """Calculate robustness score from split results.

    Args:
        splits: List of split results.
        positive_weight: Weight for positive OOS ratio component.
        consistency_weight: Weight for consistency component.
        degradation_weight: Weight for low degradation component.

    Returns:
        Robustness score between 0 and 1.
    """
    if not splits:
        return 0.0

    # Positive OOS ratio (0-1)
    positive_oos = sum(1 for s in splits if s.test_score > 0) / len(splits)

    # Consistency (inverse of CV, capped at 1)
    test_scores = [s.test_score for s in splits]
    std_val = float(np.std(test_scores))
    mean_val = float(np.mean(test_scores))
    if std_val == 0 or mean_val == 0:
        consistency = 1.0
    else:
        cv = std_val / abs(mean_val)
        consistency = float(max(0, min(1, 1 - cv)))

    # Low degradation score (0-1)
    degradations = [s.degradation_pct for s in splits]
    mean_deg = float(np.mean(degradations))
    # 0% degradation = 1.0, 100% degradation = 0.0
    low_degradation = float(max(0, min(1, 1 - mean_deg / 100)))

    score = (
        positive_weight * positive_oos +
        consistency_weight * consistency +
        degradation_weight * low_degradation
    )

    return float(score)
