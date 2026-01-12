"""Purged K-Fold Cross-Validation with embargo.

Implements purged k-fold cross-validation that eliminates data leakage
from overlapping time periods, especially important when using
features with lookback windows.

Based on the methodology described by Marcos López de Prado in
"Advances in Financial Machine Learning".
"""

from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

from .splitters import BaseSplitter, SplitInfo


@dataclass
class PurgedSplitInfo(SplitInfo):
    """Extended split info with purge and embargo details."""

    purged_samples: int = 0
    embargo_samples: int = 0
    effective_train_size: int = 0


class PurgedKFold(BaseSplitter):
    """Purged K-Fold Cross-Validation with embargo.

    This splitter addresses data leakage in time series by:

    1. **Purging**: Removes training samples that overlap temporally
       with test samples. This is crucial when features use lookback
       windows that could leak future information.

    2. **Embargo**: Adds a buffer period after each test set where
       training samples are excluded, preventing information from
       test period bleeding into training.

    Example:
        >>> pkf = PurgedKFold(n_splits=5, purge_gap=20, embargo_pct=0.02)
        >>> for train_idx, test_idx in pkf.split(data):
        ...     train_data = data.iloc[train_idx]
        ...     test_data = data.iloc[test_idx]

    The splitter ensures that:
    - No training sample is within `purge_gap` samples of any test sample
    - An additional `embargo_pct` of data after test set is excluded from training
    """

    def __init__(
        self,
        n_splits: int = 5,
        purge_gap: int = 0,
        embargo_pct: float = 0.01,
    ):
        """Initialize PurgedKFold.

        Args:
            n_splits: Number of folds (must be >= 2).
            purge_gap: Number of samples to purge before test set.
                       This should match the lookback window of your features.
            embargo_pct: Percentage of data to embargo after each test set (0-0.1).
        """
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        if purge_gap < 0:
            raise ValueError("purge_gap must be non-negative")
        if not (0 <= embargo_pct <= 0.1):
            raise ValueError("embargo_pct must be between 0 and 0.1")

        self.n_splits = n_splits
        self.purge_gap = purge_gap
        self.embargo_pct = embargo_pct

    def split(
        self, data: pd.DataFrame
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate purged train/test indices.

        Args:
            data: DataFrame with time series data.

        Yields:
            Tuple of (train_indices, test_indices) as numpy arrays.
        """
        self._validate_data(data)
        n_samples = len(data)
        indices = np.arange(n_samples)

        # Calculate fold sizes
        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[: n_samples % self.n_splits] += 1

        # Calculate embargo size
        embargo_size = int(n_samples * self.embargo_pct)

        current = 0
        for fold_idx in range(self.n_splits):
            test_start = current
            test_end = current + fold_sizes[fold_idx]

            # Purge: remove samples too close before test
            purge_start = max(0, test_start - self.purge_gap)

            # Embargo: exclude samples after test
            embargo_end = min(n_samples, test_end + embargo_size)

            # Build train indices
            train_indices_before = indices[:purge_start]
            train_indices_after = indices[embargo_end:]

            train_idx = np.concatenate([train_indices_before, train_indices_after])
            test_idx = indices[test_start:test_end]

            if len(train_idx) > 0 and len(test_idx) > 0:
                yield train_idx, test_idx

            current = test_end

    def get_n_splits(self, data: Optional[pd.DataFrame] = None) -> int:
        """Return the number of splits."""
        return self.n_splits

    def get_split_info(self, data: pd.DataFrame) -> List[PurgedSplitInfo]:
        """Get detailed information about each purged split.

        Args:
            data: DataFrame with time series data.

        Returns:
            List of PurgedSplitInfo with purge/embargo details.
        """
        splits_info = []
        has_datetime_index = isinstance(data.index, pd.DatetimeIndex)
        n_samples = len(data)

        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[: n_samples % self.n_splits] += 1
        embargo_size = int(n_samples * self.embargo_pct)

        current = 0
        for idx, (train_idx, test_idx) in enumerate(self.split(data)):
            test_start = current
            test_end = current + fold_sizes[idx]
            purge_start = max(0, test_start - self.purge_gap)
            embargo_end = min(n_samples, test_end + embargo_size)

            purged_samples = test_start - purge_start
            embargo_samples = embargo_end - test_end

            info = PurgedSplitInfo(
                split_idx=idx,
                train_start_idx=int(train_idx[0]) if len(train_idx) > 0 else 0,
                train_end_idx=int(train_idx[-1]) if len(train_idx) > 0 else 0,
                test_start_idx=int(test_idx[0]),
                test_end_idx=int(test_idx[-1]),
                train_size=len(train_idx),
                test_size=len(test_idx),
                purged_samples=purged_samples,
                embargo_samples=embargo_samples,
                effective_train_size=len(train_idx),
            )

            if has_datetime_index:
                dt_index = data.index  # type: ignore[assignment]
                if len(train_idx) > 0:
                    info.train_start_date = dt_index[train_idx[0]].to_pydatetime()  # type: ignore[union-attr]
                    info.train_end_date = dt_index[train_idx[-1]].to_pydatetime()  # type: ignore[union-attr]
                info.test_start_date = dt_index[test_idx[0]].to_pydatetime()  # type: ignore[union-attr]
                info.test_end_date = dt_index[test_idx[-1]].to_pydatetime()  # type: ignore[union-attr]

            splits_info.append(info)
            current = test_end

        return splits_info

    def print_split_summary(self, data: pd.DataFrame) -> None:
        """Print a summary of the splits with purge/embargo info.

        Args:
            data: DataFrame to be split.
        """
        splits = self.get_split_info(data)
        n_samples = len(data)

        print("=" * 60)
        print("PURGED K-FOLD SPLIT SUMMARY")
        print("=" * 60)
        print(f"Total samples: {n_samples}")
        print(f"Number of folds: {self.n_splits}")
        print(f"Purge gap: {self.purge_gap} samples")
        print(f"Embargo: {self.embargo_pct:.1%} ({int(n_samples * self.embargo_pct)} samples)")
        print()

        for split in splits:
            print(f"Fold {split.split_idx + 1}:")
            print(f"  Train: {split.train_size} samples (effective)")
            print(f"  Test:  {split.test_size} samples")
            print(f"  Purged: {split.purged_samples} samples")
            print(f"  Embargoed: {split.embargo_samples} samples")
            print()

        print("=" * 60)


class CombinatorialPurgedCV(BaseSplitter):
    """Combinatorial Purged Cross-Validation (CPCV).

    Generates all possible train/test combinations while
    applying purging and embargo to prevent data leakage.

    This provides more robust out-of-sample performance estimation
    by using all possible data configurations.

    Note: The number of combinations grows quickly with n_splits,
    so this is computationally expensive for large n_splits.
    """

    def __init__(
        self,
        n_splits: int = 5,
        n_test_splits: int = 2,
        purge_gap: int = 0,
        embargo_pct: float = 0.01,
    ):
        """Initialize CombinatorialPurgedCV.

        Args:
            n_splits: Total number of data groups.
            n_test_splits: Number of groups to use for testing per combination.
            purge_gap: Samples to purge before test groups.
            embargo_pct: Percentage to embargo after test groups.
        """
        if n_splits < 3:
            raise ValueError("n_splits must be at least 3")
        if n_test_splits < 1 or n_test_splits >= n_splits:
            raise ValueError("n_test_splits must be between 1 and n_splits-1")
        if purge_gap < 0:
            raise ValueError("purge_gap must be non-negative")
        if not (0 <= embargo_pct <= 0.1):
            raise ValueError("embargo_pct must be between 0 and 0.1")

        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.purge_gap = purge_gap
        self.embargo_pct = embargo_pct

    def _get_combinations(self) -> List[Tuple[int, ...]]:
        """Generate all test group combinations."""
        from itertools import combinations
        return list(combinations(range(self.n_splits), self.n_test_splits))

    def split(
        self, data: pd.DataFrame
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate combinatorial purged train/test indices.

        Args:
            data: DataFrame with time series data.

        Yields:
            Tuple of (train_indices, test_indices) as numpy arrays.
        """
        self._validate_data(data)
        n_samples = len(data)
        indices = np.arange(n_samples)

        # Calculate group boundaries
        group_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        group_sizes[: n_samples % self.n_splits] += 1
        group_bounds = np.cumsum(np.concatenate([[0], group_sizes]))

        embargo_size = int(n_samples * self.embargo_pct)

        for test_groups in self._get_combinations():
            # Determine test indices
            test_idx_list = []
            for group in test_groups:
                start, end = group_bounds[group], group_bounds[group + 1]
                test_idx_list.append(indices[start:end])
            test_idx = np.concatenate(test_idx_list)

            # Determine train indices with purging and embargo
            train_mask = np.ones(n_samples, dtype=bool)

            for group in test_groups:
                test_start = group_bounds[group]
                test_end = group_bounds[group + 1]

                # Purge before test
                purge_start = max(0, test_start - self.purge_gap)
                train_mask[purge_start:test_start] = False

                # Exclude test
                train_mask[test_start:test_end] = False

                # Embargo after test
                embargo_end = min(n_samples, test_end + embargo_size)
                train_mask[test_end:embargo_end] = False

            train_idx = indices[train_mask]

            if len(train_idx) > 0 and len(test_idx) > 0:
                yield train_idx, test_idx

    def get_n_splits(self, data: Optional[pd.DataFrame] = None) -> int:
        """Return the number of combinations."""
        from math import comb
        return comb(self.n_splits, self.n_test_splits)

    def print_summary(self) -> None:
        """Print summary of combinatorial setup."""
        n_combinations = self.get_n_splits()
        print(f"Combinatorial Purged CV:")
        print(f"  Groups: {self.n_splits}")
        print(f"  Test groups per combination: {self.n_test_splits}")
        print(f"  Total combinations: {n_combinations}")
        print(f"  Purge gap: {self.purge_gap}")
        print(f"  Embargo: {self.embargo_pct:.1%}")


def calculate_optimal_purge_gap(
    lookback_window: int,
    feature_decay: float = 1.0,
) -> int:
    """Calculate optimal purge gap based on feature lookback.

    Args:
        lookback_window: Maximum lookback window used in features.
        feature_decay: Factor to account for feature influence decay.
                      1.0 means full lookback, 0.5 means half lookback.

    Returns:
        Recommended purge gap in samples.

    Example:
        >>> # If using 50-period moving average
        >>> gap = calculate_optimal_purge_gap(lookback_window=50)
        >>> pkf = PurgedKFold(n_splits=5, purge_gap=gap)
    """
    return int(lookback_window * feature_decay)


def calculate_optimal_embargo(
    prediction_horizon: int,
    n_samples: int,
) -> float:
    """Calculate optimal embargo percentage.

    Args:
        prediction_horizon: How many periods ahead the strategy predicts.
        n_samples: Total number of samples in data.

    Returns:
        Recommended embargo percentage (0-0.1).

    Example:
        >>> embargo_pct = calculate_optimal_embargo(
        ...     prediction_horizon=5,
        ...     n_samples=1000
        ... )
        >>> pkf = PurgedKFold(n_splits=5, embargo_pct=embargo_pct)
    """
    if n_samples <= 0:
        return 0.01

    # Calculate embargo as percentage based on prediction horizon
    embargo_pct = prediction_horizon / n_samples

    # Clamp to reasonable range
    return min(0.1, max(0.01, embargo_pct))
