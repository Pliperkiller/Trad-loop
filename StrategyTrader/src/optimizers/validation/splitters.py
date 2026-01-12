"""Time series split strategies for IS/OOS validation.

Provides various strategies for splitting time series data
while respecting temporal order (no future data leakage).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Iterator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


@dataclass
class SplitInfo:
    """Information about a single train/test split."""

    split_idx: int
    train_start_idx: int
    train_end_idx: int
    test_start_idx: int
    test_end_idx: int
    train_size: int
    test_size: int
    train_start_date: Optional[datetime] = None
    train_end_date: Optional[datetime] = None
    test_start_date: Optional[datetime] = None
    test_end_date: Optional[datetime] = None


class BaseSplitter(ABC):
    """Abstract base class for time series splitters."""

    @abstractmethod
    def split(
        self, data: pd.DataFrame
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate train/test indices for each split.

        Args:
            data: DataFrame with time series data (index should be datetime).

        Yields:
            Tuple of (train_indices, test_indices) as numpy arrays.
        """
        pass

    @abstractmethod
    def get_n_splits(self, data: Optional[pd.DataFrame] = None) -> int:
        """Return the number of splits."""
        pass

    def get_split_info(self, data: pd.DataFrame) -> List[SplitInfo]:
        """Get detailed information about each split.

        Args:
            data: DataFrame with time series data.

        Returns:
            List of SplitInfo objects with details for each split.
        """
        splits_info = []
        has_datetime_index = isinstance(data.index, pd.DatetimeIndex)

        for idx, (train_idx, test_idx) in enumerate(self.split(data)):
            info = SplitInfo(
                split_idx=idx,
                train_start_idx=int(train_idx[0]),
                train_end_idx=int(train_idx[-1]),
                test_start_idx=int(test_idx[0]),
                test_end_idx=int(test_idx[-1]),
                train_size=len(train_idx),
                test_size=len(test_idx),
            )

            if has_datetime_index:
                dt_index = data.index  # type: ignore[assignment]
                info.train_start_date = dt_index[train_idx[0]].to_pydatetime()  # type: ignore[union-attr]
                info.train_end_date = dt_index[train_idx[-1]].to_pydatetime()  # type: ignore[union-attr]
                info.test_start_date = dt_index[test_idx[0]].to_pydatetime()  # type: ignore[union-attr]
                info.test_end_date = dt_index[test_idx[-1]].to_pydatetime()  # type: ignore[union-attr]

            splits_info.append(info)

        return splits_info

    def _validate_data(self, data: pd.DataFrame, min_rows: int = 10) -> None:
        """Validate input data.

        Args:
            data: DataFrame to validate.
            min_rows: Minimum required rows.

        Raises:
            ValueError: If data is invalid.
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame")
        if len(data) < min_rows:
            raise ValueError(f"Data must have at least {min_rows} rows")


class TimeSeriesSplit(BaseSplitter):
    """K-Fold time series cross-validation.

    Provides train/test indices to split time series data samples
    that are observed at fixed time intervals, in train/test sets.
    In each split, test indices must be higher than before, and thus
    shuffling in cross validator is inappropriate.

    This is similar to sklearn's TimeSeriesSplit but with additional
    features like gap between train and test.

    Example with n_splits=3:
        Split 1: Train[0:33] → Test[33:50]
        Split 2: Train[0:50] → Test[50:66]
        Split 3: Train[0:66] → Test[66:100]
    """

    def __init__(
        self,
        n_splits: int = 5,
        test_size: Optional[int] = None,
        gap: int = 0,
        max_train_size: Optional[int] = None,
    ):
        """Initialize TimeSeriesSplit.

        Args:
            n_splits: Number of splits (folds).
            test_size: Size of test set. If None, uses n_samples // (n_splits + 1).
            gap: Number of samples to exclude from end of train and beginning of test.
            max_train_size: Maximum size for train set (for rolling window behavior).
        """
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        if gap < 0:
            raise ValueError("gap must be non-negative")
        if max_train_size is not None and max_train_size < 1:
            raise ValueError("max_train_size must be positive")

        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap
        self.max_train_size = max_train_size

    def split(
        self, data: pd.DataFrame
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate train/test indices."""
        self._validate_data(data)
        n_samples = len(data)
        indices = np.arange(n_samples)

        # Calculate test size if not provided
        test_size = self.test_size
        if test_size is None:
            test_size = n_samples // (self.n_splits + 1)

        if test_size < 1:
            raise ValueError("test_size must be positive")

        # Calculate minimum train size needed
        min_train_size = n_samples - (test_size * self.n_splits) - (self.gap * self.n_splits)
        if min_train_size < 1:
            raise ValueError(
                f"Not enough samples ({n_samples}) for {self.n_splits} splits "
                f"with test_size={test_size} and gap={self.gap}"
            )

        # Generate splits
        test_starts = range(
            n_samples - self.n_splits * test_size,
            n_samples,
            test_size
        )

        for test_start in test_starts:
            train_end = test_start - self.gap
            if train_end < 1:
                continue

            train_start = 0
            if self.max_train_size is not None:
                train_start = max(0, train_end - self.max_train_size)

            test_end = min(test_start + test_size, n_samples)

            yield (
                indices[train_start:train_end],
                indices[test_start:test_end]
            )

    def get_n_splits(self, data: Optional[pd.DataFrame] = None) -> int:
        """Return the number of splits."""
        return self.n_splits


class RollingWindowSplit(BaseSplitter):
    """Rolling window (sliding window) time series split.

    Fixed-size training window that slides forward through the data.
    Each test period immediately follows its training period.

    Example with train_size=50, test_size=20, step=20:
        Split 1: Train[0:50]   → Test[50:70]
        Split 2: Train[20:70]  → Test[70:90]
        Split 3: Train[40:90]  → Test[90:110]
    """

    def __init__(
        self,
        train_size: int,
        test_size: int,
        step: Optional[int] = None,
        gap: int = 0,
    ):
        """Initialize RollingWindowSplit.

        Args:
            train_size: Fixed size of training window.
            test_size: Size of test set.
            step: Step size to move window. Defaults to test_size.
            gap: Number of samples between train and test.
        """
        if train_size < 1:
            raise ValueError("train_size must be positive")
        if test_size < 1:
            raise ValueError("test_size must be positive")
        if gap < 0:
            raise ValueError("gap must be non-negative")

        self.train_size = train_size
        self.test_size = test_size
        self.step = step if step is not None else test_size
        self.gap = gap

        if self.step < 1:
            raise ValueError("step must be positive")

    def split(
        self, data: pd.DataFrame
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate train/test indices."""
        self._validate_data(data, min_rows=self.train_size + self.test_size + self.gap)
        n_samples = len(data)
        indices = np.arange(n_samples)

        train_start = 0
        while True:
            train_end = train_start + self.train_size
            test_start = train_end + self.gap
            test_end = test_start + self.test_size

            if test_end > n_samples:
                break

            yield (
                indices[train_start:train_end],
                indices[test_start:test_end]
            )

            train_start += self.step

    def get_n_splits(self, data: Optional[pd.DataFrame] = None) -> int:
        """Return the number of splits."""
        if data is None:
            raise ValueError("data is required to compute n_splits")

        n_samples = len(data)
        min_required = self.train_size + self.test_size + self.gap

        if n_samples < min_required:
            return 0

        # Calculate number of windows that fit
        remaining = n_samples - min_required
        return 1 + remaining // self.step


class ExpandingWindowSplit(BaseSplitter):
    """Expanding (anchored) window time series split.

    Training window starts at the beginning and expands forward.
    Also known as anchored walk-forward.

    Example with initial_train_size=50, test_size=20, step=20:
        Split 1: Train[0:50]  → Test[50:70]
        Split 2: Train[0:70]  → Test[70:90]
        Split 3: Train[0:90]  → Test[90:110]
    """

    def __init__(
        self,
        initial_train_size: int,
        test_size: int,
        step: Optional[int] = None,
        gap: int = 0,
    ):
        """Initialize ExpandingWindowSplit.

        Args:
            initial_train_size: Initial size of training window.
            test_size: Size of test set.
            step: Step size to expand window. Defaults to test_size.
            gap: Number of samples between train and test.
        """
        if initial_train_size < 1:
            raise ValueError("initial_train_size must be positive")
        if test_size < 1:
            raise ValueError("test_size must be positive")
        if gap < 0:
            raise ValueError("gap must be non-negative")

        self.initial_train_size = initial_train_size
        self.test_size = test_size
        self.step = step if step is not None else test_size
        self.gap = gap

        if self.step < 1:
            raise ValueError("step must be positive")

    def split(
        self, data: pd.DataFrame
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate train/test indices."""
        min_rows = self.initial_train_size + self.test_size + self.gap
        self._validate_data(data, min_rows=min_rows)
        n_samples = len(data)
        indices = np.arange(n_samples)

        train_end = self.initial_train_size
        while True:
            test_start = train_end + self.gap
            test_end = test_start + self.test_size

            if test_end > n_samples:
                break

            yield (
                indices[0:train_end],  # Always starts at 0 (anchored)
                indices[test_start:test_end]
            )

            train_end += self.step

    def get_n_splits(self, data: Optional[pd.DataFrame] = None) -> int:
        """Return the number of splits."""
        if data is None:
            raise ValueError("data is required to compute n_splits")

        n_samples = len(data)
        min_required = self.initial_train_size + self.test_size + self.gap

        if n_samples < min_required:
            return 0

        # Calculate number of windows that fit
        remaining = n_samples - min_required
        return 1 + remaining // self.step


class TrainValTestSplit:
    """Static train/validation/test split.

    Splits data into three non-overlapping sequential sets.
    Useful for final model evaluation with held-out test set.
    """

    def __init__(
        self,
        train_pct: float = 0.6,
        val_pct: float = 0.2,
        test_pct: float = 0.2,
        gap: int = 0,
    ):
        """Initialize TrainValTestSplit.

        Args:
            train_pct: Percentage of data for training (0-1).
            val_pct: Percentage of data for validation (0-1).
            test_pct: Percentage of data for testing (0-1).
            gap: Number of samples between each split.
        """
        if not (0 < train_pct < 1):
            raise ValueError("train_pct must be between 0 and 1")
        if not (0 < val_pct < 1):
            raise ValueError("val_pct must be between 0 and 1")
        if not (0 < test_pct < 1):
            raise ValueError("test_pct must be between 0 and 1")

        total = train_pct + val_pct + test_pct
        if not (0.99 <= total <= 1.01):  # Allow small floating point error
            raise ValueError(
                f"train_pct + val_pct + test_pct must equal 1.0, got {total}"
            )

        if gap < 0:
            raise ValueError("gap must be non-negative")

        self.train_pct = train_pct
        self.val_pct = val_pct
        self.test_pct = test_pct
        self.gap = gap

    def split(
        self, data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train, validation, and test sets.

        Args:
            data: DataFrame to split.

        Returns:
            Tuple of (train_df, val_df, test_df).
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame")

        n_samples = len(data)
        min_rows = 10 + 2 * self.gap  # At least some data in each split
        if n_samples < min_rows:
            raise ValueError(f"Data must have at least {min_rows} rows")

        # Calculate split points
        train_end = int(n_samples * self.train_pct)
        val_start = train_end + self.gap
        val_end = val_start + int(n_samples * self.val_pct)
        test_start = val_end + self.gap

        if test_start >= n_samples:
            raise ValueError(
                f"Not enough data for gaps. Reduce gap or increase data size."
            )

        train_df = data.iloc[:train_end].copy()
        val_df = data.iloc[val_start:val_end].copy()
        test_df = data.iloc[test_start:].copy()

        return train_df, val_df, test_df

    def split_indices(
        self, data: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get indices for train, validation, and test sets.

        Args:
            data: DataFrame to split.

        Returns:
            Tuple of (train_indices, val_indices, test_indices).
        """
        n_samples = len(data)
        indices = np.arange(n_samples)

        train_end = int(n_samples * self.train_pct)
        val_start = train_end + self.gap
        val_end = val_start + int(n_samples * self.val_pct)
        test_start = val_end + self.gap

        return (
            indices[:train_end],
            indices[val_start:val_end],
            indices[test_start:]
        )

    def get_split_info(self, data: pd.DataFrame) -> dict:
        """Get information about the split.

        Args:
            data: DataFrame to split.

        Returns:
            Dictionary with split information.
        """
        train_idx, val_idx, test_idx = self.split_indices(data)
        has_datetime_index = isinstance(data.index, pd.DatetimeIndex)

        info = {
            "train_size": len(train_idx),
            "val_size": len(val_idx),
            "test_size": len(test_idx),
            "train_pct_actual": len(train_idx) / len(data),
            "val_pct_actual": len(val_idx) / len(data),
            "test_pct_actual": len(test_idx) / len(data),
        }

        if has_datetime_index and len(train_idx) > 0:
            info["train_start"] = data.index[train_idx[0]]
            info["train_end"] = data.index[train_idx[-1]]
            info["val_start"] = data.index[val_idx[0]]
            info["val_end"] = data.index[val_idx[-1]]
            info["test_start"] = data.index[test_idx[0]]
            info["test_end"] = data.index[test_idx[-1]]

        return info


class WalkForwardSplit(BaseSplitter):
    """Classical Walk-Forward split with non-overlapping test periods.

    This implements the traditional walk-forward analysis where:
    - Training window can be rolling (fixed size) or expanding (anchored)
    - Test periods are strictly sequential and non-overlapping
    - Optional gap between train and test to avoid lookahead bias

    Example with anchored=True, n_splits=3:
        Split 1: Train[0:60]  → Test[60:80]
        Split 2: Train[0:80]  → Test[80:100]
        Split 3: Train[0:100] → Test[100:120]

    Example with anchored=False (rolling), train_size=60:
        Split 1: Train[0:60]   → Test[60:80]
        Split 2: Train[20:80]  → Test[80:100]
        Split 3: Train[40:100] → Test[100:120]
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
        """Initialize WalkForwardSplit.

        Args:
            n_splits: Number of walk-forward splits.
            train_pct: Percentage of available data for training in first split.
            anchored: If True, use expanding window. If False, use rolling window.
            gap: Number of samples between train and test.
            min_train_size: Minimum training samples required.
            min_test_size: Minimum test samples required.
        """
        if n_splits < 1:
            raise ValueError("n_splits must be at least 1")
        if not (0.1 <= train_pct <= 0.9):
            raise ValueError("train_pct must be between 0.1 and 0.9")
        if gap < 0:
            raise ValueError("gap must be non-negative")

        self.n_splits = n_splits
        self.train_pct = train_pct
        self.anchored = anchored
        self.gap = gap
        self.min_train_size = min_train_size or 10
        self.min_test_size = min_test_size or 5

    def split(
        self, data: pd.DataFrame
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate train/test indices for walk-forward analysis."""
        self._validate_data(data)
        n_samples = len(data)
        indices = np.arange(n_samples)

        # Calculate initial train size and test size per split
        initial_train_size = int(n_samples * self.train_pct)
        remaining = n_samples - initial_train_size - (self.gap * self.n_splits)
        test_size_per_split = remaining // self.n_splits

        if test_size_per_split < self.min_test_size:
            raise ValueError(
                f"Test size per split ({test_size_per_split}) is less than "
                f"min_test_size ({self.min_test_size})"
            )

        if initial_train_size < self.min_train_size:
            raise ValueError(
                f"Initial train size ({initial_train_size}) is less than "
                f"min_train_size ({self.min_train_size})"
            )

        # For rolling window, calculate fixed train size
        rolling_train_size = initial_train_size

        for split_idx in range(self.n_splits):
            if self.anchored:
                # Expanding: train starts at 0
                train_start = 0
                train_end = initial_train_size + (split_idx * test_size_per_split)
            else:
                # Rolling: fixed train size, window slides
                train_start = split_idx * test_size_per_split
                train_end = train_start + rolling_train_size

            test_start = train_end + self.gap
            test_end = test_start + test_size_per_split

            # Ensure we don't exceed data bounds
            if test_end > n_samples:
                test_end = n_samples

            if test_start >= n_samples:
                break

            yield (
                indices[train_start:train_end],
                indices[test_start:test_end]
            )

    def get_n_splits(self, data: Optional[pd.DataFrame] = None) -> int:
        """Return the number of splits."""
        return self.n_splits


def create_splitter(
    method: str,
    **kwargs
) -> Union[BaseSplitter, TrainValTestSplit]:
    """Factory function to create a splitter by name.

    Args:
        method: Splitter method name. One of:
            - 'time_series': TimeSeriesSplit
            - 'rolling': RollingWindowSplit
            - 'expanding': ExpandingWindowSplit
            - 'train_val_test': TrainValTestSplit
            - 'walk_forward': WalkForwardSplit
        **kwargs: Arguments passed to the splitter constructor.

    Returns:
        Splitter instance.

    Raises:
        ValueError: If method is unknown.
    """
    splitters = {
        'time_series': TimeSeriesSplit,
        'rolling': RollingWindowSplit,
        'expanding': ExpandingWindowSplit,
        'train_val_test': TrainValTestSplit,
        'walk_forward': WalkForwardSplit,
    }

    if method not in splitters:
        raise ValueError(
            f"Unknown splitter method: {method}. "
            f"Available: {list(splitters.keys())}"
        )

    return splitters[method](**kwargs)
