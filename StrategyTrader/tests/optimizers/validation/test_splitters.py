"""Tests for time series splitters."""

import pytest
import numpy as np
import pandas as pd

from src.optimizers.validation.splitters import (
    TimeSeriesSplit,
    RollingWindowSplit,
    ExpandingWindowSplit,
    TrainValTestSplit,
    WalkForwardSplit,
    create_splitter,
)


class TestTimeSeriesSplit:
    """Tests for TimeSeriesSplit."""

    def test_basic_split(self, sample_time_series):
        """Basic split produces correct number of folds."""
        splitter = TimeSeriesSplit(n_splits=5)
        splits = list(splitter.split(sample_time_series))

        assert len(splits) == 5

    def test_train_test_ordering(self, sample_time_series):
        """Train indices always come before test indices."""
        splitter = TimeSeriesSplit(n_splits=5)

        for train_idx, test_idx in splitter.split(sample_time_series):
            assert train_idx[-1] < test_idx[0]

    def test_test_indices_sequential(self, sample_time_series):
        """Test indices are sequential across folds."""
        splitter = TimeSeriesSplit(n_splits=5)
        splits = list(splitter.split(sample_time_series))

        for i in range(1, len(splits)):
            prev_test_end = splits[i-1][1][-1]
            curr_test_start = splits[i][1][0]
            # Test periods should be sequential
            assert curr_test_start > prev_test_end

    def test_gap_parameter(self, sample_time_series):
        """Gap parameter creates separation between train and test."""
        gap = 10
        splitter = TimeSeriesSplit(n_splits=3, gap=gap)

        for train_idx, test_idx in splitter.split(sample_time_series):
            actual_gap = test_idx[0] - train_idx[-1] - 1
            assert actual_gap >= gap

    def test_max_train_size(self, sample_time_series):
        """Max train size limits training data."""
        max_train = 100
        splitter = TimeSeriesSplit(n_splits=3, max_train_size=max_train)

        for train_idx, test_idx in splitter.split(sample_time_series):
            assert len(train_idx) <= max_train

    def test_custom_test_size(self, sample_time_series):
        """Custom test size is respected."""
        test_size = 30
        splitter = TimeSeriesSplit(n_splits=3, test_size=test_size)

        for train_idx, test_idx in splitter.split(sample_time_series):
            assert len(test_idx) == test_size

    def test_invalid_n_splits(self):
        """Invalid n_splits raises error."""
        with pytest.raises(ValueError):
            TimeSeriesSplit(n_splits=1)

    def test_get_n_splits(self, sample_time_series):
        """get_n_splits returns correct value."""
        splitter = TimeSeriesSplit(n_splits=5)
        assert splitter.get_n_splits(sample_time_series) == 5

    def test_get_split_info(self, sample_time_series):
        """get_split_info returns correct information."""
        splitter = TimeSeriesSplit(n_splits=3)
        info = splitter.get_split_info(sample_time_series)

        assert len(info) == 3
        for split_info in info:
            assert split_info.train_size > 0
            assert split_info.test_size > 0
            assert split_info.train_start_date is not None


class TestRollingWindowSplit:
    """Tests for RollingWindowSplit."""

    def test_basic_split(self, sample_time_series):
        """Basic rolling window split works."""
        splitter = RollingWindowSplit(train_size=100, test_size=20)
        splits = list(splitter.split(sample_time_series))

        assert len(splits) > 0

    def test_fixed_train_size(self, sample_time_series):
        """Train size is fixed across splits."""
        train_size = 100
        splitter = RollingWindowSplit(train_size=train_size, test_size=20)

        for train_idx, test_idx in splitter.split(sample_time_series):
            assert len(train_idx) == train_size

    def test_fixed_test_size(self, sample_time_series):
        """Test size is fixed across splits."""
        test_size = 20
        splitter = RollingWindowSplit(train_size=100, test_size=test_size)

        for train_idx, test_idx in splitter.split(sample_time_series):
            assert len(test_idx) == test_size

    def test_window_slides(self, sample_time_series):
        """Window slides forward between splits."""
        splitter = RollingWindowSplit(train_size=100, test_size=20, step=20)
        splits = list(splitter.split(sample_time_series))

        for i in range(1, len(splits)):
            prev_train_start = splits[i-1][0][0]
            curr_train_start = splits[i][0][0]
            assert curr_train_start > prev_train_start

    def test_custom_step(self, sample_time_series):
        """Custom step size is respected."""
        step = 50
        splitter = RollingWindowSplit(train_size=100, test_size=20, step=step)
        splits = list(splitter.split(sample_time_series))

        if len(splits) > 1:
            diff = splits[1][0][0] - splits[0][0][0]
            assert diff == step

    def test_gap_parameter(self, sample_time_series):
        """Gap creates separation between train and test."""
        gap = 5
        splitter = RollingWindowSplit(train_size=100, test_size=20, gap=gap)

        for train_idx, test_idx in splitter.split(sample_time_series):
            actual_gap = test_idx[0] - train_idx[-1] - 1
            assert actual_gap >= gap

    def test_invalid_train_size(self):
        """Invalid train size raises error."""
        with pytest.raises(ValueError):
            RollingWindowSplit(train_size=0, test_size=20)

    def test_get_n_splits(self, sample_time_series):
        """get_n_splits calculates correctly."""
        splitter = RollingWindowSplit(train_size=100, test_size=50, step=50)
        n_splits = splitter.get_n_splits(sample_time_series)

        actual_splits = list(splitter.split(sample_time_series))
        assert n_splits == len(actual_splits)


class TestExpandingWindowSplit:
    """Tests for ExpandingWindowSplit."""

    def test_basic_split(self, sample_time_series):
        """Basic expanding window split works."""
        splitter = ExpandingWindowSplit(initial_train_size=100, test_size=20)
        splits = list(splitter.split(sample_time_series))

        assert len(splits) > 0

    def test_train_expands(self, sample_time_series):
        """Train size expands between splits."""
        splitter = ExpandingWindowSplit(initial_train_size=100, test_size=20)
        splits = list(splitter.split(sample_time_series))

        for i in range(1, len(splits)):
            prev_train_size = len(splits[i-1][0])
            curr_train_size = len(splits[i][0])
            assert curr_train_size > prev_train_size

    def test_train_always_starts_at_zero(self, sample_time_series):
        """Train always starts at index 0 (anchored)."""
        splitter = ExpandingWindowSplit(initial_train_size=100, test_size=20)

        for train_idx, test_idx in splitter.split(sample_time_series):
            assert train_idx[0] == 0

    def test_fixed_test_size(self, sample_time_series):
        """Test size is fixed across splits."""
        test_size = 30
        splitter = ExpandingWindowSplit(initial_train_size=100, test_size=test_size)

        for train_idx, test_idx in splitter.split(sample_time_series):
            assert len(test_idx) == test_size

    def test_gap_parameter(self, sample_time_series):
        """Gap creates separation between train and test."""
        gap = 10
        splitter = ExpandingWindowSplit(initial_train_size=100, test_size=20, gap=gap)

        for train_idx, test_idx in splitter.split(sample_time_series):
            actual_gap = test_idx[0] - train_idx[-1] - 1
            assert actual_gap >= gap


class TestTrainValTestSplit:
    """Tests for TrainValTestSplit."""

    def test_basic_split(self, sample_time_series):
        """Basic train/val/test split works."""
        splitter = TrainValTestSplit(train_pct=0.6, val_pct=0.2, test_pct=0.2)
        train, val, test = splitter.split(sample_time_series)

        assert len(train) > 0
        assert len(val) > 0
        assert len(test) > 0

    def test_no_overlap(self, sample_time_series):
        """Train, val, test sets don't overlap."""
        splitter = TrainValTestSplit(train_pct=0.6, val_pct=0.2, test_pct=0.2)
        train, val, test = splitter.split(sample_time_series)

        train_set = set(train.index)
        val_set = set(val.index)
        test_set = set(test.index)

        assert len(train_set & val_set) == 0
        assert len(val_set & test_set) == 0
        assert len(train_set & test_set) == 0

    def test_correct_ordering(self, sample_time_series):
        """Sets are in correct chronological order."""
        splitter = TrainValTestSplit(train_pct=0.6, val_pct=0.2, test_pct=0.2)
        train, val, test = splitter.split(sample_time_series)

        assert train.index[-1] < val.index[0]
        assert val.index[-1] < test.index[0]

    def test_approximate_sizes(self, sample_time_series):
        """Sizes are approximately correct."""
        splitter = TrainValTestSplit(train_pct=0.6, val_pct=0.2, test_pct=0.2)
        train, val, test = splitter.split(sample_time_series)
        total = len(sample_time_series)

        # Allow some tolerance due to rounding
        assert abs(len(train) / total - 0.6) < 0.05
        assert abs(len(val) / total - 0.2) < 0.05

    def test_gap_parameter(self, sample_time_series):
        """Gap creates separation between sets."""
        gap = 10
        splitter = TrainValTestSplit(train_pct=0.6, val_pct=0.2, test_pct=0.2, gap=gap)
        train_idx, val_idx, test_idx = splitter.split_indices(sample_time_series)

        # Check gap between train and val
        assert val_idx[0] - train_idx[-1] > gap

    def test_invalid_percentages(self):
        """Invalid percentages raise error."""
        with pytest.raises(ValueError):
            TrainValTestSplit(train_pct=0.5, val_pct=0.5, test_pct=0.5)

    def test_get_split_info(self, sample_time_series):
        """get_split_info returns correct information."""
        splitter = TrainValTestSplit(train_pct=0.6, val_pct=0.2, test_pct=0.2)
        info = splitter.get_split_info(sample_time_series)

        assert 'train_size' in info
        assert 'val_size' in info
        assert 'test_size' in info


class TestWalkForwardSplit:
    """Tests for WalkForwardSplit."""

    def test_basic_split(self, sample_time_series):
        """Basic walk-forward split works."""
        splitter = WalkForwardSplit(n_splits=5, train_pct=0.6)
        splits = list(splitter.split(sample_time_series))

        assert len(splits) == 5

    def test_non_overlapping_tests(self, sample_time_series):
        """Test periods don't overlap (classical WF)."""
        splitter = WalkForwardSplit(n_splits=5, train_pct=0.6)
        splits = list(splitter.split(sample_time_series))

        for i in range(1, len(splits)):
            prev_test = set(splits[i-1][1])
            curr_test = set(splits[i][1])
            assert len(prev_test & curr_test) == 0

    def test_anchored_mode(self, sample_time_series):
        """Anchored mode expands training window."""
        splitter = WalkForwardSplit(n_splits=3, train_pct=0.6, anchored=True)
        splits = list(splitter.split(sample_time_series))

        # All trains should start at 0
        for train_idx, _ in splits:
            assert train_idx[0] == 0

        # Train sizes should increase
        for i in range(1, len(splits)):
            assert len(splits[i][0]) > len(splits[i-1][0])

    def test_rolling_mode(self, sample_time_series):
        """Rolling mode maintains fixed training size."""
        splitter = WalkForwardSplit(n_splits=3, train_pct=0.6, anchored=False)
        splits = list(splitter.split(sample_time_series))

        # Train sizes should be approximately equal
        train_sizes = [len(train_idx) for train_idx, _ in splits]
        assert max(train_sizes) - min(train_sizes) <= 10  # Small tolerance

    def test_gap_parameter(self, sample_time_series):
        """Gap creates separation between train and test."""
        gap = 10
        splitter = WalkForwardSplit(n_splits=3, train_pct=0.6, gap=gap)

        for train_idx, test_idx in splitter.split(sample_time_series):
            actual_gap = test_idx[0] - train_idx[-1] - 1
            assert actual_gap >= gap


class TestCreateSplitter:
    """Tests for create_splitter factory function."""

    def test_create_time_series(self):
        """Create time_series splitter."""
        splitter = create_splitter('time_series', n_splits=5)
        assert isinstance(splitter, TimeSeriesSplit)

    def test_create_rolling(self):
        """Create rolling splitter."""
        splitter = create_splitter('rolling', train_size=100, test_size=20)
        assert isinstance(splitter, RollingWindowSplit)

    def test_create_expanding(self):
        """Create expanding splitter."""
        splitter = create_splitter('expanding', initial_train_size=100, test_size=20)
        assert isinstance(splitter, ExpandingWindowSplit)

    def test_create_train_val_test(self):
        """Create train_val_test splitter."""
        splitter = create_splitter('train_val_test')
        assert isinstance(splitter, TrainValTestSplit)

    def test_create_walk_forward(self):
        """Create walk_forward splitter."""
        splitter = create_splitter('walk_forward', n_splits=5)
        assert isinstance(splitter, WalkForwardSplit)

    def test_invalid_method(self):
        """Invalid method raises error."""
        with pytest.raises(ValueError):
            create_splitter('invalid_method')
