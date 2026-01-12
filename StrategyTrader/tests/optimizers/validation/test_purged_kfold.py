"""Tests for Purged K-Fold cross-validation."""

import pytest
import numpy as np
import pandas as pd

from src.optimizers.validation.purged_kfold import (
    PurgedKFold,
    CombinatorialPurgedCV,
    calculate_optimal_purge_gap,
    calculate_optimal_embargo,
)


class TestPurgedKFold:
    """Tests for PurgedKFold class."""

    def test_initialization(self):
        """PurgedKFold initializes correctly."""
        pkf = PurgedKFold(n_splits=5, purge_gap=10, embargo_pct=0.02)

        assert pkf.n_splits == 5
        assert pkf.purge_gap == 10
        assert pkf.embargo_pct == 0.02

    def test_invalid_n_splits(self):
        """Invalid n_splits raises error."""
        with pytest.raises(ValueError):
            PurgedKFold(n_splits=1)

    def test_invalid_purge_gap(self):
        """Invalid purge_gap raises error."""
        with pytest.raises(ValueError):
            PurgedKFold(n_splits=5, purge_gap=-1)

    def test_invalid_embargo(self):
        """Invalid embargo_pct raises error."""
        with pytest.raises(ValueError):
            PurgedKFold(n_splits=5, embargo_pct=0.2)

    def test_basic_split(self, sample_time_series):
        """Basic purged split works."""
        pkf = PurgedKFold(n_splits=5)
        splits = list(pkf.split(sample_time_series))

        assert len(splits) == 5

    def test_no_overlap_train_test(self, sample_time_series):
        """Train and test sets don't overlap."""
        pkf = PurgedKFold(n_splits=5)

        for train_idx, test_idx in pkf.split(sample_time_series):
            train_set = set(train_idx)
            test_set = set(test_idx)
            assert len(train_set & test_set) == 0

    def test_purge_gap_applied(self, sample_time_series):
        """Purge gap removes samples before test."""
        purge_gap = 20
        pkf = PurgedKFold(n_splits=5, purge_gap=purge_gap)

        for train_idx, test_idx in pkf.split(sample_time_series):
            test_start = test_idx[0]

            # No train sample should be within purge_gap of test start
            for train_i in train_idx:
                if train_i < test_start:
                    assert test_start - train_i > purge_gap or train_i < test_start - purge_gap

    def test_embargo_applied(self, sample_time_series):
        """Embargo excludes samples after test."""
        embargo_pct = 0.05
        pkf = PurgedKFold(n_splits=5, embargo_pct=embargo_pct)
        n_samples = len(sample_time_series)
        embargo_size = int(n_samples * embargo_pct)

        for train_idx, test_idx in pkf.split(sample_time_series):
            test_end = test_idx[-1]

            # No train sample should be within embargo of test end
            for train_i in train_idx:
                if train_i > test_end:
                    assert train_i - test_end >= embargo_size

    def test_get_n_splits(self, sample_time_series):
        """get_n_splits returns correct value."""
        pkf = PurgedKFold(n_splits=5)
        assert pkf.get_n_splits(sample_time_series) == 5

    def test_get_split_info(self, sample_time_series):
        """get_split_info returns detailed info."""
        pkf = PurgedKFold(n_splits=5, purge_gap=10, embargo_pct=0.02)
        info = pkf.get_split_info(sample_time_series)

        assert len(info) == 5
        for split_info in info:
            assert hasattr(split_info, 'purged_samples')
            assert hasattr(split_info, 'embargo_samples')
            assert hasattr(split_info, 'effective_train_size')

    def test_reduced_train_size(self, sample_time_series):
        """Purging and embargo reduce effective train size."""
        pkf_no_purge = PurgedKFold(n_splits=5, purge_gap=0, embargo_pct=0)
        pkf_with_purge = PurgedKFold(n_splits=5, purge_gap=20, embargo_pct=0.05)

        splits_no_purge = list(pkf_no_purge.split(sample_time_series))
        splits_with_purge = list(pkf_with_purge.split(sample_time_series))

        # Training sets should be smaller with purging
        for (train_np, _), (train_p, _) in zip(splits_no_purge, splits_with_purge):
            assert len(train_p) <= len(train_np)


class TestCombinatorialPurgedCV:
    """Tests for Combinatorial Purged CV."""

    def test_initialization(self):
        """CPCV initializes correctly."""
        cpcv = CombinatorialPurgedCV(n_splits=5, n_test_splits=2)

        assert cpcv.n_splits == 5
        assert cpcv.n_test_splits == 2

    def test_invalid_n_splits(self):
        """Invalid n_splits raises error."""
        with pytest.raises(ValueError):
            CombinatorialPurgedCV(n_splits=2, n_test_splits=1)

    def test_invalid_n_test_splits(self):
        """Invalid n_test_splits raises error."""
        with pytest.raises(ValueError):
            CombinatorialPurgedCV(n_splits=5, n_test_splits=5)

    def test_correct_number_combinations(self, sample_time_series):
        """Generates correct number of combinations."""
        from math import comb

        cpcv = CombinatorialPurgedCV(n_splits=5, n_test_splits=2)
        expected_combinations = comb(5, 2)

        n_splits = cpcv.get_n_splits(sample_time_series)
        assert n_splits == expected_combinations

    def test_basic_split(self, sample_time_series):
        """Basic CPCV split works."""
        cpcv = CombinatorialPurgedCV(n_splits=5, n_test_splits=2)
        splits = list(cpcv.split(sample_time_series))

        assert len(splits) > 0

    def test_no_overlap_train_test(self, sample_time_series):
        """Train and test don't overlap in any combination."""
        cpcv = CombinatorialPurgedCV(n_splits=5, n_test_splits=2)

        for train_idx, test_idx in cpcv.split(sample_time_series):
            train_set = set(train_idx)
            test_set = set(test_idx)
            assert len(train_set & test_set) == 0

    def test_purge_applied(self, sample_time_series):
        """Purge gap is applied in CPCV."""
        cpcv = CombinatorialPurgedCV(n_splits=5, n_test_splits=2, purge_gap=10)

        # Should run without error
        splits = list(cpcv.split(sample_time_series))
        assert len(splits) > 0

    def test_embargo_applied(self, sample_time_series):
        """Embargo is applied in CPCV."""
        cpcv = CombinatorialPurgedCV(n_splits=5, n_test_splits=2, embargo_pct=0.02)

        # Should run without error
        splits = list(cpcv.split(sample_time_series))
        assert len(splits) > 0


class TestOptimalPurgeGap:
    """Tests for calculate_optimal_purge_gap."""

    def test_basic_calculation(self):
        """Basic purge gap calculation."""
        gap = calculate_optimal_purge_gap(lookback_window=50)
        assert gap == 50

    def test_with_decay_factor(self):
        """Decay factor reduces gap."""
        gap = calculate_optimal_purge_gap(lookback_window=50, feature_decay=0.5)
        assert gap == 25

    def test_zero_lookback(self):
        """Zero lookback returns zero gap."""
        gap = calculate_optimal_purge_gap(lookback_window=0)
        assert gap == 0


class TestOptimalEmbargo:
    """Tests for calculate_optimal_embargo."""

    def test_basic_calculation(self):
        """Basic embargo calculation."""
        embargo = calculate_optimal_embargo(
            prediction_horizon=5,
            n_samples=1000,
        )

        assert 0.01 <= embargo <= 0.1

    def test_larger_horizon_larger_embargo(self):
        """Larger prediction horizon results in larger embargo."""
        embargo_short = calculate_optimal_embargo(prediction_horizon=1, n_samples=1000)
        embargo_long = calculate_optimal_embargo(prediction_horizon=20, n_samples=1000)

        assert embargo_long > embargo_short

    def test_minimum_embargo(self):
        """Embargo has minimum value."""
        embargo = calculate_optimal_embargo(
            prediction_horizon=0,
            n_samples=10000,
        )

        assert embargo >= 0.01

    def test_maximum_embargo(self):
        """Embargo has maximum value."""
        embargo = calculate_optimal_embargo(
            prediction_horizon=1000,
            n_samples=100,
        )

        assert embargo <= 0.1

    def test_zero_samples(self):
        """Zero samples returns minimum embargo."""
        embargo = calculate_optimal_embargo(
            prediction_horizon=5,
            n_samples=0,
        )

        assert embargo == 0.01
