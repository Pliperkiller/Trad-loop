"""Tests for Pivot Points indicator."""

import pytest
import numpy as np
import pandas as pd

from src.indicators.technical.pivots import pivot_points, pivot_points_series


class TestPivotPoints:
    """Tests for Pivot Points."""

    def test_pivot_classic_basic(self):
        """Classic pivot points produces valid output."""
        result = pivot_points(
            high=110.0,
            low=100.0,
            close=105.0,
            method="classic"
        )

        assert hasattr(result, "pivot")
        assert hasattr(result, "r1")
        assert hasattr(result, "r2")
        assert hasattr(result, "r3")
        assert hasattr(result, "s1")
        assert hasattr(result, "s2")
        assert hasattr(result, "s3")
        assert result.method == "classic"

    def test_pivot_classic_calculation(self):
        """Classic pivot calculation is correct."""
        high, low, close = 110.0, 100.0, 105.0

        result = pivot_points(high, low, close, method="classic")

        # Pivot = (H + L + C) / 3
        expected_pivot = (110 + 100 + 105) / 3
        assert abs(result.pivot - expected_pivot) < 0.01

        # R1 = 2P - L
        expected_r1 = (2 * expected_pivot) - low
        assert abs(result.r1 - expected_r1) < 0.01

        # S1 = 2P - H
        expected_s1 = (2 * expected_pivot) - high
        assert abs(result.s1 - expected_s1) < 0.01

    def test_pivot_fibonacci(self):
        """Fibonacci pivot points produces valid output."""
        result = pivot_points(
            high=110.0,
            low=100.0,
            close=105.0,
            method="fibonacci"
        )

        assert result.method == "fibonacci"

        # Fib levels use 38.2%, 61.8%, 100%
        range_ = 110.0 - 100.0
        pivot = (110 + 100 + 105) / 3

        expected_r1 = pivot + (0.382 * range_)
        expected_r2 = pivot + (0.618 * range_)

        assert abs(result.r1 - expected_r1) < 0.01
        assert abs(result.r2 - expected_r2) < 0.01

    def test_pivot_woodie(self):
        """Woodie pivot points produces valid output."""
        result = pivot_points(
            high=110.0,
            low=100.0,
            close=105.0,
            open_=102.0,
            method="woodie"
        )

        assert result.method == "woodie"

        # Woodie Pivot = (H + L + 2C) / 4
        expected_pivot = (110 + 100 + 2 * 105) / 4
        assert abs(result.pivot - expected_pivot) < 0.01

    def test_pivot_camarilla(self):
        """Camarilla pivot points produces valid output."""
        result = pivot_points(
            high=110.0,
            low=100.0,
            close=105.0,
            method="camarilla"
        )

        assert result.method == "camarilla"

        # Camarilla R1 = C + (Range * 1.1/12)
        range_ = 110.0 - 100.0
        expected_r1 = 105 + (range_ * 1.1 / 12)
        assert abs(result.r1 - expected_r1) < 0.01

    def test_pivot_resistance_support_ordering(self):
        """R3 > R2 > R1 > Pivot > S1 > S2 > S3."""
        result = pivot_points(high=110.0, low=100.0, close=105.0)

        assert result.r3 > result.r2
        assert result.r2 > result.r1
        assert result.r1 > result.pivot
        assert result.pivot > result.s1
        assert result.s1 > result.s2
        assert result.s2 > result.s3

    def test_pivot_accepts_series(self, sample_ohlcv):
        """Pivot points accepts Series (uses last value)."""
        result = pivot_points(
            high=sample_ohlcv["high"],
            low=sample_ohlcv["low"],
            close=sample_ohlcv["close"]
        )

        # Should use last values from series
        expected_pivot = (
            sample_ohlcv["high"].iloc[-1] +
            sample_ohlcv["low"].iloc[-1] +
            sample_ohlcv["close"].iloc[-1]
        ) / 3

        assert abs(result.pivot - expected_pivot) < 0.01

    def test_pivot_invalid_method(self):
        """Invalid method raises error."""
        with pytest.raises(ValueError):
            pivot_points(high=110.0, low=100.0, close=105.0, method="invalid")

    def test_pivot_woodie_requires_open(self):
        """Woodie method requires open_ parameter."""
        with pytest.raises(ValueError):
            pivot_points(high=110.0, low=100.0, close=105.0, method="woodie")


class TestPivotPointsSeries:
    """Tests for Pivot Points Series calculation."""

    def test_pivot_series_basic(self, sample_ohlcv):
        """Pivot points series produces DataFrame."""
        result = pivot_points_series(
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"]
        )

        assert isinstance(result, pd.DataFrame)
        assert "pivot" in result.columns
        assert "r1" in result.columns
        assert "s1" in result.columns
        assert len(result) == len(sample_ohlcv)

    def test_pivot_series_first_row_nan(self, sample_ohlcv):
        """First row should be NaN (no previous data)."""
        result = pivot_points_series(
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"]
        )

        assert pd.isna(result["pivot"].iloc[0])
        assert pd.isna(result["r1"].iloc[0])

    def test_pivot_series_uses_previous_bar(self, sample_ohlcv):
        """Each row uses previous bar's data."""
        result = pivot_points_series(
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"]
        )

        # Row i should be calculated from bar i-1
        i = 5
        expected = pivot_points(
            sample_ohlcv["high"].iloc[i-1],
            sample_ohlcv["low"].iloc[i-1],
            sample_ohlcv["close"].iloc[i-1]
        )

        assert abs(result["pivot"].iloc[i] - expected.pivot) < 0.01
        assert abs(result["r1"].iloc[i] - expected.r1) < 0.01

    def test_pivot_series_all_methods(self, sample_ohlcv):
        """Pivot series works with all methods."""
        for method in ["classic", "fibonacci", "camarilla"]:
            result = pivot_points_series(
                sample_ohlcv["high"],
                sample_ohlcv["low"],
                sample_ohlcv["close"],
                method=method
            )
            assert result["pivot"].iloc[1:].notna().all()

    def test_pivot_series_woodie_with_open(self, sample_ohlcv):
        """Woodie series works with open data."""
        result = pivot_points_series(
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"],
            open_=sample_ohlcv["open"],
            method="woodie"
        )

        assert result["pivot"].iloc[1:].notna().all()
