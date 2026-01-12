"""Pivot Points indicator (Classic, Fibonacci, Woodie, Camarilla)."""

from typing import Union, Literal
import pandas as pd
import numpy as np

from src.indicators.base import (
    validate_series,
    PivotPointsResult,
)


PivotMethod = Literal["classic", "fibonacci", "woodie", "camarilla"]


def pivot_points(
    high: Union[pd.Series, np.ndarray, list, float],
    low: Union[pd.Series, np.ndarray, list, float],
    close: Union[pd.Series, np.ndarray, list, float],
    open_: Union[pd.Series, np.ndarray, list, float, None] = None,
    method: PivotMethod = "classic"
) -> PivotPointsResult:
    """
    Calculate Pivot Points with support and resistance levels.

    Pivot points are calculated from previous period's OHLC data.
    Can use single values (daily) or Series (for multiple periods).

    Methods:
    - classic: Standard pivot point calculation
    - fibonacci: Uses Fibonacci ratios for S/R levels
    - woodie: Gives more weight to closing price
    - camarilla: Focuses on intraday trading levels

    Args:
        high: Previous period high (float or Series)
        low: Previous period low (float or Series)
        close: Previous period close (float or Series)
        open_: Previous period open (only used for woodie method)
        method: Calculation method ('classic', 'fibonacci', 'woodie', 'camarilla')

    Returns:
        PivotPointsResult with pivot, r1-r3, s1-s3, and method

    Raises:
        ValueError: If method is invalid or open_ is missing for woodie method
    """
    # Convert to float if single values
    if isinstance(high, (pd.Series, np.ndarray, list)):
        high = validate_series(high, "high")
        high = float(high.iloc[-1])

    if isinstance(low, (pd.Series, np.ndarray, list)):
        low = validate_series(low, "low")
        low = float(low.iloc[-1])

    if isinstance(close, (pd.Series, np.ndarray, list)):
        close = validate_series(close, "close")
        close = float(close.iloc[-1])

    if open_ is not None and isinstance(open_, (pd.Series, np.ndarray, list)):
        open_ = validate_series(open_, "open")
        open_ = float(open_.iloc[-1])

    valid_methods = ["classic", "fibonacci", "woodie", "camarilla"]
    if method not in valid_methods:
        raise ValueError(f"method must be one of {valid_methods}")

    if method == "woodie" and open_ is None:
        raise ValueError("open_ is required for woodie method")

    range_ = high - low

    if method == "classic":
        return _classic_pivots(high, low, close, range_)
    elif method == "fibonacci":
        return _fibonacci_pivots(high, low, close, range_)
    elif method == "woodie":
        return _woodie_pivots(high, low, close, open_, range_)
    else:  # camarilla
        return _camarilla_pivots(high, low, close, range_)


def _classic_pivots(high: float, low: float, close: float, range_: float) -> PivotPointsResult:
    """
    Classic (Floor) Pivot Points.

    Pivot = (H + L + C) / 3
    R1 = 2P - L
    R2 = P + Range
    R3 = R1 + Range
    S1 = 2P - H
    S2 = P - Range
    S3 = S1 - Range
    """
    pivot = (high + low + close) / 3

    r1 = (2 * pivot) - low
    r2 = pivot + range_
    r3 = r1 + range_

    s1 = (2 * pivot) - high
    s2 = pivot - range_
    s3 = s1 - range_

    return PivotPointsResult(
        pivot=pivot,
        r1=r1, r2=r2, r3=r3,
        s1=s1, s2=s2, s3=s3,
        method="classic"
    )


def _fibonacci_pivots(high: float, low: float, close: float, range_: float) -> PivotPointsResult:
    """
    Fibonacci Pivot Points.

    Uses Fibonacci retracement levels (38.2%, 61.8%, 100%)
    Pivot = (H + L + C) / 3
    R1 = P + (0.382 * Range)
    R2 = P + (0.618 * Range)
    R3 = P + Range
    S1 = P - (0.382 * Range)
    S2 = P - (0.618 * Range)
    S3 = P - Range
    """
    pivot = (high + low + close) / 3

    r1 = pivot + (0.382 * range_)
    r2 = pivot + (0.618 * range_)
    r3 = pivot + range_

    s1 = pivot - (0.382 * range_)
    s2 = pivot - (0.618 * range_)
    s3 = pivot - range_

    return PivotPointsResult(
        pivot=pivot,
        r1=r1, r2=r2, r3=r3,
        s1=s1, s2=s2, s3=s3,
        method="fibonacci"
    )


def _woodie_pivots(
    high: float, low: float, close: float, open_: float, range_: float
) -> PivotPointsResult:
    """
    Woodie Pivot Points.

    Gives more weight to closing price.
    Pivot = (H + L + 2C) / 4
    R1 = 2P - L
    R2 = P + Range
    R3 = H + 2 * (P - L)
    S1 = 2P - H
    S2 = P - Range
    S3 = L - 2 * (H - P)
    """
    pivot = (high + low + (2 * close)) / 4

    r1 = (2 * pivot) - low
    r2 = pivot + range_
    r3 = high + (2 * (pivot - low))

    s1 = (2 * pivot) - high
    s2 = pivot - range_
    s3 = low - (2 * (high - pivot))

    return PivotPointsResult(
        pivot=pivot,
        r1=r1, r2=r2, r3=r3,
        s1=s1, s2=s2, s3=s3,
        method="woodie"
    )


def _camarilla_pivots(high: float, low: float, close: float, range_: float) -> PivotPointsResult:
    """
    Camarilla Pivot Points.

    Designed for intraday trading, levels are tighter.
    Pivot = (H + L + C) / 3
    R1 = C + (Range * 1.1/12)
    R2 = C + (Range * 1.1/6)
    R3 = C + (Range * 1.1/4)
    S1 = C - (Range * 1.1/12)
    S2 = C - (Range * 1.1/6)
    S3 = C - (Range * 1.1/4)
    """
    pivot = (high + low + close) / 3

    r1 = close + (range_ * 1.1 / 12)
    r2 = close + (range_ * 1.1 / 6)
    r3 = close + (range_ * 1.1 / 4)

    s1 = close - (range_ * 1.1 / 12)
    s2 = close - (range_ * 1.1 / 6)
    s3 = close - (range_ * 1.1 / 4)

    return PivotPointsResult(
        pivot=pivot,
        r1=r1, r2=r2, r3=r3,
        s1=s1, s2=s2, s3=s3,
        method="camarilla"
    )


def pivot_points_series(
    high: Union[pd.Series, np.ndarray, list],
    low: Union[pd.Series, np.ndarray, list],
    close: Union[pd.Series, np.ndarray, list],
    open_: Union[pd.Series, np.ndarray, list, None] = None,
    method: PivotMethod = "classic"
) -> pd.DataFrame:
    """
    Calculate Pivot Points for each bar in a series.

    Each row uses the previous bar's OHLC to calculate pivot points
    for the current bar.

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        open_: Open prices (only used for woodie method)
        method: Calculation method

    Returns:
        DataFrame with pivot, r1-r3, s1-s3 columns
    """
    high = validate_series(high, "high")
    low = validate_series(low, "low")
    close = validate_series(close, "close")

    if open_ is not None:
        open_ = validate_series(open_, "open")

    n = len(close)
    results = {
        'pivot': np.zeros(n),
        'r1': np.zeros(n),
        'r2': np.zeros(n),
        'r3': np.zeros(n),
        's1': np.zeros(n),
        's2': np.zeros(n),
        's3': np.zeros(n),
    }

    for i in range(1, n):
        pp = pivot_points(
            high.iloc[i-1],
            low.iloc[i-1],
            close.iloc[i-1],
            open_.iloc[i-1] if open_ is not None else None,
            method=method
        )
        results['pivot'][i] = pp.pivot
        results['r1'][i] = pp.r1
        results['r2'][i] = pp.r2
        results['r3'][i] = pp.r3
        results['s1'][i] = pp.s1
        results['s2'][i] = pp.s2
        results['s3'][i] = pp.s3

    # Set first row to NaN (no previous data)
    for key in results:
        results[key][0] = np.nan

    return pd.DataFrame(results, index=close.index)
