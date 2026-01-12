"""Validation utilities for indicators."""

from typing import Union, Tuple, Optional
import pandas as pd
import numpy as np


def validate_series(
    data: Union[pd.Series, np.ndarray, list],
    name: str = "data"
) -> pd.Series:
    """
    Convert input to pandas Series and validate.

    Args:
        data: Input data
        name: Parameter name for error messages

    Returns:
        Validated pandas Series

    Raises:
        TypeError: If data type is not supported
        ValueError: If data is empty
    """
    if isinstance(data, list):
        data = pd.Series(data)
    elif isinstance(data, np.ndarray):
        data = pd.Series(data)
    elif not isinstance(data, pd.Series):
        raise TypeError(f"{name} must be a pandas Series, numpy array, or list")

    if len(data) == 0:
        raise ValueError(f"{name} cannot be empty")

    return data


def validate_period(
    period: int,
    min_val: int = 1,
    max_val: Optional[int] = None,
    name: str = "period"
) -> int:
    """
    Validate period parameter.

    Args:
        period: Period value to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value (optional)
        name: Parameter name for error messages

    Returns:
        Validated period

    Raises:
        TypeError: If period is not an integer
        ValueError: If period is out of range
    """
    if not isinstance(period, int):
        raise TypeError(f"{name} must be an integer")
    if period < min_val:
        raise ValueError(f"{name} must be >= {min_val}")
    if max_val is not None and period > max_val:
        raise ValueError(f"{name} must be <= {max_val}")
    return period


def validate_positive(
    value: Union[int, float],
    name: str = "value",
    allow_zero: bool = False
) -> Union[int, float]:
    """
    Validate that value is positive.

    Args:
        value: Value to validate
        name: Parameter name for error messages
        allow_zero: Whether zero is allowed

    Returns:
        Validated value

    Raises:
        TypeError: If value is not numeric
        ValueError: If value is not positive
    """
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    if allow_zero:
        if value < 0:
            raise ValueError(f"{name} must be >= 0")
    else:
        if value <= 0:
            raise ValueError(f"{name} must be > 0")
    return value


def validate_range(
    value: Union[int, float],
    min_val: Union[int, float],
    max_val: Union[int, float],
    name: str = "value"
) -> Union[int, float]:
    """
    Validate that value is within range.

    Args:
        value: Value to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        name: Parameter name for error messages

    Returns:
        Validated value

    Raises:
        TypeError: If value is not numeric
        ValueError: If value is out of range
    """
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    if value < min_val or value > max_val:
        raise ValueError(f"{name} must be between {min_val} and {max_val}")
    return value


def validate_ohlcv(
    high: Union[pd.Series, np.ndarray],
    low: Union[pd.Series, np.ndarray],
    close: Union[pd.Series, np.ndarray],
    volume: Optional[Union[pd.Series, np.ndarray]] = None,
    open_: Optional[Union[pd.Series, np.ndarray]] = None
) -> Tuple[pd.Series, pd.Series, pd.Series, Optional[pd.Series], Optional[pd.Series]]:
    """
    Validate and convert OHLCV data.

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        volume: Volume (optional)
        open_: Open prices (optional)

    Returns:
        Tuple of validated pandas Series

    Raises:
        TypeError: If data types are not supported
        ValueError: If lengths don't match or data is empty
    """
    high = validate_series(high, "high")
    low = validate_series(low, "low")
    close = validate_series(close, "close")

    if len(high) != len(low) or len(high) != len(close):
        raise ValueError("high, low, and close must have the same length")

    if volume is not None:
        volume = validate_series(volume, "volume")
        if len(volume) != len(high):
            raise ValueError("volume must have the same length as price data")

    if open_ is not None:
        open_ = validate_series(open_, "open")
        if len(open_) != len(high):
            raise ValueError("open must have the same length as price data")

    return high, low, close, volume, open_
