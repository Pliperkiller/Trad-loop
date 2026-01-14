"""Volatility indicators: ATR, Bollinger Bands, Keltner Channels, Donchian Channels."""

from typing import Union
import pandas as pd
import numpy as np

from src.indicators.base import (
    validate_series,
    validate_period,
    validate_ohlcv,
    BollingerBandsResult,
    KeltnerChannelsResult,
    DonchianChannelsResult,
)


def atr(
    high: Union[pd.Series, np.ndarray, list],
    low: Union[pd.Series, np.ndarray, list],
    close: Union[pd.Series, np.ndarray, list],
    period: int = 14
) -> pd.Series:
    """
    Calculate Average True Range.

    ATR measures market volatility by decomposing the entire range of an asset.

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: ATR period (default 14)

    Returns:
        ATR values as pandas Series
    """
    high, low, close, _, _ = validate_ohlcv(high, low, close)
    period = validate_period(period, min_val=1, name="period")

    # Calculate True Range components
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))

    # True Range is the maximum of the three components
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Calculate ATR using Wilder's smoothing
    atr_values = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

    return atr_values


def bollinger_bands(
    data: Union[pd.Series, np.ndarray, list],
    period: int = 20,
    std_dev: float = 2.0
) -> BollingerBandsResult:
    """
    Calculate Bollinger Bands.

    Bollinger Bands consist of a middle band (SMA) and upper/lower bands
    at standard deviations from the middle band.

    Args:
        data: Price data (typically close prices)
        period: SMA period (default 20)
        std_dev: Number of standard deviations (default 2.0)

    Returns:
        BollingerBandsResult with middle, upper, lower, bandwidth, and percent_b
    """
    data = validate_series(data, "data")
    period = validate_period(period, min_val=1, name="period")

    if std_dev <= 0:
        raise ValueError("std_dev must be positive")

    # Calculate middle band (SMA)
    middle = data.rolling(window=period).mean()

    # Calculate standard deviation
    std = data.rolling(window=period).std()

    # Calculate upper and lower bands
    upper = middle + (std_dev * std)
    lower = middle - (std_dev * std)

    # Calculate bandwidth (volatility measure)
    bandwidth = (upper - lower) / middle * 100

    # Calculate %B (position within bands)
    percent_b = (data - lower) / (upper - lower)

    return BollingerBandsResult(
        upper=upper,
        middle=middle,
        lower=lower,
        bandwidth=bandwidth,
        percent_b=percent_b
    )


def keltner_channels(
    high: Union[pd.Series, np.ndarray, list],
    low: Union[pd.Series, np.ndarray, list],
    close: Union[pd.Series, np.ndarray, list],
    period: int = 20,
    atr_period: int = 10,
    multiplier: float = 2.0
) -> KeltnerChannelsResult:
    """
    Calculate Keltner Channels.

    Keltner Channels are volatility-based envelopes using ATR instead of
    standard deviation (unlike Bollinger Bands).

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: EMA period for middle band (default 20)
        atr_period: ATR period (default 10)
        multiplier: ATR multiplier (default 2.0)

    Returns:
        KeltnerChannelsResult with middle, upper, and lower bands
    """
    high, low, close, _, _ = validate_ohlcv(high, low, close)
    period = validate_period(period, min_val=1, name="period")
    atr_period = validate_period(atr_period, min_val=1, name="atr_period")

    if multiplier <= 0:
        raise ValueError("multiplier must be positive")

    # Calculate middle band (EMA)
    middle = close.ewm(span=period, adjust=False).mean()

    # Calculate ATR
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_values = tr.ewm(alpha=1/atr_period, min_periods=atr_period, adjust=False).mean()

    # Calculate upper and lower bands
    upper = middle + (multiplier * atr_values)
    lower = middle - (multiplier * atr_values)

    return KeltnerChannelsResult(
        middle=middle,
        upper=upper,
        lower=lower
    )


def donchian_channels(
    high: Union[pd.Series, np.ndarray, list],
    low: Union[pd.Series, np.ndarray, list],
    period: int = 20
) -> DonchianChannelsResult:
    """
    Calculate Donchian Channels.

    Donchian Channels show the highest high and lowest low over a period.
    Used for breakout trading strategies.

    Args:
        high: High prices
        low: Low prices
        period: Lookback period (default 20)

    Returns:
        DonchianChannelsResult with upper, middle, and lower bands
    """
    high = validate_series(high, "high")
    low = validate_series(low, "low")
    period = validate_period(period, min_val=1, name="period")

    if len(high) != len(low):
        raise ValueError("high and low must have the same length")

    # Calculate upper and lower channels
    upper = high.rolling(window=period).max()
    lower = low.rolling(window=period).min()

    # Calculate middle channel (average of upper and lower)
    middle = (upper + lower) / 2

    return DonchianChannelsResult(
        upper=upper,
        middle=middle,
        lower=lower
    )
