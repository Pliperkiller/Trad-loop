"""Base classes and protocols for indicators module."""

from dataclasses import dataclass, field
from typing import Protocol, Union, Tuple, Optional
import pandas as pd
import numpy as np


class IndicatorProtocol(Protocol):
    """Protocol for indicator functions."""

    def __call__(
        self,
        data: Union[pd.Series, np.ndarray],
        *args,
        **kwargs
    ) -> Union[pd.Series, np.ndarray, "IndicatorResult"]:
        """Calculate indicator value."""
        ...


@dataclass
class IndicatorResult:
    """Base class for multi-value indicator results."""
    pass


@dataclass
class BollingerBandsResult(IndicatorResult):
    """Result from Bollinger Bands calculation."""
    upper: pd.Series
    middle: pd.Series
    lower: pd.Series
    bandwidth: Optional[pd.Series] = field(default=None)
    percent_b: Optional[pd.Series] = field(default=None)

    def __iter__(self):
        """Support tuple unpacking: upper, middle, lower = bollinger_bands(...)"""
        return iter((self.upper, self.middle, self.lower))


@dataclass
class MACDResult(IndicatorResult):
    """Result from MACD calculation."""
    macd_line: pd.Series
    signal_line: pd.Series
    histogram: pd.Series

    def __iter__(self):
        """Support tuple unpacking: macd, signal, hist = macd(...)"""
        return iter((self.macd_line, self.signal_line, self.histogram))


@dataclass
class StochasticResult(IndicatorResult):
    """Result from Stochastic calculation."""
    k: pd.Series
    d: pd.Series

    def __iter__(self):
        """Support tuple unpacking: k, d = stochastic(...)"""
        return iter((self.k, self.d))


@dataclass
class ADXResult(IndicatorResult):
    """Result from ADX calculation."""
    adx: pd.Series
    plus_di: pd.Series
    minus_di: pd.Series


@dataclass
class KeltnerChannelsResult(IndicatorResult):
    """Result from Keltner Channels calculation."""
    middle: pd.Series
    upper: pd.Series
    lower: pd.Series


@dataclass
class DonchianChannelsResult(IndicatorResult):
    """Result from Donchian Channels calculation."""
    upper: pd.Series
    middle: pd.Series
    lower: pd.Series


@dataclass
class IchimokuCloudResult(IndicatorResult):
    """Result from Ichimoku Cloud calculation."""
    tenkan_sen: pd.Series      # Conversion Line (9-period)
    kijun_sen: pd.Series       # Base Line (26-period)
    senkou_span_a: pd.Series   # Leading Span A
    senkou_span_b: pd.Series   # Leading Span B (52-period)
    chikou_span: pd.Series     # Lagging Span


@dataclass
class PivotPointsResult(IndicatorResult):
    """Result from Pivot Points calculation."""
    pivot: float
    r1: float
    r2: float
    r3: float
    s1: float
    s2: float
    s3: float
    method: str


@dataclass
class SupertrendResult(IndicatorResult):
    """Result from Supertrend calculation."""
    supertrend: pd.Series
    direction: pd.Series  # 1 for uptrend, -1 for downtrend


@dataclass
class ParabolicSARResult(IndicatorResult):
    """Result from Parabolic SAR calculation."""
    sar: pd.Series
    trend: pd.Series  # 1 for uptrend, -1 for downtrend


@dataclass
class LinearRegressionResult(IndicatorResult):
    """Result from Linear Regression calculation."""
    slope: pd.Series
    slope_normalized_1: pd.Series  # slope / avg price of window
    slope_normalized_2: pd.Series  # slope / close price (last candle)
    slope_normalized_3: pd.Series  # slope / open price of window
    intercept: pd.Series
    r_squared: pd.Series
    residual: pd.Series           # actual price - predicted price
    residual_std: pd.Series       # std deviation of residuals in window
    residual_zscore: pd.Series    # residual / residual_std


def validate_series(
    data: Union[pd.Series, np.ndarray, list],
    name: str = "data"
) -> pd.Series:
    """Convert input to pandas Series and validate."""
    if isinstance(data, list):
        data = pd.Series(data)
    elif isinstance(data, np.ndarray):
        data = pd.Series(data)
    elif not isinstance(data, pd.Series):
        raise TypeError(f"{name} must be a pandas Series, numpy array, or list")

    if len(data) == 0:
        raise ValueError(f"{name} cannot be empty")

    return data


def validate_period(period: int, min_val: int = 1, name: str = "period") -> int:
    """Validate period parameter."""
    if not isinstance(period, int):
        raise TypeError(f"{name} must be an integer")
    if period < min_val:
        raise ValueError(f"{name} must be >= {min_val}")
    return period


def validate_ohlcv(
    high: Union[pd.Series, np.ndarray],
    low: Union[pd.Series, np.ndarray],
    close: Union[pd.Series, np.ndarray],
    volume: Optional[Union[pd.Series, np.ndarray]] = None,
    open_: Optional[Union[pd.Series, np.ndarray]] = None
) -> Tuple[pd.Series, pd.Series, pd.Series, Optional[pd.Series], Optional[pd.Series]]:
    """Validate and convert OHLCV data."""
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
