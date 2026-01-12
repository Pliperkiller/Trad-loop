"""Ichimoku Cloud indicator."""

from typing import Union
import pandas as pd
import numpy as np

from src.indicators.base import (
    validate_series,
    validate_period,
    validate_ohlcv,
    IchimokuCloudResult,
)


def ichimoku_cloud(
    high: Union[pd.Series, np.ndarray, list],
    low: Union[pd.Series, np.ndarray, list],
    close: Union[pd.Series, np.ndarray, list],
    tenkan_period: int = 9,
    kijun_period: int = 26,
    senkou_b_period: int = 52,
    displacement: int = 26
) -> IchimokuCloudResult:
    """
    Calculate Ichimoku Cloud (Ichimoku Kinko Hyo).

    The Ichimoku Cloud consists of five lines:
    - Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    - Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    - Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2, plotted 26 periods ahead
    - Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2, plotted 26 periods ahead
    - Chikou Span (Lagging Span): Close plotted 26 periods back

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        tenkan_period: Tenkan-sen period (default 9)
        kijun_period: Kijun-sen period (default 26)
        senkou_b_period: Senkou Span B period (default 52)
        displacement: Forward/backward shift for Senkou/Chikou (default 26)

    Returns:
        IchimokuCloudResult with all five components
    """
    high, low, close, _, _ = validate_ohlcv(high, low, close)
    tenkan_period = validate_period(tenkan_period, min_val=1, name="tenkan_period")
    kijun_period = validate_period(kijun_period, min_val=1, name="kijun_period")
    senkou_b_period = validate_period(senkou_b_period, min_val=1, name="senkou_b_period")
    displacement = validate_period(displacement, min_val=1, name="displacement")

    def _midpoint(high_data: pd.Series, low_data: pd.Series, period: int) -> pd.Series:
        """Calculate midpoint of high-low range over period."""
        return (high_data.rolling(window=period).max() +
                low_data.rolling(window=period).min()) / 2

    # Tenkan-sen (Conversion Line)
    tenkan_sen = _midpoint(high, low, tenkan_period)

    # Kijun-sen (Base Line)
    kijun_sen = _midpoint(high, low, kijun_period)

    # Senkou Span A (Leading Span A)
    # Average of Tenkan and Kijun, shifted forward
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(displacement)

    # Senkou Span B (Leading Span B)
    # Midpoint of 52-period high-low, shifted forward
    senkou_span_b = _midpoint(high, low, senkou_b_period).shift(displacement)

    # Chikou Span (Lagging Span)
    # Close price shifted backward
    chikou_span = close.shift(-displacement)

    return IchimokuCloudResult(
        tenkan_sen=tenkan_sen,
        kijun_sen=kijun_sen,
        senkou_span_a=senkou_span_a,
        senkou_span_b=senkou_span_b,
        chikou_span=chikou_span
    )


def ichimoku_signals(
    high: Union[pd.Series, np.ndarray, list],
    low: Union[pd.Series, np.ndarray, list],
    close: Union[pd.Series, np.ndarray, list],
    tenkan_period: int = 9,
    kijun_period: int = 26,
    senkou_b_period: int = 52,
    displacement: int = 26
) -> pd.DataFrame:
    """
    Generate Ichimoku trading signals.

    Signals generated:
    - tenkan_kijun_cross: 1 when Tenkan crosses above Kijun (bullish), -1 for bearish
    - price_cloud_position: 1 above cloud, -1 below cloud, 0 inside
    - chikou_position: 1 when Chikou is above price, -1 below

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        tenkan_period: Tenkan-sen period (default 9)
        kijun_period: Kijun-sen period (default 26)
        senkou_b_period: Senkou Span B period (default 52)
        displacement: Forward/backward shift (default 26)

    Returns:
        DataFrame with signal columns
    """
    ichi = ichimoku_cloud(
        high, low, close,
        tenkan_period, kijun_period, senkou_b_period, displacement
    )

    close = validate_series(close, "close")

    # Tenkan-Kijun cross signals
    tk_cross = np.where(
        (ichi.tenkan_sen > ichi.kijun_sen) &
        (ichi.tenkan_sen.shift(1) <= ichi.kijun_sen.shift(1)),
        1,
        np.where(
            (ichi.tenkan_sen < ichi.kijun_sen) &
            (ichi.tenkan_sen.shift(1) >= ichi.kijun_sen.shift(1)),
            -1,
            0
        )
    )

    # Cloud position (need to look at non-shifted Senkou spans for current position)
    cloud_top = pd.concat([ichi.senkou_span_a, ichi.senkou_span_b], axis=1).max(axis=1)
    cloud_bottom = pd.concat([ichi.senkou_span_a, ichi.senkou_span_b], axis=1).min(axis=1)

    price_cloud = np.where(
        close > cloud_top.shift(-displacement),
        1,
        np.where(close < cloud_bottom.shift(-displacement), -1, 0)
    )

    # Chikou span position (compare to close 26 periods ago)
    chikou_position = np.where(
        ichi.chikou_span.shift(displacement) > close,
        1,
        np.where(ichi.chikou_span.shift(displacement) < close, -1, 0)
    )

    signals = pd.DataFrame({
        'tenkan_kijun_cross': tk_cross,
        'price_cloud_position': price_cloud,
        'chikou_position': chikou_position
    }, index=close.index)

    return signals
