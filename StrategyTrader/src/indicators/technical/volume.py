"""Volume indicators: VWAP, OBV, CMF, MFI."""

from typing import Union, Optional
import pandas as pd
import numpy as np

from src.indicators.base import (
    validate_series,
    validate_period,
    validate_ohlcv,
)


def vwap(
    high: Union[pd.Series, np.ndarray, list],
    low: Union[pd.Series, np.ndarray, list],
    close: Union[pd.Series, np.ndarray, list],
    volume: Union[pd.Series, np.ndarray, list],
    anchor: Optional[str] = None
) -> pd.Series:
    """
    Calculate Volume Weighted Average Price.

    VWAP = Cumulative(Typical Price * Volume) / Cumulative(Volume)
    Typical Price = (High + Low + Close) / 3

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        volume: Volume data
        anchor: Optional anchor period ('D' for daily reset, etc.)
               If None, calculates cumulative VWAP from start

    Returns:
        VWAP values as pandas Series
    """
    high, low, close, volume, _ = validate_ohlcv(high, low, close, volume)

    # Calculate typical price
    typical_price = (high + low + close) / 3

    # Calculate VWAP
    tp_volume = typical_price * volume

    if anchor is not None and hasattr(close, 'index') and hasattr(close.index, 'date'):
        # Group by anchor period and calculate cumulative within each group
        if anchor == 'D':
            groups = close.index.date
        else:
            groups = close.index.to_period(anchor)

        cumulative_tp_vol = tp_volume.groupby(groups).cumsum()
        cumulative_vol = volume.groupby(groups).cumsum()
    else:
        # Calculate cumulative from start
        cumulative_tp_vol = tp_volume.cumsum()
        cumulative_vol = volume.cumsum()

    vwap_values = cumulative_tp_vol / cumulative_vol

    return vwap_values


def obv(
    close: Union[pd.Series, np.ndarray, list],
    volume: Union[pd.Series, np.ndarray, list]
) -> pd.Series:
    """
    Calculate On-Balance Volume.

    OBV adds volume on up days and subtracts volume on down days.
    Used to detect buying/selling pressure.

    Args:
        close: Close prices
        volume: Volume data

    Returns:
        OBV values as pandas Series
    """
    close = validate_series(close, "close")
    volume = validate_series(volume, "volume")

    if len(close) != len(volume):
        raise ValueError("close and volume must have the same length")

    # Calculate price direction
    # First value has no previous price, so set direction to 0
    direction = np.zeros(len(close))
    for i in range(1, len(close)):
        if close.iloc[i] > close.iloc[i-1]:
            direction[i] = 1
        elif close.iloc[i] < close.iloc[i-1]:
            direction[i] = -1
        else:
            direction[i] = 0

    # Calculate OBV as cumulative sum of signed volumes
    obv_values = (volume.values * direction).cumsum()

    return pd.Series(obv_values, index=close.index)


def cmf(
    high: Union[pd.Series, np.ndarray, list],
    low: Union[pd.Series, np.ndarray, list],
    close: Union[pd.Series, np.ndarray, list],
    volume: Union[pd.Series, np.ndarray, list],
    period: int = 20
) -> pd.Series:
    """
    Calculate Chaikin Money Flow.

    CMF measures buying and selling pressure over a period.
    Positive values indicate buying pressure, negative indicate selling pressure.

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        volume: Volume data
        period: Lookback period (default 20)

    Returns:
        CMF values (-1 to 1) as pandas Series
    """
    high, low, close, volume, _ = validate_ohlcv(high, low, close, volume)
    period = validate_period(period, min_val=1, name="period")

    # Calculate Money Flow Multiplier
    # MFM = [(Close - Low) - (High - Close)] / (High - Low)
    hl_diff = high - low
    mfm = ((close - low) - (high - close)) / hl_diff

    # Handle division by zero (when high == low)
    mfm = mfm.fillna(0)

    # Calculate Money Flow Volume
    mfv = mfm * volume

    # Calculate CMF (sum of MFV / sum of Volume over period)
    cmf_values = mfv.rolling(window=period).sum() / volume.rolling(window=period).sum()

    return cmf_values


def mfi(
    high: Union[pd.Series, np.ndarray, list],
    low: Union[pd.Series, np.ndarray, list],
    close: Union[pd.Series, np.ndarray, list],
    volume: Union[pd.Series, np.ndarray, list],
    period: int = 14
) -> pd.Series:
    """
    Calculate Money Flow Index.

    MFI is a volume-weighted RSI. Values above 80 indicate overbought,
    below 20 indicate oversold.

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        volume: Volume data
        period: MFI period (default 14)

    Returns:
        MFI values (0-100) as pandas Series
    """
    high, low, close, volume, _ = validate_ohlcv(high, low, close, volume)
    period = validate_period(period, min_val=1, name="period")

    # Calculate typical price
    typical_price = (high + low + close) / 3

    # Calculate raw money flow
    raw_money_flow = typical_price * volume

    # Determine positive and negative money flow
    tp_change = typical_price.diff()

    positive_flow = np.where(tp_change > 0, raw_money_flow, 0)
    negative_flow = np.where(tp_change < 0, raw_money_flow, 0)

    positive_flow = pd.Series(positive_flow, index=close.index)
    negative_flow = pd.Series(negative_flow, index=close.index)

    # Calculate sum of flows over period
    positive_sum = positive_flow.rolling(window=period).sum()
    negative_sum = negative_flow.rolling(window=period).sum()

    # Calculate Money Flow Ratio
    mfr = positive_sum / negative_sum

    # Calculate MFI
    mfi_values = 100 - (100 / (1 + mfr))

    # Handle division by zero
    mfi_values = mfi_values.fillna(50)  # Neutral when no negative flow

    return mfi_values
