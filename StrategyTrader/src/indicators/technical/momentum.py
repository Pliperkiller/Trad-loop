"""Momentum indicators: RSI, MACD, Stochastic, Williams %R, ADX."""

from typing import Union
import pandas as pd
import numpy as np

from src.indicators.base import (
    validate_series,
    validate_period,
    validate_ohlcv,
    MACDResult,
    StochasticResult,
    ADXResult,
)


def rsi(
    data: Union[pd.Series, np.ndarray, list],
    period: int = 14
) -> pd.Series:
    """
    Calculate Relative Strength Index.

    RSI measures the speed and magnitude of price movements.

    Args:
        data: Price data (typically close prices)
        period: RSI period (default 14)

    Returns:
        RSI values (0-100) as pandas Series
    """
    data = validate_series(data, "data")
    period = validate_period(period, min_val=1, name="period")

    # Calculate price changes
    delta = data.diff()

    # Separate gains and losses
    gains = delta.where(delta > 0, 0.0)
    losses = (-delta).where(delta < 0, 0.0)

    # Calculate average gains and losses using Wilder's smoothing
    avg_gain = gains.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

    # Calculate RS and RSI
    rs = avg_gain / avg_loss
    rsi_values = 100 - (100 / (1 + rs))

    # Handle division by zero (when avg_loss is 0)
    rsi_values = rsi_values.fillna(100)

    return rsi_values


def macd(
    data: Union[pd.Series, np.ndarray, list],
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> MACDResult:
    """
    Calculate Moving Average Convergence Divergence.

    Args:
        data: Price data (typically close prices)
        fast_period: Fast EMA period (default 12)
        slow_period: Slow EMA period (default 26)
        signal_period: Signal line EMA period (default 9)

    Returns:
        MACDResult with macd_line, signal_line, and histogram
    """
    data = validate_series(data, "data")
    fast_period = validate_period(fast_period, min_val=1, name="fast_period")
    slow_period = validate_period(slow_period, min_val=1, name="slow_period")
    signal_period = validate_period(signal_period, min_val=1, name="signal_period")

    if fast_period >= slow_period:
        raise ValueError("fast_period must be less than slow_period")

    # Calculate EMAs
    fast_ema = data.ewm(span=fast_period, adjust=False).mean()
    slow_ema = data.ewm(span=slow_period, adjust=False).mean()

    # Calculate MACD line
    macd_line = fast_ema - slow_ema

    # Calculate signal line
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()

    # Calculate histogram
    histogram = macd_line - signal_line

    return MACDResult(
        macd_line=macd_line,
        signal_line=signal_line,
        histogram=histogram
    )


def stochastic(
    high: Union[pd.Series, np.ndarray, list],
    low: Union[pd.Series, np.ndarray, list],
    close: Union[pd.Series, np.ndarray, list],
    k_period: int = 14,
    d_period: int = 3,
    smooth_k: int = 3
) -> StochasticResult:
    """
    Calculate Stochastic Oscillator (%K and %D).

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        k_period: %K period (default 14)
        d_period: %D smoothing period (default 3)
        smooth_k: %K smoothing period (default 3)

    Returns:
        StochasticResult with k and d values (0-100)
    """
    high, low, close, _, _ = validate_ohlcv(high, low, close)
    k_period = validate_period(k_period, min_val=1, name="k_period")
    d_period = validate_period(d_period, min_val=1, name="d_period")
    smooth_k = validate_period(smooth_k, min_val=1, name="smooth_k")

    # Calculate lowest low and highest high over k_period
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()

    # Calculate raw %K
    raw_k = 100 * (close - lowest_low) / (highest_high - lowest_low)

    # Smooth %K
    k = raw_k.rolling(window=smooth_k).mean()

    # Calculate %D (SMA of %K)
    d = k.rolling(window=d_period).mean()

    return StochasticResult(k=k, d=d)


def williams_r(
    high: Union[pd.Series, np.ndarray, list],
    low: Union[pd.Series, np.ndarray, list],
    close: Union[pd.Series, np.ndarray, list],
    period: int = 14
) -> pd.Series:
    """
    Calculate Williams %R.

    Williams %R is a momentum indicator that shows overbought/oversold levels.
    Range: -100 to 0 (typically -80 to -100 is oversold, -20 to 0 is overbought)

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: Lookback period (default 14)

    Returns:
        Williams %R values (-100 to 0) as pandas Series
    """
    high, low, close, _, _ = validate_ohlcv(high, low, close)
    period = validate_period(period, min_val=1, name="period")

    # Calculate highest high and lowest low over period
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()

    # Calculate Williams %R
    williams = -100 * (highest_high - close) / (highest_high - lowest_low)

    return williams


def adx(
    high: Union[pd.Series, np.ndarray, list],
    low: Union[pd.Series, np.ndarray, list],
    close: Union[pd.Series, np.ndarray, list],
    period: int = 14
) -> ADXResult:
    """
    Calculate Average Directional Index (ADX).

    ADX measures trend strength regardless of direction.
    Values > 25 indicate trending market, < 20 indicate ranging market.

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: ADX period (default 14)

    Returns:
        ADXResult with adx, plus_di, and minus_di values
    """
    high, low, close, _, _ = validate_ohlcv(high, low, close)
    period = validate_period(period, min_val=1, name="period")

    # Calculate True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Calculate directional movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    # +DM and -DM
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    plus_dm = pd.Series(plus_dm, index=high.index)
    minus_dm = pd.Series(minus_dm, index=high.index)

    # Smooth TR, +DM, -DM using Wilder's smoothing
    atr = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    smooth_plus_dm = plus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    smooth_minus_dm = minus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

    # Calculate +DI and -DI
    plus_di = 100 * smooth_plus_dm / atr
    minus_di = 100 * smooth_minus_dm / atr

    # Calculate DX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)

    # Calculate ADX (smoothed DX)
    adx_values = dx.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

    return ADXResult(
        adx=adx_values,
        plus_di=plus_di,
        minus_di=minus_di
    )
