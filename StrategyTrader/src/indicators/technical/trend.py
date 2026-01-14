"""Trend indicators: SMA, EMA, VWMA, Parabolic SAR, Supertrend."""

from typing import Union
import pandas as pd
import numpy as np

from src.indicators.base import (
    validate_series,
    validate_period,
    validate_ohlcv,
    ParabolicSARResult,
    SupertrendResult,
    LinearRegressionResult,
)


def sma(
    data: Union[pd.Series, np.ndarray, list],
    period: int
) -> pd.Series:
    """
    Calculate Simple Moving Average.

    Args:
        data: Price data (typically close prices)
        period: Number of periods for the moving average

    Returns:
        SMA values as pandas Series
    """
    data = validate_series(data, "data")
    period = validate_period(period, min_val=1, name="period")

    return data.rolling(window=period).mean()


def ema(
    data: Union[pd.Series, np.ndarray, list],
    period: int,
    adjust: bool = True
) -> pd.Series:
    """
    Calculate Exponential Moving Average.

    Args:
        data: Price data (typically close prices)
        period: Number of periods for the moving average
        adjust: Whether to adjust the EMA calculation (default True)

    Returns:
        EMA values as pandas Series
    """
    data = validate_series(data, "data")
    period = validate_period(period, min_val=1, name="period")

    return data.ewm(span=period, adjust=adjust).mean()


def vwma(
    close: Union[pd.Series, np.ndarray, list],
    volume: Union[pd.Series, np.ndarray, list],
    period: int
) -> pd.Series:
    """
    Calculate Volume Weighted Moving Average.

    VWMA = Sum(Price * Volume) / Sum(Volume) over period

    Args:
        close: Close prices
        volume: Volume data
        period: Number of periods for the moving average

    Returns:
        VWMA values as pandas Series
    """
    close = validate_series(close, "close")
    volume = validate_series(volume, "volume")
    period = validate_period(period, min_val=1, name="period")

    if len(close) != len(volume):
        raise ValueError("close and volume must have the same length")

    price_volume = close * volume
    sum_pv = price_volume.rolling(window=period).sum()
    sum_v = volume.rolling(window=period).sum()

    # Protección contra división por cero cuando el volumen total es cero
    # Convertir a Series si es necesario y reemplazar ceros con NaN
    if isinstance(sum_v, pd.Series):
        sum_v_safe = sum_v.replace(0, np.nan)
    else:
        sum_v_safe = pd.Series(sum_v).replace(0, np.nan)

    return sum_pv / sum_v_safe


def parabolic_sar(
    high: Union[pd.Series, np.ndarray, list],
    low: Union[pd.Series, np.ndarray, list],
    close: Union[pd.Series, np.ndarray, list],
    af_start: float = 0.02,
    af_increment: float = 0.02,
    af_max: float = 0.20
) -> ParabolicSARResult:
    """
    Calculate Parabolic Stop and Reverse (SAR).

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        af_start: Starting acceleration factor (default 0.02)
        af_increment: AF increment on new extreme (default 0.02)
        af_max: Maximum acceleration factor (default 0.20)

    Returns:
        ParabolicSARResult with sar values and trend direction
    """
    high, low, close, _, _ = validate_ohlcv(high, low, close)

    if af_start <= 0 or af_increment <= 0 or af_max <= 0:
        raise ValueError("Acceleration factors must be positive")
    if af_start > af_max:
        raise ValueError("af_start must be <= af_max")

    n = len(close)
    sar = np.zeros(n)
    trend = np.zeros(n)  # 1 for uptrend, -1 for downtrend
    ep = np.zeros(n)  # Extreme point
    af = np.zeros(n)  # Acceleration factor

    # Initialize
    trend[0] = 1 if close.iloc[1] > close.iloc[0] else -1

    if trend[0] == 1:
        sar[0] = low.iloc[0]
        ep[0] = high.iloc[0]
    else:
        sar[0] = high.iloc[0]
        ep[0] = low.iloc[0]
    af[0] = af_start

    for i in range(1, n):
        # Calculate new SAR
        sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])

        # Adjust SAR based on trend
        if trend[i-1] == 1:  # Uptrend
            sar[i] = min(sar[i], low.iloc[i-1])
            if i >= 2:
                sar[i] = min(sar[i], low.iloc[i-2])

            # Check for trend reversal
            if low.iloc[i] < sar[i]:
                trend[i] = -1
                sar[i] = ep[i-1]
                ep[i] = low.iloc[i]
                af[i] = af_start
            else:
                trend[i] = 1
                # Update EP and AF
                if high.iloc[i] > ep[i-1]:
                    ep[i] = high.iloc[i]
                    af[i] = min(af[i-1] + af_increment, af_max)
                else:
                    ep[i] = ep[i-1]
                    af[i] = af[i-1]
        else:  # Downtrend
            sar[i] = max(sar[i], high.iloc[i-1])
            if i >= 2:
                sar[i] = max(sar[i], high.iloc[i-2])

            # Check for trend reversal
            if high.iloc[i] > sar[i]:
                trend[i] = 1
                sar[i] = ep[i-1]
                ep[i] = high.iloc[i]
                af[i] = af_start
            else:
                trend[i] = -1
                # Update EP and AF
                if low.iloc[i] < ep[i-1]:
                    ep[i] = low.iloc[i]
                    af[i] = min(af[i-1] + af_increment, af_max)
                else:
                    ep[i] = ep[i-1]
                    af[i] = af[i-1]

    sar_series = pd.Series(sar, index=close.index)
    trend_series = pd.Series(trend, index=close.index)

    return ParabolicSARResult(sar=sar_series, trend=trend_series)


def supertrend(
    high: Union[pd.Series, np.ndarray, list],
    low: Union[pd.Series, np.ndarray, list],
    close: Union[pd.Series, np.ndarray, list],
    period: int = 10,
    multiplier: float = 3.0
) -> SupertrendResult:
    """
    Calculate Supertrend indicator.

    Supertrend uses ATR and a multiplier to determine trend direction.

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: ATR period (default 10)
        multiplier: ATR multiplier (default 3.0)

    Returns:
        SupertrendResult with supertrend values and direction
    """
    high, low, close, _, _ = validate_ohlcv(high, low, close)
    period = validate_period(period, min_val=1, name="period")

    if multiplier <= 0:
        raise ValueError("multiplier must be positive")

    n = len(close)

    # Calculate ATR
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_values = tr.rolling(window=period).mean()

    # Calculate basic bands
    hl2 = (high + low) / 2
    upper_band = hl2 + (multiplier * atr_values)
    lower_band = hl2 - (multiplier * atr_values)

    # Initialize arrays
    supertrend_arr = np.zeros(n)
    direction = np.ones(n)  # 1 for uptrend, -1 for downtrend
    final_upper = np.zeros(n)
    final_lower = np.zeros(n)

    # First valid values
    first_valid = period - 1
    final_upper[:first_valid+1] = upper_band.iloc[:first_valid+1].values
    final_lower[:first_valid+1] = lower_band.iloc[:first_valid+1].values

    for i in range(first_valid + 1, n):
        # Update final upper band
        if upper_band.iloc[i] < final_upper[i-1] or close.iloc[i-1] > final_upper[i-1]:
            final_upper[i] = upper_band.iloc[i]
        else:
            final_upper[i] = final_upper[i-1]

        # Update final lower band
        if lower_band.iloc[i] > final_lower[i-1] or close.iloc[i-1] < final_lower[i-1]:
            final_lower[i] = lower_band.iloc[i]
        else:
            final_lower[i] = final_lower[i-1]

    # Calculate supertrend and direction
    for i in range(first_valid + 1, n):
        if direction[i-1] == 1:  # Previous was uptrend
            if close.iloc[i] < final_lower[i]:
                direction[i] = -1
                supertrend_arr[i] = final_upper[i]
            else:
                direction[i] = 1
                supertrend_arr[i] = final_lower[i]
        else:  # Previous was downtrend
            if close.iloc[i] > final_upper[i]:
                direction[i] = 1
                supertrend_arr[i] = final_lower[i]
            else:
                direction[i] = -1
                supertrend_arr[i] = final_upper[i]

    # Set initial values
    supertrend_arr[:first_valid+1] = np.nan
    direction[:first_valid+1] = np.nan

    supertrend_series = pd.Series(supertrend_arr, index=close.index)
    direction_series = pd.Series(direction, index=close.index)

    return SupertrendResult(supertrend=supertrend_series, direction=direction_series)


def linear_regression(
    close: Union[pd.Series, np.ndarray, list],
    open_: Union[pd.Series, np.ndarray, list],
    period: int = 20
) -> LinearRegressionResult:
    """
    Calculate Linear Regression over a rolling window.

    Fits a linear regression line to the last N candles and returns
    the slope, intercept, R², normalized slope values, and residual metrics.

    Args:
        close: Close prices (used for regression calculation)
        open_: Open prices (used for slope_normalized_3)
        period: Number of periods for regression window (default 20)

    Returns:
        LinearRegressionResult with:
        - slope: Regression line slope
        - slope_normalized_1: slope / average price of window
        - slope_normalized_2: slope / close price of last candle
        - slope_normalized_3: slope / open price of first candle in window
        - intercept: Regression line intercept
        - r_squared: Coefficient of determination (R²)
        - residual: actual price - predicted price at current point
        - residual_std: standard deviation of residuals in window
        - residual_zscore: residual / residual_std (clamped to avoid inf)
    """
    close = validate_series(close, "close")
    open_ = validate_series(open_, "open_")
    period = validate_period(period, min_val=2, name="period")

    if len(close) != len(open_):
        raise ValueError("close and open_ must have the same length")

    n = len(close)

    # Initialize result arrays
    slope_arr = np.full(n, np.nan)
    slope_norm1_arr = np.full(n, np.nan)
    slope_norm2_arr = np.full(n, np.nan)
    slope_norm3_arr = np.full(n, np.nan)
    intercept_arr = np.full(n, np.nan)
    r_squared_arr = np.full(n, np.nan)
    residual_arr = np.full(n, np.nan)
    residual_std_arr = np.full(n, np.nan)
    residual_zscore_arr = np.full(n, np.nan)

    # X values for regression (0, 1, 2, ..., period-1)
    x = np.arange(period)

    for i in range(period - 1, n):
        # Get window of close prices
        y = close.iloc[i - period + 1:i + 1].values

        # Calculate linear regression using polyfit
        coeffs = np.polyfit(x, y, 1)
        slope = coeffs[0]
        intercept = coeffs[1]

        # Calculate predicted values for all points in window
        y_pred = slope * x + intercept

        # Calculate R² (coefficient of determination)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 1.0

        # Calculate residual metrics
        # Predicted price at current point (last point in window)
        predicted_current = intercept + slope * (period - 1)
        # Residual: actual - predicted
        residual = close.iloc[i] - predicted_current
        # Standard deviation of all residuals in window
        all_residuals = y - y_pred
        residual_std = np.std(all_residuals, ddof=0)
        # Z-score with protection against division by zero
        if residual_std > 1e-10:
            residual_zscore = residual / residual_std
        else:
            # If std is ~0, prices are perfectly linear, zscore is 0
            residual_zscore = 0.0

        # Calculate normalized slopes
        avg_price = np.mean(y)
        close_price = close.iloc[i]
        open_price_window = open_.iloc[i - period + 1]

        slope_norm1 = slope / avg_price if avg_price != 0 else 0.0
        slope_norm2 = slope / close_price if close_price != 0 else 0.0
        slope_norm3 = slope / open_price_window if open_price_window != 0 else 0.0

        # Store results
        slope_arr[i] = slope
        slope_norm1_arr[i] = slope_norm1
        slope_norm2_arr[i] = slope_norm2
        slope_norm3_arr[i] = slope_norm3
        intercept_arr[i] = intercept
        r_squared_arr[i] = r_squared
        residual_arr[i] = residual
        residual_std_arr[i] = residual_std
        residual_zscore_arr[i] = residual_zscore

    # Convert to pandas Series with original index
    return LinearRegressionResult(
        slope=pd.Series(slope_arr, index=close.index),
        slope_normalized_1=pd.Series(slope_norm1_arr, index=close.index),
        slope_normalized_2=pd.Series(slope_norm2_arr, index=close.index),
        slope_normalized_3=pd.Series(slope_norm3_arr, index=close.index),
        intercept=pd.Series(intercept_arr, index=close.index),
        r_squared=pd.Series(r_squared_arr, index=close.index),
        residual=pd.Series(residual_arr, index=close.index),
        residual_std=pd.Series(residual_std_arr, index=close.index),
        residual_zscore=pd.Series(residual_zscore_arr, index=close.index),
    )
