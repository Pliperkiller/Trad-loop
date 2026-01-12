"""
Indicators module - Technical and Fundamental analysis tools.

This module provides:
- Technical indicators (trend, momentum, volatility, volume)
- Fundamental indicators (market data, on-chain metrics, sentiment)

Technical Indicators
--------------------
Trend: SMA, EMA, VWMA, Parabolic SAR, Supertrend
Momentum: RSI, MACD, Stochastic, Williams %R, ADX
Volatility: ATR, Bollinger Bands, Keltner Channels, Donchian Channels
Volume: VWAP, OBV, CMF, MFI
Complex: Ichimoku Cloud, Pivot Points

Fundamental Indicators
----------------------
Free APIs:
- CoinGecko: Market data, prices, market cap
- DefiLlama: DeFi TVL data
- Alternative.me: Fear & Greed Index

Paid APIs (structure provided):
- Glassnode: On-chain metrics (NVT, MVRV, SOPR)
- Santiment: Social metrics, sentiment

Example usage:
    >>> from src.indicators import sma, rsi, macd, ichimoku_cloud
    >>> sma_20 = sma(close_prices, period=20)
    >>> rsi_14 = rsi(close_prices, period=14)

    >>> from src.indicators.fundamental import CoinGeckoClient, FearGreedClient
    >>> cg = CoinGeckoClient()
    >>> btc_data = cg.get_market_data('bitcoin')

Backward Compatibility:
    >>> from src.indicators import TechnicalIndicators
    >>> TechnicalIndicators.sma(prices, 20)  # Still works
"""

# Base classes and result types
from src.indicators.base import (
    IndicatorProtocol,
    IndicatorResult,
    BollingerBandsResult,
    MACDResult,
    StochasticResult,
    ADXResult,
    KeltnerChannelsResult,
    DonchianChannelsResult,
    IchimokuCloudResult,
    PivotPointsResult,
    SupertrendResult,
    ParabolicSARResult,
    validate_series,
    validate_period,
    validate_ohlcv,
)

# Technical indicators
from src.indicators.technical import (
    # Trend
    sma,
    ema,
    vwma,
    parabolic_sar,
    supertrend,
    # Momentum
    rsi,
    macd,
    stochastic,
    williams_r,
    adx,
    # Volatility
    atr,
    bollinger_bands,
    keltner_channels,
    donchian_channels,
    # Volume
    vwap,
    obv,
    cmf,
    mfi,
    # Complex
    ichimoku_cloud,
    pivot_points,
)


class TechnicalIndicators:
    """
    Backward-compatible class for technical indicators.

    This class provides the same interface as the original TechnicalIndicators
    class in strategy.py, ensuring existing code continues to work.

    Example:
        >>> TechnicalIndicators.sma(prices, 20)
        >>> TechnicalIndicators.rsi(prices, 14)
    """

    # Trend indicators
    sma = staticmethod(sma)
    ema = staticmethod(ema)
    vwma = staticmethod(vwma)
    parabolic_sar = staticmethod(parabolic_sar)
    supertrend = staticmethod(supertrend)

    # Momentum indicators
    rsi = staticmethod(rsi)
    macd = staticmethod(macd)
    stochastic = staticmethod(stochastic)
    williams_r = staticmethod(williams_r)
    adx = staticmethod(adx)

    # Volatility indicators
    atr = staticmethod(atr)
    bollinger_bands = staticmethod(bollinger_bands)
    keltner_channels = staticmethod(keltner_channels)
    donchian_channels = staticmethod(donchian_channels)

    # Volume indicators
    vwap = staticmethod(vwap)
    obv = staticmethod(obv)
    cmf = staticmethod(cmf)
    mfi = staticmethod(mfi)

    # Complex indicators
    ichimoku_cloud = staticmethod(ichimoku_cloud)
    pivot_points = staticmethod(pivot_points)


__all__ = [
    # Base
    "IndicatorProtocol",
    "IndicatorResult",
    "BollingerBandsResult",
    "MACDResult",
    "StochasticResult",
    "ADXResult",
    "KeltnerChannelsResult",
    "DonchianChannelsResult",
    "IchimokuCloudResult",
    "PivotPointsResult",
    "SupertrendResult",
    "ParabolicSARResult",
    "validate_series",
    "validate_period",
    "validate_ohlcv",
    # Trend
    "sma",
    "ema",
    "vwma",
    "parabolic_sar",
    "supertrend",
    # Momentum
    "rsi",
    "macd",
    "stochastic",
    "williams_r",
    "adx",
    # Volatility
    "atr",
    "bollinger_bands",
    "keltner_channels",
    "donchian_channels",
    # Volume
    "vwap",
    "obv",
    "cmf",
    "mfi",
    # Complex
    "ichimoku_cloud",
    "pivot_points",
    # Backward compatibility
    "TechnicalIndicators",
]
