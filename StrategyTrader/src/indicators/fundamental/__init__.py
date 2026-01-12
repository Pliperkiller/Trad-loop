"""Fundamental indicators module - API clients for market and on-chain data."""

from src.indicators.fundamental.models import (
    MarketData,
    GlobalMarketData,
    FearGreedIndex,
    TVLData,
    ProtocolData,
    OnChainMetrics,
    SocialMetrics,
)
from src.indicators.fundamental.base_client import BaseAPIClient
from src.indicators.fundamental.coingecko import CoinGeckoClient
from src.indicators.fundamental.defillama import DefiLlamaClient
from src.indicators.fundamental.alternative_me import FearGreedClient
from src.indicators.fundamental.glassnode import GlassnodeClient
from src.indicators.fundamental.santiment import SantimentClient

__all__ = [
    # Models
    "MarketData",
    "GlobalMarketData",
    "FearGreedIndex",
    "TVLData",
    "ProtocolData",
    "OnChainMetrics",
    "SocialMetrics",
    # Base
    "BaseAPIClient",
    # Clients
    "CoinGeckoClient",
    "DefiLlamaClient",
    "FearGreedClient",
    "GlassnodeClient",
    "SantimentClient",
]
