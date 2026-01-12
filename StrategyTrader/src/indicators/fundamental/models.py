"""Data models for fundamental indicators."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any


@dataclass
class MarketData:
    """Market data for a single asset."""
    symbol: str
    name: str
    price: float
    market_cap: float
    volume_24h: float
    change_1h: float
    change_24h: float
    change_7d: float
    change_30d: float
    ath: float
    ath_date: Optional[datetime]
    atl: float
    atl_date: Optional[datetime]
    circulating_supply: float
    total_supply: Optional[float]
    max_supply: Optional[float]
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "name": self.name,
            "price": self.price,
            "market_cap": self.market_cap,
            "volume_24h": self.volume_24h,
            "changes": {
                "1h": self.change_1h,
                "24h": self.change_24h,
                "7d": self.change_7d,
                "30d": self.change_30d,
            },
            "ath": self.ath,
            "ath_date": self.ath_date.isoformat() if self.ath_date else None,
            "atl": self.atl,
            "atl_date": self.atl_date.isoformat() if self.atl_date else None,
            "supply": {
                "circulating": self.circulating_supply,
                "total": self.total_supply,
                "max": self.max_supply,
            },
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class GlobalMarketData:
    """Global cryptocurrency market data."""
    total_market_cap: float
    total_volume_24h: float
    btc_dominance: float
    eth_dominance: float
    active_cryptocurrencies: int
    market_cap_change_24h: float
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_market_cap": self.total_market_cap,
            "total_volume_24h": self.total_volume_24h,
            "btc_dominance": self.btc_dominance,
            "eth_dominance": self.eth_dominance,
            "active_cryptocurrencies": self.active_cryptocurrencies,
            "market_cap_change_24h": self.market_cap_change_24h,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class FearGreedIndex:
    """Crypto Fear & Greed Index."""
    value: int  # 0-100
    classification: str  # "Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"
    timestamp: datetime = field(default_factory=datetime.now)

    @classmethod
    def classify(cls, value: int) -> str:
        """Get classification from value."""
        if value <= 20:
            return "Extreme Fear"
        elif value <= 40:
            return "Fear"
        elif value <= 60:
            return "Neutral"
        elif value <= 80:
            return "Greed"
        else:
            return "Extreme Greed"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "value": self.value,
            "classification": self.classification,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class TVLData:
    """Total Value Locked data for DeFi protocols."""
    protocol: str
    tvl: float
    chain: str
    change_1d: float
    change_7d: float
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "protocol": self.protocol,
            "tvl": self.tvl,
            "chain": self.chain,
            "change_1d": self.change_1d,
            "change_7d": self.change_7d,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ProtocolData:
    """Protocol data from DefiLlama."""
    name: str
    symbol: str
    chain: str
    tvl: float
    change_1d: float
    change_7d: float
    category: str
    chains: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "symbol": self.symbol,
            "chain": self.chain,
            "tvl": self.tvl,
            "change_1d": self.change_1d,
            "change_7d": self.change_7d,
            "category": self.category,
            "chains": self.chains,
        }


@dataclass
class OnChainMetrics:
    """On-chain metrics from Glassnode or similar providers."""
    symbol: str
    active_addresses: Optional[int] = None
    transaction_count: Optional[int] = None
    nvt_ratio: Optional[float] = None  # Network Value to Transactions
    mvrv_ratio: Optional[float] = None  # Market Value to Realized Value
    sopr: Optional[float] = None  # Spent Output Profit Ratio
    exchange_inflow: Optional[float] = None
    exchange_outflow: Optional[float] = None
    exchange_netflow: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "active_addresses": self.active_addresses,
            "transaction_count": self.transaction_count,
            "nvt_ratio": self.nvt_ratio,
            "mvrv_ratio": self.mvrv_ratio,
            "sopr": self.sopr,
            "exchange_flows": {
                "inflow": self.exchange_inflow,
                "outflow": self.exchange_outflow,
                "netflow": self.exchange_netflow,
            },
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class SocialMetrics:
    """Social metrics from Santiment or similar providers."""
    symbol: str
    social_volume: Optional[int] = None
    social_dominance: Optional[float] = None
    sentiment_score: Optional[float] = None  # -1 to 1
    dev_activity: Optional[float] = None
    github_commits: Optional[int] = None
    twitter_followers: Optional[int] = None
    reddit_subscribers: Optional[int] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "social_volume": self.social_volume,
            "social_dominance": self.social_dominance,
            "sentiment_score": self.sentiment_score,
            "dev_activity": self.dev_activity,
            "github_commits": self.github_commits,
            "twitter_followers": self.twitter_followers,
            "reddit_subscribers": self.reddit_subscribers,
            "timestamp": self.timestamp.isoformat(),
        }
