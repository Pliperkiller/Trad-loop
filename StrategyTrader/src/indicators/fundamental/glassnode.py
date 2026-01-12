"""Glassnode API client for on-chain metrics (Paid API)."""

from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import pandas as pd

from src.indicators.fundamental.base_client import BaseAPIClient, AuthenticationError
from src.indicators.fundamental.models import OnChainMetrics


class GlassnodeClient(BaseAPIClient):
    """
    Glassnode API client for on-chain metrics.

    PAID API - Requires API key from https://glassnode.com/
    Free tier: Limited metrics, 10 API calls per day
    Paid tiers: More metrics, higher limits

    On-chain metrics available:
    - Active addresses
    - Transaction count
    - NVT ratio (Network Value to Transactions)
    - MVRV ratio (Market Value to Realized Value)
    - SOPR (Spent Output Profit Ratio)
    - Exchange flows (inflow/outflow/netflow)
    - Whale movements
    """

    BASE_URL = "https://api.glassnode.com/v1"

    def __init__(
        self,
        api_key: str,
        cache_ttl: int = 3600,
        rate_limit_per_min: int = 10,
    ):
        """
        Initialize Glassnode client.

        Args:
            api_key: Glassnode API key (required)
            cache_ttl: Cache TTL in seconds (default 1 hour - on-chain data updates slowly)
            rate_limit_per_min: Rate limit (default 10)

        Raises:
            ValueError: If api_key is not provided
        """
        if not api_key:
            raise ValueError("Glassnode API key is required")

        super().__init__(
            base_url=self.BASE_URL,
            api_key=api_key,
            cache_ttl=cache_ttl,
            rate_limit_per_min=rate_limit_per_min,
        )

    def _add_auth_header(self) -> None:
        """Add Glassnode API key as query parameter."""
        pass  # Glassnode uses query param, not header

    def get(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Override to add API key to params."""
        if params is None:
            params = {}
        params["api_key"] = self.api_key
        return super().get(endpoint, params=params, **kwargs)

    def _get_metric(
        self,
        metric: str,
        asset: str = "BTC",
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        resolution: str = "24h",
    ) -> pd.DataFrame:
        """
        Get a specific on-chain metric.

        Args:
            metric: Metric name (e.g., "addresses/active_count")
            asset: Asset symbol (default "BTC")
            since: Start date
            until: End date
            resolution: Data resolution ("1h", "24h", "10m")

        Returns:
            DataFrame with timestamp and value columns
        """
        params: Dict[str, Any] = {
            "a": asset,
            "i": resolution,
        }

        if since:
            params["s"] = int(since.timestamp())
        if until:
            params["u"] = int(until.timestamp())

        result = self.get(f"metrics/{metric}", params=params)

        if not result:
            return pd.DataFrame(columns=["timestamp", "value"])

        df = pd.DataFrame(result)
        if "t" in df.columns:
            df["timestamp"] = pd.to_datetime(df["t"], unit="s")
            df.rename(columns={"v": "value"}, inplace=True)
            df = df[["timestamp", "value"]]
            df.set_index("timestamp", inplace=True)

        return df

    def get_active_addresses(
        self,
        asset: str = "BTC",
        days: int = 30,
    ) -> pd.DataFrame:
        """
        Get active address count.

        Args:
            asset: Asset symbol
            days: Number of days

        Returns:
            DataFrame with daily active address counts
        """
        since = datetime.now() - timedelta(days=days)
        return self._get_metric("addresses/active_count", asset=asset, since=since)

    def get_nvt_ratio(
        self,
        asset: str = "BTC",
        days: int = 30,
    ) -> pd.DataFrame:
        """
        Get NVT (Network Value to Transactions) ratio.

        High NVT suggests overvaluation, low NVT suggests undervaluation.

        Args:
            asset: Asset symbol
            days: Number of days

        Returns:
            DataFrame with NVT values
        """
        since = datetime.now() - timedelta(days=days)
        return self._get_metric("indicators/nvt", asset=asset, since=since)

    def get_mvrv_ratio(
        self,
        asset: str = "BTC",
        days: int = 30,
    ) -> pd.DataFrame:
        """
        Get MVRV (Market Value to Realized Value) ratio.

        MVRV > 3.5: Market may be overheated
        MVRV < 1: Market may be undervalued

        Args:
            asset: Asset symbol
            days: Number of days

        Returns:
            DataFrame with MVRV values
        """
        since = datetime.now() - timedelta(days=days)
        return self._get_metric("indicators/mvrv", asset=asset, since=since)

    def get_sopr(
        self,
        asset: str = "BTC",
        days: int = 30,
    ) -> pd.DataFrame:
        """
        Get SOPR (Spent Output Profit Ratio).

        SOPR > 1: Sellers are in profit
        SOPR < 1: Sellers are at loss
        SOPR = 1: Breakeven

        Args:
            asset: Asset symbol
            days: Number of days

        Returns:
            DataFrame with SOPR values
        """
        since = datetime.now() - timedelta(days=days)
        return self._get_metric("indicators/sopr", asset=asset, since=since)

    def get_exchange_netflow(
        self,
        asset: str = "BTC",
        days: int = 30,
    ) -> pd.DataFrame:
        """
        Get exchange netflow (inflow - outflow).

        Positive: More coins going to exchanges (selling pressure)
        Negative: More coins leaving exchanges (accumulation)

        Args:
            asset: Asset symbol
            days: Number of days

        Returns:
            DataFrame with netflow values
        """
        since = datetime.now() - timedelta(days=days)
        return self._get_metric("transactions/transfers_to_exchanges_count", asset=asset, since=since)

    def get_on_chain_metrics(
        self,
        asset: str = "BTC",
    ) -> OnChainMetrics:
        """
        Get latest on-chain metrics snapshot.

        Args:
            asset: Asset symbol

        Returns:
            OnChainMetrics object with latest values
        """
        # Get latest values (1 day of data)
        active = self.get_active_addresses(asset, days=1)
        nvt = self.get_nvt_ratio(asset, days=1)
        mvrv = self.get_mvrv_ratio(asset, days=1)
        sopr = self.get_sopr(asset, days=1)

        return OnChainMetrics(
            symbol=asset,
            active_addresses=int(active["value"].iloc[-1]) if not active.empty else None,
            nvt_ratio=float(nvt["value"].iloc[-1]) if not nvt.empty else None,
            mvrv_ratio=float(mvrv["value"].iloc[-1]) if not mvrv.empty else None,
            sopr=float(sopr["value"].iloc[-1]) if not sopr.empty else None,
        )

    def get_available_metrics(self) -> List[str]:
        """
        Get list of available metrics.

        Returns:
            List of metric endpoint names
        """
        # This is a subset of commonly used metrics
        return [
            "addresses/active_count",
            "addresses/new_non_zero_count",
            "transactions/count",
            "transactions/transfers_to_exchanges_count",
            "transactions/transfers_from_exchanges_count",
            "indicators/nvt",
            "indicators/mvrv",
            "indicators/sopr",
            "market/price_usd",
            "supply/current",
        ]

    def get_supported_assets(self) -> List[str]:
        """
        Get list of supported assets.

        Returns:
            List of asset symbols
        """
        return ["BTC", "ETH", "LTC", "AAVE", "UNI", "LINK", "MKR", "COMP"]
