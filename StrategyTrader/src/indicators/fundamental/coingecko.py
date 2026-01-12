"""CoinGecko API client for market data."""

from datetime import datetime
from typing import Dict, List, Optional, Any
import pandas as pd

from src.indicators.fundamental.base_client import BaseAPIClient
from src.indicators.fundamental.models import MarketData, GlobalMarketData


class CoinGeckoClient(BaseAPIClient):
    """
    CoinGecko API client for cryptocurrency market data.

    Free tier limits: 10-30 calls/minute (varies)
    API docs: https://www.coingecko.com/en/api/documentation

    Features:
    - Price data
    - Market cap and volume
    - Historical prices
    - Global market data
    """

    BASE_URL = "https://api.coingecko.com/api/v3"

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_ttl: int = 300,
        rate_limit_per_min: int = 10,
    ):
        """
        Initialize CoinGecko client.

        Args:
            api_key: Optional Pro API key for higher limits
            cache_ttl: Cache TTL in seconds (default 5 min)
            rate_limit_per_min: Rate limit (default 10 for free tier)
        """
        super().__init__(
            base_url=self.BASE_URL,
            api_key=api_key,
            cache_ttl=cache_ttl,
            rate_limit_per_min=rate_limit_per_min,
        )

    def _add_auth_header(self) -> None:
        """Add CoinGecko Pro API key header."""
        if self.api_key:
            self._session.headers["x-cg-pro-api-key"] = self.api_key

    def ping(self) -> bool:
        """Check API status."""
        try:
            result = self.get("ping", use_cache=False)
            return "gecko_says" in result
        except Exception:
            return False

    def get_price(
        self,
        coin_id: str,
        vs_currency: str = "usd",
    ) -> float:
        """
        Get current price for a coin.

        Args:
            coin_id: CoinGecko coin ID (e.g., "bitcoin", "ethereum")
            vs_currency: Target currency (default "usd")

        Returns:
            Current price
        """
        result = self.get(
            "simple/price",
            params={
                "ids": coin_id,
                "vs_currencies": vs_currency,
            }
        )
        return result.get(coin_id, {}).get(vs_currency, 0.0)

    def get_prices(
        self,
        coin_ids: List[str],
        vs_currency: str = "usd",
        include_24h_change: bool = True,
    ) -> Dict[str, Dict[str, float]]:
        """
        Get prices for multiple coins.

        Args:
            coin_ids: List of CoinGecko coin IDs
            vs_currency: Target currency
            include_24h_change: Include 24h change percentage

        Returns:
            Dict mapping coin_id to price data
        """
        params = {
            "ids": ",".join(coin_ids),
            "vs_currencies": vs_currency,
        }
        if include_24h_change:
            params["include_24hr_change"] = "true"

        return self.get("simple/price", params=params)

    def get_market_data(
        self,
        coin_id: str,
        vs_currency: str = "usd",
    ) -> MarketData:
        """
        Get detailed market data for a coin.

        Args:
            coin_id: CoinGecko coin ID
            vs_currency: Target currency

        Returns:
            MarketData object with comprehensive data
        """
        result = self.get(
            f"coins/{coin_id}",
            params={
                "localization": "false",
                "tickers": "false",
                "market_data": "true",
                "community_data": "false",
                "developer_data": "false",
            }
        )

        market = result.get("market_data", {})

        # Parse ATH date
        ath_date = None
        ath_date_str = market.get("ath_date", {}).get(vs_currency)
        if ath_date_str:
            try:
                ath_date = datetime.fromisoformat(ath_date_str.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass

        # Parse ATL date
        atl_date = None
        atl_date_str = market.get("atl_date", {}).get(vs_currency)
        if atl_date_str:
            try:
                atl_date = datetime.fromisoformat(atl_date_str.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass

        return MarketData(
            symbol=result.get("symbol", "").upper(),
            name=result.get("name", ""),
            price=market.get("current_price", {}).get(vs_currency, 0.0),
            market_cap=market.get("market_cap", {}).get(vs_currency, 0.0),
            volume_24h=market.get("total_volume", {}).get(vs_currency, 0.0),
            change_1h=market.get("price_change_percentage_1h_in_currency", {}).get(vs_currency, 0.0) or 0.0,
            change_24h=market.get("price_change_percentage_24h", 0.0) or 0.0,
            change_7d=market.get("price_change_percentage_7d", 0.0) or 0.0,
            change_30d=market.get("price_change_percentage_30d", 0.0) or 0.0,
            ath=market.get("ath", {}).get(vs_currency, 0.0),
            ath_date=ath_date,
            atl=market.get("atl", {}).get(vs_currency, 0.0),
            atl_date=atl_date,
            circulating_supply=market.get("circulating_supply", 0.0) or 0.0,
            total_supply=market.get("total_supply"),
            max_supply=market.get("max_supply"),
        )

    def get_market_chart(
        self,
        coin_id: str,
        vs_currency: str = "usd",
        days: int = 30,
    ) -> pd.DataFrame:
        """
        Get historical market data.

        Args:
            coin_id: CoinGecko coin ID
            vs_currency: Target currency
            days: Number of days (1, 7, 14, 30, 90, 180, 365, max)

        Returns:
            DataFrame with columns: timestamp, price, market_cap, volume
        """
        result = self.get(
            f"coins/{coin_id}/market_chart",
            params={
                "vs_currency": vs_currency,
                "days": days,
            }
        )

        prices = result.get("prices", [])
        market_caps = result.get("market_caps", [])
        volumes = result.get("total_volumes", [])

        df = pd.DataFrame({
            "timestamp": [p[0] for p in prices],
            "price": [p[1] for p in prices],
            "market_cap": [m[1] for m in market_caps] if market_caps else [None] * len(prices),
            "volume": [v[1] for v in volumes] if volumes else [None] * len(prices),
        })

        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)

        return df

    def get_global_data(self) -> GlobalMarketData:
        """
        Get global cryptocurrency market data.

        Returns:
            GlobalMarketData with market overview
        """
        result = self.get("global")
        data = result.get("data", {})

        return GlobalMarketData(
            total_market_cap=data.get("total_market_cap", {}).get("usd", 0.0),
            total_volume_24h=data.get("total_volume", {}).get("usd", 0.0),
            btc_dominance=data.get("market_cap_percentage", {}).get("btc", 0.0),
            eth_dominance=data.get("market_cap_percentage", {}).get("eth", 0.0),
            active_cryptocurrencies=data.get("active_cryptocurrencies", 0),
            market_cap_change_24h=data.get("market_cap_change_percentage_24h_usd", 0.0),
        )

    def get_trending(self) -> List[Dict[str, Any]]:
        """
        Get trending coins.

        Returns:
            List of trending coin data
        """
        result = self.get("search/trending")
        coins = result.get("coins", [])
        return [coin.get("item", {}) for coin in coins]

    def search(self, query: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Search for coins, exchanges, and categories.

        Args:
            query: Search query

        Returns:
            Dict with coins, exchanges, categories lists
        """
        return self.get("search", params={"query": query})

    def get_coin_list(self) -> List[Dict[str, str]]:
        """
        Get list of all supported coins with IDs.

        Returns:
            List of dicts with id, symbol, name
        """
        return self.get("coins/list")
