"""DefiLlama API client for DeFi TVL data."""

from datetime import datetime
from typing import List, Optional, Dict, Any
import pandas as pd

from src.indicators.fundamental.base_client import BaseAPIClient
from src.indicators.fundamental.models import TVLData, ProtocolData


class DefiLlamaClient(BaseAPIClient):
    """
    DefiLlama API client for DeFi protocol data.

    No rate limit documented, generous free access.
    API docs: https://defillama.com/docs/api

    Features:
    - Protocol TVL data
    - Chain TVL data
    - Historical TVL
    - Protocol listings
    """

    BASE_URL = "https://api.llama.fi"

    def __init__(
        self,
        cache_ttl: int = 300,
        rate_limit_per_min: int = 60,
    ):
        """
        Initialize DefiLlama client.

        Args:
            cache_ttl: Cache TTL in seconds (default 5 min)
            rate_limit_per_min: Rate limit (default 60)
        """
        super().__init__(
            base_url=self.BASE_URL,
            cache_ttl=cache_ttl,
            rate_limit_per_min=rate_limit_per_min,
        )

    def get_protocol_tvl(self, protocol: str) -> float:
        """
        Get current TVL for a protocol.

        Args:
            protocol: Protocol slug (e.g., "aave", "uniswap")

        Returns:
            TVL in USD
        """
        result = self.get(f"protocol/{protocol}")
        return result.get("tvl", 0.0)

    def get_protocol_data(self, protocol: str) -> ProtocolData:
        """
        Get detailed protocol data.

        Args:
            protocol: Protocol slug

        Returns:
            ProtocolData object
        """
        result = self.get(f"protocol/{protocol}")

        return ProtocolData(
            name=result.get("name", ""),
            symbol=result.get("symbol", ""),
            chain=result.get("chain", ""),
            tvl=result.get("tvl", 0.0),
            change_1d=result.get("change_1d", 0.0) or 0.0,
            change_7d=result.get("change_7d", 0.0) or 0.0,
            category=result.get("category", ""),
            chains=result.get("chains", []),
        )

    def get_protocol_tvl_history(self, protocol: str) -> pd.DataFrame:
        """
        Get historical TVL for a protocol.

        Args:
            protocol: Protocol slug

        Returns:
            DataFrame with date and tvl columns
        """
        result = self.get(f"protocol/{protocol}")
        tvl_history = result.get("tvl", [])

        if not tvl_history or not isinstance(tvl_history, list):
            return pd.DataFrame(columns=["date", "tvl"])

        # Extract from chain TVLs if needed
        chain_tvls = result.get("chainTvls", {})
        if chain_tvls:
            # Use first chain's TVL history
            first_chain = list(chain_tvls.values())[0]
            tvl_history = first_chain.get("tvl", [])

        df = pd.DataFrame(tvl_history)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], unit="s")
            df.set_index("date", inplace=True)
        return df

    def get_chain_tvl(self, chain: str) -> float:
        """
        Get current TVL for a blockchain.

        Args:
            chain: Chain name (e.g., "ethereum", "bsc", "polygon")

        Returns:
            TVL in USD
        """
        result = self.get(f"v2/chains")

        for chain_data in result:
            if chain_data.get("name", "").lower() == chain.lower():
                return chain_data.get("tvl", 0.0)

        return 0.0

    def get_all_chains_tvl(self) -> Dict[str, float]:
        """
        Get TVL for all chains.

        Returns:
            Dict mapping chain name to TVL
        """
        result = self.get("v2/chains")

        return {
            chain.get("name", ""): chain.get("tvl", 0.0)
            for chain in result
        }

    def get_top_protocols(
        self,
        n: int = 20,
        chain: Optional[str] = None,
    ) -> List[ProtocolData]:
        """
        Get top protocols by TVL.

        Args:
            n: Number of protocols to return
            chain: Filter by chain (optional)

        Returns:
            List of ProtocolData sorted by TVL
        """
        result = self.get("protocols")

        protocols = []
        for item in result:
            if chain and chain.lower() not in [c.lower() for c in item.get("chains", [])]:
                continue

            protocols.append(ProtocolData(
                name=item.get("name", ""),
                symbol=item.get("symbol", ""),
                chain=item.get("chain", ""),
                tvl=item.get("tvl", 0.0),
                change_1d=item.get("change_1d", 0.0) or 0.0,
                change_7d=item.get("change_7d", 0.0) or 0.0,
                category=item.get("category", ""),
                chains=item.get("chains", []),
            ))

        # Sort by TVL descending
        protocols.sort(key=lambda p: p.tvl, reverse=True)

        return protocols[:n]

    def get_global_tvl(self) -> float:
        """
        Get total TVL across all DeFi protocols.

        Returns:
            Global TVL in USD
        """
        result = self.get("v2/chains")
        return sum(chain.get("tvl", 0.0) for chain in result)

    def get_tvl_history(self) -> pd.DataFrame:
        """
        Get historical global TVL.

        Returns:
            DataFrame with date and totalLiquidityUSD columns
        """
        result = self.get("v2/historicalChainTvl")

        df = pd.DataFrame(result)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], unit="s")
            df.set_index("date", inplace=True)
        return df

    def get_chain_tvl_history(self, chain: str) -> pd.DataFrame:
        """
        Get historical TVL for a chain.

        Args:
            chain: Chain name

        Returns:
            DataFrame with date and tvl columns
        """
        result = self.get(f"v2/historicalChainTvl/{chain}")

        df = pd.DataFrame(result)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], unit="s")
            df.set_index("date", inplace=True)
        return df

    def get_stablecoins(self) -> List[Dict[str, Any]]:
        """
        Get list of stablecoins with market cap.

        Returns:
            List of stablecoin data
        """
        result = self.get("stablecoins")
        return result.get("peggedAssets", [])

    def get_yields(self, chain: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get DeFi yield/APY data.

        Args:
            chain: Filter by chain (optional)

        Returns:
            List of yield pool data
        """
        result = self.get("pools")
        pools = result.get("data", [])

        if chain:
            pools = [p for p in pools if p.get("chain", "").lower() == chain.lower()]

        return pools
