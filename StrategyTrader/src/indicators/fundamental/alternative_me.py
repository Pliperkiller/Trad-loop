"""Alternative.me API client for Fear & Greed Index."""

from datetime import datetime
from typing import Optional
import pandas as pd

from src.indicators.fundamental.base_client import BaseAPIClient
from src.indicators.fundamental.models import FearGreedIndex


class FearGreedClient(BaseAPIClient):
    """
    Alternative.me Fear & Greed Index API client.

    No documented rate limit, generous free access.
    API docs: https://alternative.me/crypto/fear-and-greed-index/

    The Fear & Greed Index ranges from 0 (Extreme Fear) to 100 (Extreme Greed).

    Classifications:
    - 0-24: Extreme Fear
    - 25-44: Fear
    - 45-55: Neutral
    - 56-75: Greed
    - 76-100: Extreme Greed
    """

    BASE_URL = "https://api.alternative.me/fng"

    def __init__(
        self,
        cache_ttl: int = 600,
        rate_limit_per_min: int = 30,
    ):
        """
        Initialize Fear & Greed client.

        Args:
            cache_ttl: Cache TTL in seconds (default 10 min - updates slowly)
            rate_limit_per_min: Rate limit (default 30)
        """
        super().__init__(
            base_url=self.BASE_URL,
            cache_ttl=cache_ttl,
            rate_limit_per_min=rate_limit_per_min,
        )

    def _build_url(self, endpoint: str) -> str:
        """Override to handle base URL structure."""
        if endpoint:
            return f"{self.base_url}/{endpoint.lstrip('/')}"
        return self.base_url

    def get_current(self) -> FearGreedIndex:
        """
        Get current Fear & Greed Index.

        Returns:
            FearGreedIndex with current value and classification
        """
        result = self.get("", params={"limit": 1})
        data = result.get("data", [{}])[0]

        value = int(data.get("value", 50))
        classification = data.get("value_classification", FearGreedIndex.classify(value))

        timestamp_str = data.get("timestamp")
        if timestamp_str:
            timestamp = datetime.fromtimestamp(int(timestamp_str))
        else:
            timestamp = datetime.now()

        return FearGreedIndex(
            value=value,
            classification=classification,
            timestamp=timestamp,
        )

    def get_historical(
        self,
        days: int = 30,
        date_format: str = "world",
    ) -> pd.DataFrame:
        """
        Get historical Fear & Greed Index data.

        Args:
            days: Number of days to fetch (max varies, usually up to 365+)
            date_format: Date format ("world" for DD-MM-YYYY, "us" for MM-DD-YYYY)

        Returns:
            DataFrame with date, value, and classification columns
        """
        result = self.get(
            "",
            params={
                "limit": days,
                "date_format": date_format,
            }
        )

        data = result.get("data", [])

        records = []
        for item in data:
            value = int(item.get("value", 50))
            timestamp_str = item.get("timestamp")

            if timestamp_str:
                timestamp = datetime.fromtimestamp(int(timestamp_str))
            else:
                continue

            records.append({
                "timestamp": timestamp,
                "value": value,
                "classification": item.get("value_classification", FearGreedIndex.classify(value)),
            })

        df = pd.DataFrame(records)
        if not df.empty:
            df.set_index("timestamp", inplace=True)
            df.sort_index(inplace=True)

        return df

    def get_value(self) -> int:
        """
        Get current index value only.

        Returns:
            Fear & Greed value (0-100)
        """
        return self.get_current().value

    def get_classification(self) -> str:
        """
        Get current classification only.

        Returns:
            Classification string
        """
        return self.get_current().classification

    def is_extreme_fear(self) -> bool:
        """Check if market is in Extreme Fear (value <= 20)."""
        return self.get_value() <= 20

    def is_fear(self) -> bool:
        """Check if market is in Fear (value <= 40)."""
        return self.get_value() <= 40

    def is_greed(self) -> bool:
        """Check if market is in Greed (value >= 60)."""
        return self.get_value() >= 60

    def is_extreme_greed(self) -> bool:
        """Check if market is in Extreme Greed (value >= 80)."""
        return self.get_value() >= 80

    def get_average(self, days: int = 7) -> float:
        """
        Get average Fear & Greed value over period.

        Args:
            days: Number of days to average

        Returns:
            Average value
        """
        df = self.get_historical(days=days)
        if df.empty:
            return 50.0
        return float(df["value"].mean())

    def get_trend(self, days: int = 7) -> str:
        """
        Get Fear & Greed trend direction.

        Args:
            days: Number of days to analyze

        Returns:
            "improving", "worsening", or "stable"
        """
        df = self.get_historical(days=days)
        if df.empty or len(df) < 2:
            return "stable"

        first_half = df.iloc[:len(df)//2]["value"].mean()
        second_half = df.iloc[len(df)//2:]["value"].mean()

        diff = second_half - first_half
        if diff > 5:
            return "improving"
        elif diff < -5:
            return "worsening"
        else:
            return "stable"
