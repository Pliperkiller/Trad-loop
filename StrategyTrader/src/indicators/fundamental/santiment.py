"""Santiment API client for social and dev metrics (Paid API)."""

from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import pandas as pd

from src.indicators.fundamental.base_client import BaseAPIClient
from src.indicators.fundamental.models import SocialMetrics


class SantimentClient(BaseAPIClient):
    """
    Santiment API client for social and development metrics.

    PAID API - Requires API key from https://santiment.net/
    Free tier: Limited access
    Paid tiers: Full access to all metrics

    Metrics available:
    - Social volume (mentions across platforms)
    - Social dominance (share of social discussions)
    - Sentiment score (positive/negative sentiment)
    - Dev activity (GitHub commits, activity)
    - Price correlation with social metrics
    """

    BASE_URL = "https://api.santiment.net/graphql"

    def __init__(
        self,
        api_key: str,
        cache_ttl: int = 1800,
        rate_limit_per_min: int = 10,
    ):
        """
        Initialize Santiment client.

        Args:
            api_key: Santiment API key (required)
            cache_ttl: Cache TTL in seconds (default 30 min)
            rate_limit_per_min: Rate limit (default 10)

        Raises:
            ValueError: If api_key is not provided
        """
        if not api_key:
            raise ValueError("Santiment API key is required")

        super().__init__(
            base_url=self.BASE_URL,
            api_key=api_key,
            cache_ttl=cache_ttl,
            rate_limit_per_min=rate_limit_per_min,
        )

    def _add_auth_header(self) -> None:
        """Add Santiment authorization header."""
        self._session.headers["Authorization"] = f"Apikey {self.api_key}"

    def _graphql_query(
        self,
        query: str,
        variables: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute GraphQL query.

        Args:
            query: GraphQL query string
            variables: Query variables

        Returns:
            Query result data
        """
        payload = {"query": query}
        if variables:
            payload["variables"] = variables

        result = self.post("", data=payload)
        return result.get("data", {})

    def _get_metric(
        self,
        metric: str,
        slug: str,
        from_date: datetime,
        to_date: datetime,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Get a specific metric for an asset.

        Args:
            metric: Metric name
            slug: Asset slug (e.g., "bitcoin", "ethereum")
            from_date: Start date
            to_date: End date
            interval: Data interval ("1h", "1d")

        Returns:
            DataFrame with datetime and value columns
        """
        query = """
        query getMetric($metric: String!, $slug: String!, $from: DateTime!, $to: DateTime!, $interval: String!) {
            getMetric(metric: $metric) {
                timeseriesData(slug: $slug, from: $from, to: $to, interval: $interval) {
                    datetime
                    value
                }
            }
        }
        """

        variables = {
            "metric": metric,
            "slug": slug,
            "from": from_date.isoformat(),
            "to": to_date.isoformat(),
            "interval": interval,
        }

        result = self._graphql_query(query, variables)
        data = result.get("getMetric", {}).get("timeseriesData", [])

        if not data:
            return pd.DataFrame(columns=["datetime", "value"])

        df = pd.DataFrame(data)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df.set_index("datetime", inplace=True)

        return df

    def get_social_volume(
        self,
        slug: str = "bitcoin",
        days: int = 30,
    ) -> pd.DataFrame:
        """
        Get social volume (total mentions across platforms).

        Args:
            slug: Asset slug
            days: Number of days

        Returns:
            DataFrame with social volume data
        """
        to_date = datetime.now()
        from_date = to_date - timedelta(days=days)

        return self._get_metric(
            "social_volume_total",
            slug=slug,
            from_date=from_date,
            to_date=to_date,
        )

    def get_social_dominance(
        self,
        slug: str = "bitcoin",
        days: int = 30,
    ) -> pd.DataFrame:
        """
        Get social dominance (share of all crypto discussions).

        Args:
            slug: Asset slug
            days: Number of days

        Returns:
            DataFrame with dominance percentages
        """
        to_date = datetime.now()
        from_date = to_date - timedelta(days=days)

        return self._get_metric(
            "social_dominance_total",
            slug=slug,
            from_date=from_date,
            to_date=to_date,
        )

    def get_sentiment(
        self,
        slug: str = "bitcoin",
        days: int = 30,
    ) -> pd.DataFrame:
        """
        Get sentiment score.

        Positive values indicate bullish sentiment,
        negative values indicate bearish sentiment.

        Args:
            slug: Asset slug
            days: Number of days

        Returns:
            DataFrame with sentiment scores
        """
        to_date = datetime.now()
        from_date = to_date - timedelta(days=days)

        return self._get_metric(
            "sentiment_balance_total",
            slug=slug,
            from_date=from_date,
            to_date=to_date,
        )

    def get_dev_activity(
        self,
        slug: str = "bitcoin",
        days: int = 30,
    ) -> pd.DataFrame:
        """
        Get development activity (GitHub commits, PRs, etc.).

        Args:
            slug: Asset slug
            days: Number of days

        Returns:
            DataFrame with dev activity scores
        """
        to_date = datetime.now()
        from_date = to_date - timedelta(days=days)

        return self._get_metric(
            "dev_activity",
            slug=slug,
            from_date=from_date,
            to_date=to_date,
        )

    def get_github_activity(
        self,
        slug: str = "bitcoin",
        days: int = 30,
    ) -> pd.DataFrame:
        """
        Get GitHub-specific activity.

        Args:
            slug: Asset slug
            days: Number of days

        Returns:
            DataFrame with GitHub activity data
        """
        to_date = datetime.now()
        from_date = to_date - timedelta(days=days)

        return self._get_metric(
            "github_activity",
            slug=slug,
            from_date=from_date,
            to_date=to_date,
        )

    def get_social_metrics(
        self,
        slug: str = "bitcoin",
    ) -> SocialMetrics:
        """
        Get latest social metrics snapshot.

        Args:
            slug: Asset slug

        Returns:
            SocialMetrics object with latest values
        """
        # Get latest values (1 day of data)
        social_vol = self.get_social_volume(slug, days=1)
        social_dom = self.get_social_dominance(slug, days=1)
        sentiment = self.get_sentiment(slug, days=1)
        dev = self.get_dev_activity(slug, days=1)

        return SocialMetrics(
            symbol=slug.upper(),
            social_volume=int(social_vol["value"].iloc[-1]) if not social_vol.empty else None,
            social_dominance=float(social_dom["value"].iloc[-1]) if not social_dom.empty else None,
            sentiment_score=float(sentiment["value"].iloc[-1]) if not sentiment.empty else None,
            dev_activity=float(dev["value"].iloc[-1]) if not dev.empty else None,
        )

    def get_trending_words(self, size: int = 10) -> List[Dict[str, Any]]:
        """
        Get trending words in crypto discussions.

        Args:
            size: Number of words to return

        Returns:
            List of trending words with scores
        """
        query = """
        query getTrendingWords($size: Int!) {
            getTrendingWords(size: $size) {
                word
                score
            }
        }
        """

        result = self._graphql_query(query, {"size": size})
        return result.get("getTrendingWords", [])

    def search_projects(self, query: str) -> List[Dict[str, Any]]:
        """
        Search for projects by name or ticker.

        Args:
            query: Search query

        Returns:
            List of matching projects
        """
        gql_query = """
        query searchProjects($query: String!) {
            allProjects(page: 1, pageSize: 20) {
                slug
                name
                ticker
            }
        }
        """

        result = self._graphql_query(gql_query, {"query": query})
        projects = result.get("allProjects", [])

        # Filter by query
        query_lower = query.lower()
        return [
            p for p in projects
            if query_lower in p.get("name", "").lower()
            or query_lower in p.get("ticker", "").lower()
            or query_lower in p.get("slug", "").lower()
        ]

    def get_supported_assets(self) -> List[str]:
        """
        Get list of commonly supported asset slugs.

        Returns:
            List of asset slugs
        """
        return [
            "bitcoin",
            "ethereum",
            "ripple",
            "cardano",
            "solana",
            "polkadot",
            "dogecoin",
            "avalanche",
            "chainlink",
            "uniswap",
        ]
