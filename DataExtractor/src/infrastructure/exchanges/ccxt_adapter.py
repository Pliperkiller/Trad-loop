"""CCXT Adapter - Universal exchange support for 100+ exchanges."""

from datetime import datetime
from typing import Any, Callable, Dict, Optional

import ccxt
import pandas as pd
from loguru import logger

from ...domain.repositories import IMarketRepository
from ...domain.entities import Timeframe


class CCXTAdapter(IMarketRepository):
    """
    Universal exchange adapter using CCXT library.

    Supports 100+ exchanges including:
    - Binance, Coinbase, Kraken, Bitfinex
    - Bybit, OKX, KuCoin, Gate.io
    - Huobi, Bitget, MEXC, and many more

    Features:
    - Unified API across all exchanges
    - Automatic rate limiting
    - Error handling and retries
    - Progress callbacks
    """

    def __init__(self, exchange_id: str, api_credentials: Optional[Dict[str, str]] = None):
        """
        Initialize CCXT adapter.

        Args:
            exchange_id: Exchange ID (e.g., 'binance', 'coinbase', 'bybit')
            api_credentials: Optional API credentials for authenticated endpoints
                Example: {"apiKey": "...", "secret": "..."}
        """
        self.exchange_id = exchange_id.lower()

        # Initialize CCXT exchange
        try:
            exchange_class = getattr(ccxt, self.exchange_id)
        except AttributeError:
            raise ValueError(
                f"Exchange '{exchange_id}' not supported by CCXT. "
                f"Available exchanges: {', '.join(ccxt.exchanges[:10])}..."
            )

        # Build exchange config
        config = {
            "enableRateLimit": True,  # Automatic rate limiting
            # Usar rate limit nativo del exchange (50ms para Binance)
            # en lugar de 1000ms que era excesivamente conservador
        }
        self._markets_loaded = False  # Cache para evitar load_markets() repetidos

        if api_credentials:
            config.update(api_credentials)

        self.exchange = exchange_class(config)

        logger.info(
            f"Initialized CCXTAdapter for {self.exchange_id} "
            f"(supports {len(ccxt.exchanges)} exchanges)"
        )

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: Timeframe,
        start_date: datetime,
        end_date: datetime,
        market_type: str = "SPOT",
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data from exchange using CCXT.

        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Timeframe enum
            start_date: Start date
            end_date: End date
            market_type: SPOT, FUTURES, etc. (CCXT handles automatically)
            progress_callback: Optional callback for progress updates

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        try:
            # Load markets (solo una vez, cacheado)
            if not self._markets_loaded:
                self.exchange.load_markets()
                self._markets_loaded = True

            # Validate symbol
            if symbol not in self.exchange.symbols:
                raise ValueError(
                    f"Symbol '{symbol}' not available on {self.exchange_id}. "
                    f"Use format like 'BTC/USDT', 'ETH/USD', etc."
                )

            # Map Timeframe enum to CCXT timeframe string
            tf_str = self._map_timeframe(timeframe)

            # Convert dates to milliseconds
            since = int(start_date.timestamp() * 1000)
            end_ms = int(end_date.timestamp() * 1000)

            logger.info(
                f"Fetching {symbol} {tf_str} data from {self.exchange_id} "
                f"({start_date.date()} to {end_date.date()})"
            )

            # Fetch data in chunks (CCXT has limits per request)
            all_ohlcv = []
            current_since = since
            chunk_limit = 1000  # Most exchanges limit to 500-1000 candles per request

            total_expected = int((end_ms - since) / self._timeframe_to_ms(tf_str))
            fetched_count = 0

            while current_since < end_ms:
                try:
                    # Fetch chunk
                    ohlcv = self.exchange.fetch_ohlcv(
                        symbol=symbol,
                        timeframe=tf_str,
                        since=current_since,
                        limit=chunk_limit
                    )

                    if not ohlcv:
                        break

                    # Filter out data beyond end_date
                    ohlcv = [candle for candle in ohlcv if candle[0] <= end_ms]

                    all_ohlcv.extend(ohlcv)
                    fetched_count += len(ohlcv)

                    # Progress callback
                    if progress_callback:
                        progress_callback(fetched_count, total_expected)

                    # Update since for next chunk
                    if ohlcv:
                        last_timestamp = ohlcv[-1][0]
                        current_since = last_timestamp + 1
                    else:
                        break

                    # Prevent infinite loops
                    if len(all_ohlcv) > 100000:
                        logger.warning(
                            f"Fetched {len(all_ohlcv)} candles, stopping to prevent memory issues"
                        )
                        break

                except ccxt.RateLimitExceeded:
                    logger.warning("Rate limit exceeded, waiting...")
                    import time
                    time.sleep(self.exchange.rateLimit / 1000)
                    continue

                except ccxt.NetworkError as e:
                    logger.error(f"Network error: {e}")
                    raise

            # Convert to DataFrame
            if not all_ohlcv:
                logger.warning(f"No data fetched for {symbol} {tf_str}")
                return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

            df = pd.DataFrame(
                all_ohlcv,
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )

            # Convert timestamp to datetime
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

            # Sort by timestamp
            df = df.sort_values("timestamp").reset_index(drop=True)

            # Remove duplicates
            df = df.drop_duplicates(subset=["timestamp"], keep="last")

            logger.info(
                f"Successfully fetched {len(df)} candles from {self.exchange_id} "
                f"({df['timestamp'].min()} to {df['timestamp'].max()})"
            )

            return df

        except Exception as e:
            logger.error(f"Error fetching data from {self.exchange_id}: {e}")
            raise

    def _map_timeframe(self, timeframe: Timeframe) -> str:
        """Map Timeframe enum to CCXT timeframe string."""
        mapping = {
            Timeframe.ONE_MINUTE: "1m",
            Timeframe.FIVE_MINUTES: "5m",
            Timeframe.FIFTEEN_MINUTES: "15m",
            Timeframe.THIRTY_MINUTES: "30m",
            Timeframe.ONE_HOUR: "1h",
            Timeframe.TWO_HOURS: "2h",
            Timeframe.FOUR_HOURS: "4h",
            Timeframe.SIX_HOURS: "6h",
            Timeframe.TWELVE_HOURS: "12h",
            Timeframe.ONE_DAY: "1d",
            Timeframe.ONE_WEEK: "1w",
            Timeframe.ONE_MONTH: "1M",
        }

        tf_str = mapping.get(timeframe)

        if not tf_str:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        # Check if exchange supports this timeframe
        if hasattr(self.exchange, "timeframes") and tf_str not in self.exchange.timeframes:
            available = list(self.exchange.timeframes.keys())
            raise ValueError(
                f"Timeframe '{tf_str}' not supported by {self.exchange_id}. "
                f"Available: {available}"
            )

        return tf_str

    def _timeframe_to_ms(self, timeframe_str: str) -> int:
        """Convert timeframe string to milliseconds."""
        mapping = {
            "1m": 60 * 1000,
            "5m": 5 * 60 * 1000,
            "15m": 15 * 60 * 1000,
            "30m": 30 * 60 * 1000,
            "1h": 60 * 60 * 1000,
            "2h": 2 * 60 * 60 * 1000,
            "4h": 4 * 60 * 60 * 1000,
            "6h": 6 * 60 * 60 * 1000,
            "12h": 12 * 60 * 60 * 1000,
            "1d": 24 * 60 * 60 * 1000,
            "1w": 7 * 24 * 60 * 60 * 1000,
            "1M": 30 * 24 * 60 * 60 * 1000,
        }
        return mapping.get(timeframe_str, 60 * 60 * 1000)

    def get_supported_symbols(self) -> list[str]:
        """Get list of supported trading symbols."""
        try:
            self.exchange.load_markets()
            return self.exchange.symbols
        except Exception as e:
            logger.error(f"Error loading markets: {e}")
            return []

    def get_exchange_info(self) -> Dict[str, Any]:
        """Get exchange information."""
        info = {
            "exchange_id": self.exchange_id,
            "name": self.exchange.name if hasattr(self.exchange, "name") else self.exchange_id,
            "has_ohlcv": self.exchange.has.get("fetchOHLCV", False),
            "supported_timeframes": list(self.exchange.timeframes.keys()) if hasattr(self.exchange, "timeframes") else [],
            "rate_limit": self.exchange.rateLimit,
        }

        return info

    # ============ IMarketRepository Interface Implementation ============

    def get_candles(
        self,
        symbol: str,
        timeframe: Timeframe,
        start_date,
        end_date,
        market_type=None,
        progress_callback=None
    ):
        """Get candles (required by IMarketRepository)."""
        df = self.fetch_ohlcv(symbol, timeframe, start_date, end_date, progress_callback=progress_callback)
        # Convert DataFrame to list of Candle objects if needed
        return df

    def get_exchange_name(self) -> str:
        """Get exchange name (required by IMarketRepository)."""
        return self.exchange.name if hasattr(self.exchange, "name") else self.exchange_id

    def get_supported_market_types(self):
        """Get supported market types (required by IMarketRepository)."""
        from ...domain.entities import MarketType
        # Most exchanges support at least spot
        return [MarketType.SPOT]

    def get_supported_timeframes(self):
        """Get supported timeframes (required by IMarketRepository)."""
        return list(self.exchange.timeframes.keys()) if hasattr(self.exchange, "timeframes") else []

    def validate_symbol(self, symbol: str, market_type=None) -> bool:
        """Validate if symbol is valid (required by IMarketRepository)."""
        try:
            self.exchange.load_markets()
            return symbol in self.exchange.symbols
        except Exception:
            return False


# Convenience function to list all available exchanges
def list_available_exchanges() -> list[str]:
    """List all exchanges supported by CCXT."""
    return ccxt.exchanges


# Convenience function to get exchange by category
def get_exchanges_by_category() -> Dict[str, list[str]]:
    """Get exchanges grouped by category."""
    return {
        "tier_1": [
            "binance", "coinbase", "kraken", "bitfinex", "bitstamp"
        ],
        "tier_2": [
            "bybit", "okx", "kucoin", "gateio", "huobi", "bitget", "mexc"
        ],
        "futures": [
            "binance", "bybit", "okx", "deribit", "bitmex"
        ],
        "dex": [
            "uniswap", "pancakeswap", "sushiswap"
        ],
        "all": ccxt.exchanges
    }
