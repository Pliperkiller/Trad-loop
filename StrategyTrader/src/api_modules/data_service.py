"""
Servicio de datos para la API.

Encapsula la lógica de obtención de datos OHLCV y gestión de exchanges.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging

import pandas as pd

logger = logging.getLogger(__name__)

# Intentar importar DataExtractor
try:
    from DataExtractor.src.application.services import DataExtractionService
    from DataExtractor.src.infrastructure.exchanges.ccxt_adapter import (
        CCXTAdapter,
        list_available_exchanges,
        get_exchanges_by_category
    )
    from DataExtractor.src.domain import MarketConfig, MarketType, Timeframe
    DATA_EXTRACTOR_AVAILABLE = True
except ImportError as e:
    logger.warning(f"DataExtractor not available: {e}")
    DATA_EXTRACTOR_AVAILABLE = False


# Mapeos de timeframe
TIMEFRAME_MAP = {
    "1m": "ONE_MINUTE",
    "5m": "FIVE_MINUTES",
    "15m": "FIFTEEN_MINUTES",
    "30m": "THIRTY_MINUTES",
    "1h": "ONE_HOUR",
    "2h": "TWO_HOURS",
    "4h": "FOUR_HOURS",
    "6h": "SIX_HOURS",
    "12h": "TWELVE_HOURS",
    "1d": "ONE_DAY",
    "3d": "THREE_DAYS",
    "1w": "ONE_WEEK",
    "1M": "ONE_MONTH",
}

TIMEFRAME_MINUTES = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "2h": 120,
    "4h": 240,
    "6h": 360,
    "12h": 720,
    "1d": 1440,
    "3d": 4320,
    "1w": 10080,
    "1M": 43200,
}


class DataService:
    """
    Servicio para obtención de datos de mercado.

    Encapsula la lógica de comunicación con exchanges via CCXT.
    """

    def __init__(self):
        self._ccxt_instances: Dict[str, Any] = {}
        self._data_service = None

        if DATA_EXTRACTOR_AVAILABLE:
            self._data_service = DataExtractionService()

    def is_available(self) -> bool:
        """Verifica si el servicio de datos está disponible."""
        return DATA_EXTRACTOR_AVAILABLE

    def get_ccxt_adapter(self, exchange_id: str):
        """
        Obtiene un adaptador CCXT (con cache).

        Args:
            exchange_id: ID del exchange

        Returns:
            CCXTAdapter instance
        """
        if not DATA_EXTRACTOR_AVAILABLE:
            raise RuntimeError("DataExtractor no disponible")

        if exchange_id not in self._ccxt_instances:
            self._ccxt_instances[exchange_id] = CCXTAdapter(exchange_id)

        return self._ccxt_instances[exchange_id]

    def get_exchanges(self) -> Dict[str, Any]:
        """
        Lista exchanges disponibles.

        Returns:
            Dict con exchanges, by_category, configured
        """
        if not DATA_EXTRACTOR_AVAILABLE:
            raise RuntimeError("DataExtractor no disponible")

        return {
            "exchanges": list_available_exchanges(),
            "by_category": get_exchanges_by_category(),
            "configured": self._data_service.get_available_exchanges()
        }

    def get_symbols(self, exchange_id: str) -> Dict[str, Any]:
        """
        Lista símbolos de un exchange.

        Args:
            exchange_id: ID del exchange

        Returns:
            Dict con exchange, count, symbols
        """
        adapter = self.get_ccxt_adapter(exchange_id)
        symbols = adapter.get_supported_symbols()

        return {
            "exchange": exchange_id,
            "count": len(symbols),
            "symbols": symbols
        }

    def get_symbols_catalog(self, exchange_id: str) -> Dict[str, Any]:
        """
        Obtiene catálogo organizado de símbolos.

        Args:
            exchange_id: ID del exchange

        Returns:
            Dict con popular, by_quote, etc.
        """
        adapter = self.get_ccxt_adapter(exchange_id)
        symbols = adapter.get_supported_symbols()

        # Símbolos populares
        popular_bases = [
            'BTC', 'ETH', 'BNB', 'SOL', 'XRP', 'DOGE', 'ADA', 'AVAX',
            'DOT', 'LINK', 'MATIC', 'UNI', 'ATOM', 'LTC', 'ETC', 'XLM'
        ]
        popular_quotes = ['USDT', 'USDC', 'USD', 'BUSD']

        popular = []
        for base in popular_bases:
            for quote in popular_quotes:
                pair = f"{base}/{quote}"
                if pair in symbols:
                    popular.append(pair)
                    break

        # Agrupar por quote
        by_quote: Dict[str, List[str]] = {}
        for symbol in symbols:
            if '/' in symbol:
                parts = symbol.split('/')
                quote = parts[1].split(':')[0]
                if quote not in by_quote:
                    by_quote[quote] = []
                by_quote[quote].append(symbol)

        sorted_quotes = sorted(by_quote.keys(), key=lambda q: len(by_quote[q]), reverse=True)
        by_quote_sorted = {q: sorted(by_quote[q]) for q in sorted_quotes}

        return {
            "exchange": exchange_id,
            "total": len(symbols),
            "popular": popular,
            "by_quote": by_quote_sorted,
            "quote_currencies": sorted_quotes
        }

    def get_exchange_info(self, exchange_id: str) -> Dict[str, Any]:
        """
        Obtiene info de un exchange.

        Args:
            exchange_id: ID del exchange

        Returns:
            Info del exchange
        """
        adapter = self.get_ccxt_adapter(exchange_id)
        return adapter.get_exchange_info()

    def fetch_ohlcv(
        self,
        exchange: str,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        warmup_candles: int = 0
    ) -> Dict[str, Any]:
        """
        Obtiene datos OHLCV.

        Args:
            exchange: ID del exchange
            symbol: Par de trading
            timeframe: Temporalidad
            start: Fecha inicio
            end: Fecha fin
            warmup_candles: Velas adicionales para warmup

        Returns:
            Dict con datos OHLCV
        """
        if timeframe not in TIMEFRAME_MAP:
            raise ValueError(f"Timeframe '{timeframe}' no válido")

        adapter = self.get_ccxt_adapter(exchange)

        # Obtener enum de timeframe
        tf_enum = getattr(Timeframe, TIMEFRAME_MAP[timeframe])

        # Calcular fecha extendida para warmup
        if warmup_candles > 0:
            warmup_minutes = warmup_candles * TIMEFRAME_MINUTES[timeframe]
            extended_start = start - timedelta(minutes=warmup_minutes)
        else:
            extended_start = start

        # Fetch data
        df = adapter.fetch_ohlcv(symbol, tf_enum, extended_start, end)

        # Calcular warmup count
        actual_warmup_count = 0
        if warmup_candles > 0 and 'timestamp' in df.columns and len(df) > 0:
            try:
                start_naive = start.replace(tzinfo=None) if start.tzinfo else start
                timestamps = df['timestamp']
                if hasattr(timestamps.iloc[0], 'tzinfo') and timestamps.iloc[0].tzinfo:
                    timestamps = timestamps.dt.tz_localize(None)
                warmup_mask = timestamps < start_naive
                actual_warmup_count = int(warmup_mask.sum())
            except Exception as e:
                logger.warning(f"Could not calculate warmup count: {e}")

        # Convertir timestamps
        data = df.to_dict(orient='records')
        for row in data:
            if 'timestamp' in row and hasattr(row['timestamp'], 'isoformat'):
                row['timestamp'] = row['timestamp'].isoformat()

        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "exchange": exchange,
            "count": len(data),
            "warmup_count": actual_warmup_count,
            "data": data
        }


# Instancia global
_data_service: Optional[DataService] = None


def get_data_service() -> DataService:
    """Obtiene el servicio de datos global."""
    global _data_service
    if _data_service is None:
        _data_service = DataService()
    return _data_service
