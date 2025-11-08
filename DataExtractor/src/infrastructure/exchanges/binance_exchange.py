from typing import List, Callable, Optional
from datetime import datetime, timedelta
import requests
from .base_exchange import BaseExchange
from ...domain import Candle, MarketType, Timeframe


class BinanceExchange(BaseExchange):
    """
    Implementacion del repositorio para Binance.

    Utiliza la API publica de Binance para obtener datos historicos de velas.
    Soporta mercados SPOT y FUTURES.
    """

    SPOT_BASE_URL = "https://api.binance.com"
    FUTURES_BASE_URL = "https://fapi.binance.com"

    TIMEFRAME_MAPPING = {
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
        Timeframe.THREE_DAYS: "3d",
        Timeframe.ONE_WEEK: "1w",
        Timeframe.ONE_MONTH: "1M"
    }

    def __init__(self):
        super().__init__(rate_limit_delay=0.1)

    def get_exchange_name(self) -> str:
        return "Binance"

    def get_supported_market_types(self) -> List[MarketType]:
        return [MarketType.SPOT, MarketType.FUTURES]

    def _get_base_url(self, market_type: MarketType) -> str:
        """Obtiene la URL base segun el tipo de mercado."""
        if market_type == MarketType.FUTURES:
            return self.FUTURES_BASE_URL
        return self.SPOT_BASE_URL

    def _convert_timeframe(self, timeframe: Timeframe) -> str:
        """Convierte el timeframe interno al formato de Binance."""
        return self.TIMEFRAME_MAPPING.get(timeframe, "1h")

    def validate_symbol(self, symbol: str, market_type: MarketType) -> bool:
        """
        Valida si un simbolo existe en Binance.

        Args:
            symbol: Par de trading
            market_type: Tipo de mercado

        Returns:
            True si el simbolo es valido
        """
        try:
            base_url = self._get_base_url(market_type)
            normalized_symbol = self._normalize_symbol(symbol)

            if market_type == MarketType.FUTURES:
                url = f"{base_url}/fapi/v1/exchangeInfo"
            else:
                url = f"{base_url}/api/v3/exchangeInfo"

            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                return False

            data = response.json()
            symbols = [s['symbol'] for s in data.get('symbols', [])]
            return normalized_symbol in symbols

        except Exception:
            return False

    def get_candles(
        self,
        symbol: str,
        timeframe: Timeframe,
        start_date: datetime,
        end_date: datetime,
        market_type: MarketType = MarketType.SPOT,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> List[Candle]:
        """
        Obtiene velas historicas de Binance.

        La API de Binance tiene un limite de 1000 velas por request,
        por lo que se hacen multiples requests si es necesario.
        """
        base_url = self._get_base_url(market_type)
        normalized_symbol = self._normalize_symbol(symbol)
        interval = self._convert_timeframe(timeframe)

        if market_type == MarketType.FUTURES:
            endpoint = f"{base_url}/fapi/v1/klines"
        else:
            endpoint = f"{base_url}/api/v3/klines"

        all_candles = []
        current_start = start_date
        batch_size = 1000

        self._call_progress_callback(
            progress_callback, 0, 100,
            f"Iniciando extraccion de {normalized_symbol} desde {self.get_exchange_name()}"
        )

        while current_start < end_date:
            self._wait_for_rate_limit()

            params = {
                'symbol': normalized_symbol,
                'interval': interval,
                'startTime': int(current_start.timestamp() * 1000),
                'endTime': int(end_date.timestamp() * 1000),
                'limit': batch_size
            }

            try:
                response = requests.get(endpoint, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()

                if not data:
                    break

                for candle_data in data:
                    candle = Candle(
                        timestamp=datetime.fromtimestamp(candle_data[0] / 1000),
                        open=float(candle_data[1]),
                        high=float(candle_data[2]),
                        low=float(candle_data[3]),
                        close=float(candle_data[4]),
                        volume=float(candle_data[5]),
                        symbol=symbol
                    )
                    all_candles.append(candle)

                if len(data) < batch_size:
                    break

                current_start = datetime.fromtimestamp(data[-1][0] / 1000) + timedelta(milliseconds=1)

                progress = min(100, int((current_start.timestamp() - start_date.timestamp()) /
                                       (end_date.timestamp() - start_date.timestamp()) * 100))
                self._call_progress_callback(
                    progress_callback, progress, 100,
                    f"Descargadas {len(all_candles)} velas..."
                )

            except requests.exceptions.RequestException as e:
                raise Exception(f"Error al obtener datos de Binance: {str(e)}")

        self._call_progress_callback(
            progress_callback, 100, 100,
            f"Extraccion completada: {len(all_candles)} velas obtenidas"
        )

        return all_candles
