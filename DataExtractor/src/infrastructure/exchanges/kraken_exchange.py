from typing import List, Callable, Optional
from datetime import datetime, timedelta
import requests
from .base_exchange import BaseExchange
from ...domain import Candle, MarketType, Timeframe


class KrakenExchange(BaseExchange):
    """
    Implementacion del repositorio para Kraken.

    Utiliza la API publica de Kraken para obtener datos historicos de velas.
    Solo soporta mercado SPOT.
    """

    BASE_URL = "https://api.kraken.com"

    TIMEFRAME_MAPPING = {
        Timeframe.ONE_MINUTE: 1,
        Timeframe.FIVE_MINUTES: 5,
        Timeframe.FIFTEEN_MINUTES: 15,
        Timeframe.THIRTY_MINUTES: 30,
        Timeframe.ONE_HOUR: 60,
        Timeframe.FOUR_HOURS: 240,
        Timeframe.ONE_DAY: 1440,
        Timeframe.ONE_WEEK: 10080,
        Timeframe.FIFTEEN_MINUTES: 21600
    }

    def __init__(self):
        super().__init__(rate_limit_delay=0.2)

    def get_exchange_name(self) -> str:
        return "Kraken"

    def get_supported_market_types(self) -> List[MarketType]:
        return [MarketType.SPOT]

    def get_supported_timeframes(self) -> List[Timeframe]:
        """Kraken solo soporta ciertas temporalidades."""
        return [
            Timeframe.ONE_MINUTE,
            Timeframe.FIVE_MINUTES,
            Timeframe.FIFTEEN_MINUTES,
            Timeframe.THIRTY_MINUTES,
            Timeframe.ONE_HOUR,
            Timeframe.FOUR_HOURS,
            Timeframe.ONE_DAY,
            Timeframe.ONE_WEEK
        ]

    def _convert_timeframe(self, timeframe: Timeframe) -> int:
        """Convierte el timeframe interno al formato de Kraken (minutos)."""
        return self.TIMEFRAME_MAPPING.get(timeframe, 60)

    def _normalize_symbol_for_kraken(self, symbol: str) -> str:
        """
        Normaliza el simbolo para Kraken.

        Kraken usa formato como XXBTZUSD en lugar de BTC/USD.
        """
        symbol = symbol.upper().replace('/', '').replace('-', '').replace('_', '')

        replacements = {
            'BTC': 'XBT',
            'USDT': 'USDT',
            'USD': 'USD',
            'EUR': 'EUR'
        }

        for old, new in replacements.items():
            if symbol.startswith(old):
                base = new
                quote = symbol[len(old):]
                if quote in replacements:
                    quote = replacements[quote]
                return f"{base}{quote}"

        return symbol

    def validate_symbol(self, symbol: str, market_type: MarketType) -> bool:
        """
        Valida si un simbolo existe en Kraken.
        """
        try:
            url = f"{self.BASE_URL}/0/public/AssetPairs"
            response = requests.get(url, timeout=10)

            if response.status_code != 200:
                return False

            data = response.json()
            if data.get('error'):
                return False

            normalized = self._normalize_symbol_for_kraken(symbol)
            pairs = data.get('result', {})

            for pair_name in pairs.keys():
                if normalized.lower() in pair_name.lower():
                    return True

            return False

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
        Obtiene velas historicas de Kraken.

        Kraken limita a 720 velas por request.
        """
        normalized_symbol = self._normalize_symbol_for_kraken(symbol)
        interval = self._convert_timeframe(timeframe)
        endpoint = f"{self.BASE_URL}/0/public/OHLC"

        all_candles = []
        current_start = start_date

        self._call_progress_callback(
            progress_callback, 0, 100,
            f"Iniciando extraccion de {normalized_symbol} desde {self.get_exchange_name()}"
        )

        while current_start < end_date:
            self._wait_for_rate_limit()

            params = {
                'pair': normalized_symbol,
                'interval': interval,
                'since': int(current_start.timestamp())
            }

            try:
                response = requests.get(endpoint, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()

                if data.get('error'):
                    raise Exception(f"Error de Kraken API: {data['error']}")

                result = data.get('result', {})
                pair_data = None
                for key in result.keys():
                    if key != 'last':
                        pair_data = result[key]
                        break

                if not pair_data:
                    break

                for candle_data in pair_data:
                    candle_time = datetime.fromtimestamp(candle_data[0])
                    if candle_time > end_date:
                        break

                    candle = Candle(
                        timestamp=candle_time,
                        open=float(candle_data[1]),
                        high=float(candle_data[2]),
                        low=float(candle_data[3]),
                        close=float(candle_data[4]),
                        volume=float(candle_data[6]),
                        symbol=symbol
                    )
                    all_candles.append(candle)

                if len(pair_data) < 720:
                    break

                current_start = datetime.fromtimestamp(pair_data[-1][0]) + timedelta(minutes=interval)

                if current_start >= end_date:
                    break

                progress = min(100, int((current_start.timestamp() - start_date.timestamp()) /
                                       (end_date.timestamp() - start_date.timestamp()) * 100))
                self._call_progress_callback(
                    progress_callback, progress, 100,
                    f"Descargadas {len(all_candles)} velas..."
                )

            except requests.exceptions.RequestException as e:
                raise Exception(f"Error al obtener datos de Kraken: {str(e)}")

        all_candles = [c for c in all_candles if start_date <= c.timestamp <= end_date]

        self._call_progress_callback(
            progress_callback, 100, 100,
            f"Extraccion completada: {len(all_candles)} velas obtenidas"
        )

        return all_candles
