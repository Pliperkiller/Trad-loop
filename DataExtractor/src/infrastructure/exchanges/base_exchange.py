from abc import ABC
from typing import List, Callable, Optional
from datetime import datetime
import time
from ...domain import IMarketRepository, Candle, MarketType, Timeframe


class BaseExchange(IMarketRepository, ABC):
    """
    Clase base abstracta para implementaciones de exchanges.

    Proporciona funcionalidad comun para todos los exchanges como
    manejo de rate limiting, reintentos y logging.
    """

    def __init__(self, rate_limit_delay: float = 0.1):
        """
        Inicializa el exchange base.

        Args:
            rate_limit_delay: Delay en segundos entre requests para respetar rate limits
        """
        self.rate_limit_delay = rate_limit_delay
        self._last_request_time = 0

    def _wait_for_rate_limit(self):
        """Espera el tiempo necesario para respetar el rate limit."""
        current_time = time.time()
        time_since_last_request = current_time - self._last_request_time
        if time_since_last_request < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last_request)
        self._last_request_time = time.time()

    def _call_progress_callback(
        self,
        callback: Optional[Callable[[int, int, str], None]],
        current: int,
        total: int,
        message: str
    ):
        """
        Llama al callback de progreso de forma segura.

        Args:
            callback: Funcion callback a llamar
            current: Valor actual del progreso
            total: Valor total del progreso
            message: Mensaje descriptivo
        """
        if callback:
            try:
                callback(current, total, message)
            except Exception as e:
                print(f"Error en callback de progreso: {e}")

    def _normalize_symbol(self, symbol: str) -> str:
        """
        Normaliza el simbolo del par de trading.

        Args:
            symbol: Simbolo original (ej: BTC/USDT, BTCUSDT)

        Returns:
            Simbolo normalizado
        """
        return symbol.upper().replace('/', '').replace('-', '').replace('_', '')

    def get_supported_market_types(self) -> List[MarketType]:
        """
        Implementacion por defecto: solo soporta SPOT.
        Los exchanges que soporten mas tipos deben sobreescribir este metodo.
        """
        return [MarketType.SPOT]

    def get_supported_timeframes(self) -> List[Timeframe]:
        """
        Implementacion por defecto: soporta todas las temporalidades.
        Los exchanges pueden sobreescribir para restringir.
        """
        return list(Timeframe)
