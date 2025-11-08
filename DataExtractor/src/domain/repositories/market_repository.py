from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Callable, Optional
from ..entities import Candle, MarketType, Timeframe


class IMarketRepository(ABC):
    """
    Interfaz que define el contrato para acceder a datos de mercado.

    Todas las implementaciones de exchanges deben cumplir con esta interfaz
    siguiendo el principio de inversion de dependencias de Clean Architecture.
    """

    @abstractmethod
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
        Obtiene velas historicas del mercado.

        Args:
            symbol: Par de trading (ej: BTC/USDT)
            timeframe: Temporalidad de las velas
            start_date: Fecha de inicio
            end_date: Fecha de fin
            market_type: Tipo de mercado (spot, futures, margin)
            progress_callback: Callback opcional para reportar progreso.
                              Recibe (current, total, message)

        Returns:
            Lista de objetos Candle

        Raises:
            Exception: Si ocurre un error al obtener los datos
        """
        pass

    @abstractmethod
    def get_exchange_name(self) -> str:
        """
        Retorna el nombre del exchange.

        Returns:
            Nombre del exchange (ej: 'Binance', 'Kraken')
        """
        pass

    @abstractmethod
    def get_supported_market_types(self) -> List[MarketType]:
        """
        Retorna los tipos de mercado soportados por el exchange.

        Returns:
            Lista de tipos de mercado soportados
        """
        pass

    @abstractmethod
    def get_supported_timeframes(self) -> List[Timeframe]:
        """
        Retorna las temporalidades soportadas por el exchange.

        Returns:
            Lista de temporalidades soportadas
        """
        pass

    @abstractmethod
    def validate_symbol(self, symbol: str, market_type: MarketType) -> bool:
        """
        Valida si un simbolo es valido para el exchange.

        Args:
            symbol: Par de trading a validar
            market_type: Tipo de mercado

        Returns:
            True si el simbolo es valido, False en caso contrario
        """
        pass
