from typing import Dict, List, Callable, Optional
from ...domain import IMarketRepository, MarketConfig, MarketType, Timeframe
from ...infrastructure import BinanceExchange, KrakenExchange
from ..use_cases import ExtractMarketDataUseCase


class DataExtractionService:
    """
    Servicio de aplicacion que coordina la extraccion de datos de mercado.

    Este servicio actua como fachada para la aplicacion, proporcionando
    una interfaz simple para acceder a todas las funcionalidades.
    """

    def __init__(self):
        """Inicializa el servicio con todos los exchanges disponibles."""
        self._exchanges: Dict[str, IMarketRepository] = {
            'Binance': BinanceExchange(),
            'Kraken': KrakenExchange()
        }

    def get_available_exchanges(self) -> List[str]:
        """
        Obtiene la lista de exchanges disponibles.

        Returns:
            Lista con nombres de exchanges
        """
        return list(self._exchanges.keys())

    def get_supported_market_types(self, exchange_name: str) -> List[MarketType]:
        """
        Obtiene los tipos de mercado soportados por un exchange.

        Args:
            exchange_name: Nombre del exchange

        Returns:
            Lista de tipos de mercado soportados
        """
        exchange = self._exchanges.get(exchange_name)
        if not exchange:
            return []
        return exchange.get_supported_market_types()

    def get_supported_timeframes(self, exchange_name: str) -> List[Timeframe]:
        """
        Obtiene las temporalidades soportadas por un exchange.

        Args:
            exchange_name: Nombre del exchange

        Returns:
            Lista de temporalidades soportadas
        """
        exchange = self._exchanges.get(exchange_name)
        if not exchange:
            return list(Timeframe)
        return exchange.get_supported_timeframes()

    def extract_market_data(
        self,
        config: MarketConfig,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> tuple[bool, str]:
        """
        Extrae datos de mercado segun la configuracion proporcionada.

        Args:
            config: Configuracion de la extraccion
            progress_callback: Callback opcional para reportar progreso

        Returns:
            Tupla (exito, mensaje)
        """
        exchange = self._exchanges.get(config.exchange)
        if not exchange:
            return False, f"Exchange '{config.exchange}' no disponible"

        if config.market_type not in exchange.get_supported_market_types():
            return False, f"El exchange {config.exchange} no soporta el tipo de mercado {config.market_type.value}"

        if config.timeframe not in exchange.get_supported_timeframes():
            return False, f"El exchange {config.exchange} no soporta la temporalidad {config.timeframe.value}"

        use_case = ExtractMarketDataUseCase(exchange)
        return use_case.execute(config, progress_callback)

    def validate_symbol(self, exchange_name: str, symbol: str, market_type: MarketType) -> bool:
        """
        Valida si un simbolo es valido para un exchange.

        Args:
            exchange_name: Nombre del exchange
            symbol: Simbolo a validar
            market_type: Tipo de mercado

        Returns:
            True si el simbolo es valido
        """
        exchange = self._exchanges.get(exchange_name)
        if not exchange:
            return False
        return exchange.validate_symbol(symbol, market_type)
