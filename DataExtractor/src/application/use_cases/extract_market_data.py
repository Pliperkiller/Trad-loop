from typing import Callable, Optional
from ...domain import IMarketRepository, MarketConfig, Candle
from ...infrastructure import CSVExporter


class ExtractMarketDataUseCase:
    """
    Caso de uso para extraer datos historicos de mercado y exportarlos a CSV.

    Este caso de uso orquesta la extraccion de datos del exchange correspondiente
    y su exportacion a formato CSV, siguiendo los principios de Clean Architecture.
    """

    def __init__(self, repository: IMarketRepository):
        """
        Inicializa el caso de uso.

        Args:
            repository: Implementacion del repositorio de mercado
        """
        self.repository = repository
        self.csv_exporter = CSVExporter()

    def execute(
        self,
        config: MarketConfig,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> tuple[bool, str]:
        """
        Ejecuta la extraccion de datos de mercado.

        Args:
            config: Configuracion de la extraccion
            progress_callback: Callback opcional para reportar progreso

        Returns:
            Tupla (exito, mensaje)
        """
        try:
            is_valid, error_message = config.validate()
            if not is_valid:
                return False, f"Configuracion invalida: {error_message}"

            if progress_callback:
                progress_callback(0, 100, "Validando simbolo...")

            if not self.repository.validate_symbol(config.symbol, config.market_type):
                return False, f"El simbolo '{config.symbol}' no es valido para {self.repository.get_exchange_name()}"

            if progress_callback:
                progress_callback(5, 100, "Obteniendo datos del exchange...")

            candles = self.repository.get_candles(
                symbol=config.symbol,
                timeframe=config.timeframe,
                start_date=config.start_date,
                end_date=config.end_date,
                market_type=config.market_type,
                progress_callback=self._create_sub_progress_callback(progress_callback, 5, 85)
            )

            if not candles:
                return False, "No se obtuvieron datos del exchange. Verifica las fechas y el simbolo."

            if progress_callback:
                progress_callback(90, 100, f"Exportando {len(candles)} velas a CSV...")

            self.csv_exporter.export_candles(candles, config.output_path)

            if progress_callback:
                progress_callback(100, 100, f"Completado: {len(candles)} velas exportadas exitosamente")

            return True, f"Extraccion completada exitosamente. {len(candles)} velas guardadas en {config.output_path}"

        except Exception as e:
            error_msg = f"Error durante la extraccion: {str(e)}"
            if progress_callback:
                progress_callback(0, 100, error_msg)
            return False, error_msg

    def _create_sub_progress_callback(
        self,
        parent_callback: Optional[Callable[[int, int, str], None]],
        start_percent: int,
        end_percent: int
    ) -> Optional[Callable[[int, int, str], None]]:
        """
        Crea un callback de progreso que mapea un rango parcial al callback padre.

        Args:
            parent_callback: Callback padre
            start_percent: Porcentaje de inicio en el callback padre
            end_percent: Porcentaje de fin en el callback padre

        Returns:
            Nuevo callback que mapea el progreso al rango especificado
        """
        if not parent_callback:
            return None

        def sub_callback(current: int, total: int, message: str):
            if total > 0:
                normalized_progress = (current / total) * (end_percent - start_percent) + start_percent
                parent_callback(int(normalized_progress), 100, message)

        return sub_callback
