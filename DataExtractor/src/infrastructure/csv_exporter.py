import csv
from typing import List
from datetime import datetime
from ..domain import Candle


class CSVExporter:
    """
    Clase responsable de exportar datos de velas a formato CSV.
    """

    @staticmethod
    def export_candles(candles: List[Candle], output_path: str) -> None:
        """
        Exporta una lista de velas a un archivo CSV.

        Args:
            candles: Lista de velas a exportar
            output_path: Ruta donde guardar el archivo CSV

        Raises:
            Exception: Si ocurre un error al escribir el archivo
        """
        if not candles:
            raise ValueError("No hay velas para exportar")

        try:
            with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'symbol']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                writer.writeheader()
                for candle in candles:
                    writer.writerow(candle.to_dict())

        except IOError as e:
            raise Exception(f"Error al escribir archivo CSV: {str(e)}")

    @staticmethod
    def get_default_filename(symbol: str, timeframe: str, start_date: datetime, end_date: datetime) -> str:
        """
        Genera un nombre de archivo por defecto.

        Args:
            symbol: Simbolo del par
            timeframe: Temporalidad
            start_date: Fecha de inicio
            end_date: Fecha de fin

        Returns:
            Nombre de archivo sugerido
        """
        symbol_clean = symbol.replace('/', '_').replace('\\', '_')
        start_str = start_date.strftime('%Y%m%d')
        end_str = end_date.strftime('%Y%m%d')
        return f"{symbol_clean}_{timeframe}_{start_str}_{end_str}.csv"
