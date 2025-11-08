from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional


class MarketType(Enum):
    """Tipos de mercado disponibles."""
    SPOT = "spot"
    FUTURES = "futures"
    MARGIN = "margin"


class Timeframe(Enum):
    """Temporalidades disponibles para las velas."""
    ONE_MINUTE = "1m"
    FIVE_MINUTES = "5m"
    FIFTEEN_MINUTES = "15m"
    THIRTY_MINUTES = "30m"
    ONE_HOUR = "1h"
    TWO_HOURS = "2h"
    FOUR_HOURS = "4h"
    SIX_HOURS = "6h"
    TWELVE_HOURS = "12h"
    ONE_DAY = "1d"
    THREE_DAYS = "3d"
    ONE_WEEK = "1w"
    ONE_MONTH = "1M"


@dataclass
class MarketConfig:
    """
    Configuracion para la extraccion de datos de mercado.

    Attributes:
        exchange: Nombre del exchange (binance, kraken, etc)
        symbol: Par de trading (ej: BTC/USDT)
        market_type: Tipo de mercado (spot, futures, margin)
        timeframe: Temporalidad de las velas
        start_date: Fecha de inicio para la extraccion
        end_date: Fecha de fin para la extraccion
        output_path: Ruta donde guardar el archivo CSV
    """
    exchange: str
    symbol: str
    market_type: MarketType
    timeframe: Timeframe
    start_date: datetime
    end_date: datetime
    output_path: str

    def validate(self) -> tuple[bool, Optional[str]]:
        """
        Valida la configuracion.

        Returns:
            Tupla (es_valido, mensaje_error)
        """
        if self.start_date >= self.end_date:
            return False, "La fecha de inicio debe ser anterior a la fecha de fin"

        if not self.symbol or not self.symbol.strip():
            return False, "El simbolo no puede estar vacio"

        if not self.output_path or not self.output_path.strip():
            return False, "La ruta de salida no puede estar vacia"

        return True, None
