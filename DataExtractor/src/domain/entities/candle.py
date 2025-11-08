from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class Candle:
    """
    Representa una vela (candlestick) en los mercados financieros.

    Attributes:
        timestamp: Fecha y hora de apertura de la vela
        open: Precio de apertura
        high: Precio maximo alcanzado
        low: Precio minimo alcanzado
        close: Precio de cierre
        volume: Volumen negociado
        symbol: Simbolo del par de trading (ej: BTC/USDT)
    """
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str

    def to_dict(self) -> dict:
        """Convierte la vela a un diccionario para exportacion."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'symbol': self.symbol
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Candle':
        """Crea una vela desde un diccionario."""
        return cls(
            timestamp=datetime.fromisoformat(data['timestamp']) if isinstance(data['timestamp'], str) else data['timestamp'],
            open=float(data['open']),
            high=float(data['high']),
            low=float(data['low']),
            close=float(data['close']),
            volume=float(data['volume']),
            symbol=data['symbol']
        )
