"""
Ordenes de Ejecucion Algoritmica.

Implementa algoritmos de ejecucion profesional:
- TWAP: Time-Weighted Average Price
- VWAP: Volume-Weighted Average Price
- Iceberg: Ordenes con cantidad parcial visible
- Hidden: Ordenes completamente ocultas
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import uuid

from .enums import OrderType, OrderSide, OrderStatus, TimeInForce


# Soporte nativo por exchange
TWAP_SUPPORT = {
    "okx": True,
    "binance": False,
    "bybit": False,
    "kraken": False,
}

VWAP_SUPPORT = {
    "okx": True,
    "binance": False,
    "bybit": False,
}

ICEBERG_SUPPORT = {
    "binance": True,
    "bybit": True,
    "okx": True,
    "kraken": True,
    "kucoin": True,
}


@dataclass
class TWAPOrder:
    """
    Orden TWAP (Time-Weighted Average Price).

    Divide una orden grande en partes iguales ejecutadas
    a intervalos regulares de tiempo.

    Attributes:
        symbol: Par de trading
        side: BUY o SELL
        total_quantity: Cantidad total a ejecutar
        duration_seconds: Duracion total de ejecucion
        slice_count: Numero de slices (default: 1 por minuto)
        size_variation: Variacion aleatoria de tamano (0.1 = 10%)
        price_limit: Precio limite opcional

    Example:
        # Comprar 10 BTC en 1 hora
        order = TWAPOrder(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            total_quantity=10.0,
            duration_seconds=3600,
            slice_count=60  # 1 slice por minuto
        )
    """
    symbol: str
    side: OrderSide
    total_quantity: float
    duration_seconds: int
    slice_count: Optional[int] = None
    size_variation: float = 0.0
    price_limit: Optional[float] = None
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    start_time: Optional[datetime] = None
    time_in_force: TimeInForce = TimeInForce.GTC

    # Estado de ejecucion
    filled_quantity: float = 0.0
    average_price: float = 0.0
    slices_executed: int = 0

    def __post_init__(self):
        if self.slice_count is None:
            # Default: 1 slice por minuto
            self.slice_count = max(1, self.duration_seconds // 60)

    def get_slice_size(self) -> float:
        """Calcula tamano de cada slice"""
        return self.total_quantity / self.slice_count

    def get_slice_interval(self) -> float:
        """Calcula intervalo entre slices en segundos"""
        return self.duration_seconds / self.slice_count

    def get_remaining_quantity(self) -> float:
        """Cantidad pendiente de ejecutar"""
        return self.total_quantity - self.filled_quantity

    def get_progress_pct(self) -> float:
        """Porcentaje de progreso"""
        return (self.filled_quantity / self.total_quantity) * 100

    def to_ccxt_params(self, exchange_id: str) -> Dict[str, Any]:
        """Convierte a parametros CCXT para exchanges que soportan TWAP"""
        if exchange_id.lower() == "okx":
            return {
                "instId": self.symbol.replace("/", "-"),
                "tdMode": "cash",
                "side": self.side.value,
                "ordType": "twap",
                "sz": str(self.total_quantity),
                "twapPxVar": "0",  # Sin variacion de precio
                "twapSzVar": str(self.size_variation * 100),
                "twapTotalSz": str(self.total_quantity),
                "twapInterval": str(int(self.get_slice_interval())),
            }
        return {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": OrderType.TWAP.value,
            "symbol": self.symbol,
            "side": self.side.value,
            "total_quantity": self.total_quantity,
            "filled_quantity": self.filled_quantity,
            "average_price": self.average_price,
            "duration_seconds": self.duration_seconds,
            "slice_count": self.slice_count,
            "slices_executed": self.slices_executed,
            "progress_pct": self.get_progress_pct(),
            "status": self.status.value,
        }


@dataclass
class VWAPOrder:
    """
    Orden VWAP (Volume-Weighted Average Price).

    Ejecuta proporcional al volumen del mercado para
    minimizar impacto.

    Attributes:
        symbol: Par de trading
        side: BUY o SELL
        total_quantity: Cantidad total
        duration_seconds: Ventana de ejecucion
        max_participation: Participacion maxima en volumen (0.05 = 5%)
        price_limit: Precio limite opcional
    """
    symbol: str
    side: OrderSide
    total_quantity: float
    duration_seconds: int
    max_participation: float = 0.05
    price_limit: Optional[float] = None
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    time_in_force: TimeInForce = TimeInForce.GTC

    # Estado
    filled_quantity: float = 0.0
    average_price: float = 0.0
    market_vwap: float = 0.0  # VWAP del mercado durante ejecucion

    def get_remaining_quantity(self) -> float:
        return self.total_quantity - self.filled_quantity

    def get_progress_pct(self) -> float:
        return (self.filled_quantity / self.total_quantity) * 100

    def calculate_slice_for_volume(self, market_volume: float) -> float:
        """
        Calcula cantidad a ejecutar basada en volumen de mercado.

        Args:
            market_volume: Volumen actual del mercado

        Returns:
            Cantidad a ejecutar
        """
        max_qty = market_volume * self.max_participation
        remaining = self.get_remaining_quantity()
        return min(max_qty, remaining)

    def to_ccxt_params(self, exchange_id: str) -> Dict[str, Any]:
        if exchange_id.lower() == "okx":
            return {
                "instId": self.symbol.replace("/", "-"),
                "tdMode": "cash",
                "side": self.side.value,
                "ordType": "vwap",
                "sz": str(self.total_quantity),
                "priceLimit": str(self.price_limit) if self.price_limit else None,
            }
        return {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": OrderType.VWAP.value,
            "symbol": self.symbol,
            "side": self.side.value,
            "total_quantity": self.total_quantity,
            "filled_quantity": self.filled_quantity,
            "average_price": self.average_price,
            "market_vwap": self.market_vwap,
            "max_participation": self.max_participation,
            "progress_pct": self.get_progress_pct(),
            "status": self.status.value,
        }


@dataclass
class IcebergOrder:
    """
    Orden Iceberg.

    Muestra solo una parte de la cantidad total en el order book.
    Cuando la parte visible se ejecuta, se recarga automaticamente.

    Attributes:
        symbol: Par de trading
        side: BUY o SELL
        total_quantity: Cantidad total
        display_quantity: Cantidad visible en order book
        price: Precio de la orden
        variance: Variacion de cantidad/precio entre recargas
    """
    symbol: str
    side: OrderSide
    total_quantity: float
    display_quantity: float
    price: float
    variance: float = 0.0  # Variacion porcentual
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    time_in_force: TimeInForce = TimeInForce.GTC

    # Estado
    filled_quantity: float = 0.0
    average_price: float = 0.0
    reloads: int = 0  # Numero de recargas

    def __post_init__(self):
        if self.display_quantity > self.total_quantity:
            raise ValueError("display_quantity no puede ser mayor que total_quantity")

    def get_remaining_quantity(self) -> float:
        return self.total_quantity - self.filled_quantity

    def get_current_display(self) -> float:
        """Cantidad actualmente visible"""
        remaining = self.get_remaining_quantity()
        return min(self.display_quantity, remaining)

    def get_progress_pct(self) -> float:
        return (self.filled_quantity / self.total_quantity) * 100

    def can_execute_on_exchange(self, exchange_id: str) -> bool:
        return ICEBERG_SUPPORT.get(exchange_id.lower(), False)

    def to_ccxt_params(self, exchange_id: str) -> Dict[str, Any]:
        """Convierte a parametros CCXT"""
        exchange = exchange_id.lower()

        base = {
            "symbol": self.symbol,
            "side": self.side.value,
            "type": "limit",
            "amount": self.total_quantity,
            "price": self.price,
        }

        params = {}

        if exchange == "binance":
            params["icebergQty"] = self.display_quantity
        elif exchange == "bybit":
            params["orderFilter"] = "Order"
            # Bybit no tiene iceberg nativo en spot
        elif exchange == "okx":
            base["type"] = "iceberg"
            params["displayQty"] = str(self.display_quantity)
        elif exchange == "kraken":
            params["displayvol"] = self.display_quantity

        base["params"] = params
        return base

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": OrderType.ICEBERG.value,
            "symbol": self.symbol,
            "side": self.side.value,
            "total_quantity": self.total_quantity,
            "display_quantity": self.display_quantity,
            "current_display": self.get_current_display(),
            "price": self.price,
            "filled_quantity": self.filled_quantity,
            "average_price": self.average_price,
            "reloads": self.reloads,
            "progress_pct": self.get_progress_pct(),
            "status": self.status.value,
        }


@dataclass
class HiddenOrder:
    """
    Orden Hidden (Dark Pool Order).

    No se muestra en el order book publico.
    Solo se ejecuta cuando alguien cruza su precio.

    Attributes:
        symbol: Par de trading
        side: BUY o SELL
        quantity: Cantidad
        price: Precio de la orden
    """
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    time_in_force: TimeInForce = TimeInForce.GTC

    # Estado
    filled_quantity: float = 0.0
    filled_price: float = 0.0

    def get_remaining_quantity(self) -> float:
        return self.quantity - self.filled_quantity

    def is_complete(self) -> bool:
        return self.filled_quantity >= self.quantity

    def to_ccxt_params(self, exchange_id: str) -> Dict[str, Any]:
        """
        Convierte a parametros CCXT.

        La mayoria de exchanges no soportan hidden orders en APIs publicas.
        """
        return {
            "symbol": self.symbol,
            "side": self.side.value,
            "type": "limit",
            "amount": self.quantity,
            "price": self.price,
            "params": {
                "hidden": True,  # No todos los exchanges lo respetan
            }
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": OrderType.HIDDEN.value,
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": self.quantity,
            "price": self.price,
            "filled_quantity": self.filled_quantity,
            "filled_price": self.filled_price,
            "status": self.status.value,
        }
