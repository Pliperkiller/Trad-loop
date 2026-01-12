"""
Capacidades de exchanges CCXT.

Define que tipos de ordenes y features soporta cada exchange.
"""

from typing import Dict, Any, List

from ...core.enums import BrokerType, AssetClass
from ...core.models import BrokerCapabilities


# Capacidades conocidas por exchange
EXCHANGE_CAPABILITIES: Dict[str, Dict[str, Any]] = {
    # Binance Spot
    "binance": {
        "trailing_stop": True,
        "oco": True,
        "iceberg": True,
        "bracket": False,
        "twap": False,
        "vwap": False,
        "fok": True,
        "ioc": True,
        "reduce_only": False,
        "post_only": False,
        "hedging": False,
        "leverage": False,
    },
    # Binance USDM Futures
    "binanceusdm": {
        "trailing_stop": True,
        "oco": False,
        "iceberg": False,
        "bracket": False,
        "twap": False,
        "vwap": False,
        "fok": True,
        "ioc": True,
        "reduce_only": True,
        "post_only": True,
        "hedging": True,
        "leverage": True,
        "max_leverage": 125,
    },
    # Binance COINM Futures
    "binancecoinm": {
        "trailing_stop": True,
        "oco": False,
        "iceberg": False,
        "bracket": False,
        "twap": False,
        "vwap": False,
        "fok": True,
        "ioc": True,
        "reduce_only": True,
        "post_only": True,
        "hedging": True,
        "leverage": True,
        "max_leverage": 125,
    },
    # Bybit
    "bybit": {
        "trailing_stop": True,
        "oco": False,
        "iceberg": False,
        "bracket": True,
        "twap": False,
        "vwap": False,
        "fok": True,
        "ioc": True,
        "reduce_only": True,
        "post_only": True,
        "hedging": False,
        "leverage": True,
        "max_leverage": 100,
    },
    # OKX
    "okx": {
        "trailing_stop": True,
        "oco": True,
        "iceberg": True,
        "bracket": False,
        "twap": True,
        "vwap": True,
        "fok": True,
        "ioc": True,
        "reduce_only": True,
        "post_only": True,
        "hedging": True,
        "leverage": True,
        "max_leverage": 100,
    },
    # Kraken
    "kraken": {
        "trailing_stop": False,
        "oco": False,
        "iceberg": False,
        "bracket": False,
        "twap": False,
        "vwap": False,
        "fok": True,
        "ioc": True,
        "reduce_only": True,
        "post_only": True,
        "hedging": False,
        "leverage": True,
        "max_leverage": 5,
    },
    # Kraken Futures
    "krakenfutures": {
        "trailing_stop": False,
        "oco": False,
        "iceberg": False,
        "bracket": False,
        "twap": False,
        "vwap": False,
        "fok": True,
        "ioc": True,
        "reduce_only": True,
        "post_only": True,
        "hedging": False,
        "leverage": True,
        "max_leverage": 50,
    },
    # KuCoin
    "kucoin": {
        "trailing_stop": True,
        "oco": False,
        "iceberg": True,
        "bracket": False,
        "twap": False,
        "vwap": False,
        "fok": True,
        "ioc": True,
        "reduce_only": False,
        "post_only": True,
        "hedging": False,
        "leverage": False,
    },
    # KuCoin Futures
    "kucoinfutures": {
        "trailing_stop": True,
        "oco": False,
        "iceberg": False,
        "bracket": False,
        "twap": False,
        "vwap": False,
        "fok": True,
        "ioc": True,
        "reduce_only": True,
        "post_only": True,
        "hedging": False,
        "leverage": True,
        "max_leverage": 100,
    },
    # Gate.io
    "gate": {
        "trailing_stop": False,
        "oco": False,
        "iceberg": True,
        "bracket": False,
        "twap": False,
        "vwap": False,
        "fok": True,
        "ioc": True,
        "reduce_only": False,
        "post_only": True,
        "hedging": False,
        "leverage": False,
    },
    # Bitget
    "bitget": {
        "trailing_stop": True,
        "oco": False,
        "iceberg": False,
        "bracket": False,
        "twap": False,
        "vwap": False,
        "fok": True,
        "ioc": True,
        "reduce_only": True,
        "post_only": True,
        "hedging": True,
        "leverage": True,
        "max_leverage": 125,
    },
    # MEXC
    "mexc": {
        "trailing_stop": False,
        "oco": False,
        "iceberg": False,
        "bracket": False,
        "twap": False,
        "vwap": False,
        "fok": True,
        "ioc": True,
        "reduce_only": True,
        "post_only": True,
        "hedging": False,
        "leverage": True,
        "max_leverage": 200,
    },
    # HTX (Huobi)
    "htx": {
        "trailing_stop": False,
        "oco": False,
        "iceberg": True,
        "bracket": False,
        "twap": False,
        "vwap": False,
        "fok": True,
        "ioc": True,
        "reduce_only": False,
        "post_only": True,
        "hedging": False,
        "leverage": False,
    },
}

# Capacidades por defecto para exchanges no listados
DEFAULT_CAPABILITIES: Dict[str, Any] = {
    "trailing_stop": False,
    "oco": False,
    "iceberg": False,
    "bracket": False,
    "twap": False,
    "vwap": False,
    "fok": True,
    "ioc": True,
    "reduce_only": False,
    "post_only": False,
    "hedging": False,
    "leverage": False,
    "max_leverage": 1,
}


def get_exchange_capabilities(exchange_id: str) -> BrokerCapabilities:
    """
    Obtener capacidades de un exchange.

    Args:
        exchange_id: Identificador del exchange (ej: 'binance', 'bybit')

    Returns:
        BrokerCapabilities con las capacidades del exchange
    """
    # Normalizar ID
    exchange_id = exchange_id.lower()

    # Obtener capacidades o usar defaults
    caps = EXCHANGE_CAPABILITIES.get(exchange_id, DEFAULT_CAPABILITIES)

    return BrokerCapabilities(
        broker_type=BrokerType.CCXT,
        exchange_id=exchange_id,
        # Basicos - siempre soportados
        supports_market=True,
        supports_limit=True,
        supports_stop_loss=True,
        supports_stop_limit=True,
        supports_take_profit=True,
        # Avanzados
        supports_trailing_stop=caps.get("trailing_stop", False),
        supports_bracket=caps.get("bracket", False),
        supports_oco=caps.get("oco", False),
        supports_twap=caps.get("twap", False),
        supports_vwap=caps.get("vwap", False),
        supports_iceberg=caps.get("iceberg", False),
        supports_hidden=False,  # Generalmente no soportado
        # Time in force
        supports_gtc=True,
        supports_fok=caps.get("fok", True),
        supports_ioc=caps.get("ioc", True),
        supports_gtd=False,  # Raro en crypto
        supports_day=False,  # No aplica a crypto
        # Features
        supports_reduce_only=caps.get("reduce_only", False),
        supports_post_only=caps.get("post_only", False),
        supports_hedging=caps.get("hedging", False),
        supports_leverage=caps.get("leverage", False),
        supports_margin=caps.get("leverage", False),
        # Asset classes
        asset_classes=[AssetClass.CRYPTO],
        # Limites
        max_leverage=float(caps.get("max_leverage", 1)),
    )


def get_supported_exchanges() -> List[str]:
    """
    Obtener lista de exchanges con capacidades conocidas.

    Returns:
        Lista de IDs de exchanges soportados
    """
    return list(EXCHANGE_CAPABILITIES.keys())


def is_futures_exchange(exchange_id: str) -> bool:
    """
    Verificar si un exchange es de futuros.

    Args:
        exchange_id: Identificador del exchange

    Returns:
        True si es un exchange de futuros
    """
    futures_keywords = ["usdm", "coinm", "futures", "perp", "swap"]
    exchange_lower = exchange_id.lower()
    return any(kw in exchange_lower for kw in futures_keywords)
