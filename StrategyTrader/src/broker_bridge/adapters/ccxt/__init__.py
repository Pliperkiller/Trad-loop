"""
CCXT adapter for broker_bridge.

Implementacion del adaptador para exchanges de criptomonedas via CCXT.
"""

from .ccxt_broker import CCXTBroker
from .ccxt_capabilities import get_exchange_capabilities, EXCHANGE_CAPABILITIES
from .ccxt_order_mapper import CCXTOrderMapper

__all__ = [
    "CCXTBroker",
    "get_exchange_capabilities",
    "EXCHANGE_CAPABILITIES",
    "CCXTOrderMapper",
]
