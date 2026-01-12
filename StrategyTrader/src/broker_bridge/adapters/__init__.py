"""
Adapters module for broker_bridge.

Contiene implementaciones especificas para cada broker.
"""

from .ccxt import CCXTBroker
from .ibkr import IBKRBroker

__all__ = [
    "CCXTBroker",
    "IBKRBroker",
]
