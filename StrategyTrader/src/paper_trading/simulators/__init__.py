"""
Simuladores especializados para ordenes avanzadas.

Este modulo contiene la logica de simulacion para diferentes tipos de ordenes:
- trailing_simulator: Simulacion de trailing stops
- algo_simulator: Simulacion de ordenes algoritmicas (TWAP, VWAP, Iceberg)
- composite_simulator: Simulacion de ordenes compuestas (Bracket, OCO, OTOCO)
"""

from .trailing_simulator import TrailingStopSimulator
from .algo_simulator import AlgoOrderSimulator
from .composite_simulator import CompositeOrderSimulator

__all__ = [
    "TrailingStopSimulator",
    "AlgoOrderSimulator",
    "CompositeOrderSimulator",
]
