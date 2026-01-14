"""
Estrategias de trading disponibles.

Para agregar una nueva estrategia:
1. Crear un archivo .py en esta carpeta
2. Heredar de TradingStrategy
3. Implementar calculate_indicators() y generate_signals()
4. Importar la estrategia en este __init__.py
5. Registrarla en backtest_api.py
"""

from .ma_crossover import MovingAverageCrossoverStrategy
from .trend_following_ema import TrendFollowingEMAStrategy
from .mean_reversion_lr import MeanReversionLinearRegressionStrategy

__all__ = [
    'MovingAverageCrossoverStrategy',
    'TrendFollowingEMAStrategy',
    'MeanReversionLinearRegressionStrategy',
]
