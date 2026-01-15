"""
Módulo de Estrategias de Trading para Trad-loop.

Estructura modular:
- base.py: Clases base (TradingStrategy, dataclasses)
- strategies/: Estrategias individuales

Indicadores técnicos: Disponibles desde src.indicators

Para agregar una nueva estrategia:
1. Crear archivo en strategies/ (ej: mi_estrategia.py)
2. Heredar de TradingStrategy
3. Implementar calculate_indicators() y generate_signals()
4. Importar aquí y agregar a __all__
5. Registrar en backtest_api.py

Ejemplo:
    from src.strategy import TradingStrategy, StrategyConfig, TechnicalIndicators

    class MiEstrategia(TradingStrategy):
        def __init__(self, config: StrategyConfig, param1: int = 10):
            super().__init__(config)
            self.param1 = param1

        def calculate_indicators(self):
            self.data['ema'] = TechnicalIndicators.ema(self.data['close'], self.param1)

        def generate_signals(self):
            # ... lógica de señales
            pass
"""

# Clases base
from .base import (
    TradeSignal,
    Position,
    StrategyConfig,
    TradingStrategy,
)

# Indicadores técnicos (desde módulo unificado)
from src.indicators import TechnicalIndicators

# Estrategias disponibles
from .strategies import (
    MovingAverageCrossoverStrategy,
    TrendFollowingEMAStrategy,
    MeanReversionLinearRegressionStrategy,
    MeanReversionLRLongOnlyStrategy,
)

__all__ = [
    # Base
    'TradeSignal',
    'Position',
    'StrategyConfig',
    'TradingStrategy',
    # Indicadores
    'TechnicalIndicators',
    # Estrategias
    'MovingAverageCrossoverStrategy',
    'TrendFollowingEMAStrategy',
    'MeanReversionLinearRegressionStrategy',
    'MeanReversionLRLongOnlyStrategy',
]
