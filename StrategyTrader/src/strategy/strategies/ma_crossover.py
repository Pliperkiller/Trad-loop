"""
Estrategia de Cruce de Medias Móviles con filtro RSI.
"""

import pandas as pd
import numpy as np

from ..base import TradingStrategy, StrategyConfig
from src.indicators import TechnicalIndicators


class MovingAverageCrossoverStrategy(TradingStrategy):
    """
    Estrategia de cruce de medias móviles con filtro RSI.

    Señales:
    - BUY: EMA rápida cruza por encima de EMA lenta + RSI en rango
    - SELL: EMA rápida cruza por debajo de EMA lenta o RSI muy alto
    """

    def __init__(
        self,
        config: StrategyConfig,
        fast_period: int = 10,
        slow_period: int = 30,
        rsi_period: int = 14,
        rsi_lower_bound: int = 30,
        rsi_upper_bound: int = 70,
        rsi_sell_threshold: int = 80
    ):
        super().__init__(config)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.rsi_period = rsi_period
        self.rsi_lower_bound = rsi_lower_bound
        self.rsi_upper_bound = rsi_upper_bound
        self.rsi_sell_threshold = rsi_sell_threshold

    def calculate_indicators(self):
        """Calcula EMAs y RSI"""
        self.data['ema_fast'] = TechnicalIndicators.ema(self.data['close'], self.fast_period)
        self.data['ema_slow'] = TechnicalIndicators.ema(self.data['close'], self.slow_period)
        self.data['rsi'] = TechnicalIndicators.rsi(self.data['close'], self.rsi_period)

    def generate_signals(self) -> pd.Series:
        """Genera señales basadas en cruces de EMAs y RSI"""
        signals = pd.Series(index=self.data.index, dtype=object)

        self.data['ema_cross'] = np.where(
            self.data['ema_fast'] > self.data['ema_slow'], 1, -1
        )
        self.data['ema_cross_signal'] = self.data['ema_cross'].diff()

        for i in range(1, len(self.data)):
            if (self.data['ema_cross_signal'].iloc[i] == 2 and
                self.rsi_lower_bound < self.data['rsi'].iloc[i] < self.rsi_upper_bound):
                signals.iloc[i] = 'BUY'

            elif (self.data['ema_cross_signal'].iloc[i] == -2 or
                  self.data['rsi'].iloc[i] > self.rsi_sell_threshold):
                signals.iloc[i] = 'SELL'

        return signals
