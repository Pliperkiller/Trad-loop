"""
Estrategia de Mean Reversion con Regresion Lineal - Solo Longs.

Version de la estrategia MeanReversionLinearRegression optimizada para mercados spot
donde solo se pueden ejecutar operaciones long (compra).

Usa regresion lineal para identificar desviaciones del precio respecto a la linea
de mejor ajuste. Opera cuando el precio se desvia significativamente hacia abajo
(z-score negativo) en mercados laterales (R² bajo).
"""

import pandas as pd
import numpy as np
from typing import Optional

from ..base import TradingStrategy, StrategyConfig, TradeSignal
from src.indicators import TechnicalIndicators, atr


class MeanReversionLRLongOnlyStrategy(TradingStrategy):
    """
    Estrategia de Mean Reversion con Regresion Lineal - Solo Longs.

    Disenada para operar en mercados spot donde no se pueden hacer shorts.
    Ajusta una linea de regresion a los precios recientes y compra cuando
    el precio se desvia significativamente hacia abajo.

    Ventajas:
    - Menor lag que medias moviles
    - R² como filtro de regimen (evita operar en tendencias)
    - Pendiente normalizada para confirmar direccion
    - Ideal para spot trading

    Senales:
    - LONG: R² < max_r2 AND |slope_norm| < max_slope AND zscore < -entry_zscore

    Salidas:
    - |zscore| < exit_zscore (precio volvio a la linea)
    - R² > 0.5 (mercado entro en tendencia)
    - Stop loss o take profit
    """

    def __init__(
        self,
        config: StrategyConfig,
        lr_period: int = 20,
        entry_zscore: float = 2.0,
        exit_zscore: float = 0.5,
        max_r2: float = 0.4,
        max_slope_pct: float = 0.3,
        atr_period: int = 14,
        atr_sl_multiplier: float = 0.5,
        tp1_ratio: float = 1.5,
        tp2_ratio: float = 2.5,
        require_candle_confirmation: bool = False,
        regime_stability_lookback: int = 5,
        min_bars_between_trades: int = 3
    ):
        """
        Args:
            config: Configuracion base de la estrategia
            lr_period: Periodos para calcular regresion lineal (default 20)
            entry_zscore: Umbral de z-score para entrar (default 2.0)
            exit_zscore: Umbral de z-score para salir (default 0.5)
            max_r2: Maximo R² permitido para operar (default 0.4)
            max_slope_pct: Maxima pendiente normalizada % permitida (default 0.3)
            atr_period: Periodo para ATR (default 14)
            atr_sl_multiplier: Multiplicador ATR para stop loss (default 0.5)
            tp1_ratio: Ratio riesgo/beneficio para TP1 (default 1.5)
            tp2_ratio: Ratio riesgo/beneficio para TP2 (default 2.5)
            require_candle_confirmation: Requerir vela de confirmacion (default False)
            regime_stability_lookback: Barras para validar estabilidad del regimen lateral (default 5)
            min_bars_between_trades: Minimo de barras entre trades para evitar sobre-operar (default 3)
        """
        super().__init__(config)
        self.lr_period = max(lr_period, 5)
        self.entry_zscore = max(entry_zscore, 0.5)
        self.exit_zscore = max(exit_zscore, 0.0)
        self.max_r2 = max_r2 if max_r2 > 0 else 0.4
        self.max_slope_pct = max_slope_pct if max_slope_pct > 0 else 0.3
        self.atr_period = max(atr_period, 5)
        self.atr_sl_multiplier = max(atr_sl_multiplier, 0.1)
        self.tp1_ratio = max(tp1_ratio, 0.5)
        self.tp2_ratio = max(tp2_ratio, 1.0)
        self.require_candle_confirmation = require_candle_confirmation
        self.regime_stability_lookback = max(regime_stability_lookback, 2)
        self.min_bars_between_trades = max(min_bars_between_trades, 0)
        self._last_trade_bar = -9999

    def calculate_indicators(self):
        """Calcula regresion lineal y ATR."""
        lr_result = TechnicalIndicators.linear_regression(
            self.data['close'],
            self.data['open'],
            self.lr_period
        )
        self.data['lr_slope'] = lr_result.slope
        self.data['lr_slope_norm'] = lr_result.slope_normalized_2 * 100
        self.data['lr_intercept'] = lr_result.intercept
        self.data['lr_r_squared'] = lr_result.r_squared
        self.data['lr_residual'] = lr_result.residual
        self.data['lr_residual_std'] = lr_result.residual_std
        self.data['lr_zscore'] = lr_result.residual_zscore

        self.data['lr_predicted'] = lr_result.intercept + lr_result.slope * (self.lr_period - 1)

        self.data['lr_upper'] = self.data['lr_predicted'] + (2 * lr_result.residual_std)
        self.data['lr_lower'] = self.data['lr_predicted'] - (2 * lr_result.residual_std)

        self.data['atr'] = atr(
            self.data['high'],
            self.data['low'],
            self.data['close'],
            self.atr_period
        )

    def _is_ranging_market(self, i: int) -> bool:
        """Verifica si el mercado esta en rango (lateral)."""
        r2 = self.data['lr_r_squared'].iloc[i]
        slope_norm = self.data['lr_slope_norm'].iloc[i]

        if pd.isna(r2) or pd.isna(slope_norm):
            return False

        return r2 < self.max_r2 and abs(slope_norm) < self.max_slope_pct

    def _is_stable_ranging_market(self, i: int) -> bool:
        """
        Verifica que el mercado haya estado en regimen lateral de forma consistente.
        """
        lookback = self.regime_stability_lookback

        if i < lookback:
            return self._is_ranging_market(i)

        r2_values = self.data['lr_r_squared'].iloc[i - lookback + 1:i + 1]
        slope_values = self.data['lr_slope_norm'].iloc[i - lookback + 1:i + 1].abs()

        if r2_values.isna().any() or slope_values.isna().any():
            return False

        r2_ok = (r2_values < self.max_r2).all()
        slope_ok = (slope_values < self.max_slope_pct).all()

        return r2_ok and slope_ok

    def _is_cooldown_active(self, i: int) -> bool:
        """Verifica si estamos en periodo de cooldown despues de un trade."""
        if self.min_bars_between_trades <= 0:
            return False
        return (i - self._last_trade_bar) < self.min_bars_between_trades

    def _is_bullish_candle(self, i: int) -> bool:
        """Detecta vela alcista."""
        return self.data['close'].iloc[i] > self.data['open'].iloc[i]

    def _find_signal(self, i: int) -> Optional[str]:
        """Busca senal de entrada long con filtros anti-overfitting."""
        if self._is_cooldown_active(i):
            return None

        if not self._is_stable_ranging_market(i):
            return None

        zscore = self.data['lr_zscore'].iloc[i]
        if pd.isna(zscore):
            return None

        # Solo buscamos sobreventa extrema -> LONG
        if zscore < -self.entry_zscore:
            if not self.require_candle_confirmation or self._is_bullish_candle(i):
                return 'BUY'

        return None

    def _should_exit_by_zscore(self, i: int) -> bool:
        """Verifica si el precio volvio a la linea de regresion."""
        zscore = self.data['lr_zscore'].iloc[i]
        if pd.isna(zscore):
            return False
        return abs(zscore) < self.exit_zscore

    def _should_exit_by_regime_change(self, i: int) -> bool:
        """Verifica si el mercado entro en tendencia."""
        r2 = self.data['lr_r_squared'].iloc[i]
        if pd.isna(r2):
            return False
        return r2 > 0.5

    def generate_signals(self) -> pd.Series:
        """Genera senales basadas en z-score de regresion lineal (solo longs)."""
        signals = pd.Series(index=self.data.index, dtype=object)

        for i in range(self.lr_period, len(self.data)):
            signal = self._find_signal(i)
            if signal:
                signals.iloc[i] = signal

        return signals

    def backtest(self):
        """
        Ejecuta el backtest con gestion de riesgo (solo posiciones long).
        Stop loss basado en canal de regresion + ATR.
        Take profits escalonados basados en z-score.
        """
        if self.data is None:
            raise ValueError("Primero debes cargar los datos con load_data()")

        self._last_trade_bar = -9999

        self.calculate_indicators()
        signals = self.generate_signals()

        for i in range(len(self.data)):
            current_bar = self.data.iloc[i]
            current_time = self.data.index[i]
            current_price = current_bar['close']
            current_high = current_bar['high']
            current_low = current_bar['low']

            # Gestionar posiciones abiertas (solo longs)
            for position in self.positions.copy():
                # Check stop loss
                if current_low <= position.stop_loss:
                    self.close_position(position, position.stop_loss, current_time, 'Stop Loss')
                    continue
                if position.take_profit and current_high >= position.take_profit:
                    self.close_position(position, position.take_profit, current_time, 'Take Profit')
                    continue

                # Salida por reversion completada (zscore cerca de 0)
                if self._should_exit_by_zscore(i):
                    self.close_position(position, current_price, current_time, 'Mean Reversion Complete')
                    continue

                # Salida por cambio de regimen (mercado entro en tendencia)
                if self._should_exit_by_regime_change(i):
                    self.close_position(position, current_price, current_time, 'Regime Change')
                    continue

            # Evaluar nuevas senales (solo BUY)
            if pd.notna(signals.iloc[i]) and len(self.positions) < self.config.max_positions:
                signal_type = signals.iloc[i]
                lr_predicted = self.data['lr_predicted'].iloc[i]
                atr_val = self.data['atr'].iloc[i]

                if pd.isna(lr_predicted) or pd.isna(atr_val) or atr_val == 0:
                    continue

                if signal_type == 'BUY':
                    stop_loss = current_price - (atr_val * self.atr_sl_multiplier)
                    risk = atr_val * self.atr_sl_multiplier
                    take_profit = current_price + (risk * self.tp1_ratio)

                    signal = TradeSignal(
                        timestamp=current_time,
                        signal='LONG',
                        price=current_price,
                        confidence=1.0,
                        indicators={
                            'zscore': self.data['lr_zscore'].iloc[i],
                            'r_squared': self.data['lr_r_squared'].iloc[i],
                            'slope_norm': self.data['lr_slope_norm'].iloc[i],
                            'predicted': lr_predicted,
                            'atr': atr_val
                        }
                    )
                    self.open_position(signal, stop_loss, take_profit)
                    self._last_trade_bar = i

            # Actualizar equity curve
            total_equity = self.capital
            for position in self.positions:
                total_equity += position.quantity * current_price
            self.equity_curve.append(total_equity)
