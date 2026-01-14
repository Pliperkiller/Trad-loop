"""
Estrategia de Trend Following con Triple EMA y filtro ADX.
"""

import pandas as pd
from typing import Optional

from ..base import TradingStrategy, StrategyConfig, TradeSignal
from src.indicators import TechnicalIndicators


class TrendFollowingEMAStrategy(TradingStrategy):
    """
    Estrategia de Trend Following con Triple EMA y filtro ADX.

    Basada en el principio de "la tendencia es tu amiga", identifica la direccion
    del mercado usando tres EMAs y opera a favor de ella.

    Senales:
    - BULLISH: EMA21 > EMA55 > EMA200
    - BEARISH: EMA21 < EMA55 < EMA200

    Tipos de entrada:
    - Crossover: Cruce de EMA rapida sobre EMA media
    - Pullback: Precio retrocede a EMA21 o EMA55 con confirmacion

    Filtros:
    - ADX > threshold (fuerza de tendencia)
    - Distancia precio-EMA < 5% (no sobre-extendido)
    """

    def __init__(
        self,
        config: StrategyConfig,
        ema_fast: int = 21,
        ema_medium: int = 55,
        ema_slow: int = 200,
        adx_period: int = 14,
        adx_threshold: int = 25,
        atr_period: int = 14,
        entry_type: str = 'pullback',  # 'pullback' o 'crossover'
        max_distance_pct: float = 5.0,
        atr_sl_multiplier: float = 0.5,
        tp1_ratio: float = 1.5,
        tp2_ratio: float = 2.5,
        use_trailing: bool = True
    ):
        super().__init__(config)
        self.ema_fast = ema_fast
        self.ema_medium = ema_medium
        self.ema_slow = ema_slow
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.atr_period = atr_period
        self.entry_type = entry_type
        self.max_distance_pct = max_distance_pct
        self.atr_sl_multiplier = atr_sl_multiplier
        self.tp1_ratio = tp1_ratio
        self.tp2_ratio = tp2_ratio
        self.use_trailing = use_trailing

    def calculate_indicators(self):
        """Calcula EMAs, ADX y ATR"""
        self.data['ema_fast'] = TechnicalIndicators.ema(self.data['close'], self.ema_fast)
        self.data['ema_medium'] = TechnicalIndicators.ema(self.data['close'], self.ema_medium)
        self.data['ema_slow'] = TechnicalIndicators.ema(self.data['close'], self.ema_slow)

        # ADX
        adx_result = TechnicalIndicators.adx(
            self.data['high'],
            self.data['low'],
            self.data['close'],
            self.adx_period
        )
        self.data['adx'] = adx_result.adx
        self.data['plus_di'] = adx_result.plus_di
        self.data['minus_di'] = adx_result.minus_di

        # ATR
        self.data['atr'] = TechnicalIndicators.atr(
            self.data['high'],
            self.data['low'],
            self.data['close'],
            self.atr_period
        )

    def _identify_trend(self, i: int) -> str:
        """Identifica la tendencia basada en la estructura de EMAs"""
        ema_f = self.data['ema_fast'].iloc[i]
        ema_m = self.data['ema_medium'].iloc[i]
        ema_s = self.data['ema_slow'].iloc[i]

        if pd.isna(ema_f) or pd.isna(ema_m) or pd.isna(ema_s):
            return 'NONE'

        if ema_f > ema_m > ema_s:
            return 'BULLISH'
        elif ema_f < ema_m < ema_s:
            return 'BEARISH'
        else:
            return 'NONE'

    def _is_trend_valid(self, i: int) -> bool:
        """Verifica si la tendencia es valida usando ADX y distancia"""
        adx_val = self.data['adx'].iloc[i]
        if pd.isna(adx_val) or adx_val < self.adx_threshold:
            return False

        # Verificar que no esta sobre-extendido
        price = self.data['close'].iloc[i]
        ema_f = self.data['ema_fast'].iloc[i]
        if pd.isna(ema_f) or ema_f == 0:
            return False

        distance_pct = abs(price - ema_f) / ema_f * 100
        if distance_pct > self.max_distance_pct:
            return False

        return True

    def _find_crossover_signal(self, i: int, trend: str) -> Optional[str]:
        """Detecta senal de cruce de EMAs"""
        if i < 1:
            return None

        ema_f_prev = self.data['ema_fast'].iloc[i - 1]
        ema_m_prev = self.data['ema_medium'].iloc[i - 1]
        ema_f_curr = self.data['ema_fast'].iloc[i]
        ema_m_curr = self.data['ema_medium'].iloc[i]

        if pd.isna(ema_f_prev) or pd.isna(ema_m_prev):
            return None

        # Golden Cross
        if ema_f_prev <= ema_m_prev and ema_f_curr > ema_m_curr:
            if trend == 'BULLISH':
                return 'BUY'

        # Death Cross
        if ema_f_prev >= ema_m_prev and ema_f_curr < ema_m_curr:
            if trend == 'BEARISH':
                return 'SELL'

        return None

    def _find_pullback_signal(self, i: int, trend: str) -> Optional[str]:
        """Detecta senal de pullback a EMA con confirmacion"""
        candle = self.data.iloc[i]
        ema_f = self.data['ema_fast'].iloc[i]
        ema_m = self.data['ema_medium'].iloc[i]

        if pd.isna(ema_f) or pd.isna(ema_m):
            return None

        if trend == 'BULLISH':
            # Precio toco EMA21 o EMA55 (low <= EMA)
            touched_fast = candle['low'] <= ema_f <= candle['high']
            touched_medium = candle['low'] <= ema_m <= candle['high']

            # Vela alcista de confirmacion
            bullish_candle = candle['close'] > candle['open']

            if (touched_fast or touched_medium) and bullish_candle:
                return 'BUY'

        elif trend == 'BEARISH':
            # Precio toco EMA21 o EMA55 (high >= EMA)
            touched_fast = candle['low'] <= ema_f <= candle['high']
            touched_medium = candle['low'] <= ema_m <= candle['high']

            # Vela bajista de confirmacion
            bearish_candle = candle['close'] < candle['open']

            if (touched_fast or touched_medium) and bearish_candle:
                return 'SELL'

        return None

    def generate_signals(self) -> pd.Series:
        """Genera senales basadas en estructura de EMAs, ADX y tipo de entrada"""
        signals = pd.Series(index=self.data.index, dtype=object)

        for i in range(1, len(self.data)):
            trend = self._identify_trend(i)

            if trend == 'NONE':
                continue

            if not self._is_trend_valid(i):
                continue

            # Buscar senal segun tipo de entrada
            if self.entry_type == 'crossover':
                signal = self._find_crossover_signal(i, trend)
            else:  # pullback
                signal = self._find_pullback_signal(i, trend)

            if signal:
                signals.iloc[i] = signal

        return signals

    def backtest(self):
        """
        Ejecuta el backtest con gestion de riesgo personalizada.
        Stop loss basado en EMAs + ATR, take profit con ratio R.
        """
        if self.data is None:
            raise ValueError("Primero debes cargar los datos con load_data()")

        self.calculate_indicators()
        signals = self.generate_signals()

        for i in range(len(self.data)):
            current_bar = self.data.iloc[i]
            current_time = self.data.index[i]
            current_price = current_bar['close']
            current_high = current_bar['high']
            current_low = current_bar['low']

            # Gestionar posiciones abiertas
            for position in self.positions.copy():
                # Check stop loss
                if position.position_type == 'LONG':
                    if current_low <= position.stop_loss:
                        self.close_position(position, position.stop_loss, current_time, 'Stop Loss')
                        continue
                    if position.take_profit and current_high >= position.take_profit:
                        self.close_position(position, position.take_profit, current_time, 'Take Profit')
                        continue
                    # Trailing stop basado en EMA21
                    if self.use_trailing:
                        ema_f = self.data['ema_fast'].iloc[i]
                        atr = self.data['atr'].iloc[i]
                        if not pd.isna(ema_f) and not pd.isna(atr):
                            new_stop = ema_f - (atr * self.atr_sl_multiplier)
                            if new_stop > position.stop_loss:
                                position.stop_loss = new_stop

                else:  # SHORT
                    if current_high >= position.stop_loss:
                        self.close_position(position, position.stop_loss, current_time, 'Stop Loss')
                        continue
                    if position.take_profit and current_low <= position.take_profit:
                        self.close_position(position, position.take_profit, current_time, 'Take Profit')
                        continue
                    # Trailing stop para short
                    if self.use_trailing:
                        ema_f = self.data['ema_fast'].iloc[i]
                        atr = self.data['atr'].iloc[i]
                        if not pd.isna(ema_f) and not pd.isna(atr):
                            new_stop = ema_f + (atr * self.atr_sl_multiplier)
                            if new_stop < position.stop_loss:
                                position.stop_loss = new_stop

                # Detectar cambio de tendencia
                trend = self._identify_trend(i)
                if position.position_type == 'LONG' and trend == 'BEARISH':
                    self.close_position(position, current_price, current_time, 'Trend Change')
                elif position.position_type == 'SHORT' and trend == 'BULLISH':
                    self.close_position(position, current_price, current_time, 'Trend Change')

            # Evaluar nuevas senales
            if pd.notna(signals.iloc[i]) and len(self.positions) < self.config.max_positions:
                signal_type = signals.iloc[i]
                ema_f = self.data['ema_fast'].iloc[i]
                ema_m = self.data['ema_medium'].iloc[i]
                atr = self.data['atr'].iloc[i]

                if pd.isna(ema_f) or pd.isna(ema_m) or pd.isna(atr) or atr == 0:
                    continue

                if signal_type == 'BUY':
                    # Stop loss: min(EMA21, EMA55) - ATR * multiplier
                    stop_loss = min(ema_f, ema_m) - (atr * self.atr_sl_multiplier)
                    risk = current_price - stop_loss
                    take_profit = current_price + (risk * self.tp1_ratio)

                    signal = TradeSignal(
                        timestamp=current_time,
                        signal='LONG',
                        price=current_price,
                        confidence=1.0,
                        indicators={
                            'ema_fast': ema_f,
                            'ema_medium': ema_m,
                            'adx': self.data['adx'].iloc[i],
                            'atr': atr
                        }
                    )
                    self.open_position(signal, stop_loss, take_profit)

                elif signal_type == 'SELL':
                    # Stop loss: max(EMA21, EMA55) + ATR * multiplier
                    stop_loss = max(ema_f, ema_m) + (atr * self.atr_sl_multiplier)
                    risk = stop_loss - current_price
                    take_profit = current_price - (risk * self.tp1_ratio)

                    signal = TradeSignal(
                        timestamp=current_time,
                        signal='SHORT',
                        price=current_price,
                        confidence=1.0,
                        indicators={
                            'ema_fast': ema_f,
                            'ema_medium': ema_m,
                            'adx': self.data['adx'].iloc[i],
                            'atr': atr
                        }
                    )
                    self.open_position(signal, stop_loss, take_profit)

            # Actualizar equity curve
            total_equity = self.capital
            for position in self.positions:
                if position.position_type == 'LONG':
                    total_equity += position.quantity * current_price
                else:
                    total_equity += position.quantity * (2 * position.entry_price - current_price)
            self.equity_curve.append(total_equity)
