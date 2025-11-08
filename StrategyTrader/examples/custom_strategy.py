"""
Ejemplo: Creación de Estrategia Personalizada
Muestra cómo crear tu propia estrategia desde cero
"""

import pandas as pd
import numpy as np
import sys
sys.path.append('..')

from src.strategy import TradingStrategy, StrategyConfig, TechnicalIndicators


class RSIBollingerStrategy(TradingStrategy):
    """
    Estrategia personalizada: RSI + Bollinger Bands
    
    Señales:
    - COMPRA: RSI < 30 y precio toca banda inferior de Bollinger
    - VENTA: RSI > 70 y precio toca banda superior de Bollinger
    """
    
    def __init__(self, config: StrategyConfig, 
                 rsi_period: int = 14,
                 bb_period: int = 20,
                 bb_std: float = 2.0,
                 rsi_oversold: float = 30,
                 rsi_overbought: float = 70):
        """
        Args:
            config: Configuración de la estrategia
            rsi_period: Período del RSI
            bb_period: Período de las Bandas de Bollinger
            bb_std: Desviación estándar para BB
            rsi_oversold: Nivel de sobreventa del RSI
            rsi_overbought: Nivel de sobrecompra del RSI
        """
        super().__init__(config)
        self.rsi_period = rsi_period
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        
    def calculate_indicators(self):
        """Calcula RSI y Bollinger Bands"""
        # RSI
        self.data['rsi'] = TechnicalIndicators.rsi(
            self.data['close'], 
            self.rsi_period
        )
        
        # Bollinger Bands
        upper, middle, lower = TechnicalIndicators.bollinger_bands(
            self.data['close'],
            self.bb_period,
            self.bb_std
        )
        
        self.data['bb_upper'] = upper
        self.data['bb_middle'] = middle
        self.data['bb_lower'] = lower
        
        # Calcular ancho de banda (volatilidad)
        self.data['bb_width'] = (upper - lower) / middle * 100
        
    def generate_signals(self) -> pd.Series:
        """Genera señales de trading"""
        signals = pd.Series(index=self.data.index, dtype=object)
        
        for i in range(1, len(self.data)):
            current_price = self.data['close'].iloc[i]
            rsi = self.data['rsi'].iloc[i]
            bb_lower = self.data['bb_lower'].iloc[i]
            bb_upper = self.data['bb_upper'].iloc[i]
            bb_width = self.data['bb_width'].iloc[i]
            
            # Señal de COMPRA
            # RSI en sobreventa Y precio cerca o debajo de banda inferior
            if (rsi < self.rsi_oversold and 
                current_price <= bb_lower * 1.01 and
                bb_width > 2):  # Asegurar que hay volatilidad
                signals.iloc[i] = 'BUY'
            
            # Señal de VENTA
            # RSI en sobrecompra Y precio cerca o arriba de banda superior
            elif (rsi > self.rsi_overbought and 
                  current_price >= bb_upper * 0.99):
                signals.iloc[i] = 'SELL'
        
        return signals


class TripleEMAStrategy(TradingStrategy):
    """
    Estrategia de Triple EMA con volumen
    
    Usa 3 EMAs y confirma señales con volumen
    """
    
    def __init__(self, config: StrategyConfig,
                 fast_period: int = 9,
                 medium_period: int = 21,
                 slow_period: int = 55,
                 volume_ma_period: int = 20):
        super().__init__(config)
        self.fast_period = fast_period
        self.medium_period = medium_period
        self.slow_period = slow_period
        self.volume_ma_period = volume_ma_period
        
    def calculate_indicators(self):
        """Calcula las 3 EMAs y media de volumen"""
        self.data['ema_fast'] = TechnicalIndicators.ema(
            self.data['close'], self.fast_period
        )
        self.data['ema_medium'] = TechnicalIndicators.ema(
            self.data['close'], self.medium_period
        )
        self.data['ema_slow'] = TechnicalIndicators.ema(
            self.data['close'], self.slow_period
        )
        self.data['volume_ma'] = self.data['volume'].rolling(
            window=self.volume_ma_period
        ).mean()
        
    def generate_signals(self) -> pd.Series:
        """Genera señales basadas en alineación de EMAs"""
        signals = pd.Series(index=self.data.index, dtype=object)
        
        for i in range(1, len(self.data)):
            fast = self.data['ema_fast'].iloc[i]
            medium = self.data['ema_medium'].iloc[i]
            slow = self.data['ema_slow'].iloc[i]
            volume = self.data['volume'].iloc[i]
            volume_ma = self.data['volume_ma'].iloc[i]
            
            fast_prev = self.data['ema_fast'].iloc[i-1]
            medium_prev = self.data['ema_medium'].iloc[i-1]
            
            # Señal de COMPRA: EMAs alineadas alcista + volumen alto
            if (fast > medium > slow and
                fast_prev <= medium_prev and  # Cruce reciente
                volume > volume_ma * 1.2):  # Volumen 20% arriba del promedio
                signals.iloc[i] = 'BUY'
            
            # Señal de VENTA: EMAs alineadas bajista
            elif (fast < medium < slow and
                  fast_prev >= medium_prev):
                signals.iloc[i] = 'SELL'
        
        return signals


def demo_custom_strategies():
    """Demuestra el uso de estrategias personalizadas"""
    print("="*70)
    print("DEMO: ESTRATEGIAS PERSONALIZADAS")
    print("="*70)
    
    # Generar datos de ejemplo
    dates = pd.date_range(start='2024-01-01', periods=500, freq='1H')
    np.random.seed(42)
    
    close_prices = 100 + np.random.randn(500).cumsum() * 2
    data = pd.DataFrame({
        'open': close_prices + np.random.randn(500) * 0.5,
        'high': close_prices + np.abs(np.random.randn(500)) * 2,
        'low': close_prices - np.abs(np.random.randn(500)) * 2,
        'close': close_prices,
        'volume': np.random.randint(1000, 10000, 500)
    }, index=dates)
    
    data['high'] = data[['open', 'close', 'high']].max(axis=1)
    data['low'] = data[['open', 'close', 'low']].min(axis=1)
    
    # Configuración
    config = StrategyConfig(
        symbol='BTC/USD',
        timeframe='1H',
        initial_capital=10000,
        risk_per_trade=2.0,
        max_positions=2,
        commission=0.1,
        slippage=0.05
    )
    
    # Probar Estrategia 1: RSI + Bollinger
    print("\n[ESTRATEGIA 1: RSI + Bollinger Bands]")
    strategy1 = RSIBollingerStrategy(config)
    strategy1.load_data(data)
    strategy1.backtest()
    
    metrics1 = strategy1.get_performance_metrics()
    print(f"  Trades: {metrics1.get('total_trades', 0)}")
    print(f"  Win Rate: {metrics1.get('win_rate', 0):.2f}%")
    print(f"  Sharpe: {metrics1.get('sharpe_ratio', 0):.2f}")
    print(f"  Return: {metrics1.get('total_return_pct', 0):.2f}%")
    
    # Probar Estrategia 2: Triple EMA
    print("\n[ESTRATEGIA 2: Triple EMA]")
    strategy2 = TripleEMAStrategy(config)
    strategy2.load_data(data)
    strategy2.backtest()
    
    metrics2 = strategy2.get_performance_metrics()
    print(f"  Trades: {metrics2.get('total_trades', 0)}")
    print(f"  Win Rate: {metrics2.get('win_rate', 0):.2f}%")
    print(f"  Sharpe: {metrics2.get('sharpe_ratio', 0):.2f}")
    print(f"  Return: {metrics2.get('total_return_pct', 0):.2f}%")
    
    # Comparación
    print("\n[COMPARACION]")
    print(f"  RSI+BB Sharpe: {metrics1.get('sharpe_ratio', 0):.2f}")
    print(f"  Triple EMA Sharpe: {metrics2.get('sharpe_ratio', 0):.2f}")
    
    if metrics1.get('sharpe_ratio', 0) > metrics2.get('sharpe_ratio', 0):
        print("\n  Ganador: RSI + Bollinger Bands")
    else:
        print("\n  Ganador: Triple EMA")
    
    print("\n" + "="*70)
    print("TIP: Puedes crear infinitas variaciones combinando indicadores!")
    print("="*70)


if __name__ == "__main__":
    demo_custom_strategies()
