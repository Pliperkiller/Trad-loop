# Guía de Estrategias

## Creando Tu Primera Estrategia

### Estructura Básica

Toda estrategia debe heredar de `TradingStrategy` e implementar dos métodos:

```python
from src.strategy import TradingStrategy, StrategyConfig

class MiEstrategia(TradingStrategy):
    def __init__(self, config, param1, param2):
        super().__init__(config)
        self.param1 = param1
        self.param2 = param2
    
    def calculate_indicators(self):
        # Calcular tus indicadores técnicos
        pass
    
    def generate_signals(self) -> pd.Series:
        # Generar señales BUY/SELL
        pass
```

### Paso 1: Definir Parámetros

```python
def __init__(self, config, fast_period=10, slow_period=30):
    super().__init__(config)
    self.fast_period = fast_period
    self.slow_period = slow_period
```

### Paso 2: Calcular Indicadores

```python
def calculate_indicators(self):
    # Usar la clase TechnicalIndicators
    self.data['ema_fast'] = TechnicalIndicators.ema(
        self.data['close'], 
        self.fast_period
    )
    self.data['ema_slow'] = TechnicalIndicators.ema(
        self.data['close'], 
        self.slow_period
    )
```

### Paso 3: Generar Señales

```python
def generate_signals(self) -> pd.Series:
    signals = pd.Series(index=self.data.index, dtype=object)
    
    for i in range(1, len(self.data)):
        if self.data['ema_fast'].iloc[i] > self.data['ema_slow'].iloc[i]:
            if self.data['ema_fast'].iloc[i-1] <= self.data['ema_slow'].iloc[i-1]:
                signals.iloc[i] = 'BUY'
        elif self.data['ema_fast'].iloc[i] < self.data['ema_slow'].iloc[i]:
            signals.iloc[i] = 'SELL'
    
    return signals
```

## Indicadores Disponibles

### Medias Móviles
- `TechnicalIndicators.sma(data, period)` - Simple Moving Average
- `TechnicalIndicators.ema(data, period)` - Exponential Moving Average

### Osciladores
- `TechnicalIndicators.rsi(data, period)` - Relative Strength Index
- `TechnicalIndicators.stochastic(high, low, close, period)` - Stochastic Oscillator

### Volatilidad
- `TechnicalIndicators.bollinger_bands(data, period, std_dev)` - Bandas de Bollinger
- `TechnicalIndicators.atr(high, low, close, period)` - Average True Range

### Momentum
- `TechnicalIndicators.macd(data, fast, slow, signal)` - MACD

## Ejemplos de Estrategias

### 1. Cruce de Medias Móviles

```python
class MAStrategy(TradingStrategy):
    def calculate_indicators(self):
        self.data['ma_fast'] = TechnicalIndicators.sma(self.data['close'], 20)
        self.data['ma_slow'] = TechnicalIndicators.sma(self.data['close'], 50)
    
    def generate_signals(self):
        # Señal cuando MA rápida cruza MA lenta
        ...
```

### 2. RSI Oversold/Overbought

```python
class RSIStrategy(TradingStrategy):
    def calculate_indicators(self):
        self.data['rsi'] = TechnicalIndicators.rsi(self.data['close'], 14)
    
    def generate_signals(self):
        # Comprar cuando RSI < 30
        # Vender cuando RSI > 70
        ...
```

### 3. Bollinger Bands Breakout

```python
class BBStrategy(TradingStrategy):
    def calculate_indicators(self):
        upper, middle, lower = TechnicalIndicators.bollinger_bands(
            self.data['close'], 20, 2.0
        )
        self.data['bb_upper'] = upper
        self.data['bb_lower'] = lower
    
    def generate_signals(self):
        # Comprar cuando precio toca banda inferior
        # Vender cuando precio toca banda superior
        ...
```

## Buenas Prácticas

1. **Validación de Datos**: Verifica que tienes suficientes datos antes de calcular indicadores
2. **Manejo de NaN**: Los primeros valores de indicadores serán NaN
3. **Confirmación Múltiple**: Usa varios indicadores para confirmar señales
4. **Gestión de Riesgo**: Siempre implementa stop loss y take profit
5. **Testing**: Prueba con diferentes períodos temporales

## Configuración Recomendada

```python
config = StrategyConfig(
    symbol='BTC/USD',
    timeframe='1H',
    initial_capital=10000,
    risk_per_trade=2.0,      # 2% de riesgo por trade
    max_positions=3,          # Máximo 3 posiciones simultáneas
    commission=0.1,           # 0.1% de comisión
    slippage=0.05             # 0.05% de slippage
)
```

## Próximos Pasos

1. Crea tu estrategia siguiendo estos ejemplos
2. Pruébala con `backtest()`
3. Analiza resultados con `PerformanceAnalyzer`
4. Optimiza parámetros con `StrategyOptimizer`
5. Valida con Walk Forward

Ver `examples/custom_strategy.py` para más ejemplos completos.
