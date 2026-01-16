# Guía de Estrategias

## Creando Tu Primera Estrategia

### Estructura Básica

Toda estrategia debe heredar de `TradingStrategy` e implementar dos métodos:

```python
from src.strategy import TradingStrategy, StrategyConfig
import pandas as pd

class MiEstrategia(TradingStrategy):
    def __init__(self, config, param1=10, param2=30):
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
def __init__(self, config, fast_period=10, slow_period=30, rsi_threshold=30):
    super().__init__(config)
    self.fast_period = fast_period
    self.slow_period = slow_period
    self.rsi_threshold = rsi_threshold
```

### Paso 2: Calcular Indicadores

```python
def calculate_indicators(self):
    from src.indicators import TechnicalIndicators

    # Medias móviles
    self.data['ema_fast'] = TechnicalIndicators.ema(
        self.data['close'],
        self.fast_period
    )
    self.data['ema_slow'] = TechnicalIndicators.ema(
        self.data['close'],
        self.slow_period
    )

    # RSI
    self.data['rsi'] = TechnicalIndicators.rsi(
        self.data['close'],
        14
    )
```

### Paso 3: Generar Señales

```python
def generate_signals(self) -> pd.Series:
    signals = pd.Series(index=self.data.index, dtype=object)

    for i in range(1, len(self.data)):
        # Cruce alcista + RSI sobreventa
        if (self.data['ema_fast'].iloc[i] > self.data['ema_slow'].iloc[i] and
            self.data['ema_fast'].iloc[i-1] <= self.data['ema_slow'].iloc[i-1] and
            self.data['rsi'].iloc[i] < self.rsi_threshold):
            signals.iloc[i] = 'BUY'

        # Cruce bajista
        elif (self.data['ema_fast'].iloc[i] < self.data['ema_slow'].iloc[i] and
              self.data['ema_fast'].iloc[i-1] >= self.data['ema_slow'].iloc[i-1]):
            signals.iloc[i] = 'SELL'

    return signals
```

---

## Indicadores Disponibles (30+)

### Medias Móviles (Trend)

```python
from src.indicators import TechnicalIndicators as ti

# Simple Moving Average
sma = ti.sma(data['close'], period=20)

# Exponential Moving Average
ema = ti.ema(data['close'], period=20)

# Weighted Moving Average
wma = ti.wma(data['close'], period=20)

# Volume Weighted Moving Average
vwma = ti.vwma(data['close'], data['volume'], period=20)
```

### Osciladores (Momentum)

```python
# RSI - Relative Strength Index
rsi = ti.rsi(data['close'], period=14)

# MACD
macd_line, signal_line, histogram = ti.macd(
    data['close'],
    fast=12,
    slow=26,
    signal=9
)

# Stochastic Oscillator
k, d = ti.stochastic(
    data['high'],
    data['low'],
    data['close'],
    k_period=14,
    d_period=3
)

# Williams %R
williams = ti.williams_r(
    data['high'],
    data['low'],
    data['close'],
    period=14
)

# CCI - Commodity Channel Index
cci = ti.cci(
    data['high'],
    data['low'],
    data['close'],
    period=20
)

# ROC - Rate of Change
roc = ti.roc(data['close'], period=12)
```

### Volatilidad

```python
# Bollinger Bands
upper, middle, lower = ti.bollinger_bands(
    data['close'],
    period=20,
    std_dev=2.0
)

# ATR - Average True Range
atr = ti.atr(
    data['high'],
    data['low'],
    data['close'],
    period=14
)

# Keltner Channels
upper, middle, lower = ti.keltner_channels(
    data['high'],
    data['low'],
    data['close'],
    period=20,
    multiplier=2.0
)

# Donchian Channels
upper, middle, lower = ti.donchian_channels(
    data['high'],
    data['low'],
    period=20
)
```

### Trend

```python
# Parabolic SAR
sar = ti.parabolic_sar(
    data['high'],
    data['low'],
    data['close'],
    af_start=0.02,
    af_max=0.2
)

# Supertrend
supertrend, direction = ti.supertrend(
    data['high'],
    data['low'],
    data['close'],
    period=10,
    multiplier=3.0
)

# ADX - Average Directional Index
adx, plus_di, minus_di = ti.adx(
    data['high'],
    data['low'],
    data['close'],
    period=14
)

# Ichimoku Cloud
tenkan, kijun, senkou_a, senkou_b, chikou = ti.ichimoku(
    data['high'],
    data['low'],
    data['close']
)
```

### Volumen

```python
# OBV - On Balance Volume
obv = ti.obv(data['close'], data['volume'])

# CMF - Chaikin Money Flow
cmf = ti.cmf(
    data['high'],
    data['low'],
    data['close'],
    data['volume'],
    period=20
)

# MFI - Money Flow Index
mfi = ti.mfi(
    data['high'],
    data['low'],
    data['close'],
    data['volume'],
    period=14
)

# VWAP - Volume Weighted Average Price
vwap = ti.vwap(
    data['high'],
    data['low'],
    data['close'],
    data['volume']
)
```

---

## Ejemplos de Estrategias

### 1. Cruce de Medias Móviles

```python
class MAStrategy(TradingStrategy):
    def __init__(self, config, fast_period=20, slow_period=50):
        super().__init__(config)
        self.fast_period = fast_period
        self.slow_period = slow_period

    def calculate_indicators(self):
        self.data['ma_fast'] = TechnicalIndicators.sma(
            self.data['close'], self.fast_period
        )
        self.data['ma_slow'] = TechnicalIndicators.sma(
            self.data['close'], self.slow_period
        )

    def generate_signals(self):
        signals = pd.Series(index=self.data.index, dtype=object)

        for i in range(1, len(self.data)):
            if (self.data['ma_fast'].iloc[i] > self.data['ma_slow'].iloc[i] and
                self.data['ma_fast'].iloc[i-1] <= self.data['ma_slow'].iloc[i-1]):
                signals.iloc[i] = 'BUY'
            elif (self.data['ma_fast'].iloc[i] < self.data['ma_slow'].iloc[i] and
                  self.data['ma_fast'].iloc[i-1] >= self.data['ma_slow'].iloc[i-1]):
                signals.iloc[i] = 'SELL'

        return signals
```

### 2. RSI Oversold/Overbought

```python
class RSIStrategy(TradingStrategy):
    def __init__(self, config, rsi_period=14, oversold=30, overbought=70):
        super().__init__(config)
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought

    def calculate_indicators(self):
        self.data['rsi'] = TechnicalIndicators.rsi(
            self.data['close'], self.rsi_period
        )

    def generate_signals(self):
        signals = pd.Series(index=self.data.index, dtype=object)

        for i in range(1, len(self.data)):
            # Comprar cuando RSI cruza hacia arriba desde sobreventa
            if (self.data['rsi'].iloc[i] > self.oversold and
                self.data['rsi'].iloc[i-1] <= self.oversold):
                signals.iloc[i] = 'BUY'
            # Vender cuando RSI cruza hacia abajo desde sobrecompra
            elif (self.data['rsi'].iloc[i] < self.overbought and
                  self.data['rsi'].iloc[i-1] >= self.overbought):
                signals.iloc[i] = 'SELL'

        return signals
```

### 3. Bollinger Bands Mean Reversion

```python
class BBStrategy(TradingStrategy):
    def __init__(self, config, bb_period=20, bb_std=2.0):
        super().__init__(config)
        self.bb_period = bb_period
        self.bb_std = bb_std

    def calculate_indicators(self):
        upper, middle, lower = TechnicalIndicators.bollinger_bands(
            self.data['close'], self.bb_period, self.bb_std
        )
        self.data['bb_upper'] = upper
        self.data['bb_middle'] = middle
        self.data['bb_lower'] = lower

    def generate_signals(self):
        signals = pd.Series(index=self.data.index, dtype=object)

        for i in range(1, len(self.data)):
            # Comprar cuando precio toca banda inferior
            if self.data['close'].iloc[i] <= self.data['bb_lower'].iloc[i]:
                signals.iloc[i] = 'BUY'
            # Vender cuando precio toca banda superior
            elif self.data['close'].iloc[i] >= self.data['bb_upper'].iloc[i]:
                signals.iloc[i] = 'SELL'

        return signals
```

### 4. Supertrend Strategy

```python
class SupertrendStrategy(TradingStrategy):
    def __init__(self, config, period=10, multiplier=3.0):
        super().__init__(config)
        self.period = period
        self.multiplier = multiplier

    def calculate_indicators(self):
        supertrend, direction = TechnicalIndicators.supertrend(
            self.data['high'],
            self.data['low'],
            self.data['close'],
            self.period,
            self.multiplier
        )
        self.data['supertrend'] = supertrend
        self.data['st_direction'] = direction

    def generate_signals(self):
        signals = pd.Series(index=self.data.index, dtype=object)

        for i in range(1, len(self.data)):
            # Cambio a tendencia alcista
            if (self.data['st_direction'].iloc[i] == 1 and
                self.data['st_direction'].iloc[i-1] == -1):
                signals.iloc[i] = 'BUY'
            # Cambio a tendencia bajista
            elif (self.data['st_direction'].iloc[i] == -1 and
                  self.data['st_direction'].iloc[i-1] == 1):
                signals.iloc[i] = 'SELL'

        return signals
```

---

## Configuración de Estrategia

### StrategyConfig

```python
from src.strategy import StrategyConfig

config = StrategyConfig(
    symbol='BTC/USD',
    timeframe='1H',
    initial_capital=10000,
    risk_per_trade=2.0,      # 2% de riesgo por trade
    max_positions=3,          # Máximo 3 posiciones simultáneas
    commission=0.1,           # 0.1% de comisión
    slippage=0.05,            # 0.05% de slippage
    use_stop_loss=True,       # Activar stop loss
    stop_loss_pct=2.0,        # Stop loss al 2%
    use_take_profit=True,     # Activar take profit
    take_profit_pct=4.0       # Take profit al 4%
)
```

### Parámetros Importantes

| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `symbol` | str | Par de trading (ej: 'BTC/USD') |
| `timeframe` | str | Temporalidad ('1m', '5m', '1H', '4H', '1D') |
| `initial_capital` | float | Capital inicial |
| `risk_per_trade` | float | % de riesgo por operación |
| `max_positions` | int | Máximo de posiciones simultáneas |
| `commission` | float | % de comisión por operación |
| `slippage` | float | % de slippage estimado |
| `use_stop_loss` | bool | Activar stop loss |
| `stop_loss_pct` | float | % de stop loss |
| `use_take_profit` | bool | Activar take profit |
| `take_profit_pct` | float | % de take profit |

---

## Position Sizing

El sistema soporta múltiples métodos de position sizing:

### Fixed Size

```python
from src.risk_management import PositionSizerFactory

sizer = PositionSizerFactory.create('fixed', size=0.1)  # 10% del capital
```

### Risk-Based (Kelly Criterion)

```python
sizer = PositionSizerFactory.create('kelly', win_rate=0.55, avg_win=2.0, avg_loss=1.0)
```

### Volatility-Based

```python
sizer = PositionSizerFactory.create('volatility', risk_pct=2.0, atr_multiplier=2.0)
```

---

## Ejecutando Backtests

### Backtest Básico

```python
# Crear estrategia
strategy = MiEstrategia(config, fast_period=10, slow_period=30)

# Cargar datos
strategy.load_data(dataframe)

# Ejecutar backtest
strategy.backtest()

# Ver resultados
metrics = strategy.get_performance_metrics()
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
print(f"Total Trades: {metrics['total_trades']}")
```

### Accediendo a Resultados

```python
# Equity curve
equity = strategy.equity_curve

# Trades cerrados
trades = strategy.closed_trades

# Señales generadas
signals = strategy.signals
```

---

## Optimización de Parámetros

### Configurar Optimizador

```python
from src.optimizer import StrategyOptimizer

optimizer = StrategyOptimizer(
    strategy_class=MiEstrategia,
    data=dataframe,
    config_template=config,
    objective_metric='sharpe_ratio'
)

# Definir espacio de parámetros
optimizer.add_parameter('fast_period', 'int', low=5, high=30, step=5)
optimizer.add_parameter('slow_period', 'int', low=20, high=100, step=10)
optimizer.add_parameter('rsi_threshold', 'int', low=20, high=40, step=5)
```

### Métodos de Optimización

```python
# Random Search (rápido, exploratorio)
result = optimizer.random_search(n_iter=100, n_jobs=4)

# Bayesian Optimization (eficiente)
result = optimizer.bayesian_optimization(n_calls=50, n_jobs=4)

# Walk Forward (validación robusta)
wf_result = optimizer.walk_forward_optimization(
    optimization_method='bayesian',
    n_splits=5,
    train_size=0.6
)
```

---

## Buenas Prácticas

### 1. Validación de Datos

```python
def calculate_indicators(self):
    # Verificar que hay suficientes datos
    if len(self.data) < self.slow_period:
        raise ValueError(f"Necesitas al menos {self.slow_period} candles")

    # Los primeros valores serán NaN
    self.data['ema'] = TechnicalIndicators.ema(
        self.data['close'], self.slow_period
    )
```

### 2. Confirmación Múltiple

```python
def generate_signals(self):
    signals = pd.Series(index=self.data.index, dtype=object)

    for i in range(1, len(self.data)):
        # Usar múltiples confirmaciones
        ma_bullish = self.data['ema_fast'].iloc[i] > self.data['ema_slow'].iloc[i]
        rsi_oversold = self.data['rsi'].iloc[i] < 30
        volume_high = self.data['volume'].iloc[i] > self.data['volume'].mean()

        if ma_bullish and rsi_oversold and volume_high:
            signals.iloc[i] = 'BUY'

    return signals
```

### 3. Gestión de Riesgo

- Nunca arriesgues más del 2% por trade
- Usa siempre stop loss
- Diversifica en múltiples estrategias
- Monitorea el drawdown constantemente

### 4. Testing

- Prueba con diferentes períodos temporales
- Usa Walk Forward para evitar overfitting
- Mínimo 100+ trades para validez estadística
- Valida con datos out-of-sample

---

## Próximos Pasos

1. Crea tu estrategia siguiendo estos ejemplos
2. Pruébala con `backtest()`
3. Analiza resultados con `PerformanceAnalyzer`
4. Optimiza parámetros con `StrategyOptimizer`
5. Valida con Walk Forward
6. Prueba en Paper Trading antes de ir a producción

Ver `examples/custom_strategy.py` para más ejemplos completos.
