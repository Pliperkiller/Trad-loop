# Quick Start - Strategy Trader

## Instalación Rápida

```bash
# Clonar repositorio
git clone https://github.com/tu-usuario/strategy-trader.git
cd strategy-trader

# Instalar dependencias
pip install -r requirements.txt

# (Opcional) Para optimización Bayesiana
pip install -r requirements-full.txt
```

## Tu Primera Estrategia en 5 Minutos

### 1. Preparar Datos

```python
import pandas as pd
import numpy as np

# Crear datos de ejemplo (o carga tus propios datos)
dates = pd.date_range('2024-01-01', periods=500, freq='1H')
data = pd.DataFrame({
    'open': np.random.randn(500).cumsum() + 100,
    'high': np.random.randn(500).cumsum() + 102,
    'low': np.random.randn(500).cumsum() + 98,
    'close': np.random.randn(500).cumsum() + 100,
    'volume': np.random.randint(1000, 10000, 500)
}, index=dates)
```

### 2. Crear Estrategia

```python
from src.strategy import MovingAverageCrossoverStrategy, StrategyConfig

# Configurar
config = StrategyConfig(
    symbol='BTC/USD',
    timeframe='1H',
    initial_capital=10000,
    risk_per_trade=2.0,
    max_positions=3,
    commission=0.1,
    slippage=0.05
)

# Crear estrategia
strategy = MovingAverageCrossoverStrategy(
    config=config,
    fast_period=10,
    slow_period=30,
    rsi_period=14
)
```

### 3. Ejecutar Backtest

```python
strategy.load_data(data)
strategy.backtest()

# Ver resultados
metrics = strategy.get_performance_metrics()
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
print(f"Total Return: {metrics['total_return_pct']:.2f}%")
print(f"Win Rate: {metrics['win_rate']:.2f}%")
```

### 4. Analizar Performance

```python
from src.performance import PerformanceAnalyzer

analyzer = PerformanceAnalyzer(
    equity_curve=strategy.equity_curve,
    trades=pd.DataFrame(strategy.closed_trades),
    initial_capital=10000
)

analyzer.print_report()
```

### 5. Optimizar Parámetros

```python
from src.optimizer import StrategyOptimizer

optimizer = StrategyOptimizer(
    strategy_class=MovingAverageCrossoverStrategy,
    data=data,
    config_template=config,
    objective_metric='sharpe_ratio'
)

# Definir parámetros a optimizar
optimizer.add_parameter('fast_period', 'int', low=5, high=20, step=5)
optimizer.add_parameter('slow_period', 'int', low=20, high=50, step=10)

# Optimizar
result = optimizer.random_search(n_iter=30, verbose=True)
result.print_summary()
```

## Ejemplos Incluidos

```bash
# Workflow completo
python examples/complete_workflow.py

# Crear estrategias personalizadas
python examples/custom_strategy.py

# Comparar métodos de optimización
python examples/optimization_demo.py
```

## Próximos Pasos

1. Lee la [Guía de Estrategias](docs/strategy_guide.md)
2. Familiarízate con las [Métricas](docs/metrics_guide.md)
3. Crea tu propia estrategia personalizada
4. Optimiza y valida con Walk Forward

## Estructura del Proyecto

```
strategy-trader/
├── src/
│   ├── strategy.py      # Framework de estrategias
│   ├── performance.py   # Análisis y visualización
│   └── optimizer.py     # Optimización de parámetros
├── examples/
│   ├── complete_workflow.py
│   ├── custom_strategy.py
│   └── optimization_demo.py
├── docs/
│   ├── strategy_guide.md
│   └── metrics_guide.md
└── README.md
```

## Ayuda

- Documentación completa: [README.md](README.md)
- Issues: GitHub Issues
- Ejemplos: Carpeta `examples/`

## Notas Importantes

### Archivos Completos

Los archivos `src/performance.py` y `src/optimizer.py` contienen placeholders con instrucciones. Para obtener el código completo (~3500 líneas):

1. Consulta las conversaciones donde desarrollamos estos módulos
2. Implementa progresivamente según necesites
3. O contacta al repositorio

### Dependencias Opcionales

- `scikit-optimize`: Para optimización Bayesiana (recomendado)

```bash
pip install scikit-optimize
```

## Happy Trading!

Recuerda: Este software es solo para fines educativos. El trading conlleva riesgos.
