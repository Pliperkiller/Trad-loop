# Strategy Trader

Sistema completo de desarrollo, backtesting, análisis y optimización de estrategias de trading algorítmico en Python.

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## Características Principales

- **Sistema de Trading Modular**: Framework base para crear estrategias personalizadas
- **Indicadores Técnicos**: Biblioteca completa de indicadores (SMA, EMA, RSI, MACD, Bollinger Bands, ATR, Stochastic)
- **Backtesting Robusto**: Motor de backtesting con gestión de riesgo, stop loss y take profit
- **Análisis de Performance**: +30 métricas cuantitativas (Sharpe, Sortino, Calmar, Profit Factor, etc.)
- **Visualizaciones Avanzadas**: Dashboards interactivos y gráficos de análisis
- **Optimización Multi-Algoritmo**: 4 métodos de optimización de parámetros
- **Walk Forward Validation**: Validación robusta anti-overfitting

## Tabla de Contenidos

- [Instalación](#instalación)
- [Quick Start](#quick-start)
- [Componentes](#componentes)
- [Ejemplo Completo](#ejemplo-completo)
- [Métricas de Performance](#métricas-de-performance)
- [Optimización](#optimización)
- [Documentación](#documentación)
- [Contribuir](#contribuir)
- [Licencia](#licencia)

## Instalación

### Requisitos

- Python 3.8 o superior
- pip

### Instalación Básica

```bash
# Clonar el repositorio
git clone https://github.com/tu-usuario/strategy-trader.git
cd strategy-trader

# Instalar dependencias
pip install -r requirements.txt
```

### Instalación con Optimización Bayesiana (Opcional)

```bash
pip install -r requirements-full.txt
```

## Quick Start

### 1. Crear una Estrategia Simple

```python
from src.strategy import TradingStrategy, StrategyConfig
from src.indicators import TechnicalIndicators
import pandas as pd
import numpy as np

class MiEstrategia(TradingStrategy):
    def __init__(self, config, fast_period=10, slow_period=30):
        super().__init__(config)
        self.fast_period = fast_period
        self.slow_period = slow_period
    
    def calculate_indicators(self):
        self.data['ema_fast'] = TechnicalIndicators.ema(
            self.data['close'], self.fast_period
        )
        self.data['ema_slow'] = TechnicalIndicators.ema(
            self.data['close'], self.slow_period
        )
    
    def generate_signals(self) -> pd.Series:
        signals = pd.Series(index=self.data.index, dtype=object)
        
        cross = np.where(
            self.data['ema_fast'] > self.data['ema_slow'], 1, -1
        )
        cross_signal = pd.Series(cross).diff()
        
        signals[cross_signal == 2] = 'BUY'
        signals[cross_signal == -2] = 'SELL'
        
        return signals

# Configurar y ejecutar
config = StrategyConfig(
    symbol='BTC/USD',
    timeframe='1H',
    initial_capital=10000,
    risk_per_trade=2.0,
    max_positions=3,
    commission=0.1,
    slippage=0.05
)

strategy = MiEstrategia(config)
strategy.load_data(tu_dataframe)
strategy.backtest()

# Ver resultados
metrics = strategy.get_performance_metrics()
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
print(f"Total Return: {metrics['total_return_pct']:.2f}%")
```

### 2. Analizar Performance

```python
from src.performance import PerformanceAnalyzer, PerformanceVisualizer

# Crear analizador
analyzer = PerformanceAnalyzer(
    equity_curve=strategy.equity_curve,
    trades=pd.DataFrame(strategy.closed_trades),
    initial_capital=10000
)

# Imprimir reporte completo
analyzer.print_report()

# Visualizaciones
visualizer = PerformanceVisualizer(analyzer)
visualizer.plot_comprehensive_dashboard()
visualizer.plot_rolling_metrics(window=50)
visualizer.plot_trade_analysis()
```

### 3. Optimizar Parámetros

```python
from src.optimizer import StrategyOptimizer

# Crear optimizador
optimizer = StrategyOptimizer(
    strategy_class=MiEstrategia,
    data=tu_dataframe,
    config_template=config,
    objective_metric='sharpe_ratio'
)

# Definir espacio de parámetros
optimizer.add_parameter('fast_period', 'int', low=5, high=20, step=5)
optimizer.add_parameter('slow_period', 'int', low=20, high=50, step=10)

# Optimizar (elige un método)
result = optimizer.bayesian_optimization(n_calls=50)
result.print_summary()

print(f"Mejores parámetros: {result.best_params}")
```

## Componentes

### 1. Strategy Framework (`src/strategy.py`)

Sistema base para crear estrategias de trading:

- **TradingStrategy**: Clase abstracta base
- **TechnicalIndicators**: Biblioteca de indicadores técnicos
- **StrategyConfig**: Configuración de estrategia
- **TradeSignal**: Estructura de señales de trading
- **Position**: Gestión de posiciones

**Indicadores Disponibles:**
- SMA (Simple Moving Average)
- EMA (Exponential Moving Average)
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- ATR (Average True Range)
- Stochastic Oscillator

### 2. Performance Analysis (`src/performance.py`)

Análisis cuantitativo exhaustivo:

**Métricas de Rentabilidad:**
- Total Return
- CAGR (Compound Annual Growth Rate)
- Expectancy

**Métricas de Riesgo:**
- Maximum Drawdown
- Volatilidad
- Value at Risk (VaR)
- Conditional VaR

**Métricas Ajustadas por Riesgo:**
- Sharpe Ratio
- Sortino Ratio
- Calmar Ratio
- Omega Ratio

**Métricas de Consistencia:**
- Win Rate
- Profit Factor
- Risk/Reward Ratio
- Recovery Factor

### 3. Visualization (`src/performance.py` - PerformanceVisualizer)

Visualizaciones profesionales:

- **Dashboard Completo**: 6 gráficos de análisis
- **Métricas Rodantes**: Evolución temporal de KPIs
- **Análisis de Trades**: Distribución de wins/losses
- **Análisis de Riesgo**: Drawdowns, VaR, normalidad

### 4. Optimization (`src/optimizer.py`)

4 métodos de optimización de parámetros:

#### Grid Search
- **Uso**: Espacios pequeños, 2-3 parámetros
- **Pros**: Encuentra óptimo global garantizado
- **Contras**: Computacionalmente costoso

```python
result = optimizer.grid_search(verbose=True)
```

#### Random Search
- **Uso**: Exploración rápida, 3-5 parámetros
- **Pros**: Eficiente, simple
- **Contras**: No garantías

```python
result = optimizer.random_search(n_iter=100, verbose=True)
```

#### Bayesian Optimization
- **Uso**: Parámetros continuos, presupuesto limitado
- **Pros**: Muy eficiente, sample efficient
- **Contras**: Requiere scikit-optimize

```python
result = optimizer.bayesian_optimization(n_calls=50, verbose=True)
```

#### Genetic Algorithm
- **Uso**: Espacios complejos, 4-8 parámetros
- **Pros**: Robusto, evita óptimos locales
- **Contras**: Muchas evaluaciones

```python
result = optimizer.genetic_algorithm(
    population_size=20,
    max_generations=50,
    verbose=True
)
```

#### Walk Forward Optimization
- **Uso**: Validación final robusta
- **Pros**: Evita overfitting
- **Contras**: Muy costoso computacionalmente

```python
wf_result = optimizer.walk_forward_optimization(
    optimization_method='bayesian',
    n_splits=5,
    train_size=0.6
)
```

## Ejemplo Completo

Ver `examples/complete_workflow.py` para un ejemplo completo de:
1. Carga de datos
2. Creación de estrategia personalizada
3. Backtesting
4. Análisis de performance
5. Optimización de parámetros
6. Walk forward validation

```bash
python examples/complete_workflow.py
```

## Métricas de Performance

### Criterios de Viabilidad

Una estrategia se considera **VIABLE** si cumple:

- Sharpe Ratio > 1.0
- Profit Factor > 1.5
- Max Drawdown < 30%
- Win Rate > 40% (o Risk/Reward > 2)
- Total Trades > 30 (validez estadística)
- Calmar Ratio > 1.0
- Recovery Factor > 2.0

### Interpretación de Métricas

**Sharpe Ratio:**
- < 0: Malo
- 0-1: Subóptimo
- 1-2: Bueno
- 2-3: Muy Bueno
- \> 3: Excelente

**Profit Factor:**
- < 1.0: Perdedora
- 1.0-1.5: Marginal
- 1.5-2.0: Bueno
- 2.0-3.0: Muy Bueno
- \> 3.0: Excelente

**Max Drawdown:**
- < 10%: Excelente
- 10-20%: Muy Bueno
- 20-30%: Aceptable
- \> 40%: Peligroso

## Estructura del Proyecto

```
strategy-trader/
│
├── src/
│   ├── __init__.py
│   ├── strategy.py          # Framework de estrategias
│   ├── performance.py       # Análisis y visualización
│   └── optimizer.py         # Optimización de parámetros
│
├── examples/
│   ├── complete_workflow.py # Ejemplo completo
│   ├── custom_strategy.py   # Estrategia personalizada
│   └── optimization_demo.py # Demo de optimización
│
├── tests/
│   ├── test_strategy.py
│   ├── test_performance.py
│   └── test_optimizer.py
│
├── docs/
│   ├── strategy_guide.md
│   ├── metrics_guide.md
│   └── optimization_guide.md
│
├── requirements.txt
├── requirements-full.txt
├── setup.py
├── LICENSE
└── README.md
```

## Workflow Recomendado

### 1. Desarrollo de Estrategia

```
Idea → Implementación → Backtesting Inicial → Análisis Básico
```

### 2. Optimización

```
Random Search (exploración) → Bayesian Opt (refinamiento) → Grid Search (verificación)
```

### 3. Validación

```
Walk Forward Optimization → Análisis de Robustez → Trading en Papel
```

### 4. Producción

```
Trading en Vivo con Capital Pequeño → Monitoreo → Escalamiento
```

## Buenas Prácticas

### 1. Evitar Overfitting

- Usa Walk Forward Optimization
- Mínimo 100+ trades para validez estadística
- No optimices demasiados parámetros (máximo 5-6)
- Deja un período out-of-sample sin tocar

### 2. Gestión de Riesgo

- Nunca arriesgues más del 2% por trade
- Usa siempre stop loss
- Diversifica en múltiples estrategias
- Monitorea el drawdown constantemente

### 3. Validación

- Verifica estabilidad de parámetros en walk forward
- Asegura que degradación train→test < 30%
- Prueba en diferentes períodos temporales
- Valida con datos de diferentes mercados

## Documentación Adicional

- [Guía de Estrategias](docs/strategy_guide.md)
- [Guía de Métricas](docs/metrics_guide.md)
- [Guía de Optimización](docs/optimization_guide.md)

## Contribuir

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit tus cambios (`git commit -m 'Añadir nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Abre un Pull Request

## Roadmap

- [ ] Integración con APIs de exchanges (Binance, Coinbase, etc.)
- [ ] Trading en vivo
- [ ] Backtesting multi-asset
- [ ] Análisis de correlaciones
- [ ] Machine Learning integration
- [ ] Dashboard web interactivo
- [ ] Alertas y notificaciones
- [ ] Backtesting con datos de order book

## Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.

## Disclaimer

Este software es solo para fines educativos y de investigación. El trading conlleva riesgos significativos. No nos hacemos responsables por pérdidas financieras derivadas del uso de este software. Siempre realiza tu propia investigación y considera consultar con un asesor financiero profesional.

## Contacto

Para preguntas, sugerencias o reportar bugs, por favor abre un issue en GitHub.

---

**Hecho con ❤️ para la comunidad de trading algorítmico**
