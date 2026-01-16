# Strategy Trader

Sistema completo de desarrollo, backtesting, análisis y optimización de estrategias de trading algorítmico en Python.

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## Características Principales

- **Sistema de Trading Modular**: Framework base para crear estrategias personalizadas
- **Indicadores Técnicos**: Biblioteca completa de 30+ indicadores (SMA, EMA, RSI, MACD, Bollinger Bands, ATR, Stochastic, Supertrend, Ichimoku, etc.)
- **Backtesting Robusto**: Motor de backtesting con gestión de riesgo, stop loss y take profit
- **Análisis de Performance**: 36+ métricas cuantitativas (Sharpe, Sortino, Calmar, Profit Factor, métricas de mean reversion, etc.)
- **Optimización Multi-Algoritmo**: 5 métodos de optimización con soporte de paralelización
- **Walk Forward Validation**: Validación robusta anti-overfitting
- **Paper Trading**: Motor de simulación en tiempo real con 17+ tipos de órdenes
- **API REST/WebSocket**: Integración completa para aplicaciones frontend
- **Multi-Broker**: Soporte para 100+ exchanges vía CCXT e Interactive Brokers

## Tabla de Contenidos

- [Instalación](#instalación)
- [Quick Start](#quick-start)
- [Componentes](#componentes)
- [API REST](#api-rest)
- [Paper Trading](#paper-trading)
- [Ejemplo Completo](#ejemplo-completo)
- [Métricas de Performance](#métricas-de-performance)
- [Optimización](#optimización)
- [Documentación](#documentación)
- [Contribuir](#contribuir)
- [Licencia](#licencia)

## Instalación

### Requisitos

- Python 3.10 o superior
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

# Optimizar con paralelización
result = optimizer.bayesian_optimization(n_calls=50, n_jobs=4)
result.print_summary()

print(f"Mejores parámetros: {result.best_params}")
```

## Componentes

### 1. Strategy Framework (`src/strategy/`)

Sistema base para crear estrategias de trading:

- **TradingStrategy**: Clase abstracta base
- **StrategyConfig**: Configuración de estrategia
- **Position**: Gestión de posiciones
- **SignalGenerator**: Generación de señales

### 2. Indicadores Técnicos (`src/indicators/`)

30+ indicadores técnicos organizados por categoría:

**Trend:**
- SMA, EMA, WMA, VWMA
- Parabolic SAR
- Supertrend
- Ichimoku Cloud
- ADX

**Momentum:**
- RSI (Relative Strength Index)
- MACD
- Stochastic Oscillator
- Williams %R
- CCI (Commodity Channel Index)
- ROC (Rate of Change)

**Volatility:**
- Bollinger Bands
- ATR (Average True Range)
- Keltner Channels
- Donchian Channels

**Volume:**
- OBV (On Balance Volume)
- CMF (Chaikin Money Flow)
- MFI (Money Flow Index)
- VWAP

### 3. Performance Analysis (`src/performance.py`)

36+ métricas de análisis cuantitativo:

**Métricas de Rentabilidad:**
- Total Return, CAGR, Expectancy

**Métricas de Riesgo:**
- Maximum Drawdown, Volatilidad, VaR, CVaR

**Métricas Ajustadas por Riesgo:**
- Sharpe Ratio, Sortino Ratio, Calmar Ratio, Omega Ratio

**Métricas de Consistencia:**
- Win Rate, Profit Factor, Risk/Reward Ratio, Recovery Factor

**Métricas de Mean Reversion:**
- MRQS (Mean Reversion Quality Score)
- Target Hit Rate, Average Excursion

### 4. Optimization (`src/optimizer.py`, `src/optimizers/`)

5 métodos de optimización con soporte de paralelización (`n_jobs`):

#### Grid Search
```python
result = optimizer.grid_search(verbose=True)
```

#### Random Search
```python
result = optimizer.random_search(n_iter=100, n_jobs=4, verbose=True)
```

#### Bayesian Optimization
```python
result = optimizer.bayesian_optimization(n_calls=50, n_jobs=4, verbose=True)
```

#### Genetic Algorithm
```python
result = optimizer.genetic_algorithm(
    population_size=20,
    max_generations=50,
    verbose=True
)
```

#### Walk Forward Optimization
```python
wf_result = optimizer.walk_forward_optimization(
    optimization_method='bayesian',
    n_splits=5,
    train_size=0.6,
    n_jobs=4
)
```

### 5. Paper Trading (`src/paper_trading/`)

Motor de simulación en tiempo real:

- 17+ tipos de órdenes (Market, Limit, Stop, OCO, Trailing, Bracket, etc.)
- Gestión de posiciones y riesgo
- WebSocket para actualizaciones en vivo
- Integración con datos de mercado en tiempo real

### 6. Broker Bridge (`src/broker_bridge/`)

Capa de abstracción multi-broker:

- **CCXT Adapter**: 100+ exchanges de criptomonedas
- **IBKR Adapter**: Interactive Brokers para mercados tradicionales
- **Unified Executor**: API unificada para todos los brokers

### 7. Risk Management (`src/risk_management/`)

- Position Sizing (Fixed, Kelly, Volatility-based)
- Límites de exposición
- VaR y correlación de portfolio
- Stop Loss dinámico

### 8. Portfolio Management (`src/portfolio/`)

- Asset allocation multi-activo
- Rebalanceo automático
- Métricas de portfolio

## API REST

### Endpoints Principales

```
GET  /api/v1/exchanges                    # Exchanges disponibles
GET  /api/v1/symbols/{exchange}           # Símbolos por exchange
GET  /api/v1/ohlcv/{exchange}/{symbol}    # Datos OHLCV
GET  /api/v1/strategies/available         # Estrategias disponibles

POST /api/v1/backtest                     # Ejecutar backtest (async)
GET  /api/v1/backtest/{jobId}             # Estado del backtest

POST /api/v1/optimize                     # Ejecutar optimización (async)
GET  /api/v1/optimize/{jobId}             # Estado de optimización

POST /api/v1/paper-trading/start          # Iniciar paper trading
POST /api/v1/paper-trading/stop           # Detener paper trading
```

### WebSocket

```
WS /ws/paper-trading/{sessionId}          # Paper trading en vivo
WS /ws/live-candles                       # Velas en tiempo real
```

### Iniciar el Servidor

```bash
cd Trad-loop/StrategyTrader
uvicorn src.api:app --host 0.0.0.0 --port 8000
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
StrategyTrader/
├── src/
│   ├── strategy/              # Framework de estrategias
│   │   ├── base.py            # Clase base TradingStrategy
│   │   ├── strategies/        # Estrategias implementadas
│   │   └── signals.py         # Generación de señales
│   │
│   ├── indicators/            # 30+ indicadores técnicos
│   │   ├── technical/         # Indicadores técnicos
│   │   └── fundamental/       # APIs de datos fundamentales
│   │
│   ├── optimizers/            # Métodos de optimización
│   │   ├── bayesian.py        # Optimización Bayesiana (paralela)
│   │   ├── genetic.py         # Algoritmo genético
│   │   ├── random_search.py   # Búsqueda aleatoria (paralela)
│   │   └── walk_forward.py    # Walk Forward Validation
│   │
│   ├── paper_trading/         # Motor de paper trading
│   │   ├── engine.py          # Motor principal
│   │   └── orders/            # 17+ tipos de órdenes
│   │
│   ├── broker_bridge/         # Integración multi-broker
│   │   ├── adapters/          # CCXT, IBKR
│   │   └── unified_executor.py
│   │
│   ├── risk_management/       # Gestión de riesgo
│   ├── portfolio/             # Gestión de portfolio
│   ├── stress_testing/        # Monte Carlo, escenarios
│   │
│   ├── api.py                 # API REST FastAPI
│   ├── websocket_api.py       # WebSocket endpoints
│   ├── performance.py         # 36+ métricas
│   └── optimizer.py           # Orchestrador de optimización
│
├── tests/                     # Tests con pytest
├── docs/                      # Documentación
├── requirements.txt
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
Walk Forward Optimization → Análisis de Robustez → Paper Trading
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

## Contribuir

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit tus cambios (`git commit -m 'Añadir nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Abre un Pull Request

## Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.

## Disclaimer

Este software es solo para fines educativos y de investigación. El trading conlleva riesgos significativos. No nos hacemos responsables por pérdidas financieras derivadas del uso de este software. Siempre realiza tu propia investigación y considera consultar con un asesor financiero profesional.

## Contacto

Para preguntas, sugerencias o reportar bugs, por favor abre un issue en GitHub.
