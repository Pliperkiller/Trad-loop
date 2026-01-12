# Trad-Loop

**Framework profesional de trading algoritmico multi-activo**

Sistema completo para desarrollo, backtesting, optimizacion y ejecucion de estrategias de trading. Soporta criptomonedas (100+ exchanges via CCXT) y mercados tradicionales (Interactive Brokers).

## Caracteristicas Principales

- **Multi-Broker**: CCXT (Binance, Bybit, OKX, etc.) + Interactive Brokers
- **Backtesting Avanzado**: Validacion con Time Series CV, Walk-Forward, Purged K-Fold
- **Optimizacion**: Grid Search, Random Search, Bayesian, Genetic, Walk-Forward
- **30+ Indicadores Tecnicos**: RSI, MACD, Bollinger, Ichimoku, ATR, etc.
- **Datos Fundamentales**: CoinGecko, Glassnode, DeFi Llama, Santiment
- **Paper Trading**: 17+ tipos de orden (TWAP, VWAP, Trailing, Bracket, OCO)
- **Gestion de Portafolio**: Risk Parity, Mean-Variance, Inverse Volatility
- **Risk Management**: Kelly Criterion, VaR, Position Sizing, Correlation Tracking
- **Stress Testing**: Monte Carlo, Scenario Analysis, Sensitivity Analysis
- **API REST**: FastAPI con endpoints para estrategias, trades, performance

## Estructura del Proyecto

```
Trad-loop/
├── DataExtractor/          # Extraccion de datos de mercado
│   ├── src/
│   │   ├── domain/         # Entidades y repositorios
│   │   ├── application/    # Servicios y casos de uso
│   │   ├── infrastructure/ # Adaptadores de exchanges
│   │   └── presentation/   # GUI (Tkinter) y CLI
│   └── main.py
│
├── StrategyTrader/         # Framework de estrategias
│   ├── src/
│   │   ├── strategy.py     # Base de estrategias
│   │   ├── performance.py  # Analisis de rendimiento
│   │   ├── optimizer.py    # Optimizacion de parametros
│   │   ├── api.py          # REST API
│   │   ├── broker_bridge/  # Capa multi-broker
│   │   ├── indicators/     # Indicadores tecnicos/fundamentales
│   │   ├── optimizers/     # Algoritmos de optimizacion
│   │   ├── paper_trading/  # Motor de paper trading
│   │   ├── portfolio/      # Gestion de portafolio
│   │   ├── risk_management/# Gestion de riesgo
│   │   └── stress_testing/ # Pruebas de estres
│   ├── tests/              # Suite de tests
│   └── examples/           # Ejemplos de uso
│
└── docs/                   # Documentacion
    ├── manual_tecnico.md   # Diagramas C4 y arquitectura
    ├── manual_uso.md       # Guia general de uso
    └── componentes/        # Manuales por componente
```

## Instalacion

```bash
# Clonar repositorio
git clone https://github.com/Pliperkiller/Trad-loop.git
cd Trad-loop

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Instalar dependencias
pip install -r StrategyTrader/requirements.txt
pip install -r DataExtractor/requirements.txt
```

## Inicio Rapido

### 1. Crear una Estrategia

```python
from src.strategy import TradingStrategy, TradeSignal, TechnicalIndicators
import pandas as pd

class MiEstrategia(TradingStrategy):
    def __init__(self, fast_period=10, slow_period=20):
        super().__init__()
        self.fast_period = fast_period
        self.slow_period = slow_period

    def calculate_indicators(self):
        self.data['ema_fast'] = TechnicalIndicators.ema(
            self.data['close'], self.fast_period
        )
        self.data['ema_slow'] = TechnicalIndicators.ema(
            self.data['close'], self.slow_period
        )
        self.data['rsi'] = TechnicalIndicators.rsi(
            self.data['close'], 14
        )

    def generate_signals(self):
        signals = []
        for i in range(len(self.data)):
            if (self.data['ema_fast'].iloc[i] > self.data['ema_slow'].iloc[i]
                and self.data['rsi'].iloc[i] < 70):
                signals.append(TradeSignal(
                    timestamp=self.data.index[i],
                    signal='BUY',
                    confidence=0.8
                ))
            elif self.data['ema_fast'].iloc[i] < self.data['ema_slow'].iloc[i]:
                signals.append(TradeSignal(
                    timestamp=self.data.index[i],
                    signal='SELL',
                    confidence=0.8
                ))
            else:
                signals.append(TradeSignal(
                    timestamp=self.data.index[i],
                    signal='HOLD',
                    confidence=0.5
                ))
        return signals
```

### 2. Ejecutar Backtest

```python
import pandas as pd

# Cargar datos
data = pd.read_csv('BTCUSDT_1h.csv', parse_dates=['timestamp'], index_col='timestamp')

# Crear y ejecutar estrategia
strategy = MiEstrategia(fast_period=12, slow_period=26)
strategy.load_data(data)
strategy.config.initial_capital = 10000
strategy.config.risk_per_trade = 0.02

trades = strategy.backtest()
metrics = strategy.get_performance_metrics()

print(f"Total Return: {metrics['total_return_pct']:.2f}%")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
```

### 3. Analizar Rendimiento

```python
from src.performance import PerformanceAnalyzer, PerformanceVisualizer

analyzer = PerformanceAnalyzer(
    equity_curve=strategy.equity_curve,
    trades_df=trades,
    initial_capital=10000
)

# 30+ metricas
metrics = analyzer.calculate_all_metrics()
print(f"Calmar Ratio: {metrics['calmar_ratio']:.2f}")
print(f"Sortino Ratio: {metrics['sortino_ratio']:.2f}")
print(f"VaR 95%: {metrics['value_at_risk_95']:.2f}%")

# Visualizacion
visualizer = PerformanceVisualizer(analyzer)
visualizer.plot_comprehensive_dashboard()
```

### 4. Optimizar Parametros

```python
from src.optimizer import StrategyOptimizer

optimizer = StrategyOptimizer(
    strategy_class=MiEstrategia,
    data=data,
    initial_capital=10000
)

optimizer.add_parameter('fast_period', 'int', 5, 20)
optimizer.add_parameter('slow_period', 'int', 20, 50)

# Optimizacion Bayesiana
results = optimizer.bayesian_optimize(
    n_iterations=50,
    objective='sharpe_ratio'
)

print(f"Mejores parametros: {results.best_params}")
print(f"Mejor Sharpe: {results.best_value:.2f}")
```

### 5. Trading en Vivo (Paper)

```python
from src.paper_trading import PaperTradingEngine, RealtimeStrategy

class MiEstrategiaRealtime(RealtimeStrategy):
    def on_candle(self, candle):
        # Logica de trading
        if self.should_buy(candle):
            self.buy(candle.close, quantity=0.1, stop_loss=candle.close*0.98)

    def on_tick(self, symbol, price):
        # Actualizar trailing stops, etc.
        pass

engine = PaperTradingEngine(initial_capital=10000)
engine.register_strategy(MiEstrategiaRealtime())
await engine.run_live()
```

### 6. Ejecucion Multi-Broker

```python
from src.broker_bridge import UnifiedExecutor, CCXTBroker, IBKRBroker, BrokerOrder

executor = UnifiedExecutor()

# Registrar brokers
executor.register_broker(CCXTBroker("binance", api_key="...", api_secret="..."))
executor.register_broker(IBKRBroker(port=7497))

await executor.connect_all()

# Crypto -> CCXT automaticamente
await executor.submit_order(BrokerOrder(
    symbol="BTC/USDT",
    side=OrderSide.BUY,
    order_type=OrderType.LIMIT,
    quantity=0.1,
    price=50000
))

# Stock -> IBKR automaticamente
await executor.submit_order(BrokerOrder(
    symbol="AAPL",
    side=OrderSide.BUY,
    order_type=OrderType.MARKET,
    quantity=10
))
```

## Documentacion

| Documento | Descripcion |
|-----------|-------------|
| [Manual Tecnico](docs/manual_tecnico.md) | Arquitectura, diagramas C4, flujos |
| [Manual de Uso](docs/manual_uso.md) | Guia completa de uso |
| [Broker Bridge](docs/componentes/broker_bridge.md) | Multi-broker execution |
| [Paper Trading](docs/componentes/paper_trading.md) | Simulacion de ordenes |
| [Indicadores](docs/componentes/indicadores.md) | Indicadores tecnicos y fundamentales |
| [Optimizadores](docs/componentes/optimizadores.md) | Algoritmos de optimizacion |
| [Portfolio](docs/componentes/portfolio.md) | Gestion de portafolio |
| [Risk Management](docs/componentes/risk_management.md) | Gestion de riesgo |

## API REST

```bash
# Iniciar servidor
uvicorn src.api:app --reload --port 8000

# Endpoints disponibles
GET  /api/v1/strategies           # Listar estrategias
GET  /api/v1/trades/{id}          # Obtener trades
GET  /api/v1/performance/{id}     # Metricas de rendimiento
GET  /api/v1/equity/{id}          # Curva de equity
```

Swagger UI disponible en: `http://localhost:8000/docs`

## Tests

```bash
# Ejecutar todos los tests
pytest

# Con coverage
pytest --cov=src --cov-report=html

# Tests especificos
pytest tests/test_strategy.py -v
pytest src/broker_bridge/tests/ -v
```

## Dependencias Principales

| Categoria | Paquetes |
|-----------|----------|
| Data | pandas, numpy, scipy |
| ML | scikit-learn, scikit-optimize |
| Exchanges | ccxt, python-binance, ib_insync |
| Web | fastapi, uvicorn |
| Viz | matplotlib, seaborn, mplfinance |
| Testing | pytest, pytest-asyncio |

## Roadmap

- [ ] Soporte para mas exchanges via CCXT
- [ ] Machine Learning strategies
- [ ] Backtesting distribuido
- [ ] Dashboard web interactivo
- [ ] Alertas y notificaciones
- [ ] Integracion con TradingView

## Licencia

MIT License - ver [LICENSE](LICENSE) para detalles.

## Autor

Carlos Caro ([@Pliperkiller](https://github.com/Pliperkiller))
