# Trad-Loop

**Professional Multi-Asset Algorithmic Trading Framework**

A complete system for developing, backtesting, optimizing, and executing trading strategies. Supports cryptocurrencies (100+ exchanges via CCXT) and traditional markets (Interactive Brokers).

## Key Features

- **Multi-Broker**: CCXT (Binance, Bybit, OKX, etc.) + Interactive Brokers
- **Advanced Backtesting**: Validation with Time Series CV, Walk-Forward, Purged K-Fold
- **Optimization**: Grid Search, Random Search, Bayesian, Genetic, Walk-Forward
- **30+ Technical Indicators**: RSI, MACD, Bollinger, Ichimoku, ATR, etc.
- **Fundamental Data**: CoinGecko, Glassnode, DeFi Llama, Santiment
- **Paper Trading**: 17+ order types (TWAP, VWAP, Trailing, Bracket, OCO)
- **Portfolio Management**: Risk Parity, Mean-Variance, Inverse Volatility
- **Risk Management**: Kelly Criterion, VaR, Position Sizing, Correlation Tracking
- **Stress Testing**: Monte Carlo, Scenario Analysis, Sensitivity Analysis
- **REST & WebSocket APIs**: FastAPI with endpoints for strategies, trades, performance
- **Async Job System**: Background task execution for backtests and optimizations

## Project Structure

```
Trad-loop/
├── DataExtractor/              # Market data extraction
│   ├── src/
│   │   ├── domain/             # Entities and repositories
│   │   ├── application/        # Services and use cases
│   │   ├── infrastructure/     # Exchange adapters
│   │   └── presentation/       # GUI (Tkinter) and CLI
│   └── main.py
│
├── StrategyTrader/             # Strategy framework
│   ├── src/
│   │   ├── strategy/           # Strategy base classes
│   │   │   ├── base.py         # TradingStrategy ABC
│   │   │   └── strategies/     # Built-in strategies
│   │   ├── performance.py      # Performance analysis
│   │   ├── optimizer.py        # Parameter optimization
│   │   ├── job_manager.py      # Async job system
│   │   ├── api.py              # Main REST API
│   │   ├── backtest_api.py     # Backtest API
│   │   ├── websocket_api.py    # WebSocket API
│   │   ├── api_paper_trading.py# Paper Trading API
│   │   ├── broker_bridge/      # Multi-broker layer
│   │   ├── indicators/         # Technical/fundamental indicators
│   │   ├── optimizers/         # Optimization algorithms
│   │   │   ├── validation/     # Cross-validation methods
│   │   │   └── analysis/       # Overfitting detection
│   │   ├── paper_trading/      # Paper trading engine
│   │   │   ├── orders/         # Order types
│   │   │   └── simulators/     # Specialized simulators
│   │   ├── portfolio/          # Portfolio management
│   │   ├── risk_management/    # Risk management
│   │   ├── stress_testing/     # Stress tests
│   │   ├── interfaces/         # Protocols and DI container
│   │   ├── api_modules/        # API helper modules
│   │   └── config/             # Configuration
│   ├── tests/                  # Test suite
│   └── examples/               # Usage examples
│
└── docs/                       # Documentation
    ├── technical_manual.md     # C4 diagrams and architecture
    ├── user_guide.md           # User guide
    └── components/             # Component manuals
```

## Installation

```bash
# Clone repository
git clone https://github.com/Pliperkiller/Trad-loop.git
cd Trad-loop

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r StrategyTrader/requirements.txt
pip install -r DataExtractor/requirements.txt
```

## Quick Start

### 1. Create a Strategy

```python
from src.strategy.base import TradingStrategy, TradeSignal, TechnicalIndicators
import pandas as pd

class MyStrategy(TradingStrategy):
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

### 2. Run Backtest

```python
import pandas as pd

# Load data
data = pd.read_csv('BTCUSDT_1h.csv', parse_dates=['timestamp'], index_col='timestamp')

# Create and run strategy
strategy = MyStrategy(fast_period=12, slow_period=26)
strategy.load_data(data)
strategy.config.initial_capital = 10000
strategy.config.risk_per_trade = 0.02

trades = strategy.backtest()
metrics = strategy.get_performance_metrics()

print(f"Total Return: {metrics['total_return_pct']:.2f}%")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
```

### 3. Analyze Performance

```python
from src.performance import PerformanceAnalyzer, PerformanceVisualizer

analyzer = PerformanceAnalyzer(
    equity_curve=strategy.equity_curve,
    trades_df=trades,
    initial_capital=10000
)

# 30+ metrics
metrics = analyzer.calculate_all_metrics()
print(f"Calmar Ratio: {metrics['calmar_ratio']:.2f}")
print(f"Sortino Ratio: {metrics['sortino_ratio']:.2f}")
print(f"VaR 95%: {metrics['value_at_risk_95']:.2f}%")

# Visualization
visualizer = PerformanceVisualizer(analyzer)
visualizer.plot_comprehensive_dashboard()
```

### 4. Optimize Parameters

```python
from src.optimizer import StrategyOptimizer

optimizer = StrategyOptimizer(
    strategy_class=MyStrategy,
    data=data,
    initial_capital=10000
)

optimizer.add_parameter('fast_period', 'int', 5, 20)
optimizer.add_parameter('slow_period', 'int', 20, 50)

# Bayesian optimization
results = optimizer.bayesian_optimize(
    n_iterations=50,
    objective='sharpe_ratio'
)

print(f"Best parameters: {results.best_params}")
print(f"Best Sharpe: {results.best_value:.2f}")
```

### 5. Paper Trading

```python
from src.paper_trading import PaperTradingEngine, RealtimeStrategy

class MyRealtimeStrategy(RealtimeStrategy):
    def on_candle(self, candle):
        # Trading logic
        if self.should_buy(candle):
            self.buy(candle.close, quantity=0.1, stop_loss=candle.close*0.98)

    def on_tick(self, symbol, price):
        # Update trailing stops, etc.
        pass

engine = PaperTradingEngine(initial_capital=10000)
engine.register_strategy(MyRealtimeStrategy())
await engine.run_live()
```

### 6. Multi-Broker Execution

```python
from src.broker_bridge import UnifiedExecutor, CCXTBroker, IBKRBroker, BrokerOrder
from src.broker_bridge.core.enums import OrderSide, OrderType

executor = UnifiedExecutor()

# Register brokers
executor.register_broker(CCXTBroker("binance", api_key="...", api_secret="..."))
executor.register_broker(IBKRBroker(port=7497))

await executor.connect_all()

# Crypto -> CCXT automatically
await executor.submit_order(BrokerOrder(
    symbol="BTC/USDT",
    side=OrderSide.BUY,
    order_type=OrderType.LIMIT,
    quantity=0.1,
    price=50000
))

# Stock -> IBKR automatically
await executor.submit_order(BrokerOrder(
    symbol="AAPL",
    side=OrderSide.BUY,
    order_type=OrderType.MARKET,
    quantity=10
))
```

## Documentation

| Document | Description |
|----------|-------------|
| [Technical Manual](docs/technical_manual.md) | Architecture, C4 diagrams, flows |
| [User Guide](docs/user_guide.md) | Complete usage guide |
| [API Reference](docs/components/api_reference.md) | REST & WebSocket API documentation |
| [Job Manager](docs/components/job_manager.md) | Async task system |
| [Broker Bridge](docs/components/broker_bridge.md) | Multi-broker execution |
| [Paper Trading](docs/components/paper_trading.md) | Order simulation |
| [Indicators](docs/components/indicators.md) | Technical and fundamental indicators |
| [Optimizers](docs/components/optimizers.md) | Optimization algorithms |
| [Portfolio](docs/components/portfolio.md) | Portfolio management |
| [Risk Management](docs/components/risk_management.md) | Risk management |
| [Stress Testing](docs/components/stress_testing.md) | Stress testing |
| [Interfaces](docs/components/interfaces.md) | Protocol system and DI |

## REST API

```bash
# Start server
cd StrategyTrader
uvicorn src.api:app --reload --port 8000

# Main endpoints
GET  /api/v1/strategies           # List strategies
GET  /api/v1/trades/{id}          # Get trades
GET  /api/v1/performance/{id}     # Performance metrics
GET  /api/v1/equity/{id}          # Equity curve
GET  /api/v1/exchanges            # Available exchanges
GET  /api/v1/symbols/{exchange}   # Symbols for exchange
GET  /api/v1/ohlcv/{exchange}/{symbol}  # OHLCV data
```

Swagger UI available at: `http://localhost:8000/docs`

## WebSocket API

```python
import websockets
import json

async def subscribe_to_candles():
    async with websockets.connect('ws://localhost:8000/ws/candles') as ws:
        await ws.send(json.dumps({
            'action': 'subscribe',
            'exchange': 'binance',
            'symbol': 'BTC/USDT',
            'timeframe': '1h'
        }))
        async for message in ws:
            candle = json.loads(message)
            print(f"New candle: {candle}")
```

## Tests

```bash
# Run all tests
cd StrategyTrader
pytest

# With coverage
pytest --cov=src --cov-report=html

# Specific tests
pytest tests/test_strategy.py -v
pytest src/broker_bridge/tests/ -v
```

## Main Dependencies

| Category | Packages |
|----------|----------|
| Data | pandas, numpy, scipy |
| ML | scikit-learn, scikit-optimize |
| Exchanges | ccxt, python-binance, ib_insync |
| Web | fastapi, uvicorn, websockets |
| Viz | matplotlib, seaborn, mplfinance |
| Testing | pytest, pytest-asyncio |

## Roadmap

- [ ] Support for more exchanges via CCXT
- [ ] Machine Learning strategies
- [ ] Distributed backtesting
- [ ] Interactive web dashboard
- [ ] Alerts and notifications
- [ ] TradingView integration

## License

MIT License - see [LICENSE](LICENSE) for details.

## Author

Carlos Caro ([@Pliperkiller](https://github.com/Pliperkiller))
