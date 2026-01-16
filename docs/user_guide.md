# User Guide - Trad-Loop

## Table of Contents

1. [Introduction](#1-introduction)
2. [Installation and Configuration](#2-installation-and-configuration)
3. [Data Extraction (DataExtractor)](#3-data-extraction-dataextractor)
4. [Strategy Development](#4-strategy-development)
5. [Backtesting](#5-backtesting)
6. [Performance Analysis](#6-performance-analysis)
7. [Parameter Optimization](#7-parameter-optimization)
8. [Paper Trading](#8-paper-trading)
9. [Live Trading (Multi-Broker)](#9-live-trading-multi-broker)
10. [Portfolio Management](#10-portfolio-management)
11. [Risk Management](#11-risk-management)
12. [Stress Testing](#12-stress-testing)
13. [REST and WebSocket API](#13-rest-and-websocket-api)
14. [Job Manager](#14-job-manager)
15. [FAQ](#15-faq)

---

## 1. Introduction

**Trad-Loop** is a complete framework for algorithmic trading that allows you to:

- Extract historical market data
- Develop and test trading strategies
- Optimize parameters with multiple algorithms
- Execute paper trading with advanced orders
- Trade live with CCXT (crypto) and Interactive Brokers (traditional)
- Manage multi-asset portfolios
- Control risk professionally
- Perform stress testing on strategies

### Typical Workflow

```
1. Extract Data → 2. Develop Strategy → 3. Backtest → 4. Optimize
        ↓                                                 ↓
5. Validate (OOS) ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ←
        ↓
6. Paper Trading → 7. Live Trading → 8. Monitoring and Adjustments
```

---

## 2. Installation and Configuration

### 2.1 Prerequisites

- Python 3.9 or higher
- pip (package manager)
- Git

### 2.2 Installation

```bash
# Clone repository
git clone https://github.com/Pliperkiller/Trad-loop.git
cd Trad-loop

# Create virtual environment
python -m venv venv

# Activate environment
source venv/bin/activate      # Linux/Mac
# venv\Scripts\activate       # Windows

# Install dependencies
pip install -r StrategyTrader/requirements.txt
pip install -r DataExtractor/requirements.txt
```

### 2.3 API Keys Configuration

Create a `.env` file in the project root:

```env
# Crypto Exchanges (CCXT)
BINANCE_API_KEY=your_api_key
BINANCE_API_SECRET=your_api_secret
BYBIT_API_KEY=your_api_key
BYBIT_API_SECRET=your_api_secret

# Interactive Brokers
IBKR_HOST=127.0.0.1
IBKR_PORT=7497          # TWS Paper: 7497, TWS Live: 7496
IBKR_CLIENT_ID=1

# Data APIs
COINGECKO_API_KEY=your_api_key
GLASSNODE_API_KEY=your_api_key
```

### 2.4 Verify Installation

```bash
# Verify imports
python -c "from src.strategy import TradingStrategy; print('OK')"

# Run tests
pytest --version
pytest tests/ -v --tb=short
```

---

## 3. Data Extraction (DataExtractor)

### 3.1 Graphical Interface (GUI)

```bash
cd DataExtractor
python main.py
```

The GUI allows you to:
- Select exchange (Binance, Kraken, etc.)
- Choose trading pair (BTC/USDT, ETH/USD)
- Define date range
- Select timeframe (1m, 5m, 1h, 1d)
- Export to CSV

### 3.2 Command Line (CLI)

```bash
cd DataExtractor
python cli_extract.py --exchange binance --symbol BTC/USDT --timeframe 1h --start 2024-01-01 --end 2024-12-31 --output data/btc_usdt_1h.csv
```

### 3.3 Programmatic Usage

```python
from src.infrastructure.exchanges.binance_exchange import BinanceExchange
from src.infrastructure.exchanges.csv_exporter import CSVExporter
from datetime import datetime

# Create exchange
exchange = BinanceExchange()

# Extract data
candles = exchange.fetch_ohlcv(
    symbol="BTC/USDT",
    timeframe="1h",
    since=datetime(2024, 1, 1),
    until=datetime(2024, 12, 31)
)

# Export to CSV
exporter = CSVExporter()
exporter.export(candles, "data/btc_usdt_1h.csv")
```

### 3.4 Data Format

The resulting CSV has the following format:

```csv
timestamp,open,high,low,close,volume
2024-01-01 00:00:00,42000.50,42100.00,41900.00,42050.25,1500.5
2024-01-01 01:00:00,42050.25,42200.00,42000.00,42150.00,1800.3
...
```

---

## 4. Strategy Development

### 4.1 Basic Structure

```python
from src.strategy import TradingStrategy, TradeSignal, TechnicalIndicators
import pandas as pd

class MyStrategy(TradingStrategy):
    """
    Custom strategy.

    Must implement:
    - calculate_indicators(): Calculate technical indicators
    - generate_signals(): Generate trading signals
    """

    def __init__(self, param1=10, param2=20):
        super().__init__()
        self.param1 = param1
        self.param2 = param2

    def calculate_indicators(self):
        """Calculate required technical indicators."""
        self.data['ema_fast'] = TechnicalIndicators.ema(
            self.data['close'], self.param1
        )
        self.data['ema_slow'] = TechnicalIndicators.ema(
            self.data['close'], self.param2
        )
        self.data['rsi'] = TechnicalIndicators.rsi(
            self.data['close'], 14
        )

    def generate_signals(self):
        """Generate trading signals."""
        signals = []

        for i in range(len(self.data)):
            row = self.data.iloc[i]

            # Buy condition
            if (row['ema_fast'] > row['ema_slow'] and
                row['rsi'] < 70):
                signal = TradeSignal(
                    timestamp=self.data.index[i],
                    signal='BUY',
                    confidence=0.8,
                    metadata={'reason': 'EMA crossover + RSI ok'}
                )
            # Sell condition
            elif row['ema_fast'] < row['ema_slow']:
                signal = TradeSignal(
                    timestamp=self.data.index[i],
                    signal='SELL',
                    confidence=0.8,
                    metadata={'reason': 'EMA crossover down'}
                )
            else:
                signal = TradeSignal(
                    timestamp=self.data.index[i],
                    signal='HOLD',
                    confidence=0.5
                )

            signals.append(signal)

        return signals
```

### 4.2 Available Indicators

```python
from src.strategy import TechnicalIndicators

# Moving averages
sma = TechnicalIndicators.sma(data['close'], period=20)
ema = TechnicalIndicators.ema(data['close'], period=20)

# Momentum
rsi = TechnicalIndicators.rsi(data['close'], period=14)
macd, signal, hist = TechnicalIndicators.macd(
    data['close'], fast=12, slow=26, signal=9
)

# Volatility
upper, middle, lower = TechnicalIndicators.bollinger_bands(
    data['close'], period=20, std=2.0
)
atr = TechnicalIndicators.atr(
    data['high'], data['low'], data['close'], period=14
)

# Oscillators
k, d = TechnicalIndicators.stochastic(
    data['high'], data['low'], data['close'], period=14
)
```

### 4.3 Advanced Indicators (indicators/ module)

```python
from src.indicators.technical import momentum, trend, volatility, volume
from src.indicators.technical.ichimoku import IchimokuCloud

# Advanced momentum
cci = momentum.cci(data['high'], data['low'], data['close'], period=20)
williams_r = momentum.williams_r(data['high'], data['low'], data['close'], period=14)

# Advanced trend
adx_result = trend.adx(data['high'], data['low'], data['close'], period=14)
supertrend = trend.supertrend(data['high'], data['low'], data['close'], period=10, multiplier=3)

# Advanced volatility
keltner = volatility.keltner_channels(data['high'], data['low'], data['close'])
donchian = volatility.donchian_channels(data['high'], data['low'], period=20)

# Volume
obv = volume.obv(data['close'], data['volume'])
vwap = volume.vwap(data['high'], data['low'], data['close'], data['volume'])
mfi = volume.mfi(data['high'], data['low'], data['close'], data['volume'], period=14)

# Ichimoku
ichimoku = IchimokuCloud()
result = ichimoku.calculate(data['high'], data['low'], data['close'])
# result.tenkan_sen, result.kijun_sen, result.senkou_span_a, etc.
```

### 4.4 Strategy Configuration

```python
strategy = MyStrategy(param1=12, param2=26)

# Configure parameters
strategy.config.initial_capital = 10000
strategy.config.risk_per_trade = 0.02      # 2% per trade
strategy.config.stop_loss_pct = 0.02       # 2% stop loss
strategy.config.take_profit_pct = 0.04     # 4% take profit
strategy.config.commission_pct = 0.001     # 0.1% commission
strategy.config.slippage_pct = 0.0005      # 0.05% slippage
```

---

## 5. Backtesting

### 5.1 Basic Backtest

```python
import pandas as pd

# Load data
data = pd.read_csv(
    'data/btc_usdt_1h.csv',
    parse_dates=['timestamp'],
    index_col='timestamp'
)

# Create strategy
strategy = MyStrategy(param1=12, param2=26)
strategy.config.initial_capital = 10000
strategy.config.risk_per_trade = 0.02

# Load data and run backtest
strategy.load_data(data)
trades = strategy.backtest()

# View results
print(f"Total trades: {len(trades)}")
print(f"Trades DataFrame:\n{trades.head()}")

# Basic metrics
metrics = strategy.get_performance_metrics()
print(f"\nTotal Return: {metrics['total_return_pct']:.2f}%")
print(f"Win Rate: {metrics['win_rate']:.2f}%")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
```

### 5.2 Accessing Equity Curve

```python
# Equity curve
equity = strategy.equity_curve
print(equity.head())

# Plot equity
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(equity.index, equity['equity'])
plt.title('Equity Curve')
plt.xlabel('Date')
plt.ylabel('Capital ($)')
plt.grid(True)
plt.show()
```

### 5.3 Trade Analysis

```python
# Trades DataFrame
trades_df = strategy.get_trades_dataframe()

# Filter winning trades
winners = trades_df[trades_df['pnl'] > 0]
losers = trades_df[trades_df['pnl'] < 0]

print(f"Winning trades: {len(winners)}")
print(f"Losing trades: {len(losers)}")
print(f"Average winner PnL: ${winners['pnl'].mean():.2f}")
print(f"Average loser PnL: ${losers['pnl'].mean():.2f}")
```

---

## 6. Performance Analysis

### 6.1 Performance Analyzer

```python
from src.performance import PerformanceAnalyzer, PerformanceVisualizer

# Create analyzer
analyzer = PerformanceAnalyzer(
    equity_curve=strategy.equity_curve,
    trades_df=trades,
    initial_capital=10000
)

# Calculate all metrics (30+)
metrics = analyzer.calculate_all_metrics()

# Profitability metrics
print("=== PROFITABILITY ===")
print(f"Total Return: {metrics['total_return_pct']:.2f}%")
print(f"CAGR: {metrics['cagr_pct']:.2f}%")
print(f"Expectancy: ${metrics['expectancy']:.2f}")

# Risk metrics
print("\n=== RISK ===")
print(f"Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
print(f"Max DD Duration: {metrics['max_drawdown_duration']} days")
print(f"Volatility: {metrics['volatility_pct']:.2f}%")
print(f"VaR 95%: {metrics['value_at_risk_95']:.2f}%")
print(f"CVaR 95%: {metrics['conditional_var_95']:.2f}%")

# Risk-adjusted ratios
print("\n=== RATIOS ===")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
print(f"Sortino Ratio: {metrics['sortino_ratio']:.2f}")
print(f"Calmar Ratio: {metrics['calmar_ratio']:.2f}")
print(f"Omega Ratio: {metrics['omega_ratio']:.2f}")

# Consistency
print("\n=== CONSISTENCY ===")
print(f"Win Rate: {metrics['win_rate_pct']:.2f}%")
print(f"Profit Factor: {metrics['profit_factor']:.2f}")
print(f"Risk/Reward: {metrics['risk_reward_ratio']:.2f}")
print(f"Recovery Factor: {metrics['recovery_factor']:.2f}")
```

### 6.2 Visualization

```python
# Create visualizer
visualizer = PerformanceVisualizer(analyzer)

# Complete dashboard (6 panels)
visualizer.plot_comprehensive_dashboard()

# Individual charts
visualizer.plot_equity_curve()
visualizer.plot_drawdown()
visualizer.plot_monthly_returns()
visualizer.plot_trade_distribution()
visualizer.plot_rolling_metrics()
```

### 6.3 Complete Dashboard

```python
# The dashboard includes:
# 1. Equity Curve with benchmark
# 2. Drawdown underwater chart
# 3. Monthly returns heatmap
# 4. Trade P&L distribution
# 5. Rolling Sharpe ratio
# 6. Win/Loss analysis

visualizer.plot_comprehensive_dashboard(
    figsize=(16, 12),
    save_path='reports/dashboard.png'
)
```

---

## 7. Parameter Optimization

### 7.1 Basic Configuration

```python
from src.optimizer import StrategyOptimizer

# Create optimizer
optimizer = StrategyOptimizer(
    strategy_class=MyStrategy,
    data=data,
    initial_capital=10000
)

# Add parameters to optimize
optimizer.add_parameter('param1', 'int', 5, 30, step=5)    # Fast EMA
optimizer.add_parameter('param2', 'int', 20, 60, step=10)  # Slow EMA

# Define objective
# Options: 'sharpe_ratio', 'profit_factor', 'total_return', 'calmar_ratio'
objective = 'sharpe_ratio'
```

### 7.2 Grid Search

```python
# Exhaustive search of all combinations
results = optimizer.grid_optimize(objective=objective)

print(f"Best parameters: {results.best_params}")
print(f"Best {objective}: {results.best_value:.4f}")
print(f"Total combinations: {len(results.all_results)}")
```

### 7.3 Random Search

```python
# Random search (useful for many parameters)
# Supports parallel execution with n_jobs
results = optimizer.random_optimize(
    n_iterations=100,
    objective=objective,
    n_jobs=4  # Use 4 parallel workers
)

print(f"Best parameters: {results.best_params}")
```

### 7.4 Bayesian Optimization

```python
# Intelligent optimization with Gaussian Process
# Supports parallel execution with n_jobs
results = optimizer.bayesian_optimize(
    n_iterations=50,
    objective=objective,
    n_initial_points=10,  # Initial random points
    n_jobs=4              # Use 4 parallel workers
)

print(f"Best parameters: {results.best_params}")

# View convergence
import matplotlib.pyplot as plt
plt.plot(results.convergence)
plt.xlabel('Iteration')
plt.ylabel(objective)
plt.title('Bayesian Optimization Convergence')
plt.show()
```

### 7.5 Genetic Algorithm

```python
# Evolutionary optimization
results = optimizer.genetic_optimize(
    population_size=50,
    n_generations=20,
    mutation_rate=0.1,
    crossover_rate=0.8,
    objective=objective
)

print(f"Best parameters: {results.best_params}")
```

### 7.6 Walk-Forward Analysis

```python
# Robust validation with walk-forward
results = optimizer.walk_forward_optimize(
    n_splits=5,           # Number of splits
    train_ratio=0.8,      # 80% train, 20% test
    objective=objective
)

print(f"Average IS Sharpe: {results.is_metrics['sharpe_ratio'].mean():.2f}")
print(f"Average OOS Sharpe: {results.oos_metrics['sharpe_ratio'].mean():.2f}")

# Check for overfitting
if results.oos_metrics['sharpe_ratio'].mean() < results.is_metrics['sharpe_ratio'].mean() * 0.5:
    print("WARNING: Possible overfitting detected")
```

### 7.7 Overfitting Analysis

```python
from src.optimizers.analysis.overfitting_detection import OverfittingDetector

detector = OverfittingDetector()

# Compare IS vs OOS
report = detector.analyze(
    is_results=results.is_results,
    oos_results=results.oos_results
)

print(f"Performance degradation: {report.performance_degradation:.2f}%")
print(f"Parameter stability: {report.parameter_stability:.2f}")
print(f"Overfitting risk: {report.overfitting_risk}")
```

---

## 8. Paper Trading

### 8.1 Configuration

```python
from src.paper_trading import PaperTradingEngine, PaperTradingConfig
from src.paper_trading import RealtimeStrategy

# Configuration
config = PaperTradingConfig(
    initial_capital=10000,
    commission_pct=0.001,
    slippage_pct=0.0005,
    enable_shorting=True,
    max_positions=5
)

# Create engine
engine = PaperTradingEngine(config)
```

### 8.2 Realtime Strategy

```python
class MyRealtimeStrategy(RealtimeStrategy):
    def __init__(self):
        super().__init__()
        self.ema_fast = []
        self.ema_slow = []

    def on_start(self):
        """Called when engine starts."""
        print("Strategy started")

    def on_candle(self, candle):
        """Called with each new candle."""
        # Update indicators
        self.update_emas(candle.close)

        # Generate signal
        if len(self.ema_fast) < 2:
            return

        # Bullish crossover
        if (self.ema_fast[-1] > self.ema_slow[-1] and
            self.ema_fast[-2] <= self.ema_slow[-2]):

            self.buy(
                price=candle.close,
                quantity=0.1,
                stop_loss=candle.close * 0.98,
                take_profit=candle.close * 1.04
            )

        # Bearish crossover
        elif (self.ema_fast[-1] < self.ema_slow[-1] and
              self.ema_fast[-2] >= self.ema_slow[-2]):

            self.close_all_positions()

    def on_tick(self, symbol, price):
        """Called with each tick (optional)."""
        pass

    def on_stop(self):
        """Called when engine stops."""
        print("Strategy stopped")
```

### 8.3 Running Paper Trading

```python
# Register strategy
engine.register_strategy(MyRealtimeStrategy())

# Backtest mode (with historical data)
backtest_result = engine.run_backtest(data)

# Live mode (real-time data)
await engine.run_live()
```

### 8.4 Advanced Order Types

```python
from src.paper_trading.orders import (
    TrailingStopOrder,
    BracketOrder,
    TWAPOrder,
    VWAPOrder,
    IcebergOrder
)

# Trailing Stop
trailing = TrailingStopOrder(
    symbol="BTC/USDT",
    side="SELL",
    quantity=0.1,
    trail_percent=0.02  # 2% trailing
)

# Bracket Order (Entry + SL + TP)
bracket = BracketOrder(
    symbol="BTC/USDT",
    side="BUY",
    quantity=0.1,
    entry_price=50000,
    stop_loss=49000,
    take_profit=52000
)

# TWAP (Time Weighted Average Price)
twap = TWAPOrder(
    symbol="BTC/USDT",
    side="BUY",
    total_quantity=1.0,
    duration_minutes=60,
    num_slices=12
)

# VWAP (Volume Weighted Average Price)
vwap = VWAPOrder(
    symbol="BTC/USDT",
    side="BUY",
    total_quantity=1.0,
    participation_rate=0.1  # 10% of volume
)

# Iceberg
iceberg = IcebergOrder(
    symbol="BTC/USDT",
    side="BUY",
    total_quantity=10.0,
    visible_quantity=1.0  # Show only 1.0
)
```

---

## 9. Live Trading (Multi-Broker)

### 9.1 Broker Configuration

```python
from src.broker_bridge import (
    UnifiedExecutor,
    CCXTBroker,
    IBKRBroker,
    BrokerOrder,
    OrderSide,
    OrderType
)

# Create executor
executor = UnifiedExecutor()

# Register crypto broker (CCXT)
ccxt_broker = CCXTBroker(
    exchange_id="binance",
    api_key="your_api_key",
    api_secret="your_api_secret",
    testnet=True  # Use testnet for testing
)
executor.register_broker(ccxt_broker)

# Register traditional broker (IBKR)
ibkr_broker = IBKRBroker(
    host="127.0.0.1",
    port=7497,       # TWS Paper
    client_id=1
)
executor.register_broker(ibkr_broker)
```

### 9.2 Connect and Trade

```python
# Connect all brokers
await executor.connect_all()

# Routing is automatic based on symbol:
# - BTC/USDT, ETH/BTC → CCXT (crypto)
# - AAPL, MSFT → IBKR (stocks)
# - EUR/USD → IBKR (forex)

# Crypto order
crypto_order = BrokerOrder(
    symbol="BTC/USDT",
    side=OrderSide.BUY,
    order_type=OrderType.LIMIT,
    quantity=0.1,
    price=50000
)
report = await executor.submit_order(crypto_order)
print(f"Order ID: {report.order_id}, Status: {report.status}")

# Stock order (automatically goes to IBKR)
stock_order = BrokerOrder(
    symbol="AAPL",
    side=OrderSide.BUY,
    order_type=OrderType.MARKET,
    quantity=10
)
report = await executor.submit_order(stock_order)
```

### 9.3 Position Management

```python
# Get positions from all brokers
all_positions = await executor.get_all_positions()

for broker_type, positions in all_positions.items():
    print(f"\n{broker_type.value}:")
    for pos in positions:
        print(f"  {pos.symbol}: {pos.quantity} @ {pos.entry_price}")
        print(f"  Unrealized PnL: ${pos.unrealized_pnl:.2f}")

# Get balances
balances = await executor.get_all_balances()
print(f"\nBalances:")
for broker_type, balance in balances.items():
    print(f"  {broker_type.value}: {balance}")
```

### 9.4 Context Manager

```python
# Using context manager (auto-connect/disconnect)
async with UnifiedExecutor() as executor:
    executor.register_broker(ccxt_broker)
    executor.register_broker(ibkr_broker)

    # Trade...
    await executor.submit_order(order)

# Automatic disconnection on exit
```

---

## 10. Portfolio Management

### 10.1 Configuration

```python
from src.portfolio import PortfolioManager, PortfolioConfig, AllocationMethod

config = PortfolioConfig(
    initial_capital=100000,
    symbols=['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'AAPL', 'MSFT'],
    allocation_method=AllocationMethod.RISK_PARITY,
    rebalance_frequency='monthly',
    min_position_weight=0.05,
    max_position_weight=0.40
)

manager = PortfolioManager(config)
```

### 10.2 Allocation Methods

```python
from src.portfolio import AllocationMethod

# Equal weight
config.allocation_method = AllocationMethod.EQUAL_WEIGHT

# Market cap weighted
config.allocation_method = AllocationMethod.MARKET_CAP_WEIGHT

# Risk Parity (equal risk contribution)
config.allocation_method = AllocationMethod.RISK_PARITY

# Inverse volatility
config.allocation_method = AllocationMethod.INVERSE_VOLATILITY

# Minimum variance
config.allocation_method = AllocationMethod.MINIMUM_VARIANCE

# Maximum Sharpe
config.allocation_method = AllocationMethod.MAXIMUM_SHARPE

# Hierarchical Risk Parity
config.allocation_method = AllocationMethod.HRP
```

### 10.3 Portfolio Backtesting

```python
# Load multi-asset data
portfolio_data = {
    'BTC/USDT': pd.read_csv('data/btc.csv', index_col='timestamp', parse_dates=True),
    'ETH/USDT': pd.read_csv('data/eth.csv', index_col='timestamp', parse_dates=True),
    'SOL/USDT': pd.read_csv('data/sol.csv', index_col='timestamp', parse_dates=True),
}

# Run backtest
result = manager.backtest(portfolio_data)

# Results
print(f"Portfolio Return: {result.metrics['total_return']:.2f}%")
print(f"Portfolio Sharpe: {result.metrics['sharpe_ratio']:.2f}")
print(f"Portfolio Volatility: {result.metrics['volatility']:.2f}%")

# Allocation history
print(result.allocation_history.tail())
```

### 10.4 Rebalancing

```python
# Get current allocation
current_allocation = manager.get_current_allocation()
print(f"Current allocation: {current_allocation}")

# Get target allocation
target_allocation = manager.get_target_allocation()
print(f"Target allocation: {target_allocation}")

# Calculate required rebalancing trades
rebalance_trades = manager.calculate_rebalance_trades()
for trade in rebalance_trades:
    print(f"{trade['action']} {trade['quantity']} {trade['symbol']}")

# Execute rebalancing
await manager.execute_rebalance()
```

---

## 11. Risk Management

### 11.1 Configuration

```python
from src.risk_management import RiskManager, RiskConfig

config = RiskConfig(
    max_position_size_pct=0.10,      # Max 10% per position
    max_portfolio_risk_pct=0.20,     # Max 20% total risk
    daily_loss_limit_pct=0.05,       # Max 5% daily loss
    max_drawdown_pct=0.15,           # Max 15% drawdown
    max_correlation=0.7              # Max correlation between positions
)

risk_manager = RiskManager(config)
```

### 11.2 Position Sizing

```python
from src.risk_management import PositionSizer, SizingMethod

sizer = PositionSizer(
    method=SizingMethod.FIXED_FRACTIONAL,
    risk_per_trade=0.02  # 2% of capital
)

# Calculate position size
position_size = sizer.calculate(
    capital=10000,
    entry_price=50000,
    stop_loss=49000
)
print(f"Position size: {position_size:.4f} BTC")

# Kelly Criterion
kelly_sizer = PositionSizer(
    method=SizingMethod.KELLY_CRITERION,
    win_rate=0.55,
    avg_win=100,
    avg_loss=80
)
kelly_size = kelly_sizer.calculate(capital=10000)
print(f"Kelly suggests: {kelly_size:.2f}% of capital")
```

### 11.3 Trade Assessment

```python
# Assess if a trade meets risk rules
assessment = risk_manager.assess_trade(
    symbol="BTC/USDT",
    side="BUY",
    quantity=0.1,
    entry_price=50000,
    stop_loss=49000
)

if assessment.approved:
    print("Trade approved")
    print(f"Risk: ${assessment.risk_amount:.2f}")
    print(f"% of capital: {assessment.risk_pct:.2f}%")
else:
    print(f"Trade rejected: {assessment.rejection_reason}")
```

### 11.4 Correlation Monitoring

```python
from src.risk_management import CorrelationManager

corr_manager = CorrelationManager()

# Update with recent data
corr_manager.update(portfolio_data)

# Get correlation matrix
corr_matrix = corr_manager.get_correlation_matrix()
print(corr_matrix)

# Detect risk concentration
clusters = corr_manager.get_correlation_clusters(threshold=0.7)
print(f"High correlation clusters: {clusters}")
```

---

## 12. Stress Testing

### 12.1 Monte Carlo

```python
from src.stress_testing import StressTester, MonteCarloSimulator

tester = StressTester()

# Monte Carlo simulation
mc_result = tester.monte_carlo(
    strategy=strategy,
    n_simulations=1000,
    n_periods=252  # 1 year
)

print(f"Mean return: {mc_result.mean_return:.2f}%")
print(f"5th percentile return: {mc_result.percentile_5:.2f}%")
print(f"95th percentile return: {mc_result.percentile_95:.2f}%")
print(f"Mean max drawdown: {mc_result.mean_max_drawdown:.2f}%")
print(f"Ruin probability (<-50%): {mc_result.ruin_probability:.2f}%")
```

### 12.2 Scenario Analysis

```python
from src.stress_testing import ScenarioAnalyzer

analyzer = ScenarioAnalyzer()

# Predefined scenarios
scenarios = {
    'bull_market': {'return_modifier': 1.5, 'volatility_modifier': 0.8},
    'bear_market': {'return_modifier': -0.5, 'volatility_modifier': 1.5},
    'crash': {'return_modifier': -0.3, 'volatility_modifier': 3.0},
    'low_volatility': {'return_modifier': 1.0, 'volatility_modifier': 0.5}
}

results = analyzer.run_scenarios(strategy, scenarios)

for scenario, result in results.items():
    print(f"\n{scenario}:")
    print(f"  Return: {result['total_return']:.2f}%")
    print(f"  Max DD: {result['max_drawdown']:.2f}%")
    print(f"  Sharpe: {result['sharpe_ratio']:.2f}")
```

### 12.3 Sensitivity Analysis

```python
from src.stress_testing import SensitivityAnalyzer

sensitivity = SensitivityAnalyzer()

# Analyze parameter sensitivity
result = sensitivity.analyze(
    strategy_class=MyStrategy,
    data=data,
    parameter='param1',
    range_pct=0.20  # +/- 20%
)

print(f"Sensitivity of {result.parameter}:")
print(f"  Impact on Sharpe: {result.sharpe_sensitivity:.4f}")
print(f"  Impact on Return: {result.return_sensitivity:.4f}")

# Plot
sensitivity.plot_sensitivity(result)
```

---

## 13. REST and WebSocket API

### 13.1 Starting the Server

```bash
cd StrategyTrader
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```

### 13.2 Available REST Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/api/v1/strategies` | GET | List registered strategies |
| `/api/v1/trades/{strategy_id}` | GET | Get strategy trades |
| `/api/v1/performance/{strategy_id}` | GET | Performance metrics |
| `/api/v1/equity/{strategy_id}` | GET | Equity curve |
| `/api/v1/backtest/run` | POST | Run backtest |
| `/api/v1/backtest/optimize` | POST | Run optimization |
| `/api/v1/paper-trading/sessions` | GET | List sessions |
| `/api/v1/paper-trading/sessions` | POST | Create session |
| `/api/v1/paper-trading/sessions/{id}/orders` | POST | Submit order |
| `/api/v1/jobs` | GET | List jobs |
| `/api/v1/jobs/{id}` | GET | Get job status |

### 13.3 Usage Examples

```bash
# Health check
curl http://localhost:8000/

# List strategies
curl http://localhost:8000/api/v1/strategies

# Get trades (with filters)
curl "http://localhost:8000/api/v1/trades/strategy1?start_time=2024-01-01&profitable_only=true"

# Get performance
curl http://localhost:8000/api/v1/performance/strategy1

# Get equity curve
curl http://localhost:8000/api/v1/equity/strategy1

# Run backtest
curl -X POST http://localhost:8000/api/v1/backtest/run \
  -H "Content-Type: application/json" \
  -d '{"strategy": "ma_crossover", "symbol": "BTC/USDT", "start": "2024-01-01", "end": "2024-06-01"}'
```

### 13.4 WebSocket API

```python
import asyncio
import websockets
import json

async def subscribe_to_updates():
    uri = "ws://localhost:8000/ws"

    async with websockets.connect(uri) as websocket:
        # Subscribe to channels
        await websocket.send(json.dumps({
            "action": "subscribe",
            "channels": ["trades", "equity", "positions"]
        }))

        # Receive updates
        while True:
            message = await websocket.recv()
            data = json.loads(message)
            print(f"Update: {data}")

# Run
asyncio.run(subscribe_to_updates())
```

### 13.5 Interactive Documentation

Swagger UI available at: `http://localhost:8000/docs`

ReDoc available at: `http://localhost:8000/redoc`

### 13.6 Register Strategy via API

```python
import requests

# From your Python code
from src.api import register_strategy

strategy = MyStrategy()
strategy.load_data(data)
strategy.backtest()

register_strategy("my_strategy", strategy)

# Now accessible via API
# GET /api/v1/trades/my_strategy
```

---

## 14. Job Manager

The Job Manager allows running long tasks asynchronously in the background.

### 14.1 Creating Jobs

```python
from src.job_manager import get_job_manager, JobType

manager = get_job_manager()

# Create backtest job
job_id = await manager.create_job(
    job_type=JobType.BACKTEST,
    params={
        'strategy_class': 'MyStrategy',
        'symbol': 'BTC/USDT',
        'start_date': '2024-01-01',
        'end_date': '2024-06-01'
    }
)

print(f"Job created: {job_id}")
```

### 14.2 Monitoring Progress

```python
# Get job status
job = await manager.get_job(job_id)

print(f"Status: {job.status}")
print(f"Progress: {job.progress.percentage}%")
print(f"Current step: {job.progress.current_step}")

# Wait for completion
result = await manager.wait_for_completion(job_id)
print(f"Result: {result}")
```

### 14.3 Job Types

| Type | Description |
|------|-------------|
| `BACKTEST` | Run strategy backtest |
| `OPTIMIZATION` | Run parameter optimization |
| `DATA_EXTRACTION` | Extract historical data |
| `STRESS_TEST` | Run stress testing |

### 14.4 Via REST API

```bash
# Create job
curl -X POST http://localhost:8000/api/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{"type": "backtest", "params": {...}}'

# Get job status
curl http://localhost:8000/api/v1/jobs/{job_id}

# Cancel job
curl -X DELETE http://localhost:8000/api/v1/jobs/{job_id}
```

---

## 15. FAQ

### Q: How do I avoid overfitting?

**A:** Use walk-forward validation and compare IS vs OOS metrics:

```python
results = optimizer.walk_forward_optimize(n_splits=5)
if results.oos_sharpe < results.is_sharpe * 0.6:
    print("Possible overfitting - reduce complexity")
```

### Q: What is the best optimization method?

**A:** It depends on the case:
- **Few parameters (<5)**: Grid Search
- **Many parameters**: Bayesian or Genetic
- **Robust validation**: Walk-Forward

### Q: How do I handle partially filled orders?

**A:** The broker bridge handles this automatically:

```python
report = await executor.submit_order(order)
if report.status == OrderStatus.PARTIAL:
    print(f"Filled: {report.filled_quantity} of {order.quantity}")
```

### Q: Can I use multiple strategies simultaneously?

**A:** Yes, with the PortfolioManager:

```python
strategies = {
    'momentum': MomentumStrategy(),
    'mean_reversion': MeanReversionStrategy()
}
manager = PortfolioManager(strategies, allocation='equal')
```

### Q: How do I configure alerts?

**A:** Implement callbacks in your strategy:

```python
def on_trade(self, trade):
    if trade.pnl < -100:
        send_alert(f"Significant loss: ${trade.pnl}")
```

### Q: Does it support futures and options?

**A:** Yes, via Interactive Brokers:

```python
# Futures
order = BrokerOrder(symbol="ES2403", ...)  # E-mini S&P 500

# Options (OCC format)
order = BrokerOrder(symbol="AAPL240315C00175000", ...)
```

### Q: How do I run long optimizations?

**A:** Use the Job Manager:

```python
job_id = await manager.create_job(
    job_type=JobType.OPTIMIZATION,
    params={'n_iterations': 1000, ...}
)
# Check progress via API or monitor directly
```

---

## Support

- **Issues**: https://github.com/Pliperkiller/Trad-loop/issues
- **Documentation**: `/docs/` in the repository

---

*User Guide v1.0 - Trad-Loop*
