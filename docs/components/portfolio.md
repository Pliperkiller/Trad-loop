# Portfolio Management

The **portfolio** module provides tools for multi-asset portfolio management with various allocation and rebalancing methods.

## Architecture

```
portfolio/
├── portfolio_manager.py   # Main orchestrator
├── models.py              # PortfolioConfig, PortfolioState
├── allocator.py           # Allocation methods
├── rebalancer.py          # Rebalancing strategies
├── backtester.py          # Portfolio backtesting
└── metrics.py             # Portfolio metrics
```

## Basic Usage

### Configuration

```python
from src.portfolio import PortfolioManager, PortfolioConfig, AllocationMethod

config = PortfolioConfig(
    initial_capital=100000,
    symbols=['BTC/USDT', 'ETH/USDT', 'SOL/USDT'],
    allocation_method=AllocationMethod.RISK_PARITY,
    rebalance_frequency='monthly',
    min_position_weight=0.05,     # Min 5% per asset
    max_position_weight=0.40,     # Max 40% per asset
    transaction_cost=0.001        # 0.1% commission
)

manager = PortfolioManager(config)
```

## Allocation Methods

### Equal Weight

```python
config.allocation_method = AllocationMethod.EQUAL_WEIGHT

# Result: 33.3% BTC, 33.3% ETH, 33.3% SOL
```

### Market Cap Weight

```python
config.allocation_method = AllocationMethod.MARKET_CAP_WEIGHT

# Weights based on market capitalization
# Requires market cap data
manager.set_market_caps({
    'BTC/USDT': 800_000_000_000,
    'ETH/USDT': 300_000_000_000,
    'SOL/USDT': 50_000_000_000
})
```

### Risk Parity

Each asset contributes equally to total risk.

```python
config.allocation_method = AllocationMethod.RISK_PARITY

# More volatile assets receive less weight
# to equalize risk contribution
```

### Inverse Volatility

Weight inverse to volatility.

```python
config.allocation_method = AllocationMethod.INVERSE_VOLATILITY

# volatility_weight[i] = 1/vol[i] / sum(1/vol)
```

### Minimum Variance

Markowitz optimization for minimum variance.

```python
config.allocation_method = AllocationMethod.MINIMUM_VARIANCE

# Minimizes: w' * Cov * w
# Subject to: sum(w) = 1, w >= 0
```

### Maximum Sharpe

Optimization for maximum Sharpe Ratio.

```python
config.allocation_method = AllocationMethod.MAXIMUM_SHARPE

# Maximizes: (w' * returns - rf) / sqrt(w' * Cov * w)
```

### Risk Budgeting

Allocation with specific risk budget.

```python
from src.portfolio import RiskBudgetAllocator

allocator = RiskBudgetAllocator()
weights = allocator.allocate(
    returns=historical_returns,
    risk_budget={
        'BTC/USDT': 0.5,   # 50% of risk
        'ETH/USDT': 0.3,   # 30% of risk
        'SOL/USDT': 0.2    # 20% of risk
    }
)
```

### Hierarchical Risk Parity (HRP)

Advanced method based on hierarchical clustering.

```python
config.allocation_method = AllocationMethod.HRP

# 1. Calculate correlation matrix
# 2. Hierarchical clustering
# 3. Assign weights recursively
```

## Rebalancing

### By Frequency

```python
config.rebalance_frequency = 'daily'    # Daily
config.rebalance_frequency = 'weekly'   # Weekly
config.rebalance_frequency = 'monthly'  # Monthly
config.rebalance_frequency = 'quarterly'# Quarterly
```

### By Threshold (Drift)

```python
from src.portfolio import ThresholdRebalancer

rebalancer = ThresholdRebalancer(
    drift_threshold=0.05  # Rebalance if drift > 5%
)

# Drift is calculated as:
# drift = max(|actual_weight - target_weight|)
```

### By Volatility

```python
from src.portfolio import VolatilityTargetRebalancer

rebalancer = VolatilityTargetRebalancer(
    target_volatility=0.15,  # 15% annualized
    lookback=20              # Calculation window
)
```

## Portfolio Backtesting

```python
# Load multi-asset data
portfolio_data = {
    'BTC/USDT': pd.read_csv('btc.csv', index_col='timestamp', parse_dates=True),
    'ETH/USDT': pd.read_csv('eth.csv', index_col='timestamp', parse_dates=True),
    'SOL/USDT': pd.read_csv('sol.csv', index_col='timestamp', parse_dates=True),
}

# Run backtest
result = manager.backtest(portfolio_data)

# Results
print(f"Total Return: {result.metrics['total_return']:.2f}%")
print(f"Annualized Return: {result.metrics['annualized_return']:.2f}%")
print(f"Volatility: {result.metrics['volatility']:.2f}%")
print(f"Sharpe Ratio: {result.metrics['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {result.metrics['max_drawdown']:.2f}%")
print(f"Calmar Ratio: {result.metrics['calmar_ratio']:.2f}")

# Equity curve
result.equity_curve.plot(title='Portfolio Equity')

# Allocation history
print(result.allocation_history.tail())

# Rebalance dates
print(f"Rebalances: {len(result.rebalance_dates)}")
```

## Portfolio Metrics

```python
from src.portfolio import PortfolioMetrics

metrics = PortfolioMetrics()

# Calculate all metrics
report = metrics.calculate(result.equity_curve, result.weights_history)

# Basic metrics
print(f"Return: {report['total_return']:.2f}%")
print(f"Volatility: {report['volatility']:.2f}%")
print(f"Sharpe: {report['sharpe_ratio']:.2f}")

# Diversification metrics
print(f"Diversification Ratio: {report['diversification_ratio']:.2f}")
print(f"Concentration (HHI): {report['hhi']:.4f}")

# Risk contribution by asset
for asset, contribution in report['risk_contribution'].items():
    print(f"{asset}: {contribution*100:.1f}%")

# Correlation matrix
print("\nCorrelation Matrix:")
print(report['correlation_matrix'])
```

## Live Operations

### Get Current Allocation

```python
# Current allocation
current = manager.get_current_allocation()
print(f"Current weights: {current}")

# Target allocation
target = manager.get_target_allocation()
print(f"Target weights: {target}")

# Drift
drift = manager.calculate_drift()
print(f"Current drift: {drift:.2%}")
```

### Calculate Rebalance Trades

```python
# Required trades
trades = manager.calculate_rebalance_trades(current_prices)

for trade in trades:
    print(f"{trade['action']} {trade['quantity']:.4f} {trade['symbol']}")
    print(f"  Value: ${trade['value']:.2f}")
```

### Execute Rebalance

```python
# With UnifiedExecutor
from src.broker_bridge import UnifiedExecutor

executor = UnifiedExecutor()
# ... configure brokers

# Execute rebalance
await manager.execute_rebalance(executor)
```

## Advanced Analysis

### Efficient Frontier

```python
from src.portfolio import EfficientFrontier

ef = EfficientFrontier(returns=historical_returns)

# Calculate frontier
frontier = ef.calculate(n_points=100)

# Plot
ef.plot_frontier()

# Minimum variance portfolio
min_var = ef.get_minimum_variance_portfolio()

# Maximum Sharpe portfolio
max_sharpe = ef.get_maximum_sharpe_portfolio()
```

### Portfolio Stress Testing

```python
from src.portfolio import PortfolioStressTester

stress_tester = PortfolioStressTester(manager)

# Scenarios
scenarios = {
    'crypto_crash': {'BTC/USDT': -0.30, 'ETH/USDT': -0.40, 'SOL/USDT': -0.50},
    'correlation_spike': {'correlation_increase': 0.3},
    'volatility_shock': {'volatility_multiplier': 2.0}
}

results = stress_tester.run_scenarios(scenarios)

for scenario, result in results.items():
    print(f"\n{scenario}:")
    print(f"  Portfolio Loss: {result['portfolio_loss']:.2%}")
    print(f"  Worst Asset: {result['worst_asset']}")
```

### Attribution Analysis

```python
from src.portfolio import AttributionAnalyzer

analyzer = AttributionAnalyzer()

# Contribution analysis
attribution = analyzer.calculate(
    portfolio_returns=result.returns,
    weights_history=result.weights_history,
    benchmark_returns=benchmark_returns
)

print("Return Attribution:")
for asset, contrib in attribution['return_contribution'].items():
    print(f"  {asset}: {contrib:.2%}")

print(f"\nActive Return: {attribution['active_return']:.2%}")
print(f"Tracking Error: {attribution['tracking_error']:.2%}")
print(f"Information Ratio: {attribution['information_ratio']:.2f}")
```

## API Reference

### PortfolioManager

| Method | Description |
|--------|-------------|
| `backtest(data)` | Run backtest |
| `get_current_allocation()` | Current allocation |
| `get_target_allocation()` | Target allocation |
| `calculate_drift()` | Calculate drift |
| `calculate_rebalance_trades(prices)` | Required trades |
| `execute_rebalance(executor)` | Execute rebalance |

### PortfolioConfig

| Field | Description |
|-------|-------------|
| `initial_capital` | Initial capital |
| `symbols` | List of symbols |
| `allocation_method` | Allocation method |
| `rebalance_frequency` | Rebalance frequency |
| `min_position_weight` | Minimum weight per position |
| `max_position_weight` | Maximum weight per position |
| `transaction_cost` | Transaction cost |

### PortfolioResult

| Field | Description |
|-------|-------------|
| `equity_curve` | Equity time series |
| `weights_history` | Weight history |
| `allocation_history` | Allocation history |
| `rebalance_dates` | Rebalance dates |
| `metrics` | Metrics dictionary |
| `trades` | List of executed trades |

## Tests

```bash
pytest tests/portfolio/ -v
```

## Related Documentation

- [Risk Management](risk_management.md) - Risk controls
- [Optimizers](optimizers.md) - Parameter optimization
- [Stress Testing](stress_testing.md) - Stress tests
