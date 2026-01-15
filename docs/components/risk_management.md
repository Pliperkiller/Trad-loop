# Risk Management

The **risk_management** module provides professional tools for risk management in algorithmic trading.

## Architecture

```
risk_management/
├── risk_manager.py        # Main orchestrator
├── models.py              # Data models
├── config.py              # Configuration
├── position_sizer.py      # Position sizing
├── risk_limits.py         # Risk limits
└── correlation_manager.py # Correlation management
```

## Basic Usage

### Configuration

```python
from src.risk_management import RiskManager, RiskConfig

config = RiskConfig(
    max_position_size_pct=0.10,      # Max 10% per position
    max_portfolio_risk_pct=0.20,     # Max 20% total risk
    max_correlation=0.7,              # Max correlation
    daily_loss_limit_pct=0.05,       # Max 5% daily loss
    max_drawdown_pct=0.15,           # Max 15% drawdown
    max_consecutive_losses=5,         # Max consecutive losses
    risk_free_rate=0.02              # Risk-free rate
)

risk_manager = RiskManager(config)
```

## Position Sizing

### Fixed Fractional

Risk a fixed percentage of capital per trade.

```python
from src.risk_management import PositionSizer, SizingMethod

sizer = PositionSizer(
    method=SizingMethod.FIXED_FRACTIONAL,
    risk_per_trade=0.02  # 2% of capital
)

# Calculate size
position_size = sizer.calculate(
    capital=10000,
    entry_price=50000,
    stop_loss=49000  # 2% price risk
)

# Result: quantity that risks 2% of capital
print(f"Position size: {position_size:.4f} BTC")
print(f"Risk amount: ${10000 * 0.02:.2f}")
```

### Kelly Criterion

Optimal sizing based on edge and probabilities.

```python
sizer = PositionSizer(
    method=SizingMethod.KELLY_CRITERION,
    win_rate=0.55,        # 55% win rate
    avg_win=100,          # Average win $100
    avg_loss=80,          # Average loss $80
    kelly_fraction=0.5    # Half Kelly (more conservative)
)

kelly_pct = sizer.calculate_kelly()
print(f"Kelly suggests: {kelly_pct:.2%} of capital")

# Kelly formula:
# f = (p * b - q) / b
# where p = win_rate, q = 1-p, b = avg_win/avg_loss
```

### Optimal-f

Based on the largest historical loss.

```python
sizer = PositionSizer(
    method=SizingMethod.OPTIMAL_F,
    largest_loss=500,     # Largest historical loss
    fraction=0.25         # 25% of optimal-f
)

optimal_size = sizer.calculate(capital=10000)
```

### ATR-Based

Sizing based on volatility (ATR).

```python
sizer = PositionSizer(
    method=SizingMethod.ATR_BASED,
    atr_multiplier=2.0,   # Stop at 2x ATR
    risk_per_trade=0.02
)

position_size = sizer.calculate(
    capital=10000,
    entry_price=50000,
    atr=1000  # Current ATR
)
```

## Risk Limits

### Position Limits

```python
from src.risk_management import RiskLimitChecker

checker = RiskLimitChecker(config)

# Check position limit
can_open = checker.check_position_limit(
    symbol="BTC/USDT",
    quantity=0.5,
    price=50000,
    current_portfolio_value=100000
)

if not can_open['approved']:
    print(f"Rejected: {can_open['reason']}")
```

### Exposure Limits

```python
# Check total exposure
exposure_check = checker.check_exposure_limit(
    current_positions=[...],
    portfolio_value=100000
)

print(f"Current Exposure: {exposure_check['current_exposure']:.2%}")
print(f"Limit: {exposure_check['limit']:.2%}")
print(f"Available: {exposure_check['available']:.2%}")
```

### Daily Loss Limit

```python
# Check daily limit
daily_check = checker.check_daily_loss_limit(
    today_pnl=-400,
    initial_capital=10000
)

if daily_check['limit_reached']:
    print("ALERT: Daily loss limit reached")
    print(f"Loss: {daily_check['current_loss']:.2%}")
```

### Drawdown Protection

```python
# Check drawdown
dd_check = checker.check_drawdown_limit(
    current_equity=8500,
    peak_equity=10000
)

if dd_check['limit_breached']:
    print("ALERT: Maximum drawdown exceeded")
    print(f"Current DD: {dd_check['current_drawdown']:.2%}")
    print(f"Limit: {dd_check['limit']:.2%}")
```

## Trade Assessment

```python
# Assess a trade before execution
assessment = risk_manager.assess_trade(
    symbol="BTC/USDT",
    side="BUY",
    quantity=0.1,
    entry_price=50000,
    stop_loss=49000,
    take_profit=52000
)

print(f"Approved: {assessment.approved}")
print(f"Risk Amount: ${assessment.risk_amount:.2f}")
print(f"Risk %: {assessment.risk_pct:.2%}")
print(f"Reward/Risk: {assessment.reward_risk_ratio:.2f}")
print(f"Position Value: ${assessment.position_value:.2f}")
print(f"Portfolio Impact: {assessment.portfolio_impact:.2%}")

if not assessment.approved:
    print(f"Rejection Reason: {assessment.rejection_reason}")
    for suggestion in assessment.suggestions:
        print(f"  - {suggestion}")
```

## Correlation Manager

### Correlation Monitoring

```python
from src.risk_management import CorrelationManager

corr_manager = CorrelationManager(
    lookback_period=60,    # 60 periods
    correlation_threshold=0.7
)

# Update with data
corr_manager.update(price_data)

# Get matrix
corr_matrix = corr_manager.get_correlation_matrix()
print(corr_matrix)
```

### Cluster Detection

```python
# Find highly correlated assets
clusters = corr_manager.get_correlation_clusters(threshold=0.7)

for cluster in clusters:
    print(f"Cluster: {cluster}")
    # Example: ['BTC/USDT', 'ETH/USDT'] - high correlation
```

### Concentration Risk

```python
# Assess concentration risk
concentration = corr_manager.calculate_concentration_risk(
    positions=[
        {'symbol': 'BTC/USDT', 'weight': 0.4},
        {'symbol': 'ETH/USDT', 'weight': 0.3},
        {'symbol': 'SOL/USDT', 'weight': 0.3}
    ]
)

print(f"Effective Positions: {concentration['effective_positions']:.1f}")
print(f"Concentration Risk: {concentration['risk_score']:.2f}")

if concentration['risk_score'] > 0.7:
    print("WARNING: High risk concentration")
```

### Regime Change Detection

```python
# Detect correlation changes
regime_change = corr_manager.detect_regime_change(
    current_window=20,
    historical_window=60
)

if regime_change['detected']:
    print(f"Regime change detected!")
    print(f"Correlation shift: {regime_change['shift']:.2f}")
    print(f"Affected pairs: {regime_change['affected_pairs']}")
```

## Value at Risk (VaR)

### Historical VaR

```python
from src.risk_management import VaRCalculator

var_calc = VaRCalculator()

# Historical VaR
var_95 = var_calc.historical_var(
    returns=portfolio_returns,
    confidence=0.95
)
print(f"VaR 95%: {var_95:.2%}")

# With specific window
var_99 = var_calc.historical_var(
    returns=portfolio_returns[-252:],  # Last year
    confidence=0.99
)
print(f"VaR 99%: {var_99:.2%}")
```

### Parametric VaR

```python
# Parametric VaR (assumes normality)
var_parametric = var_calc.parametric_var(
    mean=portfolio_returns.mean(),
    std=portfolio_returns.std(),
    confidence=0.95
)
print(f"Parametric VaR 95%: {var_parametric:.2%}")
```

### CVaR (Expected Shortfall)

```python
# CVaR - expected loss when VaR is exceeded
cvar = var_calc.conditional_var(
    returns=portfolio_returns,
    confidence=0.95
)
print(f"CVaR 95%: {cvar:.2%}")

# Interpretation: "In the worst 5% of cases,
# the average loss is {cvar}%"
```

### Portfolio VaR

```python
# VaR considering correlations
portfolio_var = var_calc.portfolio_var(
    weights={'BTC/USDT': 0.5, 'ETH/USDT': 0.3, 'SOL/USDT': 0.2},
    returns_data=multi_asset_returns,
    confidence=0.95
)
print(f"Portfolio VaR 95%: {portfolio_var:.2%}")
```

## Risk Dashboard

```python
from src.risk_management import RiskDashboard

dashboard = RiskDashboard(risk_manager)

# Generate report
report = dashboard.generate_report(
    positions=current_positions,
    equity_curve=equity_curve
)

# Key metrics
print("=== RISK DASHBOARD ===")
print(f"Total Exposure: {report['total_exposure']:.2%}")
print(f"Portfolio Beta: {report['portfolio_beta']:.2f}")
print(f"VaR 95%: {report['var_95']:.2%}")
print(f"CVaR 95%: {report['cvar_95']:.2%}")
print(f"Current Drawdown: {report['current_drawdown']:.2%}")
print(f"Correlation Risk: {report['correlation_risk']:.2f}")

# Active alerts
for alert in report['active_alerts']:
    print(f"ALERT: {alert['type']} - {alert['message']}")

# Plot
dashboard.plot_risk_metrics()
```

## Trading Integration

```python
from src.paper_trading import PaperTradingEngine
from src.risk_management import RiskManager

# Configure risk manager
risk_manager = RiskManager(config)

class RiskAwareStrategy(RealtimeStrategy):
    def __init__(self, risk_manager):
        super().__init__()
        self.risk_manager = risk_manager

    def on_candle(self, candle):
        if self.should_buy(candle):
            # Assess risk before trading
            assessment = self.risk_manager.assess_trade(
                symbol=candle.symbol,
                side="BUY",
                entry_price=candle.close,
                stop_loss=candle.close * 0.98
            )

            if assessment.approved:
                # Use suggested position size
                self.buy(
                    price=candle.close,
                    quantity=assessment.suggested_quantity,
                    stop_loss=assessment.stop_loss
                )
            else:
                print(f"Trade rejected: {assessment.rejection_reason}")
```

## API Reference

### RiskManager

| Method | Description |
|--------|-------------|
| `assess_trade(symbol, side, qty, entry, sl)` | Assess trade |
| `check_limits()` | Check all limits |
| `get_available_risk()` | Available risk |
| `update_state(positions, equity)` | Update state |

### PositionSizer

| Method | Description |
|--------|-------------|
| `calculate(capital, entry, sl)` | Calculate size |
| `calculate_kelly()` | Calculate Kelly % |
| `set_method(method)` | Change method |

### RiskConfig

| Field | Description |
|-------|-------------|
| `max_position_size_pct` | Max % per position |
| `max_portfolio_risk_pct` | Max % total risk |
| `daily_loss_limit_pct` | Daily loss limit |
| `max_drawdown_pct` | Max drawdown |
| `max_correlation` | Max correlation |
| `max_consecutive_losses` | Max consecutive losses |

## Tests

```bash
pytest tests/risk_management/ -v
```

## Related Documentation

- [Portfolio](portfolio.md) - Portfolio management
- [Paper Trading](paper_trading.md) - Simulated trading
- [Interfaces](interfaces.md) - IRiskManager protocol
