# Paper Trading

The **paper_trading** module provides a complete simulation engine for testing strategies with advanced orders in real-time or with historical data.

## Architecture

```
paper_trading/
├── engine.py              # Main engine
├── models.py              # Order, Position, TradeRecord
├── config.py              # Configuration
├── order_simulator.py     # Execution simulator
├── position_manager.py    # Position management
├── performance_tracker.py # Real-time metrics
├── realtime_feed.py       # Data feeds
├── orders/                # Advanced order types
│   ├── base.py
│   ├── enums.py
│   ├── conditional_orders.py
│   ├── dynamic_orders.py
│   ├── execution_algos.py
│   └── risk_control.py
└── simulators/            # Specialized simulators
    ├── trailing_simulator.py
    ├── algo_simulator.py
    └── composite_simulator.py
```

## Basic Usage

### Configuration

```python
from src.paper_trading import PaperTradingEngine, PaperTradingConfig

config = PaperTradingConfig(
    initial_capital=10000,
    commission_pct=0.001,      # 0.1%
    slippage_pct=0.0005,       # 0.05%
    enable_shorting=True,
    max_positions=10,
    margin_requirement=0.5     # 50% margin
)

engine = PaperTradingEngine(config)
```

### Realtime Strategy

```python
from src.paper_trading import RealtimeStrategy

class MyStrategy(RealtimeStrategy):
    def __init__(self):
        super().__init__()
        self.sma_20 = []

    def on_start(self):
        """Initialization."""
        print("Strategy started")

    def on_candle(self, candle):
        """Process new candle."""
        self.sma_20.append(candle.close)
        if len(self.sma_20) > 20:
            self.sma_20.pop(0)

        sma = sum(self.sma_20) / len(self.sma_20)

        if candle.close > sma and not self.has_position():
            self.buy(candle.close, quantity=0.1)
        elif candle.close < sma and self.has_position():
            self.close_all_positions()

    def on_tick(self, symbol, price):
        """Process tick (optional)."""
        pass

    def on_stop(self):
        """Finalization."""
        print("Strategy stopped")
```

### Run

```python
# Register strategy
engine.register_strategy(MyStrategy())

# Backtest with historical data
result = engine.run_backtest(historical_data)

# Or run live
await engine.run_live()
```

## Order Types (17+)

### Basic Orders

```python
from src.paper_trading.orders import MarketOrder, LimitOrder, StopOrder

# Market
market = MarketOrder(
    symbol="BTC/USDT",
    side="BUY",
    quantity=0.1
)

# Limit
limit = LimitOrder(
    symbol="BTC/USDT",
    side="BUY",
    quantity=0.1,
    price=50000
)

# Stop Loss
stop = StopOrder(
    symbol="BTC/USDT",
    side="SELL",
    quantity=0.1,
    stop_price=49000
)
```

### Conditional Orders

```python
from src.paper_trading.orders.conditional_orders import (
    IfTouchedOrder,
    OCOOrder,
    OTOCOOrder
)

# If-Touched: Trigger when price hits level
if_touched = IfTouchedOrder(
    symbol="BTC/USDT",
    side="BUY",
    quantity=0.1,
    trigger_price=50000,
    order_type="LIMIT",
    limit_price=50100
)

# OCO: One Cancels Other
oco = OCOOrder(
    symbol="BTC/USDT",
    quantity=0.1,
    take_profit_price=52000,  # Take profit
    stop_loss_price=48000     # Stop loss
)

# OTOCO: One Triggers OCO
otoco = OTOCOOrder(
    entry_order=LimitOrder(side="BUY", price=50000, quantity=0.1),
    take_profit_price=52000,
    stop_loss_price=48000
)
```

### Dynamic Orders

```python
from src.paper_trading.orders.dynamic_orders import (
    TrailingStopOrder,
    TrailingStopLimitOrder
)

# Trailing Stop
trailing = TrailingStopOrder(
    symbol="BTC/USDT",
    side="SELL",
    quantity=0.1,
    trail_percent=0.02  # 2%
)

# Trailing Stop with Limit
trailing_limit = TrailingStopLimitOrder(
    symbol="BTC/USDT",
    side="SELL",
    quantity=0.1,
    trail_percent=0.02,
    limit_offset=0.001  # 0.1% below trigger
)
```

### Risk Control Orders

```python
from src.paper_trading.orders.risk_control import (
    ProfitTargetOrder,
    BreakEvenOrder,
    ScaleOutOrder,
    TimedExitOrder
)

# Profit Target with trailing
profit_target = ProfitTargetOrder(
    position_id="pos123",
    profit_target_pct=0.05,     # 5% profit
    enable_trailing=True,
    trail_after_pct=0.03       # Trail after 3%
)

# Break Even
break_even = BreakEvenOrder(
    position_id="pos123",
    trigger_profit_pct=0.02,   # Move SL to break even at 2%
    lock_profit_pct=0.001      # Lock 0.1% profit
)

# Scale Out (partial exit)
scale_out = ScaleOutOrder(
    position_id="pos123",
    levels=[
        {'profit_pct': 0.02, 'close_pct': 0.25},  # 25% at 2%
        {'profit_pct': 0.04, 'close_pct': 0.25},  # 25% at 4%
        {'profit_pct': 0.06, 'close_pct': 0.50}   # 50% at 6%
    ]
)

# Timed Exit
timed_exit = TimedExitOrder(
    position_id="pos123",
    exit_after_minutes=60  # Close after 1 hour
)
```

### Execution Algorithms

```python
from src.paper_trading.orders.execution_algos import (
    TWAPOrder,
    VWAPOrder,
    IcebergOrder,
    POVOrder
)

# TWAP (Time Weighted Average Price)
twap = TWAPOrder(
    symbol="BTC/USDT",
    side="BUY",
    total_quantity=1.0,
    duration_minutes=60,
    num_slices=12  # 12 orders in 60 min
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
    visible_quantity=1.0,
    variance_pct=0.1  # +/- 10% variance
)

# POV (Percentage of Volume)
pov = POVOrder(
    symbol="BTC/USDT",
    side="BUY",
    total_quantity=5.0,
    target_pct=0.05  # 5% of volume
)
```

### Composite Orders

```python
from src.paper_trading.orders import BracketOrder, MultiLegOrder

# Bracket (Entry + SL + TP)
bracket = BracketOrder(
    symbol="BTC/USDT",
    side="BUY",
    quantity=0.1,
    entry_price=50000,
    stop_loss=49000,
    take_profit=52000
)

# Multi-Leg (multiple related orders)
multi_leg = MultiLegOrder(
    legs=[
        LimitOrder(symbol="BTC/USDT", side="BUY", quantity=0.05, price=49500),
        LimitOrder(symbol="BTC/USDT", side="BUY", quantity=0.05, price=49000),
    ],
    fill_requirement="ALL"  # "ALL", "ANY", "PARTIAL"
)
```

## Specialized Simulators

The paper trading engine includes specialized simulators for complex order types.

### Trailing Stop Simulator

Handles trailing stop orders with real-time price tracking.

```python
from src.paper_trading.simulators.trailing_simulator import TrailingStopSimulator

simulator = TrailingStopSimulator()

# Register trailing order
order = TrailingStopOrder(
    symbol="BTC/USDT",
    side="SELL",
    quantity=0.1,
    trail_percent=0.02
)
simulator.register_order(order)

# Update with price changes
simulator.on_price_update("BTC/USDT", 51000)  # Price rises, stop follows
simulator.on_price_update("BTC/USDT", 52000)  # Price rises more
simulator.on_price_update("BTC/USDT", 50960)  # Price drops 2%, stop triggers
```

### Algo Execution Simulator

Simulates execution algorithms (TWAP, VWAP, Iceberg).

```python
from src.paper_trading.simulators.algo_simulator import AlgoSimulator

simulator = AlgoSimulator()

# TWAP execution
twap = TWAPOrder(
    symbol="BTC/USDT",
    side="BUY",
    total_quantity=1.0,
    duration_minutes=60,
    num_slices=12
)

simulator.start_algo(twap)

# Simulator slices orders over time
# Each slice: 1.0 / 12 = 0.0833 BTC every 5 minutes
```

### Composite Simulator

Orchestrates multiple specialized simulators.

```python
from src.paper_trading.simulators.composite_simulator import CompositeSimulator

# Combines all simulators
composite = CompositeSimulator(
    trailing_simulator=TrailingStopSimulator(),
    algo_simulator=AlgoSimulator(),
    # ... other simulators
)

# Automatically routes orders to appropriate simulator
composite.submit_order(trailing_order)  # -> TrailingStopSimulator
composite.submit_order(twap_order)      # -> AlgoSimulator
```

## Position Manager

```python
from src.paper_trading import PositionManager

pm = engine.position_manager

# Get positions
positions = pm.get_all_positions()
btc_position = pm.get_position("BTC/USDT")

# Position metrics
if btc_position:
    print(f"Entry: {btc_position.entry_price}")
    print(f"Quantity: {btc_position.quantity}")
    print(f"Unrealized PnL: ${btc_position.unrealized_pnl:.2f}")
    print(f"ROI: {btc_position.roi_pct:.2f}%")

# Close position
pm.close_position("BTC/USDT", price=51000, reason="manual")
```

## Performance Tracker

```python
from src.paper_trading import PerformanceTracker

tracker = engine.performance_tracker

# Real-time metrics
metrics = tracker.get_realtime_metrics()
print(f"Equity: ${metrics['equity']:.2f}")
print(f"Unrealized PnL: ${metrics['unrealized_pnl']:.2f}")
print(f"Win Rate: {metrics['win_rate']:.2f}%")
print(f"Current Drawdown: {metrics['current_drawdown']:.2f}%")

# Trade history
trades = tracker.get_trade_history()
```

## Data Feeds

### Mock Feed (Backtest)

```python
from src.paper_trading.realtime_feed import MockFeedManager

feed = MockFeedManager()
feed.load_data(historical_data)

engine.set_feed(feed)
engine.run_backtest()
```

### Realtime Feed (Live)

```python
from src.paper_trading.realtime_feed import RealtimeFeedManager
from src.paper_trading.websocket_handlers import BinanceWebSocketHandler

# Configure WebSocket
ws_handler = BinanceWebSocketHandler(
    symbols=["BTC/USDT", "ETH/USDT"],
    timeframe="1m"
)

feed = RealtimeFeedManager(ws_handler)
engine.set_feed(feed)

await engine.run_live()
```

## Execution Simulation

### Slippage

```python
from src.paper_trading import SlippageModel

# Fixed slippage
config.slippage_model = SlippageModel.FIXED
config.slippage_pct = 0.0005

# Volume-based slippage
config.slippage_model = SlippageModel.VOLUME_BASED
config.slippage_base = 0.0001
config.slippage_volume_impact = 0.0001
```

### Commissions

```python
# Percentage commission
config.commission_type = "percentage"
config.commission_pct = 0.001

# Fixed commission
config.commission_type = "fixed"
config.commission_fixed = 1.0  # $1 per trade
```

## API Reference

### PaperTradingEngine

| Method | Description |
|--------|-------------|
| `register_strategy(strategy)` | Register strategy |
| `run_backtest(data)` | Run backtest |
| `run_live()` | Run live |
| `submit_order(order)` | Submit order |
| `cancel_order(order_id)` | Cancel order |
| `get_positions()` | Get positions |
| `get_balance()` | Get balance |
| `get_metrics()` | Get metrics |

### RealtimeStrategy (ABC)

| Method | Description |
|--------|-------------|
| `on_start()` | Called on start |
| `on_candle(candle)` | Called with each candle |
| `on_tick(symbol, price)` | Called with each tick |
| `on_stop()` | Called on stop |
| `buy(price, quantity, sl?, tp?)` | Buy |
| `sell(price, quantity, sl?, tp?)` | Sell |
| `close_all_positions()` | Close all positions |

### Order Types Summary

| Category | Types |
|----------|-------|
| Basic | Market, Limit, Stop, StopLimit |
| Conditional | IfTouched, OCO, OTOCO |
| Dynamic | TrailingStop, TrailingStopLimit |
| Risk Control | ProfitTarget, BreakEven, ScaleOut, TimedExit |
| Execution | TWAP, VWAP, Iceberg, POV, Hidden |
| Composite | Bracket, MultiLeg |

## Tests

```bash
pytest tests/paper_trading/ -v
pytest src/paper_trading/tests/ -v
```

Coverage: ~220 tests for advanced orders and simulators.

## Related Documentation

- [API Reference](api_reference.md) - Paper Trading API endpoints
- [Broker Bridge](broker_bridge.md) - Live trading execution
- [Risk Management](risk_management.md) - Risk controls
