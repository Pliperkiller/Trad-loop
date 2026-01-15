# Broker Bridge

The **broker_bridge** module provides a unified abstraction layer for trading with multiple brokers (crypto and traditional) through a common interface.

## Architecture

```
broker_bridge/
├── core/
│   ├── enums.py         # BrokerType, AssetClass, OrderType, OrderSide
│   ├── models.py        # BrokerOrder, BrokerPosition, ExecutionReport
│   ├── interfaces.py    # IBrokerAdapter (ABC)
│   └── exceptions.py    # Custom exceptions
├── adapters/
│   ├── ccxt/            # Adapter for 100+ crypto exchanges
│   └── ibkr/            # Adapter for Interactive Brokers
└── execution/
    ├── unified_executor.py   # Main orchestrator
    ├── symbol_router.py      # Automatic routing by symbol
    └── fallback_simulator.py # Simulation for unsupported orders
```

## Basic Usage

### Initialization

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

# Register brokers
executor.register_broker(CCXTBroker("binance", "api_key", "api_secret"))
executor.register_broker(IBKRBroker(port=7497))

# Connect
await executor.connect_all()
```

### Submit Orders

```python
# Crypto order (auto-routed to CCXT)
order = BrokerOrder(
    symbol="BTC/USDT",
    side=OrderSide.BUY,
    order_type=OrderType.LIMIT,
    quantity=0.1,
    price=50000
)
report = await executor.submit_order(order)

# Stock order (auto-routed to IBKR)
order = BrokerOrder(
    symbol="AAPL",
    side=OrderSide.BUY,
    order_type=OrderType.MARKET,
    quantity=10
)
report = await executor.submit_order(order)
```

## Automatic Routing

The `SymbolRouter` automatically determines the appropriate broker:

| Pattern | Broker | Asset Class |
|---------|--------|-------------|
| `BTC/USDT`, `ETH/BTC` | CCXT | CRYPTO |
| `EUR/USD`, `GBP/JPY` | IBKR | FOREX |
| `AAPL`, `MSFT` | IBKR | STOCK |
| `SPX`, `NDX`, `VIX` | IBKR | INDEX |
| `ES2403`, `NQM24` | IBKR | FUTURES |
| `AAPL240315C00175000` | IBKR | OPTIONS |

### Customize Routing

```python
router = executor.get_router()

# Add crypto pattern
router.add_crypto_pattern(r"^MY_TOKEN.*$")

# Add forex currency
router.add_forex_currency("MXN")

# Manual override
router.set_override("SPECIAL", BrokerType.PAPER, AssetClass.STOCK)
```

## Supported Order Types

```python
from src.broker_bridge.core.enums import OrderType

# Basic
OrderType.MARKET
OrderType.LIMIT
OrderType.STOP_LOSS
OrderType.STOP_LIMIT

# Advanced
OrderType.TRAILING_STOP
OrderType.BRACKET
OrderType.OCO
OrderType.ICEBERG

# Algorithmic
OrderType.TWAP
OrderType.VWAP
```

## Broker Capabilities

```python
# Check capabilities
ccxt = executor.get_broker(BrokerType.CCXT)
caps = ccxt.get_capabilities()

print(f"Trailing Stop: {caps.supports_trailing_stop}")
print(f"OCO: {caps.supports_oco}")
print(f"Bracket: {caps.supports_bracket}")
```

### CCXT Capabilities by Exchange

| Exchange | Trailing | OCO | Bracket | Iceberg |
|----------|----------|-----|---------|---------|
| Binance | Yes | Yes | No | Yes |
| Bybit | Yes | No | Yes | No |
| OKX | Yes | Yes | No | Yes |
| Kraken | No | No | No | No |

## CCXT Broker

### Configuration

```python
from src.broker_bridge.adapters.ccxt import CCXTBroker

broker = CCXTBroker(
    exchange_id="binance",      # Exchange ID
    api_key="your_api_key",
    api_secret="your_api_secret",
    testnet=True,               # Use testnet
    sandbox=False,              # Sandbox mode
    options={                   # Additional options
        'defaultType': 'spot',  # 'spot', 'future', 'margin'
        'adjustForTimeDifference': True
    }
)
```

### Supported Exchanges

```python
import ccxt
print(ccxt.exchanges)  # List of 100+ exchanges
```

Main: binance, bybit, okx, kraken, kucoin, coinbase, huobi, gate, bitget

## IBKR Broker

### Configuration

```python
from src.broker_bridge.adapters.ibkr import IBKRBroker

broker = IBKRBroker(
    host="127.0.0.1",
    port=7497,          # TWS Paper: 7497, TWS Live: 7496
    client_id=1,        # Gateway Paper: 4002, Gateway Live: 4001
    readonly=False
)
```

### Contract Types

```python
# The broker automatically detects contract type
# based on symbol format:

# Stock
order = BrokerOrder(symbol="AAPL", ...)

# Forex
order = BrokerOrder(symbol="EUR/USD", ...)

# Futures (format: ROOT + YYYYMM or ROOT + MONTHCODE + YY)
order = BrokerOrder(symbol="ES2403", ...)  # March 2024
order = BrokerOrder(symbol="ESH24", ...)   # March 2024

# Options (OCC format)
order = BrokerOrder(symbol="AAPL240315C00175000", ...)
# AAPL, March 15 2024, Call, Strike $175.00
```

## Error Handling

```python
from src.broker_bridge.core.exceptions import (
    BrokerNotRegisteredError,
    BrokerConnectionError,
    OrderError,
    InsufficientFundsError,
    UnsupportedOrderTypeError
)

try:
    report = await executor.submit_order(order)
except BrokerNotRegisteredError:
    print("Broker not registered")
except BrokerConnectionError:
    print("Connection error")
except InsufficientFundsError as e:
    print(f"Insufficient funds: {e}")
except UnsupportedOrderTypeError:
    print("Order type not supported")
```

## Fallback Simulator

When a broker doesn't support an order type, the system uses local simulation:

```python
# If Kraken doesn't support trailing stop,
# the FallbackSimulator handles it locally
order = BrokerOrder(
    symbol="BTC/USD",
    order_type=OrderType.TRAILING_STOP,
    trail_percent=0.02
)

# The executor detects Kraken doesn't support trailing
# and uses the simulator automatically
report = await executor.submit_order(order, broker_type=BrokerType.CCXT)
```

## Context Manager

```python
async with UnifiedExecutor() as executor:
    executor.register_broker(ccxt_broker)
    executor.register_broker(ibkr_broker)

    # Automatic connection
    await executor.submit_order(order)

# Automatic disconnection
```

## API Reference

### UnifiedExecutor

| Method | Description |
|--------|-------------|
| `register_broker(broker)` | Register a broker |
| `unregister_broker(broker_type)` | Unregister broker |
| `connect_all()` | Connect all brokers |
| `disconnect_all()` | Disconnect all |
| `submit_order(order, broker_type?)` | Submit order |
| `cancel_order(order_id)` | Cancel order |
| `get_all_positions()` | Positions from all brokers |
| `get_all_balances()` | Balances from all brokers |
| `get_ticker(symbol)` | Get ticker |
| `get_orderbook(symbol)` | Get order book |

### BrokerOrder

| Field | Type | Description |
|-------|------|-------------|
| `symbol` | str | Asset symbol |
| `side` | OrderSide | BUY or SELL |
| `order_type` | OrderType | Order type |
| `quantity` | float | Quantity |
| `price` | float? | Limit price |
| `stop_price` | float? | Stop price |
| `trail_percent` | float? | % for trailing |
| `take_profit` | float? | Take profit price |
| `stop_loss` | float? | Stop loss price |
| `time_in_force` | TimeInForce | GTC, IOC, FOK |
| `reduce_only` | bool | Reduce position only |
| `post_only` | bool | Maker only |

## Tests

```bash
pytest src/broker_bridge/tests/ -v
```

Tests available: 76 tests covering all functionality.

## Related Documentation

- [API Reference](api_reference.md) - REST API endpoints
- [Paper Trading](paper_trading.md) - Simulated trading
- [Risk Management](risk_management.md) - Risk controls
