# Interfaces and Dependency Injection

The Interfaces module provides a protocol-based dependency injection system for Trad-Loop. It enables loose coupling between components, making the framework extensible and testable.

## Overview

The system consists of:
- **Protocols**: Define contracts using Python's `typing.Protocol` (structural typing)
- **Default Implementations**: Ready-to-use implementations of each protocol
- **Dependency Container**: Manages service registration and resolution

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Dependency Container                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌───────────────────┐    ┌───────────────────┐                │
│  │    Protocols      │    │  Implementations  │                │
│  │  (interfaces.py)  │    │ (implementations) │                │
│  ├───────────────────┤    ├───────────────────┤                │
│  │ IDataValidator    │───▶│ DefaultValidator  │                │
│  │ IMetricsCalculator│───▶│ DefaultMetrics    │                │
│  │ IPositionSizer    │───▶│ FixedFractional   │                │
│  │ IRiskManager      │───▶│ DefaultRiskMgr    │                │
│  │ ISignalGenerator  │    │ (custom)          │                │
│  │ IOrderExecutor    │    │ (custom)          │                │
│  │ IPositionManager  │    │ (custom)          │                │
│  │ IOHLCVFetcher     │    │ (custom)          │                │
│  │ IDataProvider     │    │ (custom)          │                │
│  │ IOptimizer        │    │ (custom)          │                │
│  └───────────────────┘    └───────────────────┘                │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                      Container                             │  │
│  ├───────────────────────────────────────────────────────────┤  │
│  │ + register_singleton(name, instance)                      │  │
│  │ + register_factory(name, factory, singleton=False)        │  │
│  │ + register_type(name, type, singleton=False, **kwargs)    │  │
│  │ + resolve(name) -> instance                               │  │
│  │ + try_resolve(name) -> Optional[instance]                 │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Available Protocols

### Core Interfaces

| Protocol | Description |
|----------|-------------|
| `IDataValidator` | Validates OHLCV data quality |
| `IMetricsCalculator` | Calculates performance metrics |
| `IPositionSizer` | Determines position size |
| `ISignalGenerator` | Generates trading signals |
| `IRiskManager` | Manages risk and validates trades |

### Data Provider Interfaces

| Protocol | Description |
|----------|-------------|
| `IOHLCVFetcher` | Fetches OHLCV data |
| `IDataProvider` | Extended data provider with metadata |

### Trading Interfaces

| Protocol | Description |
|----------|-------------|
| `IOrderExecutor` | Executes trading orders |
| `IPositionManager` | Manages open positions |

### Optimization Interfaces

| Protocol | Description |
|----------|-------------|
| `IObjectiveFunction` | Defines optimization objective |
| `IOptimizer` | Optimization algorithm interface |

## Data Types

The module defines common data types used across protocols:

```python
@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str]
    warnings: List[str]

@dataclass
class Signal:
    timestamp: datetime
    symbol: str
    action: str  # 'BUY', 'SELL', 'HOLD'
    price: float
    confidence: float
    metadata: Dict[str, Any]

@dataclass
class PositionInfo:
    id: str
    symbol: str
    side: str  # 'LONG', 'SHORT'
    entry_price: float
    quantity: float
    current_price: float
    unrealized_pnl: float
    entry_time: datetime

@dataclass
class TradeResult:
    id: str
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    return_pct: float
    entry_time: datetime
    exit_time: datetime
    exit_reason: str

@dataclass
class PerformanceMetrics:
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    total_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    avg_win: float
    avg_loss: float
    expectancy: float
```

## Protocol Details

### IDataValidator

Validates OHLCV data quality and integrity.

```python
@runtime_checkable
class IDataValidator(Protocol):
    def validate(self, data: pd.DataFrame) -> ValidationResult:
        """Validates an OHLCV DataFrame."""
        ...

    def sanitize(self, data: pd.DataFrame) -> pd.DataFrame:
        """Cleans data by removing problematic rows."""
        ...
```

**Usage:**
```python
from src.interfaces import IDataValidator, ValidationResult

class MyValidator:
    def validate(self, data: pd.DataFrame) -> ValidationResult:
        errors = []
        warnings = []

        # Check for missing values
        if data.isnull().any().any():
            errors.append("Data contains null values")

        # Check OHLC relationships
        if (data['high'] < data['low']).any():
            errors.append("High < Low in some rows")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )

    def sanitize(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.dropna()
```

### IMetricsCalculator

Calculates performance metrics from trades and equity curve.

```python
@runtime_checkable
class IMetricsCalculator(Protocol):
    def calculate(
        self,
        trades: List[TradeResult],
        equity_curve: List[float],
        initial_capital: float
    ) -> PerformanceMetrics:
        """Calculates all performance metrics."""
        ...
```

### IPositionSizer

Determines position size based on various strategies.

```python
@runtime_checkable
class IPositionSizer(Protocol):
    def calculate_size(
        self,
        capital: float,
        price: float,
        stop_loss: float,
        volatility: Optional[float] = None
    ) -> float:
        """Calculates position size."""
        ...
```

**Default Implementations:**
- `FixedFractionalSizer`: Risk fixed percentage per trade
- Extensible for Kelly Criterion, volatility-based, etc.

### IRiskManager

Validates trades against risk limits.

```python
@runtime_checkable
class IRiskManager(Protocol):
    def can_open_position(
        self,
        symbol: str,
        side: str,
        size: float,
        price: float,
        current_positions: List[PositionInfo],
        current_equity: float
    ) -> Tuple[bool, str]:
        """Checks if a position can be opened."""
        ...

    def calculate_stop_loss(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        atr: Optional[float] = None
    ) -> float:
        """Calculates stop loss price."""
        ...
```

### ISignalGenerator

Generates trading signals from data.

```python
@runtime_checkable
class ISignalGenerator(Protocol):
    def generate(self, data: pd.DataFrame) -> pd.Series:
        """Generates signals for all rows."""
        ...

    def get_current_signal(self, data: pd.DataFrame) -> Optional[Signal]:
        """Gets current signal (last row)."""
        ...
```

### IOrderExecutor

Executes trading orders (paper, live, or backtest).

```python
@runtime_checkable
class IOrderExecutor(Protocol):
    def execute_market_order(
        self,
        symbol: str,
        side: str,
        quantity: float
    ) -> Dict[str, Any]:
        """Executes market order."""
        ...

    def execute_limit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float
    ) -> Dict[str, Any]:
        """Executes limit order."""
        ...

    def cancel_order(self, order_id: str) -> bool:
        """Cancels an order."""
        ...
```

## Dependency Container

### Basic Usage

```python
from src.interfaces.container import Container

# Create container
container = Container()

# Register singleton instance
container.register_singleton('validator', MyValidator())

# Register factory (new instance each time)
container.register_factory('sizer', lambda: FixedFractionalSizer(risk_pct=2.0))

# Register type with constructor args
container.register_type('risk_manager', DefaultRiskManager, singleton=True, max_positions=5)

# Resolve services
validator = container.resolve('validator')
sizer = container.resolve('sizer')
```

### Using Default Configuration

```python
from src.interfaces.container import Container

# Create with default implementations
container = Container.with_defaults()

# Default services available:
# - 'validator': DefaultDataValidator
# - 'metrics_calculator': DefaultMetricsCalculator
# - 'position_sizer': FixedFractionalSizer
# - 'risk_manager': DefaultRiskManager

validator = container.resolve('validator')
```

### Global Container

```python
from src.interfaces.container import get_container, resolve

# Get global container (creates with defaults if needed)
container = get_container()

# Shorthand for resolving from global container
validator = resolve('validator')
```

### Container Methods

| Method | Description |
|--------|-------------|
| `register_singleton(name, instance)` | Register single instance |
| `register_factory(name, factory, singleton=False)` | Register factory function |
| `register_type(name, type, singleton=False, **kwargs)` | Register class type |
| `resolve(name)` | Get service instance |
| `try_resolve(name)` | Get instance or None |
| `has_service(name)` | Check if registered |
| `unregister(name)` | Remove service |
| `clear()` | Remove all services |

## Custom Implementation Example

Create a custom position sizer using Kelly Criterion:

```python
from src.interfaces import IPositionSizer

class KellyCriterionSizer:
    """Position sizer using Kelly Criterion."""

    def __init__(self, win_rate: float, avg_win: float, avg_loss: float, fraction: float = 0.5):
        self.win_rate = win_rate
        self.avg_win = avg_win
        self.avg_loss = avg_loss
        self.fraction = fraction  # Fractional Kelly (safer)

    def calculate_size(
        self,
        capital: float,
        price: float,
        stop_loss: float,
        volatility: Optional[float] = None
    ) -> float:
        # Kelly formula
        win_loss_ratio = self.avg_win / abs(self.avg_loss)
        kelly_pct = (self.win_rate * win_loss_ratio - (1 - self.win_rate)) / win_loss_ratio

        # Apply fraction and ensure non-negative
        kelly_pct = max(0, kelly_pct * self.fraction)

        # Calculate position size
        position_value = capital * kelly_pct
        quantity = position_value / price

        return quantity


# Register custom implementation
from src.interfaces.container import get_container

container = get_container()
container.register_singleton(
    'position_sizer',
    KellyCriterionSizer(win_rate=0.55, avg_win=100, avg_loss=80, fraction=0.25)
)
```

## Using with Strategy

```python
from src.interfaces.container import resolve

class MyStrategy(TradingStrategy):
    def __init__(self):
        super().__init__()
        # Inject dependencies
        self.validator = resolve('validator')
        self.sizer = resolve('position_sizer')
        self.risk_manager = resolve('risk_manager')

    def load_data(self, data: pd.DataFrame):
        # Validate data
        result = self.validator.validate(data)
        if not result.is_valid:
            raise ValueError(f"Invalid data: {result.errors}")

        # Sanitize if needed
        if result.warnings:
            data = self.validator.sanitize(data)

        super().load_data(data)

    def calculate_position_size(self, price: float, stop_loss: float) -> float:
        return self.sizer.calculate_size(
            capital=self.current_capital,
            price=price,
            stop_loss=stop_loss
        )

    def open_position(self, signal, stop_loss, take_profit):
        # Check with risk manager
        can_open, reason = self.risk_manager.can_open_position(
            symbol=self.config.symbol,
            side='LONG' if signal.signal == 'BUY' else 'SHORT',
            size=self.calculate_position_size(signal.price, stop_loss),
            price=signal.price,
            current_positions=self.positions,
            current_equity=self.current_capital
        )

        if not can_open:
            logger.warning(f"Cannot open position: {reason}")
            return None

        return super().open_position(signal, stop_loss, take_profit)
```

## Testing with Mock Implementations

```python
import pytest

class MockValidator:
    """Mock validator for testing."""

    def __init__(self, always_valid: bool = True):
        self.always_valid = always_valid
        self.validate_calls = []

    def validate(self, data):
        self.validate_calls.append(data)
        return ValidationResult(
            is_valid=self.always_valid,
            errors=[] if self.always_valid else ["Test error"],
            warnings=[]
        )

    def sanitize(self, data):
        return data


def test_strategy_with_mock():
    from src.interfaces.container import Container

    # Create container with mocks
    container = Container()
    container.register_singleton('validator', MockValidator(always_valid=True))

    # Test strategy...
```

## Best Practices

1. **Use protocols for abstraction**: Define interfaces using `Protocol` for duck typing
2. **Prefer constructor injection**: Pass dependencies via constructor, not global state
3. **Use singletons sparingly**: Only for truly shared resources
4. **Test with mocks**: Create mock implementations for unit tests
5. **Keep interfaces small**: Follow Interface Segregation Principle
6. **Document contracts**: Clearly document expected behavior in protocols

## Related Modules

- [Optimizers](optimizers.md) - Uses IOptimizer and IObjectiveFunction
- [Risk Management](risk_management.md) - Implements IRiskManager
- [User Guide](../user_guide.md) - Strategy development guide
