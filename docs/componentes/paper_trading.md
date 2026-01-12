# Paper Trading - Manual de Componente

## Descripcion

El modulo **paper_trading** proporciona un motor completo de simulacion para probar estrategias con ordenes avanzadas en tiempo real o con datos historicos.

## Arquitectura

```
paper_trading/
├── engine.py              # Motor principal
├── models.py              # Order, Position, TradeRecord
├── config.py              # Configuracion
├── order_simulator.py     # Simulador de ejecucion
├── position_manager.py    # Gestion de posiciones
├── performance_tracker.py # Metricas en tiempo real
├── realtime_feed.py       # Feeds de datos
├── orders/                # Tipos de orden avanzados
│   ├── base.py
│   ├── conditional_orders.py
│   ├── dynamic_orders.py
│   ├── execution_algos.py
│   └── risk_control.py
└── simulators/            # Simuladores especializados
    ├── trailing_simulator.py
    ├── algo_simulator.py
    └── composite_simulator.py
```

## Uso Basico

### Configuracion

```python
from src.paper_trading import PaperTradingEngine, PaperTradingConfig

config = PaperTradingConfig(
    initial_capital=10000,
    commission_pct=0.001,      # 0.1%
    slippage_pct=0.0005,       # 0.05%
    enable_shorting=True,
    max_positions=10,
    margin_requirement=0.5     # 50% margen
)

engine = PaperTradingEngine(config)
```

### Estrategia Realtime

```python
from src.paper_trading import RealtimeStrategy

class MiEstrategia(RealtimeStrategy):
    def __init__(self):
        super().__init__()
        self.sma_20 = []

    def on_start(self):
        """Inicializacion."""
        print("Estrategia iniciada")

    def on_candle(self, candle):
        """Procesar nueva vela."""
        self.sma_20.append(candle.close)
        if len(self.sma_20) > 20:
            self.sma_20.pop(0)

        sma = sum(self.sma_20) / len(self.sma_20)

        if candle.close > sma and not self.has_position():
            self.buy(candle.close, quantity=0.1)
        elif candle.close < sma and self.has_position():
            self.close_all_positions()

    def on_tick(self, symbol, price):
        """Procesar tick (opcional)."""
        pass

    def on_stop(self):
        """Finalizacion."""
        print("Estrategia detenida")
```

### Ejecutar

```python
# Registrar estrategia
engine.register_strategy(MiEstrategia())

# Backtest con datos historicos
result = engine.run_backtest(historical_data)

# O en tiempo real
await engine.run_live()
```

## Tipos de Orden (17+)

### Ordenes Basicas

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

### Ordenes Condicionales

```python
from src.paper_trading.orders.conditional_orders import (
    IfThenOrder,
    OneCancelsOther,
    OneCancelsAll
)

# If-Then: Si se ejecuta la primera, enviar la segunda
if_then = IfThenOrder(
    trigger_order=LimitOrder(...),
    then_order=StopOrder(...)
)

# OCO: Una cancela la otra
oco = OneCancelsOther(
    order1=LimitOrder(side="SELL", price=52000),  # Take profit
    order2=StopOrder(side="SELL", stop_price=48000)  # Stop loss
)

# OCA: Una cancela todas
oca = OneCancelsAll(orders=[order1, order2, order3])
```

### Ordenes Dinamicas

```python
from src.paper_trading.orders.dynamic_orders import (
    TrailingStopOrder,
    TrailingStopLimitOrder,
    DynamicTrailingOrder
)

# Trailing Stop basico
trailing = TrailingStopOrder(
    symbol="BTC/USDT",
    side="SELL",
    quantity=0.1,
    trail_percent=0.02  # 2%
)

# Trailing Stop con Limit
trailing_limit = TrailingStopLimitOrder(
    symbol="BTC/USDT",
    side="SELL",
    quantity=0.1,
    trail_percent=0.02,
    limit_offset=0.001  # 0.1% debajo del trigger
)

# Trailing dinamico (ajusta trail_percent segun volatilidad)
dynamic = DynamicTrailingOrder(
    symbol="BTC/USDT",
    side="SELL",
    quantity=0.1,
    base_trail_percent=0.02,
    volatility_multiplier=1.5
)
```

### Ordenes de Control de Riesgo

```python
from src.paper_trading.orders.risk_control import (
    ProfitTargetOrder,
    BreakEvenOrder,
    ScaleOutOrder,
    TimedExitOrder
)

# Profit Target con trailing
profit_target = ProfitTargetOrder(
    position_id="pos123",
    profit_target_pct=0.05,     # 5% profit
    enable_trailing=True,
    trail_after_pct=0.03       # Trail despues de 3%
)

# Break Even
break_even = BreakEvenOrder(
    position_id="pos123",
    trigger_profit_pct=0.02,   # Mover SL a break even al 2%
    lock_profit_pct=0.001      # Asegurar 0.1% profit
)

# Scale Out (salida parcial)
scale_out = ScaleOutOrder(
    position_id="pos123",
    levels=[
        {'profit_pct': 0.02, 'close_pct': 0.25},  # 25% al 2%
        {'profit_pct': 0.04, 'close_pct': 0.25},  # 25% al 4%
        {'profit_pct': 0.06, 'close_pct': 0.50}   # 50% al 6%
    ]
)

# Timed Exit
timed_exit = TimedExitOrder(
    position_id="pos123",
    exit_after_minutes=60  # Cerrar despues de 1 hora
)
```

### Algoritmos de Ejecucion

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
    num_slices=12  # 12 ordenes en 60 min
)

# VWAP (Volume Weighted Average Price)
vwap = VWAPOrder(
    symbol="BTC/USDT",
    side="BUY",
    total_quantity=1.0,
    participation_rate=0.1  # 10% del volumen
)

# Iceberg
iceberg = IcebergOrder(
    symbol="BTC/USDT",
    side="BUY",
    total_quantity=10.0,
    visible_quantity=1.0,
    variance_pct=0.1  # +/- 10% variacion
)

# POV (Percentage of Volume)
pov = POVOrder(
    symbol="BTC/USDT",
    side="BUY",
    total_quantity=5.0,
    target_pct=0.05  # 5% del volumen
)
```

### Ordenes Compuestas

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

# Multi-Leg (multiples ordenes relacionadas)
multi_leg = MultiLegOrder(
    legs=[
        LimitOrder(symbol="BTC/USDT", side="BUY", quantity=0.05, price=49500),
        LimitOrder(symbol="BTC/USDT", side="BUY", quantity=0.05, price=49000),
    ],
    fill_requirement="ALL"  # "ALL", "ANY", "PARTIAL"
)
```

## Position Manager

```python
from src.paper_trading import PositionManager

pm = engine.position_manager

# Obtener posiciones
positions = pm.get_all_positions()
btc_position = pm.get_position("BTC/USDT")

# Metricas de posicion
if btc_position:
    print(f"Entry: {btc_position.entry_price}")
    print(f"Quantity: {btc_position.quantity}")
    print(f"Unrealized PnL: ${btc_position.unrealized_pnl:.2f}")
    print(f"ROI: {btc_position.roi_pct:.2f}%")

# Cerrar posicion
pm.close_position("BTC/USDT", price=51000, reason="manual")
```

## Performance Tracker

```python
from src.paper_trading import PerformanceTracker

tracker = engine.performance_tracker

# Metricas en tiempo real
metrics = tracker.get_realtime_metrics()
print(f"Equity: ${metrics['equity']:.2f}")
print(f"Unrealized PnL: ${metrics['unrealized_pnl']:.2f}")
print(f"Win Rate: {metrics['win_rate']:.2f}%")
print(f"Current Drawdown: {metrics['current_drawdown']:.2f}%")

# Historial de trades
trades = tracker.get_trade_history()
```

## Feeds de Datos

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

# Configurar WebSocket
ws_handler = BinanceWebSocketHandler(
    symbols=["BTC/USDT", "ETH/USDT"],
    timeframe="1m"
)

feed = RealtimeFeedManager(ws_handler)
engine.set_feed(feed)

await engine.run_live()
```

## Simulacion de Ejecucion

### Slippage

```python
from src.paper_trading import SlippageModel

# Slippage fijo
config.slippage_model = SlippageModel.FIXED
config.slippage_pct = 0.0005

# Slippage basado en volumen
config.slippage_model = SlippageModel.VOLUME_BASED
config.slippage_base = 0.0001
config.slippage_volume_impact = 0.0001
```

### Comisiones

```python
# Comision porcentual
config.commission_type = "percentage"
config.commission_pct = 0.001

# Comision fija
config.commission_type = "fixed"
config.commission_fixed = 1.0  # $1 por trade
```

## API Reference

### PaperTradingEngine

| Metodo | Descripcion |
|--------|-------------|
| `register_strategy(strategy)` | Registrar estrategia |
| `run_backtest(data)` | Ejecutar backtest |
| `run_live()` | Ejecutar en vivo |
| `submit_order(order)` | Enviar orden |
| `cancel_order(order_id)` | Cancelar orden |
| `get_positions()` | Obtener posiciones |
| `get_balance()` | Obtener balance |
| `get_metrics()` | Obtener metricas |

### RealtimeStrategy (ABC)

| Metodo | Descripcion |
|--------|-------------|
| `on_start()` | Llamado al iniciar |
| `on_candle(candle)` | Llamado con cada vela |
| `on_tick(symbol, price)` | Llamado con cada tick |
| `on_stop()` | Llamado al detener |
| `buy(price, quantity, sl?, tp?)` | Comprar |
| `sell(price, quantity, sl?, tp?)` | Vender |
| `close_all_positions()` | Cerrar todas las posiciones |

## Tests

```bash
pytest src/paper_trading/tests/ -v
```

Cobertura: ~220 tests para ordenes avanzadas y simuladores.
