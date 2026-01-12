# Broker Bridge - Manual de Componente

## Descripcion

El modulo **broker_bridge** proporciona una capa de abstraccion unificada para operar con multiples brokers (crypto y tradicionales) a traves de una interfaz comun.

## Arquitectura

```
broker_bridge/
├── core/
│   ├── enums.py         # BrokerType, AssetClass, OrderType, OrderSide
│   ├── models.py        # BrokerOrder, BrokerPosition, ExecutionReport
│   ├── interfaces.py    # IBrokerAdapter (ABC)
│   └── exceptions.py    # Excepciones personalizadas
├── adapters/
│   ├── ccxt/            # Adaptador para 100+ exchanges crypto
│   └── ibkr/            # Adaptador para Interactive Brokers
└── execution/
    ├── unified_executor.py   # Orquestador principal
    ├── symbol_router.py      # Ruteo automatico por simbolo
    └── fallback_simulator.py # Simulacion para ordenes no soportadas
```

## Uso Basico

### Inicializacion

```python
from src.broker_bridge import (
    UnifiedExecutor,
    CCXTBroker,
    IBKRBroker,
    BrokerOrder,
    OrderSide,
    OrderType
)

# Crear executor
executor = UnifiedExecutor()

# Registrar brokers
executor.register_broker(CCXTBroker("binance", "api_key", "api_secret"))
executor.register_broker(IBKRBroker(port=7497))

# Conectar
await executor.connect_all()
```

### Enviar Ordenes

```python
# Orden crypto (auto-ruteada a CCXT)
order = BrokerOrder(
    symbol="BTC/USDT",
    side=OrderSide.BUY,
    order_type=OrderType.LIMIT,
    quantity=0.1,
    price=50000
)
report = await executor.submit_order(order)

# Orden stock (auto-ruteada a IBKR)
order = BrokerOrder(
    symbol="AAPL",
    side=OrderSide.BUY,
    order_type=OrderType.MARKET,
    quantity=10
)
report = await executor.submit_order(order)
```

## Ruteo Automatico

El `SymbolRouter` determina automaticamente el broker apropiado:

| Patron | Broker | Asset Class |
|--------|--------|-------------|
| `BTC/USDT`, `ETH/BTC` | CCXT | CRYPTO |
| `EUR/USD`, `GBP/JPY` | IBKR | FOREX |
| `AAPL`, `MSFT` | IBKR | STOCK |
| `SPX`, `NDX`, `VIX` | IBKR | INDEX |
| `ES2403`, `NQM24` | IBKR | FUTURES |
| `AAPL240315C00175000` | IBKR | OPTIONS |

### Personalizar Ruteo

```python
router = executor.get_router()

# Agregar patron crypto
router.add_crypto_pattern(r"^MY_TOKEN.*$")

# Agregar moneda forex
router.add_forex_currency("MXN")

# Override manual
router.set_override("SPECIAL", BrokerType.PAPER, AssetClass.STOCK)
```

## Tipos de Orden Soportados

```python
from src.broker_bridge.core.enums import OrderType

# Basicos
OrderType.MARKET
OrderType.LIMIT
OrderType.STOP_LOSS
OrderType.STOP_LIMIT

# Avanzados
OrderType.TRAILING_STOP
OrderType.BRACKET
OrderType.OCO
OrderType.ICEBERG

# Algoritmicos
OrderType.TWAP
OrderType.VWAP
```

## Capacidades por Broker

```python
# Verificar capacidades
ccxt = executor.get_broker(BrokerType.CCXT)
caps = ccxt.get_capabilities()

print(f"Trailing Stop: {caps.supports_trailing_stop}")
print(f"OCO: {caps.supports_oco}")
print(f"Bracket: {caps.supports_bracket}")
```

### Capacidades CCXT por Exchange

| Exchange | Trailing | OCO | Bracket | Iceberg |
|----------|----------|-----|---------|---------|
| Binance | Yes | Yes | No | Yes |
| Bybit | Yes | No | Yes | No |
| OKX | Yes | Yes | No | Yes |
| Kraken | No | No | No | No |

## CCXT Broker

### Configuracion

```python
from src.broker_bridge.adapters.ccxt import CCXTBroker

broker = CCXTBroker(
    exchange_id="binance",      # ID del exchange
    api_key="tu_api_key",
    api_secret="tu_api_secret",
    testnet=True,               # Usar testnet
    sandbox=False,              # Modo sandbox
    options={                   # Opciones adicionales
        'defaultType': 'spot',  # 'spot', 'future', 'margin'
        'adjustForTimeDifference': True
    }
)
```

### Exchanges Soportados

```python
import ccxt
print(ccxt.exchanges)  # Lista de 100+ exchanges
```

Principales: binance, bybit, okx, kraken, kucoin, coinbase, huobi, gate, bitget

## IBKR Broker

### Configuracion

```python
from src.broker_bridge.adapters.ibkr import IBKRBroker

broker = IBKRBroker(
    host="127.0.0.1",
    port=7497,          # TWS Paper: 7497, TWS Live: 7496
    client_id=1,        # Gateway Paper: 4002, Gateway Live: 4001
    readonly=False
)
```

### Tipos de Contrato

```python
# El broker detecta automaticamente el tipo de contrato
# basado en el formato del simbolo:

# Stock
order = BrokerOrder(symbol="AAPL", ...)

# Forex
order = BrokerOrder(symbol="EUR/USD", ...)

# Futures (formato: ROOT + YYYYMM o ROOT + MONTHCODE + YY)
order = BrokerOrder(symbol="ES2403", ...)  # Marzo 2024
order = BrokerOrder(symbol="ESH24", ...)   # Marzo 2024

# Options (formato OCC)
order = BrokerOrder(symbol="AAPL240315C00175000", ...)
# AAPL, 15 Marzo 2024, Call, Strike $175.00
```

## Manejo de Errores

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
    print("Broker no registrado")
except BrokerConnectionError:
    print("Error de conexion")
except InsufficientFundsError as e:
    print(f"Fondos insuficientes: {e}")
except UnsupportedOrderTypeError:
    print("Tipo de orden no soportado")
```

## Fallback Simulator

Cuando un broker no soporta un tipo de orden, el sistema usa simulacion local:

```python
# Si Kraken no soporta trailing stop,
# el FallbackSimulator lo maneja localmente
order = BrokerOrder(
    symbol="BTC/USD",
    order_type=OrderType.TRAILING_STOP,
    trail_percent=0.02
)

# El executor detecta que Kraken no soporta trailing
# y usa el simulador automaticamente
report = await executor.submit_order(order, broker_type=BrokerType.CCXT)
```

## Context Manager

```python
async with UnifiedExecutor() as executor:
    executor.register_broker(ccxt_broker)
    executor.register_broker(ibkr_broker)

    # Conexion automatica
    await executor.submit_order(order)

# Desconexion automatica
```

## API Reference

### UnifiedExecutor

| Metodo | Descripcion |
|--------|-------------|
| `register_broker(broker)` | Registrar un broker |
| `unregister_broker(broker_type)` | Desregistrar broker |
| `connect_all()` | Conectar todos los brokers |
| `disconnect_all()` | Desconectar todos |
| `submit_order(order, broker_type?)` | Enviar orden |
| `cancel_order(order_id)` | Cancelar orden |
| `get_all_positions()` | Posiciones de todos los brokers |
| `get_all_balances()` | Balances de todos los brokers |
| `get_ticker(symbol)` | Obtener ticker |
| `get_orderbook(symbol)` | Obtener order book |

### BrokerOrder

| Campo | Tipo | Descripcion |
|-------|------|-------------|
| `symbol` | str | Simbolo del activo |
| `side` | OrderSide | BUY o SELL |
| `order_type` | OrderType | Tipo de orden |
| `quantity` | float | Cantidad |
| `price` | float? | Precio limite |
| `stop_price` | float? | Precio de stop |
| `trail_percent` | float? | % para trailing |
| `take_profit` | float? | Precio take profit |
| `stop_loss` | float? | Precio stop loss |
| `time_in_force` | TimeInForce | GTC, IOC, FOK |
| `reduce_only` | bool | Solo reducir posicion |
| `post_only` | bool | Solo maker |

## Tests

```bash
pytest src/broker_bridge/tests/ -v
```

Tests disponibles: 76 tests cubriendo todas las funcionalidades.
