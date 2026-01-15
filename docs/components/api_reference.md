# API Reference

Trad-Loop provides multiple APIs for interacting with the trading framework:
- **Main REST API** - Strategy management, trades, and performance metrics
- **Data API** - Market data from 100+ exchanges
- **Backtest API** - Backtesting and optimization
- **Paper Trading API** - Simulated trading
- **WebSocket API** - Real-time data streaming

## Starting the Server

```bash
cd StrategyTrader
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Main REST API

### Health Check

```
GET /
```

Returns server status and available features.

**Response:**
```json
{
    "service": "Trad-loop API",
    "status": "running",
    "version": "1.0.0",
    "features": {
        "data_extractor": true,
        "websocket": true,
        "backtest": true
    }
}
```

### Strategy Management

#### List Strategies

```
GET /api/v1/strategies
```

Lists all registered strategies.

**Response:**
```json
{
    "strategies": [
        {
            "id": "my-strategy",
            "name": "MovingAverageCrossoverStrategy",
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "total_trades": 45,
            "is_active": false
        }
    ]
}
```

#### Get Trades

```
GET /api/v1/trades/{strategy_id}
```

Gets trades for a specific strategy.

**Query Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `start_time` | string | Filter from date (ISO format) |
| `end_time` | string | Filter to date (ISO format) |
| `position_type` | string | Filter by LONG or SHORT |
| `profitable_only` | boolean | Only return profitable trades |

**Example:**
```bash
curl "http://localhost:8000/api/v1/trades/my-strategy?profitable_only=true"
```

**Response:**
```json
{
    "trades": [
        {
            "id": "my-strategy-trade-0",
            "entry_time": "2024-01-15T10:30:00",
            "exit_time": "2024-01-15T14:45:00",
            "entry_price": 42500.0,
            "exit_price": 43200.0,
            "quantity": 0.1,
            "position_type": "LONG",
            "pnl": 70.0,
            "return_pct": 1.65,
            "reason": "take_profit"
        }
    ],
    "symbol": "BTC/USDT",
    "strategy_name": "MovingAverageCrossoverStrategy",
    "total_trades": 1
}
```

#### Get Performance Metrics

```
GET /api/v1/performance/{strategy_id}
```

Gets performance metrics for a strategy.

**Response:**
```json
{
    "total_trades": 45,
    "winning_trades": 28,
    "losing_trades": 17,
    "win_rate": 62.2,
    "profit_factor": 1.85,
    "total_return_pct": 23.5,
    "max_drawdown_pct": 8.3,
    "final_capital": 12350.0,
    "avg_win": 125.50,
    "avg_loss": -75.30,
    "sharpe_ratio": 1.42
}
```

#### Get Equity Curve

```
GET /api/v1/equity/{strategy_id}
```

Gets the equity curve data for a strategy.

**Response:**
```json
{
    "strategy_id": "my-strategy",
    "initial_capital": 10000,
    "equity_curve": [10000, 10150, 10050, 10300, ...],
    "final_capital": 12350.0
}
```

## Data API

### List Exchanges

```
GET /api/v1/exchanges
```

Lists all available exchanges.

**Response:**
```json
{
    "exchanges": ["binance", "bybit", "okx", "kraken", ...],
    "by_category": {
        "tier_1": ["binance", "coinbase", "kraken"],
        "tier_2": ["bybit", "okx", "kucoin"],
        "futures": ["binance", "bybit"],
        "dex": ["uniswap", "sushiswap"]
    },
    "configured": ["binance", "kraken"]
}
```

### Get Exchange Info

```
GET /api/v1/exchanges/{exchange_id}/info
```

Gets detailed information about an exchange.

**Response:**
```json
{
    "exchange_id": "binance",
    "name": "Binance",
    "has_ohlcv": true,
    "supported_timeframes": ["1m", "5m", "15m", "1h", "4h", "1d"],
    "rate_limit": 50
}
```

### Get Symbols

```
GET /api/v1/exchanges/{exchange_id}/symbols
```

Lists trading pairs available on an exchange.

**Response:**
```json
{
    "exchange": "binance",
    "count": 1500,
    "symbols": ["BTC/USDT", "ETH/USDT", "SOL/USDT", ...]
}
```

### Get Symbols Catalog

```
GET /api/v1/exchanges/{exchange_id}/symbols/catalog
```

Gets an organized catalog of symbols, grouped by quote currency.

**Response:**
```json
{
    "exchange": "binance",
    "total": 1500,
    "popular": ["BTC/USDT", "ETH/USDT", "SOL/USDT", ...],
    "by_quote": {
        "USDT": ["BTC/USDT", "ETH/USDT", ...],
        "BTC": ["ETH/BTC", "SOL/BTC", ...],
        "USDC": ["BTC/USDC", "ETH/USDC", ...]
    },
    "quote_currencies": ["USDT", "BTC", "USDC", ...]
}
```

### Get OHLCV Data

```
GET /api/v1/ohlcv
```

Gets historical OHLCV (candlestick) data.

**Query Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `exchange` | string | Yes | Exchange ID (e.g., "binance") |
| `symbol` | string | Yes | Trading pair (e.g., "BTC/USDT") |
| `timeframe` | string | Yes | Candle timeframe (1m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d, 3d, 1w, 1M) |
| `start` | string | Yes | Start date (ISO format) |
| `end` | string | Yes | End date (ISO format) |
| `warmup_candles` | integer | No | Additional candles for indicator warmup (default: 100) |

**Example:**
```bash
curl "http://localhost:8000/api/v1/ohlcv?exchange=binance&symbol=BTC/USDT&timeframe=1h&start=2024-01-01T00:00:00&end=2024-01-31T23:59:59"
```

**Response:**
```json
{
    "symbol": "BTC/USDT",
    "timeframe": "1h",
    "exchange": "binance",
    "count": 744,
    "warmup_count": 100,
    "data": [
        {
            "timestamp": "2024-01-01T00:00:00",
            "open": 42500.0,
            "high": 42800.0,
            "low": 42300.0,
            "close": 42650.0,
            "volume": 1500.5
        },
        ...
    ]
}
```

## Backtest API

### List Available Strategies

```
GET /api/v1/backtest/strategies
```

Lists strategy classes available for backtesting.

**Response:**
```json
{
    "strategies": [
        {
            "name": "MovingAverageCrossover",
            "class": "MovingAverageCrossoverStrategy",
            "description": "EMA crossover strategy with RSI filter",
            "parameters": [
                {"name": "fast_period", "type": "int", "default": 12},
                {"name": "slow_period", "type": "int", "default": 26},
                {"name": "rsi_period", "type": "int", "default": 14}
            ]
        }
    ]
}
```

### Run Backtest

```
POST /api/v1/backtest/run
```

Starts a backtest job (runs asynchronously).

**Request Body:**
```json
{
    "strategy": "MovingAverageCrossover",
    "exchange": "binance",
    "symbol": "BTC/USDT",
    "timeframe": "1h",
    "start_date": "2024-01-01",
    "end_date": "2024-12-31",
    "initial_capital": 10000,
    "strategy_params": {
        "fast_period": 12,
        "slow_period": 26
    }
}
```

**Response:**
```json
{
    "job_id": "a1b2c3d4",
    "status": "queued",
    "message": "Backtest job created"
}
```

### Get Job Status

```
GET /api/v1/jobs/{job_id}
```

Gets the status of a backtest or optimization job.

**Response:**
```json
{
    "id": "a1b2c3d4",
    "type": "backtest",
    "status": "running",
    "progress": {
        "current": 75,
        "total": 100,
        "percentage": 75.0,
        "message": "Running backtest..."
    },
    "created_at": "2024-01-15T10:30:00Z",
    "started_at": "2024-01-15T10:30:01Z",
    "completed_at": null
}
```

### Get Job Result

```
GET /api/v1/jobs/{job_id}/result
```

Gets the result of a completed job.

**Response (Backtest):**
```json
{
    "id": "a1b2c3d4",
    "type": "backtest",
    "status": "completed",
    "result": {
        "success": true,
        "data": {
            "trades": [...],
            "metrics": {
                "total_return_pct": 23.5,
                "sharpe_ratio": 1.42,
                "max_drawdown_pct": 8.3
            },
            "equity_curve": [...]
        }
    }
}
```

### Run Optimization

```
POST /api/v1/backtest/optimize
```

Starts an optimization job.

**Request Body:**
```json
{
    "strategy": "MovingAverageCrossover",
    "exchange": "binance",
    "symbol": "BTC/USDT",
    "timeframe": "1h",
    "start_date": "2024-01-01",
    "end_date": "2024-12-31",
    "initial_capital": 10000,
    "parameters": {
        "fast_period": {"type": "int", "min": 5, "max": 20, "step": 5},
        "slow_period": {"type": "int", "min": 20, "max": 50, "step": 10}
    },
    "method": "bayesian",
    "objective": "sharpe_ratio",
    "n_iterations": 50
}
```

**Response:**
```json
{
    "job_id": "e5f6g7h8",
    "status": "queued",
    "message": "Optimization job created"
}
```

### Cancel Job

```
POST /api/v1/jobs/{job_id}/cancel
```

Cancels a running job.

**Response:**
```json
{
    "success": true,
    "message": "Job cancelled"
}
```

### List Jobs

```
GET /api/v1/jobs
```

Lists all jobs, optionally filtered by type.

**Query Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `type` | string | Filter by job type (backtest, optimization) |
| `status` | string | Filter by status (queued, running, completed, failed) |

**Response:**
```json
{
    "jobs": [
        {
            "id": "a1b2c3d4",
            "type": "backtest",
            "status": "completed",
            "created_at": "2024-01-15T10:30:00Z"
        }
    ]
}
```

## Paper Trading API

### Create Session

```
POST /api/v1/paper-trading/sessions
```

Creates a new paper trading session.

**Request Body:**
```json
{
    "name": "Test Session",
    "initial_capital": 10000,
    "symbols": ["BTC/USDT", "ETH/USDT"],
    "exchange": "binance"
}
```

**Response:**
```json
{
    "session_id": "sess-12345",
    "name": "Test Session",
    "status": "active",
    "capital": 10000
}
```

### Get Session

```
GET /api/v1/paper-trading/sessions/{session_id}
```

Gets session details including positions and performance.

### Submit Order

```
POST /api/v1/paper-trading/sessions/{session_id}/orders
```

Submits an order to a paper trading session.

**Request Body:**
```json
{
    "symbol": "BTC/USDT",
    "side": "buy",
    "type": "limit",
    "quantity": 0.1,
    "price": 42000,
    "stop_loss": 41000,
    "take_profit": 44000
}
```

### Get Positions

```
GET /api/v1/paper-trading/sessions/{session_id}/positions
```

Gets open positions for a session.

### Get Orders

```
GET /api/v1/paper-trading/sessions/{session_id}/orders
```

Gets order history for a session.

## WebSocket API

### Candle Streaming

```
ws://localhost:8000/ws/candles
```

Subscribe to real-time candle updates.

**Subscribe Message:**
```json
{
    "action": "subscribe",
    "exchange": "binance",
    "symbol": "BTC/USDT",
    "timeframe": "1h"
}
```

**Candle Update:**
```json
{
    "type": "candle",
    "symbol": "BTC/USDT",
    "timeframe": "1h",
    "exchange": "binance",
    "data": {
        "time": 1705320000,
        "open": 42500.0,
        "high": 42800.0,
        "low": 42300.0,
        "close": 42650.0,
        "volume": 1500.5
    },
    "is_closed": false
}
```

**Unsubscribe:**
```json
{
    "action": "unsubscribe",
    "exchange": "binance",
    "symbol": "BTC/USDT",
    "timeframe": "1h"
}
```

### Job Progress

```
ws://localhost:8000/ws/jobs/{job_id}
```

Subscribe to job progress updates.

**Progress Update:**
```json
{
    "type": "progress",
    "job_id": "a1b2c3d4",
    "current": 75,
    "total": 100,
    "percentage": 75.0,
    "message": "Running backtest..."
}
```

**Completion Update:**
```json
{
    "type": "completed",
    "job_id": "a1b2c3d4",
    "success": true
}
```

### Python WebSocket Client Example

```python
import asyncio
import websockets
import json

async def subscribe_to_candles():
    uri = "ws://localhost:8000/ws/candles"

    async with websockets.connect(uri) as websocket:
        # Subscribe to BTC/USDT 1h candles
        await websocket.send(json.dumps({
            "action": "subscribe",
            "exchange": "binance",
            "symbol": "BTC/USDT",
            "timeframe": "1h"
        }))

        # Listen for updates
        async for message in websocket:
            data = json.loads(message)
            print(f"Candle: {data}")

            if data.get("is_closed"):
                print(f"Closed candle: {data['data']}")

asyncio.run(subscribe_to_candles())
```

### JavaScript WebSocket Client Example

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/candles');

ws.onopen = () => {
    ws.send(JSON.stringify({
        action: 'subscribe',
        exchange: 'binance',
        symbol: 'BTC/USDT',
        timeframe: '1h'
    }));
};

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log('Candle update:', data);

    if (data.is_closed) {
        console.log('New closed candle:', data.data);
    }
};

ws.onerror = (error) => {
    console.error('WebSocket error:', error);
};
```

## Error Handling

All endpoints return standard HTTP status codes:

| Status | Description |
|--------|-------------|
| 200 | Success |
| 400 | Bad Request - Invalid parameters |
| 404 | Not Found - Resource doesn't exist |
| 500 | Internal Server Error |
| 503 | Service Unavailable - Feature not available |

**Error Response Format:**
```json
{
    "detail": "Error message describing the problem"
}
```

## Rate Limiting

The API does not implement rate limiting directly, but exchange data endpoints respect the rate limits of underlying exchanges (via CCXT).

For high-frequency applications, consider:
- Caching OHLCV responses
- Using WebSocket for real-time data instead of polling
- Batching requests when possible

## Related Documentation

- [Job Manager](job_manager.md) - Async task execution
- [User Guide](../user_guide.md) - Complete usage guide
- [Broker Bridge](broker_bridge.md) - Live trading execution
