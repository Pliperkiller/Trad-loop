# Technical Manual - Trad-Loop

## Table of Contents

1. [Overview](#1-overview)
2. [C2 Diagram - System Context](#2-c2-diagram---system-context)
3. [C3 Diagram - Components](#3-c3-diagram---components)
4. [Module Architecture](#4-module-architecture)
5. [Data Flows](#5-data-flows)
6. [Design Patterns](#6-design-patterns)
7. [External Dependencies](#7-external-dependencies)
8. [Test Structure](#8-test-structure)

---

## 1. Overview

**Trad-Loop** is a professional algorithmic trading framework composed of two main systems:

| System | Purpose | Technology |
|--------|---------|------------|
| **DataExtractor** | Market data extraction | Clean Architecture, Tkinter GUI |
| **StrategyTrader** | Strategy development and execution | Modular, FastAPI, Async |

### Project Metrics

- **Lines of code**: ~34,000
- **Tests**: ~500+
- **Modules**: 10+ main modules
- **Supported exchanges**: 100+ (via CCXT)

---

## 2. C2 Diagram - System Context

The context diagram shows how Trad-Loop interacts with external systems and users.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                   CONTEXT                                        │
│                                                                                  │
│  ┌──────────────┐                                           ┌──────────────┐    │
│  │   Trader/    │                                           │   Crypto     │    │
│  │   Analyst    │                                           │  Exchanges   │    │
│  │              │                                           │  (Binance,   │    │
│  │  [Person]    │                                           │  Bybit, OKX) │    │
│  └──────┬───────┘                                           └──────┬───────┘    │
│         │                                                          │            │
│         │ Develops strategies                        Market Data,  │            │
│         │ Analyzes results                           Order Exec    │            │
│         │ Executes trades                                          │            │
│         ▼                                                          ▼            │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                          │   │
│  │                           TRAD-LOOP                                      │   │
│  │                                                                          │   │
│  │   ┌─────────────────┐              ┌─────────────────────────────────┐  │   │
│  │   │  DataExtractor  │─────────────▶│        StrategyTrader           │  │   │
│  │   │                 │   CSV Data   │                                 │  │   │
│  │   │  - GUI (Tkinter)│              │  - Backtesting                  │  │   │
│  │   │  - CLI          │              │  - Optimization                 │  │   │
│  │   │  - Exchanges    │              │  - Paper Trading                │  │   │
│  │   └─────────────────┘              │  - Live Execution               │  │   │
│  │                                    │  - REST/WebSocket API           │  │   │
│  │                                    └─────────────────────────────────┘  │   │
│  │                                                                          │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│         │                                                          │            │
│         │                                                          │            │
│         ▼                                                          ▼            │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐  ┌──────────────┐│
│  │  PostgreSQL  │      │  Data APIs   │      │ Interactive  │  │  Web Client  ││
│  │   Database   │      │  (CoinGecko, │      │   Brokers    │  │  (Dashboard) ││
│  │              │      │  Glassnode)  │      │    (IBKR)    │  │              ││
│  │  [Database]  │      │  [External]  │      │  [External]  │  │  [Frontend]  ││
│  └──────────────┘      └──────────────┘      └──────────────┘  └──────────────┘│
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Context Diagram (Mermaid)

```mermaid
C4Context
    title System Context Diagram - Trad-Loop

    Person(trader, "Trader/Analyst", "Develops and executes trading strategies")

    System_Boundary(tradloop, "Trad-Loop") {
        System(dataextractor, "DataExtractor", "Market data extraction")
        System(strategytrader, "StrategyTrader", "Trading strategy framework")
    }

    System_Ext(exchanges_crypto, "Crypto Exchanges", "Binance, Bybit, OKX, Kraken, etc.")
    System_Ext(ibkr, "Interactive Brokers", "Stocks, Forex, Futures, Options")
    System_Ext(data_apis, "Data APIs", "CoinGecko, Glassnode, DeFi Llama")
    System_Ext(database, "PostgreSQL", "Persistent storage")
    System_Ext(webclient, "Web Client", "Dashboard, Swagger UI")

    Rel(trader, dataextractor, "Extracts historical data", "GUI/CLI")
    Rel(trader, strategytrader, "Develops and executes strategies", "Python/API")
    Rel(dataextractor, exchanges_crypto, "Fetches OHLCV", "REST/WebSocket")
    Rel(strategytrader, exchanges_crypto, "Executes orders", "CCXT")
    Rel(strategytrader, ibkr, "Executes orders", "ib_insync")
    Rel(strategytrader, data_apis, "Fetches fundamental data", "REST")
    Rel(strategytrader, database, "Persists data", "SQLAlchemy")
    Rel(webclient, strategytrader, "Queries API", "REST/JSON")
```

### Actors and External Systems

| Actor/System | Type | Interaction |
|--------------|------|-------------|
| Trader/Analyst | Person | Develops strategies, analyzes results |
| Crypto Exchanges | External System | CCXT for 100+ exchanges |
| Interactive Brokers | External System | ib_insync for traditional markets |
| Data APIs | External System | On-chain fundamental data |
| PostgreSQL | Database | Storage for trades, metrics |
| Web Client | Frontend | Dashboard, Swagger UI |

---

## 3. C3 Diagram - Components

### 3.1 Component Overview

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                    STRATEGY TRADER                                       │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│  │                              PRESENTATION LAYER                                  │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────────┐  │   │
│  │  │    REST API     │  │   WebSocket     │  │         CLI Scripts             │  │   │
│  │  │   (FastAPI)     │  │     API         │  │    run_with_strategy.py         │  │   │
│  │  └────────┬────────┘  └────────┬────────┘  └────────────────┬────────────────┘  │   │
│  └───────────┼────────────────────┼─────────────────────────────┼───────────────────┘   │
│              │                    │                             │                        │
│  ┌───────────▼────────────────────▼─────────────────────────────▼───────────────────┐   │
│  │                              APPLICATION LAYER                                    │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │   │
│  │  │  Strategy   │  │ Performance │  │  Optimizer  │  │    Unified Executor     │  │   │
│  │  │  Framework  │  │  Analyzer   │  │             │  │    (Broker Bridge)      │  │   │
│  │  │             │  │             │  │             │  │                         │  │   │
│  │  │ strategy/   │  │performance. │  │ optimizer/  │  │ unified_executor.py     │  │   │
│  │  │ base.py     │  │    py       │  │             │  │                         │  │   │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └───────────┬─────────────┘  │   │
│  └─────────┼────────────────┼────────────────┼─────────────────────┼────────────────┘   │
│            │                │                │                     │                     │
│  ┌─────────▼────────────────▼────────────────▼─────────────────────▼────────────────┐   │
│  │                                DOMAIN LAYER                                       │   │
│  │                                                                                   │   │
│  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐  ┌───────────────────┐  │   │
│  │  │  Indicators   │  │  Optimizers   │  │Paper Trading  │  │    Portfolio      │  │   │
│  │  │               │  │               │  │               │  │                   │  │   │
│  │  │ - Technical   │  │ - Grid        │  │ - Engine      │  │ - Allocator       │  │   │
│  │  │ - Fundamental │  │ - Random      │  │ - Orders      │  │ - Rebalancer      │  │   │
│  │  │ - Utils       │  │ - Bayesian    │  │ - Simulators  │  │ - Backtester      │  │   │
│  │  │               │  │ - Genetic     │  │ - Position    │  │ - Metrics         │  │   │
│  │  │               │  │ - WalkForward │  │   Manager     │  │                   │  │   │
│  │  └───────────────┘  └───────────────┘  └───────────────┘  └───────────────────┘  │   │
│  │                                                                                   │   │
│  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐  ┌───────────────────┐  │   │
│  │  │    Risk       │  │    Stress     │  │     Job       │  │    Interfaces     │  │   │
│  │  │  Management   │  │   Testing     │  │    Manager    │  │    (Protocols)    │  │   │
│  │  │               │  │               │  │               │  │                   │  │   │
│  │  │ - Sizer       │  │ - MonteCarlo  │  │ - Async Jobs  │  │ - IDataValidator  │  │   │
│  │  │ - Limits      │  │ - Scenario    │  │ - Progress    │  │ - IMetricsCalc    │  │   │
│  │  │ - Correlation │  │ - Sensitivity │  │ - Scheduling  │  │ - IPositionSizer  │  │   │
│  │  └───────────────┘  └───────────────┘  └───────────────┘  └───────────────────┘  │   │
│  │                                                                                   │   │
│  │  ┌───────────────────────────────────────────────────────────────────────────┐   │   │
│  │  │                            Broker Bridge                                   │   │   │
│  │  │                                                                            │   │   │
│  │  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────────────┐  │   │   │
│  │  │  │  CCXT   │  │  IBKR   │  │  Paper  │  │ Symbol  │  │    Fallback     │  │   │   │
│  │  │  │ Adapter │  │ Adapter │  │ Trading │  │ Router  │  │    Simulator    │  │   │   │
│  │  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘  └─────────────────┘  │   │   │
│  │  └───────────────────────────────────────────────────────────────────────────┘   │   │
│  │                                                                                   │   │
│  └───────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                          │
│  ┌───────────────────────────────────────────────────────────────────────────────────┐   │
│  │                             INFRASTRUCTURE LAYER                                  │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐  │   │
│  │  │      CCXT       │  │   ib_insync     │  │   WebSocket     │  │  SQLAlchemy │  │   │
│  │  │  (100+ exch)    │  │    (IBKR)       │  │   Handlers      │  │   (ORM)     │  │   │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘  └─────────────┘  │   │
│  └───────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Component Diagram (Mermaid)

```mermaid
C4Component
    title Component Diagram - StrategyTrader

    Container_Boundary(api, "API Layer") {
        Component(rest_api, "REST API", "FastAPI", "Endpoints for strategies, trades, performance")
        Component(ws_api, "WebSocket API", "FastAPI", "Real-time streaming")
        Component(swagger, "Swagger UI", "OpenAPI", "Interactive documentation")
    }

    Container_Boundary(core, "Core Components") {
        Component(strategy, "Strategy Framework", "Python", "TradingStrategy ABC, signals, positions")
        Component(performance, "Performance Analyzer", "Python", "30+ metrics, visualization")
        Component(optimizer, "Optimizer", "Python", "Grid, Random, Bayesian, Genetic, WalkForward")
        Component(job_manager, "Job Manager", "Python", "Async task execution")
    }

    Container_Boundary(domain, "Domain Components") {
        Component(indicators, "Indicators", "Python", "Technical & Fundamental indicators")
        Component(paper_trading, "Paper Trading", "Python", "Engine, orders, simulators")
        Component(portfolio, "Portfolio", "Python", "Allocation, rebalancing, metrics")
        Component(risk, "Risk Management", "Python", "Position sizing, limits, VaR")
        Component(stress, "Stress Testing", "Python", "Monte Carlo, scenarios, sensitivity")
        Component(interfaces, "Interfaces", "Python", "Protocols and DI container")
    }

    Container_Boundary(broker, "Broker Bridge") {
        Component(unified, "Unified Executor", "Python", "Multi-broker orchestrator")
        Component(router, "Symbol Router", "Python", "Automatic routing by asset")
        Component(ccxt_adapter, "CCXT Adapter", "Python", "100+ crypto exchanges")
        Component(ibkr_adapter, "IBKR Adapter", "Python", "Stocks, forex, futures")
        Component(fallback, "Fallback Simulator", "Python", "Paper trading fallback")
    }

    Container_Boundary(infra, "Infrastructure") {
        Component(ccxt_lib, "CCXT Library", "Python", "Exchange abstraction")
        Component(ibinsync, "ib_insync", "Python", "IB TWS/Gateway connection")
        Component(websocket, "WebSocket", "Python", "Real-time data feeds")
        Component(db, "SQLAlchemy", "Python", "Database ORM")
    }

    Rel(rest_api, strategy, "Executes")
    Rel(rest_api, performance, "Calculates metrics")
    Rel(rest_api, job_manager, "Creates jobs")
    Rel(strategy, indicators, "Uses")
    Rel(strategy, performance, "Generates data for")
    Rel(optimizer, strategy, "Optimizes parameters of")
    Rel(paper_trading, strategy, "Executes in simulation")
    Rel(unified, router, "Queries route")
    Rel(unified, ccxt_adapter, "Sends crypto orders")
    Rel(unified, ibkr_adapter, "Sends traditional orders")
    Rel(unified, fallback, "Fallback for unsupported orders")
    Rel(ccxt_adapter, ccxt_lib, "Uses")
    Rel(ibkr_adapter, ibinsync, "Uses")
```

### 3.3 Detailed Components

#### Strategy Framework (`strategy/base.py`)

```
┌─────────────────────────────────────────────────────────────────┐
│                     STRATEGY FRAMEWORK                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                 TradingStrategy (ABC)                    │    │
│  ├─────────────────────────────────────────────────────────┤    │
│  │ + config: StrategyConfig                                 │    │
│  │ + data: DataFrame                                        │    │
│  │ + equity_curve: DataFrame                                │    │
│  │ + positions: List[Position]                              │    │
│  ├─────────────────────────────────────────────────────────┤    │
│  │ + load_data(df: DataFrame)                               │    │
│  │ + calculate_indicators() [abstract]                      │    │
│  │ + generate_signals() -> List[TradeSignal] [abstract]     │    │
│  │ + backtest() -> DataFrame                                │    │
│  │ + calculate_position_size(price, stop_loss) -> float     │    │
│  │ + open_position(signal, stop_loss, take_profit)          │    │
│  │ + close_position(position, exit_price, time, reason)     │    │
│  │ + get_performance_metrics() -> Dict                      │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                   │
│                              │ extends                           │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │          MovingAverageCrossoverStrategy                  │    │
│  ├─────────────────────────────────────────────────────────┤    │
│  │ + fast_period: int                                       │    │
│  │ + slow_period: int                                       │    │
│  │ + rsi_period: int                                        │    │
│  ├─────────────────────────────────────────────────────────┤    │
│  │ + calculate_indicators()                                 │    │
│  │ + generate_signals() -> List[TradeSignal]                │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌──────────────────┐  ┌────────────────┐  ┌────────────────┐   │
│  │   TradeSignal    │  │    Position    │  │ StrategyConfig │   │
│  ├──────────────────┤  ├────────────────┤  ├────────────────┤   │
│  │ timestamp        │  │ entry_time     │  │ initial_capital│   │
│  │ signal (BUY/SELL)│  │ entry_price    │  │ risk_per_trade │   │
│  │ confidence       │  │ quantity       │  │ stop_loss_pct  │   │
│  │ metadata         │  │ stop_loss      │  │ take_profit_pct│   │
│  └──────────────────┘  │ take_profit    │  │ commission_pct │   │
│                        │ position_type  │  └────────────────┘   │
│                        └────────────────┘                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### Broker Bridge (`broker_bridge/`)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              BROKER BRIDGE                                       │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────┐     │
│  │                         UnifiedExecutor                                 │     │
│  ├────────────────────────────────────────────────────────────────────────┤     │
│  │ - _brokers: Dict[BrokerType, IBrokerAdapter]                           │     │
│  │ - _router: SymbolRouter                                                │     │
│  │ - _fallback: FallbackSimulator                                         │     │
│  ├────────────────────────────────────────────────────────────────────────┤     │
│  │ + register_broker(broker: IBrokerAdapter)                              │     │
│  │ + connect_all() -> Dict[BrokerType, bool]                              │     │
│  │ + submit_order(order, broker_type?) -> ExecutionReport                 │     │
│  │ + cancel_order(order_id) -> bool                                       │     │
│  │ + get_all_positions() -> Dict[BrokerType, List[Position]]              │     │
│  │ + get_all_balances() -> Dict[BrokerType, Dict[str, float]]             │     │
│  └──────────────────────────────┬─────────────────────────────────────────┘     │
│                                 │                                                │
│              ┌──────────────────┼──────────────────┐                            │
│              │                  │                  │                            │
│              ▼                  ▼                  ▼                            │
│  ┌───────────────────┐  ┌─────────────┐  ┌───────────────────┐                 │
│  │    CCXTBroker     │  │ IBKRBroker  │  │ FallbackSimulator │                 │
│  ├───────────────────┤  ├─────────────┤  ├───────────────────┤                 │
│  │ implements        │  │ implements  │  │ implements        │                 │
│  │ IBrokerAdapter    │  │IBrokerAdapt │  │ IBrokerAdapter    │                 │
│  └─────────┬─────────┘  └──────┬──────┘  └───────────────────┘                 │
│            │                   │                                                │
│            ▼                   ▼                                                │
│  ┌───────────────────┐  ┌─────────────┐                                        │
│  │   ccxt library    │  │  ib_insync  │                                        │
│  │  (100+ exchanges) │  │   library   │                                        │
│  └───────────────────┘  └─────────────┘                                        │
│                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────┐     │
│  │                          SymbolRouter                                   │     │
│  ├────────────────────────────────────────────────────────────────────────┤     │
│  │ + route(symbol) -> Tuple[BrokerType, AssetClass]                       │     │
│  │ + group_by_broker(symbols) -> Dict[BrokerType, List[str]]              │     │
│  │ + group_by_asset_class(symbols) -> Dict[AssetClass, List[str]]         │     │
│  ├────────────────────────────────────────────────────────────────────────┤     │
│  │ Routing Rules:                                                          │     │
│  │   BTC/USDT, ETH/BTC  →  CCXT (Crypto)                                  │     │
│  │   EUR/USD, GBP/JPY   →  IBKR (Forex)                                   │     │
│  │   AAPL, MSFT         →  IBKR (Stock)                                   │     │
│  │   SPX, NDX, VIX      →  IBKR (Index)                                   │     │
│  │   ES2403, NQM24      →  IBKR (Futures)                                 │     │
│  └────────────────────────────────────────────────────────────────────────┘     │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                         IBrokerAdapter (ABC)                             │    │
│  ├─────────────────────────────────────────────────────────────────────────┤    │
│  │ + connect() -> bool                                                      │    │
│  │ + disconnect() -> None                                                   │    │
│  │ + get_capabilities() -> BrokerCapabilities                               │    │
│  │ + submit_order(order: BrokerOrder) -> ExecutionReport                    │    │
│  │ + cancel_order(order_id: str, symbol?: str) -> bool                      │    │
│  │ + get_positions(symbol?: str) -> List[BrokerPosition]                    │    │
│  │ + get_balance() -> Dict[str, float]                                      │    │
│  │ + get_ticker(symbol: str) -> Dict                                        │    │
│  │ + get_orderbook(symbol: str, limit?: int) -> Dict                        │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

#### Paper Trading (`paper_trading/`)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              PAPER TRADING                                       │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                        PaperTradingEngine                                │    │
│  ├─────────────────────────────────────────────────────────────────────────┤    │
│  │ - _position_manager: PositionManager                                     │    │
│  │ - _order_simulator: OrderSimulator                                       │    │
│  │ - _performance_tracker: RealtimePerformanceTracker                       │    │
│  │ - _feed_manager: FeedManager                                             │    │
│  ├─────────────────────────────────────────────────────────────────────────┤    │
│  │ + register_strategy(strategy: RealtimeStrategy)                          │    │
│  │ + run_live() -> None                                                     │    │
│  │ + run_backtest(data: DataFrame) -> BacktestResult                        │    │
│  │ + submit_order(order: Order) -> ExecutionReport                          │    │
│  │ + cancel_order(order_id: str) -> bool                                    │    │
│  │ + get_positions() -> List[Position]                                      │    │
│  │ + get_metrics() -> Dict                                                  │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                        Order Types (17+)                                 │    │
│  ├─────────────────────────────────────────────────────────────────────────┤    │
│  │                                                                          │    │
│  │  Basic Orders          Conditional Orders        Dynamic Orders          │    │
│  │  ├── Market            ├── IfThenOrder           ├── TrailingStop        │    │
│  │  ├── Limit             ├── OneCancelsOther       ├── TrailingStopLimit   │    │
│  │  ├── StopLoss          └── OneCancelsAll         └── DynamicTrailing     │    │
│  │  └── StopLimit                                                           │    │
│  │                                                                          │    │
│  │  Risk Control          Algo Execution                                    │    │
│  │  ├── ProfitTarget      ├── TWAP (Time Weighted)                          │    │
│  │  ├── BreakEven         ├── VWAP (Volume Weighted)                        │    │
│  │  ├── ScaleOut          ├── Iceberg                                       │    │
│  │  └── TimedExit         └── POV (Percentage of Volume)                    │    │
│  │                                                                          │    │
│  │  Composite Orders                                                        │    │
│  │  ├── BracketOrder (Entry + SL + TP)                                      │    │
│  │  └── MultiLegOrder                                                       │    │
│  │                                                                          │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
│  ┌───────────────────┐  ┌───────────────────┐  ┌───────────────────────────┐   │
│  │  PositionManager  │  │  OrderSimulator   │  │ PerformanceTracker        │   │
│  ├───────────────────┤  ├───────────────────┤  ├───────────────────────────┤   │
│  │ + open_position() │  │ + execute_order() │  │ + update_equity()         │   │
│  │ + close_position()│  │ + apply_slippage()│  │ + calculate_metrics()     │   │
│  │ + get_unrealized_ │  │ + validate_order()│  │ + get_realtime_stats()    │   │
│  │   pnl()           │  │ + check_fills()   │  │ + track_drawdown()        │   │
│  └───────────────────┘  └───────────────────┘  └───────────────────────────┘   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Module Architecture

### 4.1 Indicators Module

```
indicators/
├── base.py                    # Protocols and base interfaces
│   ├── IndicatorProtocol      # Protocol for indicator functions
│   └── IndicatorResult        # Base class for multi-value results
│
├── technical/                 # Technical indicators
│   ├── momentum.py            # RSI, MACD, Stochastic, KDJ, CCI, Williams%R
│   ├── trend.py               # SMA, EMA, DEMA, TEMA, ADX, TRIX, Supertrend
│   ├── volatility.py          # Bollinger, ATR, Keltner, Donchian
│   ├── volume.py              # OBV, VWAP, MFI, ADL
│   ├── pivots.py              # Pivot Points, Fibonacci, S/R
│   └── ichimoku.py            # Complete Ichimoku Cloud
│
├── fundamental/               # Fundamental data
│   ├── base_client.py         # Abstract base API client
│   ├── coingecko.py           # Market cap, volume, dominance
│   ├── glassnode.py           # On-chain metrics
│   ├── defillama.py           # TVL, DeFi protocols
│   ├── alternative_me.py      # Fear & Greed Index
│   └── santiment.py           # Social metrics
│
└── utils/
    ├── cache.py               # Data cache
    └── validators.py          # Input validation
```

### 4.2 Optimizers Module

All optimizers support parallel execution via `n_jobs` parameter using `ProcessPoolExecutor`.

```
optimizers/
├── grid_search.py             # Exhaustive search (parallel)
├── random_search.py           # Random search (parallel)
├── bayesian.py                # Bayesian optimization (parallel)
├── genetic.py                 # Genetic algorithm
├── walk_forward.py            # Walk-forward analysis (parallel)
├── optimization_types.py      # Types and configurations
│
├── validation/                # Cross-validation
│   ├── splitters.py           # TimeSeriesSplit, RollingWindow, PurgedKFold
│   ├── time_series_cv.py      # Temporal cross-validation
│   ├── purged_kfold.py        # K-Fold with data purge
│   └── results.py             # Result models
│
└── analysis/                  # Post-optimization analysis
    ├── overfitting_detection.py  # IS/OOS overfitting detection
    ├── parameter_stability.py    # Parameter stability
    └── visualization.py          # 3D plots, convergence
```

### 4.3 Portfolio Module

```
portfolio/
├── portfolio_manager.py       # Main orchestrator
├── models.py                  # PortfolioConfig, PortfolioState, PortfolioResult
├── allocator.py               # Allocation methods
│   ├── equal_weight()
│   ├── market_cap_weight()
│   ├── risk_parity()
│   ├── inverse_volatility()
│   ├── minimum_variance()
│   ├── maximum_sharpe()
│   └── hierarchical_risk_parity()
├── rebalancer.py              # Rebalancing strategies
│   ├── by_frequency()         # Daily, weekly, monthly
│   ├── by_threshold()         # By allocation drift
│   └── by_volatility()        # By target volatility
├── backtester.py              # Portfolio backtesting
└── metrics.py                 # Portfolio metrics
```

### 4.4 Risk Management Module

```
risk_management/
├── risk_manager.py            # Main orchestrator
├── models.py                  # TradeRiskAssessment, RiskState
├── config.py                  # RiskConfig
├── position_sizer.py          # Position sizing
│   ├── fixed_fractional()     # Fixed % of capital
│   ├── kelly_criterion()      # Optimal Kelly
│   └── optimal_f()            # Optimal-f
├── risk_limits.py             # Risk limits
│   ├── daily_loss_limit
│   ├── drawdown_protection
│   ├── position_limits
│   └── exposure_limits
└── correlation_manager.py     # Correlation management
    ├── track_correlations()
    ├── detect_regime_change()
    └── cluster_analysis()
```

### 4.5 Stress Testing Module

```
stress_testing/
├── stress_tester.py           # Main orchestrator
├── models.py                  # StressTestConfig, StressTestReport
├── monte_carlo.py             # Monte Carlo simulation
│   ├── simulate_paths()       # Path generation
│   ├── bootstrap()            # Bootstrap resampling
│   └── gbm_simulation()       # Geometric Brownian Motion
├── scenario_analysis.py       # Scenario analysis
│   ├── bull_market()
│   ├── bear_market()
│   ├── crash()
│   └── volatility_shock()
└── sensitivity.py             # Sensitivity analysis
    ├── single_variable()
    └── multi_variable()
```

### 4.6 Job Manager Module

```
job_manager/
├── job_manager.py             # Main job manager
│   ├── JobManager             # Singleton job manager
│   ├── create_job()           # Create new job
│   ├── get_job()              # Get job by ID
│   ├── cancel_job()           # Cancel running job
│   └── list_jobs()            # List all jobs
├── models.py                  # Job models
│   ├── JobType                # BACKTEST, OPTIMIZATION, DATA_EXTRACTION
│   ├── JobStatus              # QUEUED, RUNNING, COMPLETED, FAILED, CANCELLED
│   ├── JobProgress            # Progress tracking
│   └── JobResult              # Job result container
└── executor.py                # Async job execution
```

### 4.7 Interfaces Module

```
interfaces/
├── protocols.py               # Protocol definitions
│   ├── IDataValidator         # Data validation protocol
│   ├── IMetricsCalculator     # Metrics calculation protocol
│   ├── IPositionSizer         # Position sizing protocol
│   ├── IRiskManager           # Risk management protocol
│   ├── IOptimizer             # Optimization protocol
│   └── IPerformanceAnalyzer   # Performance analysis protocol
├── container.py               # Dependency injection container
│   ├── DependencyContainer    # Service container
│   ├── register()             # Register service
│   ├── resolve()              # Resolve dependency
│   └── get_container()        # Get singleton container
└── implementations/           # Default implementations
    ├── data_validator.py
    ├── metrics_calculator.py
    └── position_sizer.py
```

### 4.8 API Modules

```
api_modules/
├── data_service.py            # Data services
│   ├── DataService            # OHLCV data management
│   ├── load_data()            # Load from file
│   ├── fetch_data()           # Fetch from exchange
│   └── validate_data()        # Data validation
├── strategy_registry.py       # Strategy registration
│   ├── StrategyRegistry       # Strategy container
│   ├── register()             # Register strategy
│   ├── get()                  # Get strategy by ID
│   └── list_all()             # List all strategies
└── session_manager.py         # Session management
    ├── SessionManager         # Trading session manager
    ├── create_session()       # Create new session
    └── get_session()          # Get session by ID
```

---

## 5. Data Flows

### 5.1 Backtest Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           BACKTEST FLOW                                      │
└─────────────────────────────────────────────────────────────────────────────┘

     ┌──────────┐         ┌────────────┐         ┌───────────────┐
     │   CSV    │────────▶│   Pandas   │────────▶│   Strategy    │
     │   Data   │         │ DataFrame  │         │  load_data()  │
     └──────────┘         └────────────┘         └───────┬───────┘
                                                         │
                                                         ▼
                                               ┌───────────────────┐
                                               │ calculate_        │
                                               │ indicators()      │
                                               │                   │
                                               │ - EMA, RSI, MACD  │
                                               │ - Custom calcs    │
                                               └─────────┬─────────┘
                                                         │
                                                         ▼
                                               ┌───────────────────┐
                                               │ generate_signals()│
                                               │                   │
                                               │ Returns:          │
                                               │ List[TradeSignal] │
                                               └─────────┬─────────┘
                                                         │
                                                         ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                              backtest() loop                                  │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│   for each bar:                                                               │
│     ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────┐   │
│     │ Check SL/TP     │───▶│ Execute Signal  │───▶│ Update Equity      │   │
│     │ on positions    │    │ (BUY/SELL/HOLD) │    │ Curve              │   │
│     └─────────────────┘    └─────────────────┘    └─────────────────────┘   │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
                                                         │
                                                         ▼
                                               ┌───────────────────┐
                                               │ get_performance_  │
                                               │ metrics()         │
                                               │                   │
                                               │ Returns: Dict     │
                                               │ - total_return    │
                                               │ - sharpe_ratio    │
                                               │ - max_drawdown    │
                                               └─────────┬─────────┘
                                                         │
                                                         ▼
                                               ┌───────────────────┐
                                               │ Performance       │
                                               │ Analyzer          │
                                               │                   │
                                               │ 30+ metrics       │
                                               └─────────┬─────────┘
                                                         │
                                                         ▼
                                               ┌───────────────────┐
                                               │ Performance       │
                                               │ Visualizer        │
                                               │                   │
                                               │ 6-panel dashboard │
                                               └───────────────────┘
```

### 5.2 Optimization Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          OPTIMIZATION FLOW                                   │
└─────────────────────────────────────────────────────────────────────────────┘

     ┌───────────────────┐
     │ StrategyOptimizer │
     │ (strategy_class,  │
     │  data, config)    │
     └─────────┬─────────┘
               │
               ▼
     ┌───────────────────┐
     │ add_parameter()   │
     │ - name, type      │
     │ - min, max        │
     │ - step (optional) │
     └─────────┬─────────┘
               │
               ▼
     ┌───────────────────┐
     │ Select Algorithm  │
     │ ┌───────────────┐ │
     │ │ grid_optimize │ │
     │ │ random_opt    │ │
     │ │ bayesian_opt  │ │
     │ │ genetic_opt   │ │
     │ │ walk_forward  │ │
     │ └───────────────┘ │
     └─────────┬─────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                        OPTIMIZATION LOOP                                      │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│   Split data using TimeSeriesCV / PurgedKFold / WalkForward                  │
│                                                                               │
│   for each parameter combination:                                             │
│     ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                │
│     │ Create       │───▶│ Run Backtest │───▶│ Calculate    │                │
│     │ Strategy     │    │ on IS Data   │    │ Objective    │                │
│     │ Instance     │    │              │    │ (Sharpe, PF) │                │
│     └──────────────┘    └──────────────┘    └──────────────┘                │
│                                                     │                         │
│                                                     ▼                         │
│                                            ┌──────────────┐                  │
│                                            │ Update Best  │                  │
│                                            │ Parameters   │                  │
│                                            └──────────────┘                  │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
               │
               ▼
     ┌───────────────────┐
     │ Validate on OOS   │
     │ Data              │
     └─────────┬─────────┘
               │
               ▼
     ┌───────────────────┐
     │ Analysis          │
     │ - Overfitting     │
     │   Detection       │
     │ - Parameter       │
     │   Stability       │
     │ - Visualization   │
     └─────────┬─────────┘
               │
               ▼
     ┌───────────────────┐
     │ OptimizationResult│
     │ - best_params     │
     │ - best_value      │
     │ - all_results     │
     │ - convergence     │
     └───────────────────┘
```

### 5.3 Multi-Broker Execution Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       MULTI-BROKER EXECUTION FLOW                            │
└─────────────────────────────────────────────────────────────────────────────┘

     ┌───────────────────┐
     │ Strategy Signal   │
     │ BUY BTC/USDT      │
     └─────────┬─────────┘
               │
               ▼
     ┌───────────────────┐
     │ BrokerOrder       │
     │ - symbol          │
     │ - side            │
     │ - order_type      │
     │ - quantity        │
     │ - price           │
     └─────────┬─────────┘
               │
               ▼
     ┌───────────────────┐
     │ UnifiedExecutor   │
     │ submit_order()    │
     └─────────┬─────────┘
               │
               ▼
     ┌───────────────────┐
     │ SymbolRouter      │
     │ route(symbol)     │
     │                   │
     │ BTC/USDT → CCXT   │
     │ AAPL → IBKR       │
     │ EUR/USD → IBKR    │
     └─────────┬─────────┘
               │
               ├───────────────────────┬───────────────────────┐
               │                       │                       │
               ▼                       ▼                       ▼
     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
     │   CCXTBroker    │     │   IBKRBroker    │     │ FallbackSimul.  │
     │                 │     │                 │     │                 │
     │ - Binance       │     │ - TWS/Gateway   │     │ - Paper Trading │
     │ - Bybit         │     │ - Stocks        │     │ - Unsupported   │
     │ - OKX           │     │ - Forex         │     │   order types   │
     │ - 100+ more     │     │ - Futures       │     │                 │
     └────────┬────────┘     └────────┬────────┘     └────────┬────────┘
              │                       │                       │
              ▼                       ▼                       ▼
     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
     │ CCXT Library    │     │ ib_insync       │     │ Local Simulator │
     │ create_order()  │     │ placeOrder()    │     │ simulate_fill() │
     └────────┬────────┘     └────────┬────────┘     └────────┬────────┘
              │                       │                       │
              └───────────────────────┼───────────────────────┘
                                      │
                                      ▼
                            ┌───────────────────┐
                            │ ExecutionReport   │
                            │ - order_id        │
                            │ - status          │
                            │ - filled_qty      │
                            │ - avg_price       │
                            │ - commission      │
                            └───────────────────┘
```

---

## 6. Design Patterns

### 6.1 Patterns Used

| Pattern | Usage in Trad-Loop | Example |
|---------|-------------------|---------|
| **Strategy** | Interchangeable algorithms | TradingStrategy, Optimizers |
| **Adapter** | External system integration | CCXTBroker, IBKRBroker |
| **Factory** | Object creation | ParameterSpace, Contract creation |
| **Template Method** | Algorithm with variable steps | TradingStrategy.backtest() |
| **Observer** | Event notification | RealtimeStrategy callbacks |
| **Facade** | Simplified interface | PortfolioManager, RiskManager |
| **Decorator** | Additional functionality | Performance wrapping |
| **Repository** | Data abstraction | MarketRepository |
| **Composite** | Hierarchical structures | CompositeOrder, MultiLegOrder |
| **Chain of Responsibility** | Chained processing | Order validation pipeline |
| **Protocol** | Interface contracts | IDataValidator, IRiskManager |
| **Singleton** | Single instance | JobManager, DependencyContainer |

### 6.2 Pattern Diagrams

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          DESIGN PATTERNS                                     │
└─────────────────────────────────────────────────────────────────────────────┘

STRATEGY PATTERN                    ADAPTER PATTERN
─────────────────                   ───────────────
     ┌────────────────┐                  ┌────────────────┐
     │ <<interface>>  │                  │ <<interface>>  │
     │ TradingStrategy│                  │ IBrokerAdapter │
     └───────┬────────┘                  └───────┬────────┘
             │                                   │
    ┌────────┼────────┐              ┌──────────┼──────────┐
    │        │        │              │          │          │
    ▼        ▼        ▼              ▼          ▼          ▼
┌───────┐┌───────┐┌───────┐    ┌────────┐ ┌────────┐ ┌─────────┐
│  MA   ││ RSI   ││Custom │    │  CCXT  │ │  IBKR  │ │Fallback │
│Crossov││Strat. ││Strat. │    │Adapter │ │Adapter │ │Simulator│
└───────┘└───────┘└───────┘    └────────┘ └────────┘ └─────────┘


FACADE PATTERN                      TEMPLATE METHOD
──────────────                      ───────────────
┌─────────────────────────┐         ┌──────────────────────────┐
│    PortfolioManager     │         │    TradingStrategy       │
│         (Facade)        │         │    (Template Method)     │
├─────────────────────────┤         ├──────────────────────────┤
│ - allocator             │         │ backtest() {             │
│ - rebalancer            │         │   for bar in data:       │
│ - backtester            │         │     check_stops()        │
│ - metrics               │         │     signal = gen_signal()│
├─────────────────────────┤         │     execute(signal)      │
│ + run_backtest()        │         │     update_equity()      │
│ + rebalance()           │         │ }                        │
│ + get_allocation()      │         │                          │
└─────────────────────────┘         │ gen_signal() [abstract]  │
                                    └──────────────────────────┘


OBSERVER PATTERN                    COMPOSITE PATTERN
────────────────                    ─────────────────
┌────────────────┐                  ┌────────────────┐
│ FeedManager    │                  │ <<interface>>  │
│  (Subject)     │                  │     Order      │
└───────┬────────┘                  └───────┬────────┘
        │ notifies                          │
        ▼                          ┌────────┼────────┐
┌────────────────┐                 │        │        │
│RealtimeStrategy│            ┌────▼───┐┌───▼────┐┌──▼─────┐
│   (Observer)   │            │ Market ││ Limit  ││Bracket │
├────────────────┤            │ Order  ││ Order  ││ Order  │
│ on_candle()    │            └────────┘└────────┘│(Compos)│
│ on_tick()      │                                └────────┘
└────────────────┘                                     │
                                              ┌────────┼────────┐
                                              ▼        ▼        ▼
                                          ┌──────┐┌──────┐┌──────┐
                                          │Entry ││  SL  ││  TP  │
                                          │Order ││Order ││Order │
                                          └──────┘└──────┘└──────┘
```

---

## 7. External Dependencies

### 7.1 Main Dependencies

| Category | Package | Version | Usage |
|----------|---------|---------|-------|
| **Data** | pandas | 2.2.0 | DataFrames, time series |
| | numpy | 1.26.4 | Numerical calculations |
| | scipy | 1.12.0 | Statistics, optimization |
| **ML** | scikit-learn | 1.7.2 | Cross-validation, metrics |
| | scikit-optimize | 0.10.2 | Bayesian optimization |
| **Exchanges** | ccxt | latest | 100+ crypto exchanges |
| | python-binance | 1.0.34 | Binance specific |
| | ib_insync | 0.9.86 | Interactive Brokers |
| **Web** | fastapi | 0.109.0 | REST API |
| | uvicorn | 0.27.0 | ASGI server |
| | websockets | 15.0.1 | WebSocket support |
| **Viz** | matplotlib | 3.10.8 | Charts |
| | seaborn | 0.13.2 | Statistical visualization |
| | mplfinance | 0.12.10b0 | Financial charts |
| **DB** | sqlalchemy | 2.0.25 | ORM |
| | psycopg2-binary | 2.9.9 | PostgreSQL |
| **Async** | aiohttp | 3.13.3 | Async HTTP |
| **Testing** | pytest | 7.4.4 | Test framework |
| | pytest-asyncio | 0.23.3 | Async tests |
| **Validation** | pydantic | 2.5.3 | Data validation |

### 7.2 Dependency Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DEPENDENCY GRAPH                                     │
└─────────────────────────────────────────────────────────────────────────────┘

                              ┌─────────────────┐
                              │   Trad-Loop     │
                              └────────┬────────┘
                                       │
           ┌───────────────────────────┼───────────────────────────┐
           │                           │                           │
           ▼                           ▼                           ▼
    ┌─────────────┐            ┌─────────────┐            ┌─────────────┐
    │    Data     │            │   Trading   │            │     Web     │
    │   Layer     │            │   Layer     │            │    Layer    │
    └──────┬──────┘            └──────┬──────┘            └──────┬──────┘
           │                          │                          │
    ┌──────┴──────┐            ┌──────┴──────┐            ┌──────┴──────┐
    │             │            │             │            │             │
    ▼             ▼            ▼             ▼            ▼             ▼
┌───────┐   ┌───────┐    ┌───────┐   ┌───────┐    ┌───────┐   ┌───────┐
│pandas │   │ numpy │    │ ccxt  │   │  ib_  │    │fastapi│   │uvicorn│
│       │   │       │    │       │   │insync │    │       │   │       │
└───────┘   └───────┘    └───────┘   └───────┘    └───────┘   └───────┘
    │                          │
    ▼                          ▼
┌───────┐                ┌───────────┐
│ scipy │                │ websockets│
└───────┘                └───────────┘


              ┌─────────────────┐
              │   Visualization │
              └────────┬────────┘
                       │
           ┌───────────┼───────────┐
           │           │           │
           ▼           ▼           ▼
      ┌─────────┐ ┌─────────┐ ┌──────────┐
      │matplotlib│ │ seaborn │ │mplfinance│
      └─────────┘ └─────────┘ └──────────┘
```

---

## 8. Test Structure

### 8.1 Test Organization

```
tests/
├── conftest.py                    # Global fixtures
├── test_strategy.py               # Base strategy tests
├── test_performance.py            # Performance analyzer tests
├── test_optimizer.py              # Optimizer tests
│
├── indicators/
│   ├── conftest.py               # Indicator fixtures
│   ├── test_integration.py       # Integration tests
│   ├── test_backward_compat.py   # Backward compatibility
│   ├── technical/
│   │   ├── test_momentum.py
│   │   ├── test_trend.py
│   │   ├── test_volatility.py
│   │   ├── test_volume.py
│   │   ├── test_pivots.py
│   │   └── test_ichimoku.py
│   └── fundamental/
│       ├── test_coingecko.py
│       ├── test_glassnode.py
│       └── ...
│
├── optimizers/
│   ├── conftest.py
│   ├── analysis/
│   │   ├── test_overfitting.py
│   │   └── test_parameter_stability.py
│   └── validation/
│       ├── test_purged_kfold.py
│       └── test_splitters.py
│
├── paper_trading/
│   ├── conftest.py
│   ├── test_conditional_orders.py
│   ├── test_dynamic_orders.py
│   ├── test_execution_algos.py
│   ├── test_models.py
│   ├── test_order_simulator.py
│   ├── test_position_manager.py
│   └── test_risk_control_orders.py
│
├── portfolio/
│   ├── conftest.py
│   ├── test_allocator.py
│   ├── test_backtester.py
│   ├── test_metrics.py
│   ├── test_models.py
│   ├── test_portfolio_manager.py
│   └── test_rebalancer.py
│
├── risk_management/
│   ├── conftest.py
│   ├── test_correlation_manager.py
│   ├── test_position_sizer.py
│   ├── test_risk_limits.py
│   └── test_risk_manager.py
│
├── stress_testing/
│   ├── conftest.py
│   ├── test_models.py
│   ├── test_monte_carlo.py
│   ├── test_scenario_analysis.py
│   ├── test_sensitivity.py
│   └── test_stress_tester.py
│
└── broker_bridge/                 # In src/broker_bridge/tests/
    ├── conftest.py
    ├── test_ccxt_broker.py
    ├── test_symbol_router.py
    └── test_unified_executor.py
```

### 8.2 Test Commands

```bash
# Run all tests
pytest

# With coverage
pytest --cov=src --cov-report=html

# Specific tests
pytest tests/test_strategy.py -v
pytest tests/indicators/ -v
pytest src/broker_bridge/tests/ -v

# By marker
pytest -m "not slow"
pytest -m "integration"

# Verbose with details
pytest -v --tb=short
```

### 8.3 Coverage Metrics

| Module | Estimated Coverage |
|--------|-------------------|
| strategy/ | 85%+ |
| performance.py | 90%+ |
| broker_bridge | 95%+ |
| paper_trading | 90%+ |
| indicators | 85%+ |
| optimizers | 80%+ |
| portfolio | 85%+ |
| risk_management | 85%+ |
| stress_testing | 80%+ |

---

## Appendix A: Technical Glossary

| Term | Definition |
|------|------------|
| **ABC** | Abstract Base Class |
| **CCXT** | CryptoCurrency eXchange Trading Library |
| **IBKR** | Interactive Brokers |
| **IS/OOS** | In-Sample / Out-of-Sample (training/test data) |
| **OHLCV** | Open, High, Low, Close, Volume |
| **SL/TP** | Stop Loss / Take Profit |
| **TWAP** | Time Weighted Average Price |
| **VWAP** | Volume Weighted Average Price |
| **VaR** | Value at Risk |
| **CVaR** | Conditional Value at Risk |
| **MVO** | Mean-Variance Optimization |
| **HRP** | Hierarchical Risk Parity |
| **DI** | Dependency Injection |

---

## Appendix B: References

- [CCXT Documentation](https://docs.ccxt.com/)
- [ib_insync Documentation](https://ib-insync.readthedocs.io/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [pandas Documentation](https://pandas.pydata.org/docs/)
- [C4 Model](https://c4model.com/)
