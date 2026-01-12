# Manual de Uso - Trad-Loop

## Indice

1. [Introduccion](#1-introduccion)
2. [Instalacion y Configuracion](#2-instalacion-y-configuracion)
3. [Extraccion de Datos (DataExtractor)](#3-extraccion-de-datos-dataextractor)
4. [Desarrollo de Estrategias](#4-desarrollo-de-estrategias)
5. [Backtesting](#5-backtesting)
6. [Analisis de Rendimiento](#6-analisis-de-rendimiento)
7. [Optimizacion de Parametros](#7-optimizacion-de-parametros)
8. [Paper Trading](#8-paper-trading)
9. [Trading en Vivo (Multi-Broker)](#9-trading-en-vivo-multi-broker)
10. [Gestion de Portafolio](#10-gestion-de-portafolio)
11. [Gestion de Riesgo](#11-gestion-de-riesgo)
12. [Stress Testing](#12-stress-testing)
13. [API REST](#13-api-rest)
14. [Preguntas Frecuentes](#14-preguntas-frecuentes)

---

## 1. Introduccion

**Trad-Loop** es un framework completo para trading algoritmico que permite:

- Extraer datos historicos de mercado
- Desarrollar y probar estrategias de trading
- Optimizar parametros con multiples algoritmos
- Ejecutar paper trading con ordenes avanzadas
- Operar en vivo con CCXT (crypto) e Interactive Brokers (tradicional)
- Gestionar portafolios multi-activo
- Controlar riesgo profesionalmente
- Realizar stress testing de estrategias

### Flujo de Trabajo Tipico

```
1. Extraer Datos → 2. Desarrollar Estrategia → 3. Backtest → 4. Optimizar
        ↓                                                         ↓
5. Validar (OOS) ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ←
        ↓
6. Paper Trading → 7. Trading en Vivo → 8. Monitoreo y Ajustes
```

---

## 2. Instalacion y Configuracion

### 2.1 Requisitos Previos

- Python 3.9 o superior
- pip (gestor de paquetes)
- Git

### 2.2 Instalacion

```bash
# Clonar repositorio
git clone https://github.com/Pliperkiller/Trad-loop.git
cd Trad-loop

# Crear entorno virtual
python -m venv venv

# Activar entorno
source venv/bin/activate      # Linux/Mac
# venv\Scripts\activate       # Windows

# Instalar dependencias
pip install -r StrategyTrader/requirements.txt
pip install -r DataExtractor/requirements.txt
```

### 2.3 Configuracion de API Keys

Crear archivo `.env` en la raiz del proyecto:

```env
# Crypto Exchanges (CCXT)
BINANCE_API_KEY=tu_api_key
BINANCE_API_SECRET=tu_api_secret
BYBIT_API_KEY=tu_api_key
BYBIT_API_SECRET=tu_api_secret

# Interactive Brokers
IBKR_HOST=127.0.0.1
IBKR_PORT=7497          # TWS Paper: 7497, TWS Live: 7496
IBKR_CLIENT_ID=1

# Data APIs
COINGECKO_API_KEY=tu_api_key
GLASSNODE_API_KEY=tu_api_key
```

### 2.4 Verificar Instalacion

```bash
# Verificar imports
python -c "from src.strategy import TradingStrategy; print('OK')"

# Ejecutar tests
pytest --version
pytest tests/ -v --tb=short
```

---

## 3. Extraccion de Datos (DataExtractor)

### 3.1 Interfaz Grafica (GUI)

```bash
cd DataExtractor
python main.py
```

La GUI permite:
- Seleccionar exchange (Binance, Kraken, etc.)
- Elegir par de trading (BTC/USDT, ETH/USD)
- Definir rango de fechas
- Seleccionar timeframe (1m, 5m, 1h, 1d)
- Exportar a CSV

### 3.2 Linea de Comandos (CLI)

```bash
cd DataExtractor
python cli_extract.py --exchange binance --symbol BTC/USDT --timeframe 1h --start 2024-01-01 --end 2024-12-31 --output data/btc_usdt_1h.csv
```

### 3.3 Uso Programatico

```python
from src.infrastructure.exchanges.binance_exchange import BinanceExchange
from src.infrastructure.exchanges.csv_exporter import CSVExporter
from datetime import datetime

# Crear exchange
exchange = BinanceExchange()

# Extraer datos
candles = exchange.fetch_ohlcv(
    symbol="BTC/USDT",
    timeframe="1h",
    since=datetime(2024, 1, 1),
    until=datetime(2024, 12, 31)
)

# Exportar a CSV
exporter = CSVExporter()
exporter.export(candles, "data/btc_usdt_1h.csv")
```

### 3.4 Formato de Datos

El CSV resultante tiene el siguiente formato:

```csv
timestamp,open,high,low,close,volume
2024-01-01 00:00:00,42000.50,42100.00,41900.00,42050.25,1500.5
2024-01-01 01:00:00,42050.25,42200.00,42000.00,42150.00,1800.3
...
```

---

## 4. Desarrollo de Estrategias

### 4.1 Estructura Basica

```python
from src.strategy import TradingStrategy, TradeSignal, TechnicalIndicators
import pandas as pd

class MiEstrategia(TradingStrategy):
    """
    Estrategia personalizada.

    Debe implementar:
    - calculate_indicators(): Calcular indicadores tecnicos
    - generate_signals(): Generar senales de trading
    """

    def __init__(self, param1=10, param2=20):
        super().__init__()
        self.param1 = param1
        self.param2 = param2

    def calculate_indicators(self):
        """Calcular indicadores tecnicos necesarios."""
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
        """Generar senales de trading."""
        signals = []

        for i in range(len(self.data)):
            row = self.data.iloc[i]

            # Condicion de compra
            if (row['ema_fast'] > row['ema_slow'] and
                row['rsi'] < 70):
                signal = TradeSignal(
                    timestamp=self.data.index[i],
                    signal='BUY',
                    confidence=0.8,
                    metadata={'reason': 'EMA crossover + RSI ok'}
                )
            # Condicion de venta
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

### 4.2 Indicadores Disponibles

```python
from src.strategy import TechnicalIndicators

# Medias moviles
sma = TechnicalIndicators.sma(data['close'], period=20)
ema = TechnicalIndicators.ema(data['close'], period=20)

# Momentum
rsi = TechnicalIndicators.rsi(data['close'], period=14)
macd, signal, hist = TechnicalIndicators.macd(
    data['close'], fast=12, slow=26, signal=9
)

# Volatilidad
upper, middle, lower = TechnicalIndicators.bollinger_bands(
    data['close'], period=20, std=2.0
)
atr = TechnicalIndicators.atr(
    data['high'], data['low'], data['close'], period=14
)

# Osciladores
k, d = TechnicalIndicators.stochastic(
    data['high'], data['low'], data['close'], period=14
)
```

### 4.3 Indicadores Avanzados (Modulo indicators/)

```python
from src.indicators.technical import momentum, trend, volatility, volume
from src.indicators.technical.ichimoku import IchimokuCloud

# Momentum avanzado
cci = momentum.cci(data['high'], data['low'], data['close'], period=20)
williams_r = momentum.williams_r(data['high'], data['low'], data['close'], period=14)

# Trend avanzado
adx_result = trend.adx(data['high'], data['low'], data['close'], period=14)
supertrend = trend.supertrend(data['high'], data['low'], data['close'], period=10, multiplier=3)

# Volatilidad avanzada
keltner = volatility.keltner_channels(data['high'], data['low'], data['close'])
donchian = volatility.donchian_channels(data['high'], data['low'], period=20)

# Volumen
obv = volume.obv(data['close'], data['volume'])
vwap = volume.vwap(data['high'], data['low'], data['close'], data['volume'])
mfi = volume.mfi(data['high'], data['low'], data['close'], data['volume'], period=14)

# Ichimoku
ichimoku = IchimokuCloud()
result = ichimoku.calculate(data['high'], data['low'], data['close'])
# result.tenkan_sen, result.kijun_sen, result.senkou_span_a, etc.
```

### 4.4 Configuracion de Estrategia

```python
strategy = MiEstrategia(param1=12, param2=26)

# Configurar parametros
strategy.config.initial_capital = 10000
strategy.config.risk_per_trade = 0.02      # 2% por trade
strategy.config.stop_loss_pct = 0.02       # 2% stop loss
strategy.config.take_profit_pct = 0.04     # 4% take profit
strategy.config.commission_pct = 0.001     # 0.1% comision
strategy.config.slippage_pct = 0.0005      # 0.05% slippage
```

---

## 5. Backtesting

### 5.1 Backtest Basico

```python
import pandas as pd

# Cargar datos
data = pd.read_csv(
    'data/btc_usdt_1h.csv',
    parse_dates=['timestamp'],
    index_col='timestamp'
)

# Crear estrategia
strategy = MiEstrategia(param1=12, param2=26)
strategy.config.initial_capital = 10000
strategy.config.risk_per_trade = 0.02

# Cargar datos y ejecutar backtest
strategy.load_data(data)
trades = strategy.backtest()

# Ver resultados
print(f"Total trades: {len(trades)}")
print(f"Trades DataFrame:\n{trades.head()}")

# Metricas basicas
metrics = strategy.get_performance_metrics()
print(f"\nTotal Return: {metrics['total_return_pct']:.2f}%")
print(f"Win Rate: {metrics['win_rate']:.2f}%")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
```

### 5.2 Acceso a Equity Curve

```python
# Equity curve
equity = strategy.equity_curve
print(equity.head())

# Graficar equity
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(equity.index, equity['equity'])
plt.title('Equity Curve')
plt.xlabel('Date')
plt.ylabel('Capital ($)')
plt.grid(True)
plt.show()
```

### 5.3 Analisis de Trades

```python
# DataFrame de trades
trades_df = strategy.get_trades_dataframe()

# Filtrar trades ganadores
winners = trades_df[trades_df['pnl'] > 0]
losers = trades_df[trades_df['pnl'] < 0]

print(f"Trades ganadores: {len(winners)}")
print(f"Trades perdedores: {len(losers)}")
print(f"PnL promedio ganadores: ${winners['pnl'].mean():.2f}")
print(f"PnL promedio perdedores: ${losers['pnl'].mean():.2f}")
```

---

## 6. Analisis de Rendimiento

### 6.1 Performance Analyzer

```python
from src.performance import PerformanceAnalyzer, PerformanceVisualizer

# Crear analyzer
analyzer = PerformanceAnalyzer(
    equity_curve=strategy.equity_curve,
    trades_df=trades,
    initial_capital=10000
)

# Calcular todas las metricas (30+)
metrics = analyzer.calculate_all_metrics()

# Metricas de rentabilidad
print("=== RENTABILIDAD ===")
print(f"Total Return: {metrics['total_return_pct']:.2f}%")
print(f"CAGR: {metrics['cagr_pct']:.2f}%")
print(f"Expectancy: ${metrics['expectancy']:.2f}")

# Metricas de riesgo
print("\n=== RIESGO ===")
print(f"Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
print(f"Max DD Duration: {metrics['max_drawdown_duration']} days")
print(f"Volatility: {metrics['volatility_pct']:.2f}%")
print(f"VaR 95%: {metrics['value_at_risk_95']:.2f}%")
print(f"CVaR 95%: {metrics['conditional_var_95']:.2f}%")

# Ratios ajustados por riesgo
print("\n=== RATIOS ===")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
print(f"Sortino Ratio: {metrics['sortino_ratio']:.2f}")
print(f"Calmar Ratio: {metrics['calmar_ratio']:.2f}")
print(f"Omega Ratio: {metrics['omega_ratio']:.2f}")

# Consistencia
print("\n=== CONSISTENCIA ===")
print(f"Win Rate: {metrics['win_rate_pct']:.2f}%")
print(f"Profit Factor: {metrics['profit_factor']:.2f}")
print(f"Risk/Reward: {metrics['risk_reward_ratio']:.2f}")
print(f"Recovery Factor: {metrics['recovery_factor']:.2f}")
```

### 6.2 Visualizacion

```python
# Crear visualizador
visualizer = PerformanceVisualizer(analyzer)

# Dashboard completo (6 paneles)
visualizer.plot_comprehensive_dashboard()

# Graficos individuales
visualizer.plot_equity_curve()
visualizer.plot_drawdown()
visualizer.plot_monthly_returns()
visualizer.plot_trade_distribution()
visualizer.plot_rolling_metrics()
```

### 6.3 Dashboard Completo

```python
# El dashboard incluye:
# 1. Equity Curve con benchmark
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

## 7. Optimizacion de Parametros

### 7.1 Configuracion Basica

```python
from src.optimizer import StrategyOptimizer

# Crear optimizador
optimizer = StrategyOptimizer(
    strategy_class=MiEstrategia,
    data=data,
    initial_capital=10000
)

# Agregar parametros a optimizar
optimizer.add_parameter('param1', 'int', 5, 30, step=5)    # EMA rapida
optimizer.add_parameter('param2', 'int', 20, 60, step=10)  # EMA lenta

# Definir objetivo
# Opciones: 'sharpe_ratio', 'profit_factor', 'total_return', 'calmar_ratio'
objective = 'sharpe_ratio'
```

### 7.2 Grid Search

```python
# Busqueda exhaustiva de todas las combinaciones
results = optimizer.grid_optimize(objective=objective)

print(f"Mejores parametros: {results.best_params}")
print(f"Mejor {objective}: {results.best_value:.4f}")
print(f"Total combinaciones: {len(results.all_results)}")
```

### 7.3 Random Search

```python
# Busqueda aleatoria (util para muchos parametros)
results = optimizer.random_optimize(
    n_iterations=100,
    objective=objective
)

print(f"Mejores parametros: {results.best_params}")
```

### 7.4 Optimizacion Bayesiana

```python
# Optimizacion inteligente con Gaussian Process
results = optimizer.bayesian_optimize(
    n_iterations=50,
    objective=objective,
    n_initial_points=10  # Puntos aleatorios iniciales
)

print(f"Mejores parametros: {results.best_params}")

# Ver convergencia
import matplotlib.pyplot as plt
plt.plot(results.convergence)
plt.xlabel('Iteration')
plt.ylabel(objective)
plt.title('Bayesian Optimization Convergence')
plt.show()
```

### 7.5 Algoritmo Genetico

```python
# Optimizacion evolutiva
results = optimizer.genetic_optimize(
    population_size=50,
    n_generations=20,
    mutation_rate=0.1,
    crossover_rate=0.8,
    objective=objective
)

print(f"Mejores parametros: {results.best_params}")
```

### 7.6 Walk-Forward Analysis

```python
# Validacion robusta con walk-forward
results = optimizer.walk_forward_optimize(
    n_splits=5,           # Numero de splits
    train_ratio=0.8,      # 80% train, 20% test
    objective=objective
)

print(f"IS Sharpe promedio: {results.is_metrics['sharpe_ratio'].mean():.2f}")
print(f"OOS Sharpe promedio: {results.oos_metrics['sharpe_ratio'].mean():.2f}")

# Verificar overfitting
if results.oos_metrics['sharpe_ratio'].mean() < results.is_metrics['sharpe_ratio'].mean() * 0.5:
    print("ADVERTENCIA: Posible overfitting detectado")
```

### 7.7 Analisis de Overfitting

```python
from src.optimizers.analysis.overfitting_detection import OverfittingDetector

detector = OverfittingDetector()

# Comparar IS vs OOS
report = detector.analyze(
    is_results=results.is_results,
    oos_results=results.oos_results
)

print(f"Degradacion de rendimiento: {report.performance_degradation:.2f}%")
print(f"Estabilidad de parametros: {report.parameter_stability:.2f}")
print(f"Riesgo de overfitting: {report.overfitting_risk}")
```

---

## 8. Paper Trading

### 8.1 Configuracion

```python
from src.paper_trading import PaperTradingEngine, PaperTradingConfig
from src.paper_trading import RealtimeStrategy

# Configuracion
config = PaperTradingConfig(
    initial_capital=10000,
    commission_pct=0.001,
    slippage_pct=0.0005,
    enable_shorting=True,
    max_positions=5
)

# Crear engine
engine = PaperTradingEngine(config)
```

### 8.2 Estrategia Realtime

```python
class MiEstrategiaRealtime(RealtimeStrategy):
    def __init__(self):
        super().__init__()
        self.ema_fast = []
        self.ema_slow = []

    def on_start(self):
        """Llamado al iniciar el engine."""
        print("Estrategia iniciada")

    def on_candle(self, candle):
        """Llamado con cada nueva vela."""
        # Actualizar indicadores
        self.update_emas(candle.close)

        # Generar senal
        if len(self.ema_fast) < 2:
            return

        # Crossover alcista
        if (self.ema_fast[-1] > self.ema_slow[-1] and
            self.ema_fast[-2] <= self.ema_slow[-2]):

            self.buy(
                price=candle.close,
                quantity=0.1,
                stop_loss=candle.close * 0.98,
                take_profit=candle.close * 1.04
            )

        # Crossover bajista
        elif (self.ema_fast[-1] < self.ema_slow[-1] and
              self.ema_fast[-2] >= self.ema_slow[-2]):

            self.close_all_positions()

    def on_tick(self, symbol, price):
        """Llamado con cada tick (opcional)."""
        pass

    def on_stop(self):
        """Llamado al detener el engine."""
        print("Estrategia detenida")
```

### 8.3 Ejecutar Paper Trading

```python
# Registrar estrategia
engine.register_strategy(MiEstrategiaRealtime())

# Modo backtest (con datos historicos)
backtest_result = engine.run_backtest(data)

# Modo live (datos en tiempo real)
await engine.run_live()
```

### 8.4 Tipos de Ordenes Avanzadas

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
    participation_rate=0.1  # 10% del volumen
)

# Iceberg
iceberg = IcebergOrder(
    symbol="BTC/USDT",
    side="BUY",
    total_quantity=10.0,
    visible_quantity=1.0  # Mostrar solo 1.0
)
```

---

## 9. Trading en Vivo (Multi-Broker)

### 9.1 Configuracion de Brokers

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

# Registrar broker crypto (CCXT)
ccxt_broker = CCXTBroker(
    exchange_id="binance",
    api_key="tu_api_key",
    api_secret="tu_api_secret",
    testnet=True  # Usar testnet para pruebas
)
executor.register_broker(ccxt_broker)

# Registrar broker tradicional (IBKR)
ibkr_broker = IBKRBroker(
    host="127.0.0.1",
    port=7497,       # TWS Paper
    client_id=1
)
executor.register_broker(ibkr_broker)
```

### 9.2 Conectar y Operar

```python
# Conectar todos los brokers
await executor.connect_all()

# El ruteo es automatico basado en el simbolo:
# - BTC/USDT, ETH/BTC → CCXT (crypto)
# - AAPL, MSFT → IBKR (stocks)
# - EUR/USD → IBKR (forex)

# Orden crypto
crypto_order = BrokerOrder(
    symbol="BTC/USDT",
    side=OrderSide.BUY,
    order_type=OrderType.LIMIT,
    quantity=0.1,
    price=50000
)
report = await executor.submit_order(crypto_order)
print(f"Order ID: {report.order_id}, Status: {report.status}")

# Orden stock (automaticamente va a IBKR)
stock_order = BrokerOrder(
    symbol="AAPL",
    side=OrderSide.BUY,
    order_type=OrderType.MARKET,
    quantity=10
)
report = await executor.submit_order(stock_order)
```

### 9.3 Gestion de Posiciones

```python
# Obtener posiciones de todos los brokers
all_positions = await executor.get_all_positions()

for broker_type, positions in all_positions.items():
    print(f"\n{broker_type.value}:")
    for pos in positions:
        print(f"  {pos.symbol}: {pos.quantity} @ {pos.entry_price}")
        print(f"  Unrealized PnL: ${pos.unrealized_pnl:.2f}")

# Obtener balances
balances = await executor.get_all_balances()
print(f"\nBalances:")
for broker_type, balance in balances.items():
    print(f"  {broker_type.value}: {balance}")
```

### 9.4 Context Manager

```python
# Uso con context manager (auto-connect/disconnect)
async with UnifiedExecutor() as executor:
    executor.register_broker(ccxt_broker)
    executor.register_broker(ibkr_broker)

    # Operar...
    await executor.submit_order(order)

# Desconexion automatica al salir
```

---

## 10. Gestion de Portafolio

### 10.1 Configuracion

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

### 10.2 Metodos de Asignacion

```python
from src.portfolio import AllocationMethod

# Peso igual
config.allocation_method = AllocationMethod.EQUAL_WEIGHT

# Por capitalizacion de mercado
config.allocation_method = AllocationMethod.MARKET_CAP_WEIGHT

# Risk Parity (igual contribucion al riesgo)
config.allocation_method = AllocationMethod.RISK_PARITY

# Volatilidad inversa
config.allocation_method = AllocationMethod.INVERSE_VOLATILITY

# Minima varianza
config.allocation_method = AllocationMethod.MINIMUM_VARIANCE

# Maximo Sharpe
config.allocation_method = AllocationMethod.MAXIMUM_SHARPE

# Hierarchical Risk Parity
config.allocation_method = AllocationMethod.HRP
```

### 10.3 Backtest de Portafolio

```python
# Cargar datos multi-activo
portfolio_data = {
    'BTC/USDT': pd.read_csv('data/btc.csv', index_col='timestamp', parse_dates=True),
    'ETH/USDT': pd.read_csv('data/eth.csv', index_col='timestamp', parse_dates=True),
    'SOL/USDT': pd.read_csv('data/sol.csv', index_col='timestamp', parse_dates=True),
}

# Ejecutar backtest
result = manager.backtest(portfolio_data)

# Resultados
print(f"Portfolio Return: {result.metrics['total_return']:.2f}%")
print(f"Portfolio Sharpe: {result.metrics['sharpe_ratio']:.2f}")
print(f"Portfolio Volatility: {result.metrics['volatility']:.2f}%")

# Historial de asignaciones
print(result.allocation_history.tail())
```

### 10.4 Rebalanceo

```python
# Obtener asignacion actual
current_allocation = manager.get_current_allocation()
print(f"Asignacion actual: {current_allocation}")

# Obtener asignacion objetivo
target_allocation = manager.get_target_allocation()
print(f"Asignacion objetivo: {target_allocation}")

# Calcular trades necesarios para rebalancear
rebalance_trades = manager.calculate_rebalance_trades()
for trade in rebalance_trades:
    print(f"{trade['action']} {trade['quantity']} {trade['symbol']}")

# Ejecutar rebalanceo
await manager.execute_rebalance()
```

---

## 11. Gestion de Riesgo

### 11.1 Configuracion

```python
from src.risk_management import RiskManager, RiskConfig

config = RiskConfig(
    max_position_size_pct=0.10,      # Max 10% por posicion
    max_portfolio_risk_pct=0.20,     # Max 20% riesgo total
    daily_loss_limit_pct=0.05,       # Max 5% perdida diaria
    max_drawdown_pct=0.15,           # Max 15% drawdown
    max_correlation=0.7              # Max correlacion entre posiciones
)

risk_manager = RiskManager(config)
```

### 11.2 Position Sizing

```python
from src.risk_management import PositionSizer, SizingMethod

sizer = PositionSizer(
    method=SizingMethod.FIXED_FRACTIONAL,
    risk_per_trade=0.02  # 2% del capital
)

# Calcular tamano de posicion
position_size = sizer.calculate(
    capital=10000,
    entry_price=50000,
    stop_loss=49000
)
print(f"Tamano de posicion: {position_size:.4f} BTC")

# Kelly Criterion
kelly_sizer = PositionSizer(
    method=SizingMethod.KELLY_CRITERION,
    win_rate=0.55,
    avg_win=100,
    avg_loss=80
)
kelly_size = kelly_sizer.calculate(capital=10000)
print(f"Kelly sugiere: {kelly_size:.2f}% del capital")
```

### 11.3 Evaluacion de Trade

```python
# Evaluar si un trade cumple con las reglas de riesgo
assessment = risk_manager.assess_trade(
    symbol="BTC/USDT",
    side="BUY",
    quantity=0.1,
    entry_price=50000,
    stop_loss=49000
)

if assessment.approved:
    print("Trade aprobado")
    print(f"Riesgo: ${assessment.risk_amount:.2f}")
    print(f"% del capital: {assessment.risk_pct:.2f}%")
else:
    print(f"Trade rechazado: {assessment.rejection_reason}")
```

### 11.4 Monitoreo de Correlacion

```python
from src.risk_management import CorrelationManager

corr_manager = CorrelationManager()

# Actualizar con datos recientes
corr_manager.update(portfolio_data)

# Obtener matriz de correlacion
corr_matrix = corr_manager.get_correlation_matrix()
print(corr_matrix)

# Detectar concentracion de riesgo
clusters = corr_manager.get_correlation_clusters(threshold=0.7)
print(f"Clusters de alta correlacion: {clusters}")
```

---

## 12. Stress Testing

### 12.1 Monte Carlo

```python
from src.stress_testing import StressTester, MonteCarloSimulator

tester = StressTester()

# Simulacion Monte Carlo
mc_result = tester.monte_carlo(
    strategy=strategy,
    n_simulations=1000,
    n_periods=252  # 1 ano
)

print(f"Return medio: {mc_result.mean_return:.2f}%")
print(f"Return percentil 5%: {mc_result.percentile_5:.2f}%")
print(f"Return percentil 95%: {mc_result.percentile_95:.2f}%")
print(f"Max Drawdown medio: {mc_result.mean_max_drawdown:.2f}%")
print(f"Prob. de ruina (<-50%): {mc_result.ruin_probability:.2f}%")
```

### 12.2 Analisis de Escenarios

```python
from src.stress_testing import ScenarioAnalyzer

analyzer = ScenarioAnalyzer()

# Escenarios predefinidos
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

### 12.3 Analisis de Sensibilidad

```python
from src.stress_testing import SensitivityAnalyzer

sensitivity = SensitivityAnalyzer()

# Analizar sensibilidad a parametros
result = sensitivity.analyze(
    strategy_class=MiEstrategia,
    data=data,
    parameter='param1',
    range_pct=0.20  # +/- 20%
)

print(f"Sensibilidad de {result.parameter}:")
print(f"  Impacto en Sharpe: {result.sharpe_sensitivity:.4f}")
print(f"  Impacto en Return: {result.return_sensitivity:.4f}")

# Graficar
sensitivity.plot_sensitivity(result)
```

---

## 13. API REST

### 13.1 Iniciar Servidor

```bash
cd StrategyTrader
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```

### 13.2 Endpoints Disponibles

| Endpoint | Metodo | Descripcion |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/api/v1/strategies` | GET | Listar estrategias registradas |
| `/api/v1/trades/{strategy_id}` | GET | Obtener trades de una estrategia |
| `/api/v1/performance/{strategy_id}` | GET | Metricas de rendimiento |
| `/api/v1/equity/{strategy_id}` | GET | Curva de equity |

### 13.3 Ejemplos de Uso

```bash
# Health check
curl http://localhost:8000/

# Listar estrategias
curl http://localhost:8000/api/v1/strategies

# Obtener trades (con filtros)
curl "http://localhost:8000/api/v1/trades/strategy1?start_time=2024-01-01&profitable_only=true"

# Obtener performance
curl http://localhost:8000/api/v1/performance/strategy1

# Obtener equity curve
curl http://localhost:8000/api/v1/equity/strategy1
```

### 13.4 Documentacion Interactiva

Swagger UI disponible en: `http://localhost:8000/docs`

ReDoc disponible en: `http://localhost:8000/redoc`

### 13.5 Registrar Estrategia via API

```python
import requests

# Desde tu codigo Python
from src.api import register_strategy

strategy = MiEstrategia()
strategy.load_data(data)
strategy.backtest()

register_strategy("mi_estrategia", strategy)

# Ahora accesible via API
# GET /api/v1/trades/mi_estrategia
```

---

## 14. Preguntas Frecuentes

### P: Como evito el overfitting?

**R:** Usa validacion walk-forward y compara metricas IS vs OOS:

```python
results = optimizer.walk_forward_optimize(n_splits=5)
if results.oos_sharpe < results.is_sharpe * 0.6:
    print("Posible overfitting - reducir complejidad")
```

### P: Cual es el mejor metodo de optimizacion?

**R:** Depende del caso:
- **Pocos parametros (<5)**: Grid Search
- **Muchos parametros**: Bayesian o Genetic
- **Validacion robusta**: Walk-Forward

### P: Como manejo ordenes parcialmente ejecutadas?

**R:** El broker bridge maneja esto automaticamente:

```python
report = await executor.submit_order(order)
if report.status == OrderStatus.PARTIAL:
    print(f"Ejecutado: {report.filled_quantity} de {order.quantity}")
```

### P: Puedo usar multiples estrategias simultaneamente?

**R:** Si, con el PortfolioManager:

```python
strategies = {
    'momentum': MomentumStrategy(),
    'mean_reversion': MeanReversionStrategy()
}
manager = PortfolioManager(strategies, allocation='equal')
```

### P: Como configuro alertas?

**R:** Implementa callbacks en tu estrategia:

```python
def on_trade(self, trade):
    if trade.pnl < -100:
        send_alert(f"Perdida significativa: ${trade.pnl}")
```

### P: Soporta futuros y opciones?

**R:** Si, via Interactive Brokers:

```python
# Futuros
order = BrokerOrder(symbol="ES2403", ...)  # E-mini S&P 500

# Opciones (formato OCC)
order = BrokerOrder(symbol="AAPL240315C00175000", ...)
```

---

## Soporte

- **Issues**: https://github.com/Pliperkiller/Trad-loop/issues
- **Documentacion**: `/docs/` en el repositorio

---

*Manual de Uso v1.0 - Trad-Loop*
