# Portfolio Management - Manual de Componente

## Descripcion

El modulo **portfolio** proporciona herramientas para gestion de portafolios multi-activo con diversos metodos de asignacion y rebalanceo.

## Arquitectura

```
portfolio/
├── portfolio_manager.py   # Orquestador principal
├── models.py              # PortfolioConfig, PortfolioState
├── allocator.py           # Metodos de asignacion
├── rebalancer.py          # Estrategias de rebalanceo
├── backtester.py          # Backtesting de portafolio
└── metrics.py             # Metricas de portafolio
```

## Uso Basico

### Configuracion

```python
from src.portfolio import PortfolioManager, PortfolioConfig, AllocationMethod

config = PortfolioConfig(
    initial_capital=100000,
    symbols=['BTC/USDT', 'ETH/USDT', 'SOL/USDT'],
    allocation_method=AllocationMethod.RISK_PARITY,
    rebalance_frequency='monthly',
    min_position_weight=0.05,     # Min 5% por activo
    max_position_weight=0.40,     # Max 40% por activo
    transaction_cost=0.001        # 0.1% comision
)

manager = PortfolioManager(config)
```

## Metodos de Asignacion

### Equal Weight

```python
config.allocation_method = AllocationMethod.EQUAL_WEIGHT

# Resultado: 33.3% BTC, 33.3% ETH, 33.3% SOL
```

### Market Cap Weight

```python
config.allocation_method = AllocationMethod.MARKET_CAP_WEIGHT

# Pesos basados en capitalizacion de mercado
# Requiere datos de market cap
manager.set_market_caps({
    'BTC/USDT': 800_000_000_000,
    'ETH/USDT': 300_000_000_000,
    'SOL/USDT': 50_000_000_000
})
```

### Risk Parity

Cada activo contribuye igual al riesgo total.

```python
config.allocation_method = AllocationMethod.RISK_PARITY

# Activos mas volatiles reciben menos peso
# para igualar contribucion al riesgo
```

### Inverse Volatility

Peso inverso a la volatilidad.

```python
config.allocation_method = AllocationMethod.INVERSE_VOLATILITY

# volatility_weight[i] = 1/vol[i] / sum(1/vol)
```

### Minimum Variance

Optimizacion de Markowitz para minima varianza.

```python
config.allocation_method = AllocationMethod.MINIMUM_VARIANCE

# Minimiza: w' * Cov * w
# Sujeto a: sum(w) = 1, w >= 0
```

### Maximum Sharpe

Optimizacion para maximo Sharpe Ratio.

```python
config.allocation_method = AllocationMethod.MAXIMUM_SHARPE

# Maximiza: (w' * returns - rf) / sqrt(w' * Cov * w)
```

### Risk Budgeting

Asignacion con presupuesto de riesgo especifico.

```python
from src.portfolio import RiskBudgetAllocator

allocator = RiskBudgetAllocator()
weights = allocator.allocate(
    returns=historical_returns,
    risk_budget={
        'BTC/USDT': 0.5,   # 50% del riesgo
        'ETH/USDT': 0.3,   # 30% del riesgo
        'SOL/USDT': 0.2    # 20% del riesgo
    }
)
```

### Hierarchical Risk Parity (HRP)

Metodo avanzado basado en clustering jerarquico.

```python
config.allocation_method = AllocationMethod.HRP

# 1. Calcula matriz de correlacion
# 2. Clustering jerarquico
# 3. Asigna pesos recursivamente
```

## Rebalanceo

### Por Frecuencia

```python
config.rebalance_frequency = 'daily'    # Diario
config.rebalance_frequency = 'weekly'   # Semanal
config.rebalance_frequency = 'monthly'  # Mensual
config.rebalance_frequency = 'quarterly'# Trimestral
```

### Por Threshold (Drift)

```python
from src.portfolio import ThresholdRebalancer

rebalancer = ThresholdRebalancer(
    drift_threshold=0.05  # Rebalancear si drift > 5%
)

# El drift se calcula como:
# drift = max(|actual_weight - target_weight|)
```

### Por Volatilidad

```python
from src.portfolio import VolatilityTargetRebalancer

rebalancer = VolatilityTargetRebalancer(
    target_volatility=0.15,  # 15% anualizado
    lookback=20              # Ventana de calculo
)
```

## Backtesting de Portafolio

```python
# Cargar datos multi-activo
portfolio_data = {
    'BTC/USDT': pd.read_csv('btc.csv', index_col='timestamp', parse_dates=True),
    'ETH/USDT': pd.read_csv('eth.csv', index_col='timestamp', parse_dates=True),
    'SOL/USDT': pd.read_csv('sol.csv', index_col='timestamp', parse_dates=True),
}

# Ejecutar backtest
result = manager.backtest(portfolio_data)

# Resultados
print(f"Total Return: {result.metrics['total_return']:.2f}%")
print(f"Annualized Return: {result.metrics['annualized_return']:.2f}%")
print(f"Volatility: {result.metrics['volatility']:.2f}%")
print(f"Sharpe Ratio: {result.metrics['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {result.metrics['max_drawdown']:.2f}%")
print(f"Calmar Ratio: {result.metrics['calmar_ratio']:.2f}")

# Equity curve
result.equity_curve.plot(title='Portfolio Equity')

# Historial de asignaciones
print(result.allocation_history.tail())

# Fechas de rebalanceo
print(f"Rebalances: {len(result.rebalance_dates)}")
```

## Metricas de Portafolio

```python
from src.portfolio import PortfolioMetrics

metrics = PortfolioMetrics()

# Calcular todas las metricas
report = metrics.calculate(result.equity_curve, result.weights_history)

# Metricas basicas
print(f"Return: {report['total_return']:.2f}%")
print(f"Volatility: {report['volatility']:.2f}%")
print(f"Sharpe: {report['sharpe_ratio']:.2f}")

# Metricas de diversificacion
print(f"Diversification Ratio: {report['diversification_ratio']:.2f}")
print(f"Concentration (HHI): {report['hhi']:.4f}")

# Contribucion al riesgo por activo
for asset, contribution in report['risk_contribution'].items():
    print(f"{asset}: {contribution*100:.1f}%")

# Matriz de correlacion
print("\nCorrelation Matrix:")
print(report['correlation_matrix'])
```

## Operaciones en Vivo

### Obtener Asignacion Actual

```python
# Asignacion actual
current = manager.get_current_allocation()
print(f"Current weights: {current}")

# Asignacion objetivo
target = manager.get_target_allocation()
print(f"Target weights: {target}")

# Drift
drift = manager.calculate_drift()
print(f"Current drift: {drift:.2%}")
```

### Calcular Trades para Rebalanceo

```python
# Trades necesarios
trades = manager.calculate_rebalance_trades(current_prices)

for trade in trades:
    print(f"{trade['action']} {trade['quantity']:.4f} {trade['symbol']}")
    print(f"  Value: ${trade['value']:.2f}")
```

### Ejecutar Rebalanceo

```python
# Con UnifiedExecutor
from src.broker_bridge import UnifiedExecutor

executor = UnifiedExecutor()
# ... configurar brokers

# Ejecutar rebalanceo
await manager.execute_rebalance(executor)
```

## Analisis Avanzado

### Efficient Frontier

```python
from src.portfolio import EfficientFrontier

ef = EfficientFrontier(returns=historical_returns)

# Calcular frontera
frontier = ef.calculate(n_points=100)

# Graficar
ef.plot_frontier()

# Portafolio de minima varianza
min_var = ef.get_minimum_variance_portfolio()

# Portafolio de maximo Sharpe
max_sharpe = ef.get_maximum_sharpe_portfolio()
```

### Stress Testing de Portafolio

```python
from src.portfolio import PortfolioStressTester

stress_tester = PortfolioStressTester(manager)

# Escenarios
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

# Analisis de contribucion
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

| Metodo | Descripcion |
|--------|-------------|
| `backtest(data)` | Ejecutar backtest |
| `get_current_allocation()` | Asignacion actual |
| `get_target_allocation()` | Asignacion objetivo |
| `calculate_drift()` | Calcular drift |
| `calculate_rebalance_trades(prices)` | Trades necesarios |
| `execute_rebalance(executor)` | Ejecutar rebalanceo |

### PortfolioConfig

| Campo | Descripcion |
|-------|-------------|
| `initial_capital` | Capital inicial |
| `symbols` | Lista de simbolos |
| `allocation_method` | Metodo de asignacion |
| `rebalance_frequency` | Frecuencia de rebalanceo |
| `min_position_weight` | Peso minimo por posicion |
| `max_position_weight` | Peso maximo por posicion |
| `transaction_cost` | Costo de transaccion |

### PortfolioResult

| Campo | Descripcion |
|-------|-------------|
| `equity_curve` | Serie temporal de equity |
| `weights_history` | Historial de pesos |
| `allocation_history` | Historial de asignaciones |
| `rebalance_dates` | Fechas de rebalanceo |
| `metrics` | Diccionario de metricas |
| `trades` | Lista de trades ejecutados |

## Tests

```bash
pytest tests/portfolio/ -v
```
