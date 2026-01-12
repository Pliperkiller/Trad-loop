# Risk Management - Manual de Componente

## Descripcion

El modulo **risk_management** proporciona herramientas profesionales para gestion de riesgo en trading algoritmico.

## Arquitectura

```
risk_management/
├── risk_manager.py        # Orquestador principal
├── models.py              # Modelos de datos
├── config.py              # Configuracion
├── position_sizer.py      # Dimensionamiento de posiciones
├── risk_limits.py         # Limites de riesgo
└── correlation_manager.py # Gestion de correlacion
```

## Uso Basico

### Configuracion

```python
from src.risk_management import RiskManager, RiskConfig

config = RiskConfig(
    max_position_size_pct=0.10,      # Max 10% por posicion
    max_portfolio_risk_pct=0.20,     # Max 20% riesgo total
    max_correlation=0.7,              # Max correlacion
    daily_loss_limit_pct=0.05,       # Max 5% perdida diaria
    max_drawdown_pct=0.15,           # Max 15% drawdown
    max_consecutive_losses=5,         # Max perdidas consecutivas
    risk_free_rate=0.02              # Tasa libre de riesgo
)

risk_manager = RiskManager(config)
```

## Position Sizing

### Fixed Fractional

Arriesgar un porcentaje fijo del capital por trade.

```python
from src.risk_management import PositionSizer, SizingMethod

sizer = PositionSizer(
    method=SizingMethod.FIXED_FRACTIONAL,
    risk_per_trade=0.02  # 2% del capital
)

# Calcular tamano
position_size = sizer.calculate(
    capital=10000,
    entry_price=50000,
    stop_loss=49000  # 2% de riesgo en precio
)

# Resultado: cantidad que arriesga 2% del capital
print(f"Position size: {position_size:.4f} BTC")
print(f"Risk amount: ${10000 * 0.02:.2f}")
```

### Kelly Criterion

Dimensionamiento optimo basado en edge y probabilidades.

```python
sizer = PositionSizer(
    method=SizingMethod.KELLY_CRITERION,
    win_rate=0.55,        # 55% win rate
    avg_win=100,          # Ganancia promedio $100
    avg_loss=80,          # Perdida promedio $80
    kelly_fraction=0.5    # Half Kelly (mas conservador)
)

kelly_pct = sizer.calculate_kelly()
print(f"Kelly sugiere: {kelly_pct:.2%} del capital")

# Formula Kelly:
# f = (p * b - q) / b
# donde p = win_rate, q = 1-p, b = avg_win/avg_loss
```

### Optimal-f

Basado en la mayor perdida historica.

```python
sizer = PositionSizer(
    method=SizingMethod.OPTIMAL_F,
    largest_loss=500,     # Mayor perdida historica
    fraction=0.25         # 25% del optimal-f
)

optimal_size = sizer.calculate(capital=10000)
```

### ATR-Based

Dimensionamiento basado en volatilidad (ATR).

```python
sizer = PositionSizer(
    method=SizingMethod.ATR_BASED,
    atr_multiplier=2.0,   # Stop a 2x ATR
    risk_per_trade=0.02
)

position_size = sizer.calculate(
    capital=10000,
    entry_price=50000,
    atr=1000  # ATR actual
)
```

## Risk Limits

### Limites de Posicion

```python
from src.risk_management import RiskLimitChecker

checker = RiskLimitChecker(config)

# Verificar limite de posicion
can_open = checker.check_position_limit(
    symbol="BTC/USDT",
    quantity=0.5,
    price=50000,
    current_portfolio_value=100000
)

if not can_open['approved']:
    print(f"Rechazado: {can_open['reason']}")
```

### Limites de Exposicion

```python
# Verificar exposicion total
exposure_check = checker.check_exposure_limit(
    current_positions=[...],
    portfolio_value=100000
)

print(f"Current Exposure: {exposure_check['current_exposure']:.2%}")
print(f"Limit: {exposure_check['limit']:.2%}")
print(f"Available: {exposure_check['available']:.2%}")
```

### Limite de Perdida Diaria

```python
# Verificar limite diario
daily_check = checker.check_daily_loss_limit(
    today_pnl=-400,
    initial_capital=10000
)

if daily_check['limit_reached']:
    print("ALERTA: Limite de perdida diaria alcanzado")
    print(f"Loss: {daily_check['current_loss']:.2%}")
```

### Proteccion de Drawdown

```python
# Verificar drawdown
dd_check = checker.check_drawdown_limit(
    current_equity=8500,
    peak_equity=10000
)

if dd_check['limit_breached']:
    print("ALERTA: Drawdown maximo excedido")
    print(f"Current DD: {dd_check['current_drawdown']:.2%}")
    print(f"Limit: {dd_check['limit']:.2%}")
```

## Evaluacion de Trade

```python
# Evaluar un trade antes de ejecutar
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

### Monitoreo de Correlacion

```python
from src.risk_management import CorrelationManager

corr_manager = CorrelationManager(
    lookback_period=60,    # 60 periodos
    correlation_threshold=0.7
)

# Actualizar con datos
corr_manager.update(price_data)

# Obtener matriz
corr_matrix = corr_manager.get_correlation_matrix()
print(corr_matrix)
```

### Deteccion de Clusters

```python
# Encontrar activos altamente correlacionados
clusters = corr_manager.get_correlation_clusters(threshold=0.7)

for cluster in clusters:
    print(f"Cluster: {cluster}")
    # Ejemplo: ['BTC/USDT', 'ETH/USDT'] - alta correlacion
```

### Riesgo de Concentracion

```python
# Evaluar riesgo de concentracion
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
    print("ADVERTENCIA: Alta concentracion de riesgo")
```

### Cambio de Regimen

```python
# Detectar cambios en correlaciones
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

### VaR Historico

```python
from src.risk_management import VaRCalculator

var_calc = VaRCalculator()

# VaR historico
var_95 = var_calc.historical_var(
    returns=portfolio_returns,
    confidence=0.95
)
print(f"VaR 95%: {var_95:.2%}")

# Con ventana especifica
var_99 = var_calc.historical_var(
    returns=portfolio_returns[-252:],  # Ultimo ano
    confidence=0.99
)
print(f"VaR 99%: {var_99:.2%}")
```

### VaR Parametrico

```python
# VaR parametrico (asume normalidad)
var_parametric = var_calc.parametric_var(
    mean=portfolio_returns.mean(),
    std=portfolio_returns.std(),
    confidence=0.95
)
print(f"VaR Parametrico 95%: {var_parametric:.2%}")
```

### CVaR (Expected Shortfall)

```python
# CVaR - perdida esperada cuando se excede VaR
cvar = var_calc.conditional_var(
    returns=portfolio_returns,
    confidence=0.95
)
print(f"CVaR 95%: {cvar:.2%}")

# Interpretacion: "En el peor 5% de los casos,
# la perdida promedio es {cvar}%"
```

### VaR de Portafolio

```python
# VaR considerando correlaciones
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

# Generar reporte
report = dashboard.generate_report(
    positions=current_positions,
    equity_curve=equity_curve
)

# Metricas clave
print("=== RISK DASHBOARD ===")
print(f"Total Exposure: {report['total_exposure']:.2%}")
print(f"Portfolio Beta: {report['portfolio_beta']:.2f}")
print(f"VaR 95%: {report['var_95']:.2%}")
print(f"CVaR 95%: {report['cvar_95']:.2%}")
print(f"Current Drawdown: {report['current_drawdown']:.2%}")
print(f"Correlation Risk: {report['correlation_risk']:.2f}")

# Alertas activas
for alert in report['active_alerts']:
    print(f"ALERT: {alert['type']} - {alert['message']}")

# Graficar
dashboard.plot_risk_metrics()
```

## Integracion con Trading

```python
from src.paper_trading import PaperTradingEngine
from src.risk_management import RiskManager

# Configurar risk manager
risk_manager = RiskManager(config)

class RiskAwareStrategy(RealtimeStrategy):
    def __init__(self, risk_manager):
        super().__init__()
        self.risk_manager = risk_manager

    def on_candle(self, candle):
        if self.should_buy(candle):
            # Evaluar riesgo antes de operar
            assessment = self.risk_manager.assess_trade(
                symbol=candle.symbol,
                side="BUY",
                entry_price=candle.close,
                stop_loss=candle.close * 0.98
            )

            if assessment.approved:
                # Usar position size sugerido
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

| Metodo | Descripcion |
|--------|-------------|
| `assess_trade(symbol, side, qty, entry, sl)` | Evaluar trade |
| `check_limits()` | Verificar todos los limites |
| `get_available_risk()` | Riesgo disponible |
| `update_state(positions, equity)` | Actualizar estado |

### PositionSizer

| Metodo | Descripcion |
|--------|-------------|
| `calculate(capital, entry, sl)` | Calcular tamano |
| `calculate_kelly()` | Calcular Kelly % |
| `set_method(method)` | Cambiar metodo |

### RiskConfig

| Campo | Descripcion |
|-------|-------------|
| `max_position_size_pct` | Max % por posicion |
| `max_portfolio_risk_pct` | Max % riesgo total |
| `daily_loss_limit_pct` | Limite perdida diaria |
| `max_drawdown_pct` | Max drawdown |
| `max_correlation` | Max correlacion |
| `max_consecutive_losses` | Max perdidas seguidas |

## Tests

```bash
pytest tests/risk_management/ -v
```
