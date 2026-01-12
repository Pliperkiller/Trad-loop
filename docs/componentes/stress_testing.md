# Stress Testing - Manual de Componente

## Descripcion

El modulo **stress_testing** proporciona herramientas para evaluar la robustez de estrategias mediante simulaciones y analisis de escenarios.

## Arquitectura

```
stress_testing/
├── stress_tester.py       # Orquestador principal
├── models.py              # Modelos de configuracion
├── monte_carlo.py         # Simulacion Monte Carlo
├── scenario_analysis.py   # Analisis de escenarios
└── sensitivity.py         # Analisis de sensibilidad
```

## Monte Carlo

### Simulacion Basica

```python
from src.stress_testing import MonteCarloSimulator

mc = MonteCarloSimulator()

# Simular paths
result = mc.simulate(
    strategy=my_strategy,
    n_simulations=1000,
    n_periods=252,  # 1 ano
    method='bootstrap'
)

# Resultados
print(f"Mean Final Return: {result.mean_return:.2%}")
print(f"Median Final Return: {result.median_return:.2%}")
print(f"Std Dev: {result.std_return:.2%}")
print(f"5th Percentile: {result.percentile_5:.2%}")
print(f"95th Percentile: {result.percentile_95:.2%}")
print(f"Probability of Loss: {result.prob_loss:.2%}")
print(f"Probability of Ruin (<-50%): {result.prob_ruin:.2%}")
```

### Metodos de Simulacion

```python
# Bootstrap (resampling historico)
result = mc.simulate(
    strategy=strategy,
    method='bootstrap',
    block_size=5  # Block bootstrap
)

# Geometric Brownian Motion
result = mc.simulate(
    strategy=strategy,
    method='gbm',
    drift=0.10,        # Drift anual 10%
    volatility=0.20    # Volatilidad 20%
)

# Historical sampling
result = mc.simulate(
    strategy=strategy,
    method='historical',
    with_replacement=True
)
```

### Analisis de Drawdown

```python
# Distribucion de drawdowns
dd_analysis = mc.analyze_drawdowns(result)

print(f"Mean Max DD: {dd_analysis['mean_max_dd']:.2%}")
print(f"Median Max DD: {dd_analysis['median_max_dd']:.2%}")
print(f"95th Percentile DD: {dd_analysis['percentile_95_dd']:.2%}")
print(f"Mean DD Duration: {dd_analysis['mean_dd_duration']} days")
print(f"Max DD Duration: {dd_analysis['max_dd_duration']} days")
```

### Visualizacion

```python
# Graficar paths simulados
mc.plot_simulation_paths(result, n_paths=100)

# Distribucion de returns finales
mc.plot_return_distribution(result)

# Distribucion de drawdowns
mc.plot_drawdown_distribution(result)

# Cone de confianza
mc.plot_confidence_cone(result, confidence_levels=[0.50, 0.75, 0.95])
```

## Analisis de Escenarios

### Escenarios Predefinidos

```python
from src.stress_testing import ScenarioAnalyzer

analyzer = ScenarioAnalyzer()

# Escenarios predefinidos
result = analyzer.run_predefined_scenarios(strategy)

print("\n=== SCENARIO ANALYSIS ===")
for scenario, metrics in result.items():
    print(f"\n{scenario}:")
    print(f"  Return: {metrics['total_return']:.2%}")
    print(f"  Max DD: {metrics['max_drawdown']:.2%}")
    print(f"  Sharpe: {metrics['sharpe_ratio']:.2f}")
```

### Escenarios Personalizados

```python
# Definir escenarios custom
scenarios = {
    'mild_correction': {
        'return_modifier': -0.10,
        'volatility_modifier': 1.2
    },
    'severe_crash': {
        'return_modifier': -0.40,
        'volatility_modifier': 3.0
    },
    'v_recovery': {
        'phases': [
            {'return': -0.30, 'duration': 20},
            {'return': 0.50, 'duration': 40}
        ]
    },
    'prolonged_bear': {
        'return_modifier': -0.02,  # -2% mensual
        'duration_months': 12
    }
}

result = analyzer.run_custom_scenarios(strategy, scenarios)
```

### Escenarios Historicos

```python
# Aplicar eventos historicos reales
historical_scenarios = {
    'covid_crash_2020': {
        'start': '2020-02-20',
        'end': '2020-03-23'
    },
    'crypto_winter_2022': {
        'start': '2022-01-01',
        'end': '2022-12-31'
    },
    'luna_collapse': {
        'start': '2022-05-01',
        'end': '2022-05-15'
    }
}

result = analyzer.apply_historical_scenarios(strategy, historical_scenarios)
```

### Stress por Factor

```python
# Stress testing por factores especificos
factor_stress = analyzer.factor_stress_test(
    strategy=strategy,
    factors={
        'volatility': [0.5, 1.0, 1.5, 2.0, 3.0],
        'correlation': [0.3, 0.5, 0.7, 0.9],
        'liquidity': [0.8, 0.9, 1.0]  # Slippage multiplier
    }
)

# Ver impacto de cada factor
for factor, results in factor_stress.items():
    print(f"\n{factor} impact:")
    for level, metrics in results.items():
        print(f"  {level}: Sharpe={metrics['sharpe']:.2f}, DD={metrics['max_dd']:.2%}")
```

## Analisis de Sensibilidad

### Sensibilidad de Parametros

```python
from src.stress_testing import SensitivityAnalyzer

sensitivity = SensitivityAnalyzer()

# Analisis de un parametro
result = sensitivity.single_parameter(
    strategy_class=MyStrategy,
    data=data,
    parameter='fast_ema',
    base_value=12,
    range_pct=0.30  # +/- 30%
)

print(f"Parameter: {result.parameter}")
print(f"Sharpe Sensitivity: {result.sharpe_sensitivity:.4f}")
print(f"Return Sensitivity: {result.return_sensitivity:.4f}")
print(f"Stability Score: {result.stability_score:.2f}")

# Graficar
sensitivity.plot_sensitivity_curve(result)
```

### Tornado Chart

```python
# Analisis de multiples parametros
tornado = sensitivity.tornado_analysis(
    strategy_class=MyStrategy,
    data=data,
    parameters=['fast_ema', 'slow_ema', 'rsi_period', 'threshold'],
    range_pct=0.20
)

# Ver importancia relativa
for param, impact in tornado.sorted_impacts:
    print(f"{param}: {impact:.2%} impact on Sharpe")

# Graficar tornado chart
sensitivity.plot_tornado(tornado)
```

### Analisis de Interaccion

```python
# Interaccion entre dos parametros
interaction = sensitivity.interaction_analysis(
    strategy_class=MyStrategy,
    data=data,
    param1='fast_ema',
    param2='slow_ema',
    n_levels=10
)

# Heatmap de interaccion
sensitivity.plot_interaction_heatmap(interaction)
```

## Stress Tester Completo

### Reporte Integral

```python
from src.stress_testing import StressTester

tester = StressTester()

# Ejecutar suite completa
report = tester.run_full_suite(
    strategy=my_strategy,
    monte_carlo_sims=1000,
    scenarios='all',
    sensitivity_params=['fast_ema', 'slow_ema', 'rsi_period']
)

# Resumen ejecutivo
print("=== STRESS TEST REPORT ===")
print(f"\nMonte Carlo Results:")
print(f"  Expected Return: {report.monte_carlo.mean_return:.2%}")
print(f"  Return Range (95%): [{report.monte_carlo.percentile_5:.2%}, {report.monte_carlo.percentile_95:.2%}]")
print(f"  Probability of Loss: {report.monte_carlo.prob_loss:.2%}")
print(f"  Expected Max DD: {report.monte_carlo.mean_max_dd:.2%}")

print(f"\nWorst Case Scenarios:")
for scenario in report.worst_scenarios[:3]:
    print(f"  {scenario['name']}: {scenario['return']:.2%}")

print(f"\nParameter Stability:")
print(f"  Most Stable: {report.sensitivity.most_stable}")
print(f"  Least Stable: {report.sensitivity.least_stable}")

print(f"\nRisk Score: {report.overall_risk_score:.2f}/10")
print(f"Recommendation: {report.recommendation}")
```

### Configuracion del Test Suite

```python
from src.stress_testing import StressTestConfig

config = StressTestConfig(
    # Monte Carlo
    n_simulations=5000,
    simulation_method='bootstrap',
    block_size=10,

    # Scenarios
    include_historical=True,
    include_synthetic=True,
    custom_scenarios=my_scenarios,

    # Sensitivity
    sensitivity_range=0.25,
    n_sensitivity_levels=11,

    # Reporting
    confidence_level=0.95,
    output_format='detailed'
)

report = tester.run_full_suite(strategy, config=config)
```

### Exportar Reporte

```python
# Exportar a diferentes formatos
report.to_html('stress_test_report.html')
report.to_pdf('stress_test_report.pdf')
report.to_json('stress_test_report.json')

# Dashboard interactivo
report.launch_dashboard()
```

## API Reference

### StressTester

| Metodo | Descripcion |
|--------|-------------|
| `run_full_suite(strategy, config?)` | Suite completa |
| `monte_carlo(strategy, n_sims)` | Solo Monte Carlo |
| `scenario_analysis(strategy, scenarios)` | Solo escenarios |
| `sensitivity(strategy, params)` | Solo sensibilidad |

### MonteCarloSimulator

| Metodo | Descripcion |
|--------|-------------|
| `simulate(strategy, n_sims, n_periods)` | Ejecutar simulacion |
| `analyze_drawdowns(result)` | Analizar drawdowns |
| `calculate_var(result, confidence)` | Calcular VaR |
| `plot_simulation_paths(result)` | Graficar paths |

### ScenarioAnalyzer

| Metodo | Descripcion |
|--------|-------------|
| `run_predefined_scenarios(strategy)` | Escenarios default |
| `run_custom_scenarios(strategy, scenarios)` | Escenarios custom |
| `apply_historical_scenarios(strategy, events)` | Eventos historicos |
| `factor_stress_test(strategy, factors)` | Stress por factor |

### SensitivityAnalyzer

| Metodo | Descripcion |
|--------|-------------|
| `single_parameter(strategy, param)` | Sensibilidad simple |
| `tornado_analysis(strategy, params)` | Tornado chart |
| `interaction_analysis(strategy, p1, p2)` | Analisis interaccion |

## Tests

```bash
pytest tests/stress_testing/ -v
```
