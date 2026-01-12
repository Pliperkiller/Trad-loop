# Optimizadores - Manual de Componente

## Descripcion

El modulo **optimizers** proporciona algoritmos avanzados para optimizar parametros de estrategias de trading con validacion robusta.

## Arquitectura

```
optimizers/
├── grid_search.py         # Busqueda exhaustiva
├── random_search.py       # Busqueda aleatoria
├── bayesian.py            # Optimizacion Bayesiana
├── genetic.py             # Algoritmo genetico
├── walk_forward.py        # Walk-forward analysis
├── optimization_types.py  # Tipos y configuraciones
├── validation/            # Validacion cruzada
│   ├── splitters.py       # Splitters temporales
│   ├── time_series_cv.py  # Time Series CV
│   ├── purged_kfold.py    # Purged K-Fold
│   └── results.py         # Modelos de resultados
└── analysis/              # Post-optimizacion
    ├── overfitting_detection.py
    ├── parameter_stability.py
    └── visualization.py
```

## Uso Basico

### Configuracion

```python
from src.optimizer import StrategyOptimizer

optimizer = StrategyOptimizer(
    strategy_class=MiEstrategia,
    data=historical_data,
    initial_capital=10000,
    commission_pct=0.001
)

# Definir parametros a optimizar
optimizer.add_parameter('fast_ema', 'int', 5, 30, step=5)
optimizer.add_parameter('slow_ema', 'int', 20, 60, step=10)
optimizer.add_parameter('rsi_period', 'int', 10, 20)
optimizer.add_parameter('rsi_threshold', 'float', 25.0, 35.0, step=2.5)
```

### Tipos de Parametros

```python
# Entero con step
optimizer.add_parameter('period', 'int', 10, 50, step=5)
# Valores: 10, 15, 20, 25, 30, 35, 40, 45, 50

# Entero continuo
optimizer.add_parameter('period', 'int', 10, 50)
# Valores: 10, 11, 12, ..., 49, 50

# Float
optimizer.add_parameter('threshold', 'float', 0.01, 0.05, step=0.01)

# Categorico
optimizer.add_parameter('ma_type', 'categorical', choices=['sma', 'ema', 'wma'])
```

## Algoritmos de Optimizacion

### Grid Search

Busqueda exhaustiva de todas las combinaciones.

```python
results = optimizer.grid_optimize(
    objective='sharpe_ratio',
    n_jobs=-1  # Paralelizacion
)

print(f"Total combinaciones: {results.n_combinations}")
print(f"Mejores parametros: {results.best_params}")
print(f"Mejor Sharpe: {results.best_value:.4f}")
```

**Ventajas**: Completo, deterministico
**Desventajas**: Lento para muchos parametros

### Random Search

Busqueda aleatoria eficiente.

```python
results = optimizer.random_optimize(
    n_iterations=100,
    objective='sharpe_ratio',
    seed=42
)

print(f"Mejores parametros: {results.best_params}")
print(f"Convergencia: {results.convergence[-1]:.4f}")
```

**Ventajas**: Rapido, escalable
**Desventajas**: No garantiza optimo global

### Optimizacion Bayesiana

Usa Gaussian Process para busqueda inteligente.

```python
results = optimizer.bayesian_optimize(
    n_iterations=50,
    objective='sharpe_ratio',
    n_initial_points=10,  # Exploracion inicial
    acq_func='EI'         # Expected Improvement
)

# Funciones de adquisicion disponibles:
# 'EI' - Expected Improvement
# 'PI' - Probability of Improvement
# 'LCB' - Lower Confidence Bound
```

**Ventajas**: Eficiente, converge rapido
**Desventajas**: Mas complejo

### Algoritmo Genetico

Optimizacion evolutiva.

```python
results = optimizer.genetic_optimize(
    population_size=50,
    n_generations=30,
    mutation_rate=0.1,
    crossover_rate=0.8,
    elite_size=5,
    objective='sharpe_ratio'
)

print(f"Generaciones: {results.n_generations}")
print(f"Mejor fitness: {results.best_fitness:.4f}")

# Ver evolucion
import matplotlib.pyplot as plt
plt.plot(results.fitness_history)
plt.xlabel('Generacion')
plt.ylabel('Mejor Fitness')
plt.show()
```

**Ventajas**: Robusto, multi-objetivo posible
**Desventajas**: Lento, requiere tuning

### Walk-Forward Analysis

Validacion out-of-sample continua.

```python
results = optimizer.walk_forward_optimize(
    n_splits=5,
    train_ratio=0.7,      # 70% training
    optimization_method='bayesian',
    n_opt_iterations=30,
    objective='sharpe_ratio'
)

# Resultados por split
for i, split in enumerate(results.splits):
    print(f"\nSplit {i+1}:")
    print(f"  IS Sharpe: {split.is_sharpe:.2f}")
    print(f"  OOS Sharpe: {split.oos_sharpe:.2f}")
    print(f"  Params: {split.best_params}")

# Resumen
print(f"\nIS Sharpe promedio: {results.is_sharpe_mean:.2f}")
print(f"OOS Sharpe promedio: {results.oos_sharpe_mean:.2f}")
print(f"Degradacion: {results.performance_degradation:.2f}%")
```

## Validacion Cruzada

### Time Series Split

```python
from src.optimizers.validation import TimeSeriesSplit

splitter = TimeSeriesSplit(n_splits=5)

for train_idx, test_idx in splitter.split(data):
    train = data.iloc[train_idx]
    test = data.iloc[test_idx]
    # Entrenar y evaluar
```

### Rolling Window Split

```python
from src.optimizers.validation import RollingWindowSplit

splitter = RollingWindowSplit(
    train_size=252,    # 1 ano training
    test_size=63,      # 3 meses test
    step=21            # Avanzar 1 mes
)
```

### Expanding Window Split

```python
from src.optimizers.validation import ExpandingWindowSplit

splitter = ExpandingWindowSplit(
    min_train_size=252,
    test_size=63,
    step=21
)
```

### Purged K-Fold

K-Fold con purga temporal para evitar data leakage.

```python
from src.optimizers.validation import PurgedKFold

splitter = PurgedKFold(
    n_splits=5,
    purge_gap=5  # 5 periodos de gap entre train/test
)
```

## Objetivos de Optimizacion

```python
# Objetivos disponibles
objectives = [
    'sharpe_ratio',      # Sharpe Ratio (default)
    'sortino_ratio',     # Sortino Ratio
    'calmar_ratio',      # Calmar Ratio
    'profit_factor',     # Profit Factor
    'total_return',      # Return Total %
    'max_drawdown',      # Max Drawdown % (minimizar)
    'win_rate',          # Win Rate %
    'expectancy',        # Expectancy por trade
    'custom'             # Funcion personalizada
]

# Objetivo personalizado
def my_objective(metrics):
    return metrics['sharpe_ratio'] * (1 - metrics['max_drawdown_pct']/100)

results = optimizer.optimize(
    objective='custom',
    custom_objective=my_objective
)
```

## Analisis Post-Optimizacion

### Deteccion de Overfitting

```python
from src.optimizers.analysis import OverfittingDetector

detector = OverfittingDetector()

report = detector.analyze(
    is_results=results.is_results,
    oos_results=results.oos_results
)

print(f"Performance Degradation: {report.degradation:.2f}%")
print(f"Parameter Stability: {report.stability:.2f}")
print(f"Overfitting Risk: {report.risk_level}")  # 'low', 'medium', 'high'

# Recomendaciones
for rec in report.recommendations:
    print(f"- {rec}")
```

### Estabilidad de Parametros

```python
from src.optimizers.analysis import ParameterStabilityAnalyzer

analyzer = ParameterStabilityAnalyzer()

stability = analyzer.analyze(results)

for param, stats in stability.items():
    print(f"\n{param}:")
    print(f"  Mean: {stats['mean']:.2f}")
    print(f"  Std: {stats['std']:.2f}")
    print(f"  Stability Score: {stats['stability']:.2f}")

# Alta estabilidad = parametro robusto
# Baja estabilidad = posible overfitting
```

### Visualizacion

```python
from src.optimizers.analysis import OptimizationVisualizer

viz = OptimizationVisualizer(results)

# Superficie 3D de optimizacion
viz.plot_optimization_surface(
    x_param='fast_ema',
    y_param='slow_ema',
    z_metric='sharpe_ratio'
)

# Convergencia
viz.plot_convergence()

# Importancia de parametros
viz.plot_parameter_importance()

# Correlacion entre parametros y objetivo
viz.plot_parameter_correlation()
```

## Configuracion Avanzada

### Restricciones

```python
# Restriccion: slow_ema > fast_ema
def constraint(params):
    return params['slow_ema'] > params['fast_ema']

results = optimizer.optimize(
    objective='sharpe_ratio',
    constraints=[constraint]
)
```

### Early Stopping

```python
results = optimizer.bayesian_optimize(
    n_iterations=100,
    early_stopping_rounds=10,  # Parar si no mejora en 10 iteraciones
    min_improvement=0.01       # Mejora minima requerida
)
```

### Multi-Objetivo

```python
from src.optimizers.genetic import MultiObjectiveGeneticOptimizer

mo_optimizer = MultiObjectiveGeneticOptimizer(
    strategy_class=MiEstrategia,
    data=data
)

# Optimizar Sharpe y minimizar Drawdown
results = mo_optimizer.optimize(
    objectives=['sharpe_ratio', 'max_drawdown'],
    objective_weights=[1.0, -0.5],  # Maximizar Sharpe, minimizar DD
    population_size=100,
    n_generations=50
)

# Frente de Pareto
pareto_front = results.pareto_front
```

## API Reference

### StrategyOptimizer

| Metodo | Descripcion |
|--------|-------------|
| `add_parameter(name, type, min, max, step?, choices?)` | Agregar parametro |
| `grid_optimize(objective, n_jobs?)` | Grid search |
| `random_optimize(n_iterations, objective)` | Random search |
| `bayesian_optimize(n_iterations, objective)` | Bayesian opt |
| `genetic_optimize(pop_size, n_gen, objective)` | Genetic algorithm |
| `walk_forward_optimize(n_splits, train_ratio)` | Walk-forward |

### OptimizationResult

| Campo | Descripcion |
|-------|-------------|
| `best_params` | Mejores parametros encontrados |
| `best_value` | Mejor valor del objetivo |
| `all_results` | Todos los resultados |
| `convergence` | Historia de convergencia |
| `n_iterations` | Numero de iteraciones |
| `runtime` | Tiempo de ejecucion |

## Tests

```bash
pytest tests/optimizers/ -v
```
