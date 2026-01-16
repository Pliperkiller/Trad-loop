# Optimizers

The **optimizers** module provides advanced algorithms for optimizing trading strategy parameters with robust validation.

## Architecture

```
optimizers/
├── grid_search.py         # Exhaustive search
├── random_search.py       # Random search
├── bayesian.py            # Bayesian optimization
├── genetic.py             # Genetic algorithm
├── walk_forward.py        # Walk-forward analysis
├── optimization_types.py  # Types and configurations
├── validation/            # Cross-validation
│   ├── splitters.py       # Temporal splitters
│   ├── time_series_cv.py  # Time Series CV
│   ├── purged_kfold.py    # Purged K-Fold
│   └── results.py         # Result models
└── analysis/              # Post-optimization
    ├── overfitting_detection.py
    ├── parameter_stability.py
    └── visualization.py
```

## Basic Usage

### Configuration

```python
from src.optimizer import StrategyOptimizer

optimizer = StrategyOptimizer(
    strategy_class=MyStrategy,
    data=historical_data,
    initial_capital=10000,
    commission_pct=0.001
)

# Define parameters to optimize
optimizer.add_parameter('fast_ema', 'int', 5, 30, step=5)
optimizer.add_parameter('slow_ema', 'int', 20, 60, step=10)
optimizer.add_parameter('rsi_period', 'int', 10, 20)
optimizer.add_parameter('rsi_threshold', 'float', 25.0, 35.0, step=2.5)
```

### Parameter Types

```python
# Integer with step
optimizer.add_parameter('period', 'int', 10, 50, step=5)
# Values: 10, 15, 20, 25, 30, 35, 40, 45, 50

# Continuous integer
optimizer.add_parameter('period', 'int', 10, 50)
# Values: 10, 11, 12, ..., 49, 50

# Float
optimizer.add_parameter('threshold', 'float', 0.01, 0.05, step=0.01)

# Categorical
optimizer.add_parameter('ma_type', 'categorical', choices=['sma', 'ema', 'wma'])
```

## Optimization Algorithms

### Grid Search

Exhaustive search of all combinations.

```python
results = optimizer.grid_optimize(
    objective='sharpe_ratio',
    n_jobs=-1  # Parallelization
)

print(f"Total combinations: {results.n_combinations}")
print(f"Best parameters: {results.best_params}")
print(f"Best Sharpe: {results.best_value:.4f}")
```

**Advantages**: Complete, deterministic
**Disadvantages**: Slow for many parameters

### Random Search

Efficient random search with parallel execution support.

```python
results = optimizer.random_optimize(
    n_iterations=100,
    objective='sharpe_ratio',
    n_jobs=4,    # Parallel evaluation with 4 workers
    seed=42
)

print(f"Best parameters: {results.best_params}")
print(f"Convergence: {results.convergence[-1]:.4f}")
```

**Advantages**: Fast, scalable, parallel execution
**Disadvantages**: No global optimum guarantee

### Bayesian Optimization

Uses Gaussian Process for intelligent search with parallel execution support.

```python
results = optimizer.bayesian_optimize(
    n_iterations=50,
    objective='sharpe_ratio',
    n_initial_points=10,  # Initial exploration
    n_jobs=4,             # Parallel evaluation with 4 workers
    acq_func='EI'         # Expected Improvement
)

# Available acquisition functions:
# 'EI' - Expected Improvement
# 'PI' - Probability of Improvement
# 'LCB' - Lower Confidence Bound
```

**Advantages**: Efficient, converges quickly, parallel execution
**Disadvantages**: More complex

### Genetic Algorithm

Evolutionary optimization.

```python
results = optimizer.genetic_optimize(
    population_size=50,
    n_generations=30,
    mutation_rate=0.1,
    crossover_rate=0.8,
    elite_size=5,
    objective='sharpe_ratio'
)

print(f"Generations: {results.n_generations}")
print(f"Best fitness: {results.best_fitness:.4f}")

# View evolution
import matplotlib.pyplot as plt
plt.plot(results.fitness_history)
plt.xlabel('Generation')
plt.ylabel('Best Fitness')
plt.show()
```

**Advantages**: Robust, multi-objective possible
**Disadvantages**: Slow, requires tuning

### Walk-Forward Analysis

Continuous out-of-sample validation.

```python
results = optimizer.walk_forward_optimize(
    n_splits=5,
    train_ratio=0.7,      # 70% training
    optimization_method='bayesian',
    n_opt_iterations=30,
    objective='sharpe_ratio'
)

# Results per split
for i, split in enumerate(results.splits):
    print(f"\nSplit {i+1}:")
    print(f"  IS Sharpe: {split.is_sharpe:.2f}")
    print(f"  OOS Sharpe: {split.oos_sharpe:.2f}")
    print(f"  Params: {split.best_params}")

# Summary
print(f"\nIS Sharpe average: {results.is_sharpe_mean:.2f}")
print(f"OOS Sharpe average: {results.oos_sharpe_mean:.2f}")
print(f"Degradation: {results.performance_degradation:.2f}%")
```

## Cross-Validation

### Time Series Split

```python
from src.optimizers.validation import TimeSeriesSplit

splitter = TimeSeriesSplit(n_splits=5)

for train_idx, test_idx in splitter.split(data):
    train = data.iloc[train_idx]
    test = data.iloc[test_idx]
    # Train and evaluate
```

### Rolling Window Split

```python
from src.optimizers.validation import RollingWindowSplit

splitter = RollingWindowSplit(
    train_size=252,    # 1 year training
    test_size=63,      # 3 months test
    step=21            # Advance 1 month
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

K-Fold with temporal purge to avoid data leakage.

```python
from src.optimizers.validation import PurgedKFold

splitter = PurgedKFold(
    n_splits=5,
    purge_gap=5  # 5 periods gap between train/test
)
```

### Combinatorial Purged CV

Advanced cross-validation for time series.

```python
from src.optimizers.validation import CombinatorialPurgedCV

splitter = CombinatorialPurgedCV(
    n_splits=6,
    n_test_splits=2,
    purge_gap=5,
    embargo_gap=3
)
```

## Optimization Objectives

```python
# Available objectives
objectives = [
    'sharpe_ratio',      # Sharpe Ratio (default)
    'sortino_ratio',     # Sortino Ratio
    'calmar_ratio',      # Calmar Ratio
    'profit_factor',     # Profit Factor
    'total_return',      # Total Return %
    'max_drawdown',      # Max Drawdown % (minimize)
    'win_rate',          # Win Rate %
    'expectancy',        # Expectancy per trade
    'custom'             # Custom function
]

# Custom objective
def my_objective(metrics):
    return metrics['sharpe_ratio'] * (1 - metrics['max_drawdown_pct']/100)

results = optimizer.optimize(
    objective='custom',
    custom_objective=my_objective
)
```

## Post-Optimization Analysis

### Overfitting Detection

The analysis module provides tools to detect overfitting and assess optimization quality.

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

# Recommendations
for rec in report.recommendations:
    print(f"- {rec}")
```

**Key Metrics:**
- **Performance Degradation**: % drop from in-sample to out-of-sample
- **Parameter Stability**: How consistent parameters are across splits
- **Risk Level**: Overall overfitting risk assessment

### Parameter Stability Analysis

Analyzes how stable optimized parameters are across different time periods.

```python
from src.optimizers.analysis import ParameterStabilityAnalyzer

analyzer = ParameterStabilityAnalyzer()

stability = analyzer.analyze(results)

for param, stats in stability.items():
    print(f"\n{param}:")
    print(f"  Mean: {stats['mean']:.2f}")
    print(f"  Std: {stats['std']:.2f}")
    print(f"  Coefficient of Variation: {stats['cv']:.2f}")
    print(f"  Stability Score: {stats['stability']:.2f}")

# High stability = robust parameter
# Low stability = possible overfitting
```

**Interpretation:**
- **Stability Score > 0.8**: Robust parameter
- **Stability Score 0.5-0.8**: Moderate stability
- **Stability Score < 0.5**: Unstable, possible overfitting

### Visualization

```python
from src.optimizers.analysis import OptimizationVisualizer

viz = OptimizationVisualizer(results)

# 3D optimization surface
viz.plot_optimization_surface(
    x_param='fast_ema',
    y_param='slow_ema',
    z_metric='sharpe_ratio'
)

# Convergence plot
viz.plot_convergence()

# Parameter importance
viz.plot_parameter_importance()

# Correlation between parameters and objective
viz.plot_parameter_correlation()

# Walk-forward results
viz.plot_walk_forward_results()

# Distribution of parameter values
viz.plot_parameter_distributions()
```

## Advanced Configuration

### Constraints

```python
# Constraint: slow_ema > fast_ema
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
    early_stopping_rounds=10,  # Stop if no improvement in 10 iterations
    min_improvement=0.01       # Minimum improvement required
)
```

### Multi-Objective Optimization

```python
from src.optimizers.genetic import MultiObjectiveGeneticOptimizer

mo_optimizer = MultiObjectiveGeneticOptimizer(
    strategy_class=MyStrategy,
    data=data
)

# Optimize Sharpe and minimize Drawdown
results = mo_optimizer.optimize(
    objectives=['sharpe_ratio', 'max_drawdown'],
    objective_weights=[1.0, -0.5],  # Maximize Sharpe, minimize DD
    population_size=100,
    n_generations=50
)

# Pareto front
pareto_front = results.pareto_front
```

### Integration with Job Manager

For long-running optimizations, use the Job Manager for async execution.

```python
from src.job_manager import get_job_manager, JobType

async def run_optimization(params, progress_callback):
    optimizer = StrategyOptimizer(
        strategy_class=params['strategy_class'],
        data=params['data']
    )

    for name, config in params['parameters'].items():
        optimizer.add_parameter(name, **config)

    # Run with progress updates
    results = optimizer.bayesian_optimize(
        n_iterations=params['n_iterations'],
        objective=params['objective'],
        progress_callback=progress_callback
    )

    return results.to_dict()

manager = get_job_manager()
job_id = await manager.create_job(
    job_type=JobType.OPTIMIZATION,
    params={...},
    executor=run_optimization
)
```

## API Reference

### StrategyOptimizer

| Method | Description |
|--------|-------------|
| `add_parameter(name, type, min, max, step?, choices?)` | Add parameter |
| `grid_optimize(objective, n_jobs?)` | Grid search (parallel) |
| `random_optimize(n_iterations, objective, n_jobs?)` | Random search (parallel) |
| `bayesian_optimize(n_iterations, objective, n_jobs?)` | Bayesian optimization (parallel) |
| `genetic_optimize(pop_size, n_gen, objective)` | Genetic algorithm |
| `walk_forward_optimize(n_splits, train_ratio, n_jobs?)` | Walk-forward (parallel) |

**Note on Parallelization**: The `n_jobs` parameter controls parallel execution:
- `n_jobs=1`: Sequential execution (default)
- `n_jobs=4`: Use 4 parallel workers
- `n_jobs=-1`: Use all available CPU cores

### OptimizationResult

| Field | Description |
|-------|-------------|
| `best_params` | Best parameters found |
| `best_value` | Best objective value |
| `all_results` | All results |
| `convergence` | Convergence history |
| `n_iterations` | Number of iterations |
| `runtime` | Execution time |

### Validation Splitters

| Class | Description |
|-------|-------------|
| `TimeSeriesSplit` | Standard time series split |
| `RollingWindowSplit` | Fixed-size rolling window |
| `ExpandingWindowSplit` | Expanding training window |
| `PurgedKFold` | K-Fold with purge |
| `CombinatorialPurgedCV` | Advanced purged CV |
| `WalkForwardSplit` | Walk-forward splits |

### Analysis Classes

| Class | Description |
|-------|-------------|
| `OverfittingDetector` | Detect overfitting |
| `ParameterStabilityAnalyzer` | Analyze parameter stability |
| `OptimizationVisualizer` | Visualization tools |

## Best Practices

1. **Start with Random Search**: Quick exploration before intensive optimization
2. **Use Walk-Forward**: Most realistic validation method
3. **Check Overfitting**: Always analyze IS vs OOS performance
4. **Validate Parameter Stability**: Unstable parameters indicate overfitting
5. **Use Constraints**: Enforce logical relationships between parameters
6. **Limit Parameter Space**: Fewer parameters = less overfitting risk
7. **Use Sufficient Data**: More data = more reliable results
8. **Enable Parallelization**: Use `n_jobs=-1` or specify cores for faster optimization

## Tests

```bash
pytest tests/optimizers/ -v
```

## Related Documentation

- [Performance](../user_guide.md#performance-analysis) - Performance metrics
- [Risk Management](risk_management.md) - Risk controls
- [Interfaces](interfaces.md) - IOptimizer protocol
