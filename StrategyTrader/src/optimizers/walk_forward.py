"""
Walk Forward Optimization

Tecnica de validacion que previene el overfitting dividiendo los datos
en multiples periodos de entrenamiento y prueba secuenciales.

Flujo:
1. Dividir datos en N splits secuenciales
2. Para cada split:
   - Optimizar parametros en periodo de entrenamiento (in-sample)
   - Validar con esos parametros en periodo de prueba (out-of-sample)
3. Agregar resultados OOS para evaluar robustez real

Mejoras:
- Soporte para ventana rolling (fija) y anchored (expandible)
- Periodos de test no solapados (walk-forward clasico)
- Gap/embargo entre train y test
- Tracking de estabilidad de parametros
- Integracion con modulo de validacion
"""

import time
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional, TYPE_CHECKING

from .optimization_types import WalkForwardResult

if TYPE_CHECKING:
    from .validation.results import CrossValidationResult, SplitResult
    from .validation.splitters import WalkForwardSplit


def walk_forward_optimization(
    optimizer,
    optimization_method: str = 'random',
    n_splits: int = 5,
    train_size: float = 0.6,
    n_iter: int = 50,
    anchored: bool = True,
    gap: int = 0,
    min_train_rows: int = 100,
    min_test_rows: int = 50,
    track_params: bool = True,
    verbose: bool = True
) -> WalkForwardResult:
    """
    Walk Forward Optimization: Validacion robusta contra overfitting

    Divide los datos temporalmente y optimiza/valida en cada segmento
    para obtener una estimacion realista del performance futuro.

    IMPORTANTE: Esta version mejorada implementa walk-forward clasico donde
    los periodos de test NO se solapan, a diferencia de la version anterior.

    Args:
        optimizer: Instancia de StrategyOptimizer
        optimization_method: Metodo a usar ('grid', 'random', 'bayesian', 'genetic')
        n_splits: Numero de splits para walk-forward
        train_size: Proporcion inicial de datos para entrenamiento (0.5-0.8)
        n_iter: Iteraciones para metodos estocasticos
        anchored: Si True, ventana de train se expande. Si False, ventana fija (rolling)
        gap: Filas de separacion entre train y test (para evitar data leakage)
        min_train_rows: Minimo de filas requeridas para train
        min_test_rows: Minimo de filas requeridas para test
        track_params: Trackear estabilidad de parametros entre splits
        verbose: Mostrar progreso

    Returns:
        WalkForwardResult con metricas agregadas y score de robustez

    Ejemplo:
        result = optimizer.walk_forward_optimization(
            optimization_method='random',
            n_splits=5,
            train_size=0.6,
            anchored=True,  # Expanding window
            gap=10,         # 10 filas de separacion
            n_iter=30
        )
        result.print_summary()
    """
    start_time = time.time()

    if verbose:
        print("\n" + "="*70)
        print("WALK FORWARD OPTIMIZATION (Mejorado)")
        print("="*70)
        print(f"Splits: {n_splits}")
        print(f"Train Size inicial: {train_size*100:.0f}%")
        print(f"Metodo: {optimization_method}")
        print(f"Iteraciones por split: {n_iter}")
        print(f"Modo: {'Anchored (Expanding)' if anchored else 'Rolling (Fixed)'}")
        print(f"Gap: {gap} filas")
        print("="*70 + "\n")

    # Obtener datos originales
    original_data = optimizer.data.copy()
    total_rows = len(original_data)

    # Calcular tamanio inicial de train y test por split
    initial_train_size = int(total_rows * train_size)
    remaining = total_rows - initial_train_size - (gap * n_splits)
    test_size_per_split = remaining // n_splits

    if initial_train_size < min_train_rows:
        raise ValueError(
            f"Train inicial ({initial_train_size}) es menor que min_train_rows ({min_train_rows})"
        )
    if test_size_per_split < min_test_rows:
        raise ValueError(
            f"Test por split ({test_size_per_split}) es menor que min_test_rows ({min_test_rows})"
        )

    if verbose:
        print(f"Train inicial: {initial_train_size} filas")
        print(f"Test por split: {test_size_per_split} filas")
        print(f"Gap: {gap} filas")

    splits_results = []
    all_oos_equity = []
    all_oos_returns = []
    all_params: List[Dict[str, Any]] = []

    # Rolling window train size (solo usado si anchored=False)
    rolling_train_size = initial_train_size

    for split_idx in range(n_splits):
        if verbose:
            print(f"\n{'='*50}")
            print(f"SPLIT {split_idx + 1}/{n_splits}")
            print(f"{'='*50}")

        # Calcular indices para este split - WALK FORWARD CLASICO
        if anchored:
            # Expanding: train siempre empieza en 0
            train_start = 0
            train_end = initial_train_size + (split_idx * test_size_per_split)
        else:
            # Rolling: train tiene tamanio fijo, ventana se desliza
            train_start = split_idx * test_size_per_split
            train_end = train_start + rolling_train_size

        # Test siempre despues de train + gap
        test_start = train_end + gap
        test_end = test_start + test_size_per_split

        # Validar que no excedemos los datos
        if test_end > total_rows:
            test_end = total_rows

        if test_start >= total_rows:
            if verbose:
                print(f"Saltando split {split_idx + 1}: no hay datos de test disponibles")
            continue

        # Extraer datos
        train_data = original_data.iloc[train_start:train_end].copy()
        test_data = original_data.iloc[test_start:test_end].copy()

        if verbose:
            print(f"Train: filas {train_start}-{train_end} ({len(train_data)} filas)")
            print(f"Test:  filas {test_start}-{test_end} ({len(test_data)} filas)")
            if gap > 0:
                print(f"Gap:   filas {train_end}-{test_start} ({gap} filas)")

        if len(test_data) < 20:
            if verbose:
                print(f"Saltando split {split_idx + 1}: datos de test insuficientes ({len(test_data)} < 20)")
            continue

        # Optimizar en datos de entrenamiento
        optimizer.data = train_data
        optimizer.results_cache = {}  # Limpiar cache

        try:
            if optimization_method == 'grid':
                opt_result = optimizer.grid_search(verbose=False)
            elif optimization_method == 'random':
                opt_result = optimizer.random_search(n_iter=n_iter, verbose=False)
            elif optimization_method == 'bayesian':
                opt_result = optimizer.bayesian_optimization(n_calls=n_iter, verbose=False)
            elif optimization_method == 'genetic':
                opt_result = optimizer.genetic_algorithm(
                    population_size=max(10, n_iter // 5),
                    max_generations=max(5, n_iter // 10),
                    verbose=False
                )
            else:
                raise ValueError(f"Metodo no reconocido: {optimization_method}")

            best_params = opt_result.best_params
            train_score = opt_result.best_score
            all_params.append(best_params)

            if verbose:
                print(f"Mejores params: {best_params}")
                print(f"Train Score: {train_score:.4f}")

        except Exception as e:
            if verbose:
                print(f"Error en optimizacion: {e}")
            splits_results.append({
                'split': split_idx + 1,
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
                'train_score': -np.inf,
                'oos_score': -np.inf,
                'best_params': {},
                'error': str(e)
            })
            continue

        # Validar en datos de prueba (out-of-sample)
        try:
            optimizer.data = test_data
            optimizer.results_cache = {}

            # Evaluar con los parametros optimizados
            oos_result = optimizer._evaluate_parameters_detailed(best_params)
            oos_score = oos_result.get('score', -np.inf)

            # Obtener metricas adicionales OOS
            from src.strategy import StrategyConfig
            if isinstance(optimizer.config_template, dict):
                config = StrategyConfig(**optimizer.config_template)
            else:
                config = optimizer.config_template
            strategy = optimizer.strategy_class(config, **best_params)
            strategy.load_data(test_data)
            strategy.backtest()
            oos_metrics = strategy.get_performance_metrics()

            # Guardar equity curve OOS
            if hasattr(strategy, 'equity_curve') and len(strategy.equity_curve) > 0:
                all_oos_equity.extend(strategy.equity_curve)

            # Guardar returns OOS
            if len(strategy.equity_curve) > 1:
                equity_series = pd.Series(strategy.equity_curve)
                returns = equity_series.pct_change().dropna().tolist()
                all_oos_returns.extend(returns)

            # Calcular degradacion
            if train_score != 0:
                degradation = ((train_score - oos_score) / abs(train_score) * 100)
            else:
                degradation = 0.0

            if verbose:
                print(f"OOS Score: {oos_score:.4f}")
                print(f"Degradacion: {degradation:.1f}%")

            splits_results.append({
                'split': split_idx + 1,
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
                'train_score': train_score,
                'oos_score': oos_score,
                'degradation_pct': degradation,
                'best_params': best_params,
                'oos_metrics': oos_metrics,
                'train_rows': len(train_data),
                'test_rows': len(test_data)
            })

        except Exception as e:
            if verbose:
                print(f"Error en validacion OOS: {e}")
            splits_results.append({
                'split': split_idx + 1,
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
                'train_score': train_score,
                'oos_score': -np.inf,
                'best_params': best_params,
                'error': str(e)
            })

    # Restaurar datos originales
    optimizer.data = original_data
    optimizer.results_cache = {}

    # Calcular metricas agregadas
    aggregated_metrics = _calculate_aggregated_metrics(splits_results, all_oos_returns)

    # Calcular score de robustez
    robustness_score = _calculate_robustness_score(splits_results)

    # Calcular estabilidad de parametros si se solicita
    if track_params and all_params:
        param_stability = _calculate_parameter_stability(all_params)
        aggregated_metrics['parameter_stability'] = param_stability

    optimization_time = time.time() - start_time

    if verbose:
        print("\n" + "="*70)
        print("WALK FORWARD COMPLETADO")
        print(f"Tiempo total: {optimization_time:.2f}s")
        print(f"Score de robustez: {robustness_score:.2f}")
        if track_params and 'parameter_stability' in aggregated_metrics:
            unstable = [k for k, v in aggregated_metrics['parameter_stability'].items() if not v['is_stable']]
            if unstable:
                print(f"Parametros inestables: {', '.join(unstable)}")
            else:
                print("Todos los parametros son estables")
        print("="*70)

    # Calculate additional metrics for validation integration
    valid_splits = [s for s in splits_results if s.get('oos_score', -np.inf) > -np.inf]
    avg_is = float(np.mean([s['train_score'] for s in valid_splits])) if valid_splits else 0.0
    avg_oos = aggregated_metrics.get('avg_oos_score', 0.0)
    consistency = aggregated_metrics.get('positive_oos_ratio', 0.0)

    # Get best params (from most recent successful split)
    best_params = None
    for s in reversed(splits_results):
        if 'best_params' in s and s.get('oos_score', -np.inf) > -np.inf:
            best_params = s['best_params']
            break

    # Get parameter stability from aggregated metrics if available
    param_stability = aggregated_metrics.get('parameter_stability', None)

    return WalkForwardResult(
        splits_results=splits_results,
        aggregated_metrics=aggregated_metrics,
        out_of_sample_equity=all_oos_equity,
        robustness_score=robustness_score,
        optimization_time=optimization_time,
        n_splits=n_splits,
        train_size=train_size,
        optimization_method=optimization_method,
        best_params=best_params,
        consistency_ratio=consistency,
        avg_is_score=avg_is,
        avg_oos_score=avg_oos,
        parameter_stability=param_stability,
    )


def _calculate_aggregated_metrics(
    splits_results: List[Dict],
    all_oos_returns: List[float]
) -> Dict[str, Any]:
    """Calcula metricas agregadas de todos los periodos OOS"""

    metrics: Dict[str, Any] = {}

    # Scores promedio
    valid_splits = [s for s in splits_results if s.get('oos_score', -np.inf) > -np.inf]

    if not valid_splits:
        return {'error': 'No hay splits validos'}

    oos_scores = [s['oos_score'] for s in valid_splits]
    train_scores = [s['train_score'] for s in valid_splits]

    metrics['avg_oos_score'] = float(np.mean(oos_scores))
    metrics['std_oos_score'] = float(np.std(oos_scores))
    metrics['min_oos_score'] = float(np.min(oos_scores))
    metrics['max_oos_score'] = float(np.max(oos_scores))

    # Degradacion promedio (train vs oos)
    degradations = []
    for s in valid_splits:
        if s['train_score'] != 0:
            deg = (s['train_score'] - s['oos_score']) / abs(s['train_score'])
            degradations.append(deg)

    if degradations:
        metrics['avg_degradation_pct'] = float(np.mean(degradations) * 100)

    # Metricas de retornos OOS combinados
    if all_oos_returns:
        returns = np.array(all_oos_returns)
        metrics['total_return_pct'] = float((np.prod(1 + returns) - 1) * 100)
        metrics['volatility_pct'] = float(np.std(returns) * np.sqrt(252) * 100)

        # Sharpe OOS
        if np.std(returns) > 0:
            metrics['sharpe_ratio'] = float((np.mean(returns) / np.std(returns)) * np.sqrt(252))
        else:
            metrics['sharpe_ratio'] = 0.0

        # Max Drawdown OOS
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        metrics['max_drawdown_pct'] = float(np.min(drawdown) * 100)

    # Agregar metricas de trades si estan disponibles
    total_trades = 0
    total_wins = 0
    for s in valid_splits:
        if 'oos_metrics' in s:
            total_trades += s['oos_metrics'].get('total_trades', 0)
            total_wins += s['oos_metrics'].get('winning_trades', 0)

    if total_trades > 0:
        metrics['total_trades'] = total_trades
        metrics['win_rate_pct'] = float((total_wins / total_trades) * 100)

    # Positive OOS ratio
    positive_oos = sum(1 for s in valid_splits if s['oos_score'] > 0)
    metrics['positive_oos_ratio'] = float(positive_oos / len(valid_splits))

    return metrics


def _calculate_robustness_score(splits_results: List[Dict]) -> float:
    """
    Calcula un score de robustez (0-1) basado en:
    - Consistencia de resultados OOS
    - Degradacion train vs OOS
    - Porcentaje de splits exitosos
    """

    valid_splits = [s for s in splits_results if s.get('oos_score', -np.inf) > -np.inf]

    if not valid_splits:
        return 0.0

    scores = []

    # 1. Porcentaje de splits con OOS positivo (40% del score)
    positive_oos = sum(1 for s in valid_splits if s['oos_score'] > 0)
    pct_positive = positive_oos / len(valid_splits)
    scores.append(pct_positive * 0.4)

    # 2. Consistencia de resultados (baja varianza) (30% del score)
    oos_scores = [s['oos_score'] for s in valid_splits]
    mean_score = float(np.mean(oos_scores))
    if mean_score != 0:
        cv = float(np.std(oos_scores)) / abs(mean_score)  # Coef. de variacion
        consistency = max(0.0, 1.0 - cv)  # Menor variacion = mayor consistencia
    else:
        consistency = 0.0
    scores.append(consistency * 0.3)

    # 3. Baja degradacion train vs OOS (30% del score)
    degradations = []
    for s in valid_splits:
        if s['train_score'] > 0:
            deg = (s['train_score'] - s['oos_score']) / s['train_score']
            degradations.append(max(0.0, min(1.0, deg)))  # Clamp 0-1

    if degradations:
        avg_degradation = float(np.mean(degradations))
        degradation_score = max(0.0, 1.0 - avg_degradation)
    else:
        degradation_score = 0.0
    scores.append(degradation_score * 0.3)

    return float(sum(scores))


def _calculate_parameter_stability(
    all_params: List[Dict[str, Any]],
    stability_threshold: float = 0.3
) -> Dict[str, Dict[str, Any]]:
    """
    Calcula la estabilidad de cada parametro a traves de los splits.

    Args:
        all_params: Lista de diccionarios de parametros por split
        stability_threshold: Umbral de CV para considerar estable

    Returns:
        Diccionario con metricas de estabilidad por parametro
    """
    if not all_params:
        return {}

    param_names = set()
    for params in all_params:
        param_names.update(params.keys())

    stability: Dict[str, Dict[str, Any]] = {}

    for param_name in param_names:
        values = [p.get(param_name) for p in all_params if param_name in p]

        try:
            numeric_values = [float(v) for v in values if v is not None]
            if len(numeric_values) < 2:
                continue

            mean = float(np.mean(numeric_values))
            std = float(np.std(numeric_values))
            cv = std / abs(mean) if mean != 0 else float('inf')
            is_stable = cv < stability_threshold

            stability[param_name] = {
                'mean': mean,
                'std': std,
                'cv': cv,
                'is_stable': is_stable,
                'values': numeric_values
            }
        except (TypeError, ValueError):
            # Non-numeric parameter
            unique_values = len(set(str(v) for v in values if v is not None))
            is_stable = unique_values == 1

            stability[param_name] = {
                'unique_values': unique_values,
                'is_stable': is_stable,
                'values': values
            }

    return stability


# Funciones de conveniencia para integracion con nuevo modulo de validacion

def create_walk_forward_splitter(
    n_splits: int = 5,
    train_pct: float = 0.6,
    anchored: bool = True,
    gap: int = 0,
) -> "WalkForwardSplit":
    """
    Crea un splitter de walk-forward para usar con TimeSeriesCV.

    Args:
        n_splits: Numero de splits
        train_pct: Porcentaje inicial de train
        anchored: Si True, expanding window. Si False, rolling
        gap: Gap entre train y test

    Returns:
        WalkForwardSplit configurado
    """
    from .validation.splitters import WalkForwardSplit
    return WalkForwardSplit(
        n_splits=n_splits,
        train_pct=train_pct,
        anchored=anchored,
        gap=gap,
    )


def convert_to_validation_result(
    wf_result: WalkForwardResult,
) -> "CrossValidationResult":
    """
    Convierte un WalkForwardResult al formato CrossValidationResult
    del nuevo modulo de validacion.

    Args:
        wf_result: Resultado de walk_forward_optimization

    Returns:
        CrossValidationResult compatible con el nuevo sistema
    """
    from .validation.results import (
        SplitResult,
        CrossValidationResult,
        ParameterStability,
    )

    splits: List[SplitResult] = []
    for s in wf_result.splits_results:
        if 'error' in s:
            continue

        split_result = SplitResult(
            split_idx=s['split'] - 1,
            train_start=datetime.now(),  # Placeholder
            train_end=datetime.now(),
            test_start=datetime.now(),
            test_end=datetime.now(),
            train_rows=s.get('train_rows', 0),
            test_rows=s.get('test_rows', 0),
            best_params=s.get('best_params', {}),
            train_score=s.get('train_score', 0.0),
            test_score=s.get('oos_score', 0.0),
            degradation_pct=s.get('degradation_pct', 0.0),
            train_metrics={'score': s.get('train_score', 0.0)},
            test_metrics=s.get('oos_metrics', {}),
        )
        splits.append(split_result)

    # Convert parameter stability
    param_stability: Dict[str, ParameterStability] = {}
    if 'parameter_stability' in wf_result.aggregated_metrics:
        stability_data = wf_result.aggregated_metrics['parameter_stability']
        if isinstance(stability_data, dict):
            for name, data in stability_data.items():
                if isinstance(data, dict) and 'values' in data:
                    param_stability[name] = ParameterStability.from_values(
                        parameter=name,
                        values=data['values'],
                    )

    return CrossValidationResult(
        splits=splits,
        aggregated_metrics=wf_result.aggregated_metrics,
        robustness_score=wf_result.robustness_score,
        parameter_stability=param_stability,
        overfitting_probability=0.0,  # Would need to calculate
        combined_oos_equity=wf_result.out_of_sample_equity,
        optimization_time=wf_result.optimization_time,
        method=f"walk_forward_{wf_result.optimization_method}",
    )
