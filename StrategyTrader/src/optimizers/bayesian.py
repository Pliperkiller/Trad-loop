"""
Bayesian Optimization con paralelización por batch acquisition y respeto de step.
"""

import os
import pandas as pd
import numpy as np
import time
import math
from typing import Callable, Optional, List, Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed

from skopt import Optimizer
from skopt.space import Real, Integer

from .optimization_types import OptimizationResult, ParameterSpace
from .utils import snap_to_step, convert_point_to_params


# Alias para compatibilidad con código existente
_snap_to_step = snap_to_step
_convert_point_to_params = convert_point_to_params


def bayesian_optimization(self, n_calls: int = 50, n_initial_points: int = 10,
                         verbose: bool = True,
                         progress_callback: Optional[Callable[[int, int], None]] = None,
                         n_jobs: int = -1,
                         batch_size: int = None) -> OptimizationResult:
    """
    Bayesian Optimization: Usa un modelo probabilístico para guiar la búsqueda.

    MEJORAS:
    - Batch acquisition paralela: evalúa múltiples puntos simultáneamente
    - Respeta el step definido para cada parámetro (snap to grid)
    - Paralelización con ProcessPoolExecutor

    Args:
        n_calls: Número total de evaluaciones
        n_initial_points: Puntos aleatorios iniciales para exploración
        verbose: Mostrar progreso
        progress_callback: Callback para reportar progreso
        n_jobs: Número de workers paralelos (-1 = todos los cores - 1)
        batch_size: Tamaño del batch para evaluación paralela (None = auto)

    Pros:
    - Muy eficiente en número de evaluaciones
    - Balancea exploración vs explotación
    - Excelente para funciones costosas de evaluar
    - PARALELO por batch acquisition

    Contras:
    - Más complejo de entender
    - Puede quedar atrapado en óptimos locales

    Mejor para: Parámetros continuos, presupuesto limitado de evaluaciones
    """
    start_time = time.time()

    # Determinar número de workers
    max_cores = os.cpu_count() or 4
    if n_jobs == -1:
        n_jobs = max(1, max_cores - 1)
    else:
        n_jobs = max(1, min(n_jobs, max_cores))

    # Determinar batch size
    if batch_size is None:
        batch_size = min(n_jobs, max(1, n_calls // 10))  # Auto: basado en cores y total calls
    batch_size = max(1, min(batch_size, n_calls))

    if verbose:
        print("\n" + "="*70)
        print("INICIANDO BAYESIAN OPTIMIZATION (PARALELO)")
        print("="*70)
        print(f"Evaluaciones totales: {n_calls}")
        print(f"Puntos iniciales: {n_initial_points}")
        print(f"Workers paralelos: {n_jobs}")
        print(f"Batch size: {batch_size}")
        print(f"Parámetros con step:")
        for p in self.parameter_space:
            step_info = f"step={p.step}" if p.step else "continuo"
            print(f"  - {p.name}: [{p.low}, {p.high}] {step_info}")
        print()

    # Definir espacio de búsqueda para skopt
    space = []
    param_names = []

    for param in self.parameter_space:
        param_names.append(param.name)

        if param.param_type == 'int':
            low = int(param.low) if param.low is not None else 0
            high = int(param.high) if param.high is not None else 100
            space.append(Integer(low, high, name=param.name))
        elif param.param_type == 'float':
            low = float(param.low) if param.low is not None else 0.0
            high = float(param.high) if param.high is not None else 1.0
            space.append(Real(low, high, name=param.name))
        elif param.param_type == 'categorical' and param.values:
            space.append(Integer(0, len(param.values) - 1, name=param.name))
        else:
            if param.low is not None and param.high is not None:
                space.append(Real(float(param.low), float(param.high), name=param.name))
            else:
                space.append(Integer(0, 1, name=param.name))

    # Crear optimizador con interfaz ask/tell para batch acquisition
    optimizer = Optimizer(
        dimensions=space,
        base_estimator='GP',  # Gaussian Process
        n_initial_points=n_initial_points,
        acq_func='EI',  # Expected Improvement
        random_state=42
    )

    # Valor de penalización
    LARGE_PENALTY = 10.0

    def safe_score(raw_score) -> float:
        """Convierte score a float seguro para skopt"""
        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            return LARGE_PENALTY

        if not math.isfinite(score):
            return LARGE_PENALTY

        clamped = max(-10.0, min(10.0, score))
        return -clamped  # Skopt minimiza

    def evaluate_single(params_dict: Dict) -> Dict:
        """Evalúa un conjunto de parámetros y retorna resultado."""
        try:
            score = self._evaluate_parameters(params_dict)
            return {
                'params': params_dict,
                'score': score,
                'safe_score': safe_score(score)
            }
        except Exception as e:
            return {
                'params': params_dict,
                'score': -np.inf,
                'safe_score': LARGE_PENALTY,
                'error': str(e)
            }

    # Resultados acumulados
    all_results = []
    completed = 0
    best_score_so_far = -np.inf

    # Loop principal de optimización por batches
    while completed < n_calls:
        # Determinar tamaño del batch actual
        remaining = n_calls - completed
        current_batch_size = min(batch_size, remaining)

        # Pedir puntos al optimizador
        if current_batch_size > 1:
            points = optimizer.ask(n_points=current_batch_size)
        else:
            points = [optimizer.ask()]

        # Convertir puntos a parámetros con snap-to-step
        params_list = [_convert_point_to_params(p, self.parameter_space) for p in points]

        # Evaluar en paralelo
        results_batch = []

        if n_jobs > 1 and current_batch_size > 1:
            # Modo paralelo
            with ProcessPoolExecutor(max_workers=min(n_jobs, current_batch_size)) as executor:
                future_to_idx = {
                    executor.submit(evaluate_single, params): idx
                    for idx, params in enumerate(params_list)
                }

                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        result = future.result()
                    except Exception as e:
                        result = {
                            'params': params_list[idx],
                            'score': -np.inf,
                            'safe_score': LARGE_PENALTY,
                            'error': str(e)
                        }
                    results_batch.append((idx, result))

            # Ordenar por índice para mantener correspondencia con points
            results_batch.sort(key=lambda x: x[0])
            results_batch = [r[1] for r in results_batch]
        else:
            # Modo secuencial
            results_batch = [evaluate_single(params) for params in params_list]

        # Actualizar el optimizador con los resultados
        scores_for_optimizer = [r['safe_score'] for r in results_batch]
        optimizer.tell(points, scores_for_optimizer)

        # Guardar resultados y actualizar progreso
        for result in results_batch:
            all_results.append(result)
            completed += 1

            if result['score'] > best_score_so_far and result['score'] > -1e9:
                best_score_so_far = result['score']

            if progress_callback:
                progress_callback(completed, n_calls)

        if verbose and completed % max(1, n_calls // 10) == 0:
            print(f"Progreso: {completed}/{n_calls} ({completed/n_calls*100:.1f}%) - Best: {best_score_so_far:.4f}")

    # Procesar resultados finales
    def to_python_type(val):
        if hasattr(val, 'item'):
            return val.item()
        return val

    # Encontrar mejor resultado
    valid_results = [r for r in all_results if r['score'] > -1e9]
    if valid_results:
        best_result = max(valid_results, key=lambda r: r['score'])
        best_params = best_result['params']
        best_score = best_result['score']
    else:
        best_params = {}
        best_score = -np.inf

    # Crear DataFrame con historial
    results_data = []
    for r in all_results:
        row = r['params'].copy()
        row['score'] = to_python_type(r['score'])
        results_data.append(row)

    results_df = pd.DataFrame(results_data)

    optimization_time = time.time() - start_time

    if verbose:
        print(f"\nOptimización completada en {optimization_time:.2f}s")
        print(f"Mejor score: {best_score:.4f}")
        print(f"Mejores parámetros: {best_params}")

    return OptimizationResult(
        best_params=best_params,
        best_score=best_score,
        all_results=results_df,
        optimization_time=optimization_time,
        method='Bayesian Optimization (Parallel)',
        iterations=n_calls
    )
