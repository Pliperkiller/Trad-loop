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
    fixed_params = {}  # Parámetros con min == max (no se optimizan)

    for param in self.parameter_space:
        # Validar que hay un rango válido para optimizar
        low_val = param.low if param.low is not None else 0
        high_val = param.high if param.high is not None else 100

        # Si min >= max, el parámetro es fijo (no hay nada que optimizar)
        if low_val >= high_val:
            fixed_params[param.name] = low_val
            if verbose:
                print(f"  [INFO] Parámetro '{param.name}' fijo en {low_val} (min >= max)")
            continue

        param_names.append(param.name)

        if param.param_type == 'int':
            low = int(low_val)
            high = int(high_val)
            space.append(Integer(low, high, name=param.name))
        elif param.param_type == 'float':
            low = float(low_val)
            high = float(high_val)
            space.append(Real(low, high, name=param.name))
        elif param.param_type == 'categorical' and param.values:
            space.append(Integer(0, len(param.values) - 1, name=param.name))
        else:
            space.append(Real(float(low_val), float(high_val), name=param.name))

    # Verificar que hay al menos un parámetro para optimizar
    if not space:
        raise ValueError("No hay parámetros con rangos válidos para optimizar. "
                        "Todos los parámetros tienen min >= max.")

    # Crear lista filtrada de parameter_space (solo los que se optimizan)
    optimizable_params = [p for p in self.parameter_space if p.name in param_names]

    if verbose and fixed_params:
        print(f"  Parámetros fijos: {fixed_params}")
        print()

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
        # Usar optimizable_params (solo los que tienen rango válido)
        params_list = []
        for p in points:
            params = _convert_point_to_params(p, optimizable_params)
            # Agregar parámetros fijos
            params.update(fixed_params)
            params_list.append(params)

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
        """Convierte tipos numpy a tipos nativos de Python."""
        if hasattr(val, 'item'):
            return val.item()
        return val

    def to_native(value, param_type: str):
        """Convierte valor a tipo nativo de Python según param_type."""
        val = to_python_type(value)
        if param_type == 'int':
            return int(val)
        elif param_type == 'float':
            return float(val)
        return val

    # Crear lookup de tipos de parámetros
    param_type_lookup = {p.name: p.param_type for p in self.parameter_space}

    # Encontrar mejor resultado
    valid_results = [r for r in all_results if r['score'] > -1e9]
    if valid_results:
        best_result = max(valid_results, key=lambda r: r['score'])
        # Convertir params a tipos nativos de Python
        best_params = {
            k: to_native(v, param_type_lookup.get(k, 'float'))
            for k, v in best_result['params'].items()
        }
        best_score = float(to_python_type(best_result['score']))
    else:
        best_params = {}
        best_score = float('-inf')

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
