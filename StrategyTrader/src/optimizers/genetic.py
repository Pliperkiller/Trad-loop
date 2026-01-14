"""
Genetic Algorithm Optimizer (Differential Evolution) con paralelización y respeto de step.
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple, Callable, Optional, List
import warnings
warnings.filterwarnings('ignore')
from scipy.optimize import differential_evolution

from .optimization_types import OptimizationResult, ParameterSpace
from .utils import snap_to_step


# Alias para compatibilidad con código existente
_snap_to_step = snap_to_step


def genetic_algorithm(self, population_size: int = 20, max_generations: int = 50,
                     mutation: Tuple[float, float] = (0.5, 1.0),
                     recombination: float = 0.7,
                     verbose: bool = True,
                     progress_callback: Optional[Callable[[int, int], None]] = None,
                     n_jobs: int = -1) -> OptimizationResult:
    """
    Genetic Algorithm (Differential Evolution): Evoluciona población de soluciones.

    MEJORAS:
    - Respeta el step definido para cada parámetro (snap to grid)
    - Paralelización con workers múltiples para evaluaciones concurrentes

    Args:
        population_size: Tamaño de la población (default 20)
        max_generations: Número máximo de generaciones (default 50)
        mutation: Rango de mutación (default (0.5, 1.0))
        recombination: Probabilidad de recombinación (default 0.7)
        verbose: Mostrar progreso
        progress_callback: Callback para reportar progreso
        n_jobs: Número de workers paralelos (-1 = todos los cores)

    Pros:
    - Excelente para espacios no convexos
    - Robusto ante óptimos locales
    - No requiere gradientes
    - Balancea exploración global y local
    - PARALELO para máximo rendimiento

    Contras:
    - Requiere muchas evaluaciones
    - Parámetros del algoritmo afectan rendimiento

    Mejor para: Espacios complejos con múltiples óptimos locales, 4-8 parámetros
    """
    import time
    start_time = time.time()

    # Determinar número de workers
    # Para mejor rendimiento, usar N-1 cores (dejar 1 para el sistema)
    max_cores = os.cpu_count() or 4
    if n_jobs == -1:
        n_jobs = max(1, max_cores - 1)  # Dejar 1 core libre para el sistema
    else:
        n_jobs = max(1, min(n_jobs, max_cores))

    if verbose:
        print("\n" + "="*70)
        print("INICIANDO GENETIC ALGORITHM (Differential Evolution) - PARALELO")
        print("="*70)
        print(f"Tamaño de población: {population_size}")
        print(f"Generaciones máximas: {max_generations}")
        print(f"Workers paralelos: {n_jobs}")
        print(f"Parámetros con step:")
        for p in self.parameter_space:
            step_info = f"step={p.step}" if p.step else "continuo"
            print(f"  - {p.name}: [{p.low}, {p.high}] {step_info}")
        print()

    # Definir bounds
    bounds = []
    param_names = []
    categorical_params = {}

    for param in self.parameter_space:
        param_names.append(param.name)

        if param.param_type == 'categorical' and param.values:
            bounds.append((0, len(param.values) - 1))
            categorical_params[param.name] = param.values
        elif param.param_type == 'int':
            low = int(param.low) if param.low is not None else 0
            high = int(param.high) if param.high is not None else 100
            bounds.append((low, high))
        elif param.low is not None and param.high is not None:
            bounds.append((float(param.low), float(param.high)))
        else:
            bounds.append((0, 1))

    # Función objetivo con snap to step
    iteration_results = []
    total_evaluations = (max_generations + 1) * population_size * max(1, len(bounds))

    def objective(x):
        # Verificar si hay NaN en los valores
        if np.any(np.isnan(x)):
            return float('inf')

        params = {}
        try:
            for i, (param, value) in enumerate(zip(self.parameter_space, x)):
                if param.param_type == 'categorical' and param.name in categorical_params:
                    idx = int(np.clip(round(value), 0, len(categorical_params[param.name]) - 1))
                    params[param.name] = categorical_params[param.name][idx]
                elif param.param_type == 'int':
                    low = param.low if param.low is not None else 0
                    high = param.high if param.high is not None else 100
                    # Snap to step grid
                    params[param.name] = _snap_to_step(value, low, high, param.step, 'int')
                else:
                    low = param.low if param.low is not None else 0
                    high = param.high if param.high is not None else 1
                    # Snap to step grid
                    params[param.name] = _snap_to_step(value, low, high, param.step, 'float')
        except (ValueError, OverflowError):
            return float('inf')

        score = self._evaluate_parameters(params)

        # Verificar si el score es válido
        if np.isnan(score) or np.isinf(score):
            return float('inf')

        # Guardar resultado
        result_dict = params.copy()
        result_dict['score'] = score
        iteration_results.append(result_dict)

        # Llamar progress_callback en cada evaluación
        if progress_callback:
            current_evals = len(iteration_results)
            progress_callback(current_evals, total_evaluations)

        return -score  # Scipy minimiza

    # Ejecutar optimización con paralelización
    # Nota: workers > 1 requiere updating='deferred'
    result = differential_evolution(
        objective,
        bounds,
        strategy='best1bin',
        maxiter=max_generations,
        popsize=population_size,
        mutation=mutation,
        recombination=recombination,
        seed=42,
        disp=verbose,
        workers=n_jobs,  # Paralelización habilitada
        updating='deferred' if n_jobs > 1 else 'immediate'  # Requerido para paralelo
    )

    # Procesar mejor resultado con snap to step
    best_params = {}
    for i, param in enumerate(self.parameter_space):
        if param.param_type == 'categorical' and param.name in categorical_params:
            best_params[param.name] = categorical_params[param.name][int(round(result.x[i]))]
        elif param.param_type == 'int':
            low = param.low if param.low is not None else 0
            high = param.high if param.high is not None else 100
            best_params[param.name] = _snap_to_step(result.x[i], low, high, param.step, 'int')
        else:
            low = param.low if param.low is not None else 0
            high = param.high if param.high is not None else 1
            best_params[param.name] = _snap_to_step(result.x[i], low, high, param.step, 'float')

    best_score = -result.fun

    results_df = pd.DataFrame(iteration_results)

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
        method='Genetic Algorithm (Parallel)',
        iterations=len(iteration_results)
    )
