"""
Random Search Optimizer con paralelización y respeto de step.
"""

import os
import numpy as np
import pandas as pd
from typing import Callable, Optional, List, Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed

from .optimization_types import OptimizationResult, ParameterSpace


def _generate_random_params(parameter_space: List[ParameterSpace]) -> Dict[str, Any]:
    """
    Genera un conjunto de parámetros aleatorios respetando el step.

    Para floats: genera valores discretos como 0.1, 0.15, 0.2, 0.25...
    Para ints: respeta el step como 10, 15, 20, 25...

    IMPORTANTE: Retorna tipos nativos de Python (int, float) no numpy types.
    """
    params = {}
    for param in parameter_space:
        if param.param_type == 'int':
            # Respetar step para enteros
            step = int(param.step) if param.step and param.step >= 1 else 1
            low = int(param.low) if param.low is not None else 0
            high = int(param.high) if param.high is not None else 100

            # Calcular número de pasos posibles
            num_steps = (high - low) // step
            random_step = int(np.random.randint(0, num_steps + 1))
            # Convertir explícitamente a int nativo de Python
            params[param.name] = int(low + (random_step * step))

        elif param.param_type == 'float':
            # Respetar step para floats
            step = float(param.step) if param.step and param.step > 0 else 0.1
            low = float(param.low) if param.low is not None else 0.0
            high = float(param.high) if param.high is not None else 1.0

            # Calcular número de pasos posibles
            num_steps = int((high - low) / step)
            if num_steps < 1:
                num_steps = 1
            random_step = int(np.random.randint(0, num_steps + 1))
            # Redondear para evitar errores de punto flotante
            value = low + (random_step * step)
            # Convertir explícitamente a float nativo de Python
            params[param.name] = float(round(value, 10))

        elif param.param_type == 'categorical' and param.values:
            choice = np.random.choice(param.values)
            # Convertir numpy types a Python natives si es necesario
            if hasattr(choice, 'item'):
                params[param.name] = choice.item()
            else:
                params[param.name] = choice
        else:
            # Fallback: tratar como float con step
            if param.low is not None and param.high is not None:
                step = float(param.step) if param.step and param.step > 0 else 0.1
                num_steps = int((param.high - param.low) / step)
                if num_steps < 1:
                    num_steps = 1
                random_step = int(np.random.randint(0, num_steps + 1))
                params[param.name] = float(round(param.low + (random_step * step), 10))
            else:
                params[param.name] = 0

    return params


def random_search(self, n_iter: int = 100, verbose: bool = True,
                  progress_callback: Optional[Callable[[int, int], None]] = None,
                  n_jobs: int = -1) -> OptimizationResult:
    """
    Random Search: Muestrea aleatoriamente del espacio de parámetros.

    MEJORAS:
    - Respeta el step definido para cada parámetro
    - Paralelización con ProcessPoolExecutor para evaluaciones concurrentes

    Args:
        n_iter: Número de iteraciones/combinaciones a evaluar
        verbose: Mostrar progreso
        progress_callback: Callback para reportar progreso (current, total)
        n_jobs: Número de workers paralelos (-1 = todos los cores)

    Pros:
    - Más eficiente que Grid Search
    - Funciona bien con espacios discretos
    - Explora mejor el espacio en alta dimensionalidad
    - Paralelizado para máximo rendimiento

    Contras:
    - No garantiza encontrar el óptimo
    - Puede repetir regiones ya exploradas

    Mejor para: 3-5 parámetros, exploración rápida
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
        print("INICIANDO RANDOM SEARCH (PARALELO)")
        print("="*70)
        print(f"Iteraciones: {n_iter}")
        print(f"Workers paralelos: {n_jobs}")
        print(f"Parámetros con step:")
        for p in self.parameter_space:
            step_info = f"step={p.step}" if p.step else "continuo"
            print(f"  - {p.name}: [{p.low}, {p.high}] {step_info}")
        print()

    # Generar todos los conjuntos de parámetros por adelantado
    all_params = [_generate_random_params(self.parameter_space) for _ in range(n_iter)]

    results = []
    completed = 0

    # Ejecutar evaluaciones en paralelo
    if n_jobs > 1 and n_iter > 1:
        # Modo paralelo
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            # Enviar todas las tareas
            future_to_params = {
                executor.submit(self._evaluate_parameters_detailed, params): params
                for params in all_params
            }

            # Recolectar resultados a medida que completan
            for future in as_completed(future_to_params):
                try:
                    result = future.result()

                    # Verificar si el score es válido
                    if result.get('score') is None or np.isnan(result['score']) or np.isinf(result['score']):
                        result['score'] = -1e10

                    results.append(result)
                    completed += 1

                    # Reportar progreso
                    if progress_callback:
                        progress_callback(completed, n_iter)

                    if verbose and completed % max(1, n_iter // 10) == 0:
                        valid_scores = [r['score'] for r in results
                                       if r['score'] > -1e9]
                        best_so_far = max(valid_scores) if valid_scores else 0
                        print(f"Progreso: {completed}/{n_iter} ({completed/n_iter*100:.1f}%) - Best: {best_so_far:.4f}")

                except Exception as e:
                    if verbose:
                        print(f"Error en evaluación: {e}")
                    results.append({'score': -1e10})
                    completed += 1
    else:
        # Modo secuencial (fallback para n_jobs=1 o debug)
        for i, params in enumerate(all_params):
            result = self._evaluate_parameters_detailed(params)

            if result.get('score') is None or np.isnan(result['score']) or np.isinf(result['score']):
                result['score'] = -1e10

            results.append(result)
            completed += 1

            if progress_callback:
                progress_callback(completed, n_iter)

            if verbose and (completed) % max(1, n_iter // 10) == 0:
                valid_scores = [r['score'] for r in results if r['score'] > -1e9]
                best_so_far = max(valid_scores) if valid_scores else 0
                print(f"Progreso: {completed}/{n_iter} ({completed/n_iter*100:.1f}%) - Best: {best_so_far:.4f}")

    # Procesar resultados
    results_df = pd.DataFrame(results)

    if results_df.empty or 'score' not in results_df.columns:
        return OptimizationResult(
            best_params={},
            best_score=-np.inf,
            all_results=results_df,
            optimization_time=time.time() - start_time,
            method='Random Search (Parallel)',
            iterations=n_iter
        )

    best_idx = results_df['score'].idxmax()
    best_result = results_df.iloc[best_idx]

    # Función para convertir tipos numpy a tipos nativos de Python
    def to_native(value, param_type: str):
        """Convierte numpy types a Python natives."""
        if hasattr(value, 'item'):
            value = value.item()
        if param_type == 'int':
            return int(value)
        elif param_type == 'float':
            return float(value)
        return value

    # Crear lookup de tipos de parámetros
    param_type_lookup = {p.name: p.param_type for p in self.parameter_space}

    param_names = [p.name for p in self.parameter_space]
    best_params = {
        name: to_native(best_result[name], param_type_lookup.get(name, 'float'))
        for name in param_names if name in best_result
    }
    best_score = float(best_result['score']) if hasattr(best_result['score'], 'item') else best_result['score']

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
        method='Random Search (Parallel)',
        iterations=n_iter
    )
