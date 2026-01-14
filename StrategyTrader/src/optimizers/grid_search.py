from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools
import numpy as np
import pandas as pd
from typing import Callable, Optional

from .optimization_types import OptimizationResult


def grid_search(self, verbose: bool = True,
                progress_callback: Optional[Callable[[int, int], None]] = None) -> OptimizationResult:
    """
    Grid Search: Prueba todas las combinaciones posibles
    
    Pros:
    - Garantiza encontrar el óptimo global en el espacio discretizado
    - Fácil de entender y reproducible
    
    Contras:
    - Computacionalmente muy costoso
    - Crece exponencialmente con el número de parámetros
    - No funciona bien con espacios continuos grandes
    
    Mejor para: 2-3 parámetros con rangos pequeños
    """
    import time
    start_time = time.time()
    
    if verbose:
        print("\n" + "="*70)
        print("INICIANDO GRID SEARCH")
        print("="*70)
        print("Parámetros con step:")
        for p in self.parameter_space:
            step_info = f"step={p.step}" if p.step else "auto"
            print(f"  - {p.name}: [{p.low}, {p.high}] {step_info} (type={p.param_type})")
        print()

    # Generar grid de parámetros
    param_grids = []
    param_names = []
    
    for param in self.parameter_space:
        param_names.append(param.name)
        
        if param.param_type == 'int':
            step = param.step if param.step else 1
            values = list(range(int(param.low), int(param.high) + 1, int(step)))
        elif param.param_type == 'float':
            step = param.step if param.step else (param.high - param.low) / 10
            values = list(np.arange(param.low, param.high + step, step))
        elif param.param_type == 'categorical' and param.values:
            values = param.values
        else:
            # Fallback: tratar como float si tiene low/high definidos
            if param.low is not None and param.high is not None:
                step = param.step if param.step else (param.high - param.low) / 10
                values = list(np.arange(param.low, param.high + step, step))
            else:
                values = [0]  # Valor por defecto seguro
        
        param_grids.append(values)
        if verbose:
            # Mostrar primeros y últimos valores del grid
            if len(values) <= 5:
                print(f"    {param.name}: {values}")
            else:
                print(f"    {param.name}: {values[:3]} ... {values[-2:]} ({len(values)} valores)")

    # Generar todas las combinaciones
    all_combinations = list(itertools.product(*param_grids))
    total_combinations = len(all_combinations)
    
    if verbose:
        print(f"Total de combinaciones a evaluar: {total_combinations}")
        print(f"Parametros: {param_names}\n")
    
    # Evaluar todas las combinaciones
    results = []
    
    if self.n_jobs and self.n_jobs != 1:
        # Evaluación paralela
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = {}
            for combo in all_combinations:
                params = dict(zip(param_names, combo))
                future = executor.submit(self._evaluate_parameters_detailed, params)
                futures[future] = params
            
            completed = 0
            for future in as_completed(futures):
                result = future.result()
                # Verificar si el score es valido
                if result.get('score') is None or np.isnan(result['score']) or np.isinf(result['score']):
                    result['score'] = -1e10
                results.append(result)
                completed += 1

                if verbose and completed % max(1, total_combinations // 20) == 0:
                    print(f"Progreso: {completed}/{total_combinations} ({completed/total_combinations*100:.1f}%)")

                if progress_callback:
                    progress_callback(completed, total_combinations)
    else:
        # Evaluación secuencial
        for i, combo in enumerate(all_combinations):
            params = dict(zip(param_names, combo))
            result = self._evaluate_parameters_detailed(params)
            # Verificar si el score es valido
            if result.get('score') is None or np.isnan(result['score']) or np.isinf(result['score']):
                result['score'] = -1e10
            results.append(result)

            if verbose and (i + 1) % max(1, total_combinations // 20) == 0:
                print(f"Progreso: {i+1}/{total_combinations} ({(i+1)/total_combinations*100:.1f}%)")

            if progress_callback:
                progress_callback(i + 1, total_combinations)
    
    # Procesar resultados
    results_df = pd.DataFrame(results)
    best_idx = results_df['score'].idxmax()
    best_result = results_df.iloc[best_idx]
    
    best_params = {name: best_result[name] for name in param_names}
    best_score = best_result['score']
    
    optimization_time = time.time() - start_time
    
    return OptimizationResult(
        best_params=best_params,
        best_score=best_score,
        all_results=results_df,
        optimization_time=optimization_time,
        method='Grid Search',
        iterations=total_combinations
    )