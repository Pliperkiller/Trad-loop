import numpy as np
import pandas as pd

from .optimization_types import OptimizationResult


def random_search(self, n_iter: int = 100, verbose: bool = True) -> OptimizationResult:
    """
    Random Search: Muestrea aleatoriamente del espacio de parámetros
    
    Pros:
    - Más eficiente que Grid Search
    - Funciona bien con espacios continuos
    - Explora mejor el espacio en alta dimensionalidad
    
    Contras:
    - No garantiza encontrar el óptimo
    - Puede repetir regiones ya exploradas
    
    Mejor para: 3-5 parámetros, exploración rápida
    """
    import time
    start_time = time.time()
    
    if verbose:
        print("\n" + "="*70)
        print("INICIANDO RANDOM SEARCH")
        print("="*70)
        print(f"Iteraciones: {n_iter}\n")
    
    results = []
    
    for i in range(n_iter):
        # Generar parámetros aleatorios
        params = {}
        for param in self.parameter_space:
            if param.param_type == 'int':
                params[param.name] = np.random.randint(param.low, param.high + 1)
            elif param.param_type == 'float':
                params[param.name] = np.random.uniform(param.low, param.high)
            else:  # categorical
                params[param.name] = np.random.choice(param.values)
        
        # Evaluar
        result = self._evaluate_parameters_detailed(params)

        # Verificar si el score es valido
        if result.get('score') is None or np.isnan(result['score']) or np.isinf(result['score']):
            result['score'] = -1e10  # Penalizacion para scores invalidos

        results.append(result)

        # Calcular mejor score valido para mostrar progreso
        valid_scores = [r['score'] for r in results if not np.isnan(r['score']) and not np.isinf(r['score'])]
        best_so_far = max(valid_scores) if valid_scores else 0

        if verbose and (i + 1) % max(1, n_iter // 20) == 0:
            print(f"Progreso: {i+1}/{n_iter} ({(i+1)/n_iter*100:.1f}%) - Best score: {best_so_far:.4f}")
    
    # Procesar resultados
    results_df = pd.DataFrame(results)
    best_idx = results_df['score'].idxmax()
    best_result = results_df.iloc[best_idx]
    
    param_names = [p.name for p in self.parameter_space]
    best_params = {name: best_result[name] for name in param_names}
    best_score = best_result['score']
    
    optimization_time = time.time() - start_time
    
    return OptimizationResult(
        best_params=best_params,
        best_score=best_score,
        all_results=results_df,
        optimization_time=optimization_time,
        method='Random Search',
        iterations=n_iter
    )