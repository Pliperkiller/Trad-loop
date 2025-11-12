import pandas as pd
import time
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args

from .optimization_types import OptimizationResult


def bayesian_optimization(self, n_calls: int = 50, n_initial_points: int = 10, 
                        verbose: bool = True) -> OptimizationResult:
    """
    Bayesian Optimization: Usa un modelo probabilístico para guiar la búsqueda
    
    Pros:
    - Muy eficiente en número de evaluaciones
    - Balancea exploración vs explotación
    - Excelente para funciones costosas de evaluar
    - Maneja bien espacios continuos
    
    Contras:
    - Requiere librería adicional (scikit-optimize)
    - Más complejo de entender
    - Puede quedar atrapado en óptimos locales
    
    Mejor para: Parámetros continuos, presupuesto limitado de evaluaciones
    """


    start_time = time.time()
    
    if verbose:
        print("\n" + "="*70)
        print("INICIANDO BAYESIAN OPTIMIZATION")
        print("="*70)
        print(f"Evaluaciones: {n_calls}")
        print(f"Puntos iniciales aleatorios: {n_initial_points}\n")
    
    # Definir espacio de búsqueda para skopt
    space = []
    param_names = []
    
    for param in self.parameter_space:
        param_names.append(param.name)
        
        if param.param_type == 'int':
            space.append(Integer(int(param.low), int(param.high), name=param.name))
        elif param.param_type == 'float':
            space.append(Real(param.low, param.high, name=param.name))
        else:  # categorical
            space.append(Integer(0, len(param.values) - 1, name=param.name))
    
    # Función objetivo para skopt
    @use_named_args(space)
    def objective(**params):
        # Convertir categorical indices a valores
        for param in self.parameter_space:
            if param.param_type == 'categorical':
                params[param.name] = param.values[params[param.name]]
        
        score = self._evaluate_parameters(params)
        return -score  # Skopt minimiza, nosotros maximizamos
    
    # Ejecutar optimización
    result = gp_minimize(
        objective,
        space,
        n_calls=n_calls,
        n_initial_points=n_initial_points,
        random_state=42,
        verbose=verbose
    )
    
    # Procesar resultados
    best_params = {}
    for i, param in enumerate(self.parameter_space):
        if param.param_type == 'categorical':
            best_params[param.name] = param.values[result.x[i]]
        else:
            best_params[param.name] = result.x[i]
    
    best_score = -result.fun
    
    # Crear DataFrame con historial
    results_data = []
    for x_vals, y_val in zip(result.x_iters, result.func_vals):
        params_dict = {}
        for i, param in enumerate(self.parameter_space):
            if param.param_type == 'categorical':
                params_dict[param.name] = param.values[x_vals[i]]
            else:
                params_dict[param.name] = x_vals[i]
        params_dict['score'] = -y_val
        results_data.append(params_dict)
    
    results_df = pd.DataFrame(results_data)
    
    optimization_time = time.time() - start_time
    
    return OptimizationResult(
        best_params=best_params,
        best_score=best_score,
        all_results=results_df,
        optimization_time=optimization_time,
        method='Bayesian Optimization',
        iterations=n_calls
    )