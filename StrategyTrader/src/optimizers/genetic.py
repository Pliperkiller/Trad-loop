import pandas as pd
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')
from scipy.optimize import differential_evolution

from .optimization_types import OptimizationResult

def genetic_algorithm(self, population_size: int = 20, max_generations: int = 50,
                         mutation: Tuple[float, float] = (0.5, 1.0),
                         recombination: float = 0.7,
                         verbose: bool = True) -> OptimizationResult:
        """
        Genetic Algorithm (Differential Evolution): Evoluciona población de soluciones
        
        Pros:
        - Excelente para espacios no convexos
        - Robusto ante óptimos locales
        - No requiere gradientes
        - Balancea exploración global y local
        
        Contras:
        - Requiere muchas evaluaciones
        - Parámetros del algoritmo afectan rendimiento
        - Puede ser lento
        
        Mejor para: Espacios complejos con múltiples óptimos locales, 4-8 parámetros
        """
        import time
        start_time = time.time()
        
        if verbose:
            print("\n" + "="*70)
            print("INICIANDO GENETIC ALGORITHM (Differential Evolution)")
            print("="*70)
            print(f"Tamaño de poblacion: {population_size}")
            print(f"Generaciones maximas: {max_generations}\n")
        
        # Definir bounds
        bounds = []
        param_names = []
        categorical_params = {}
        
        for param in self.parameter_space:
            param_names.append(param.name)
            
            if param.param_type == 'categorical':
                bounds.append((0, len(param.values) - 1))
                categorical_params[param.name] = param.values
            elif param.param_type == 'int':
                bounds.append((param.low, param.high))
            else:  # float
                bounds.append((param.low, param.high))
        
        # Función objetivo
        iteration_results = []
        
        def objective(x):
            params = {}
            for i, (param, value) in enumerate(zip(self.parameter_space, x)):
                if param.param_type == 'categorical':
                    params[param.name] = categorical_params[param.name][int(round(value))]
                elif param.param_type == 'int':
                    params[param.name] = int(round(value))
                else:
                    params[param.name] = value
            
            score = self._evaluate_parameters(params)
            
            # Guardar resultado
            result_dict = params.copy()
            result_dict['score'] = score
            iteration_results.append(result_dict)
            
            return -score  # Scipy minimiza
        
        # Ejecutar optimización
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
            workers=1  # Serializado para mantener historial
        )
        
        # Procesar mejor resultado
        best_params = {}
        for i, param in enumerate(self.parameter_space):
            if param.param_type == 'categorical':
                best_params[param.name] = categorical_params[param.name][int(round(result.x[i]))]
            elif param.param_type == 'int':
                best_params[param.name] = int(round(result.x[i]))
            else:
                best_params[param.name] = result.x[i]
        
        best_score = -result.fun
        
        results_df = pd.DataFrame(iteration_results)
        
        optimization_time = time.time() - start_time
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            all_results=results_df,
            optimization_time=optimization_time,
            method='Genetic Algorithm',
            iterations=len(iteration_results)
        )