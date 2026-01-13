"""
Módulo de Optimización de Parámetros

NOTA: Este archivo contiene más de 2000 líneas de código.
Para obtener el código completo, consulta la conversación donde desarrollamos:
- StrategyOptimizer con 5 métodos de optimización
- Visualizador de resultados de optimización

ESTRUCTURA DEL MÓDULO:
====================

1. ParameterSpace (dataclass)
2. OptimizationResult (dataclass)
3. StrategyOptimizer
   - grid_search()
   - random_search()
   - bayesian_optimization()
   - genetic_algorithm()
   - walk_forward_optimization()
4. OptimizationVisualizer
   - plot_optimization_surface()
   - plot_convergence()
   - plot_parameter_importance()

PARA IMPLEMENTAR ESTE MÓDULO:
============================
Ver conversación donde desarrollamos este código completo
"""


import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Callable, Optional, Any
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

from .optimizers.optimization_types import ParameterSpace, OptimizationResult, WalkForwardResult
from .optimizers.grid_search import grid_search
from .optimizers.random_search import random_search
from .optimizers.bayesian import bayesian_optimization
from .optimizers.genetic import genetic_algorithm
from .optimizers.walk_forward import walk_forward_optimization

# Validation module imports
from .optimizers.validation import (
    TimeSeriesSplit,
    RollingWindowSplit,
    ExpandingWindowSplit,
    WalkForwardSplit,
    PurgedKFold,
    CombinatorialPurgedCV,
    TimeSeriesCV,
    WalkForwardCV,
    CrossValidationResult,
    SplitResult,
)
from .optimizers.analysis import (
    ParameterStabilityAnalyzer,
    OverfittingDetector,
    ValidationVisualizer,
)


class StrategyOptimizer:
    """
    Optimizador de parámetros con múltiples algoritmos
    
    Ejemplo de uso:
    ---------------
    optimizer = StrategyOptimizer(
        strategy_class=MyStrategy,
        data=dataframe,
        config_template=config,
        objective_metric='sharpe_ratio'
    )
    
    optimizer.add_parameter('fast_period', 'int', low=5, high=20, step=5)
    optimizer.add_parameter('slow_period', 'int', low=20, high=50, step=10)
    
    result = optimizer.bayesian_optimization(n_calls=50)
    result.print_summary()
    """
    
    def __init__(self, strategy_class: type, data: pd.DataFrame,
                 config_template: Dict, objective_metric: str = 'sharpe_ratio',
                 n_jobs: int = -1):
        self.strategy_class = strategy_class
        self.data = data
        self.config_template = config_template
        self.objective_metric = objective_metric
        self.n_jobs = n_jobs if n_jobs > 0 else None
        self.parameter_space: List[ParameterSpace] = []
        self.results_cache: Dict = {}
    
    def add_parameter(self, name: str, param_type: str, **kwargs):
        """
        Añade un parámetro al espacio de búsqueda
        
        Args:
            name: Nombre del parámetro
            param_type: Tipo ('int', 'float', 'categorical')
            **kwargs: low, high, step, values según el tipo
        """
        param = ParameterSpace(name=name, param_type=param_type, **kwargs)
        self.parameter_space.append(param)

    def _evaluate_parameters(self, params: Dict) -> float:
        """
        Evalúa un conjunto de parámetros y retorna el score
        """
        from .strategy import StrategyConfig

        # Crear hash para cache
        params_tuple = tuple(sorted(params.items()))
        if params_tuple in self.results_cache:
            return self.results_cache[params_tuple]

        try:
            # Crear config object desde dict si es necesario
            if isinstance(self.config_template, dict):
                config = StrategyConfig(**self.config_template)
            else:
                config = self.config_template

            # Crear instancia de estrategia con parámetros
            strategy = self.strategy_class(config, **params)
            
            # Ejecutar backtest
            strategy.load_data(self.data)
            strategy.backtest()
            
            # Obtener métricas
            metrics = strategy.get_performance_metrics()
            
            # Obtener score de la métrica objetivo
            score = metrics.get(self.objective_metric, -np.inf)
            
            # Validaciones adicionales
            if metrics.get('total_trades', 0) < 10:
                score = -np.inf  # Penalizar estrategias con muy pocos trades
            
            # Cachear resultado
            self.results_cache[params_tuple] = score
            
            return score
            
        except Exception as e:
            print(f"Error evaluando parametros {params}: {str(e)}")
            return -np.inf
    
    def _evaluate_parameters_detailed(self, params: Dict) -> Dict:
        """Evalúa parámetros y retorna todas las métricas"""
        from .strategy import StrategyConfig

        try:
            # Crear config object desde dict si es necesario
            if isinstance(self.config_template, dict):
                config = StrategyConfig(**self.config_template)
            else:
                config = self.config_template

            strategy = self.strategy_class(config, **params)
            strategy.load_data(self.data)
            strategy.backtest()
            metrics = strategy.get_performance_metrics()
            
            result = {'score': metrics.get(self.objective_metric, -np.inf)}
            result.update(params)
            result.update(metrics)
            
            return result
        except Exception as e:
            result = {'score': -np.inf}
            result.update(params)
            return result
    
    def grid_search(self, verbose: bool = True):
        return grid_search(self, verbose=verbose)


    def random_search(self, n_iter: int = 100, verbose: bool = True):
        return random_search(self, n_iter, verbose)


    def bayesian_optimization(self, n_calls: int = 50, verbose: bool = True):
        return bayesian_optimization(self, n_calls, verbose)

    def genetic_algorithm(self, population_size: int = 20,
                         max_generations: int = 50, verbose: bool = True):
        return genetic_algorithm(self, population_size, max_generations, verbose)
    
    def walk_forward_optimization(self, optimization_method: str = 'random',
                                 n_splits: int = 5, train_size: float = 0.6,
                                 n_iter: int = 50, verbose: bool = True) -> WalkForwardResult:
        """
        Walk Forward Optimization: Validacion robusta contra overfitting

        Divide los datos temporalmente y optimiza/valida en cada segmento
        para obtener una estimacion realista del performance futuro.

        Args:
            optimization_method: Metodo a usar ('grid', 'random', 'bayesian', 'genetic')
            n_splits: Numero de splits para walk-forward
            train_size: Proporcion de datos para entrenamiento (0.5-0.8)
            n_iter: Iteraciones para metodos estocasticos
            verbose: Mostrar progreso

        Returns:
            WalkForwardResult con metricas agregadas y score de robustez
        """
        return walk_forward_optimization(
            self,
            optimization_method=optimization_method,
            n_splits=n_splits,
            train_size=train_size,
            n_iter=n_iter,
            verbose=verbose
        )

    def walk_forward_advanced(
        self,
        optimization_method: str = 'random',
        n_splits: int = 5,
        train_pct: float = 0.6,
        n_iter: int = 50,
        anchored: bool = True,
        gap: int = 0,
        min_train_rows: int = 100,
        min_test_rows: int = 50,
        track_params: bool = True,
        verbose: bool = True,
    ) -> WalkForwardResult:
        """
        Walk-Forward avanzado con test periods no solapados.

        A diferencia de walk_forward_optimization(), este método:
        - Usa test periods estrictamente secuenciales (sin solapamiento)
        - Soporta modo anchored (expanding) y rolling
        - Incluye gap/embargo entre train y test
        - Trackea estabilidad de parámetros

        Args:
            optimization_method: 'grid', 'random', 'bayesian', 'genetic'
            n_splits: Número de splits walk-forward
            train_pct: Porcentaje inicial de datos para training
            n_iter: Iteraciones para métodos estocásticos
            anchored: True=expanding window, False=rolling window
            gap: Filas de separación entre train y test
            min_train_rows: Mínimo de filas para training
            min_test_rows: Mínimo de filas para test
            track_params: Trackear estabilidad de parámetros
            verbose: Mostrar progreso

        Returns:
            WalkForwardResult con métricas y análisis de robustez
        """
        return walk_forward_optimization(
            self,
            optimization_method=optimization_method,
            n_splits=n_splits,
            train_size=train_pct,
            n_iter=n_iter,
            anchored=anchored,
            gap=gap,
            min_train_rows=min_train_rows,
            min_test_rows=min_test_rows,
            track_params=track_params,
            verbose=verbose,
        )

    def cross_validate(
        self,
        optimization_method: str = 'random',
        n_splits: int = 5,
        n_iter: int = 50,
        gap: int = 0,
        max_train_size: Optional[int] = None,
        verbose: bool = True,
    ) -> CrossValidationResult:
        """
        Cross-validation temporal con K-Fold.

        Ejecuta optimización en cada fold y agrega resultados
        para obtener métricas robustas de generalización.

        Args:
            optimization_method: 'grid', 'random', 'bayesian', 'genetic'
            n_splits: Número de folds
            n_iter: Iteraciones para métodos estocásticos
            gap: Filas de separación entre train y test
            max_train_size: Tamaño máximo de train (None=expanding)
            verbose: Mostrar progreso

        Returns:
            CrossValidationResult con métricas agregadas
        """
        cv = TimeSeriesCV(
            n_splits=n_splits,
            gap=gap,
            max_train_size=max_train_size,
        )

        return cv.cross_validate(
            optimizer=self,
            optimization_method=optimization_method,
            n_iter=n_iter,
            verbose=verbose,
        )

    def purged_cross_validate(
        self,
        optimization_method: str = 'random',
        n_splits: int = 5,
        n_iter: int = 50,
        purge_gap: int = 0,
        embargo_pct: float = 0.01,
        verbose: bool = True,
    ) -> CrossValidationResult:
        """
        Cross-validation con Purged K-Fold para evitar data leakage.

        Útil cuando las features tienen lookback temporal que podría
        causar contaminación entre train y test sets.

        Args:
            optimization_method: 'grid', 'random', 'bayesian', 'genetic'
            n_splits: Número de folds
            n_iter: Iteraciones para métodos estocásticos
            purge_gap: Filas a purgar antes del test set
            embargo_pct: Porcentaje de embargo después del test
            verbose: Mostrar progreso

        Returns:
            CrossValidationResult con métricas agregadas
        """
        # Calculate gap from purge_gap + embargo
        total_gap = purge_gap + int(len(self.data) * embargo_pct)

        # Use WalkForwardCV with gap for similar effect
        cv = WalkForwardCV(
            n_splits=n_splits,
            train_pct=0.6,
            anchored=True,
            gap=total_gap,
        )

        return cv.cross_validate(
            optimizer=self,
            optimization_method=optimization_method,
            n_iter=n_iter,
            verbose=verbose,
        )

    def analyze_stability(
        self,
        wf_result: WalkForwardResult,
        stability_threshold: float = 0.3,
    ) -> Dict[str, Any]:
        """
        Analiza la estabilidad de parámetros a través de splits.

        Identifica parámetros que varían demasiado entre splits,
        lo cual puede indicar overfitting o inestabilidad.

        Args:
            wf_result: Resultado de walk_forward_optimization
            stability_threshold: Umbral de CV para considerar inestable

        Returns:
            Dict con análisis de estabilidad por parámetro
        """
        analyzer = ParameterStabilityAnalyzer(stability_threshold=stability_threshold)

        # Extraer resultados de splits
        splits_data = []
        if hasattr(wf_result, 'splits_results') and wf_result.splits_results:
            # Extract best_params from each split result
            for s in wf_result.splits_results:
                if isinstance(s, dict) and 'best_params' in s:
                    splits_data.append(s['best_params'])

        if not splits_data:
            return {
                'error': 'No split results available for stability analysis',
                'stable_params': [],
                'unstable_params': [],
            }

        stability = analyzer.analyze(splits_data)  # type: ignore[arg-type]

        return {
            'stability': {k: v.__dict__ for k, v in stability.items()},
            'stable_params': analyzer.get_stable_parameters(),
            'unstable_params': analyzer.get_unstable_parameters(),
            'summary': analyzer.get_summary(),
        }

    def detect_overfitting(
        self,
        is_sharpes: List[float],
        oos_sharpes: List[float],
        is_returns: Optional[np.ndarray] = None,
        oos_returns: Optional[np.ndarray] = None,
        n_trials: Optional[int] = None,
        significance_level: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Detecta señales de overfitting usando tests estadísticos.

        Implementa:
        - Deflated Sharpe Ratio (DSR)
        - Probability of Backtest Overfitting (PBO)
        - Test de degradación de performance

        Args:
            is_sharpes: Sharpe ratios in-sample por split/estrategia
            oos_sharpes: Sharpe ratios out-of-sample
            is_returns: Retornos in-sample (opcional, para test de degradación)
            oos_returns: Retornos out-of-sample (opcional)
            n_trials: Número de trials/estrategias probadas
            significance_level: Nivel de significancia para tests

        Returns:
            Dict con resultados de todos los tests de overfitting
        """
        detector = OverfittingDetector(significance_level=significance_level)
        results: Dict[str, Any] = {}

        # PBO
        if len(is_sharpes) >= 2 and len(oos_sharpes) >= 2:
            try:
                pbo = detector.probability_of_backtest_overfitting(
                    is_sharpes=is_sharpes,
                    oos_sharpes=oos_sharpes,
                )
                results['pbo'] = pbo
                results['pbo_interpretation'] = (
                    'High overfitting risk' if pbo > 0.5 else 'Acceptable'
                )
            except Exception as e:
                results['pbo_error'] = str(e)

        # DSR
        if n_trials and len(oos_sharpes) > 0:
            try:
                observed_sharpe = float(np.mean(oos_sharpes))
                variance_sharpe = float(np.var(oos_sharpes)) if len(oos_sharpes) > 1 else 0.5
                dsr_result = detector.deflated_sharpe_ratio(
                    observed_sharpe=observed_sharpe,
                    n_trials=n_trials,
                    variance_sharpe=max(0.01, variance_sharpe),
                )
                results['dsr'] = {
                    'statistic': dsr_result.statistic,
                    'p_value': dsr_result.p_value,
                    'is_overfitted': dsr_result.is_overfitted,
                }
            except Exception as e:
                results['dsr_error'] = str(e)

        # Degradation test
        if is_returns is not None and oos_returns is not None:
            if len(is_returns) >= 10 and len(oos_returns) >= 10:
                try:
                    deg_result = detector.performance_degradation_test(
                        is_returns=is_returns,
                        oos_returns=oos_returns,
                    )
                    results['degradation'] = {
                        'degradation_pct': deg_result.details.get('degradation', 0),
                        'p_value': deg_result.p_value,
                        'is_overfitted': deg_result.is_overfitted,
                    }
                except Exception as e:
                    results['degradation_error'] = str(e)

        # Overall assessment
        overfitting_signals = 0
        if results.get('pbo', 0) > 0.5:
            overfitting_signals += 1
        if results.get('dsr', {}).get('is_overfitted', False):
            overfitting_signals += 1
        if results.get('degradation', {}).get('is_overfitted', False):
            overfitting_signals += 1

        results['overfitting_signals'] = overfitting_signals
        results['overall_assessment'] = (
            'High risk' if overfitting_signals >= 2 else
            'Moderate risk' if overfitting_signals == 1 else
            'Low risk'
        )

        return results

    def visualize_validation(
        self,
        wf_result: WalkForwardResult,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Genera visualizaciones de resultados de validación.

        Args:
            wf_result: Resultado de walk_forward_optimization
            save_path: Ruta para guardar figuras (opcional)
        """
        # Convert WalkForwardResult to CrossValidationResult for visualization
        from .optimizers.walk_forward import convert_to_validation_result

        try:
            cv_result = convert_to_validation_result(wf_result)
            viz = ValidationVisualizer()

            # Equity curves
            viz.plot_equity_curves(cv_result)

            # Degradation heatmap si hay suficientes datos
            if len(cv_result.splits) > 1:
                viz.plot_degradation_heatmap(cv_result)

            # Parameter stability
            if cv_result.parameter_stability:
                viz.plot_parameter_stability(cv_result.parameter_stability)

            # Robustness breakdown
            viz.plot_robustness_breakdown(cv_result)

        except Exception as e:
            print(f"Error generating visualizations: {e}")
            print("Visualization requires matplotlib. Install with: pip install matplotlib")

    def comprehensive_validation(
        self,
        optimization_method: str = 'random',
        n_iter: int = 50,
        n_splits: int = 5,
        train_pct: float = 0.6,
        purge_gap: int = 0,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Validación completa: Walk-Forward + Stability + Overfitting detection.

        Ejecuta todo el pipeline de validación y retorna un reporte completo.

        Args:
            optimization_method: 'grid', 'random', 'bayesian', 'genetic'
            n_iter: Iteraciones para métodos estocásticos
            n_splits: Número de splits
            train_pct: Porcentaje de datos para training
            purge_gap: Gap de purga (0 si no hay lookback features)
            verbose: Mostrar progreso

        Returns:
            Dict con todos los resultados de validación
        """
        results: Dict[str, Any] = {}

        # 1. Walk-Forward avanzado
        if verbose:
            print("=" * 60)
            print("COMPREHENSIVE VALIDATION")
            print("=" * 60)
            print("\n[1/3] Running Walk-Forward Analysis...")

        wf_result = self.walk_forward_advanced(
            optimization_method=optimization_method,
            n_splits=n_splits,
            train_pct=train_pct,
            n_iter=n_iter,
            gap=purge_gap,
            track_params=True,
            verbose=verbose,
        )

        results['walk_forward'] = {
            'robustness_score': wf_result.robustness_score,
            'avg_is_score': wf_result.avg_is_score,
            'avg_oos_score': wf_result.avg_oos_score,
            'consistency_ratio': wf_result.consistency_ratio,
            'best_params': wf_result.best_params,
        }

        # 2. Stability Analysis
        if verbose:
            print("\n[2/3] Analyzing Parameter Stability...")

        stability = self.analyze_stability(wf_result)
        results['stability'] = stability

        # 3. Overfitting Detection
        if verbose:
            print("\n[3/3] Detecting Overfitting Signals...")

        is_sharpes = [s.get('sharpe_ratio', s.get('score', 0))
                      for s in wf_result.splits_results] if wf_result.splits_results else []
        oos_sharpes = [s.get('oos_sharpe', s.get('oos_score', 0))
                       for s in wf_result.splits_results] if wf_result.splits_results else []

        # Estimate n_trials from parameter space
        n_trials = n_iter * n_splits

        overfitting = self.detect_overfitting(
            is_sharpes=is_sharpes,
            oos_sharpes=oos_sharpes,
            n_trials=n_trials,
        )
        results['overfitting'] = overfitting

        # Summary
        results['summary'] = {
            'robustness_score': wf_result.robustness_score,
            'unstable_params': stability.get('unstable_params', []),
            'overfitting_risk': overfitting.get('overall_assessment', 'Unknown'),
            'recommendation': self._get_validation_recommendation(
                wf_result.robustness_score,
                len(stability.get('unstable_params', [])),
                overfitting.get('overfitting_signals', 0),
            ),
        }

        if verbose:
            print("\n" + "=" * 60)
            print("VALIDATION SUMMARY")
            print("=" * 60)
            print(f"Robustness Score: {results['summary']['robustness_score']:.2f}")
            print(f"Unstable Parameters: {results['summary']['unstable_params']}")
            print(f"Overfitting Risk: {results['summary']['overfitting_risk']}")
            print(f"Recommendation: {results['summary']['recommendation']}")
            print("=" * 60)

        return results

    def _get_validation_recommendation(
        self,
        robustness: float,
        n_unstable: int,
        overfitting_signals: int,
    ) -> str:
        """Genera recomendación basada en resultados de validación."""
        if robustness >= 0.7 and n_unstable == 0 and overfitting_signals == 0:
            return "Strategy appears robust. Consider paper trading."
        elif robustness >= 0.5 and overfitting_signals <= 1:
            return "Moderate robustness. Review unstable parameters before deployment."
        elif robustness >= 0.3:
            return "Low robustness. Simplify strategy or gather more data."
        else:
            return "High overfitting risk. Do not deploy. Redesign strategy."


class OptimizationVisualizer:
    """Visualizador de resultados de optimización"""
    
    def __init__(self, optimization_result: OptimizationResult):
        self.result = optimization_result
    
    def plot_optimization_surface(self, param1: str, param2: str):
        """Superficie de optimización - IMPLEMENTAR"""
        print("NOTA: Implementar visualización de superficie")
    
    def plot_convergence(self):
        """Convergencia del algoritmo - IMPLEMENTAR"""
        print("NOTA: Implementar gráfico de convergencia")


# ============================================================================
# INSTRUCCIONES PARA IMPLEMENTAR
# ============================================================================

"""
COMPARACIÓN DE MÉTODOS:

| Método          | Mejor Para           | Evaluaciones | Garantías        |
|-----------------|---------------------|--------------|------------------|
| Grid Search     | 2-3 parámetros      | MUY ALTO     | Óptimo global    |
| Random Search   | Exploración rápida  | MEDIO        | Ninguna          |
| Bayesian Opt    | Presupuesto limitado| BAJO         | Sample efficient |
| Genetic Algo    | Espacios complejos  | ALTO         | Robusto          |
| Walk Forward    | Validación final    | MUY ALTO     | Anti-overfitting |

WORKFLOW RECOMENDADO:
1. Random Search (100 iter) - Exploración
2. Bayesian Opt (50 calls) - Refinamiento
3. Walk Forward (5 splits) - Validación

"""
