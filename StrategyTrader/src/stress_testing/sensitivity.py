"""
Analisis de sensibilidad para stress testing.

Evalua como cambian los resultados de la estrategia
al variar los parametros de entrada.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Callable, Any
import numpy as np

from .models import SensitivityConfig, SensitivityResult


@dataclass
class ParameterRange:
    """Rango de valores para un parametro"""
    name: str
    base_value: float
    min_value: float
    max_value: float
    step: Optional[float] = None
    n_steps: int = 5

    def get_values(self) -> List[float]:
        """Genera lista de valores a probar"""
        if self.step:
            values = np.arange(self.min_value, self.max_value + self.step, self.step)
        else:
            values = np.linspace(self.min_value, self.max_value, self.n_steps)
        return values.tolist()


class SensitivityAnalyzer:
    """
    Analizador de sensibilidad de parametros.

    Evalua la robustez de una estrategia variando sus parametros
    y midiendo el impacto en las metricas de rendimiento.

    Ejemplo de uso:
        analyzer = SensitivityAnalyzer()
        result = analyzer.analyze_parameter(
            strategy_func=run_strategy,
            param_name="sma_period",
            base_value=20,
            variation_pct=0.30,  # ±30%
        )
    """

    def __init__(self, config: Optional[SensitivityConfig] = None):
        """
        Args:
            config: Configuracion del analisis
        """
        self.config = config or SensitivityConfig()

    def analyze_parameter(
        self,
        strategy_func: Callable[[Dict[str, Any]], Dict[str, float]],
        param_name: str,
        base_value: float,
        variation_pct: Optional[float] = None,
        values: Optional[List[float]] = None,
        other_params: Optional[Dict[str, Any]] = None,
        metric: str = "sharpe_ratio"
    ) -> SensitivityResult:
        """
        Analiza sensibilidad de un parametro.

        Args:
            strategy_func: Funcion que recibe dict de parametros y retorna metricas
            param_name: Nombre del parametro a variar
            base_value: Valor base del parametro
            variation_pct: Variacion porcentual (ej: 0.20 = ±20%)
            values: Lista especifica de valores a probar
            other_params: Otros parametros fijos
            metric: Metrica a evaluar

        Returns:
            SensitivityResult
        """
        other_params = other_params or {}

        # Determinar valores a probar
        if values:
            test_values = values
        else:
            var_pct = variation_pct or self.config.variation_pct
            min_val = base_value * (1 - var_pct)
            max_val = base_value * (1 + var_pct)
            test_values = np.linspace(min_val, max_val, self.config.n_steps).tolist()

        # Ejecutar estrategia para cada valor
        metric_values = []

        for val in test_values:
            params = {**other_params, param_name: val}
            try:
                result = strategy_func(params)
                metric_values.append(result.get(metric, 0.0))
            except Exception:
                metric_values.append(float('nan'))

        # Analizar resultados
        return self._build_result(
            param_name=param_name,
            base_value=base_value,
            tested_values=test_values,
            metric_values=metric_values,
            metric=metric,
        )

    def analyze_multiple_parameters(
        self,
        strategy_func: Callable[[Dict[str, Any]], Dict[str, float]],
        parameters: Dict[str, ParameterRange],
        metric: str = "sharpe_ratio"
    ) -> List[SensitivityResult]:
        """
        Analiza sensibilidad de multiples parametros.

        Args:
            strategy_func: Funcion de estrategia
            parameters: Dict de parametros con sus rangos
            metric: Metrica a evaluar

        Returns:
            Lista de SensitivityResult
        """
        results = []

        for param_name, param_range in parameters.items():
            # Crear dict con valores base de otros parametros
            other_params = {
                name: pr.base_value
                for name, pr in parameters.items()
                if name != param_name
            }

            result = self.analyze_parameter(
                strategy_func=strategy_func,
                param_name=param_name,
                base_value=param_range.base_value,
                values=param_range.get_values(),
                other_params=other_params,
                metric=metric,
            )
            results.append(result)

        return results

    def analyze_grid(
        self,
        strategy_func: Callable[[Dict[str, Any]], Dict[str, float]],
        param1: ParameterRange,
        param2: ParameterRange,
        metric: str = "sharpe_ratio"
    ) -> Dict:
        """
        Analisis de sensibilidad 2D (grid).

        Args:
            strategy_func: Funcion de estrategia
            param1: Primer parametro
            param2: Segundo parametro
            metric: Metrica a evaluar

        Returns:
            Dict con resultados del grid
        """
        values1 = param1.get_values()
        values2 = param2.get_values()

        # Matriz de resultados
        results_matrix = np.zeros((len(values1), len(values2)))

        for i, v1 in enumerate(values1):
            for j, v2 in enumerate(values2):
                params = {param1.name: v1, param2.name: v2}
                try:
                    result = strategy_func(params)
                    results_matrix[i, j] = result.get(metric, 0.0)
                except Exception:
                    results_matrix[i, j] = float('nan')

        # Encontrar optimo
        if not np.all(np.isnan(results_matrix)):
            max_idx = np.unravel_index(np.nanargmax(results_matrix), results_matrix.shape)
            optimal = {
                param1.name: values1[max_idx[0]],
                param2.name: values2[max_idx[1]],
                "metric_value": float(results_matrix[max_idx]),
            }
        else:
            optimal = None

        return {
            "param1": {
                "name": param1.name,
                "values": values1,
            },
            "param2": {
                "name": param2.name,
                "values": values2,
            },
            "results_matrix": results_matrix.tolist(),
            "metric": metric,
            "optimal": optimal,
        }

    def _build_result(
        self,
        param_name: str,
        base_value: float,
        tested_values: List[float],
        metric_values: List[float],
        metric: str
    ) -> SensitivityResult:
        """Construye el resultado del analisis"""
        # Filtrar NaN
        valid_mask = ~np.isnan(metric_values)
        valid_values = np.array(tested_values)[valid_mask]
        valid_metrics = np.array(metric_values)[valid_mask]

        if len(valid_metrics) == 0:
            return SensitivityResult(
                parameter_name=param_name,
                base_value=base_value,
                tested_values=tested_values,
                metric_values=metric_values,
                is_robust=False,
                sensitivity_score=1.0,
            )

        # Encontrar optimo
        optimal_idx = np.argmax(valid_metrics)
        optimal_value = float(valid_values[optimal_idx])
        optimal_metric = float(valid_metrics[optimal_idx])

        # Calcular estadisticas
        metric_mean = float(np.mean(valid_metrics))
        metric_std = float(np.std(valid_metrics))
        metric_min = float(np.min(valid_metrics))
        metric_max = float(np.max(valid_metrics))

        # Calcular score de sensibilidad (0 = no sensible, 1 = muy sensible)
        if metric_mean != 0:
            sensitivity_score = metric_std / abs(metric_mean)
        else:
            sensitivity_score = 1.0

        sensitivity_score = min(sensitivity_score, 1.0)

        # Determinar si es robusto
        # Es robusto si al menos 60% de los valores dan resultados positivos
        threshold = metric_mean * 0.5 if metric_mean > 0 else 0
        working_count = int(np.sum(valid_metrics > threshold))
        is_robust = bool(working_count >= len(valid_metrics) * 0.6)

        # Determinar rango de trabajo
        working_indices = np.where(valid_metrics > threshold)[0]
        if len(working_indices) > 0:
            working_range = (
                float(valid_values[working_indices[0]]),
                float(valid_values[working_indices[-1]])
            )
        else:
            working_range = (0.0, 0.0)

        return SensitivityResult(
            parameter_name=param_name,
            base_value=base_value,
            tested_values=tested_values,
            metric_values=metric_values,
            optimal_value=optimal_value,
            optimal_metric=optimal_metric,
            is_robust=is_robust,
            working_range=working_range,
            sensitivity_score=sensitivity_score,
            metric_mean=metric_mean,
            metric_std=metric_std,
            metric_min=metric_min,
            metric_max=metric_max,
        )


class RobustnessAnalyzer:
    """Analisis avanzado de robustez"""

    @staticmethod
    def calculate_robustness_score(results: List[SensitivityResult]) -> float:
        """
        Calcula score de robustez global (0-100).

        Args:
            results: Lista de resultados de sensibilidad

        Returns:
            Score de robustez (0-100)
        """
        if not results:
            return 0.0

        # Componentes del score
        scores = []

        for r in results:
            # Penalizar alta sensibilidad
            sensitivity_penalty = 1 - r.sensitivity_score

            # Bonus por robustez
            robust_bonus = 1.0 if r.is_robust else 0.5

            # Penalizar si el optimo esta lejos del base
            if r.base_value != 0:
                distance = abs(r.optimal_value - r.base_value) / abs(r.base_value)
                distance_penalty = max(0, 1 - distance)
            else:
                distance_penalty = 1.0

            param_score = (sensitivity_penalty + robust_bonus + distance_penalty) / 3
            scores.append(param_score)

        return float(np.mean(scores) * 100)

    @staticmethod
    def identify_fragile_parameters(
        results: List[SensitivityResult],
        threshold: float = 0.5
    ) -> List[str]:
        """
        Identifica parametros fragiles (alta sensibilidad).

        Args:
            results: Lista de resultados
            threshold: Umbral de sensibilidad (0-1)

        Returns:
            Lista de nombres de parametros fragiles
        """
        fragile = []
        for r in results:
            if r.sensitivity_score > threshold or not r.is_robust:
                fragile.append(r.parameter_name)
        return fragile

    @staticmethod
    def get_optimal_parameters(results: List[SensitivityResult]) -> Dict[str, float]:
        """
        Obtiene valores optimos de todos los parametros.

        Args:
            results: Lista de resultados

        Returns:
            Dict parametro -> valor optimo
        """
        return {r.parameter_name: r.optimal_value for r in results}

    @staticmethod
    def generate_report(results: List[SensitivityResult]) -> Dict:
        """Genera reporte de sensibilidad"""
        if not results:
            return {}

        robustness_score = RobustnessAnalyzer.calculate_robustness_score(results)
        fragile = RobustnessAnalyzer.identify_fragile_parameters(results)
        optimal = RobustnessAnalyzer.get_optimal_parameters(results)

        # Clasificar robustez
        if robustness_score >= 80:
            rating = "ROBUST"
        elif robustness_score >= 60:
            rating = "MODERATE"
        elif robustness_score >= 40:
            rating = "FRAGILE"
        else:
            rating = "VERY_FRAGILE"

        # Estadisticas
        avg_sensitivity = np.mean([r.sensitivity_score for r in results])
        robust_count = sum(1 for r in results if r.is_robust)

        return {
            "robustness_score": f"{robustness_score:.1f}/100",
            "rating": rating,
            "total_parameters": len(results),
            "robust_parameters": robust_count,
            "fragile_parameters": len(fragile),
            "fragile_list": fragile,
            "average_sensitivity": f"{avg_sensitivity:.2f}",
            "optimal_values": optimal,
            "recommendations": _generate_recommendations(results, fragile),
        }


def _generate_recommendations(
    results: List[SensitivityResult],
    fragile: List[str]
) -> List[str]:
    """Genera recomendaciones basadas en el analisis"""
    recommendations = []

    if len(fragile) > len(results) / 2:
        recommendations.append(
            "ALERTA: Mas del 50% de parametros son fragiles. "
            "La estrategia depende mucho de la configuracion exacta."
        )

    for r in results:
        if r.sensitivity_score > 0.7:
            recommendations.append(
                f"Parametro '{r.parameter_name}' es MUY sensible. "
                f"Pequenos cambios afectan significativamente los resultados."
            )
        elif not r.is_robust:
            recommendations.append(
                f"Parametro '{r.parameter_name}' tiene rango de trabajo limitado "
                f"({r.working_range[0]:.2f} - {r.working_range[1]:.2f})."
            )

        # Sugerir usar valor optimo si difiere mucho del base
        if r.base_value != 0:
            diff_pct = abs(r.optimal_value - r.base_value) / abs(r.base_value)
            if diff_pct > 0.20:
                recommendations.append(
                    f"Considerar cambiar '{r.parameter_name}' de {r.base_value:.2f} "
                    f"a {r.optimal_value:.2f} (+{diff_pct:.0%} mejora potencial)."
                )

    if not recommendations:
        recommendations.append(
            "La estrategia parece robusta. Los parametros actuales son razonables."
        )

    return recommendations
