"""Tests para el modulo de analisis de sensibilidad"""

import pytest
import numpy as np
from typing import Dict, Any

from src.stress_testing.sensitivity import (
    SensitivityAnalyzer,
    RobustnessAnalyzer,
    ParameterRange,
)
from src.stress_testing.models import (
    SensitivityConfig,
    SensitivityResult,
)


class TestParameterRange:
    """Tests para ParameterRange"""

    def test_creation(self):
        """Creacion basica"""
        pr = ParameterRange(
            name="sma_period",
            base_value=20.0,
            min_value=10.0,
            max_value=30.0,
            n_steps=5,
        )
        assert pr.name == "sma_period"
        assert pr.base_value == 20.0

    def test_get_values_with_n_steps(self):
        """Obtener valores con n_steps"""
        pr = ParameterRange(
            name="param",
            base_value=20.0,
            min_value=10.0,
            max_value=30.0,
            n_steps=5,
        )
        values = pr.get_values()
        assert len(values) == 5
        assert values[0] == 10.0
        assert values[-1] == 30.0

    def test_get_values_with_step(self):
        """Obtener valores con step fijo"""
        pr = ParameterRange(
            name="param",
            base_value=20.0,
            min_value=10.0,
            max_value=30.0,
            step=5.0,
        )
        values = pr.get_values()
        assert 10.0 in values
        assert 15.0 in values
        assert 20.0 in values
        assert 25.0 in values
        assert 30.0 in values

    def test_get_values_includes_extremes(self):
        """Valores incluyen min y max"""
        pr = ParameterRange(
            name="param",
            base_value=50.0,
            min_value=0.0,
            max_value=100.0,
            n_steps=11,
        )
        values = pr.get_values()
        assert min(values) == 0.0
        assert max(values) == 100.0


class TestSensitivityAnalyzer:
    """Tests para SensitivityAnalyzer"""

    def test_initialization_default(self):
        """Inicializacion con valores por defecto"""
        analyzer = SensitivityAnalyzer()
        assert analyzer.config.variation_pct == 0.20
        assert analyzer.config.n_steps == 5

    def test_initialization_custom(self, sensitivity_config):
        """Inicializacion con config personalizada"""
        analyzer = SensitivityAnalyzer(sensitivity_config)
        assert analyzer.config.variation_pct == 0.20
        assert analyzer.config.n_steps == 5

    def test_analyze_parameter_returns_result(self, mock_strategy_func):
        """analyze_parameter retorna SensitivityResult"""
        analyzer = SensitivityAnalyzer()
        result = analyzer.analyze_parameter(
            strategy_func=mock_strategy_func,
            param_name="sma_period",
            base_value=20.0,
            variation_pct=0.20,
        )
        assert isinstance(result, SensitivityResult)

    def test_analyze_parameter_name(self, mock_strategy_func):
        """Verifica nombre del parametro en resultado"""
        analyzer = SensitivityAnalyzer()
        result = analyzer.analyze_parameter(
            strategy_func=mock_strategy_func,
            param_name="my_param",
            base_value=50.0,
        )
        assert result.parameter_name == "my_param"

    def test_analyze_parameter_base_value(self, mock_strategy_func):
        """Verifica valor base en resultado"""
        analyzer = SensitivityAnalyzer()
        result = analyzer.analyze_parameter(
            strategy_func=mock_strategy_func,
            param_name="sma_period",
            base_value=25.0,
        )
        assert result.base_value == 25.0

    def test_analyze_parameter_tested_values_count(self, mock_strategy_func):
        """Verifica cantidad de valores testeados"""
        config = SensitivityConfig(n_steps=7)
        analyzer = SensitivityAnalyzer(config)
        result = analyzer.analyze_parameter(
            strategy_func=mock_strategy_func,
            param_name="sma_period",
            base_value=20.0,
        )
        assert len(result.tested_values) == 7

    def test_analyze_parameter_with_specific_values(self, mock_strategy_func):
        """Analisis con valores especificos"""
        analyzer = SensitivityAnalyzer()
        specific_values = [15.0, 20.0, 25.0, 30.0]
        result = analyzer.analyze_parameter(
            strategy_func=mock_strategy_func,
            param_name="sma_period",
            base_value=20.0,
            values=specific_values,
        )
        assert result.tested_values == specific_values

    def test_analyze_parameter_metric_values(self, mock_strategy_func):
        """Verifica que se calculan valores de metrica"""
        analyzer = SensitivityAnalyzer()
        result = analyzer.analyze_parameter(
            strategy_func=mock_strategy_func,
            param_name="sma_period",
            base_value=20.0,
        )
        assert len(result.metric_values) == len(result.tested_values)

    def test_analyze_parameter_finds_optimal(self, mock_strategy_func):
        """Encuentra valor optimo"""
        analyzer = SensitivityAnalyzer()
        result = analyzer.analyze_parameter(
            strategy_func=mock_strategy_func,
            param_name="sma_period",
            base_value=20.0,
            variation_pct=0.30,
        )
        # El mock tiene optimo en 25
        assert 20 <= result.optimal_value <= 26

    def test_analyze_parameter_sensitivity_score(self, mock_strategy_func):
        """Calcula score de sensibilidad"""
        analyzer = SensitivityAnalyzer()
        result = analyzer.analyze_parameter(
            strategy_func=mock_strategy_func,
            param_name="sma_period",
            base_value=20.0,
        )
        assert 0.0 <= result.sensitivity_score <= 1.0

    def test_analyze_parameter_is_robust(self, mock_strategy_func):
        """Verifica flag de robustez"""
        analyzer = SensitivityAnalyzer()
        result = analyzer.analyze_parameter(
            strategy_func=mock_strategy_func,
            param_name="sma_period",
            base_value=20.0,
        )
        assert isinstance(result.is_robust, bool)

    def test_analyze_multiple_parameters(
        self, mock_strategy_func, base_parameters
    ):
        """Analiza multiples parametros"""
        analyzer = SensitivityAnalyzer()
        param_ranges = {
            name: ParameterRange(
                name=name,
                base_value=value,
                min_value=value * 0.8,
                max_value=value * 1.2,
                n_steps=5,
            )
            for name, value in base_parameters.items()
        }

        results = analyzer.analyze_multiple_parameters(
            strategy_func=mock_strategy_func,
            parameters=param_ranges,
        )

        assert len(results) == len(base_parameters)
        assert all(isinstance(r, SensitivityResult) for r in results)

    def test_analyze_grid_2d(self, mock_strategy_func):
        """Analisis de grid 2D"""
        analyzer = SensitivityAnalyzer()
        param1 = ParameterRange("sma_period", 20.0, 15.0, 25.0, n_steps=3)
        param2 = ParameterRange("rsi_threshold", 30.0, 25.0, 35.0, n_steps=3)

        grid_result = analyzer.analyze_grid(
            strategy_func=mock_strategy_func,
            param1=param1,
            param2=param2,
        )

        assert "param1" in grid_result
        assert "param2" in grid_result
        assert "results_matrix" in grid_result
        assert "optimal" in grid_result

    def test_analyze_grid_matrix_shape(self, mock_strategy_func):
        """Verifica forma de matriz de grid"""
        analyzer = SensitivityAnalyzer()
        param1 = ParameterRange("p1", 20.0, 15.0, 25.0, n_steps=4)
        param2 = ParameterRange("p2", 30.0, 25.0, 35.0, n_steps=5)

        grid_result = analyzer.analyze_grid(
            strategy_func=mock_strategy_func,
            param1=param1,
            param2=param2,
        )

        matrix = grid_result["results_matrix"]
        assert len(matrix) == 4  # rows = param1 steps
        assert len(matrix[0]) == 5  # cols = param2 steps


class TestRobustnessAnalyzer:
    """Tests para RobustnessAnalyzer"""

    def test_calculate_robustness_score_range(self, mock_strategy_func):
        """Score de robustez en rango 0-100"""
        analyzer = SensitivityAnalyzer()
        result = analyzer.analyze_parameter(
            strategy_func=mock_strategy_func,
            param_name="sma_period",
            base_value=20.0,
        )

        score = RobustnessAnalyzer.calculate_robustness_score([result])
        assert 0.0 <= score <= 100.0

    def test_calculate_robustness_score_empty(self):
        """Score con lista vacia"""
        score = RobustnessAnalyzer.calculate_robustness_score([])
        assert score == 0.0

    def test_identify_fragile_parameters_robust(self, robust_strategy_func):
        """Identificar parametros fragiles - estrategia robusta"""
        analyzer = SensitivityAnalyzer()
        result = analyzer.analyze_parameter(
            strategy_func=robust_strategy_func,
            param_name="sma_period",
            base_value=20.0,
        )

        fragile = RobustnessAnalyzer.identify_fragile_parameters([result])
        # Estrategia robusta no deberia tener parametros fragiles
        assert len(fragile) == 0

    def test_identify_fragile_parameters_fragile(self, fragile_strategy_func):
        """Identificar parametros fragiles - estrategia fragil"""
        analyzer = SensitivityAnalyzer()
        result = analyzer.analyze_parameter(
            strategy_func=fragile_strategy_func,
            param_name="sma_period",
            base_value=20.0,
            variation_pct=0.50,  # ±50% para capturar fragilidad
        )

        fragile = RobustnessAnalyzer.identify_fragile_parameters([result])
        # Estrategia fragil deberia identificar el parametro
        assert "sma_period" in fragile

    def test_get_optimal_parameters(self, mock_strategy_func, base_parameters):
        """Obtener parametros optimos"""
        analyzer = SensitivityAnalyzer()
        param_ranges = {
            name: ParameterRange(
                name=name,
                base_value=value,
                min_value=value * 0.8,
                max_value=value * 1.2,
                n_steps=5,
            )
            for name, value in base_parameters.items()
        }

        results = analyzer.analyze_multiple_parameters(
            strategy_func=mock_strategy_func,
            parameters=param_ranges,
        )

        optimal = RobustnessAnalyzer.get_optimal_parameters(results)
        assert "sma_period" in optimal
        assert "rsi_threshold" in optimal

    def test_generate_report(self, mock_strategy_func):
        """Generar reporte de robustez"""
        analyzer = SensitivityAnalyzer()
        result = analyzer.analyze_parameter(
            strategy_func=mock_strategy_func,
            param_name="sma_period",
            base_value=20.0,
        )

        report = RobustnessAnalyzer.generate_report([result])
        assert "robustness_score" in report
        assert "rating" in report
        assert "total_parameters" in report
        assert "fragile_parameters" in report
        assert "recommendations" in report

    def test_report_rating_values(self, mock_strategy_func):
        """Ratings validos en reporte"""
        analyzer = SensitivityAnalyzer()
        result = analyzer.analyze_parameter(
            strategy_func=mock_strategy_func,
            param_name="sma_period",
            base_value=20.0,
        )

        report = RobustnessAnalyzer.generate_report([result])
        valid_ratings = ["ROBUST", "MODERATE", "FRAGILE", "VERY_FRAGILE"]
        assert report["rating"] in valid_ratings


class TestSensitivityEdgeCases:
    """Tests de casos limite"""

    def test_strategy_with_exception(self):
        """Estrategia que lanza excepcion"""
        def failing_strategy(params: Dict[str, Any]) -> Dict[str, float]:
            if params.get("sma_period", 0) > 25:
                raise ValueError("Parametro fuera de rango")
            return {"sharpe_ratio": 1.0}

        analyzer = SensitivityAnalyzer()
        result = analyzer.analyze_parameter(
            strategy_func=failing_strategy,
            param_name="sma_period",
            base_value=20.0,
            variation_pct=0.50,  # Algunos valores causaran excepcion
        )

        # Deberia manejar las excepciones y tener NaN en algunos valores
        assert isinstance(result, SensitivityResult)

    def test_constant_metric_values(self):
        """Metricas constantes (sensibilidad 0)"""
        def constant_strategy(params: Dict[str, Any]) -> Dict[str, float]:
            return {"sharpe_ratio": 1.5}

        analyzer = SensitivityAnalyzer()
        result = analyzer.analyze_parameter(
            strategy_func=constant_strategy,
            param_name="any_param",
            base_value=50.0,
        )

        # Con metricas constantes, sensibilidad deberia ser 0
        assert result.sensitivity_score == 0.0
        assert result.is_robust is True

    def test_all_nan_metrics(self):
        """Todos los valores son NaN"""
        def nan_strategy(params: Dict[str, Any]) -> Dict[str, float]:
            raise ValueError("Always fails")

        analyzer = SensitivityAnalyzer()
        result = analyzer.analyze_parameter(
            strategy_func=nan_strategy,
            param_name="param",
            base_value=10.0,
        )

        # Deberia manejar todos NaN
        assert result.is_robust is False

    def test_single_step(self):
        """Un solo paso (min = max)"""
        def simple_strategy(params: Dict[str, Any]) -> Dict[str, float]:
            return {"sharpe_ratio": 1.0}

        config = SensitivityConfig(n_steps=1)
        analyzer = SensitivityAnalyzer(config)

        # Esto deberia funcionar sin error
        result = analyzer.analyze_parameter(
            strategy_func=simple_strategy,
            param_name="param",
            base_value=10.0,
            variation_pct=0.0,  # Sin variacion
        )
        assert isinstance(result, SensitivityResult)

    def test_negative_base_value(self):
        """Valor base negativo"""
        def simple_strategy(params: Dict[str, Any]) -> Dict[str, float]:
            val = params.get("param", 0)
            return {"sharpe_ratio": 1.0 + val * 0.01}

        analyzer = SensitivityAnalyzer()
        result = analyzer.analyze_parameter(
            strategy_func=simple_strategy,
            param_name="param",
            base_value=-50.0,
            variation_pct=0.20,
        )

        assert result.base_value == -50.0
        # Deberia generar valores negativos
        assert any(v < 0 for v in result.tested_values)

    def test_zero_base_value(self):
        """Valor base cero"""
        def simple_strategy(params: Dict[str, Any]) -> Dict[str, float]:
            return {"sharpe_ratio": 1.0}

        analyzer = SensitivityAnalyzer()
        result = analyzer.analyze_parameter(
            strategy_func=simple_strategy,
            param_name="param",
            base_value=0.0,
            values=[-1.0, -0.5, 0.0, 0.5, 1.0],
        )

        assert result.base_value == 0.0

    def test_very_small_variation(self):
        """Variacion muy pequena"""
        def sensitive_strategy(params: Dict[str, Any]) -> Dict[str, float]:
            val = params.get("param", 20)
            # Muy sensible a cambios pequenos
            return {"sharpe_ratio": 1.0 / (abs(val - 20) + 0.001)}

        config = SensitivityConfig(variation_pct=0.01, n_steps=5)  # Solo ±1%
        analyzer = SensitivityAnalyzer(config)

        result = analyzer.analyze_parameter(
            strategy_func=sensitive_strategy,
            param_name="param",
            base_value=20.0,
        )

        # Todos los valores deberian estar muy cerca de 20
        assert all(19.5 <= v <= 20.5 for v in result.tested_values)
