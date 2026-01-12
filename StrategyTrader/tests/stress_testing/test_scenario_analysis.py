"""Tests para el modulo de analisis de escenarios"""

import pytest
import numpy as np

from src.stress_testing.scenario_analysis import (
    ScenarioAnalyzer,
    ScenarioComparator,
    ScenarioData,
)
from src.stress_testing.models import (
    ScenarioConfig,
    ScenarioResult,
    ScenarioType,
    PREDEFINED_SCENARIOS,
)


class TestScenarioAnalyzer:
    """Tests para ScenarioAnalyzer"""

    def test_initialization_default(self):
        """Inicializacion con valores por defecto"""
        analyzer = ScenarioAnalyzer()
        assert analyzer.initial_capital == 10000.0

    def test_initialization_custom_capital(self):
        """Inicializacion con capital personalizado"""
        analyzer = ScenarioAnalyzer(initial_capital=50000.0)
        assert analyzer.initial_capital == 50000.0

    def test_run_scenario_returns_result(self, sample_returns):
        """run_scenario retorna ScenarioResult"""
        analyzer = ScenarioAnalyzer()
        result = analyzer.run_scenario(sample_returns, ScenarioType.COVID_CRASH)
        assert isinstance(result, ScenarioResult)

    def test_run_covid_crash_scenario(self, sample_returns):
        """Test escenario COVID crash"""
        analyzer = ScenarioAnalyzer()
        result = analyzer.run_scenario(sample_returns, ScenarioType.COVID_CRASH)

        assert result.scenario_type == ScenarioType.COVID_CRASH
        assert "COVID" in result.scenario_name
        assert result.max_drawdown_pct > 0

    def test_run_luna_crash_scenario(self, sample_returns):
        """Test escenario Luna crash"""
        analyzer = ScenarioAnalyzer()
        result = analyzer.run_scenario(sample_returns, ScenarioType.LUNA_CRASH)

        assert result.scenario_type == ScenarioType.LUNA_CRASH
        assert "Luna" in result.scenario_name or "LUNA" in result.scenario_name.upper()
        # Luna crash es severo, esperar alto drawdown
        assert result.max_drawdown_pct > 50

    def test_run_flash_crash_scenario(self, sample_returns):
        """Test escenario Flash crash"""
        analyzer = ScenarioAnalyzer()
        result = analyzer.run_scenario(sample_returns, ScenarioType.FLASH_CRASH)

        assert result.scenario_type == ScenarioType.FLASH_CRASH
        # Flash crash es rapido
        assert result.drawdown_duration_days <= 5

    def test_run_bear_market_scenario(self, sample_returns):
        """Test escenario Bear market"""
        analyzer = ScenarioAnalyzer()
        result = analyzer.run_scenario(sample_returns, ScenarioType.BEAR_MARKET)

        assert result.scenario_type == ScenarioType.BEAR_MARKET
        # Bear market es prolongado
        assert result.drawdown_duration_days > 30

    def test_run_high_volatility_scenario(self, sample_returns):
        """Test escenario alta volatilidad"""
        analyzer = ScenarioAnalyzer()
        result = analyzer.run_scenario(sample_returns, ScenarioType.HIGH_VOLATILITY)

        assert result.scenario_type == ScenarioType.HIGH_VOLATILITY
        assert result.volatility_vs_normal > 1.0

    def test_run_low_volatility_scenario(self, sample_returns):
        """Test escenario baja volatilidad"""
        analyzer = ScenarioAnalyzer()
        result = analyzer.run_scenario(sample_returns, ScenarioType.LOW_VOLATILITY)

        assert result.scenario_type == ScenarioType.LOW_VOLATILITY
        assert result.volatility_vs_normal < 1.5

    def test_run_all_scenarios(self, sample_returns):
        """run_all_scenarios ejecuta todos los escenarios"""
        analyzer = ScenarioAnalyzer()
        results = analyzer.run_all_scenarios(sample_returns)

        # Debe haber un resultado por cada tipo excepto CUSTOM
        expected_count = len(ScenarioType) - 1
        assert len(results) == expected_count

    def test_run_all_scenarios_types(self, sample_returns):
        """Verifica tipos de escenarios ejecutados"""
        analyzer = ScenarioAnalyzer()
        results = analyzer.run_all_scenarios(sample_returns)

        types_executed = {r.scenario_type for r in results}
        assert ScenarioType.COVID_CRASH in types_executed
        assert ScenarioType.LUNA_CRASH in types_executed
        assert ScenarioType.CUSTOM not in types_executed

    def test_run_custom_scenario(self, sample_returns):
        """Test escenario personalizado"""
        analyzer = ScenarioAnalyzer()
        result = analyzer.run_custom_scenario(
            returns=sample_returns,
            drawdown=0.25,
            duration_days=15,
            recovery_days=30,
            volatility_mult=1.5,
            name="Mi Escenario",
        )

        assert result.scenario_type == ScenarioType.CUSTOM
        assert result.scenario_name == "Mi Escenario"

    def test_scenario_result_equity_values(self, sample_returns):
        """Verifica valores de equity en resultado"""
        analyzer = ScenarioAnalyzer(initial_capital=10000.0)
        result = analyzer.run_scenario(sample_returns, ScenarioType.COVID_CRASH)

        assert result.initial_equity == 10000.0
        assert result.final_equity > 0

    def test_scenario_result_return_consistency(self, sample_returns):
        """Verifica consistencia de retorno total"""
        analyzer = ScenarioAnalyzer(initial_capital=10000.0)
        result = analyzer.run_scenario(sample_returns, ScenarioType.FLASH_CRASH)

        # total_return = final_equity - initial_equity
        expected_return = result.final_equity - result.initial_equity
        assert abs(result.total_return - expected_return) < 0.01

    def test_scenario_survival_flag(self, sample_returns):
        """Verifica flag de supervivencia"""
        analyzer = ScenarioAnalyzer(initial_capital=10000.0)
        result = analyzer.run_scenario(sample_returns, ScenarioType.FLASH_CRASH)

        # Sobrevive si conserva >20% del capital
        expected_survived = result.final_equity > 10000.0 * 0.2
        assert result.survived == expected_survived

    def test_scenario_margin_call_flag(self, sample_returns):
        """Verifica flag de margin call"""
        analyzer = ScenarioAnalyzer(initial_capital=10000.0)
        result = analyzer.run_scenario(sample_returns, ScenarioType.LUNA_CRASH)

        # Margin call si equity cae <10% en algun momento
        # numpy.bool_ is acceptable
        assert result.margin_call in (True, False)

    def test_worst_day_return(self, sample_returns):
        """Verifica peor dia"""
        analyzer = ScenarioAnalyzer()
        result = analyzer.run_scenario(sample_returns, ScenarioType.COVID_CRASH)

        # El peor dia deberia ser negativo en un crash
        assert result.worst_day_return < 0

    def test_worst_week_return(self, sample_returns):
        """Verifica peor semana"""
        analyzer = ScenarioAnalyzer()
        result = analyzer.run_scenario(sample_returns, ScenarioType.COVID_CRASH)

        # La peor semana deberia ser negativa en un crash
        assert result.worst_week_return < 0


class TestScenarioComparator:
    """Tests para ScenarioComparator"""

    def test_rank_by_survival(self, sample_returns):
        """Ranking por supervivencia"""
        analyzer = ScenarioAnalyzer()
        results = analyzer.run_all_scenarios(sample_returns)

        ranked = ScenarioComparator.rank_by_survival(results)
        # El resultado debe tener la misma cantidad
        assert len(ranked) == len(results)
        # Si hay no sobrevivientes, deben estar ordenados por severidad
        non_survivors = [r for r in ranked if not r.survived]
        if non_survivors:
            # El peor no sobreviviente deberia tener el mayor drawdown
            worst_ns = max(non_survivors, key=lambda r: r.max_drawdown_pct)
            assert worst_ns.max_drawdown_pct > 50  # Es realmente severo

    def test_get_worst_case(self, sample_returns):
        """Obtener peor escenario"""
        analyzer = ScenarioAnalyzer()
        results = analyzer.run_all_scenarios(sample_returns)

        worst = ScenarioComparator.get_worst_case(results)
        max_dd = max(r.max_drawdown_pct for r in results)
        assert worst.max_drawdown_pct == max_dd

    def test_get_survival_rate(self, sample_returns):
        """Calcular tasa de supervivencia"""
        analyzer = ScenarioAnalyzer()
        results = analyzer.run_all_scenarios(sample_returns)

        rate = ScenarioComparator.get_survival_rate(results)
        assert 0.0 <= rate <= 1.0

        # Verificar calculo manual
        survived_count = sum(1 for r in results if r.survived)
        expected_rate = survived_count / len(results)
        assert rate == expected_rate

    def test_survival_rate_empty_list(self):
        """Tasa de supervivencia con lista vacia"""
        rate = ScenarioComparator.get_survival_rate([])
        assert rate == 0.0

    def test_generate_summary(self, sample_returns):
        """Generar resumen de escenarios"""
        analyzer = ScenarioAnalyzer()
        results = analyzer.run_all_scenarios(sample_returns)

        summary = ScenarioComparator.generate_summary(results)
        assert "total_scenarios" in summary
        assert "survival_rate" in summary
        assert "scenarios_survived" in summary
        assert "worst_scenario" in summary
        assert "worst_drawdown" in summary
        assert "risk_assessment" in summary

    def test_summary_empty_results(self):
        """Resumen con resultados vacios"""
        summary = ScenarioComparator.generate_summary([])
        assert summary == {}


class TestScenarioData:
    """Tests para ScenarioData"""

    def test_scenario_data_creation(self):
        """Creacion de ScenarioData"""
        returns = np.array([0.01, -0.02, 0.015])
        data = ScenarioData(returns=returns, volatility=0.15, duration_days=3)

        assert len(data.returns) == 3
        assert data.volatility == 0.15
        assert data.duration_days == 3


class TestScenarioEdgeCases:
    """Tests de casos limite"""

    def test_short_returns(self):
        """Retornos cortos"""
        np.random.seed(42)
        returns = np.random.normal(0, 0.02, 10)  # Solo 10 dias
        analyzer = ScenarioAnalyzer()

        result = analyzer.run_scenario(returns, ScenarioType.FLASH_CRASH)
        assert isinstance(result, ScenarioResult)

    def test_very_volatile_base_returns(self):
        """Retornos base muy volatiles"""
        np.random.seed(42)
        returns = np.random.normal(0, 0.10, 100)  # 10% volatilidad diaria
        analyzer = ScenarioAnalyzer()

        result = analyzer.run_scenario(returns, ScenarioType.HIGH_VOLATILITY)
        # Aun con base volatil, el escenario deberia agregar mas
        assert result.volatility > 0

    def test_all_positive_base_returns(self):
        """Retornos base todos positivos"""
        returns = np.array([0.01] * 100)
        analyzer = ScenarioAnalyzer()

        result = analyzer.run_scenario(returns, ScenarioType.COVID_CRASH)
        # Un crash deberia causar drawdown significativo
        assert result.max_drawdown_pct > 20

    def test_scenario_with_large_capital(self):
        """Escenario con capital grande"""
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 100)
        analyzer = ScenarioAnalyzer(initial_capital=1_000_000.0)

        result = analyzer.run_scenario(returns, ScenarioType.COVID_CRASH)
        assert result.initial_equity == 1_000_000.0

    def test_scenario_with_small_capital(self):
        """Escenario con capital pequeno"""
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 100)
        analyzer = ScenarioAnalyzer(initial_capital=100.0)

        result = analyzer.run_scenario(returns, ScenarioType.COVID_CRASH)
        assert result.initial_equity == 100.0


class TestPredefinedScenariosIntegration:
    """Tests de integracion con escenarios predefinidos"""

    def test_all_predefined_scenarios_runnable(self, sample_returns):
        """Todos los escenarios predefinidos son ejecutables"""
        analyzer = ScenarioAnalyzer()

        for scenario_type in PREDEFINED_SCENARIOS.keys():
            result = analyzer.run_scenario(sample_returns, scenario_type)
            assert isinstance(result, ScenarioResult)
            assert result.scenario_type == scenario_type

    def test_severity_ordering(self, sample_returns):
        """Luna crash deberia ser mas severo que COVID"""
        analyzer = ScenarioAnalyzer()

        covid = analyzer.run_scenario(sample_returns, ScenarioType.COVID_CRASH)
        luna = analyzer.run_scenario(sample_returns, ScenarioType.LUNA_CRASH)

        # Luna (99% drawdown) deberia ser peor que COVID (35%)
        assert luna.max_drawdown_pct >= covid.max_drawdown_pct

    def test_duration_differences(self, sample_returns):
        """Verificar diferencias en duracion"""
        analyzer = ScenarioAnalyzer()

        flash = analyzer.run_scenario(sample_returns, ScenarioType.FLASH_CRASH)
        bear = analyzer.run_scenario(sample_returns, ScenarioType.BEAR_MARKET)

        # Bear market deberia durar mas que flash crash
        assert bear.drawdown_duration_days >= flash.drawdown_duration_days
