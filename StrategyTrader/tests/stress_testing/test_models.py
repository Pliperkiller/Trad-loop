"""Tests para los modelos del modulo de stress testing"""

import pytest
from dataclasses import asdict

from src.stress_testing.models import (
    StressTestType,
    ScenarioType,
    MonteCarloConfig,
    ScenarioConfig,
    SensitivityConfig,
    StressTestConfig,
    MonteCarloResult,
    ScenarioResult,
    SensitivityResult,
    StressTestReport,
    PREDEFINED_SCENARIOS,
)


class TestEnums:
    """Tests para enums"""

    def test_stress_test_type_values(self):
        """Verifica valores de StressTestType"""
        assert StressTestType.MONTE_CARLO.value == "monte_carlo"
        assert StressTestType.SCENARIO.value == "scenario"
        assert StressTestType.SENSITIVITY.value == "sensitivity"

    def test_scenario_type_values(self):
        """Verifica valores de ScenarioType"""
        assert ScenarioType.COVID_CRASH.value == "covid_crash"
        assert ScenarioType.LUNA_CRASH.value == "luna_crash"
        assert ScenarioType.FTX_COLLAPSE.value == "ftx_collapse"
        assert ScenarioType.FLASH_CRASH.value == "flash_crash"
        assert ScenarioType.BEAR_MARKET.value == "bear_market"
        assert ScenarioType.HIGH_VOLATILITY.value == "high_volatility"
        assert ScenarioType.LOW_VOLATILITY.value == "low_volatility"
        assert ScenarioType.CUSTOM.value == "custom"

    def test_all_scenario_types_have_predefined_params(self):
        """Todos los escenarios excepto CUSTOM tienen parametros predefinidos"""
        for scenario in ScenarioType:
            if scenario != ScenarioType.CUSTOM:
                assert scenario in PREDEFINED_SCENARIOS


class TestMonteCarloConfig:
    """Tests para MonteCarloConfig"""

    def test_default_values(self):
        """Verifica valores por defecto"""
        config = MonteCarloConfig()
        assert config.n_simulations == 1000
        assert config.confidence_levels == [0.95, 0.99]
        assert config.shuffle_returns is True
        assert config.block_size is None

    def test_custom_values(self, monte_carlo_config):
        """Verifica valores personalizados"""
        assert monte_carlo_config.n_simulations == 100
        assert monte_carlo_config.confidence_levels == [0.90, 0.95]
        assert monte_carlo_config.block_size == 5

    def test_validation_min_simulations(self):
        """Validacion de simulaciones minimas"""
        with pytest.raises(ValueError):
            MonteCarloConfig(n_simulations=50)  # Menos de 100


class TestScenarioConfig:
    """Tests para ScenarioConfig"""

    def test_default_values(self):
        """Verifica valores por defecto"""
        config = ScenarioConfig()
        assert config.scenario_type == ScenarioType.COVID_CRASH
        assert config.custom_drawdown == 0.0
        assert config.custom_duration_days == 0
        assert config.custom_recovery_days == 0
        assert config.custom_volatility_mult == 1.0

    def test_custom_scenario_params(self, scenario_config):
        """Verifica parametros de escenario personalizado"""
        assert scenario_config.custom_drawdown == 0.30
        assert scenario_config.custom_duration_days == 20


class TestSensitivityConfig:
    """Tests para SensitivityConfig"""

    def test_default_values(self):
        """Verifica valores por defecto"""
        config = SensitivityConfig()
        assert config.variation_pct == 0.20
        assert config.n_steps == 5
        assert config.metric_to_optimize == "sharpe_ratio"

    def test_custom_values(self, sensitivity_config):
        """Verifica valores personalizados"""
        assert sensitivity_config.variation_pct == 0.20
        assert sensitivity_config.n_steps == 5


class TestStressTestConfig:
    """Tests para StressTestConfig"""

    def test_default_values(self):
        """Verifica valores por defecto"""
        config = StressTestConfig()
        assert config.initial_capital == 10000.0
        assert config.risk_free_rate == 0.02
        assert isinstance(config.monte_carlo, MonteCarloConfig)
        assert isinstance(config.sensitivity, SensitivityConfig)

    def test_nested_configs(self, stress_test_config):
        """Verifica configs anidadas"""
        assert stress_test_config.monte_carlo.n_simulations == 100
        assert stress_test_config.sensitivity.variation_pct == 0.20

    def test_to_dict(self):
        """Verifica conversion a diccionario"""
        config = StressTestConfig()
        d = config.to_dict()
        assert d["initial_capital"] == 10000.0
        assert "monte_carlo" in d
        assert "scenario" in d


class TestMonteCarloResult:
    """Tests para MonteCarloResult"""

    def test_creation(self):
        """Verifica creacion de resultado"""
        result = MonteCarloResult(
            n_simulations=1000,
            mean_return=15.5,
            std_return=12.3,
            percentile_5=-10.0,
            percentile_25=5.0,
            percentile_75=25.0,
            percentile_95=35.0,
            prob_profit=0.72,
            prob_loss_20pct=0.05,
            mean_max_drawdown=18.5,
            mean_sharpe=0.85,
            var_95=12.0,
            cvar_95=15.0,
        )
        assert result.n_simulations == 1000
        assert result.prob_profit == 0.72
        assert result.mean_sharpe == 0.85

    def test_default_values(self):
        """Verifica valores por defecto"""
        result = MonteCarloResult()
        assert result.n_simulations == 0
        assert result.mean_return == 0.0
        assert result.all_final_returns == []

    def test_to_dict(self):
        """Verifica conversion a dict"""
        result = MonteCarloResult(n_simulations=100, mean_return=10.0)
        d = result.to_dict()
        assert d["n_simulations"] == 100
        assert d["returns"]["mean"] == 10.0


class TestScenarioResult:
    """Tests para ScenarioResult"""

    def test_creation(self):
        """Verifica creacion de resultado"""
        result = ScenarioResult(
            scenario_type=ScenarioType.COVID_CRASH,
            scenario_name="COVID-19 Crash",
            initial_equity=10000.0,
            final_equity=7500.0,
            total_return=-2500.0,
            total_return_pct=-25.0,
            max_drawdown=3000.0,
            max_drawdown_pct=30.0,
            drawdown_duration_days=45,
            recovery_days=60,
            volatility=0.45,
            volatility_vs_normal=2.5,
            worst_day_return=-8.5,
            worst_week_return=-15.0,
            sharpe_during_stress=-1.2,
            survived=True,
            margin_call=False,
        )
        assert result.scenario_type == ScenarioType.COVID_CRASH
        assert result.survived is True
        assert result.max_drawdown_pct == 30.0

    def test_default_values(self):
        """Verifica valores por defecto"""
        result = ScenarioResult()
        assert result.scenario_type == ScenarioType.CUSTOM
        assert result.survived is True
        assert result.margin_call is False

    def test_to_dict(self):
        """Verifica conversion a dict"""
        result = ScenarioResult(
            scenario_type=ScenarioType.FLASH_CRASH,
            scenario_name="Flash Crash"
        )
        d = result.to_dict()
        assert d["scenario_type"] == "flash_crash"
        assert d["scenario_name"] == "Flash Crash"


class TestSensitivityResult:
    """Tests para SensitivityResult"""

    def test_creation(self):
        """Verifica creacion de resultado"""
        result = SensitivityResult(
            parameter_name="sma_period",
            base_value=20.0,
            tested_values=[16.0, 18.0, 20.0, 22.0, 24.0],
            metric_values=[0.8, 1.0, 1.2, 1.1, 0.9],
            optimal_value=20.0,
            optimal_metric=1.2,
            is_robust=True,
            working_range=(16.0, 24.0),
            sensitivity_score=0.15,
            metric_mean=1.0,
            metric_std=0.15,
            metric_min=0.8,
            metric_max=1.2,
        )
        assert result.parameter_name == "sma_period"
        assert result.is_robust is True
        assert result.sensitivity_score == 0.15

    def test_fragile_parameter(self):
        """Verifica parametro fragil"""
        result = SensitivityResult(
            parameter_name="threshold",
            base_value=0.5,
            tested_values=[0.4, 0.45, 0.5, 0.55, 0.6],
            metric_values=[0.2, 1.5, 0.3, -0.5, -1.0],
            optimal_value=0.45,
            optimal_metric=1.5,
            is_robust=False,  # Solo 1-2 valores funcionan
            working_range=(0.45, 0.45),
            sensitivity_score=0.85,  # Alta sensibilidad
        )
        assert result.is_robust is False
        assert result.sensitivity_score > 0.5

    def test_to_dict(self):
        """Verifica conversion a dict"""
        result = SensitivityResult(
            parameter_name="test_param",
            base_value=10.0
        )
        d = result.to_dict()
        assert d["parameter"] == "test_param"
        assert d["base_value"] == 10.0


class TestStressTestReport:
    """Tests para StressTestReport"""

    def test_creation_minimal(self):
        """Verifica creacion minima"""
        report = StressTestReport()
        assert report.config is None
        assert report.monte_carlo is None
        assert report.scenarios == []
        assert report.sensitivity == []

    def test_creation_with_config(self):
        """Verifica creacion con config"""
        config = StressTestConfig()
        report = StressTestReport(config=config)
        assert report.config == config

    def test_creation_complete(self):
        """Verifica creacion completa"""
        config = StressTestConfig()
        report = StressTestReport(
            config=config,
            overall_robustness_score=75.5,
            risk_rating="MEDIUM",
            key_risks=["Alto drawdown en COVID crash"],
            recommendations=["Reducir exposicion"],
            execution_time=5.5,
        )
        assert report.overall_robustness_score == 75.5
        assert report.risk_rating == "MEDIUM"
        assert len(report.key_risks) == 1

    def test_to_dict(self):
        """Verifica conversion a dict"""
        report = StressTestReport(
            overall_robustness_score=80.0,
            risk_rating="LOW"
        )
        d = report.to_dict()
        assert d["summary"]["robustness_score"] == 80.0
        assert d["summary"]["risk_rating"] == "LOW"


class TestPredefinedScenarios:
    """Tests para escenarios predefinidos"""

    def test_covid_crash_params(self):
        """Verifica parametros de COVID crash"""
        params = PREDEFINED_SCENARIOS[ScenarioType.COVID_CRASH]
        assert "COVID" in params["name"]
        assert params["drawdown"] == 0.35
        assert params["duration_days"] == 33
        assert params["recovery_days"] == 140

    def test_luna_crash_params(self):
        """Verifica parametros de Luna crash"""
        params = PREDEFINED_SCENARIOS[ScenarioType.LUNA_CRASH]
        assert "Luna" in params["name"]
        assert params["drawdown"] == 0.60
        assert params["recovery_days"] == float("inf")

    def test_ftx_collapse_params(self):
        """Verifica parametros de FTX collapse"""
        params = PREDEFINED_SCENARIOS[ScenarioType.FTX_COLLAPSE]
        assert "FTX" in params["name"]
        assert params["drawdown"] == 0.25

    def test_flash_crash_params(self):
        """Verifica parametros de Flash crash"""
        params = PREDEFINED_SCENARIOS[ScenarioType.FLASH_CRASH]
        assert params["duration_days"] == 1
        assert params["recovery_days"] == 3

    def test_bear_market_params(self):
        """Verifica parametros de Bear market"""
        params = PREDEFINED_SCENARIOS[ScenarioType.BEAR_MARKET]
        assert params["duration_days"] == 365
        assert params["drawdown"] == 0.50

    def test_all_scenarios_have_required_fields(self):
        """Todos los escenarios tienen campos requeridos"""
        required_fields = ["name", "drawdown", "duration_days", "recovery_days"]
        for scenario_type, params in PREDEFINED_SCENARIOS.items():
            for field in required_fields:
                assert field in params, f"{scenario_type} missing {field}"
