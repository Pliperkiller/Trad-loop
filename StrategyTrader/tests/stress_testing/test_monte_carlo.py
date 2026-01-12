"""Tests para el modulo de Monte Carlo simulation"""

import pytest
import numpy as np

from src.stress_testing.monte_carlo import (
    MonteCarloSimulator,
    MonteCarloAnalyzer,
)
from src.stress_testing.models import MonteCarloConfig, MonteCarloResult


class TestMonteCarloSimulator:
    """Tests para MonteCarloSimulator"""

    def test_initialization_default(self):
        """Inicializacion con config por defecto"""
        simulator = MonteCarloSimulator()
        assert simulator.config.n_simulations == 1000
        assert simulator.config.shuffle_returns is True

    def test_initialization_custom(self, monte_carlo_config):
        """Inicializacion con config personalizada"""
        simulator = MonteCarloSimulator(monte_carlo_config)
        assert simulator.config.n_simulations == 100
        assert simulator.config.block_size == 5

    def test_run_returns_result(self, sample_returns, monte_carlo_config):
        """run() retorna MonteCarloResult"""
        simulator = MonteCarloSimulator(monte_carlo_config)
        result = simulator.run(sample_returns)
        assert isinstance(result, MonteCarloResult)

    def test_run_simulation_count(self, sample_returns, monte_carlo_config):
        """Verifica numero de simulaciones"""
        simulator = MonteCarloSimulator(monte_carlo_config)
        result = simulator.run(sample_returns)
        assert result.n_simulations == 100

    def test_run_shuffle_method(self, sample_returns):
        """Test metodo shuffle (sin block_size)"""
        config = MonteCarloConfig(n_simulations=100)
        simulator = MonteCarloSimulator(config)
        result = simulator.run(sample_returns)
        assert result.n_simulations == 100

    def test_run_block_bootstrap_method(self, sample_returns):
        """Test metodo block bootstrap"""
        config = MonteCarloConfig(n_simulations=100, block_size=5)
        simulator = MonteCarloSimulator(config)
        result = simulator.run(sample_returns)
        assert result.n_simulations == 100

    def test_percentiles_ordered(self, sample_returns, monte_carlo_config):
        """Percentiles estan ordenados correctamente"""
        simulator = MonteCarloSimulator(monte_carlo_config)
        result = simulator.run(sample_returns)
        assert result.percentile_5 <= result.percentile_25
        assert result.percentile_25 <= result.percentile_75
        assert result.percentile_75 <= result.percentile_95

    def test_prob_profit_range(self, sample_returns, monte_carlo_config):
        """Probabilidad de ganancia en rango [0, 1]"""
        simulator = MonteCarloSimulator(monte_carlo_config)
        result = simulator.run(sample_returns)
        assert 0.0 <= result.prob_profit <= 1.0

    def test_prob_loss_range(self, sample_returns, monte_carlo_config):
        """Probabilidad de perdida en rango [0, 1]"""
        simulator = MonteCarloSimulator(monte_carlo_config)
        result = simulator.run(sample_returns)
        assert 0.0 <= result.prob_loss_20pct <= 1.0

    def test_mean_max_drawdown_positive(self, sample_returns, monte_carlo_config):
        """Max drawdown promedio es positivo"""
        simulator = MonteCarloSimulator(monte_carlo_config)
        result = simulator.run(sample_returns)
        assert result.mean_max_drawdown >= 0

    def test_var_cvar_relationship(self, sample_returns, monte_carlo_config):
        """CVaR <= VaR (CVaR is expected loss below VaR)"""
        simulator = MonteCarloSimulator(monte_carlo_config)
        result = simulator.run(sample_returns)
        # CVaR (expected loss) should be more negative or equal to VaR
        assert result.cvar_95 <= result.var_95

    def test_trending_returns_higher_profit_prob(
        self, trending_returns, monte_carlo_config
    ):
        """Retornos con tendencia tienen mayor prob de ganancia"""
        simulator = MonteCarloSimulator(monte_carlo_config)
        result = simulator.run(trending_returns)
        # Con retornos positivos consistentes, prob ganancia deberia ser alta
        assert result.prob_profit > 0.5

    def test_losing_returns_lower_profit_prob(self, losing_returns, monte_carlo_config):
        """Retornos perdedores tienen menor prob de ganancia"""
        simulator = MonteCarloSimulator(monte_carlo_config)
        result = simulator.run(losing_returns)
        # Con retornos negativos consistentes, prob ganancia deberia ser baja
        assert result.prob_profit < 0.5

    def test_volatile_returns_higher_drawdown(self, volatile_returns, sample_returns):
        """Retornos volatiles tienen mayor drawdown promedio"""
        config = MonteCarloConfig(n_simulations=100)
        simulator = MonteCarloSimulator(config)

        result_volatile = simulator.run(volatile_returns)
        result_normal = simulator.run(sample_returns)

        # Mayor volatilidad base deberia resultar en mayor drawdown
        assert result_volatile.mean_max_drawdown >= result_normal.mean_max_drawdown * 0.8

    def test_with_initial_capital(self, sample_returns, monte_carlo_config):
        """Test con capital inicial especifico"""
        simulator = MonteCarloSimulator(monte_carlo_config)
        result = simulator.run(sample_returns, initial_capital=50000.0)
        assert isinstance(result, MonteCarloResult)

    def test_with_risk_free_rate(self, sample_returns, monte_carlo_config):
        """Test con tasa libre de riesgo"""
        simulator = MonteCarloSimulator(monte_carlo_config)
        result = simulator.run(sample_returns, risk_free_rate=0.05)
        assert isinstance(result, MonteCarloResult)

    def test_short_returns(self, short_returns):
        """Test con serie corta de retornos"""
        config = MonteCarloConfig(n_simulations=100)
        simulator = MonteCarloSimulator(config)
        result = simulator.run(short_returns)
        assert isinstance(result, MonteCarloResult)

    def test_raw_data_stored(self, sample_returns, monte_carlo_config):
        """Verifica que datos raw se almacenan"""
        simulator = MonteCarloSimulator(monte_carlo_config)
        result = simulator.run(sample_returns)

        assert len(result.all_final_returns) == monte_carlo_config.n_simulations
        assert len(result.all_max_drawdowns) == monte_carlo_config.n_simulations
        assert len(result.all_sharpe_ratios) == monte_carlo_config.n_simulations


class TestMonteCarloAnalyzer:
    """Tests para MonteCarloAnalyzer"""

    def test_get_confidence_interval_95(self, sample_returns, monte_carlo_config):
        """Calculo de intervalo de confianza 95%"""
        simulator = MonteCarloSimulator(monte_carlo_config)
        result = simulator.run(sample_returns)

        ci = MonteCarloAnalyzer.get_confidence_interval(result, 0.95)
        assert len(ci) == 2
        assert ci[0] < ci[1]  # lower < upper

    def test_get_confidence_interval_90(self, sample_returns, monte_carlo_config):
        """Calculo de intervalo de confianza 90%"""
        simulator = MonteCarloSimulator(monte_carlo_config)
        result = simulator.run(sample_returns)

        ci_90 = MonteCarloAnalyzer.get_confidence_interval(result, 0.90)
        ci_95 = MonteCarloAnalyzer.get_confidence_interval(result, 0.95)

        # IC 95% deberia ser mas amplio que IC 90%
        assert ci_95[0] <= ci_90[0]
        assert ci_95[1] >= ci_90[1]

    def test_get_risk_of_ruin_calculation(self, sample_returns, monte_carlo_config):
        """Calculo de riesgo de ruina"""
        simulator = MonteCarloSimulator(monte_carlo_config)
        result = simulator.run(sample_returns)

        ror = MonteCarloAnalyzer.get_risk_of_ruin(result, ruin_threshold=-50.0)
        assert 0.0 <= ror <= 1.0

    def test_get_risk_of_ruin_stricter_threshold(self, sample_returns, monte_carlo_config):
        """Threshold mas estricto = mayor riesgo de ruina"""
        simulator = MonteCarloSimulator(monte_carlo_config)
        result = simulator.run(sample_returns)

        ror_50 = MonteCarloAnalyzer.get_risk_of_ruin(result, ruin_threshold=-50.0)
        ror_30 = MonteCarloAnalyzer.get_risk_of_ruin(result, ruin_threshold=-30.0)

        # Perder 30% es mas probable que perder 50%
        assert ror_30 >= ror_50

    def test_get_expected_shortfall(self, sample_returns, monte_carlo_config):
        """Calculo de expected shortfall"""
        simulator = MonteCarloSimulator(monte_carlo_config)
        result = simulator.run(sample_returns)

        es = MonteCarloAnalyzer.get_expected_shortfall(result, percentile=5)
        assert isinstance(es, float)

    def test_get_probability_target(self, sample_returns, monte_carlo_config):
        """Calculo de probabilidad de objetivo"""
        simulator = MonteCarloSimulator(monte_carlo_config)
        result = simulator.run(sample_returns)

        prob = MonteCarloAnalyzer.get_probability_target(result, target_return=0.0)
        # Deberia ser igual a prob_profit
        assert abs(prob - result.prob_profit) < 0.01

    def test_generate_summary(self, sample_returns, monte_carlo_config):
        """Generacion de resumen"""
        simulator = MonteCarloSimulator(monte_carlo_config)
        result = simulator.run(sample_returns)

        summary = MonteCarloAnalyzer.generate_summary(result)
        assert "risk_level" in summary
        assert "robustness" in summary
        assert "expected_return" in summary
        assert "probability_of_profit" in summary
        assert "key_insight" in summary

    def test_summary_risk_levels(self, sample_returns, monte_carlo_config):
        """Niveles de riesgo validos"""
        simulator = MonteCarloSimulator(monte_carlo_config)
        result = simulator.run(sample_returns)

        summary = MonteCarloAnalyzer.generate_summary(result)
        valid_risk_levels = ["LOW", "MEDIUM", "HIGH", "EXTREME"]
        assert summary["risk_level"] in valid_risk_levels

    def test_summary_robustness_levels(self, sample_returns, monte_carlo_config):
        """Niveles de robustez validos"""
        simulator = MonteCarloSimulator(monte_carlo_config)
        result = simulator.run(sample_returns)

        summary = MonteCarloAnalyzer.generate_summary(result)
        valid_robustness = ["ROBUST", "MODERATE", "FRAGILE"]
        assert summary["robustness"] in valid_robustness


class TestMonteCarloEdgeCases:
    """Tests de casos limite"""

    def test_all_positive_returns(self):
        """Todos los retornos positivos"""
        returns = np.array([0.01] * 100)
        config = MonteCarloConfig(n_simulations=100)
        simulator = MonteCarloSimulator(config)
        result = simulator.run(returns)

        assert result.prob_profit == 1.0
        assert result.mean_return > 0

    def test_all_negative_returns(self):
        """Todos los retornos negativos"""
        returns = np.array([-0.01] * 100)
        config = MonteCarloConfig(n_simulations=100)
        simulator = MonteCarloSimulator(config)
        result = simulator.run(returns)

        assert result.prob_profit == 0.0
        assert result.mean_return < 0

    def test_alternating_returns(self):
        """Retornos alternantes"""
        returns = np.array([0.02, -0.02] * 50)
        config = MonteCarloConfig(n_simulations=100)
        simulator = MonteCarloSimulator(config)
        result = simulator.run(returns)

        # Retornos alternantes deberian dar resultado cercano a 0
        assert abs(result.mean_return) < 20

    def test_extreme_volatility(self):
        """Volatilidad extrema causa alto drawdown"""
        np.random.seed(42)
        returns = np.random.normal(0, 0.20, 100)  # 20% diario
        config = MonteCarloConfig(n_simulations=100)
        simulator = MonteCarloSimulator(config)
        result = simulator.run(returns)

        # Alta volatilidad causa perdidas significativas
        assert result.mean_max_drawdown > 50 or result.prob_loss_20pct > 0.3

    def test_reproducibility_with_seed(self):
        """Reproducibilidad con semilla"""
        returns = np.random.RandomState(42).normal(0.001, 0.02, 100)

        config1 = MonteCarloConfig(n_simulations=100, random_seed=123)
        simulator1 = MonteCarloSimulator(config1)
        result1 = simulator1.run(returns)

        config2 = MonteCarloConfig(n_simulations=100, random_seed=123)
        simulator2 = MonteCarloSimulator(config2)
        result2 = simulator2.run(returns)

        assert result1.mean_return == result2.mean_return

    def test_progress_callback(self, sample_returns):
        """Test progress callback"""
        config = MonteCarloConfig(n_simulations=100)
        simulator = MonteCarloSimulator(config)

        progress_calls = []

        def callback(current, total):
            progress_calls.append((current, total))

        simulator.run(sample_returns, progress_callback=callback)

        assert len(progress_calls) == 100
        assert progress_calls[-1] == (100, 100)

    def test_to_dict_method(self, sample_returns, monte_carlo_config):
        """Test conversion a dict"""
        simulator = MonteCarloSimulator(monte_carlo_config)
        result = simulator.run(sample_returns)

        d = result.to_dict()
        assert "n_simulations" in d
        assert "returns" in d
        assert "percentiles" in d
        assert "probabilities" in d
