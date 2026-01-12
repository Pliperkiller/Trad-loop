"""
Tests para portfolio/metrics.py
"""

import pytest
import sys
from pathlib import Path
import numpy as np
import pandas as pd

src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from portfolio.metrics import (
    PortfolioMetricsCalculator,
    DiversificationMetrics,
    CorrelationMetrics,
    ContributionMetrics,
)


class TestPortfolioMetricsCalculator:
    """Tests para PortfolioMetricsCalculator"""

    @pytest.fixture
    def calculator(self, sample_returns, equal_weights):
        """Calculator con datos de ejemplo"""
        return PortfolioMetricsCalculator(
            returns=sample_returns,
            weights=equal_weights,
            risk_free_rate=0.02,
        )

    @pytest.fixture
    def calculator_with_benchmark(self, sample_returns, equal_weights, benchmark_returns):
        """Calculator con benchmark"""
        return PortfolioMetricsCalculator(
            returns=sample_returns,
            weights=equal_weights,
            risk_free_rate=0.02,
            benchmark_returns=benchmark_returns,
        )

    def test_create_calculator(self, calculator):
        """Test crear calculator"""
        assert calculator is not None
        assert calculator.risk_free_rate == 0.02

    def test_calculate_diversification(self, calculator):
        """Test metricas de diversificacion"""
        metrics = calculator.calculate_diversification()

        assert isinstance(metrics, DiversificationMetrics)
        assert metrics.diversification_ratio >= 0
        assert metrics.concentration_hhi > 0
        assert metrics.effective_n > 0

    def test_concentration_hhi(self, sample_returns):
        """Test HHI para diferentes pesos"""
        # Equal weight: HHI deberia ser bajo (1/n)
        equal = {s: 0.25 for s in sample_returns.columns}
        calc_equal = PortfolioMetricsCalculator(sample_returns, equal)
        div_equal = calc_equal.calculate_diversification()

        # Concentrated: HHI deberia ser alto
        concentrated = {
            "BTC/USDT": 0.70,
            "ETH/USDT": 0.20,
            "SOL/USDT": 0.07,
            "LINK/USDT": 0.03,
        }
        calc_conc = PortfolioMetricsCalculator(sample_returns, concentrated)
        div_conc = calc_conc.calculate_diversification()

        # Equal weight deberia tener menor HHI
        assert div_equal.concentration_hhi < div_conc.concentration_hhi

    def test_calculate_correlation_metrics(self, calculator):
        """Test metricas de correlacion"""
        metrics = calculator.calculate_correlation_metrics()

        assert isinstance(metrics, CorrelationMetrics)
        assert -1 <= metrics.avg_correlation <= 1
        assert -1 <= metrics.max_correlation <= 1
        assert -1 <= metrics.min_correlation <= 1

    def test_calculate_contributions(self, calculator):
        """Test contribuciones"""
        metrics = calculator.calculate_contributions()

        assert isinstance(metrics, ContributionMetrics)
        assert len(metrics.return_contribution) > 0
        assert len(metrics.risk_contribution) > 0

    def test_calculate_portfolio_return(self, calculator):
        """Test retorno del portfolio"""
        ret = calculator.calculate_portfolio_return()

        assert isinstance(ret, float)

    def test_calculate_portfolio_volatility(self, calculator):
        """Test volatilidad del portfolio"""
        vol = calculator.calculate_portfolio_volatility()

        assert isinstance(vol, float)
        assert vol >= 0

    def test_calculate_sharpe_ratio(self, calculator):
        """Test Sharpe ratio"""
        sharpe = calculator.calculate_sharpe_ratio()

        assert isinstance(sharpe, float)

    def test_calculate_var(self, calculator):
        """Test VaR"""
        var_95 = calculator.calculate_var(0.95)

        assert isinstance(var_95, float)
        assert var_95 <= 0  # VaR es tipicamente negativo

    def test_calculate_cvar(self, calculator):
        """Test CVaR"""
        var_95 = calculator.calculate_var(0.95)
        cvar_95 = calculator.calculate_cvar(0.95)

        assert isinstance(cvar_95, float)
        # CVaR deberia ser peor (mas negativo) que VaR
        assert cvar_95 <= var_95

    def test_tracking_error_no_benchmark(self, calculator):
        """Test tracking error sin benchmark"""
        te = calculator.calculate_tracking_error()

        assert te == 0.0  # Sin benchmark, deberia ser 0

    def test_tracking_error_with_benchmark(self, calculator_with_benchmark):
        """Test tracking error con benchmark"""
        te = calculator_with_benchmark.calculate_tracking_error()

        assert isinstance(te, float)
        assert te >= 0

    def test_information_ratio_with_benchmark(self, calculator_with_benchmark):
        """Test information ratio con benchmark"""
        ir = calculator_with_benchmark.calculate_information_ratio()

        assert isinstance(ir, float)

    def test_beta_with_benchmark(self, calculator_with_benchmark):
        """Test beta con benchmark"""
        beta = calculator_with_benchmark.calculate_beta()

        assert isinstance(beta, float)

    def test_alpha_with_benchmark(self, calculator_with_benchmark):
        """Test alpha con benchmark"""
        alpha = calculator_with_benchmark.calculate_alpha()

        assert isinstance(alpha, float)

    def test_generate_report(self, calculator):
        """Test generar reporte"""
        report = calculator.generate_report()

        assert isinstance(report, dict)
        assert "portfolio_metrics" in report
        assert "diversification" in report
        assert "correlation" in report
        assert "contributions" in report
        assert "weights" in report

    def test_get_risk_parity_weights(self, calculator):
        """Test obtener pesos de risk parity"""
        weights = calculator.get_risk_parity_weights()

        assert isinstance(weights, dict)
        assert len(weights) > 0

        # Deberia sumar ~1
        total = sum(weights.values())
        assert total == pytest.approx(1.0, rel=0.01)


class TestDiversificationMetrics:
    """Tests para DiversificationMetrics"""

    def test_create_metrics(self):
        """Test crear metricas"""
        metrics = DiversificationMetrics(
            diversification_ratio=1.5,
            concentration_hhi=0.25,
            effective_n=4.0,
        )

        assert metrics.diversification_ratio == 1.5
        assert metrics.concentration_hhi == 0.25
        assert metrics.effective_n == 4.0


class TestCorrelationMetrics:
    """Tests para CorrelationMetrics"""

    def test_create_metrics(self):
        """Test crear metricas"""
        metrics = CorrelationMetrics(
            avg_correlation=0.5,
            max_correlation=0.8,
            min_correlation=0.1,
            highly_correlated_pairs=2,
        )

        assert metrics.avg_correlation == 0.5
        assert metrics.highly_correlated_pairs == 2


class TestContributionMetrics:
    """Tests para ContributionMetrics"""

    def test_create_metrics(self):
        """Test crear metricas"""
        metrics = ContributionMetrics(
            return_contribution={"BTC/USDT": 0.05, "ETH/USDT": 0.03},
            risk_contribution={"BTC/USDT": 0.40, "ETH/USDT": 0.60},
        )

        assert metrics.return_contribution["BTC/USDT"] == 0.05
        assert metrics.risk_contribution["ETH/USDT"] == 0.60
