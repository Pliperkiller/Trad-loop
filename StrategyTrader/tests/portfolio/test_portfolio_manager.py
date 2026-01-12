"""
Tests para portfolio/portfolio_manager.py
"""

import pytest
import sys
from pathlib import Path
import pandas as pd
import numpy as np

src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from portfolio.portfolio_manager import (
    PortfolioManager,
    PortfolioManagerFactory,
)
from portfolio.models import (
    PortfolioConfig,
    AllocationMethod,
    RebalanceFrequency,
)


class TestPortfolioManager:
    """Tests para PortfolioManager"""

    @pytest.fixture
    def manager(self, default_config):
        """Manager con config por defecto"""
        return PortfolioManager(default_config)

    def test_create_manager(self, manager):
        """Test crear manager"""
        assert manager is not None
        assert manager.config is not None
        assert manager.allocator is not None
        assert manager.rebalancer is not None

    def test_backtest(self, manager, sample_ohlcv_data):
        """Test ejecutar backtest"""
        result = manager.backtest(sample_ohlcv_data, warmup_period=30)

        assert result is not None
        assert len(result.equity_curve) > 0
        assert manager.last_result is not None
        assert manager.returns_data is not None

    def test_optimize_weights(self, manager, sample_ohlcv_data):
        """Test optimizar pesos"""
        # Primero hacer backtest para tener datos
        manager.backtest(sample_ohlcv_data, warmup_period=30)

        result = manager.optimize_weights()

        assert result is not None
        assert len(result.weights) > 0

    def test_optimize_weights_different_method(self, manager, sample_ohlcv_data):
        """Test optimizar con diferente metodo"""
        manager.backtest(sample_ohlcv_data, warmup_period=30)

        result = manager.optimize_weights(method=AllocationMethod.RISK_PARITY)

        assert result is not None
        assert result.method == AllocationMethod.RISK_PARITY

    def test_set_target_weights(self, manager):
        """Test establecer pesos objetivo"""
        weights = {
            "BTC/USDT": 0.40,
            "ETH/USDT": 0.30,
            "SOL/USDT": 0.20,
            "LINK/USDT": 0.10,
        }

        manager.set_target_weights(weights)

        assert manager.config.target_weights == weights
        assert manager.config.allocation_method == AllocationMethod.CUSTOM

    def test_set_target_weights_invalid(self, manager):
        """Test error con pesos invalidos"""
        weights = {
            "BTC/USDT": 0.50,
            "ETH/USDT": 0.20,
        }  # No suman 1

        with pytest.raises(ValueError):
            manager.set_target_weights(weights)

    def test_get_metrics(self, manager, sample_ohlcv_data):
        """Test obtener metricas"""
        manager.backtest(sample_ohlcv_data, warmup_period=30)

        metrics = manager.get_metrics()

        assert metrics is not None

    def test_get_metrics_no_backtest(self, manager):
        """Test metricas sin backtest"""
        metrics = manager.get_metrics()

        assert metrics is None

    def test_get_current_weights(self, manager, sample_ohlcv_data):
        """Test obtener pesos actuales"""
        manager.backtest(sample_ohlcv_data, warmup_period=30)

        weights = manager.get_current_weights()

        assert len(weights) > 0
        total = sum(weights.values())
        assert total == pytest.approx(1.0, rel=0.05)

    def test_get_weight_drift(self, manager, sample_ohlcv_data):
        """Test obtener drift de pesos"""
        manager.backtest(sample_ohlcv_data, warmup_period=30)

        drift = manager.get_weight_drift()

        assert len(drift) > 0

    def test_get_correlation_matrix(self, manager, sample_ohlcv_data):
        """Test obtener matriz de correlacion"""
        manager.backtest(sample_ohlcv_data, warmup_period=30)

        corr = manager.get_correlation_matrix()

        assert corr is not None
        assert isinstance(corr, pd.DataFrame)

    def test_get_efficient_frontier(self, manager, sample_ohlcv_data):
        """Test obtener frontera eficiente"""
        manager.backtest(sample_ohlcv_data, warmup_period=30)

        frontier = manager.get_efficient_frontier(n_points=10)

        assert len(frontier) > 0

    def test_calculate_portfolio_stats(self, manager, sample_ohlcv_data):
        """Test calcular estadisticas"""
        manager.backtest(sample_ohlcv_data, warmup_period=30)

        stats = manager.calculate_portfolio_stats()

        assert isinstance(stats, dict)
        assert "portfolio_metrics" in stats

    def test_compare_allocations(self, manager, sample_ohlcv_data):
        """Test comparar allocations"""
        manager.backtest(sample_ohlcv_data, warmup_period=30)

        comparison = manager.compare_allocations()

        assert isinstance(comparison, pd.DataFrame)
        assert len(comparison) > 0

    def test_get_equity_curve(self, manager, sample_ohlcv_data):
        """Test obtener equity curve"""
        manager.backtest(sample_ohlcv_data, warmup_period=30)

        equity = manager.get_equity_curve()

        assert equity is not None
        assert isinstance(equity, pd.Series)

    def test_get_returns(self, manager, sample_ohlcv_data):
        """Test obtener retornos"""
        manager.backtest(sample_ohlcv_data, warmup_period=30)

        returns = manager.get_returns()

        assert returns is not None
        assert isinstance(returns, pd.Series)

    def test_get_weight_history(self, manager, sample_ohlcv_data):
        """Test obtener historial de pesos"""
        manager.backtest(sample_ohlcv_data, warmup_period=30)

        weights = manager.get_weight_history()

        assert weights is not None
        assert isinstance(weights, pd.DataFrame)


class TestPortfolioManagerFactory:
    """Tests para PortfolioManagerFactory"""

    def test_create_equal_weight(self, symbols):
        """Test crear manager equal weight"""
        manager = PortfolioManagerFactory.create_equal_weight(
            symbols=symbols,
            initial_capital=10000,
        )

        assert manager.config.allocation_method == AllocationMethod.EQUAL_WEIGHT

    def test_create_risk_parity(self, symbols):
        """Test crear manager risk parity"""
        manager = PortfolioManagerFactory.create_risk_parity(
            symbols=symbols,
            initial_capital=10000,
        )

        assert manager.config.allocation_method == AllocationMethod.RISK_PARITY

    def test_create_max_sharpe(self, symbols):
        """Test crear manager max Sharpe"""
        manager = PortfolioManagerFactory.create_max_sharpe(
            symbols=symbols,
            initial_capital=10000,
        )

        assert manager.config.allocation_method == AllocationMethod.MAX_SHARPE

    def test_create_custom(self, symbols):
        """Test crear manager custom"""
        weights = {
            "BTC/USDT": 0.50,
            "ETH/USDT": 0.25,
            "SOL/USDT": 0.15,
            "LINK/USDT": 0.10,
        }

        manager = PortfolioManagerFactory.create_custom(
            symbols=symbols,
            weights=weights,
            initial_capital=10000,
        )

        assert manager.config.allocation_method == AllocationMethod.CUSTOM
        assert manager.config.target_weights == weights


class TestPortfolioManagerIntegration:
    """Tests de integracion"""

    def test_full_workflow(self, sample_ohlcv_data, symbols):
        """Test flujo completo"""
        # 1. Crear manager
        manager = PortfolioManagerFactory.create_risk_parity(
            symbols=symbols,
            initial_capital=10000,
            rebalance_frequency=RebalanceFrequency.WEEKLY,
        )

        # 2. Ejecutar backtest
        result = manager.backtest(sample_ohlcv_data, warmup_period=50)

        # 3. Verificar resultados
        assert result is not None
        assert len(result.equity_curve) > 0

        # 4. Obtener metricas
        metrics = manager.get_metrics()
        assert metrics is not None

        # 5. Comparar con otros metodos
        comparison = manager.compare_allocations()
        assert len(comparison) >= 2

        # 6. Calcular estadisticas
        stats = manager.calculate_portfolio_stats()
        assert "portfolio_metrics" in stats

    def test_multiple_backtests(self, sample_ohlcv_data, symbols):
        """Test multiples backtests con diferentes configs"""
        results = {}

        for method in [AllocationMethod.EQUAL_WEIGHT, AllocationMethod.RISK_PARITY]:
            config = PortfolioConfig(
                initial_capital=10000,
                symbols=symbols,
                allocation_method=method,
                rebalance_frequency=RebalanceFrequency.MONTHLY,
            )

            manager = PortfolioManager(config)
            result = manager.backtest(sample_ohlcv_data, warmup_period=50)
            results[method] = result

        # Verificar que ambos backtests funcionaron
        assert len(results) == 2
        for method, result in results.items():
            assert len(result.equity_curve) > 0

    def test_different_rebalance_frequencies(self, sample_ohlcv_data, symbols):
        """Test diferentes frecuencias de rebalanceo"""
        frequencies = [
            RebalanceFrequency.NEVER,
            RebalanceFrequency.WEEKLY,
            RebalanceFrequency.MONTHLY,
        ]

        for freq in frequencies:
            config = PortfolioConfig(
                initial_capital=10000,
                symbols=symbols,
                allocation_method=AllocationMethod.EQUAL_WEIGHT,
                rebalance_frequency=freq,
            )

            manager = PortfolioManager(config)
            result = manager.backtest(sample_ohlcv_data, warmup_period=30)

            assert result is not None
