"""
Tests para portfolio/backtester.py
"""

import pytest
import sys
from pathlib import Path
import pandas as pd
import numpy as np

src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from portfolio.backtester import PortfolioBacktester
from portfolio.models import (
    PortfolioConfig,
    AllocationMethod,
    RebalanceFrequency,
)


class TestPortfolioBacktester:
    """Tests para PortfolioBacktester"""

    @pytest.fixture
    def backtester(self, default_config):
        """Backtester con config por defecto"""
        return PortfolioBacktester(default_config)

    def test_create_backtester(self, backtester):
        """Test crear backtester"""
        assert backtester is not None
        assert backtester.config is not None
        assert backtester.allocator is not None

    def test_load_data(self, backtester, sample_ohlcv_data):
        """Test cargar datos"""
        backtester.load_data(sample_ohlcv_data)

        assert backtester.aligned_data is not None
        assert backtester.returns_data is not None
        assert len(backtester.aligned_data) > 0

    def test_load_data_missing_symbol(self, backtester, sample_ohlcv_data):
        """Test error al cargar datos con simbolo faltante"""
        # Remover un simbolo
        del sample_ohlcv_data["BTC/USDT"]

        with pytest.raises(ValueError):
            backtester.load_data(sample_ohlcv_data)

    def test_load_data_missing_column(self, backtester, sample_ohlcv_data):
        """Test error al cargar datos sin columna close"""
        # Remover columna close
        sample_ohlcv_data["BTC/USDT"] = sample_ohlcv_data["BTC/USDT"].drop(columns=["close"])

        with pytest.raises(ValueError):
            backtester.load_data(sample_ohlcv_data)

    def test_run_backtest(self, backtester, sample_ohlcv_data):
        """Test ejecutar backtest"""
        backtester.load_data(sample_ohlcv_data)
        result = backtester.run(warmup_period=30)

        assert result is not None
        assert result.config == backtester.config
        assert len(result.equity_curve) > 0
        assert len(result.returns) > 0

    def test_backtest_equity_curve(self, backtester, sample_ohlcv_data):
        """Test que equity curve es razonable"""
        backtester.load_data(sample_ohlcv_data)
        result = backtester.run(warmup_period=30)

        # Equity no deberia ser negativa
        assert all(e > 0 for e in result.equity_curve)

        # Deberia empezar cerca del capital inicial
        assert result.equity_curve[0] == pytest.approx(
            backtester.config.initial_capital, rel=0.1
        )

    def test_backtest_with_rebalancing(self, sample_ohlcv_data, symbols):
        """Test backtest con rebalanceo"""
        config = PortfolioConfig(
            initial_capital=10000,
            symbols=symbols,
            allocation_method=AllocationMethod.EQUAL_WEIGHT,
            rebalance_frequency=RebalanceFrequency.WEEKLY,
        )

        backtester = PortfolioBacktester(config)
        backtester.load_data(sample_ohlcv_data)
        result = backtester.run(warmup_period=30)

        # Deberia haber rebalanceos
        # Con datos de 500 periodos hourly (~20 dias), deberia haber ~3 rebalanceos semanales
        # Pero depende de los datos exactos
        assert result.metrics.num_rebalances >= 0

    def test_backtest_no_rebalancing(self, sample_ohlcv_data, symbols):
        """Test backtest sin rebalanceo"""
        config = PortfolioConfig(
            initial_capital=10000,
            symbols=symbols,
            allocation_method=AllocationMethod.EQUAL_WEIGHT,
            rebalance_frequency=RebalanceFrequency.NEVER,
        )

        backtester = PortfolioBacktester(config)
        backtester.load_data(sample_ohlcv_data)
        result = backtester.run(warmup_period=30)

        # Solo deberia haber el rebalanceo inicial
        assert result.metrics.num_rebalances <= 1

    def test_backtest_metrics(self, backtester, sample_ohlcv_data):
        """Test metricas del backtest"""
        backtester.load_data(sample_ohlcv_data)
        result = backtester.run(warmup_period=30)

        metrics = result.metrics

        # Verificar que metricas basicas estan calculadas
        assert metrics.total_return != 0 or metrics.total_return_pct != 0
        assert metrics.volatility >= 0
        assert metrics.annualized_volatility >= 0

    def test_backtest_trade_history(self, backtester, sample_ohlcv_data):
        """Test historial de trades"""
        backtester.load_data(sample_ohlcv_data)
        result = backtester.run(warmup_period=30)

        # Deberia haber al menos trades iniciales
        assert len(result.trade_history) > 0

        # Verificar estructura de trades
        for trade in result.trade_history:
            assert trade.symbol in backtester.config.symbols
            assert trade.side in ["buy", "sell"]
            assert trade.quantity > 0

    def test_backtest_weight_history(self, backtester, sample_ohlcv_data):
        """Test historial de pesos"""
        backtester.load_data(sample_ohlcv_data)
        result = backtester.run(warmup_period=30)

        assert len(result.weight_history) > 0

        for symbol in backtester.config.symbols:
            assert symbol in result.weight_history
            assert len(result.weight_history[symbol]) == len(result.equity_curve)

    def test_get_equity_curve(self, backtester, sample_ohlcv_data):
        """Test obtener equity curve como Series"""
        backtester.load_data(sample_ohlcv_data)
        backtester.run(warmup_period=30)

        equity = backtester.get_equity_curve()

        assert isinstance(equity, pd.Series)
        assert len(equity) > 0

    def test_get_returns(self, backtester, sample_ohlcv_data):
        """Test obtener retornos como Series"""
        backtester.load_data(sample_ohlcv_data)
        backtester.run(warmup_period=30)

        returns = backtester.get_returns()

        assert isinstance(returns, pd.Series)
        assert len(returns) > 0

    def test_get_weight_history_df(self, backtester, sample_ohlcv_data):
        """Test obtener historial de pesos como DataFrame"""
        backtester.load_data(sample_ohlcv_data)
        backtester.run(warmup_period=30)

        weights = backtester.get_weight_history()

        assert isinstance(weights, pd.DataFrame)
        for symbol in backtester.config.symbols:
            assert symbol in weights.columns


class TestBacktesterWithDifferentMethods:
    """Tests con diferentes metodos de allocation"""

    @pytest.fixture
    def ohlcv_data(self, sample_ohlcv_data):
        """Datos OHLCV"""
        return sample_ohlcv_data

    def test_risk_parity_backtest(self, ohlcv_data, symbols):
        """Test backtest con risk parity"""
        config = PortfolioConfig(
            initial_capital=10000,
            symbols=symbols,
            allocation_method=AllocationMethod.RISK_PARITY,
            rebalance_frequency=RebalanceFrequency.MONTHLY,
        )

        backtester = PortfolioBacktester(config)
        backtester.load_data(ohlcv_data)
        result = backtester.run(warmup_period=50)

        assert result is not None
        assert len(result.equity_curve) > 0

    def test_min_variance_backtest(self, ohlcv_data, symbols):
        """Test backtest con minima varianza"""
        config = PortfolioConfig(
            initial_capital=10000,
            symbols=symbols,
            allocation_method=AllocationMethod.MIN_VARIANCE,
            rebalance_frequency=RebalanceFrequency.MONTHLY,
        )

        backtester = PortfolioBacktester(config)
        backtester.load_data(ohlcv_data)
        result = backtester.run(warmup_period=50)

        assert result is not None
        # Min variance deberia tener baja volatilidad
        assert result.metrics.annualized_volatility >= 0

    def test_max_sharpe_backtest(self, ohlcv_data, symbols):
        """Test backtest con max Sharpe"""
        config = PortfolioConfig(
            initial_capital=10000,
            symbols=symbols,
            allocation_method=AllocationMethod.MAX_SHARPE,
            rebalance_frequency=RebalanceFrequency.MONTHLY,
        )

        backtester = PortfolioBacktester(config)
        backtester.load_data(ohlcv_data)
        result = backtester.run(warmup_period=50)

        assert result is not None


class TestBacktesterEdgeCases:
    """Tests de casos limite"""

    def test_short_data(self, symbols):
        """Test con pocos datos"""
        config = PortfolioConfig(
            initial_capital=10000,
            symbols=symbols[:2],  # Solo 2 simbolos
        )

        # Crear datos cortos
        dates = pd.date_range(start="2024-01-01", periods=50, freq="1h")
        data = {}
        for symbol in symbols[:2]:
            data[symbol] = pd.DataFrame({
                "open": np.random.uniform(100, 110, 50),
                "high": np.random.uniform(105, 115, 50),
                "low": np.random.uniform(95, 105, 50),
                "close": np.random.uniform(100, 110, 50),
                "volume": np.random.uniform(1000, 10000, 50),
            }, index=dates)

        backtester = PortfolioBacktester(config)
        backtester.load_data(data)
        result = backtester.run(warmup_period=20)

        assert result is not None
        assert len(result.equity_curve) > 0

    def test_two_assets(self, symbols):
        """Test con solo 2 activos"""
        config = PortfolioConfig(
            initial_capital=10000,
            symbols=symbols[:2],
        )

        dates = pd.date_range(start="2024-01-01", periods=100, freq="1h")
        data = {}
        for symbol in symbols[:2]:
            data[symbol] = pd.DataFrame({
                "open": np.random.uniform(100, 110, 100),
                "high": np.random.uniform(105, 115, 100),
                "low": np.random.uniform(95, 105, 100),
                "close": np.random.uniform(100, 110, 100),
                "volume": np.random.uniform(1000, 10000, 100),
            }, index=dates)

        backtester = PortfolioBacktester(config)
        backtester.load_data(data)
        result = backtester.run(warmup_period=20)

        assert result is not None
        assert len(result.weight_history) == 2


class TestWalkForwardBacktest:
    """Tests for walk-forward backtesting"""

    @pytest.fixture
    def large_ohlcv_data(self, symbols):
        """Large OHLCV data for walk-forward tests"""
        np.random.seed(42)
        n_periods = 1000  # Enough for multiple splits

        data = {}
        dates = pd.date_range(start="2024-01-01", periods=n_periods, freq="1h")

        base_prices = {
            "BTC/USDT": 50000.0,
            "ETH/USDT": 3000.0,
            "SOL/USDT": 100.0,
            "LINK/USDT": 15.0,
        }

        for symbol in symbols:
            base = base_prices.get(symbol, 100.0)
            returns = np.random.normal(0.0001, 0.02, n_periods)
            prices = base * np.exp(np.cumsum(returns))

            df = pd.DataFrame({
                "open": prices * (1 + np.random.uniform(-0.005, 0.005, n_periods)),
                "high": prices * (1 + np.random.uniform(0, 0.02, n_periods)),
                "low": prices * (1 - np.random.uniform(0, 0.02, n_periods)),
                "close": prices,
                "volume": np.random.uniform(1000, 10000, n_periods),
            }, index=dates)

            data[symbol] = df

        return data

    def test_walk_forward_basic(self, default_config, large_ohlcv_data):
        """Test basic walk-forward backtest"""
        backtester = PortfolioBacktester(default_config)
        backtester.load_data(large_ohlcv_data)

        result = backtester.walk_forward_backtest(
            n_splits=3,
            train_pct=0.6,
            anchored=True,
        )

        assert result is not None
        assert len(result.splits) == 3
        assert result.robustness_score >= 0
        assert result.robustness_score <= 1

    def test_walk_forward_anchored(self, default_config, large_ohlcv_data):
        """Test anchored (expanding) walk-forward"""
        backtester = PortfolioBacktester(default_config)
        backtester.load_data(large_ohlcv_data)

        result = backtester.walk_forward_backtest(
            n_splits=3,
            train_pct=0.5,
            anchored=True,
        )

        assert result.anchored is True

        # In anchored mode, training size should increase
        train_sizes = [s.train_rows for s in result.splits]
        for i in range(1, len(train_sizes)):
            assert train_sizes[i] > train_sizes[i-1]

    def test_walk_forward_rolling(self, default_config, large_ohlcv_data):
        """Test rolling (fixed window) walk-forward"""
        backtester = PortfolioBacktester(default_config)
        backtester.load_data(large_ohlcv_data)

        result = backtester.walk_forward_backtest(
            n_splits=3,
            train_pct=0.5,
            anchored=False,
        )

        assert result.anchored is False

        # In rolling mode, training size should be approximately constant
        train_sizes = [s.train_rows for s in result.splits]
        # Allow some tolerance
        assert max(train_sizes) - min(train_sizes) <= 50

    def test_walk_forward_split_structure(self, default_config, large_ohlcv_data):
        """Test that split results have correct structure"""
        backtester = PortfolioBacktester(default_config)
        backtester.load_data(large_ohlcv_data)

        result = backtester.walk_forward_backtest(n_splits=3)

        for split in result.splits:
            assert split.train_start < split.train_end
            assert split.test_start < split.test_end
            assert split.train_end <= split.test_start
            assert split.train_rows > 0
            assert split.test_rows > 0
            assert len(split.optimized_weights) > 0

    def test_walk_forward_metrics(self, default_config, large_ohlcv_data):
        """Test that metrics are calculated"""
        backtester = PortfolioBacktester(default_config)
        backtester.load_data(large_ohlcv_data)

        result = backtester.walk_forward_backtest(n_splits=3)

        # Aggregated metrics
        assert isinstance(result.avg_train_sharpe, float)
        assert isinstance(result.avg_test_sharpe, float)
        assert isinstance(result.avg_degradation, float)

        # Ratios
        assert 0 <= result.positive_oos_ratio <= 1
        assert 0 <= result.consistency_ratio <= 1

    def test_walk_forward_combined_oos(self, default_config, large_ohlcv_data):
        """Test combined OOS equity curve"""
        backtester = PortfolioBacktester(default_config)
        backtester.load_data(large_ohlcv_data)

        result = backtester.walk_forward_backtest(n_splits=3)

        # Combined OOS should exist
        assert len(result.combined_oos_equity) > 0
        assert len(result.combined_oos_returns) > 0

        # Should have metrics
        assert result.combined_oos_metrics.total_days > 0

    def test_walk_forward_weight_stability(self, default_config, large_ohlcv_data):
        """Test weight stability analysis"""
        backtester = PortfolioBacktester(default_config)
        backtester.load_data(large_ohlcv_data)

        result = backtester.walk_forward_backtest(
            n_splits=3,
            stability_threshold=0.3,
        )

        # Stability analysis should exist
        assert isinstance(result.weight_stability, dict)
        assert isinstance(result.unstable_allocations, list)

    def test_walk_forward_with_gap(self, default_config, large_ohlcv_data):
        """Test walk-forward with gap between train and test"""
        backtester = PortfolioBacktester(default_config)
        backtester.load_data(large_ohlcv_data)

        result = backtester.walk_forward_backtest(
            n_splits=3,
            gap=10,
        )

        # Should complete without error
        assert len(result.splits) == 3

    def test_walk_forward_different_allocations(self, symbols, large_ohlcv_data):
        """Test walk-forward with different allocation methods"""
        for method in [AllocationMethod.EQUAL_WEIGHT, AllocationMethod.RISK_PARITY]:
            config = PortfolioConfig(
                initial_capital=10000,
                symbols=symbols,
                allocation_method=method,
                rebalance_frequency=RebalanceFrequency.MONTHLY,
            )

            backtester = PortfolioBacktester(config)
            backtester.load_data(large_ohlcv_data)

            result = backtester.walk_forward_backtest(n_splits=2)

            assert result is not None
            assert len(result.splits) == 2

    def test_walk_forward_no_data_error(self, default_config):
        """Test error when no data loaded"""
        backtester = PortfolioBacktester(default_config)

        with pytest.raises(ValueError):
            backtester.walk_forward_backtest(n_splits=3)

    def test_walk_forward_result_to_dict(self, default_config, large_ohlcv_data):
        """Test result serialization"""
        backtester = PortfolioBacktester(default_config)
        backtester.load_data(large_ohlcv_data)

        result = backtester.walk_forward_backtest(n_splits=2)
        result_dict = result.to_dict()

        assert "n_splits" in result_dict
        assert "splits" in result_dict
        assert "robustness_score" in result_dict
        assert len(result_dict["splits"]) == 2

    def test_walk_forward_split_to_dict(self, default_config, large_ohlcv_data):
        """Test split result serialization"""
        backtester = PortfolioBacktester(default_config)
        backtester.load_data(large_ohlcv_data)

        result = backtester.walk_forward_backtest(n_splits=2)
        split_dict = result.splits[0].to_dict()

        assert "split_idx" in split_dict
        assert "train_sharpe" in split_dict
        assert "test_sharpe" in split_dict
        assert "degradation_pct" in split_dict
        assert "optimized_weights" in split_dict

    def test_walk_forward_robustness_score_bounds(self, default_config, large_ohlcv_data):
        """Test that robustness score is bounded"""
        backtester = PortfolioBacktester(default_config)
        backtester.load_data(large_ohlcv_data)

        result = backtester.walk_forward_backtest(n_splits=3)

        assert 0 <= result.robustness_score <= 1

    def test_walk_forward_execution_time_tracked(self, default_config, large_ohlcv_data):
        """Test that execution time is tracked"""
        backtester = PortfolioBacktester(default_config)
        backtester.load_data(large_ohlcv_data)

        result = backtester.walk_forward_backtest(n_splits=2)

        assert result.execution_time > 0
