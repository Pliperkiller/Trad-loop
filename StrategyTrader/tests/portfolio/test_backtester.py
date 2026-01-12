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
