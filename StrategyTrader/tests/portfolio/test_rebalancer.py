"""
Tests para portfolio/rebalancer.py
"""

import pytest
import sys
from pathlib import Path
from datetime import datetime, timedelta

src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from portfolio.rebalancer import (
    PortfolioRebalancer,
    RebalanceDecision,
    RebalancerFactory,
)
from portfolio.models import (
    PortfolioConfig,
    PortfolioState,
    PortfolioPosition,
    RebalanceFrequency,
    RebalanceReason,
)


class TestPortfolioRebalancer:
    """Tests para PortfolioRebalancer"""

    @pytest.fixture
    def rebalancer(self, default_config):
        """Rebalancer con config por defecto"""
        return PortfolioRebalancer(default_config)

    @pytest.fixture
    def threshold_rebalancer(self, threshold_config):
        """Rebalancer por threshold"""
        return PortfolioRebalancer(threshold_config)

    def test_create_rebalancer(self, rebalancer):
        """Test crear rebalancer"""
        assert rebalancer is not None
        assert rebalancer.rebalance_count == 0

    def test_should_rebalance_initial(self, rebalancer, sample_state):
        """Test rebalanceo inicial"""
        # Sin rebalanceo previo, deberia recomendar rebalancear
        decision = rebalancer.should_rebalance(sample_state)

        assert decision.should_rebalance == True
        assert decision.reason == RebalanceReason.SCHEDULED

    def test_should_not_rebalance_recent(self, rebalancer, sample_state):
        """Test no rebalancear si es reciente"""
        # Marcar que ya rebalanceamos hoy
        rebalancer.last_rebalance_date = datetime.now()

        decision = rebalancer.should_rebalance(sample_state)

        # Con frecuencia mensual, no deberia rebalancear
        assert decision.should_rebalance == False

    def test_threshold_rebalance_trigger(self, threshold_rebalancer):
        """Test que threshold trigger funciona"""
        state = PortfolioState(
            total_equity=10000,
            cash=0,
            current_weights={
                "BTC/USDT": 0.35,  # 10% drift
                "ETH/USDT": 0.35,
                "SOL/USDT": 0.20,
                "LINK/USDT": 0.10,
            },
            target_weights={
                "BTC/USDT": 0.25,
                "ETH/USDT": 0.25,
                "SOL/USDT": 0.25,
                "LINK/USDT": 0.25,
            },
        )

        decision = threshold_rebalancer.should_rebalance(state)

        assert decision.should_rebalance == True
        assert decision.reason == RebalanceReason.THRESHOLD_BREACH
        assert decision.max_drift >= 0.05

    def test_threshold_no_rebalance_small_drift(self, threshold_rebalancer):
        """Test no rebalancear con drift pequeno"""
        state = PortfolioState(
            total_equity=10000,
            current_weights={
                "BTC/USDT": 0.26,  # 1% drift
                "ETH/USDT": 0.26,
                "SOL/USDT": 0.24,
                "LINK/USDT": 0.24,
            },
            target_weights={
                "BTC/USDT": 0.25,
                "ETH/USDT": 0.25,
                "SOL/USDT": 0.25,
                "LINK/USDT": 0.25,
            },
        )

        decision = threshold_rebalancer.should_rebalance(state)

        assert decision.should_rebalance == False

    def test_calculate_rebalance_trades(self, rebalancer, symbols):
        """Test calcular trades de rebalanceo"""
        state = PortfolioState(
            total_equity=10000,
            cash=2000,
            positions={
                "BTC/USDT": PortfolioPosition("BTC/USDT", 0.1, 50000, 50000),
                "ETH/USDT": PortfolioPosition("ETH/USDT", 1.0, 3000, 3000),
            },
            current_weights={
                "BTC/USDT": 0.50,
                "ETH/USDT": 0.30,
                "SOL/USDT": 0.0,
                "LINK/USDT": 0.0,
            },
            target_weights={
                "BTC/USDT": 0.25,
                "ETH/USDT": 0.25,
                "SOL/USDT": 0.25,
                "LINK/USDT": 0.25,
            },
        )

        prices = {
            "BTC/USDT": 50000,
            "ETH/USDT": 3000,
            "SOL/USDT": 100,
            "LINK/USDT": 15,
        }

        trades = rebalancer.calculate_rebalance_trades(state, prices)

        assert len(trades) > 0

        # Deberia haber venta de BTC (sobre-pesado)
        btc_trades = [t for t in trades if t.symbol == "BTC/USDT"]
        assert any(t.side == "sell" for t in btc_trades)

        # Deberia haber compra de SOL y LINK (sub-pesados)
        sol_trades = [t for t in trades if t.symbol == "SOL/USDT"]
        if sol_trades:
            assert sol_trades[0].side == "buy"

    def test_estimate_transaction_costs(self, rebalancer):
        """Test estimar costos de transaccion"""
        from portfolio.models import RebalanceTrade

        trades = [
            RebalanceTrade(
                symbol="BTC/USDT",
                side="sell",
                quantity=0.05,
                estimated_price=50000,
                estimated_value=2500,
                estimated_commission=2.5,
            ),
            RebalanceTrade(
                symbol="SOL/USDT",
                side="buy",
                quantity=25,
                estimated_price=100,
                estimated_value=2500,
                estimated_commission=2.5,
            ),
        ]

        commission, slippage, total = rebalancer.estimate_transaction_costs(trades)

        assert commission == 5.0
        assert slippage >= 0
        assert total == commission + slippage

    def test_get_weight_drifts(self, rebalancer, symbols):
        """Test obtener drifts"""
        state = PortfolioState(
            current_weights={
                "BTC/USDT": 0.30,
                "ETH/USDT": 0.30,
                "SOL/USDT": 0.20,
                "LINK/USDT": 0.20,
            },
            target_weights={
                "BTC/USDT": 0.25,
                "ETH/USDT": 0.25,
                "SOL/USDT": 0.25,
                "LINK/USDT": 0.25,
            },
        )

        drifts = rebalancer.get_weight_drifts(state)

        assert drifts["BTC/USDT"] == pytest.approx(0.05, rel=0.01)
        assert drifts["SOL/USDT"] == pytest.approx(-0.05, rel=0.01)

    def test_get_largest_drifts(self, rebalancer, symbols):
        """Test obtener mayores drifts"""
        state = PortfolioState(
            current_weights={
                "BTC/USDT": 0.40,  # +15%
                "ETH/USDT": 0.30,  # +5%
                "SOL/USDT": 0.20,  # -5%
                "LINK/USDT": 0.10,  # -15%
            },
            target_weights={
                "BTC/USDT": 0.25,
                "ETH/USDT": 0.25,
                "SOL/USDT": 0.25,
                "LINK/USDT": 0.25,
            },
        )

        largest = rebalancer.get_largest_drifts(state, n=2)

        assert len(largest) == 2
        # Los dos mayores deberian ser BTC y LINK
        symbols_in_largest = [s for s, d in largest]
        assert "BTC/USDT" in symbols_in_largest
        assert "LINK/USDT" in symbols_in_largest

    def test_estimate_turnover(self, rebalancer, symbols):
        """Test estimar turnover"""
        state = PortfolioState(
            current_weights={
                "BTC/USDT": 0.40,
                "ETH/USDT": 0.30,
                "SOL/USDT": 0.20,
                "LINK/USDT": 0.10,
            },
            target_weights={
                "BTC/USDT": 0.25,
                "ETH/USDT": 0.25,
                "SOL/USDT": 0.25,
                "LINK/USDT": 0.25,
            },
        )

        turnover = rebalancer.estimate_turnover(state)

        # Suma de diferencias absolutas
        expected = abs(0.40 - 0.25) + abs(0.30 - 0.25) + abs(0.20 - 0.25) + abs(0.10 - 0.25)
        assert turnover == pytest.approx(expected, rel=0.01)

    def test_reset(self, rebalancer):
        """Test reiniciar rebalancer"""
        rebalancer.last_rebalance_date = datetime.now()
        rebalancer.rebalance_count = 5

        rebalancer.reset()

        assert rebalancer.last_rebalance_date is None
        assert rebalancer.rebalance_count == 0


class TestRebalancerFactory:
    """Tests para RebalancerFactory"""

    def test_create_monthly(self, default_config):
        """Test crear rebalancer mensual"""
        rebalancer = RebalancerFactory.create_monthly(default_config)

        assert rebalancer.config.rebalance_frequency == RebalanceFrequency.MONTHLY

    def test_create_threshold(self, default_config):
        """Test crear rebalancer por threshold"""
        rebalancer = RebalancerFactory.create_threshold_5pct(default_config)

        assert rebalancer.config.rebalance_frequency == RebalanceFrequency.THRESHOLD
        assert rebalancer.config.rebalance_threshold == 0.05

    def test_create_quarterly(self, default_config):
        """Test crear rebalancer trimestral"""
        rebalancer = RebalancerFactory.create_quarterly(default_config)

        assert rebalancer.config.rebalance_frequency == RebalanceFrequency.QUARTERLY


class TestRebalanceDecision:
    """Tests para RebalanceDecision"""

    def test_create_decision(self):
        """Test crear decision"""
        decision = RebalanceDecision(
            should_rebalance=True,
            reason=RebalanceReason.THRESHOLD_BREACH,
            max_drift=0.10,
            avg_drift=0.05,
            message="Drift exceeded threshold",
        )

        assert decision.should_rebalance == True
        assert decision.max_drift == 0.10
