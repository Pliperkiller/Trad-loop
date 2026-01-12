"""
Tests para portfolio/models.py
"""

import pytest
import sys
from pathlib import Path
from datetime import datetime

src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from portfolio.models import (
    AllocationMethod,
    RebalanceFrequency,
    RebalanceReason,
    PortfolioPosition,
    AssetAllocation,
    RebalanceTrade,
    PortfolioState,
    PortfolioConfig,
    PortfolioTradeRecord,
    RebalanceEvent,
    PortfolioMetrics,
    PortfolioResult,
)


class TestPortfolioPosition:
    """Tests para PortfolioPosition"""

    def test_create_position(self):
        """Test crear posicion"""
        pos = PortfolioPosition(
            symbol="BTC/USDT",
            quantity=0.1,
            entry_price=50000,
            current_price=51000,
        )

        assert pos.symbol == "BTC/USDT"
        assert pos.quantity == 0.1
        assert pos.entry_price == 50000
        assert pos.current_price == 51000

    def test_position_value(self):
        """Test valor de posicion"""
        pos = PortfolioPosition(
            symbol="BTC/USDT",
            quantity=0.1,
            entry_price=50000,
            current_price=51000,
        )

        assert pos.value == 0.1 * 51000  # 5100

    def test_position_cost_basis(self):
        """Test costo base"""
        pos = PortfolioPosition(
            symbol="BTC/USDT",
            quantity=0.1,
            entry_price=50000,
            current_price=51000,
        )

        assert pos.cost_basis == 0.1 * 50000  # 5000

    def test_position_pnl(self):
        """Test PnL"""
        pos = PortfolioPosition(
            symbol="BTC/USDT",
            quantity=0.1,
            entry_price=50000,
            current_price=51000,
        )

        assert pos.pnl == 100  # 5100 - 5000

    def test_position_pnl_percent(self):
        """Test PnL porcentual"""
        pos = PortfolioPosition(
            symbol="BTC/USDT",
            quantity=0.1,
            entry_price=50000,
            current_price=51000,
        )

        assert pos.pnl_percent == pytest.approx(2.0, rel=0.01)  # 2%

    def test_position_to_dict(self):
        """Test conversion a dict"""
        pos = PortfolioPosition(
            symbol="BTC/USDT",
            quantity=0.1,
            entry_price=50000,
            current_price=51000,
        )

        d = pos.to_dict()
        assert d["symbol"] == "BTC/USDT"
        assert d["pnl"] == 100


class TestAssetAllocation:
    """Tests para AssetAllocation"""

    def test_create_allocation(self):
        """Test crear allocation"""
        alloc = AssetAllocation(
            symbol="BTC/USDT",
            target_weight=0.25,
            current_weight=0.30,
            value=3000,
            quantity=0.06,
        )

        assert alloc.symbol == "BTC/USDT"
        assert alloc.target_weight == 0.25
        assert alloc.current_weight == 0.30

    def test_drift(self):
        """Test calculo de drift"""
        alloc = AssetAllocation(
            symbol="BTC/USDT",
            target_weight=0.25,
            current_weight=0.30,
        )

        assert alloc.drift == pytest.approx(0.05, rel=0.01)

    def test_needs_rebalance(self):
        """Test si necesita rebalanceo"""
        # Drift > 5%
        alloc1 = AssetAllocation(
            symbol="BTC/USDT",
            target_weight=0.25,
            current_weight=0.35,  # 10% drift
        )
        assert alloc1.needs_rebalance == True

        # Drift < 5%
        alloc2 = AssetAllocation(
            symbol="ETH/USDT",
            target_weight=0.25,
            current_weight=0.27,  # 2% drift
        )
        assert alloc2.needs_rebalance == False


class TestPortfolioConfig:
    """Tests para PortfolioConfig"""

    def test_create_config(self, symbols):
        """Test crear configuracion"""
        config = PortfolioConfig(
            initial_capital=10000,
            symbols=symbols,
        )

        assert config.initial_capital == 10000
        assert len(config.symbols) == 4

    def test_config_validation_negative_capital(self):
        """Test validacion de capital negativo"""
        with pytest.raises(ValueError):
            PortfolioConfig(initial_capital=-1000, symbols=["BTC/USDT"])

    def test_config_validation_invalid_commission(self):
        """Test validacion de comision invalida"""
        with pytest.raises(ValueError):
            PortfolioConfig(
                initial_capital=10000,
                symbols=["BTC/USDT"],
                commission=0.5,  # 50% es demasiado
            )

    def test_config_validation_weights_sum(self):
        """Test validacion de suma de pesos"""
        with pytest.raises(ValueError):
            PortfolioConfig(
                initial_capital=10000,
                symbols=["BTC/USDT", "ETH/USDT"],
                target_weights={"BTC/USDT": 0.3, "ETH/USDT": 0.3},  # Suma 0.6
            )

    def test_config_to_dict(self, default_config):
        """Test conversion a dict"""
        d = default_config.to_dict()

        assert d["initial_capital"] == 10000
        assert d["allocation_method"] == "equal_weight"


class TestPortfolioState:
    """Tests para PortfolioState"""

    def test_create_state(self):
        """Test crear estado"""
        state = PortfolioState(
            total_equity=10000,
            cash=5000,
        )

        assert state.total_equity == 10000
        assert state.cash == 5000

    def test_update_weights(self, sample_positions):
        """Test actualizar pesos"""
        state = PortfolioState(
            total_equity=10000,
            cash=5000,
            positions=sample_positions,
            target_weights={
                "BTC/USDT": 0.5,
                "ETH/USDT": 0.5,
            },
        )

        state.update_weights()

        # BTC: 0.1 * 51000 = 5100 / 10000 = 0.51
        # ETH: 1.0 * 3100 = 3100 / 10000 = 0.31
        assert "BTC/USDT" in state.current_weights
        assert "ETH/USDT" in state.current_weights
        assert state.num_positions == 2


class TestRebalanceTrade:
    """Tests para RebalanceTrade"""

    def test_create_trade(self):
        """Test crear trade de rebalanceo"""
        trade = RebalanceTrade(
            symbol="BTC/USDT",
            side="buy",
            quantity=0.01,
            estimated_price=50000,
            estimated_value=500,
            estimated_commission=0.5,
        )

        assert trade.symbol == "BTC/USDT"
        assert trade.side == "buy"

    def test_total_cost(self):
        """Test costo total"""
        trade = RebalanceTrade(
            symbol="BTC/USDT",
            side="buy",
            quantity=0.01,
            estimated_price=50000,
            estimated_value=500,
            estimated_commission=0.5,
        )

        assert trade.total_cost == 500.5


class TestPortfolioMetrics:
    """Tests para PortfolioMetrics"""

    def test_create_metrics(self):
        """Test crear metricas"""
        metrics = PortfolioMetrics(
            total_return=1000,
            total_return_pct=10.0,
            sharpe_ratio=1.5,
        )

        assert metrics.total_return == 1000
        assert metrics.sharpe_ratio == 1.5

    def test_metrics_to_dict(self):
        """Test conversion a dict"""
        metrics = PortfolioMetrics(
            total_return=1000,
            sharpe_ratio=1.5,
        )

        d = metrics.to_dict()

        assert d["total_return"] == 1000
        assert d["sharpe_ratio"] == 1.5


class TestEnums:
    """Tests para enumeraciones"""

    def test_allocation_methods(self):
        """Test metodos de allocation"""
        assert AllocationMethod.EQUAL_WEIGHT.value == "equal_weight"
        assert AllocationMethod.RISK_PARITY.value == "risk_parity"
        assert AllocationMethod.MEAN_VARIANCE.value == "mean_variance"
        assert AllocationMethod.MIN_VARIANCE.value == "min_variance"
        assert AllocationMethod.MAX_SHARPE.value == "max_sharpe"

    def test_rebalance_frequencies(self):
        """Test frecuencias de rebalanceo"""
        assert RebalanceFrequency.DAILY.value == "daily"
        assert RebalanceFrequency.WEEKLY.value == "weekly"
        assert RebalanceFrequency.MONTHLY.value == "monthly"
        assert RebalanceFrequency.THRESHOLD.value == "threshold"

    def test_rebalance_reasons(self):
        """Test razones de rebalanceo"""
        assert RebalanceReason.SCHEDULED.value == "scheduled"
        assert RebalanceReason.THRESHOLD_BREACH.value == "threshold_breach"
