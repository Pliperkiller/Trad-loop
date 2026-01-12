"""
Tests para portfolio/allocator.py
"""

import pytest
import sys
from pathlib import Path
import numpy as np

src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from portfolio.allocator import (
    PortfolioAllocator,
    AllocationResult,
    PortfolioAllocatorFactory,
)
from portfolio.models import AllocationMethod


class TestPortfolioAllocator:
    """Tests para PortfolioAllocator"""

    @pytest.fixture
    def allocator(self):
        """Allocator por defecto"""
        return PortfolioAllocator(risk_free_rate=0.02)

    def test_create_allocator(self, allocator):
        """Test crear allocator"""
        assert allocator is not None
        assert allocator.risk_free_rate == 0.02

    def test_equal_weight(self, allocator, sample_returns, symbols):
        """Test equal weight allocation"""
        result = allocator.calculate_weights(
            returns=sample_returns.values,
            symbols=symbols,
            method=AllocationMethod.EQUAL_WEIGHT,
        )

        assert result.success == True
        assert len(result.weights) == len(symbols)

        # Verificar que todos los pesos son iguales
        expected_weight = 1.0 / len(symbols)
        for weight in result.weights.values():
            assert weight == pytest.approx(expected_weight, rel=0.01)

    def test_risk_parity(self, allocator, sample_returns, symbols):
        """Test risk parity allocation"""
        result = allocator.calculate_weights(
            returns=sample_returns.values,
            symbols=symbols,
            method=AllocationMethod.RISK_PARITY,
        )

        assert result.success == True
        assert len(result.weights) == len(symbols)

        # Verificar que suman 1
        total = sum(result.weights.values())
        assert total == pytest.approx(1.0, rel=0.01)

        # Activos mas volatiles deberian tener menor peso
        # SOL es mas volatil, deberia tener menor peso que BTC
        # Pero esto depende de los datos

    def test_min_variance(self, allocator, sample_returns, symbols):
        """Test minimum variance allocation"""
        result = allocator.calculate_weights(
            returns=sample_returns.values,
            symbols=symbols,
            method=AllocationMethod.MIN_VARIANCE,
        )

        assert result.success == True
        assert len(result.weights) == len(symbols)
        assert result.expected_volatility >= 0

    def test_max_sharpe(self, allocator, sample_returns, symbols):
        """Test max Sharpe allocation"""
        result = allocator.calculate_weights(
            returns=sample_returns.values,
            symbols=symbols,
            method=AllocationMethod.MAX_SHARPE,
        )

        assert result.success == True
        assert len(result.weights) == len(symbols)

    def test_mean_variance_with_target(self, allocator, sample_returns, symbols):
        """Test mean-variance con retorno objetivo"""
        result = allocator.calculate_weights(
            returns=sample_returns.values,
            symbols=symbols,
            method=AllocationMethod.MEAN_VARIANCE,
            target_return=0.10,  # 10% anual
        )

        assert len(result.weights) == len(symbols)

    def test_custom_weights(self, allocator, symbols):
        """Test pesos personalizados"""
        custom = {
            "BTC/USDT": 0.5,
            "ETH/USDT": 0.3,
            "SOL/USDT": 0.15,
            "LINK/USDT": 0.05,
        }

        result = allocator.calculate_weights(
            returns=np.random.randn(100, 4),
            symbols=symbols,
            method=AllocationMethod.CUSTOM,
            target_weights=custom,
        )

        assert result.success == True
        assert result.weights["BTC/USDT"] == pytest.approx(0.5, rel=0.01)

    def test_single_asset(self, allocator):
        """Test con un solo activo"""
        result = allocator.calculate_weights(
            returns=np.random.randn(100, 1),
            symbols=["BTC/USDT"],
            method=AllocationMethod.EQUAL_WEIGHT,
        )

        assert result.success == True
        assert result.weights["BTC/USDT"] == 1.0

    def test_insufficient_data(self, allocator, symbols):
        """Test con datos insuficientes"""
        result = allocator.calculate_weights(
            returns=np.random.randn(1, 4),  # Solo 1 periodo
            symbols=symbols,
            method=AllocationMethod.RISK_PARITY,
        )

        # Deberia caer a equal weight
        assert result.success == False

    def test_empty_symbols(self, allocator):
        """Test sin simbolos"""
        result = allocator.calculate_weights(
            returns=np.array([]),
            symbols=[],
            method=AllocationMethod.EQUAL_WEIGHT,
        )

        assert result.success == False
        assert len(result.weights) == 0

    def test_weights_sum_to_one(self, allocator, sample_returns, symbols):
        """Test que todos los metodos dan pesos que suman 1"""
        methods = [
            AllocationMethod.EQUAL_WEIGHT,
            AllocationMethod.RISK_PARITY,
            AllocationMethod.MIN_VARIANCE,
            AllocationMethod.MAX_SHARPE,
        ]

        for method in methods:
            result = allocator.calculate_weights(
                returns=sample_returns.values,
                symbols=symbols,
                method=method,
            )

            if result.success:
                total = sum(result.weights.values())
                assert total == pytest.approx(1.0, rel=0.01), f"Failed for {method}"

    def test_efficient_frontier(self, allocator, sample_returns, symbols):
        """Test frontera eficiente"""
        frontier = allocator.get_efficient_frontier(
            returns=sample_returns.values,
            symbols=symbols,
            n_points=10,
        )

        assert len(frontier) > 0

        for ret, vol, weights in frontier:
            # Verificar que retorno y vol son validos
            assert isinstance(ret, float)
            assert isinstance(vol, float)
            assert vol >= 0

            # Verificar que pesos suman 1
            total = sum(weights.values())
            assert total == pytest.approx(1.0, rel=0.02)


class TestPortfolioAllocatorFactory:
    """Tests para PortfolioAllocatorFactory"""

    def test_create_conservative(self):
        """Test crear allocator conservador"""
        allocator = PortfolioAllocatorFactory.create_conservative()

        assert allocator.min_weight == 0.05
        assert allocator.max_weight == 0.30

    def test_create_moderate(self):
        """Test crear allocator moderado"""
        allocator = PortfolioAllocatorFactory.create_moderate()

        assert allocator.min_weight == 0.02
        assert allocator.max_weight == 0.50

    def test_create_aggressive(self):
        """Test crear allocator agresivo"""
        allocator = PortfolioAllocatorFactory.create_aggressive()

        assert allocator.min_weight == 0.0
        assert allocator.max_weight == 1.0


class TestAllocationResult:
    """Tests para AllocationResult"""

    def test_create_result(self):
        """Test crear resultado"""
        result = AllocationResult(
            weights={"BTC/USDT": 0.5, "ETH/USDT": 0.5},
            method=AllocationMethod.EQUAL_WEIGHT,
            expected_return=0.10,
            expected_volatility=0.20,
            sharpe_ratio=0.5,
            success=True,
        )

        assert result.weights["BTC/USDT"] == 0.5
        assert result.sharpe_ratio == 0.5

    def test_result_to_dict(self):
        """Test conversion a dict"""
        result = AllocationResult(
            weights={"BTC/USDT": 1.0},
            method=AllocationMethod.EQUAL_WEIGHT,
            success=True,
        )

        d = result.to_dict()

        assert d["method"] == "equal_weight"
        assert d["success"] == True
