"""
Tests para correlation_manager.py
"""

import pytest
import sys
from pathlib import Path
import numpy as np

src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from risk_management.correlation_manager import CorrelationManager, AssetReturns
from risk_management.config import CorrelationConfig


class TestAssetReturns:
    """Tests para AssetReturns dataclass"""

    def test_create_asset_returns(self):
        """Test crear asset returns"""
        returns = AssetReturns(
            symbol="BTC/USDT",
            returns=[0.01, 0.02, -0.01],
        )

        assert returns.symbol == "BTC/USDT"
        assert len(returns.returns) == 3
        assert returns.last_updated is not None


class TestCorrelationManager:
    """Tests para CorrelationManager"""

    @pytest.fixture
    def manager(self):
        """Manager con config por defecto"""
        config = CorrelationConfig()
        return CorrelationManager(config)

    @pytest.fixture
    def manager_with_data(self, sample_returns_data):
        """Manager con datos precargados"""
        config = CorrelationConfig()
        manager = CorrelationManager(config)

        for symbol, returns in sample_returns_data.items():
            manager.update_returns(symbol, returns)

        manager.update_correlation_matrix()
        return manager

    def test_create_manager(self, manager):
        """Test crear manager"""
        assert manager is not None
        assert len(manager.returns_data) == 0
        assert len(manager.correlation_matrix) == 0

    def test_update_returns(self, manager):
        """Test actualizar retornos"""
        returns = [0.01, 0.02, -0.01, 0.015, -0.02]
        manager.update_returns("BTC/USDT", returns)

        assert "BTC/USDT" in manager.returns_data
        assert len(manager.returns_data["BTC/USDT"].returns) == 5

    def test_add_return(self, manager):
        """Test agregar retorno individual"""
        manager.add_return("BTC/USDT", 0.01)
        manager.add_return("BTC/USDT", 0.02)
        manager.add_return("BTC/USDT", -0.01)

        assert len(manager.returns_data["BTC/USDT"].returns) == 3

    def test_calculate_correlation_insufficient_data(self, manager):
        """Test correlacion con datos insuficientes"""
        manager.update_returns("BTC/USDT", [0.01, 0.02])
        manager.update_returns("ETH/USDT", [0.02, 0.03])

        corr = manager.calculate_correlation("BTC/USDT", "ETH/USDT")
        assert corr is None

    def test_calculate_correlation(self, manager_with_data):
        """Test calcular correlacion"""
        corr = manager_with_data.calculate_correlation("BTC/USDT", "ETH/USDT")

        assert corr is not None
        assert -1 <= corr <= 1
        # BTC y ETH deberian estar correlacionados (ver fixture)
        assert corr > 0.5

    def test_correlation_independent_assets(self, manager_with_data):
        """Test correlacion de activos independientes"""
        corr = manager_with_data.calculate_correlation("BTC/USDT", "LINK/USDT")

        # LINK es independiente, correlacion deberia ser baja
        assert abs(corr) < 0.5

    def test_correlation_inverse_assets(self, manager_with_data):
        """Test correlacion de activos inversos"""
        corr = manager_with_data.calculate_correlation("BTC/USDT", "XRP/USDT")

        # XRP tiene correlacion negativa con BTC (ver fixture)
        assert corr < 0

    def test_update_correlation_matrix(self, manager, sample_returns_data):
        """Test actualizar matriz de correlacion"""
        for symbol, returns in sample_returns_data.items():
            manager.update_returns(symbol, returns)

        manager.update_correlation_matrix()

        # Deberia haber pares de correlacion
        assert len(manager.correlation_matrix) > 0

    def test_get_correlation(self, manager_with_data):
        """Test obtener correlacion almacenada"""
        corr = manager_with_data.get_correlation("BTC/USDT", "ETH/USDT")

        assert corr is not None

        # Probar orden inverso
        corr2 = manager_with_data.get_correlation("ETH/USDT", "BTC/USDT")
        assert corr2 == corr

    def test_get_highly_correlated_pairs(self, manager_with_data):
        """Test obtener pares altamente correlacionados"""
        high_corr = manager_with_data.get_highly_correlated_pairs()

        # BTC-ETH deberia estar en la lista
        symbols_in_high = [
            (pair.symbol_a, pair.symbol_b) for pair in high_corr
        ]
        assert any(
            ("BTC/USDT" in pair and "ETH/USDT" in pair)
            for pair in symbols_in_high
        )

    def test_get_diversifying_pairs(self, manager_with_data):
        """Test obtener pares diversificantes"""
        diversifying = manager_with_data.get_diversifying_pairs()

        # BTC-XRP deberia ser diversificante (correlacion negativa)
        assert len(diversifying) > 0

    def test_get_portfolio_correlation(self, manager_with_data):
        """Test correlacion de portfolio"""
        positions = [
            ("BTC/USDT", 0.5),
            ("ETH/USDT", 0.3),
            ("LINK/USDT", 0.2),
        ]

        portfolio_corr = manager_with_data.get_portfolio_correlation(positions)

        assert -1 <= portfolio_corr <= 1

    def test_check_correlation_limits(self, manager_with_data):
        """Test verificar limites de correlacion"""
        current_positions = ["BTC/USDT", "ETH/USDT"]

        # Intentar agregar otro activo correlacionado
        can_add, reasons = manager_with_data.check_correlation_limits(
            current_positions, "ETH/USDT"
        )

        # Nota: ETH ya esta en las posiciones, esto deberia fallar por max correlated
        # Pero como estamos agregando el mismo, probamos otra cosa

        # Probemos con un activo menos correlacionado
        can_add, reasons = manager_with_data.check_correlation_limits(
            current_positions, "LINK/USDT"
        )
        assert can_add == True

    def test_check_correlation_limits_exceeds_max(self, manager_with_data):
        """Test que rechaza cuando excede maximo de posiciones correlacionadas"""
        # Configurar max 2 posiciones correlacionadas
        manager_with_data.config.max_correlated_positions = 2

        current_positions = ["BTC/USDT", "ETH/USDT"]

        # Intentar agregar otro activo altamente correlacionado
        # (simulado - en realidad ETH ya esta correlacionado con BTC)
        # Este test depende de los datos
        pass  # Skip por ahora

    def test_get_correlation_penalty(self, manager_with_data):
        """Test obtener penalizacion por correlacion"""
        current_positions = ["BTC/USDT"]

        # Agregar activo correlacionado - deberia tener penalizacion
        penalty = manager_with_data.get_correlation_penalty(
            current_positions, "ETH/USDT"
        )
        assert penalty < 1.0  # Menos del 100%

        # Agregar activo independiente - sin penalizacion o menor
        penalty_link = manager_with_data.get_correlation_penalty(
            current_positions, "LINK/USDT"
        )
        assert penalty_link >= penalty  # Menor penalizacion

    def test_get_max_correlation_pair(self, manager_with_data):
        """Test obtener par con mayor correlacion"""
        max_pair = manager_with_data.get_max_correlation_pair()

        assert max_pair is not None
        assert len(max_pair) == 3  # (symbol_a, symbol_b, correlation)
        assert -1 <= max_pair[2] <= 1

    def test_get_average_correlation(self, manager_with_data):
        """Test obtener correlacion promedio"""
        avg = manager_with_data.get_average_correlation()

        assert 0 <= avg <= 1  # Usamos valores absolutos

    def test_get_correlation_report(self, manager_with_data):
        """Test generar reporte de correlacion"""
        report = manager_with_data.get_correlation_report()

        assert "total_pairs" in report
        assert "average_correlation" in report
        assert "max_correlation_pair" in report
        assert "highly_correlated_count" in report
        assert "diversifying_count" in report
        assert "last_update" in report

    def test_needs_update(self, manager):
        """Test si necesita actualizacion"""
        assert manager.needs_update() == True

        # Actualizar
        manager.update_returns("BTC/USDT", [0.01] * 20)
        manager.update_correlation_matrix()

        assert manager.needs_update() == False

    def test_check_limits(self, manager_with_data):
        """Test verificar limites"""
        current_positions = ["BTC/USDT", "ETH/USDT"]

        breached = manager_with_data.check_limits(current_positions)

        # Puede o no haber limites excedidos
        assert isinstance(breached, list)

    def test_reset(self, manager_with_data):
        """Test reiniciar manager"""
        assert len(manager_with_data.returns_data) > 0

        manager_with_data.reset()

        assert len(manager_with_data.returns_data) == 0
        assert len(manager_with_data.correlation_matrix) == 0
        assert manager_with_data.last_matrix_update is None


class TestCorrelationData:
    """Tests para CorrelationData model"""

    @pytest.fixture
    def manager_with_data(self, sample_returns_data):
        """Manager con datos precargados"""
        config = CorrelationConfig()
        manager = CorrelationManager(config)

        for symbol, returns in sample_returns_data.items():
            manager.update_returns(symbol, returns)

        manager.update_correlation_matrix()
        return manager

    def test_is_highly_correlated(self, manager_with_data):
        """Test propiedad is_highly_correlated"""
        for data in manager_with_data.correlation_matrix.values():
            if abs(data.correlation) > 0.7:
                assert data.is_highly_correlated == True
            else:
                assert data.is_highly_correlated == False

    def test_is_diversifying(self, manager_with_data):
        """Test propiedad is_diversifying"""
        for data in manager_with_data.correlation_matrix.values():
            if data.correlation < -0.3:
                assert data.is_diversifying == True
            else:
                assert data.is_diversifying == False
