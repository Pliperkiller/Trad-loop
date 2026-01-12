"""
Tests para risk_manager.py
"""

import pytest
import sys
from pathlib import Path

src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from risk_management.risk_manager import (
    RiskManager,
    RiskManagerFactory,
    RiskManagerState,
)
from risk_management.risk_limits import Position
from risk_management.config import RiskManagementConfig
from risk_management.models import RiskLevel


class TestRiskManager:
    """Tests para RiskManager"""

    @pytest.fixture
    def manager(self, default_config):
        """Manager con config por defecto"""
        return RiskManager(default_config)

    @pytest.fixture
    def conservative_manager(self, conservative_config):
        """Manager conservador"""
        return RiskManager(conservative_config)

    def test_create_manager(self, manager):
        """Test crear risk manager"""
        assert manager is not None
        assert manager.current_capital == 10000
        assert len(manager.positions) == 0
        assert manager.state.is_active == True

    def test_assess_trade_approved(self, manager):
        """Test evaluar trade aprobado"""
        assessment = manager.assess_trade(
            symbol="BTC/USDT",
            side="long",
            proposed_size=0.02,  # $1000
            entry_price=50000,
        )

        assert assessment.is_approved == True
        assert assessment.approved_size > 0
        assert assessment.risk_score >= 0

    def test_assess_trade_rejected_max_positions(self, manager):
        """Test rechazo por maximo de posiciones"""
        # Agregar 5 posiciones (el maximo)
        for i in range(5):
            manager.positions.append(
                Position(f"SYMBOL{i}", "long", 0.01, 100, 100)
            )

        assessment = manager.assess_trade(
            symbol="NEW/USDT",
            side="long",
            proposed_size=0.01,
            entry_price=100,
        )

        assert assessment.is_approved == False
        assert "maximo" in assessment.rejection_reasons[0].lower()

    def test_assess_trade_rejected_exposure(self, conservative_manager):
        """Test rechazo por exposicion excesiva"""
        assessment = conservative_manager.assess_trade(
            symbol="BTC/USDT",
            side="long",
            proposed_size=0.1,  # $5000 = 50% del capital
            entry_price=50000,
        )

        # Con config conservadora, max single asset es 15%
        # 50% excede el limite
        assert assessment.is_approved == False

    def test_assess_trade_with_stop_loss(self, manager):
        """Test evaluar trade con stop loss"""
        assessment = manager.assess_trade(
            symbol="BTC/USDT",
            side="long",
            proposed_size=0.02,
            entry_price=50000,
            stop_loss=48000,
        )

        assert assessment.stop_loss == 48000
        assert assessment.is_approved == True

    def test_assess_trade_adjustments(self, manager):
        """Test que se aplican ajustes correctamente"""
        # Crear drawdown para forzar ajuste
        manager.limit_checker.drawdown_protection.update(10000)
        manager.limit_checker.drawdown_protection.update(9000)  # 10% DD

        assessment = manager.assess_trade(
            symbol="BTC/USDT",
            side="long",
            proposed_size=0.05,
            entry_price=50000,
        )

        # Deberia tener ajuste por drawdown
        if assessment.is_approved:
            assert "drawdown" in assessment.adjustments_made or assessment.approved_size < 0.05

    def test_get_optimal_position_size(self, manager):
        """Test obtener tamano optimo"""
        result = manager.get_optimal_position_size(
            symbol="BTC/USDT",
            entry_price=50000,
        )

        assert result.recommended_size > 0
        assert result.symbol == "BTC/USDT"

    def test_update(self, manager, sample_positions):
        """Test actualizar estado"""
        manager.update(
            equity=10500,
            positions=sample_positions,
            daily_return=0.05,
        )

        assert manager.current_capital == 10500
        assert len(manager.positions) == len(sample_positions)
        assert manager.state.last_update is not None

    def test_record_trade_result(self, manager):
        """Test registrar resultado de trade"""
        manager.record_trade_result(0.05)  # 5% ganancia
        manager.record_trade_result(-0.02)  # 2% perdida

        assert len(manager.trade_history.returns) == 2
        assert manager.trade_history.wins == 1
        assert manager.trade_history.losses == 1

    def test_get_risk_metrics(self, manager, sample_positions):
        """Test obtener metricas de riesgo"""
        manager.update(10000, sample_positions)

        metrics = manager.get_risk_metrics()

        assert metrics.total_exposure >= 0
        assert metrics.risk_level is not None

    def test_get_drawdown_state(self, manager):
        """Test obtener estado de drawdown"""
        manager.update(10000, [])
        manager.update(9500, [])

        dd_state = manager.get_drawdown_state()

        assert dd_state.peak_equity == 10000
        assert dd_state.current_equity == 9500
        assert dd_state.drawdown_percent == pytest.approx(5.0, rel=0.1)

    def test_is_trading_allowed(self, manager):
        """Test si trading esta permitido"""
        assert manager.is_trading_allowed() == True

        # Crear drawdown critico
        manager.update(10000, [])
        manager.update(8000, [])  # 20% DD

        assert manager.is_trading_allowed() == False

    def test_get_current_risk_level(self, manager):
        """Test obtener nivel de riesgo"""
        manager.update(10000, [])
        assert manager.get_current_risk_level() == RiskLevel.MINIMAL

        manager.update(9000, [])  # 10% DD
        assert manager.get_current_risk_level() == RiskLevel.MODERATE

    def test_should_close_all_positions(self, manager):
        """Test si debe cerrar todas las posiciones"""
        manager.update(10000, [])
        assert manager.should_close_all_positions() == False

    def test_get_position_summary(self, manager, sample_positions):
        """Test obtener resumen de posiciones"""
        manager.update(10000, sample_positions)

        summary = manager.get_position_summary()

        assert summary["total_positions"] == 2
        assert summary["long_positions"] == 2
        assert summary["short_positions"] == 0
        assert summary["total_exposure"] > 0

    def test_get_status(self, manager, sample_positions):
        """Test obtener estado completo"""
        manager.update(10000, sample_positions)

        status = manager.get_status()

        assert "is_active" in status
        assert "is_trading_allowed" in status
        assert "risk_level" in status
        assert "capital" in status
        assert "positions" in status
        assert "drawdown" in status
        assert "exposure" in status
        assert "correlation" in status

    def test_reset(self, manager, sample_positions):
        """Test reiniciar manager"""
        manager.update(10500, sample_positions)
        manager.record_trade_result(0.05)

        manager.reset()

        assert manager.current_capital == 10000  # Vuelve al inicial
        assert len(manager.positions) == 0
        assert len(manager.trade_history.returns) == 0

    def test_reset_with_new_capital(self, manager):
        """Test reiniciar con nuevo capital"""
        manager.reset(new_capital=20000)

        assert manager.current_capital == 20000

    def test_trading_paused_rejects_trades(self, manager):
        """Test que trades se rechazan cuando trading esta pausado"""
        # Forzar pausa
        manager.update(10000, [])
        manager.update(8000, [])  # 20% DD critico

        assessment = manager.assess_trade(
            symbol="BTC/USDT",
            side="long",
            proposed_size=0.01,
            entry_price=50000,
        )

        assert assessment.is_approved == False
        assert "pausado" in assessment.rejection_reasons[0].lower()

    def test_update_correlation_data(self, manager):
        """Test actualizar datos de correlacion"""
        returns = [0.01, 0.02, -0.01, 0.015] * 10

        manager.update_correlation_data("BTC/USDT", returns)

        assert "BTC/USDT" in manager.correlation_manager.returns_data

    def test_add_daily_return(self, manager):
        """Test agregar retorno diario"""
        manager.add_daily_return("BTC/USDT", 0.02)
        manager.add_daily_return("BTC/USDT", -0.01)

        assert len(manager.correlation_manager.returns_data["BTC/USDT"].returns) == 2

    def test_refresh_correlation_matrix(self, manager):
        """Test refrescar matriz de correlacion"""
        # Agregar datos
        for symbol in ["BTC/USDT", "ETH/USDT"]:
            manager.update_correlation_data(symbol, [0.01, 0.02, -0.01] * 10)

        manager.refresh_correlation_matrix()

        assert manager.correlation_manager.last_matrix_update is not None


class TestRiskManagerFactory:
    """Tests para RiskManagerFactory"""

    def test_create_conservative(self):
        """Test crear manager conservador"""
        manager = RiskManagerFactory.create_conservative(initial_capital=5000)

        assert manager.current_capital == 5000
        assert manager.config.position_sizing.max_position_percent == 0.10

    def test_create_moderate(self):
        """Test crear manager moderado"""
        manager = RiskManagerFactory.create_moderate(initial_capital=10000)

        assert manager.current_capital == 10000
        assert manager.config.position_sizing.max_position_percent == 0.20

    def test_create_aggressive(self):
        """Test crear manager agresivo"""
        manager = RiskManagerFactory.create_aggressive(initial_capital=15000)

        assert manager.current_capital == 15000
        assert manager.config.position_sizing.max_position_percent == 0.30


class TestRiskManagerState:
    """Tests para RiskManagerState"""

    @pytest.fixture
    def manager(self, default_config):
        """Manager con config por defecto"""
        return RiskManager(default_config)

    def test_create_state(self):
        """Test crear estado"""
        state = RiskManagerState()

        assert state.is_active == True
        assert state.is_trading_allowed == True
        assert state.current_risk_level == RiskLevel.MINIMAL
        assert state.trade_count == 0
        assert state.rejected_count == 0
        assert len(state.alerts) == 0

    def test_state_updates(self, manager):
        """Test que estado se actualiza correctamente"""
        initial_trade_count = manager.state.trade_count

        # Trade exitoso
        manager.assess_trade(
            symbol="BTC/USDT",
            side="long",
            proposed_size=0.01,
            entry_price=50000,
        )

        assert manager.state.trade_count == initial_trade_count + 1


class TestRiskManagerIntegration:
    """Tests de integracion del RiskManager"""

    @pytest.fixture
    def manager(self, default_config):
        """Manager con config por defecto"""
        return RiskManager(default_config)

    def test_full_trade_cycle(self, manager, sample_returns_data):
        """Test ciclo completo de trading"""
        # 1. Configurar datos de correlacion
        for symbol, returns in sample_returns_data.items():
            manager.update_correlation_data(symbol, returns)
        manager.refresh_correlation_matrix()

        # 2. Evaluar primer trade
        assessment1 = manager.assess_trade(
            symbol="BTC/USDT",
            side="long",
            proposed_size=0.02,
            entry_price=50000,
            stop_loss=48000,
        )
        assert assessment1.is_approved == True

        # 3. Actualizar posiciones
        positions = [
            Position("BTC/USDT", "long", assessment1.approved_size, 50000, 50000)
        ]
        manager.update(10000, positions)

        # 4. Evaluar segundo trade (correlacionado)
        assessment2 = manager.assess_trade(
            symbol="ETH/USDT",
            side="long",
            proposed_size=0.3,
            entry_price=3000,
        )
        # Puede haber penalizacion por correlacion
        assert assessment2.approved_size <= 0.3 or not assessment2.is_approved

        # 5. Simular ganancia
        new_equity = 10200
        manager.update(new_equity, positions)

        # 6. Registrar resultado
        manager.record_trade_result(0.02)  # 2% ganancia

        # 7. Verificar estado
        status = manager.get_status()
        assert status["capital"] == new_equity
        assert status["trade_count"] == 2  # 2 trades evaluados

    def test_drawdown_response(self, manager):
        """Test respuesta a drawdown progresivo"""
        # Simular drawdown progresivo
        equities = [10000, 9500, 9000, 8500, 8000]
        risk_levels = []

        for equity in equities:
            manager.update(equity, [])
            risk_levels.append(manager.get_current_risk_level())

        # Los niveles deberian aumentar
        assert risk_levels[0] == RiskLevel.MINIMAL
        assert risk_levels[-1] == RiskLevel.EXTREME

        # Trading deberia pausarse al final
        assert manager.is_trading_allowed() == False

    def test_size_reduction_under_stress(self, manager):
        """Test reduccion de tamano bajo estres"""
        # Sin drawdown
        assessment1 = manager.assess_trade(
            symbol="BTC/USDT",
            side="long",
            proposed_size=0.04,
            entry_price=50000,
        )
        size_normal = assessment1.approved_size

        # Con drawdown del 10%
        manager.update(10000, [])
        manager.update(9000, [])

        assessment2 = manager.assess_trade(
            symbol="BTC/USDT",
            side="long",
            proposed_size=0.04,
            entry_price=50000,
        )

        if assessment2.is_approved:
            size_stressed = assessment2.approved_size
            # El tamano bajo estres deberia ser menor
            assert size_stressed < size_normal
