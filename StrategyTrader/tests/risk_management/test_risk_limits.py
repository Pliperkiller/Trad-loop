"""
Tests para risk_limits.py
"""

import pytest
import sys
from pathlib import Path

src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from risk_management.risk_limits import (
    Position,
    ExposureLimitManager,
    DrawdownProtection,
    VaRCalculator,
    RiskLimitChecker,
)
from risk_management.config import (
    ExposureLimitsConfig,
    DrawdownConfig,
    VaRConfig,
)
from risk_management.models import DrawdownAction, RiskLevel


class TestPosition:
    """Tests para Position dataclass"""

    def test_create_position(self):
        """Test crear posicion"""
        pos = Position(
            symbol="BTC/USDT",
            side="long",
            size=0.1,
            entry_price=50000,
        )

        assert pos.symbol == "BTC/USDT"
        assert pos.side == "long"
        assert pos.size == 0.1
        assert pos.entry_price == 50000

    def test_position_value(self):
        """Test valor de posicion"""
        pos = Position(
            symbol="BTC/USDT",
            side="long",
            size=0.1,
            entry_price=50000,
            current_price=51000,
        )

        assert pos.value == 0.1 * 51000  # Usa current_price

    def test_position_value_no_current_price(self):
        """Test valor sin current_price usa entry_price"""
        pos = Position(
            symbol="BTC/USDT",
            side="long",
            size=0.1,
            entry_price=50000,
        )

        assert pos.value == 0.1 * 50000

    def test_position_pnl_long(self):
        """Test PnL para posicion long"""
        pos = Position(
            symbol="BTC/USDT",
            side="long",
            size=0.1,
            entry_price=50000,
            current_price=51000,
        )

        expected_pnl = (51000 - 50000) * 0.1  # $100
        assert pos.pnl == expected_pnl

    def test_position_pnl_short(self):
        """Test PnL para posicion short"""
        pos = Position(
            symbol="BTC/USDT",
            side="short",
            size=0.1,
            entry_price=50000,
            current_price=49000,
        )

        expected_pnl = (50000 - 49000) * 0.1  # $100
        assert pos.pnl == expected_pnl


class TestExposureLimitManager:
    """Tests para ExposureLimitManager"""

    @pytest.fixture
    def manager(self):
        """Manager con config por defecto"""
        config = ExposureLimitsConfig()
        return ExposureLimitManager(config, total_capital=10000)

    def test_create_manager(self, manager):
        """Test crear manager"""
        assert manager.total_capital == 10000
        assert len(manager.limits) > 0

    def test_get_exposure_by_symbol(self, manager, sample_positions):
        """Test exposicion por simbolo"""
        manager.update_positions(sample_positions)

        exposure = manager.get_exposure_by_symbol()

        assert "BTC/USDT" in exposure
        assert "ETH/USDT" in exposure
        # BTC: 0.04 * 51000 / 10000 = 0.204
        assert exposure["BTC/USDT"] == pytest.approx(0.204, rel=0.01)

    def test_get_total_exposure(self, manager, sample_positions):
        """Test exposicion total"""
        manager.update_positions(sample_positions)

        total = manager.get_total_exposure()

        # BTC: 0.04 * 51000 = 2040
        # ETH: 0.5 * 3100 = 1550
        # Total: 3590 / 10000 = 0.359
        assert total == pytest.approx(0.359, rel=0.01)

    def test_get_long_short_exposure(self, manager):
        """Test exposicion long vs short"""
        positions = [
            Position("BTC/USDT", "long", 0.1, 50000, 50000),
            Position("ETH/USDT", "short", 1.0, 3000, 3000),
        ]
        manager.update_positions(positions)

        long_exp = manager.get_long_exposure()
        short_exp = manager.get_short_exposure()

        assert long_exp == pytest.approx(0.5, rel=0.01)  # 5000/10000
        assert short_exp == pytest.approx(0.3, rel=0.01)  # 3000/10000

    def test_check_limits_no_breach(self, manager):
        """Test verificar limites sin exceder"""
        positions = [
            Position("BTC/USDT", "long", 0.02, 50000, 50000),  # $1000 = 10%
        ]
        manager.update_positions(positions)

        breached = manager.check_limits()
        assert len(breached) == 0

    def test_check_limits_with_breach(self, manager):
        """Test verificar limites excedidos"""
        positions = [
            Position("BTC/USDT", "long", 0.06, 50000, 50000),  # $3000 = 30%
        ]
        manager.update_positions(positions)

        breached = manager.check_limits()
        assert len(breached) > 0
        assert any("Exposure" in l.name for l in breached)

    def test_can_open_position_allowed(self, manager):
        """Test que permite abrir posicion valida"""
        can_open, reasons = manager.can_open_position(
            symbol="BTC/USDT",
            side="long",
            size=0.02,  # $1000 = 10%
            price=50000,
        )

        assert can_open == True
        assert len(reasons) == 0

    def test_can_open_position_denied(self, manager):
        """Test que rechaza posicion muy grande"""
        can_open, reasons = manager.can_open_position(
            symbol="BTC/USDT",
            side="long",
            size=0.1,  # $5000 = 50%
            price=50000,
        )

        assert can_open == False
        assert len(reasons) > 0

    def test_get_max_allowed_size(self, manager):
        """Test obtener tamano maximo permitido"""
        max_size = manager.get_max_allowed_size(
            symbol="BTC/USDT",
            side="long",
            price=50000,
        )

        # Max single asset = 25% = $2500 = 0.05 BTC
        assert max_size == pytest.approx(0.05, rel=0.01)


class TestDrawdownProtection:
    """Tests para DrawdownProtection"""

    @pytest.fixture
    def protection(self):
        """Proteccion con config por defecto"""
        config = DrawdownConfig()
        return DrawdownProtection(config, initial_capital=10000)

    def test_create_protection(self, protection):
        """Test crear proteccion"""
        assert protection.state.peak_equity == 10000
        assert protection.state.current_equity == 10000

    def test_update_no_drawdown(self, protection):
        """Test actualizar sin drawdown"""
        action = protection.update(10500)

        assert action == DrawdownAction.NONE
        assert protection.state.peak_equity == 10500
        assert protection.state.drawdown_percent == 0

    def test_update_with_drawdown(self, protection):
        """Test actualizar con drawdown"""
        protection.update(10000)  # Set peak
        action = protection.update(9500)  # 5% drawdown

        assert protection.state.drawdown_percent == pytest.approx(5.0, rel=0.1)
        assert protection.get_risk_level() == RiskLevel.LOW

    def test_drawdown_warning_level(self, protection):
        """Test nivel de warning"""
        protection.update(10000)
        protection.update(9500)  # 5% drawdown

        level = protection.get_risk_level()
        assert level == RiskLevel.LOW

    def test_drawdown_caution_level(self, protection):
        """Test nivel de caution"""
        protection.update(10000)
        protection.update(9000)  # 10% drawdown

        level = protection.get_risk_level()
        assert level == RiskLevel.MODERATE

    def test_drawdown_danger_level(self, protection):
        """Test nivel de danger"""
        protection.update(10000)
        protection.update(8500)  # 15% drawdown

        level = protection.get_risk_level()
        assert level == RiskLevel.HIGH

    def test_drawdown_critical_level(self, protection):
        """Test nivel critical"""
        protection.update(10000)
        protection.update(8000)  # 20% drawdown

        level = protection.get_risk_level()
        assert level == RiskLevel.EXTREME

    def test_trading_allowed(self, protection):
        """Test si trading esta permitido"""
        protection.update(10000)
        protection.update(9000)  # 10% drawdown

        assert protection.is_trading_allowed() == True

    def test_trading_paused(self, protection):
        """Test trading pausado por drawdown critico"""
        protection.update(10000)
        protection.update(8000)  # 20% drawdown

        assert protection.is_trading_allowed() == False

    def test_size_multiplier(self, protection):
        """Test multiplicador de tamano"""
        protection.update(10000)

        # Sin drawdown
        assert protection.get_size_multiplier() == 1.0

        # 10% drawdown - nivel caution
        protection.update(9000)
        assert protection.get_size_multiplier() == 0.75

        # 15% drawdown - nivel danger
        protection.update(8500)
        assert protection.get_size_multiplier() == 0.50

    def test_reset(self, protection):
        """Test reiniciar proteccion"""
        protection.update(10000)
        protection.update(8000)  # Crear drawdown

        protection.reset(10000)

        assert protection.state.peak_equity == 10000
        assert protection.state.drawdown_percent == 0


class TestVaRCalculator:
    """Tests para VaRCalculator"""

    @pytest.fixture
    def calculator(self):
        """Calculator con config por defecto"""
        config = VaRConfig()
        return VaRCalculator(config)

    def test_create_calculator(self, calculator):
        """Test crear calculator"""
        assert calculator is not None
        assert len(calculator.returns_history) == 0

    def test_var_insufficient_data(self, calculator):
        """Test VaR con datos insuficientes"""
        calculator.returns_history = [0.01, 0.02, -0.01]

        var = calculator.calculate_var(0.95)
        assert var == 0.0

    def test_var_with_data(self, calculator, var_returns):
        """Test VaR con datos suficientes"""
        calculator.update_returns(var_returns)

        var_95 = calculator.calculate_var(0.95)

        # VaR deberia ser negativo (perdida)
        assert var_95 < 0

    def test_cvar(self, calculator, var_returns):
        """Test CVaR"""
        calculator.update_returns(var_returns)

        var_95 = calculator.calculate_var(0.95)
        cvar_95 = calculator.calculate_cvar(0.95)

        # CVaR deberia ser peor (mas negativo) que VaR
        assert cvar_95 <= var_95

    def test_get_var_metrics(self, calculator, var_returns):
        """Test obtener metricas de VaR"""
        calculator.update_returns(var_returns)

        metrics = calculator.get_var_metrics()

        assert "var_95" in metrics
        assert "var_99" in metrics
        assert "cvar_95" in metrics
        assert "cvar_99" in metrics

    def test_check_var_limits(self, calculator, var_returns):
        """Test verificar limites de VaR"""
        calculator.update_returns(var_returns)

        breached = calculator.check_var_limits()
        # Depende de los datos, puede o no haber brechas
        assert isinstance(breached, list)


class TestRiskLimitChecker:
    """Tests para RiskLimitChecker"""

    @pytest.fixture
    def checker(self):
        """Checker con configs por defecto"""
        return RiskLimitChecker(
            ExposureLimitsConfig(),
            DrawdownConfig(),
            VaRConfig(),
            initial_capital=10000,
        )

    def test_create_checker(self, checker):
        """Test crear checker"""
        assert checker is not None
        assert checker.initial_capital == 10000

    def test_update(self, checker, sample_positions):
        """Test actualizar checker"""
        checker.update(
            equity=10500,
            positions=sample_positions,
            daily_return=0.05,
        )

        # Verificar que se actualizo correctamente
        assert checker.drawdown_protection.state.current_equity == 10500

    def test_check_all_limits(self, checker, sample_positions):
        """Test verificar todos los limites"""
        checker.update(10000, sample_positions)

        metrics = checker.check_all_limits()

        assert metrics.total_exposure >= 0
        assert metrics.current_drawdown >= 0
        assert metrics.risk_level is not None

    def test_can_open_position(self, checker):
        """Test verificar si puede abrir posicion"""
        can_open, reasons = checker.can_open_position(
            symbol="BTC/USDT",
            side="long",
            size=0.02,
            price=50000,
        )

        assert can_open == True

    def test_get_adjusted_size(self, checker):
        """Test obtener tamano ajustado"""
        # Simular drawdown
        checker.drawdown_protection.update(10000)
        checker.drawdown_protection.update(9000)  # 10% DD

        adjusted = checker.get_adjusted_size(
            symbol="BTC/USDT",
            side="long",
            requested_size=0.1,
            price=50000,
        )

        # Deberia ser menor que el solicitado por el DD
        assert adjusted < 0.1

    def test_reset(self, checker):
        """Test reiniciar checker"""
        checker.drawdown_protection.update(10000)
        checker.drawdown_protection.update(8000)

        checker.reset(10000)

        assert checker.drawdown_protection.state.drawdown_percent == 0
