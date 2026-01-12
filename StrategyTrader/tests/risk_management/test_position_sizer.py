"""
Tests para position_sizer.py
"""

import pytest
import sys
from pathlib import Path

src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from risk_management.position_sizer import (
    PositionSizer,
    PositionSizerFactory,
    TradeHistory,
)
from risk_management.config import PositionSizingConfig
from risk_management.models import SizingMethod


class TestPositionSizer:
    """Tests para PositionSizer"""

    @pytest.fixture
    def default_sizer(self):
        """Sizer con configuracion por defecto"""
        config = PositionSizingConfig()
        return PositionSizer(config)

    @pytest.fixture
    def kelly_sizer(self):
        """Sizer con Kelly Criterion"""
        config = PositionSizingConfig(
            method=SizingMethod.KELLY,
            kelly_fraction=0.5,
            min_trades_for_kelly=20,
        )
        return PositionSizer(config)

    def test_create_sizer(self, default_sizer):
        """Test crear position sizer"""
        assert default_sizer is not None
        assert default_sizer.config.method == SizingMethod.FIXED_FRACTIONAL

    def test_fixed_fractional_basic(self, default_sizer):
        """Test fixed fractional basico"""
        result = default_sizer.calculate(
            symbol="BTC/USDT",
            capital=10000,
            entry_price=50000,
        )

        assert result.symbol == "BTC/USDT"
        assert result.recommended_size > 0
        assert result.sizing_method == SizingMethod.FIXED_FRACTIONAL
        assert result.confidence > 0

    def test_fixed_fractional_with_stop_loss(self, default_sizer):
        """Test fixed fractional con stop loss"""
        result = default_sizer.calculate(
            symbol="BTC/USDT",
            capital=10000,
            entry_price=50000,
            stop_loss=48000,  # 4% stop loss
        )

        # Con stop loss, el tamano se calcula basado en el riesgo
        risk_per_unit = 50000 - 48000  # $2000 por unidad
        expected_risk = 10000 * 0.02  # 2% del capital
        calculated_size = expected_risk / risk_per_unit  # 0.1 BTC

        # Pero el tamano se limita al 25% del capital = $2500 / $50000 = 0.05 BTC
        max_size = 10000 * 0.25 / 50000  # 0.05 BTC

        # El resultado final deberia ser el minimo de ambos
        expected_size = min(calculated_size, max_size)
        assert result.recommended_size == pytest.approx(expected_size, rel=0.01)

    def test_kelly_with_insufficient_trades(self, kelly_sizer):
        """Test Kelly con trades insuficientes"""
        trade_history = TradeHistory(
            returns=[0.02, -0.01, 0.03],
            wins=2,
            losses=1,
        )

        result = kelly_sizer.calculate(
            symbol="BTC/USDT",
            capital=10000,
            entry_price=50000,
            trade_history=trade_history,
        )

        # Debe caer back a fixed fractional
        assert len(result.warnings) > 0
        assert result.confidence < 0.9

    def test_kelly_with_sufficient_trades(self, kelly_sizer, sample_trade_history):
        """Test Kelly con trades suficientes"""
        result = kelly_sizer.calculate(
            symbol="BTC/USDT",
            capital=10000,
            entry_price=50000,
            trade_history=sample_trade_history,
        )

        assert result.sizing_method == SizingMethod.KELLY
        assert result.recommended_size > 0
        assert "raw_kelly" in result.adjustments
        assert "adjusted_kelly" in result.adjustments

    def test_kelly_negative_expectancy(self, kelly_sizer):
        """Test Kelly con expectancy negativo"""
        # Historial perdedor
        trade_history = TradeHistory(
            returns=[-0.02, -0.03, 0.01, -0.02, -0.01] * 10,
            wins=10,
            losses=40,
        )

        result = kelly_sizer.calculate(
            symbol="BTC/USDT",
            capital=10000,
            entry_price=50000,
            trade_history=trade_history,
        )

        assert len(result.warnings) > 0
        assert result.confidence < 0.5

    def test_atr_based(self):
        """Test ATR-based sizing"""
        config = PositionSizingConfig(
            method=SizingMethod.ATR_BASED,
            atr_multiplier=2.0,
            risk_per_trade=0.02,
        )
        sizer = PositionSizer(config)

        result = sizer.calculate(
            symbol="BTC/USDT",
            capital=10000,
            entry_price=50000,
            atr=1000,  # ATR de $1000
        )

        assert result.sizing_method == SizingMethod.ATR_BASED
        assert "atr" in result.adjustments
        # Risk = 2% de 10000 = $200
        # Stop distance = ATR * 2 = $2000
        # Size calculated = $200 / $2000 = 0.1 BTC
        # But max position = 25% of 10000 / 50000 = 0.05 BTC
        # Result should be limited to max
        assert result.recommended_size == pytest.approx(0.05, rel=0.01)

    def test_atr_without_atr_value(self):
        """Test ATR sin valor de ATR cae a fixed fractional"""
        config = PositionSizingConfig(method=SizingMethod.ATR_BASED)
        sizer = PositionSizer(config)

        result = sizer.calculate(
            symbol="BTC/USDT",
            capital=10000,
            entry_price=50000,
            atr=None,
        )

        assert len(result.warnings) > 0

    def test_volatility_adjusted(self):
        """Test volatility adjusted sizing"""
        config = PositionSizingConfig(
            method=SizingMethod.VOLATILITY_ADJUSTED,
            target_volatility=0.15,
            fixed_fraction=0.02,
        )
        sizer = PositionSizer(config)

        # Activo con baja volatilidad - deberia tener mayor tamano
        result_low_vol = sizer.calculate(
            symbol="LOW_VOL",
            capital=10000,
            entry_price=100,
            volatility=0.10,
        )

        # Activo con alta volatilidad - deberia tener menor tamano
        result_high_vol = sizer.calculate(
            symbol="HIGH_VOL",
            capital=10000,
            entry_price=100,
            volatility=0.30,
        )

        assert result_low_vol.recommended_size > result_high_vol.recommended_size

    def test_equal_weight(self):
        """Test equal weight sizing"""
        config = PositionSizingConfig(method=SizingMethod.EQUAL_WEIGHT)
        sizer = PositionSizer(config)

        result = sizer.calculate(
            symbol="BTC/USDT",
            capital=10000,
            entry_price=50000,
        )

        assert result.sizing_method == SizingMethod.EQUAL_WEIGHT
        # 10% del capital (10 posiciones)
        expected_value = 10000 * 0.10
        expected_size = expected_value / 50000
        assert result.recommended_size == pytest.approx(expected_size, rel=0.01)

    def test_max_position_limit(self):
        """Test que se aplica el limite maximo de posicion"""
        config = PositionSizingConfig(
            method=SizingMethod.FIXED_FRACTIONAL,
            fixed_fraction=0.50,  # 50% seria muy grande
            max_position_percent=0.20,  # Pero el limite es 20%
        )
        sizer = PositionSizer(config)

        result = sizer.calculate(
            symbol="BTC/USDT",
            capital=10000,
            entry_price=50000,
        )

        # El tamano deberia estar limitado al 20%
        max_value = 10000 * 0.20
        max_size = max_value / 50000
        assert result.recommended_size <= max_size
        assert "limit_applied" in result.adjustments

    def test_min_position_limit(self):
        """Test que se aplica el limite minimo de posicion"""
        config = PositionSizingConfig(
            method=SizingMethod.FIXED_FRACTIONAL,
            fixed_fraction=0.001,  # 0.1% seria muy pequeno
            min_position_percent=0.01,  # Minimo 1%
        )
        sizer = PositionSizer(config)

        result = sizer.calculate(
            symbol="BTC/USDT",
            capital=10000,
            entry_price=50000,
        )

        # El tamano deberia ser al menos 1%
        min_value = 10000 * 0.01
        min_size = min_value / 50000
        assert result.recommended_size >= min_size

    def test_absolute_max_position_value(self):
        """Test limite absoluto de valor de posicion"""
        config = PositionSizingConfig(
            method=SizingMethod.FIXED_FRACTIONAL,
            fixed_fraction=0.10,
            max_position_value=500,  # Max $500
        )
        sizer = PositionSizer(config)

        result = sizer.calculate(
            symbol="BTC/USDT",
            capital=10000,
            entry_price=50000,
        )

        # Max 500 / 50000 = 0.01 BTC
        assert result.max_allowed_size <= 500 / 50000

    def test_final_size_property(self):
        """Test propiedad final_size"""
        config = PositionSizingConfig()
        sizer = PositionSizer(config)

        result = sizer.calculate(
            symbol="BTC/USDT",
            capital=10000,
            entry_price=50000,
        )

        assert result.final_size == min(result.recommended_size, result.max_allowed_size)


class TestPositionSizerFactory:
    """Tests para PositionSizerFactory"""

    def test_create_conservative(self):
        """Test crear sizer conservador"""
        sizer = PositionSizerFactory.create_conservative()

        result = sizer.calculate(
            symbol="BTC/USDT",
            capital=10000,
            entry_price=50000,
        )

        # Conservador: max 10% de posicion
        max_value = 10000 * 0.10
        assert result.recommended_size * 50000 <= max_value

    def test_create_moderate(self):
        """Test crear sizer moderado"""
        sizer = PositionSizerFactory.create_moderate()

        result = sizer.calculate(
            symbol="BTC/USDT",
            capital=10000,
            entry_price=50000,
        )

        # Moderado: max 20% de posicion
        max_value = 10000 * 0.20
        assert result.recommended_size * 50000 <= max_value

    def test_create_aggressive(self):
        """Test crear sizer agresivo"""
        sizer = PositionSizerFactory.create_aggressive()

        assert sizer.config.method == SizingMethod.KELLY

    def test_create_atr_based(self):
        """Test crear sizer ATR-based"""
        sizer = PositionSizerFactory.create_atr_based(atr_multiplier=3.0)

        assert sizer.config.method == SizingMethod.ATR_BASED
        assert sizer.config.atr_multiplier == 3.0


class TestTradeHistory:
    """Tests para TradeHistory"""

    def test_total_trades(self, sample_trade_history):
        """Test total de trades"""
        assert sample_trade_history.total_trades == 30

    def test_win_rate(self, sample_trade_history):
        """Test win rate"""
        win_rate = sample_trade_history.win_rate
        assert 0 < win_rate < 1

    def test_avg_win(self, sample_trade_history):
        """Test promedio de ganancias"""
        avg_win = sample_trade_history.avg_win
        assert avg_win > 0

    def test_avg_loss(self, sample_trade_history):
        """Test promedio de perdidas"""
        avg_loss = sample_trade_history.avg_loss
        assert avg_loss > 0

    def test_empty_history(self):
        """Test historial vacio"""
        history = TradeHistory(returns=[], wins=0, losses=0)

        assert history.total_trades == 0
        assert history.win_rate == 0
        assert history.avg_win == 0
        assert history.avg_loss == 0
