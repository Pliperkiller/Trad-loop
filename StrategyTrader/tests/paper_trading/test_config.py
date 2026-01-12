"""
Tests para config.py
"""

import pytest
import sys
from pathlib import Path

src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from paper_trading.config import (
    PaperTradingConfig, SlippageModel, CommissionModel,
    CONSERVATIVE_CONFIG, MODERATE_CONFIG, AGGRESSIVE_CONFIG
)


class TestPaperTradingConfig:
    """Tests para PaperTradingConfig"""

    def test_default_config(self):
        """Test configuracion por defecto"""
        config = PaperTradingConfig()

        assert config.initial_balance == 10000
        assert config.commission_rate == 0.001
        assert config.max_position_size == 0.25
        assert config.max_positions == 5
        assert config.risk_per_trade == 0.02
        assert config.exchange == "binance"

    def test_custom_config(self):
        """Test configuracion personalizada"""
        config = PaperTradingConfig(
            initial_balance=50000,
            symbols=["ETH/USDT", "BTC/USDT"],
            commission_rate=0.0005,
            max_position_size=0.10,
        )

        assert config.initial_balance == 50000
        assert len(config.symbols) == 2
        assert config.commission_rate == 0.0005
        assert config.max_position_size == 0.10

    def test_validate_valid_config(self, default_config):
        """Test validacion de configuracion valida"""
        errors = default_config.validate()
        assert len(errors) == 0
        assert default_config.is_valid() == True

    def test_validate_invalid_balance(self):
        """Test validacion con balance invalido"""
        config = PaperTradingConfig(initial_balance=-1000)
        errors = config.validate()

        assert len(errors) > 0
        assert any("initial_balance" in e for e in errors)

    def test_validate_invalid_commission(self):
        """Test validacion con comision invalida"""
        config = PaperTradingConfig(commission_rate=0.5)  # 50% es muy alto
        errors = config.validate()

        assert len(errors) > 0
        assert any("commission_rate" in e for e in errors)

    def test_validate_no_symbols(self):
        """Test validacion sin simbolos"""
        config = PaperTradingConfig(symbols=[])
        errors = config.validate()

        assert len(errors) > 0
        assert any("symbol" in e.lower() for e in errors)

    def test_get_commission_percentage(self, default_config):
        """Test calculo de comision porcentual"""
        commission = default_config.get_commission(10000)
        assert commission == 10  # 0.1% de 10000

    def test_get_commission_maker_vs_taker(self, default_config):
        """Test diferencia maker vs taker"""
        default_config.maker_fee = 0.0005
        default_config.taker_fee = 0.001

        maker_comm = default_config.get_commission(10000, is_maker=True)
        taker_comm = default_config.get_commission(10000, is_maker=False)

        assert maker_comm == 5
        assert taker_comm == 10
        assert maker_comm < taker_comm

    def test_get_max_position_value(self, default_config):
        """Test valor maximo de posicion"""
        max_value = default_config.get_max_position_value()
        assert max_value == 2500  # 25% de 10000

    def test_to_dict(self, default_config):
        """Test conversion a diccionario"""
        d = default_config.to_dict()

        assert d["initial_balance"] == 10000
        assert d["commission_rate"] == 0.001
        assert d["max_position_size"] == 0.25

    def test_from_dict(self):
        """Test creacion desde diccionario"""
        data = {
            "initial_balance": 20000,
            "symbols": ["ETH/USDT"],
            "commission_rate": 0.002,
        }

        config = PaperTradingConfig.from_dict(data)

        assert config.initial_balance == 20000
        assert config.symbols == ["ETH/USDT"]
        assert config.commission_rate == 0.002

    def test_predefined_configs(self):
        """Test configuraciones predefinidas"""
        # Conservative
        assert CONSERVATIVE_CONFIG.max_position_size == 0.10
        assert CONSERVATIVE_CONFIG.risk_per_trade == 0.01

        # Moderate
        assert MODERATE_CONFIG.max_position_size == 0.20
        assert MODERATE_CONFIG.risk_per_trade == 0.02

        # Aggressive
        assert AGGRESSIVE_CONFIG.max_position_size == 0.30
        assert AGGRESSIVE_CONFIG.risk_per_trade == 0.03

    def test_slippage_models(self):
        """Test modelos de slippage"""
        config = PaperTradingConfig(slippage_model=SlippageModel.FIXED)
        assert config.slippage_model == SlippageModel.FIXED

        config = PaperTradingConfig(slippage_model=SlippageModel.VOLATILITY)
        assert config.slippage_model == SlippageModel.VOLATILITY

    def test_commission_models(self):
        """Test modelos de comision"""
        config = PaperTradingConfig(commission_model=CommissionModel.PERCENTAGE)
        assert config.commission_model == CommissionModel.PERCENTAGE

        config = PaperTradingConfig(commission_model=CommissionModel.FIXED)
        assert config.commission_model == CommissionModel.FIXED
