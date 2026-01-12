"""
Configuracion del modulo de gestion de riesgo.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from .models import SizingMethod, DrawdownAction, RiskLevel


@dataclass
class PositionSizingConfig:
    """Configuracion de position sizing"""
    method: SizingMethod = SizingMethod.FIXED_FRACTIONAL
    risk_per_trade: float = 0.02  # 2% por defecto

    # Fixed Fractional
    fixed_fraction: float = 0.02

    # Kelly
    kelly_fraction: float = 0.5  # Half-Kelly por seguridad
    min_trades_for_kelly: int = 30  # Minimo de trades para calcular Kelly

    # ATR Based
    atr_multiplier: float = 2.0
    atr_period: int = 14

    # Volatility Adjusted
    target_volatility: float = 0.15  # 15% anual
    lookback_period: int = 20

    # Limites
    max_position_percent: float = 0.25  # Maximo 25% del capital
    min_position_percent: float = 0.01  # Minimo 1% del capital
    max_position_value: Optional[float] = None  # Valor absoluto maximo


@dataclass
class ExposureLimitsConfig:
    """Configuracion de limites de exposicion"""
    # Por activo individual
    max_single_asset_exposure: float = 0.25  # 25%

    # Por sector/categoria
    max_sector_exposure: float = 0.40  # 40%
    sectors: Dict[str, List[str]] = field(default_factory=lambda: {
        "crypto_major": ["BTC/USDT", "ETH/USDT"],
        "crypto_alt": ["SOL/USDT", "ADA/USDT", "DOT/USDT"],
        "defi": ["UNI/USDT", "AAVE/USDT", "LINK/USDT"],
    })

    # Exposicion total
    max_total_exposure: float = 1.0  # 100% del capital
    max_long_exposure: float = 1.0
    max_short_exposure: float = 0.5  # Shorts mas conservadores

    # Por correlacion
    max_correlated_exposure: float = 0.50  # Max 50% en activos correlacionados


@dataclass
class DrawdownConfig:
    """Configuracion de proteccion contra drawdown"""
    # Niveles de alerta
    warning_level: float = 0.05  # 5%
    caution_level: float = 0.10  # 10%
    danger_level: float = 0.15  # 15%
    critical_level: float = 0.20  # 20%

    # Acciones por nivel
    actions: Dict[str, DrawdownAction] = field(default_factory=lambda: {
        "warning": DrawdownAction.NONE,
        "caution": DrawdownAction.REDUCE_SIZE,
        "danger": DrawdownAction.REDUCE_SIZE,
        "critical": DrawdownAction.PAUSE_TRADING,
    })

    # Reduccion de tamano por nivel
    size_reduction: Dict[str, float] = field(default_factory=lambda: {
        "warning": 1.0,  # Sin reduccion
        "caution": 0.75,  # Reducir 25%
        "danger": 0.50,  # Reducir 50%
        "critical": 0.0,  # No operar
    })

    # Recuperacion
    recovery_threshold: float = 0.50  # Recuperar 50% del drawdown para volver a normal
    cooldown_hours: int = 24  # Horas de espera despues de critical


@dataclass
class CorrelationConfig:
    """Configuracion de gestion de correlacion"""
    enabled: bool = True
    lookback_days: int = 30
    high_correlation_threshold: float = 0.70
    max_correlated_positions: int = 3

    # Penalizacion
    correlation_penalty: bool = True
    penalty_factor: float = 0.5  # Reducir 50% size si muy correlacionado

    # Actualizacion
    update_frequency_hours: int = 24


@dataclass
class VaRConfig:
    """Configuracion de Value at Risk"""
    enabled: bool = True
    confidence_levels: List[float] = field(default_factory=lambda: [0.95, 0.99])
    lookback_days: int = 252  # 1 año
    method: str = "historical"  # historical, parametric, monte_carlo

    # Limites
    max_var_95: float = 0.05  # Max 5% VaR diario al 95%
    max_var_99: float = 0.10  # Max 10% VaR diario al 99%


@dataclass
class RiskManagementConfig:
    """Configuracion completa del modulo de riesgo"""
    # Sub-configuraciones
    position_sizing: PositionSizingConfig = field(default_factory=PositionSizingConfig)
    exposure_limits: ExposureLimitsConfig = field(default_factory=ExposureLimitsConfig)
    drawdown: DrawdownConfig = field(default_factory=DrawdownConfig)
    correlation: CorrelationConfig = field(default_factory=CorrelationConfig)
    var: VaRConfig = field(default_factory=VaRConfig)

    # Capital
    initial_capital: float = 10000.0
    max_open_positions: int = 10

    # Logging
    log_all_decisions: bool = True
    log_rejections: bool = True

    def validate(self) -> List[str]:
        """Valida la configuracion"""
        errors = []

        # Position sizing
        if self.position_sizing.risk_per_trade <= 0 or self.position_sizing.risk_per_trade > 0.10:
            errors.append("risk_per_trade debe estar entre 0 y 10%")

        if self.position_sizing.max_position_percent > 0.50:
            errors.append("max_position_percent no deberia exceder 50%")

        # Exposure
        if self.exposure_limits.max_single_asset_exposure > 0.50:
            errors.append("max_single_asset_exposure no deberia exceder 50%")

        # Drawdown
        if self.drawdown.critical_level > 0.30:
            errors.append("critical_level muy alto, riesgo de perdida excesiva")

        # Capital
        if self.initial_capital <= 0:
            errors.append("initial_capital debe ser positivo")

        if self.max_open_positions <= 0:
            errors.append("max_open_positions debe ser positivo")

        return errors

    def is_valid(self) -> bool:
        """Verifica si la configuracion es valida"""
        return len(self.validate()) == 0

    def get_risk_level_for_drawdown(self, drawdown_percent: float) -> RiskLevel:
        """Determina el nivel de riesgo segun el drawdown"""
        if drawdown_percent >= self.drawdown.critical_level:
            return RiskLevel.EXTREME
        elif drawdown_percent >= self.drawdown.danger_level:
            return RiskLevel.HIGH
        elif drawdown_percent >= self.drawdown.caution_level:
            return RiskLevel.MODERATE
        elif drawdown_percent >= self.drawdown.warning_level:
            return RiskLevel.LOW
        return RiskLevel.MINIMAL

    def get_size_multiplier_for_drawdown(self, drawdown_percent: float) -> float:
        """Obtiene el multiplicador de tamano segun drawdown"""
        if drawdown_percent >= self.drawdown.critical_level:
            return self.drawdown.size_reduction["critical"]
        elif drawdown_percent >= self.drawdown.danger_level:
            return self.drawdown.size_reduction["danger"]
        elif drawdown_percent >= self.drawdown.caution_level:
            return self.drawdown.size_reduction["caution"]
        elif drawdown_percent >= self.drawdown.warning_level:
            return self.drawdown.size_reduction["warning"]
        return 1.0

    def to_dict(self) -> dict:
        """Convierte la configuracion a diccionario"""
        return {
            "initial_capital": self.initial_capital,
            "max_open_positions": self.max_open_positions,
            "position_sizing": {
                "method": self.position_sizing.method.value,
                "risk_per_trade": self.position_sizing.risk_per_trade,
                "max_position_percent": self.position_sizing.max_position_percent,
            },
            "exposure_limits": {
                "max_single_asset": self.exposure_limits.max_single_asset_exposure,
                "max_sector": self.exposure_limits.max_sector_exposure,
                "max_total": self.exposure_limits.max_total_exposure,
            },
            "drawdown": {
                "warning": self.drawdown.warning_level,
                "caution": self.drawdown.caution_level,
                "danger": self.drawdown.danger_level,
                "critical": self.drawdown.critical_level,
            },
            "correlation": {
                "enabled": self.correlation.enabled,
                "threshold": self.correlation.high_correlation_threshold,
            },
            "var": {
                "enabled": self.var.enabled,
                "max_var_95": self.var.max_var_95,
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RiskManagementConfig":
        """Crea configuracion desde diccionario"""
        config = cls()
        if "initial_capital" in data:
            config.initial_capital = data["initial_capital"]
        if "max_open_positions" in data:
            config.max_open_positions = data["max_open_positions"]
        # ... agregar mas segun necesidad
        return config


# Configuraciones predefinidas
CONSERVATIVE_RISK_CONFIG = RiskManagementConfig(
    position_sizing=PositionSizingConfig(
        method=SizingMethod.FIXED_FRACTIONAL,
        risk_per_trade=0.01,
        max_position_percent=0.10,
    ),
    exposure_limits=ExposureLimitsConfig(
        max_single_asset_exposure=0.15,
        max_total_exposure=0.60,
    ),
    drawdown=DrawdownConfig(
        warning_level=0.03,
        caution_level=0.05,
        danger_level=0.08,
        critical_level=0.10,
    ),
    max_open_positions=5,
)

MODERATE_RISK_CONFIG = RiskManagementConfig(
    position_sizing=PositionSizingConfig(
        method=SizingMethod.FIXED_FRACTIONAL,
        risk_per_trade=0.02,
        max_position_percent=0.20,
    ),
    exposure_limits=ExposureLimitsConfig(
        max_single_asset_exposure=0.25,
        max_total_exposure=0.80,
    ),
    drawdown=DrawdownConfig(
        warning_level=0.05,
        caution_level=0.10,
        danger_level=0.15,
        critical_level=0.20,
    ),
    max_open_positions=8,
)

AGGRESSIVE_RISK_CONFIG = RiskManagementConfig(
    position_sizing=PositionSizingConfig(
        method=SizingMethod.KELLY,
        risk_per_trade=0.03,
        max_position_percent=0.30,
        kelly_fraction=0.5,
    ),
    exposure_limits=ExposureLimitsConfig(
        max_single_asset_exposure=0.35,
        max_total_exposure=1.0,
    ),
    drawdown=DrawdownConfig(
        warning_level=0.10,
        caution_level=0.15,
        danger_level=0.20,
        critical_level=0.25,
    ),
    max_open_positions=12,
)
