"""
Modelos de datos para el modulo de gestion de riesgo.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, List
import uuid


class SizingMethod(Enum):
    """Metodos de position sizing"""
    FIXED_FRACTIONAL = "fixed_fractional"
    KELLY = "kelly"
    OPTIMAL_F = "optimal_f"
    ATR_BASED = "atr_based"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    EQUAL_WEIGHT = "equal_weight"


class RiskLevel(Enum):
    """Niveles de riesgo"""
    MINIMAL = "minimal"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    EXTREME = "extreme"


class DrawdownAction(Enum):
    """Acciones ante drawdown"""
    NONE = "none"
    REDUCE_SIZE = "reduce_size"
    PAUSE_TRADING = "pause_trading"
    CLOSE_ALL = "close_all"


@dataclass
class PositionSizeResult:
    """Resultado del calculo de position size"""
    symbol: str
    recommended_size: float
    max_allowed_size: float
    risk_amount: float
    sizing_method: SizingMethod
    confidence: float  # 0-1, confianza en el calculo
    adjustments: Dict[str, str] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    @property
    def final_size(self) -> float:
        """Tamano final (el menor entre recomendado y maximo)"""
        return min(self.recommended_size, self.max_allowed_size)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "recommended_size": self.recommended_size,
            "max_allowed_size": self.max_allowed_size,
            "final_size": self.final_size,
            "risk_amount": self.risk_amount,
            "sizing_method": self.sizing_method.value,
            "confidence": self.confidence,
            "adjustments": self.adjustments,
            "warnings": self.warnings,
        }


@dataclass
class RiskMetrics:
    """Metricas de riesgo actuales"""
    timestamp: datetime = field(default_factory=datetime.now)

    # Capital
    total_equity: float = 0.0
    available_capital: float = 0.0
    margin_used: float = 0.0

    # Exposicion
    total_exposure: float = 0.0
    exposure_by_symbol: Dict[str, float] = field(default_factory=dict)
    exposure_by_sector: Dict[str, float] = field(default_factory=dict)

    # Drawdown
    current_drawdown: float = 0.0
    max_drawdown: float = 0.0
    drawdown_duration_days: int = 0

    # VaR
    var_95: float = 0.0
    var_99: float = 0.0
    cvar_95: float = 0.0

    # Correlacion
    avg_correlation: float = 0.0
    max_correlation_pair: Optional[tuple] = None

    # Estado
    risk_level: RiskLevel = RiskLevel.LOW
    is_trading_allowed: bool = True
    active_limits_breached: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "total_equity": self.total_equity,
            "available_capital": self.available_capital,
            "margin_used": self.margin_used,
            "total_exposure": self.total_exposure,
            "exposure_by_symbol": self.exposure_by_symbol,
            "exposure_by_sector": self.exposure_by_sector,
            "current_drawdown": self.current_drawdown,
            "max_drawdown": self.max_drawdown,
            "drawdown_duration_days": self.drawdown_duration_days,
            "var_95": self.var_95,
            "var_99": self.var_99,
            "cvar_95": self.cvar_95,
            "avg_correlation": self.avg_correlation,
            "max_correlation_pair": self.max_correlation_pair,
            "risk_level": self.risk_level.value,
            "is_trading_allowed": self.is_trading_allowed,
            "active_limits_breached": self.active_limits_breached,
        }


@dataclass
class RiskLimit:
    """Limite de riesgo configurable"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    limit_type: str = ""  # exposure, drawdown, correlation, var
    threshold: float = 0.0
    current_value: float = 0.0
    action: DrawdownAction = DrawdownAction.NONE
    is_breached: bool = False
    last_checked: datetime = field(default_factory=datetime.now)

    def check(self, value: float) -> bool:
        """Verifica si el limite fue excedido"""
        self.current_value = value
        self.last_checked = datetime.now()
        self.is_breached = value >= self.threshold
        return self.is_breached

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "limit_type": self.limit_type,
            "threshold": self.threshold,
            "current_value": self.current_value,
            "action": self.action.value,
            "is_breached": self.is_breached,
            "last_checked": self.last_checked.isoformat(),
        }


@dataclass
class TradeRiskAssessment:
    """Evaluacion de riesgo para un trade propuesto"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: datetime = field(default_factory=datetime.now)

    symbol: str = ""
    side: str = ""  # long/short
    proposed_size: float = 0.0
    entry_price: float = 0.0
    stop_loss: Optional[float] = None

    # Evaluacion
    is_approved: bool = False
    approved_size: float = 0.0
    risk_score: float = 0.0  # 0-100

    # Razones
    rejection_reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    adjustments_made: Dict[str, str] = field(default_factory=dict)

    # Impacto proyectado
    projected_exposure: float = 0.0
    projected_drawdown: float = 0.0
    projected_correlation_impact: float = 0.0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "side": self.side,
            "proposed_size": self.proposed_size,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "is_approved": self.is_approved,
            "approved_size": self.approved_size,
            "risk_score": self.risk_score,
            "rejection_reasons": self.rejection_reasons,
            "warnings": self.warnings,
            "adjustments_made": self.adjustments_made,
            "projected_exposure": self.projected_exposure,
            "projected_drawdown": self.projected_drawdown,
            "projected_correlation_impact": self.projected_correlation_impact,
        }


@dataclass
class CorrelationData:
    """Datos de correlacion entre activos"""
    symbol_a: str
    symbol_b: str
    correlation: float  # -1 a 1
    period_days: int = 30
    last_updated: datetime = field(default_factory=datetime.now)

    @property
    def is_highly_correlated(self) -> bool:
        """Correlacion mayor a 0.7"""
        return abs(self.correlation) > 0.7

    @property
    def is_diversifying(self) -> bool:
        """Correlacion negativa (diversifica)"""
        return self.correlation < -0.3

    def to_dict(self) -> dict:
        return {
            "symbol_a": self.symbol_a,
            "symbol_b": self.symbol_b,
            "correlation": self.correlation,
            "period_days": self.period_days,
            "last_updated": self.last_updated.isoformat(),
            "is_highly_correlated": self.is_highly_correlated,
            "is_diversifying": self.is_diversifying,
        }


@dataclass
class DrawdownState:
    """Estado actual del drawdown"""
    peak_equity: float = 0.0
    current_equity: float = 0.0
    drawdown_amount: float = 0.0
    drawdown_percent: float = 0.0
    drawdown_start: Optional[datetime] = None
    days_in_drawdown: int = 0
    max_drawdown_percent: float = 0.0
    max_drawdown_date: Optional[datetime] = None
    recovery_needed_percent: float = 0.0

    def update(self, equity: float) -> None:
        """Actualiza el estado del drawdown"""
        self.current_equity = equity

        if equity > self.peak_equity:
            # Nuevo maximo
            self.peak_equity = equity
            self.drawdown_start = None
            self.days_in_drawdown = 0
        else:
            # En drawdown
            self.drawdown_amount = self.peak_equity - equity
            self.drawdown_percent = (self.drawdown_amount / self.peak_equity) * 100 if self.peak_equity > 0 else 0

            if self.drawdown_start is None:
                self.drawdown_start = datetime.now()

            self.days_in_drawdown = (datetime.now() - self.drawdown_start).days if self.drawdown_start else 0

            if self.drawdown_percent > self.max_drawdown_percent:
                self.max_drawdown_percent = self.drawdown_percent
                self.max_drawdown_date = datetime.now()

            # Porcentaje necesario para recuperar
            if equity > 0:
                self.recovery_needed_percent = ((self.peak_equity / equity) - 1) * 100
            else:
                self.recovery_needed_percent = 100.0

    def to_dict(self) -> dict:
        return {
            "peak_equity": self.peak_equity,
            "current_equity": self.current_equity,
            "drawdown_amount": self.drawdown_amount,
            "drawdown_percent": self.drawdown_percent,
            "drawdown_start": self.drawdown_start.isoformat() if self.drawdown_start else None,
            "days_in_drawdown": self.days_in_drawdown,
            "max_drawdown_percent": self.max_drawdown_percent,
            "max_drawdown_date": self.max_drawdown_date.isoformat() if self.max_drawdown_date else None,
            "recovery_needed_percent": self.recovery_needed_percent,
        }
