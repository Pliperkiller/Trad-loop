"""
Modelos de datos para el modulo de gestion de portfolio multi-activo.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, List
import uuid


class AllocationMethod(Enum):
    """Metodos de asset allocation"""
    EQUAL_WEIGHT = "equal_weight"
    RISK_PARITY = "risk_parity"
    MEAN_VARIANCE = "mean_variance"
    MIN_VARIANCE = "min_variance"
    MAX_SHARPE = "max_sharpe"
    CUSTOM = "custom"


class RebalanceFrequency(Enum):
    """Frecuencia de rebalanceo"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    THRESHOLD = "threshold"  # Basado en drift
    NEVER = "never"


class RebalanceReason(Enum):
    """Razon del rebalanceo"""
    SCHEDULED = "scheduled"
    THRESHOLD_BREACH = "threshold_breach"
    MANUAL = "manual"
    INITIAL = "initial"


@dataclass
class PortfolioPosition:
    """Posicion en un activo del portfolio"""
    symbol: str
    quantity: float
    entry_price: float
    current_price: float = 0.0
    entry_date: datetime = field(default_factory=datetime.now)

    @property
    def value(self) -> float:
        """Valor actual de la posicion"""
        price = self.current_price if self.current_price > 0 else self.entry_price
        return self.quantity * price

    @property
    def cost_basis(self) -> float:
        """Costo de entrada"""
        return self.quantity * self.entry_price

    @property
    def pnl(self) -> float:
        """Ganancia/perdida no realizada"""
        return self.value - self.cost_basis

    @property
    def pnl_percent(self) -> float:
        """Porcentaje de ganancia/perdida"""
        if self.cost_basis == 0:
            return 0.0
        return (self.pnl / self.cost_basis) * 100

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "current_price": self.current_price,
            "entry_date": self.entry_date.isoformat(),
            "value": self.value,
            "cost_basis": self.cost_basis,
            "pnl": self.pnl,
            "pnl_percent": self.pnl_percent,
        }


@dataclass
class AssetAllocation:
    """Allocation de un activo en el portfolio"""
    symbol: str
    target_weight: float  # Peso objetivo (0-1)
    current_weight: float = 0.0  # Peso actual
    value: float = 0.0  # Valor actual
    quantity: float = 0.0  # Cantidad actual

    @property
    def drift(self) -> float:
        """Desviacion del target (current - target)"""
        return self.current_weight - self.target_weight

    @property
    def drift_percent(self) -> float:
        """Desviacion porcentual absoluta"""
        return abs(self.drift) * 100

    @property
    def needs_rebalance(self) -> bool:
        """Indica si necesita rebalanceo (drift > 5%)"""
        return abs(self.drift) > 0.05

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "target_weight": self.target_weight,
            "current_weight": self.current_weight,
            "drift": self.drift,
            "drift_percent": self.drift_percent,
            "value": self.value,
            "quantity": self.quantity,
            "needs_rebalance": self.needs_rebalance,
        }


@dataclass
class RebalanceTrade:
    """Trade necesario para rebalancear"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    symbol: str = ""
    side: str = ""  # "buy" o "sell"
    quantity: float = 0.0
    estimated_price: float = 0.0
    estimated_value: float = 0.0
    estimated_commission: float = 0.0
    reason: RebalanceReason = RebalanceReason.SCHEDULED

    @property
    def total_cost(self) -> float:
        """Costo total incluyendo comision"""
        return self.estimated_value + self.estimated_commission

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "estimated_price": self.estimated_price,
            "estimated_value": self.estimated_value,
            "estimated_commission": self.estimated_commission,
            "total_cost": self.total_cost,
            "reason": self.reason.value,
        }


@dataclass
class PortfolioState:
    """Estado actual del portfolio"""
    timestamp: datetime = field(default_factory=datetime.now)

    # Capital
    total_equity: float = 0.0
    cash: float = 0.0
    invested_value: float = 0.0

    # Posiciones
    positions: Dict[str, PortfolioPosition] = field(default_factory=dict)

    # Pesos
    current_weights: Dict[str, float] = field(default_factory=dict)
    target_weights: Dict[str, float] = field(default_factory=dict)

    # PnL
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    total_pnl: float = 0.0

    # Metricas
    max_weight_drift: float = 0.0
    avg_weight_drift: float = 0.0
    num_positions: int = 0

    def update_weights(self) -> None:
        """Actualiza los pesos actuales basado en posiciones"""
        if self.total_equity <= 0:
            return

        self.current_weights = {}
        total_drift = 0.0
        max_drift = 0.0

        for symbol, position in self.positions.items():
            weight = position.value / self.total_equity
            self.current_weights[symbol] = weight

            if symbol in self.target_weights:
                drift = abs(weight - self.target_weights[symbol])
                total_drift += drift
                max_drift = max(max_drift, drift)

        self.num_positions = len(self.positions)
        self.max_weight_drift = max_drift
        self.avg_weight_drift = total_drift / self.num_positions if self.num_positions > 0 else 0.0

        # Actualizar PnL
        self.unrealized_pnl = sum(p.pnl for p in self.positions.values())
        self.invested_value = sum(p.value for p in self.positions.values())
        self.total_pnl = self.unrealized_pnl + self.realized_pnl

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "total_equity": self.total_equity,
            "cash": self.cash,
            "invested_value": self.invested_value,
            "positions": {s: p.to_dict() for s, p in self.positions.items()},
            "current_weights": self.current_weights,
            "target_weights": self.target_weights,
            "unrealized_pnl": self.unrealized_pnl,
            "realized_pnl": self.realized_pnl,
            "total_pnl": self.total_pnl,
            "max_weight_drift": self.max_weight_drift,
            "avg_weight_drift": self.avg_weight_drift,
            "num_positions": self.num_positions,
        }


@dataclass
class PortfolioConfig:
    """Configuracion del portfolio"""
    # Capital
    initial_capital: float = 10000.0

    # Activos
    symbols: List[str] = field(default_factory=list)

    # Timeframe para backtesting
    timeframe: str = "1h"

    # Allocation
    allocation_method: AllocationMethod = AllocationMethod.EQUAL_WEIGHT
    target_weights: Dict[str, float] = field(default_factory=dict)

    # Rebalanceo
    rebalance_frequency: RebalanceFrequency = RebalanceFrequency.MONTHLY
    rebalance_threshold: float = 0.05  # 5% drift para threshold-based
    min_trade_value: float = 10.0  # Minimo valor de trade

    # Costos
    commission: float = 0.001  # 0.1%
    slippage: float = 0.0005  # 0.05%

    # Restricciones
    max_position_weight: float = 0.40  # Maximo 40% en un activo
    min_position_weight: float = 0.01  # Minimo 1%
    max_positions: int = 20

    # Risk parameters
    risk_free_rate: float = 0.02  # 2% anual
    target_volatility: Optional[float] = None  # Para volatility targeting

    def __post_init__(self):
        """Validacion post-inicializacion"""
        if self.initial_capital <= 0:
            raise ValueError("initial_capital debe ser positivo")

        if self.commission < 0 or self.commission > 0.1:
            raise ValueError("commission debe estar entre 0 y 0.1")

        if self.slippage < 0 or self.slippage > 0.1:
            raise ValueError("slippage debe estar entre 0 y 0.1")

        if self.rebalance_threshold <= 0 or self.rebalance_threshold > 0.5:
            raise ValueError("rebalance_threshold debe estar entre 0 y 0.5")

        # Validar que weights sumen ~1 si se especifican
        if self.target_weights:
            total = sum(self.target_weights.values())
            if not (0.99 <= total <= 1.01):
                raise ValueError(f"target_weights deben sumar 1.0, suma actual: {total}")

    def to_dict(self) -> dict:
        return {
            "initial_capital": self.initial_capital,
            "symbols": self.symbols,
            "timeframe": self.timeframe,
            "allocation_method": self.allocation_method.value,
            "target_weights": self.target_weights,
            "rebalance_frequency": self.rebalance_frequency.value,
            "rebalance_threshold": self.rebalance_threshold,
            "min_trade_value": self.min_trade_value,
            "commission": self.commission,
            "slippage": self.slippage,
            "max_position_weight": self.max_position_weight,
            "min_position_weight": self.min_position_weight,
            "max_positions": self.max_positions,
            "risk_free_rate": self.risk_free_rate,
            "target_volatility": self.target_volatility,
        }


@dataclass
class PortfolioTradeRecord:
    """Registro de un trade ejecutado"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: datetime = field(default_factory=datetime.now)
    symbol: str = ""
    side: str = ""  # "buy" o "sell"
    quantity: float = 0.0
    price: float = 0.0
    value: float = 0.0
    commission: float = 0.0
    slippage_cost: float = 0.0
    reason: str = ""  # "rebalance", "initial", "exit"

    @property
    def total_cost(self) -> float:
        """Costo total del trade"""
        return self.commission + self.slippage_cost

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "price": self.price,
            "value": self.value,
            "commission": self.commission,
            "slippage_cost": self.slippage_cost,
            "total_cost": self.total_cost,
            "reason": self.reason,
        }


@dataclass
class RebalanceEvent:
    """Evento de rebalanceo"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: datetime = field(default_factory=datetime.now)
    reason: RebalanceReason = RebalanceReason.SCHEDULED

    # Estado antes
    weights_before: Dict[str, float] = field(default_factory=dict)
    equity_before: float = 0.0

    # Estado despues
    weights_after: Dict[str, float] = field(default_factory=dict)
    equity_after: float = 0.0

    # Trades
    trades: List[RebalanceTrade] = field(default_factory=list)
    total_trades: int = 0
    total_value_traded: float = 0.0
    total_commission: float = 0.0

    # Resultado
    success: bool = True
    error_message: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "reason": self.reason.value,
            "weights_before": self.weights_before,
            "equity_before": self.equity_before,
            "weights_after": self.weights_after,
            "equity_after": self.equity_after,
            "trades": [t.to_dict() for t in self.trades],
            "total_trades": self.total_trades,
            "total_value_traded": self.total_value_traded,
            "total_commission": self.total_commission,
            "success": self.success,
            "error_message": self.error_message,
        }


@dataclass
class PortfolioMetrics:
    """Metricas de rendimiento del portfolio"""
    # Rentabilidad
    total_return: float = 0.0
    total_return_pct: float = 0.0
    annualized_return: float = 0.0
    cagr: float = 0.0

    # Riesgo
    volatility: float = 0.0
    annualized_volatility: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0  # dias
    var_95: float = 0.0
    cvar_95: float = 0.0

    # Ratios
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    information_ratio: float = 0.0

    # Portfolio especificos
    diversification_ratio: float = 0.0  # Vol promedio ponderada / Vol portfolio
    concentration_ratio: float = 0.0  # HHI de weights
    effective_n: float = 0.0  # Numero efectivo de activos
    avg_correlation: float = 0.0

    # Contribuciones
    contribution_to_return: Dict[str, float] = field(default_factory=dict)
    contribution_to_risk: Dict[str, float] = field(default_factory=dict)

    # Costos
    turnover: float = 0.0  # Rotacion anualizada
    total_commission: float = 0.0
    total_slippage: float = 0.0
    rebalancing_cost: float = 0.0

    # Benchmark
    tracking_error: float = 0.0
    alpha: float = 0.0
    beta: float = 0.0

    # Periodos
    total_days: int = 0
    trading_days: int = 0
    num_rebalances: int = 0

    def to_dict(self) -> dict:
        return {
            "total_return": self.total_return,
            "total_return_pct": self.total_return_pct,
            "annualized_return": self.annualized_return,
            "cagr": self.cagr,
            "volatility": self.volatility,
            "annualized_volatility": self.annualized_volatility,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_duration": self.max_drawdown_duration,
            "var_95": self.var_95,
            "cvar_95": self.cvar_95,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "information_ratio": self.information_ratio,
            "diversification_ratio": self.diversification_ratio,
            "concentration_ratio": self.concentration_ratio,
            "effective_n": self.effective_n,
            "avg_correlation": self.avg_correlation,
            "contribution_to_return": self.contribution_to_return,
            "contribution_to_risk": self.contribution_to_risk,
            "turnover": self.turnover,
            "total_commission": self.total_commission,
            "total_slippage": self.total_slippage,
            "rebalancing_cost": self.rebalancing_cost,
            "tracking_error": self.tracking_error,
            "alpha": self.alpha,
            "beta": self.beta,
            "total_days": self.total_days,
            "trading_days": self.trading_days,
            "num_rebalances": self.num_rebalances,
        }


@dataclass
class PortfolioResult:
    """Resultado completo del backtest de portfolio"""
    config: PortfolioConfig = field(default_factory=PortfolioConfig)
    metrics: PortfolioMetrics = field(default_factory=PortfolioMetrics)

    # Series temporales
    equity_curve: List[float] = field(default_factory=list)
    returns: List[float] = field(default_factory=list)
    drawdown_curve: List[float] = field(default_factory=list)
    timestamps: List[datetime] = field(default_factory=list)

    # Historiales
    weight_history: Dict[str, List[float]] = field(default_factory=dict)
    trade_history: List[PortfolioTradeRecord] = field(default_factory=list)
    rebalance_history: List[RebalanceEvent] = field(default_factory=list)

    # Estado final
    final_state: Optional[PortfolioState] = None

    # Metadata
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    execution_time: float = 0.0

    def to_dict(self) -> dict:
        return {
            "config": self.config.to_dict(),
            "metrics": self.metrics.to_dict(),
            "equity_curve": self.equity_curve,
            "returns": self.returns,
            "drawdown_curve": self.drawdown_curve,
            "timestamps": [t.isoformat() for t in self.timestamps],
            "weight_history": self.weight_history,
            "trade_history": [t.to_dict() for t in self.trade_history],
            "rebalance_history": [r.to_dict() for r in self.rebalance_history],
            "final_state": self.final_state.to_dict() if self.final_state else None,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "execution_time": self.execution_time,
        }
