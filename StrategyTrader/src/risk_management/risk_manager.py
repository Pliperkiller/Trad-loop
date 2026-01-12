"""
Risk Manager - Orquestador principal del modulo de gestion de riesgo.

Integra todos los componentes:
- Position Sizing (Kelly, Optimal-f, Fixed Fractional)
- Exposure Limits
- Drawdown Protection
- Correlation Management
- VaR Calculation
"""

from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass, field
import logging

from .models import (
    PositionSizeResult,
    RiskMetrics,
    TradeRiskAssessment,
    DrawdownState,
    RiskLevel,
)
from .config import RiskManagementConfig
from .position_sizer import PositionSizer, TradeHistory
from .risk_limits import (
    RiskLimitChecker,
    Position,
)
from .correlation_manager import CorrelationManager


logger = logging.getLogger(__name__)


@dataclass
class RiskManagerState:
    """Estado del Risk Manager"""
    is_active: bool = True
    is_trading_allowed: bool = True
    current_risk_level: RiskLevel = RiskLevel.MINIMAL
    last_update: datetime = field(default_factory=datetime.now)
    alerts: List[str] = field(default_factory=list)
    trade_count: int = 0
    rejected_count: int = 0


class RiskManager:
    """
    Gestor de riesgo principal.

    Orquesta todos los componentes de gestion de riesgo para:
    - Calcular tamanos de posicion optimos
    - Verificar limites de exposicion
    - Proteger contra drawdown excesivo
    - Gestionar correlacion entre posiciones
    - Calcular Value at Risk
    """

    def __init__(self, config: RiskManagementConfig):
        self.config = config
        self.state = RiskManagerState()

        # Inicializar componentes
        self.position_sizer = PositionSizer(config.position_sizing)

        self.limit_checker = RiskLimitChecker(
            config.exposure_limits,
            config.drawdown,
            config.var,
            config.initial_capital,
        )

        self.correlation_manager = CorrelationManager(config.correlation)

        # Trade history para Kelly/Optimal-f
        self.trade_history = TradeHistory(returns=[], wins=0, losses=0)

        # Posiciones actuales
        self.positions: List[Position] = []

        # Capital actual
        self.current_capital: float = config.initial_capital

        logger.info(f"RiskManager inicializado con capital: {config.initial_capital}")

    def assess_trade(
        self,
        symbol: str,
        side: str,
        proposed_size: float,
        entry_price: float,
        stop_loss: Optional[float] = None,
        volatility: Optional[float] = None,
        atr: Optional[float] = None,
    ) -> TradeRiskAssessment:
        """
        Evalua un trade propuesto y retorna la evaluacion completa.

        Esta es la funcion principal que debe usarse antes de abrir cualquier posicion.

        Args:
            symbol: Simbolo del activo
            side: "long" o "short"
            proposed_size: Tamano propuesto
            entry_price: Precio de entrada
            stop_loss: Precio de stop loss (opcional)
            volatility: Volatilidad del activo (opcional)
            atr: Average True Range (opcional)

        Returns:
            TradeRiskAssessment: Evaluacion completa del trade
        """
        assessment = TradeRiskAssessment(
            symbol=symbol,
            side=side,
            proposed_size=proposed_size,
            entry_price=entry_price,
            stop_loss=stop_loss,
        )

        # 1. Verificar si trading esta permitido
        if not self.state.is_trading_allowed:
            assessment.is_approved = False
            assessment.rejection_reasons.append(
                f"Trading pausado: {self.state.current_risk_level.value}"
            )
            self.state.rejected_count += 1
            return assessment

        # 2. Verificar limite de posiciones
        if len(self.positions) >= self.config.max_open_positions:
            assessment.is_approved = False
            assessment.rejection_reasons.append(
                f"Maximo de posiciones alcanzado: {len(self.positions)}/{self.config.max_open_positions}"
            )
            self.state.rejected_count += 1
            return assessment

        # 3. Calcular tamano optimo
        size_result = self.position_sizer.calculate(
            symbol=symbol,
            capital=self.current_capital,
            entry_price=entry_price,
            stop_loss=stop_loss,
            trade_history=self.trade_history,
            volatility=volatility,
            atr=atr,
        )

        # 4. Verificar limites de exposicion
        can_open, exp_reasons = self.limit_checker.can_open_position(
            symbol, side, proposed_size, entry_price
        )

        if not can_open:
            assessment.is_approved = False
            assessment.rejection_reasons.extend(exp_reasons)
            self.state.rejected_count += 1
            return assessment

        # 5. Verificar correlacion
        current_symbols = [p.symbol for p in self.positions]
        can_open_corr, corr_reasons = self.correlation_manager.check_correlation_limits(
            current_symbols, symbol
        )

        if not can_open_corr:
            assessment.is_approved = False
            assessment.rejection_reasons.extend(corr_reasons)
            self.state.rejected_count += 1
            return assessment

        # 6. Aplicar ajustes
        adjusted_size = proposed_size

        # Ajuste por drawdown
        dd_multiplier = self.limit_checker.drawdown_protection.get_size_multiplier()
        if dd_multiplier < 1.0:
            adjusted_size *= dd_multiplier
            assessment.adjustments_made["drawdown"] = f"Reducido {(1-dd_multiplier)*100:.0f}%"

        # Ajuste por correlacion
        corr_multiplier = self.correlation_manager.get_correlation_penalty(
            current_symbols, symbol
        )
        if corr_multiplier < 1.0:
            adjusted_size *= corr_multiplier
            assessment.adjustments_made["correlation"] = f"Reducido {(1-corr_multiplier)*100:.0f}%"

        # Limitar al maximo permitido
        max_exposure_size = self.limit_checker.exposure_manager.get_max_allowed_size(
            symbol, side, entry_price
        )
        if adjusted_size > max_exposure_size:
            adjusted_size = max_exposure_size
            assessment.adjustments_made["exposure"] = f"Limitado a {max_exposure_size:.6f}"

        # Limitar al tamano optimo calculado
        if adjusted_size > size_result.final_size:
            adjusted_size = size_result.final_size
            assessment.adjustments_made["sizing"] = f"Limitado a {size_result.final_size:.6f}"

        # 7. Agregar warnings de size_result
        assessment.warnings.extend(size_result.warnings)
        assessment.warnings.extend(corr_reasons)  # Warnings de correlacion

        # 8. Calcular impacto proyectado
        new_exposure = (adjusted_size * entry_price) / self.current_capital
        assessment.projected_exposure = self.limit_checker.exposure_manager.get_total_exposure() + new_exposure

        # 9. Calcular risk score (0-100, menor es mejor)
        assessment.risk_score = self._calculate_risk_score(assessment, size_result)

        # 10. Aprobar si el tamano es valido
        if adjusted_size > 0:
            assessment.is_approved = True
            assessment.approved_size = adjusted_size
            self.state.trade_count += 1
        else:
            assessment.is_approved = False
            assessment.rejection_reasons.append("Tamano ajustado es cero")
            self.state.rejected_count += 1

        if self.config.log_all_decisions:
            self._log_assessment(assessment)

        return assessment

    def _calculate_risk_score(
        self,
        assessment: TradeRiskAssessment,
        size_result: PositionSizeResult,
    ) -> float:
        """Calcula un score de riesgo (0-100)"""
        score = 0.0

        # Factor de drawdown (0-30 puntos)
        dd_state = self.limit_checker.drawdown_protection.state
        dd_score = min(30, (dd_state.drawdown_percent / 20) * 30)
        score += dd_score

        # Factor de exposicion (0-25 puntos)
        exp_score = min(25, assessment.projected_exposure * 25)
        score += exp_score

        # Factor de correlacion (0-20 puntos)
        avg_corr = self.correlation_manager.get_average_correlation()
        corr_score = avg_corr * 20
        score += corr_score

        # Factor de confianza del sizing (0-15 puntos inversos)
        conf_score = (1 - size_result.confidence) * 15
        score += conf_score

        # Factor de warnings (0-10 puntos)
        warning_score = min(10, len(assessment.warnings) * 2)
        score += warning_score

        return min(100, score)

    def get_optimal_position_size(
        self,
        symbol: str,
        entry_price: float,
        stop_loss: Optional[float] = None,
        volatility: Optional[float] = None,
        atr: Optional[float] = None,
    ) -> PositionSizeResult:
        """
        Obtiene el tamano de posicion optimo sin evaluar el trade completo.

        Util para calcular previamente antes de decidir si entrar.
        """
        return self.position_sizer.calculate(
            symbol=symbol,
            capital=self.current_capital,
            entry_price=entry_price,
            stop_loss=stop_loss,
            trade_history=self.trade_history,
            volatility=volatility,
            atr=atr,
        )

    def update(
        self,
        equity: float,
        positions: List[Position],
        daily_return: Optional[float] = None,
    ):
        """
        Actualiza el estado del risk manager.

        Debe llamarse periodicamente (idealmente cada minuto o cada trade).

        Args:
            equity: Equity actual
            positions: Lista de posiciones abiertas
            daily_return: Retorno diario (para calculos de VaR)
        """
        self.current_capital = equity
        self.positions = positions

        # Actualizar limit checker
        self.limit_checker.update(equity, positions, daily_return)

        # Actualizar estado
        self.state.current_risk_level = self.limit_checker.drawdown_protection.get_risk_level()
        self.state.is_trading_allowed = self.limit_checker.drawdown_protection.is_trading_allowed()
        self.state.last_update = datetime.now()

        # Verificar si hay alertas
        self._check_alerts()

    def _check_alerts(self):
        """Verifica y genera alertas"""
        self.state.alerts.clear()

        # Alerta de drawdown
        dd_state = self.limit_checker.drawdown_protection.state
        if dd_state.drawdown_percent >= self.config.drawdown.warning_level * 100:
            self.state.alerts.append(
                f"Drawdown warning: {dd_state.drawdown_percent:.1f}%"
            )

        # Alerta de exposicion
        total_exp = self.limit_checker.exposure_manager.get_total_exposure()
        if total_exp >= self.config.exposure_limits.max_total_exposure * 0.8:
            self.state.alerts.append(
                f"Exposicion alta: {total_exp*100:.1f}%"
            )

        # Alerta de correlacion
        high_corr = self.correlation_manager.get_highly_correlated_pairs()
        if len(high_corr) > 0:
            self.state.alerts.append(
                f"{len(high_corr)} pares con alta correlacion"
            )

    def record_trade_result(self, return_pct: float):
        """
        Registra el resultado de un trade cerrado.

        Actualiza el historial para calculos de Kelly/Optimal-f.

        Args:
            return_pct: Retorno porcentual del trade (ej: 0.05 = 5%)
        """
        self.trade_history.returns.append(return_pct)

        if return_pct > 0:
            self.trade_history.wins += 1
        else:
            self.trade_history.losses += 1

        logger.debug(f"Trade registrado: {return_pct*100:.2f}% | "
                    f"Win rate: {self.trade_history.win_rate*100:.1f}%")

    def update_correlation_data(self, symbol: str, returns: List[float]):
        """Actualiza datos de retornos para calculos de correlacion"""
        self.correlation_manager.update_returns(symbol, returns)

    def add_daily_return(self, symbol: str, daily_return: float):
        """Agrega un retorno diario para un simbolo"""
        self.correlation_manager.add_return(symbol, daily_return)

    def refresh_correlation_matrix(self, symbols: Optional[List[str]] = None):
        """Recalcula la matriz de correlacion"""
        if symbols is None:
            symbols = list(set(p.symbol for p in self.positions))
        self.correlation_manager.update_correlation_matrix(symbols)

    def get_risk_metrics(self) -> RiskMetrics:
        """Obtiene las metricas de riesgo actuales"""
        metrics = self.limit_checker.check_all_limits()

        # Agregar datos de correlacion
        metrics.avg_correlation = self.correlation_manager.get_average_correlation()
        max_corr = self.correlation_manager.get_max_correlation_pair()
        if max_corr:
            metrics.max_correlation_pair = (max_corr[0], max_corr[1])

        return metrics

    def get_drawdown_state(self) -> DrawdownState:
        """Obtiene el estado actual del drawdown"""
        return self.limit_checker.drawdown_protection.state

    def is_trading_allowed(self) -> bool:
        """Verifica si se permite trading"""
        return self.state.is_trading_allowed

    def get_current_risk_level(self) -> RiskLevel:
        """Obtiene el nivel de riesgo actual"""
        return self.state.current_risk_level

    def should_close_all_positions(self) -> bool:
        """Verifica si se deben cerrar todas las posiciones"""
        return self.limit_checker.drawdown_protection.should_close_all()

    def get_position_summary(self) -> Dict:
        """Obtiene un resumen de las posiciones actuales"""
        long_count = sum(1 for p in self.positions if p.side == "long")
        short_count = len(self.positions) - long_count

        long_value = sum(p.value for p in self.positions if p.side == "long")
        short_value = sum(p.value for p in self.positions if p.side == "short")

        total_pnl = sum(p.pnl for p in self.positions)

        return {
            "total_positions": len(self.positions),
            "long_positions": long_count,
            "short_positions": short_count,
            "long_exposure": long_value / self.current_capital if self.current_capital > 0 else 0,
            "short_exposure": short_value / self.current_capital if self.current_capital > 0 else 0,
            "total_exposure": (long_value + short_value) / self.current_capital if self.current_capital > 0 else 0,
            "unrealized_pnl": total_pnl,
            "unrealized_pnl_pct": (total_pnl / self.current_capital * 100) if self.current_capital > 0 else 0,
        }

    def get_status(self) -> Dict:
        """Obtiene el estado completo del risk manager"""
        return {
            "is_active": self.state.is_active,
            "is_trading_allowed": self.state.is_trading_allowed,
            "risk_level": self.state.current_risk_level.value,
            "capital": self.current_capital,
            "positions": len(self.positions),
            "max_positions": self.config.max_open_positions,
            "trade_count": self.state.trade_count,
            "rejected_count": self.state.rejected_count,
            "alerts": self.state.alerts,
            "last_update": self.state.last_update.isoformat(),
            "drawdown": self.get_drawdown_state().to_dict(),
            "exposure": {
                "total": self.limit_checker.exposure_manager.get_total_exposure(),
                "long": self.limit_checker.exposure_manager.get_long_exposure(),
                "short": self.limit_checker.exposure_manager.get_short_exposure(),
            },
            "correlation": self.correlation_manager.get_correlation_report(),
        }

    def reset(self, new_capital: Optional[float] = None):
        """Reinicia el risk manager"""
        capital = new_capital if new_capital is not None else self.config.initial_capital
        self.current_capital = capital
        self.positions = []
        self.trade_history = TradeHistory(returns=[], wins=0, losses=0)
        self.state = RiskManagerState()

        self.limit_checker.reset(capital)
        self.correlation_manager.reset()

        logger.info(f"RiskManager reiniciado con capital: {capital}")

    def _log_assessment(self, assessment: TradeRiskAssessment):
        """Registra la evaluacion de un trade"""
        status = "APROBADO" if assessment.is_approved else "RECHAZADO"
        logger.info(
            f"Trade {status}: {assessment.symbol} {assessment.side} | "
            f"Propuesto: {assessment.proposed_size:.6f} | "
            f"Aprobado: {assessment.approved_size:.6f} | "
            f"Risk Score: {assessment.risk_score:.1f}"
        )

        if assessment.rejection_reasons:
            for reason in assessment.rejection_reasons:
                logger.warning(f"  Rechazo: {reason}")

        if assessment.warnings:
            for warning in assessment.warnings:
                logger.debug(f"  Warning: {warning}")


class RiskManagerFactory:
    """Factory para crear Risk Managers con configuraciones predefinidas"""

    @staticmethod
    def create_conservative(initial_capital: float = 10000) -> RiskManager:
        """Crea un risk manager conservador"""
        from .config import CONSERVATIVE_RISK_CONFIG
        config = CONSERVATIVE_RISK_CONFIG
        config.initial_capital = initial_capital
        return RiskManager(config)

    @staticmethod
    def create_moderate(initial_capital: float = 10000) -> RiskManager:
        """Crea un risk manager moderado"""
        from .config import MODERATE_RISK_CONFIG
        config = MODERATE_RISK_CONFIG
        config.initial_capital = initial_capital
        return RiskManager(config)

    @staticmethod
    def create_aggressive(initial_capital: float = 10000) -> RiskManager:
        """Crea un risk manager agresivo"""
        from .config import AGGRESSIVE_RISK_CONFIG
        config = AGGRESSIVE_RISK_CONFIG
        config.initial_capital = initial_capital
        return RiskManager(config)
