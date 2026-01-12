"""
Gestion de limites de riesgo.

Incluye:
- Limites de exposicion por activo/sector
- Proteccion contra drawdown
- Limites de VaR
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import numpy as np

from .models import (
    RiskLimit,
    DrawdownState,
    DrawdownAction,
    RiskLevel,
    RiskMetrics,
)
from .config import (
    ExposureLimitsConfig,
    DrawdownConfig,
    VaRConfig,
)


@dataclass
class Position:
    """Posicion abierta (simplificada para calculos de riesgo)"""
    symbol: str
    side: str  # "long" o "short"
    size: float
    entry_price: float
    current_price: float = 0.0
    sector: Optional[str] = None

    @property
    def value(self) -> float:
        """Valor de la posicion"""
        price = self.current_price if self.current_price > 0 else self.entry_price
        return self.size * price

    @property
    def pnl(self) -> float:
        """PnL no realizado"""
        if self.current_price <= 0:
            return 0.0
        if self.side == "long":
            return (self.current_price - self.entry_price) * self.size
        else:  # short
            return (self.entry_price - self.current_price) * self.size


class ExposureLimitManager:
    """Gestor de limites de exposicion"""

    def __init__(self, config: ExposureLimitsConfig, total_capital: float):
        self.config = config
        self.total_capital = total_capital
        self.positions: List[Position] = []
        self.limits: List[RiskLimit] = []
        self._init_limits()

    def _init_limits(self):
        """Inicializa los limites"""
        self.limits = [
            RiskLimit(
                name="Max Single Asset Exposure",
                limit_type="exposure",
                threshold=self.config.max_single_asset_exposure,
                action=DrawdownAction.REDUCE_SIZE,
            ),
            RiskLimit(
                name="Max Sector Exposure",
                limit_type="exposure",
                threshold=self.config.max_sector_exposure,
                action=DrawdownAction.REDUCE_SIZE,
            ),
            RiskLimit(
                name="Max Total Exposure",
                limit_type="exposure",
                threshold=self.config.max_total_exposure,
                action=DrawdownAction.PAUSE_TRADING,
            ),
            RiskLimit(
                name="Max Long Exposure",
                limit_type="exposure",
                threshold=self.config.max_long_exposure,
                action=DrawdownAction.REDUCE_SIZE,
            ),
            RiskLimit(
                name="Max Short Exposure",
                limit_type="exposure",
                threshold=self.config.max_short_exposure,
                action=DrawdownAction.REDUCE_SIZE,
            ),
        ]

    def update_capital(self, capital: float):
        """Actualiza el capital total"""
        self.total_capital = capital

    def update_positions(self, positions: List[Position]):
        """Actualiza las posiciones actuales"""
        self.positions = positions

    def get_exposure_by_symbol(self) -> Dict[str, float]:
        """Obtiene la exposicion por simbolo como porcentaje del capital"""
        exposure = {}
        for pos in self.positions:
            symbol = pos.symbol
            pct = pos.value / self.total_capital if self.total_capital > 0 else 0
            exposure[symbol] = exposure.get(symbol, 0) + pct
        return exposure

    def get_exposure_by_sector(self) -> Dict[str, float]:
        """Obtiene la exposicion por sector"""
        exposure = {}
        for pos in self.positions:
            sector = pos.sector or self._get_sector_for_symbol(pos.symbol)
            if sector:
                pct = pos.value / self.total_capital if self.total_capital > 0 else 0
                exposure[sector] = exposure.get(sector, 0) + pct
        return exposure

    def _get_sector_for_symbol(self, symbol: str) -> Optional[str]:
        """Encuentra el sector para un simbolo"""
        for sector, symbols in self.config.sectors.items():
            if symbol in symbols:
                return sector
        return None

    def get_total_exposure(self) -> float:
        """Obtiene la exposicion total como porcentaje"""
        total = sum(pos.value for pos in self.positions)
        return total / self.total_capital if self.total_capital > 0 else 0

    def get_long_exposure(self) -> float:
        """Obtiene la exposicion long"""
        total = sum(pos.value for pos in self.positions if pos.side == "long")
        return total / self.total_capital if self.total_capital > 0 else 0

    def get_short_exposure(self) -> float:
        """Obtiene la exposicion short"""
        total = sum(pos.value for pos in self.positions if pos.side == "short")
        return total / self.total_capital if self.total_capital > 0 else 0

    def check_limits(self) -> List[RiskLimit]:
        """Verifica todos los limites y retorna los excedidos"""
        breached = []

        # Single asset exposure
        for symbol, exposure in self.get_exposure_by_symbol().items():
            limit = self.limits[0]  # Max Single Asset
            if limit.check(exposure):
                breached.append(RiskLimit(
                    name=f"Max Exposure {symbol}",
                    limit_type="exposure",
                    threshold=limit.threshold,
                    current_value=exposure,
                    action=limit.action,
                    is_breached=True,
                ))

        # Sector exposure
        for sector, exposure in self.get_exposure_by_sector().items():
            limit = self.limits[1]  # Max Sector
            if limit.check(exposure):
                breached.append(RiskLimit(
                    name=f"Max Sector {sector}",
                    limit_type="exposure",
                    threshold=limit.threshold,
                    current_value=exposure,
                    action=limit.action,
                    is_breached=True,
                ))

        # Total exposure
        total_exp = self.get_total_exposure()
        if self.limits[2].check(total_exp):
            breached.append(self.limits[2])

        # Long exposure
        long_exp = self.get_long_exposure()
        if self.limits[3].check(long_exp):
            breached.append(self.limits[3])

        # Short exposure
        short_exp = self.get_short_exposure()
        if self.limits[4].check(short_exp):
            breached.append(self.limits[4])

        return breached

    def can_open_position(
        self,
        symbol: str,
        side: str,
        size: float,
        price: float,
    ) -> Tuple[bool, List[str]]:
        """
        Verifica si se puede abrir una posicion sin exceder limites.

        Returns:
            Tuple[bool, List[str]]: (permitido, razones si no)
        """
        reasons = []
        new_value = size * price
        new_exposure = new_value / self.total_capital if self.total_capital > 0 else 0

        # Check single asset
        current_symbol_exp = self.get_exposure_by_symbol().get(symbol, 0)
        if current_symbol_exp + new_exposure > self.config.max_single_asset_exposure:
            reasons.append(
                f"Excede limite de exposicion por activo: "
                f"{(current_symbol_exp + new_exposure)*100:.1f}% > "
                f"{self.config.max_single_asset_exposure*100:.1f}%"
            )

        # Check sector
        sector = self._get_sector_for_symbol(symbol)
        if sector:
            current_sector_exp = self.get_exposure_by_sector().get(sector, 0)
            if current_sector_exp + new_exposure > self.config.max_sector_exposure:
                reasons.append(
                    f"Excede limite de sector {sector}: "
                    f"{(current_sector_exp + new_exposure)*100:.1f}% > "
                    f"{self.config.max_sector_exposure*100:.1f}%"
                )

        # Check total
        if self.get_total_exposure() + new_exposure > self.config.max_total_exposure:
            reasons.append(
                f"Excede exposicion total maxima: "
                f"{(self.get_total_exposure() + new_exposure)*100:.1f}% > "
                f"{self.config.max_total_exposure*100:.1f}%"
            )

        # Check long/short
        if side == "long":
            if self.get_long_exposure() + new_exposure > self.config.max_long_exposure:
                reasons.append("Excede exposicion long maxima")
        else:
            if self.get_short_exposure() + new_exposure > self.config.max_short_exposure:
                reasons.append("Excede exposicion short maxima")

        return len(reasons) == 0, reasons

    def get_max_allowed_size(
        self,
        symbol: str,
        side: str,
        price: float,
    ) -> float:
        """Calcula el tamano maximo permitido para una nueva posicion"""
        max_sizes = []

        # Por activo
        current_symbol_exp = self.get_exposure_by_symbol().get(symbol, 0)
        remaining_symbol = self.config.max_single_asset_exposure - current_symbol_exp
        if remaining_symbol > 0:
            max_sizes.append((remaining_symbol * self.total_capital) / price)

        # Por sector
        sector = self._get_sector_for_symbol(symbol)
        if sector:
            current_sector_exp = self.get_exposure_by_sector().get(sector, 0)
            remaining_sector = self.config.max_sector_exposure - current_sector_exp
            if remaining_sector > 0:
                max_sizes.append((remaining_sector * self.total_capital) / price)

        # Por total
        remaining_total = self.config.max_total_exposure - self.get_total_exposure()
        if remaining_total > 0:
            max_sizes.append((remaining_total * self.total_capital) / price)

        # Por long/short
        if side == "long":
            remaining_side = self.config.max_long_exposure - self.get_long_exposure()
        else:
            remaining_side = self.config.max_short_exposure - self.get_short_exposure()
        if remaining_side > 0:
            max_sizes.append((remaining_side * self.total_capital) / price)

        return min(max_sizes) if max_sizes else 0.0


class DrawdownProtection:
    """Proteccion contra drawdown"""

    def __init__(self, config: DrawdownConfig, initial_capital: float):
        self.config = config
        self.state = DrawdownState(peak_equity=initial_capital, current_equity=initial_capital)
        self.last_action_time: Optional[datetime] = None
        self.is_in_cooldown = False

    def update(self, equity: float) -> DrawdownAction:
        """
        Actualiza el estado de drawdown y retorna la accion recomendada.

        Returns:
            DrawdownAction: Accion a tomar basada en el drawdown actual
        """
        self.state.update(equity)

        # Verificar cooldown
        if self.is_in_cooldown and self.last_action_time:
            cooldown_end = self.last_action_time + timedelta(hours=self.config.cooldown_hours)
            if datetime.now() < cooldown_end:
                return DrawdownAction.PAUSE_TRADING
            else:
                self.is_in_cooldown = False

        dd_percent = self.state.drawdown_percent / 100  # Convertir a decimal

        # Determinar nivel y accion
        if dd_percent >= self.config.critical_level:
            self.last_action_time = datetime.now()
            self.is_in_cooldown = True
            return self.config.actions.get("critical", DrawdownAction.PAUSE_TRADING)
        elif dd_percent >= self.config.danger_level:
            return self.config.actions.get("danger", DrawdownAction.REDUCE_SIZE)
        elif dd_percent >= self.config.caution_level:
            return self.config.actions.get("caution", DrawdownAction.REDUCE_SIZE)
        elif dd_percent >= self.config.warning_level:
            return self.config.actions.get("warning", DrawdownAction.NONE)

        return DrawdownAction.NONE

    def get_size_multiplier(self) -> float:
        """Obtiene el multiplicador de tamano segun el drawdown actual"""
        dd_percent = self.state.drawdown_percent / 100

        if dd_percent >= self.config.critical_level:
            return self.config.size_reduction.get("critical", 0.0)
        elif dd_percent >= self.config.danger_level:
            return self.config.size_reduction.get("danger", 0.5)
        elif dd_percent >= self.config.caution_level:
            return self.config.size_reduction.get("caution", 0.75)
        elif dd_percent >= self.config.warning_level:
            return self.config.size_reduction.get("warning", 1.0)

        return 1.0

    def get_risk_level(self) -> RiskLevel:
        """Obtiene el nivel de riesgo actual"""
        dd_percent = self.state.drawdown_percent / 100

        if dd_percent >= self.config.critical_level:
            return RiskLevel.EXTREME
        elif dd_percent >= self.config.danger_level:
            return RiskLevel.HIGH
        elif dd_percent >= self.config.caution_level:
            return RiskLevel.MODERATE
        elif dd_percent >= self.config.warning_level:
            return RiskLevel.LOW

        return RiskLevel.MINIMAL

    def is_trading_allowed(self) -> bool:
        """Verifica si se permite trading"""
        action = self.update(self.state.current_equity)
        return action not in [DrawdownAction.PAUSE_TRADING, DrawdownAction.CLOSE_ALL]

    def should_close_all(self) -> bool:
        """Verifica si se deben cerrar todas las posiciones"""
        dd_percent = self.state.drawdown_percent / 100
        if dd_percent >= self.config.critical_level:
            return self.config.actions.get("critical") == DrawdownAction.CLOSE_ALL
        return False

    def reset(self, new_capital: float):
        """Reinicia el estado de drawdown"""
        self.state = DrawdownState(peak_equity=new_capital, current_equity=new_capital)
        self.last_action_time = None
        self.is_in_cooldown = False


class VaRCalculator:
    """Calculador de Value at Risk"""

    def __init__(self, config: VaRConfig):
        self.config = config
        self.returns_history: List[float] = []

    def update_returns(self, returns: List[float]):
        """Actualiza el historial de retornos"""
        self.returns_history = returns

    def add_return(self, daily_return: float):
        """Agrega un retorno diario"""
        self.returns_history.append(daily_return)
        # Mantener solo los ultimos N dias
        if len(self.returns_history) > self.config.lookback_days:
            self.returns_history = self.returns_history[-self.config.lookback_days:]

    def calculate_var(self, confidence: float = 0.95) -> float:
        """
        Calcula el VaR historico.

        Args:
            confidence: Nivel de confianza (0.95 = 95%)

        Returns:
            float: VaR como porcentaje negativo
        """
        if len(self.returns_history) < 10:
            return 0.0

        returns = np.array(self.returns_history)
        var: float = 0.0

        if self.config.method == "historical":
            var = float(np.percentile(returns, (1 - confidence) * 100))
        elif self.config.method == "parametric":
            # VaR parametrico (asume distribucion normal)
            mean = float(np.mean(returns))
            std = float(np.std(returns))
            from scipy import stats
            z_score = float(stats.norm.ppf(1 - confidence))
            var = mean + z_score * std
        else:
            # Por defecto, historico
            var = float(np.percentile(returns, (1 - confidence) * 100))

        return var

    def calculate_cvar(self, confidence: float = 0.95) -> float:
        """
        Calcula el CVaR (Conditional VaR / Expected Shortfall).

        Es el promedio de las perdidas que exceden el VaR.
        """
        if len(self.returns_history) < 10:
            return 0.0

        returns = np.array(self.returns_history)
        var = self.calculate_var(confidence)

        # Retornos peores que el VaR
        tail_returns = returns[returns <= var]

        if len(tail_returns) == 0:
            return var

        return float(np.mean(tail_returns))

    def get_var_metrics(self) -> Dict[str, float]:
        """Obtiene todas las metricas de VaR"""
        metrics = {}

        for conf in self.config.confidence_levels:
            conf_str = f"{int(conf * 100)}"
            metrics[f"var_{conf_str}"] = self.calculate_var(conf)
            metrics[f"cvar_{conf_str}"] = self.calculate_cvar(conf)

        return metrics

    def check_var_limits(self) -> List[RiskLimit]:
        """Verifica si los limites de VaR fueron excedidos"""
        breached = []

        var_95 = abs(self.calculate_var(0.95))
        if var_95 > self.config.max_var_95:
            breached.append(RiskLimit(
                name="VaR 95%",
                limit_type="var",
                threshold=self.config.max_var_95,
                current_value=var_95,
                action=DrawdownAction.REDUCE_SIZE,
                is_breached=True,
            ))

        var_99 = abs(self.calculate_var(0.99))
        if var_99 > self.config.max_var_99:
            breached.append(RiskLimit(
                name="VaR 99%",
                limit_type="var",
                threshold=self.config.max_var_99,
                current_value=var_99,
                action=DrawdownAction.REDUCE_SIZE,
                is_breached=True,
            ))

        return breached


class RiskLimitChecker:
    """
    Verificador centralizado de limites de riesgo.

    Coordina todos los componentes de limites.
    """

    def __init__(
        self,
        exposure_config: ExposureLimitsConfig,
        drawdown_config: DrawdownConfig,
        var_config: VaRConfig,
        initial_capital: float,
    ):
        self.exposure_manager = ExposureLimitManager(exposure_config, initial_capital)
        self.drawdown_protection = DrawdownProtection(drawdown_config, initial_capital)
        self.var_calculator = VaRCalculator(var_config)
        self.initial_capital = initial_capital

    def update(
        self,
        equity: float,
        positions: List[Position],
        daily_return: Optional[float] = None,
    ):
        """Actualiza todos los componentes"""
        self.exposure_manager.update_capital(equity)
        self.exposure_manager.update_positions(positions)
        self.drawdown_protection.update(equity)

        if daily_return is not None:
            self.var_calculator.add_return(daily_return)

    def check_all_limits(self) -> RiskMetrics:
        """Verifica todos los limites y retorna metricas"""
        metrics = RiskMetrics()

        # Exposicion
        metrics.total_exposure = self.exposure_manager.get_total_exposure()
        metrics.exposure_by_symbol = self.exposure_manager.get_exposure_by_symbol()
        metrics.exposure_by_sector = self.exposure_manager.get_exposure_by_sector()

        # Drawdown
        dd_state = self.drawdown_protection.state
        metrics.current_drawdown = dd_state.drawdown_percent / 100
        metrics.max_drawdown = dd_state.max_drawdown_percent / 100
        metrics.drawdown_duration_days = dd_state.days_in_drawdown

        # VaR
        var_metrics = self.var_calculator.get_var_metrics()
        metrics.var_95 = var_metrics.get("var_95", 0)
        metrics.var_99 = var_metrics.get("var_99", 0)
        metrics.cvar_95 = var_metrics.get("cvar_95", 0)

        # Estado general
        metrics.risk_level = self.drawdown_protection.get_risk_level()
        metrics.is_trading_allowed = self.drawdown_protection.is_trading_allowed()

        # Limites excedidos
        breached = []
        breached.extend([l.name for l in self.exposure_manager.check_limits()])
        breached.extend([l.name for l in self.var_calculator.check_var_limits()])
        metrics.active_limits_breached = breached

        # Capital
        metrics.total_equity = self.drawdown_protection.state.current_equity
        metrics.available_capital = metrics.total_equity * (1 - metrics.total_exposure)

        return metrics

    def can_open_position(
        self,
        symbol: str,
        side: str,
        size: float,
        price: float,
    ) -> Tuple[bool, List[str]]:
        """Verifica si se puede abrir una posicion"""
        reasons = []

        # Verificar si trading esta permitido
        if not self.drawdown_protection.is_trading_allowed():
            reasons.append(
                f"Trading pausado por drawdown: "
                f"{self.drawdown_protection.state.drawdown_percent:.1f}%"
            )
            return False, reasons

        # Verificar limites de exposicion
        _, exp_reasons = self.exposure_manager.can_open_position(
            symbol, side, size, price
        )
        reasons.extend(exp_reasons)

        return len(reasons) == 0, reasons

    def get_adjusted_size(
        self,
        symbol: str,
        side: str,
        requested_size: float,
        price: float,
    ) -> float:
        """Obtiene el tamano ajustado segun los limites"""
        # Multiplicador por drawdown
        dd_multiplier = self.drawdown_protection.get_size_multiplier()

        # Tamano maximo por exposicion
        max_exposure_size = self.exposure_manager.get_max_allowed_size(symbol, side, price)

        # Tamano final
        adjusted = requested_size * dd_multiplier
        adjusted = min(adjusted, max_exposure_size)

        return max(adjusted, 0)

    def reset(self, new_capital: float):
        """Reinicia todos los componentes"""
        self.initial_capital = new_capital
        self.exposure_manager.update_capital(new_capital)
        self.exposure_manager.positions = []
        self.drawdown_protection.reset(new_capital)
        self.var_calculator.returns_history = []
