"""
Implementaciones por defecto de las interfaces.

Estas implementaciones proporcionan funcionalidad estándar
y pueden ser reemplazadas por implementaciones personalizadas
mediante dependency injection.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import logging

from .protocols import (
    IDataValidator,
    IMetricsCalculator,
    IPositionSizer,
    IRiskManager,
    ValidationResult,
    TradeResult,
    PerformanceMetrics,
    PositionInfo,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Default Data Validator
# ============================================================================

class DefaultDataValidator:
    """
    Validador de datos OHLCV por defecto.

    Implementa IDataValidator con validaciones estándar.
    """

    def __init__(self, strict: bool = True):
        """
        Args:
            strict: Si True, cualquier error hace fallar la validación
        """
        self.strict = strict

    def validate(self, data: pd.DataFrame) -> ValidationResult:
        """Valida datos OHLCV."""
        errors = []
        warnings = []

        # Verificar columnas requeridas
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing = [c for c in required_columns if c not in data.columns]
        if missing:
            errors.append(f"Columnas faltantes: {missing}")
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings)

        # Verificar DataFrame vacío
        if len(data) == 0:
            errors.append("DataFrame vacío")
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings)

        # Verificar integridad OHLC
        invalid_high_low = (data['high'] < data['low']).sum()
        if invalid_high_low > 0:
            msg = f"Datos inválidos: high < low en {invalid_high_low} filas"
            if self.strict:
                errors.append(msg)
            else:
                warnings.append(msg)

        # Verificar valores no positivos
        for col in ['open', 'high', 'low', 'close']:
            non_positive = (data[col] <= 0).sum()
            if non_positive > 0:
                msg = f"Valores no positivos en '{col}': {non_positive} filas"
                if self.strict:
                    errors.append(msg)
                else:
                    warnings.append(msg)

        # Verificar NaN
        nan_counts = data[required_columns].isna().sum()
        nan_cols = nan_counts[nan_counts > 0]
        if len(nan_cols) > 0:
            msg = f"Valores NaN encontrados: {nan_cols.to_dict()}"
            warnings.append(msg)

        # Verificar Inf
        numeric_cols = data[required_columns].select_dtypes(include=[np.number])
        inf_counts = np.isinf(numeric_cols).sum()
        inf_cols = inf_counts[inf_counts > 0]
        if len(inf_cols) > 0:
            msg = f"Valores Inf encontrados: {inf_cols.to_dict()}"
            if self.strict:
                errors.append(msg)
            else:
                warnings.append(msg)

        is_valid = len(errors) == 0
        return ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings)

    def sanitize(self, data: pd.DataFrame) -> pd.DataFrame:
        """Limpia datos removiendo filas problemáticas."""
        df = data.copy()
        required_columns = ['open', 'high', 'low', 'close', 'volume']

        initial_len = len(df)

        # Remover NaN
        df = df.dropna(subset=required_columns)

        # Remover valores no positivos
        for col in ['open', 'high', 'low', 'close']:
            df = df[df[col] > 0]

        # Remover volumen negativo
        df = df[df['volume'] >= 0]

        # Remover Inf
        for col in required_columns:
            if col in df.columns and df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                df = df[~np.isinf(df[col])]

        # Corregir OHLC
        df['high'] = df[['open', 'high', 'low', 'close']].max(axis=1)
        df['low'] = df[['open', 'high', 'low', 'close']].min(axis=1)

        removed = initial_len - len(df)
        if removed > 0:
            logger.warning(f"DefaultDataValidator.sanitize: removidas {removed} filas")

        return df


# ============================================================================
# Default Metrics Calculator
# ============================================================================

class DefaultMetricsCalculator:
    """
    Calculador de métricas por defecto.

    Implementa IMetricsCalculator con métricas estándar de trading.
    """

    def __init__(self, risk_free_rate: float = 0.0, periods_per_year: int = 252):
        """
        Args:
            risk_free_rate: Tasa libre de riesgo anualizada
            periods_per_year: Períodos por año (252 para diario, 52 para semanal)
        """
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year

    def calculate(
        self,
        trades: List[TradeResult],
        equity_curve: List[float],
        initial_capital: float
    ) -> PerformanceMetrics:
        """Calcula métricas de rendimiento."""

        # Métricas de trades
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t.pnl > 0)
        losing_trades = sum(1 for t in trades if t.pnl < 0)

        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0

        wins = [t.pnl for t in trades if t.pnl > 0]
        losses = [t.pnl for t in trades if t.pnl < 0]

        avg_win = np.mean(wins) if wins else 0.0
        avg_loss = np.mean(losses) if losses else 0.0

        # Profit factor
        total_wins = sum(wins) if wins else 0.0
        total_losses = abs(sum(losses)) if losses else 0.0

        if total_losses > 0:
            profit_factor = total_wins / total_losses
        else:
            profit_factor = float('inf') if total_wins > 0 else 0.0

        # Expectancy
        if total_trades > 0:
            expectancy = (win_rate / 100 * avg_win) + ((1 - win_rate / 100) * avg_loss)
        else:
            expectancy = 0.0

        # Total return
        if initial_capital > 0 and equity_curve:
            total_return_pct = ((equity_curve[-1] - initial_capital) / initial_capital) * 100
        else:
            total_return_pct = 0.0

        # Drawdown
        equity_series = pd.Series(equity_curve) if equity_curve else pd.Series([initial_capital])
        rolling_max = equity_series.expanding().max()

        with np.errstate(divide='ignore', invalid='ignore'):
            drawdown = np.where(
                rolling_max > 0,
                (equity_series - rolling_max) / rolling_max * 100,
                0.0
            )
        max_drawdown_pct = np.nanmin(drawdown) if len(drawdown) > 0 else 0.0

        # Returns for ratios
        returns = equity_series.pct_change().dropna()

        # Sharpe ratio
        if len(returns) > 0:
            std_returns = returns.std()
            if std_returns > 0 and np.isfinite(std_returns):
                excess_return = returns.mean() - (self.risk_free_rate / self.periods_per_year)
                sharpe_ratio = (excess_return / std_returns) * np.sqrt(self.periods_per_year)
            else:
                sharpe_ratio = 0.0
        else:
            sharpe_ratio = 0.0

        # Sortino ratio (downside deviation)
        if len(returns) > 0:
            negative_returns = returns[returns < 0]
            if len(negative_returns) > 0:
                downside_std = negative_returns.std()
                if downside_std > 0 and np.isfinite(downside_std):
                    excess_return = returns.mean() - (self.risk_free_rate / self.periods_per_year)
                    sortino_ratio = (excess_return / downside_std) * np.sqrt(self.periods_per_year)
                else:
                    sortino_ratio = 0.0
            else:
                sortino_ratio = float('inf') if returns.mean() > 0 else 0.0
        else:
            sortino_ratio = 0.0

        # Calmar ratio (return / max drawdown)
        if max_drawdown_pct < 0:
            calmar_ratio = total_return_pct / abs(max_drawdown_pct)
        else:
            calmar_ratio = float('inf') if total_return_pct > 0 else 0.0

        return PerformanceMetrics(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_return_pct=total_return_pct,
            max_drawdown_pct=max_drawdown_pct,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            avg_win=avg_win,
            avg_loss=avg_loss,
            expectancy=expectancy
        )


# ============================================================================
# Default Position Sizer
# ============================================================================

class FixedFractionalSizer:
    """
    Position sizer con fixed fractional.

    Riesga un porcentaje fijo del capital en cada trade.
    """

    def __init__(self, risk_pct: float = 2.0, max_position_pct: float = 95.0):
        """
        Args:
            risk_pct: Porcentaje del capital a arriesgar por trade
            max_position_pct: Máximo porcentaje del capital por posición
        """
        self.risk_pct = risk_pct
        self.max_position_pct = max_position_pct

    def calculate_size(
        self,
        capital: float,
        price: float,
        stop_loss: float,
        volatility: Optional[float] = None
    ) -> float:
        """Calcula tamaño de posición."""
        if price <= 0 or capital <= 0:
            return 0.0

        if stop_loss < 0:
            return 0.0

        risk_amount = capital * (self.risk_pct / 100)
        risk_per_unit = abs(price - stop_loss)

        if risk_per_unit < 1e-10:
            return 0.0

        position_size = risk_amount / risk_per_unit

        # Limitar por porcentaje máximo
        max_position = capital * (self.max_position_pct / 100) / price
        position_size = min(position_size, max_position)

        if not np.isfinite(position_size):
            return 0.0

        return position_size


class KellySizer:
    """
    Position sizer basado en Kelly Criterion.

    Calcula tamaño óptimo basado en win rate y ratio win/loss.
    """

    def __init__(
        self,
        win_rate: float = 0.5,
        avg_win: float = 1.0,
        avg_loss: float = 1.0,
        kelly_fraction: float = 0.5,  # Half-Kelly por seguridad
        max_position_pct: float = 25.0
    ):
        """
        Args:
            win_rate: Tasa de aciertos (0-1)
            avg_win: Ganancia promedio
            avg_loss: Pérdida promedio (valor absoluto)
            kelly_fraction: Fracción del Kelly completo a usar
            max_position_pct: Máximo porcentaje del capital
        """
        self.win_rate = win_rate
        self.avg_win = avg_win
        self.avg_loss = abs(avg_loss)
        self.kelly_fraction = kelly_fraction
        self.max_position_pct = max_position_pct

    def calculate_size(
        self,
        capital: float,
        price: float,
        stop_loss: float,
        volatility: Optional[float] = None
    ) -> float:
        """Calcula tamaño usando Kelly."""
        if price <= 0 or capital <= 0 or self.avg_loss == 0:
            return 0.0

        # Kelly formula: f = (bp - q) / b
        # b = win/loss ratio, p = win rate, q = 1 - p
        b = self.avg_win / self.avg_loss if self.avg_loss > 0 else 1.0
        p = self.win_rate
        q = 1 - p

        kelly_pct = (b * p - q) / b if b > 0 else 0.0

        # Aplicar fracción y limitar
        kelly_pct = max(0, kelly_pct * self.kelly_fraction)
        kelly_pct = min(kelly_pct, self.max_position_pct / 100)

        position_value = capital * kelly_pct
        position_size = position_value / price

        return position_size


class VolatilitySizer:
    """
    Position sizer basado en volatilidad.

    Ajusta el tamaño inversamente proporcional a la volatilidad.
    """

    def __init__(
        self,
        target_risk_pct: float = 2.0,
        atr_multiplier: float = 2.0,
        max_position_pct: float = 50.0
    ):
        """
        Args:
            target_risk_pct: Riesgo objetivo por trade
            atr_multiplier: Multiplicador de ATR para stop
            max_position_pct: Máximo porcentaje del capital
        """
        self.target_risk_pct = target_risk_pct
        self.atr_multiplier = atr_multiplier
        self.max_position_pct = max_position_pct

    def calculate_size(
        self,
        capital: float,
        price: float,
        stop_loss: float,
        volatility: Optional[float] = None
    ) -> float:
        """Calcula tamaño basado en volatilidad."""
        if price <= 0 or capital <= 0:
            return 0.0

        # Si no hay volatilidad, usar distancia al stop
        if volatility is None or volatility <= 0:
            risk_per_unit = abs(price - stop_loss)
        else:
            # Usar volatilidad como base del stop
            risk_per_unit = volatility * self.atr_multiplier

        if risk_per_unit < 1e-10:
            return 0.0

        risk_amount = capital * (self.target_risk_pct / 100)
        position_size = risk_amount / risk_per_unit

        # Limitar
        max_position = capital * (self.max_position_pct / 100) / price
        position_size = min(position_size, max_position)

        return position_size if np.isfinite(position_size) else 0.0


# ============================================================================
# Default Risk Manager
# ============================================================================

class DefaultRiskManager:
    """
    Risk manager por defecto.

    Implementa controles de riesgo estándar.
    """

    def __init__(
        self,
        max_positions: int = 5,
        max_position_pct: float = 25.0,
        max_total_exposure_pct: float = 100.0,
        max_drawdown_pct: float = 25.0,
        default_stop_pct: float = 2.0
    ):
        """
        Args:
            max_positions: Máximo número de posiciones simultáneas
            max_position_pct: Máximo por posición (% del capital)
            max_total_exposure_pct: Máximo exposure total
            max_drawdown_pct: Drawdown máximo permitido
            default_stop_pct: Porcentaje default para stop loss
        """
        self.max_positions = max_positions
        self.max_position_pct = max_position_pct
        self.max_total_exposure_pct = max_total_exposure_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.default_stop_pct = default_stop_pct

    def can_open_position(
        self,
        symbol: str,
        side: str,
        size: float,
        price: float,
        current_positions: List[PositionInfo],
        current_equity: float
    ) -> Tuple[bool, str]:
        """Verifica si se puede abrir posición."""

        # Verificar número de posiciones
        if len(current_positions) >= self.max_positions:
            return False, f"Máximo de posiciones alcanzado ({self.max_positions})"

        # Verificar tamaño de posición
        position_value = size * price
        position_pct = (position_value / current_equity * 100) if current_equity > 0 else 100

        if position_pct > self.max_position_pct:
            return False, f"Posición excede límite ({position_pct:.1f}% > {self.max_position_pct}%)"

        # Verificar exposure total
        total_exposure = sum(p.quantity * p.current_price for p in current_positions)
        new_exposure = total_exposure + position_value
        exposure_pct = (new_exposure / current_equity * 100) if current_equity > 0 else 100

        if exposure_pct > self.max_total_exposure_pct:
            return False, f"Exposure total excede límite ({exposure_pct:.1f}%)"

        return True, "OK"

    def calculate_stop_loss(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        atr: Optional[float] = None
    ) -> float:
        """Calcula stop loss."""
        if atr is not None and atr > 0:
            # Stop basado en ATR
            stop_distance = atr * 2
        else:
            # Stop basado en porcentaje
            stop_distance = entry_price * (self.default_stop_pct / 100)

        if side == 'LONG':
            return entry_price - stop_distance
        else:  # SHORT
            return entry_price + stop_distance
