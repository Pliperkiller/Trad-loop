"""
Calculadores de position sizing.

Implementa multiples metodos de position sizing:
- Fixed Fractional
- Kelly Criterion
- Optimal-f
- ATR-based
- Volatility Adjusted
"""

from typing import List, Optional
from dataclasses import dataclass
import numpy as np

from .models import PositionSizeResult, SizingMethod
from .config import PositionSizingConfig


@dataclass
class TradeHistory:
    """Historial de trades para calculos estadisticos"""
    returns: List[float]  # Retornos porcentuales
    wins: int = 0
    losses: int = 0

    @property
    def total_trades(self) -> int:
        return self.wins + self.losses

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.wins / self.total_trades

    @property
    def avg_win(self) -> float:
        wins = [r for r in self.returns if r > 0]
        return sum(wins) / len(wins) if wins else 0.0

    @property
    def avg_loss(self) -> float:
        losses = [abs(r) for r in self.returns if r < 0]
        return sum(losses) / len(losses) if losses else 0.0


class PositionSizer:
    """Calculador de position sizing"""

    def __init__(self, config: PositionSizingConfig):
        self.config = config

    def calculate(
        self,
        symbol: str,
        capital: float,
        entry_price: float,
        stop_loss: Optional[float] = None,
        trade_history: Optional[TradeHistory] = None,
        volatility: Optional[float] = None,
        atr: Optional[float] = None,
    ) -> PositionSizeResult:
        """
        Calcula el tamano de posicion recomendado.

        Args:
            symbol: Simbolo del activo
            capital: Capital disponible
            entry_price: Precio de entrada
            stop_loss: Precio de stop loss (requerido para algunos metodos)
            trade_history: Historial de trades (requerido para Kelly/Optimal-f)
            volatility: Volatilidad del activo (para volatility-adjusted)
            atr: Average True Range (para ATR-based)

        Returns:
            PositionSizeResult con el tamano recomendado
        """
        method = self.config.method

        if method == SizingMethod.FIXED_FRACTIONAL:
            result = self._fixed_fractional(symbol, capital, entry_price, stop_loss)
        elif method == SizingMethod.KELLY:
            result = self._kelly_criterion(symbol, capital, entry_price, trade_history)
        elif method == SizingMethod.OPTIMAL_F:
            result = self._optimal_f(symbol, capital, entry_price, trade_history)
        elif method == SizingMethod.ATR_BASED:
            result = self._atr_based(symbol, capital, entry_price, atr)
        elif method == SizingMethod.VOLATILITY_ADJUSTED:
            result = self._volatility_adjusted(symbol, capital, entry_price, volatility)
        elif method == SizingMethod.EQUAL_WEIGHT:
            result = self._equal_weight(symbol, capital, entry_price)
        else:
            result = self._fixed_fractional(symbol, capital, entry_price, stop_loss)

        # Aplicar limites
        result = self._apply_limits(result, capital, entry_price)

        return result

    def _fixed_fractional(
        self,
        symbol: str,
        capital: float,
        entry_price: float,
        stop_loss: Optional[float] = None,
    ) -> PositionSizeResult:
        """
        Fixed Fractional: Arriesga un porcentaje fijo del capital.

        Si hay stop loss:
            Size = (Capital * Risk%) / (Entry - StopLoss)
        Si no hay stop loss:
            Size = (Capital * Risk%) / Entry
        """
        risk_amount = capital * self.config.risk_per_trade

        if stop_loss and stop_loss > 0 and entry_price != stop_loss:
            risk_per_unit = abs(entry_price - stop_loss)
            size = risk_amount / risk_per_unit
        else:
            # Sin stop loss, usar el porcentaje directamente
            size = (capital * self.config.fixed_fraction) / entry_price

        return PositionSizeResult(
            symbol=symbol,
            recommended_size=size,
            max_allowed_size=size * 2,  # Se ajustara despues
            risk_amount=risk_amount,
            sizing_method=SizingMethod.FIXED_FRACTIONAL,
            confidence=0.9,  # Alta confianza, metodo simple
        )

    def _kelly_criterion(
        self,
        symbol: str,
        capital: float,
        entry_price: float,
        trade_history: Optional[TradeHistory] = None,
    ) -> PositionSizeResult:
        """
        Kelly Criterion: f* = (bp - q) / b

        Donde:
        - b = ratio win/loss (avg_win / avg_loss)
        - p = probabilidad de ganar (win_rate)
        - q = probabilidad de perder (1 - p)

        Se usa Half-Kelly por seguridad.
        """
        warnings = []

        if not trade_history or trade_history.total_trades < self.config.min_trades_for_kelly:
            warnings.append(
                f"Insuficientes trades ({trade_history.total_trades if trade_history else 0}), "
                f"usando Fixed Fractional"
            )
            result = self._fixed_fractional(symbol, capital, entry_price)
            result.warnings = warnings
            result.confidence = 0.5
            return result

        win_rate = trade_history.win_rate
        avg_win = trade_history.avg_win
        avg_loss = trade_history.avg_loss

        if avg_loss == 0:
            warnings.append("No hay perdidas en historial, usando Fixed Fractional")
            result = self._fixed_fractional(symbol, capital, entry_price)
            result.warnings = warnings
            result.confidence = 0.5
            return result

        # Calcular Kelly
        b = avg_win / avg_loss  # Win/Loss ratio
        p = win_rate
        q = 1 - p

        kelly_fraction = (b * p - q) / b

        if kelly_fraction <= 0:
            warnings.append("Kelly negativo, la estrategia no es rentable")
            kelly_fraction = 0.01  # Minimo
            confidence = 0.3
        else:
            # Aplicar Half-Kelly o fraccion configurada
            kelly_fraction *= self.config.kelly_fraction
            confidence = min(0.95, 0.5 + (trade_history.total_trades / 200))

        # Limitar Kelly
        kelly_fraction = min(kelly_fraction, self.config.max_position_percent)
        kelly_fraction = max(kelly_fraction, self.config.min_position_percent)

        size = (capital * kelly_fraction) / entry_price
        risk_amount = capital * kelly_fraction

        result = PositionSizeResult(
            symbol=symbol,
            recommended_size=size,
            max_allowed_size=size * 1.5,
            risk_amount=risk_amount,
            sizing_method=SizingMethod.KELLY,
            confidence=confidence,
            adjustments={
                "raw_kelly": f"{(b * p - q) / b * 100:.2f}%",
                "adjusted_kelly": f"{kelly_fraction * 100:.2f}%",
                "win_rate": f"{win_rate * 100:.1f}%",
                "win_loss_ratio": f"{b:.2f}",
            },
            warnings=warnings,
        )

        return result

    def _optimal_f(
        self,
        symbol: str,
        capital: float,
        entry_price: float,
        trade_history: Optional[TradeHistory] = None,
    ) -> PositionSizeResult:
        """
        Optimal-f: Encuentra la fraccion que maximiza el crecimiento geometrico.

        TWR = Producto((1 + f * (-trade/max_loss))) para todos los trades

        Busca el f que maximiza TWR.
        """
        warnings = []

        if not trade_history or len(trade_history.returns) < self.config.min_trades_for_kelly:
            warnings.append("Insuficientes trades, usando Fixed Fractional")
            result = self._fixed_fractional(symbol, capital, entry_price)
            result.warnings = warnings
            return result

        returns = trade_history.returns
        max_loss = max(abs(min(returns)), 0.001)  # Mayor perdida

        best_f = 0.01
        best_twr = 0

        # Buscar optimal-f
        for f in np.arange(0.01, 0.50, 0.01):
            twr = 1.0
            for r in returns:
                # HPR = 1 + f * (-trade / max_loss)
                hpr = 1 + f * (r / max_loss)
                if hpr <= 0:
                    twr = 0
                    break
                twr *= hpr

            if twr > best_twr:
                best_twr = twr
                best_f = f

        # Aplicar fraccion de seguridad (similar a half-kelly)
        optimal_fraction = best_f * self.config.kelly_fraction

        # Limitar
        optimal_fraction = min(optimal_fraction, self.config.max_position_percent)
        optimal_fraction = max(optimal_fraction, self.config.min_position_percent)

        size = (capital * optimal_fraction) / entry_price
        risk_amount = capital * optimal_fraction

        return PositionSizeResult(
            symbol=symbol,
            recommended_size=size,
            max_allowed_size=size * 1.5,
            risk_amount=risk_amount,
            sizing_method=SizingMethod.OPTIMAL_F,
            confidence=0.85,
            adjustments={
                "raw_optimal_f": f"{best_f * 100:.2f}%",
                "adjusted_optimal_f": f"{optimal_fraction * 100:.2f}%",
                "max_loss": f"{max_loss * 100:.2f}%",
                "twr": f"{best_twr:.4f}",
            },
            warnings=warnings,
        )

    def _atr_based(
        self,
        symbol: str,
        capital: float,
        entry_price: float,
        atr: Optional[float] = None,
    ) -> PositionSizeResult:
        """
        ATR-based: Ajusta el tamano segun la volatilidad del activo.

        Size = (Capital * Risk%) / (ATR * Multiplier)
        """
        warnings = []

        if atr is None or atr <= 0:
            warnings.append("ATR no disponible, usando Fixed Fractional")
            result = self._fixed_fractional(symbol, capital, entry_price)
            result.warnings = warnings
            return result

        risk_amount = capital * self.config.risk_per_trade
        risk_per_unit = atr * self.config.atr_multiplier
        size = risk_amount / risk_per_unit

        return PositionSizeResult(
            symbol=symbol,
            recommended_size=size,
            max_allowed_size=size * 1.5,
            risk_amount=risk_amount,
            sizing_method=SizingMethod.ATR_BASED,
            confidence=0.85,
            adjustments={
                "atr": f"{atr:.4f}",
                "atr_multiplier": f"{self.config.atr_multiplier}x",
                "stop_distance": f"{risk_per_unit:.4f}",
            },
            warnings=warnings,
        )

    def _volatility_adjusted(
        self,
        symbol: str,
        capital: float,
        entry_price: float,
        volatility: Optional[float] = None,
    ) -> PositionSizeResult:
        """
        Volatility Adjusted: Ajusta el tamano para mantener volatilidad constante.

        Size = (Capital * Target_Vol) / (Asset_Vol * Entry_Price)
        """
        warnings = []

        if volatility is None or volatility <= 0:
            warnings.append("Volatilidad no disponible, usando Fixed Fractional")
            result = self._fixed_fractional(symbol, capital, entry_price)
            result.warnings = warnings
            return result

        target_vol = self.config.target_volatility
        vol_ratio = target_vol / volatility

        # Limitar el ratio
        vol_ratio = min(vol_ratio, 2.0)  # Max 2x
        vol_ratio = max(vol_ratio, 0.25)  # Min 0.25x

        base_fraction = self.config.fixed_fraction
        adjusted_fraction = base_fraction * vol_ratio

        size = (capital * adjusted_fraction) / entry_price
        risk_amount = capital * adjusted_fraction

        return PositionSizeResult(
            symbol=symbol,
            recommended_size=size,
            max_allowed_size=size * 1.5,
            risk_amount=risk_amount,
            sizing_method=SizingMethod.VOLATILITY_ADJUSTED,
            confidence=0.80,
            adjustments={
                "asset_volatility": f"{volatility * 100:.1f}%",
                "target_volatility": f"{target_vol * 100:.1f}%",
                "vol_ratio": f"{vol_ratio:.2f}x",
            },
            warnings=warnings,
        )

    def _equal_weight(
        self,
        symbol: str,
        capital: float,
        entry_price: float,
    ) -> PositionSizeResult:
        """
        Equal Weight: Divide el capital equitativamente entre posiciones.
        """
        # Asumimos max 10 posiciones si no se especifica
        num_positions = 10
        fraction = 1.0 / num_positions

        size = (capital * fraction) / entry_price
        risk_amount = capital * fraction

        return PositionSizeResult(
            symbol=symbol,
            recommended_size=size,
            max_allowed_size=size * 1.5,
            risk_amount=risk_amount,
            sizing_method=SizingMethod.EQUAL_WEIGHT,
            confidence=0.95,
            adjustments={
                "num_positions": str(num_positions),
                "weight_per_position": f"{fraction * 100:.1f}%",
            },
        )

    def _apply_limits(
        self,
        result: PositionSizeResult,
        capital: float,
        entry_price: float,
    ) -> PositionSizeResult:
        """Aplica limites de posicion al resultado"""
        max_value = capital * self.config.max_position_percent
        min_value = capital * self.config.min_position_percent

        if self.config.max_position_value:
            max_value = min(max_value, self.config.max_position_value)

        max_size = max_value / entry_price
        min_size = min_value / entry_price

        result.max_allowed_size = max_size

        # Ajustar si excede el maximo
        if result.recommended_size > max_size:
            result.adjustments["limit_applied"] = f"Reducido de {result.recommended_size:.6f} a {max_size:.6f}"
            result.recommended_size = max_size
            result.warnings.append("Tamano reducido por limite maximo")

        # Ajustar si es menor al minimo
        if result.recommended_size < min_size:
            if result.recommended_size > 0:
                result.adjustments["min_applied"] = f"Aumentado de {result.recommended_size:.6f} a {min_size:.6f}"
                result.recommended_size = min_size

        return result


class PositionSizerFactory:
    """Factory para crear position sizers con diferentes configuraciones"""

    @staticmethod
    def create_conservative() -> PositionSizer:
        """Crea un sizer conservador"""
        config = PositionSizingConfig(
            method=SizingMethod.FIXED_FRACTIONAL,
            risk_per_trade=0.01,
            max_position_percent=0.10,
        )
        return PositionSizer(config)

    @staticmethod
    def create_moderate() -> PositionSizer:
        """Crea un sizer moderado"""
        config = PositionSizingConfig(
            method=SizingMethod.FIXED_FRACTIONAL,
            risk_per_trade=0.02,
            max_position_percent=0.20,
        )
        return PositionSizer(config)

    @staticmethod
    def create_aggressive() -> PositionSizer:
        """Crea un sizer agresivo con Kelly"""
        config = PositionSizingConfig(
            method=SizingMethod.KELLY,
            risk_per_trade=0.03,
            max_position_percent=0.30,
            kelly_fraction=0.5,
        )
        return PositionSizer(config)

    @staticmethod
    def create_atr_based(atr_multiplier: float = 2.0) -> PositionSizer:
        """Crea un sizer basado en ATR"""
        config = PositionSizingConfig(
            method=SizingMethod.ATR_BASED,
            risk_per_trade=0.02,
            atr_multiplier=atr_multiplier,
        )
        return PositionSizer(config)
