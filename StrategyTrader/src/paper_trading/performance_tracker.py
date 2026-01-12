"""
Realtime Performance Tracker

Calcula metricas de performance en tiempo real.
Extiende el PerformanceAnalyzer existente para soportar
actualizaciones incrementales.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from .models import TradeRecord, PaperTradingState, PositionSide
from .config import PaperTradingConfig


logger = logging.getLogger(__name__)


class RealtimePerformanceTracker:
    """
    Tracker de metricas de performance en tiempo real.

    A diferencia de PerformanceAnalyzer que analiza datos historicos,
    este tracker se actualiza incrementalmente con cada trade.

    Example:
        tracker = RealtimePerformanceTracker(config)

        # Agregar trade
        tracker.add_trade(trade_record)

        # Actualizar equity
        tracker.update_equity(11000)

        # Obtener metricas
        metrics = tracker.get_metrics()

    Attributes:
        config: Configuracion de paper trading
        initial_capital: Capital inicial
        equity_curve: Lista de valores de equity
    """

    def __init__(self, config: PaperTradingConfig):
        """
        Inicializa el tracker.

        Args:
            config: Configuracion de paper trading
        """
        self.config = config
        self.initial_capital = config.initial_balance

        self._equity_curve: List[float] = [config.initial_balance]
        self._equity_timestamps: List[datetime] = [datetime.now()]
        self._trades: List[TradeRecord] = []
        self._peak_equity: float = config.initial_balance
        self._max_drawdown: float = 0.0
        self._current_drawdown: float = 0.0

        # Metricas en cache
        self._cached_metrics: Optional[Dict] = None
        self._metrics_dirty: bool = True

    def add_trade(self, trade: TradeRecord):
        """
        Agrega un trade al historial.

        Args:
            trade: Registro del trade cerrado
        """
        self._trades.append(trade)
        self._metrics_dirty = True

        logger.debug(f"Trade agregado: {trade.id} - PnL: {trade.pnl:+.2f}")

    def update_equity(self, current_equity: float):
        """
        Actualiza la curva de equity.

        Args:
            current_equity: Valor actual de equity
        """
        self._equity_curve.append(current_equity)
        self._equity_timestamps.append(datetime.now())

        # Actualizar peak y drawdown
        if current_equity > self._peak_equity:
            self._peak_equity = current_equity

        if self._peak_equity > 0:
            self._current_drawdown = (self._peak_equity - current_equity) / self._peak_equity
            self._max_drawdown = max(self._max_drawdown, self._current_drawdown)

        self._metrics_dirty = True

    def get_metrics(self) -> Dict:
        """
        Obtiene todas las metricas de performance.

        Returns:
            Diccionario con metricas
        """
        if not self._metrics_dirty and self._cached_metrics:
            return self._cached_metrics

        metrics = {}

        # Metricas de rentabilidad
        metrics.update(self._calculate_profitability_metrics())

        # Metricas de riesgo
        metrics.update(self._calculate_risk_metrics())

        # Metricas de eficiencia
        metrics.update(self._calculate_efficiency_metrics())

        # Metricas operativas
        metrics.update(self._calculate_operational_metrics())

        self._cached_metrics = metrics
        self._metrics_dirty = False

        return metrics

    def _calculate_profitability_metrics(self) -> Dict:
        """Calcula metricas de rentabilidad"""
        current_equity = self._equity_curve[-1] if self._equity_curve else self.initial_capital
        total_return = ((current_equity - self.initial_capital) / self.initial_capital) * 100

        # CAGR
        if len(self._equity_timestamps) > 1:
            days = (self._equity_timestamps[-1] - self._equity_timestamps[0]).days
            years = max(days / 365.25, 0.01)  # Minimo 1% de año
            cagr = (((current_equity / self.initial_capital) ** (1 / years)) - 1) * 100
        else:
            cagr = 0

        # Expectancy
        if self._trades:
            win_rate = len([t for t in self._trades if t.pnl > 0]) / len(self._trades)
            avg_win = np.mean([t.pnl for t in self._trades if t.pnl > 0]) if any(t.pnl > 0 for t in self._trades) else 0
            avg_loss = abs(np.mean([t.pnl for t in self._trades if t.pnl < 0])) if any(t.pnl < 0 for t in self._trades) else 0
            expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        else:
            expectancy = 0

        return {
            "total_return_pct": total_return,
            "cagr_pct": cagr,
            "expectancy": expectancy,
            "current_equity": current_equity,
            "total_pnl": current_equity - self.initial_capital,
        }

    def _calculate_risk_metrics(self) -> Dict:
        """Calcula metricas de riesgo"""
        # Volatilidad
        if len(self._equity_curve) > 1:
            returns = pd.Series(self._equity_curve).pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) * 100
        else:
            volatility = 0

        # Value at Risk (95%)
        if len(self._equity_curve) > 10:
            returns = pd.Series(self._equity_curve).pct_change().dropna()
            var_95 = np.percentile(returns, 5) * self._equity_curve[-1]
        else:
            var_95 = 0

        return {
            "max_drawdown_pct": self._max_drawdown * 100,
            "current_drawdown_pct": self._current_drawdown * 100,
            "volatility_pct": volatility,
            "value_at_risk_95": var_95,
            "peak_equity": self._peak_equity,
        }

    def _calculate_efficiency_metrics(self) -> Dict:
        """Calcula metricas de eficiencia (risk-adjusted)"""
        if len(self._equity_curve) < 2:
            return {
                "sharpe_ratio": 0,
                "sortino_ratio": 0,
                "calmar_ratio": 0,
            }

        returns = pd.Series(self._equity_curve).pct_change().dropna()

        # Sharpe Ratio
        excess_returns = returns - (0.02 / 252)  # Risk-free rate diario
        sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252) if excess_returns.std() > 0 else 0

        # Sortino Ratio
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else 0.0001
        sortino = (excess_returns.mean() / downside_std) * np.sqrt(252) if downside_std > 0 else 0

        # Calmar Ratio
        if self._max_drawdown > 0:
            days = (self._equity_timestamps[-1] - self._equity_timestamps[0]).days
            years = max(days / 252, 0.01)
            cagr = (((self._equity_curve[-1] / self.initial_capital) ** (1 / years)) - 1) * 100
            calmar = cagr / (self._max_drawdown * 100)
        else:
            calmar = 0

        return {
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "calmar_ratio": calmar,
        }

    def _calculate_operational_metrics(self) -> Dict:
        """Calcula metricas operativas"""
        if not self._trades:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate_pct": 0,
                "profit_factor": 0,
                "avg_trade_pnl": 0,
                "avg_win": 0,
                "avg_loss": 0,
                "largest_win": 0,
                "largest_loss": 0,
                "avg_trade_duration_hours": 0,
                "long_trades": 0,
                "short_trades": 0,
            }

        winning = [t for t in self._trades if t.pnl > 0]
        losing = [t for t in self._trades if t.pnl < 0]

        # Win Rate
        win_rate = (len(winning) / len(self._trades)) * 100

        # Profit Factor
        gross_profit = sum(t.pnl for t in winning)
        gross_loss = abs(sum(t.pnl for t in losing))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Duracion promedio
        durations = [
            (t.exit_time - t.entry_time).total_seconds() / 3600
            for t in self._trades
        ]
        avg_duration = np.mean(durations) if durations else 0

        # Por tipo de posicion
        long_trades = len([t for t in self._trades if t.side == PositionSide.LONG])
        short_trades = len([t for t in self._trades if t.side == PositionSide.SHORT])

        return {
            "total_trades": len(self._trades),
            "winning_trades": len(winning),
            "losing_trades": len(losing),
            "win_rate_pct": win_rate,
            "profit_factor": profit_factor,
            "avg_trade_pnl": np.mean([t.pnl for t in self._trades]),
            "avg_win": np.mean([t.pnl for t in winning]) if winning else 0,
            "avg_loss": np.mean([t.pnl for t in losing]) if losing else 0,
            "largest_win": max((t.pnl for t in winning), default=0),
            "largest_loss": min((t.pnl for t in losing), default=0),
            "avg_trade_duration_hours": avg_duration,
            "long_trades": long_trades,
            "short_trades": short_trades,
        }

    def get_equity_curve(self) -> pd.DataFrame:
        """
        Obtiene la curva de equity como DataFrame.

        Returns:
            DataFrame con timestamp y equity
        """
        return pd.DataFrame({
            "timestamp": self._equity_timestamps,
            "equity": self._equity_curve
        })

    def get_trades_dataframe(self) -> pd.DataFrame:
        """
        Obtiene historial de trades como DataFrame.

        Returns:
            DataFrame con todos los trades
        """
        if not self._trades:
            return pd.DataFrame()

        return pd.DataFrame([t.to_dict() for t in self._trades])

    def get_state(self) -> PaperTradingState:
        """
        Obtiene el estado actual para UI.

        Returns:
            Estado de paper trading
        """
        metrics = self.get_metrics()

        return PaperTradingState(
            is_running=True,
            balance=self._equity_curve[-1] if self._equity_curve else self.initial_capital,
            equity=self._equity_curve[-1] if self._equity_curve else self.initial_capital,
            total_trades=metrics.get("total_trades", 0),
            winning_trades=metrics.get("winning_trades", 0),
            realized_pnl=metrics.get("total_pnl", 0),
            last_update=datetime.now()
        )

    def get_verdict(self) -> str:
        """
        Genera veredicto sobre la viabilidad de la estrategia.

        Returns:
            Texto con el veredicto
        """
        metrics = self.get_metrics()
        score = 0

        # Criterios de evaluacion
        if metrics.get("sharpe_ratio", 0) > 1:
            score += 1
        if metrics.get("sharpe_ratio", 0) > 2:
            score += 1
        if metrics.get("profit_factor", 0) > 1.5:
            score += 1
        if metrics.get("win_rate_pct", 0) > 50:
            score += 1
        if metrics.get("max_drawdown_pct", 100) < 30:
            score += 1
        if metrics.get("total_trades", 0) >= 30:
            score += 1
        if metrics.get("calmar_ratio", 0) > 1:
            score += 1

        if score >= 6:
            return "ESTRATEGIA VIABLE - Excelentes metricas"
        elif score >= 4:
            return "ESTRATEGIA PROMETEDORA - Requiere optimizacion"
        elif score >= 2:
            return "ESTRATEGIA MARGINAL - Necesita mejoras significativas"
        else:
            return "ESTRATEGIA NO VIABLE - Redisenar completamente"

    def print_report(self):
        """Imprime un reporte formateado"""
        metrics = self.get_metrics()

        print("\n" + "=" * 60)
        print(" " * 15 + "REPORTE DE PAPER TRADING")
        print("=" * 60)

        print("\n[RENTABILIDAD]")
        print(f"  Capital Inicial:    ${self.initial_capital:,.2f}")
        print(f"  Capital Actual:     ${metrics['current_equity']:,.2f}")
        print(f"  P&L Total:          ${metrics['total_pnl']:+,.2f}")
        print(f"  Retorno Total:      {metrics['total_return_pct']:+.2f}%")

        print("\n[RIESGO]")
        print(f"  Max Drawdown:       {metrics['max_drawdown_pct']:.2f}%")
        print(f"  Drawdown Actual:    {metrics['current_drawdown_pct']:.2f}%")
        print(f"  Volatilidad Anual:  {metrics['volatility_pct']:.2f}%")

        print("\n[EFICIENCIA]")
        print(f"  Sharpe Ratio:       {metrics['sharpe_ratio']:.2f}")
        print(f"  Sortino Ratio:      {metrics['sortino_ratio']:.2f}")
        print(f"  Calmar Ratio:       {metrics['calmar_ratio']:.2f}")

        print("\n[OPERATIVAS]")
        print(f"  Total Trades:       {metrics['total_trades']}")
        print(f"  Win Rate:           {metrics['win_rate_pct']:.2f}%")
        print(f"  Profit Factor:      {metrics['profit_factor']:.2f}")
        print(f"  Duracion Promedio:  {metrics['avg_trade_duration_hours']:.1f} horas")

        print("\n" + "=" * 60)
        print(f"VEREDICTO: {self.get_verdict()}")
        print("=" * 60 + "\n")

    def reset(self):
        """Reinicia el tracker"""
        self._equity_curve = [self.config.initial_balance]
        self._equity_timestamps = [datetime.now()]
        self._trades.clear()
        self._peak_equity = self.config.initial_balance
        self._max_drawdown = 0.0
        self._current_drawdown = 0.0
        self._cached_metrics = None
        self._metrics_dirty = True

        logger.info("Performance tracker reiniciado")
