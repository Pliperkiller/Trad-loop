"""
Modulo de Analisis de Performance y Visualizacion

Este modulo contiene:
- PerformanceAnalyzer: Calcula 30+ metricas cuantitativas
- PerformanceVisualizer: Genera dashboards y graficos avanzados
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class PerformanceAnalyzer:
    """
    Clase para analizar metricas de estrategias de trading

    Ejemplo de uso:
    ---------------
    analyzer = PerformanceAnalyzer(
        equity_curve=strategy.equity_curve,
        trades=pd.DataFrame(strategy.closed_trades),
        initial_capital=10000
    )

    analyzer.print_report()
    metrics = analyzer.calculate_all_metrics()
    """

    def __init__(self, equity_curve: List[float], trades: pd.DataFrame,
                 initial_capital: float, risk_free_rate: float = 0.02):
        """
        Args:
            equity_curve: Lista con evolucion del capital
            trades: DataFrame con trades cerrados
            initial_capital: Capital inicial
            risk_free_rate: Tasa libre de riesgo anualizada
        """
        self.equity_curve = pd.Series(equity_curve)
        self.trades = trades.copy()
        self.initial_capital = initial_capital
        self.risk_free_rate = risk_free_rate
        self.returns = self.equity_curve.pct_change().dropna()

        if len(self.trades) > 0 and 'exit_time' in self.trades.columns:
            self.trades['exit_time'] = pd.to_datetime(self.trades['exit_time'])
            self.trades['entry_time'] = pd.to_datetime(self.trades['entry_time'])

    def calculate_all_metrics(self) -> Dict:
        """Calcula todas las metricas de performance"""
        metrics = {}

        # Metricas de rentabilidad
        metrics.update(self._profitability_metrics())

        # Metricas de riesgo
        metrics.update(self._risk_metrics())

        # Metricas ajustadas por riesgo
        metrics.update(self._risk_adjusted_metrics())

        # Metricas de consistencia
        metrics.update(self._consistency_metrics())

        # Metricas operativas
        metrics.update(self._operational_metrics())

        # Metricas especializadas para mean reversion
        metrics.update(self._mean_reversion_metrics())

        # Metrica compuesta MRQS (Mean Reversion Quality Score)
        metrics.update(self._mrqs_metric(metrics))

        return metrics

    def _profitability_metrics(self) -> Dict:
        """Metricas de rentabilidad"""
        final_capital = self.equity_curve.iloc[-1]
        total_return = ((final_capital - self.initial_capital) / self.initial_capital) * 100

        # Calcular CAGR
        if len(self.trades) > 0:
            days = (self.trades['exit_time'].max() - self.trades['entry_time'].min()).days
            years = days / 365.25
            if years > 0:
                cagr = (((final_capital / self.initial_capital) ** (1 / years)) - 1) * 100
            else:
                cagr = 0
        else:
            cagr = 0

        # Expectancy
        if len(self.trades) > 0:
            win_rate = len(self.trades[self.trades['pnl'] > 0]) / len(self.trades)
            avg_win = self.trades[self.trades['pnl'] > 0]['pnl'].mean() if len(self.trades[self.trades['pnl'] > 0]) > 0 else 0
            avg_loss = abs(self.trades[self.trades['pnl'] < 0]['pnl'].mean()) if len(self.trades[self.trades['pnl'] < 0]) > 0 else 0
            expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        else:
            expectancy = 0

        return {
            'total_return_pct': total_return,
            'cagr_pct': cagr,
            'expectancy': expectancy,
            'final_capital': final_capital,
            'total_pnl': final_capital - self.initial_capital
        }

    def _risk_metrics(self) -> Dict:
        """Metricas de riesgo"""
        # Maximum Drawdown
        rolling_max = self.equity_curve.expanding().max()
        drawdown = (self.equity_curve - rolling_max) / rolling_max * 100
        max_drawdown = drawdown.min()

        # Drawdown Duration
        is_drawdown = drawdown < 0
        drawdown_periods = []
        current_dd_length = 0

        for is_dd in is_drawdown:
            if is_dd:
                current_dd_length += 1
            else:
                if current_dd_length > 0:
                    drawdown_periods.append(current_dd_length)
                current_dd_length = 0

        max_drawdown_duration = max(drawdown_periods) if drawdown_periods else 0

        # Volatilidad anualizada
        if len(self.returns) > 0:
            volatility = self.returns.std() * np.sqrt(252) * 100
        else:
            volatility = 0

        # Value at Risk (95%)
        if len(self.returns) > 0:
            var_95 = np.percentile(self.returns, 5) * self.equity_curve.iloc[-1]
        else:
            var_95 = 0

        # Conditional Value at Risk (CVaR)
        if len(self.returns) > 0:
            cvar_95 = self.returns[self.returns <= np.percentile(self.returns, 5)].mean() * self.equity_curve.iloc[-1]
        else:
            cvar_95 = 0

        return {
            'max_drawdown_pct': max_drawdown,
            'max_drawdown_duration': max_drawdown_duration,
            'volatility_pct': volatility,
            'value_at_risk_95': var_95,
            'conditional_var_95': cvar_95
        }

    def _risk_adjusted_metrics(self) -> Dict:
        """Metricas ajustadas por riesgo"""
        if len(self.returns) == 0:
            return {
                'sharpe_ratio': 0,
                'sortino_ratio': 0,
                'calmar_ratio': 0,
                'omega_ratio': 0
            }

        # Sharpe Ratio
        excess_returns = self.returns - (self.risk_free_rate / 252)
        sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252) if excess_returns.std() > 0 else 0

        # Sortino Ratio
        downside_returns = self.returns[self.returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else 0.0001
        sortino = (excess_returns.mean() / downside_std) * np.sqrt(252) if downside_std > 0 else 0

        # Calmar Ratio
        rolling_max = self.equity_curve.expanding().max()
        drawdown = (self.equity_curve - rolling_max) / rolling_max * 100
        max_dd = abs(drawdown.min())

        days = len(self.equity_curve)
        years = days / 252
        if years > 0:
            cagr = (((self.equity_curve.iloc[-1] / self.initial_capital) ** (1 / years)) - 1) * 100
            calmar = cagr / max_dd if max_dd > 0 else 0
        else:
            calmar = 0

        # Omega Ratio
        threshold = self.risk_free_rate / 252
        gains = self.returns[self.returns > threshold] - threshold
        losses = threshold - self.returns[self.returns < threshold]
        omega = gains.sum() / losses.sum() if losses.sum() > 0 else 0

        return {
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            'omega_ratio': omega
        }

    def _consistency_metrics(self) -> Dict:
        """Metricas de consistencia"""
        if len(self.trades) == 0:
            return {
                'win_rate_pct': 0,
                'profit_factor': 0,
                'risk_reward_ratio': 0,
                'recovery_factor': 0
            }

        # Win Rate
        winning_trades = len(self.trades[self.trades['pnl'] > 0])
        win_rate = (winning_trades / len(self.trades)) * 100

        # Profit Factor
        gross_profit = self.trades[self.trades['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(self.trades[self.trades['pnl'] < 0]['pnl'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Risk/Reward Ratio
        avg_win = self.trades[self.trades['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = abs(self.trades[self.trades['pnl'] < 0]['pnl'].mean()) if len(self.trades[self.trades['pnl'] < 0]) > 0 else 1
        rr_ratio = avg_win / avg_loss if avg_loss > 0 else 0

        # Recovery Factor
        net_profit = self.equity_curve.iloc[-1] - self.initial_capital
        rolling_max = self.equity_curve.expanding().max()
        drawdown = rolling_max - self.equity_curve
        max_dd_value = drawdown.max()
        recovery_factor = net_profit / max_dd_value if max_dd_value > 0 else 0

        return {
            'win_rate_pct': win_rate,
            'profit_factor': profit_factor,
            'risk_reward_ratio': rr_ratio,
            'recovery_factor': recovery_factor
        }

    def _operational_metrics(self) -> Dict:
        """Metricas operativas"""
        if len(self.trades) == 0:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'avg_trade_duration_hours': 0,
                'max_consecutive_wins': 0,
                'max_consecutive_losses': 0
            }

        total_trades = len(self.trades)
        winning_trades = len(self.trades[self.trades['pnl'] > 0])
        losing_trades = len(self.trades[self.trades['pnl'] < 0])

        # Duracion promedio
        self.trades['duration'] = (self.trades['exit_time'] - self.trades['entry_time']).dt.total_seconds() / 3600
        avg_duration = self.trades['duration'].mean()

        # Rachas consecutivas
        wins = (self.trades['pnl'] > 0).astype(int)

        max_consecutive_wins = 0
        max_consecutive_losses = 0
        current_wins = 0
        current_losses = 0

        for win in wins:
            if win == 1:
                current_wins += 1
                current_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, current_losses)

        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'avg_trade_duration_hours': avg_duration,
            'max_consecutive_wins': max_consecutive_wins,
            'max_consecutive_losses': max_consecutive_losses
        }

    def _mean_reversion_metrics(self) -> Dict:
        """
        Metricas especializadas para estrategias de mean reversion.
        Estas metricas estan optimizadas para estrategias que generan
        muchos trades con pequenas ganancias y pocas perdidas grandes.
        """
        default_metrics = {
            'adjusted_profit_factor': 0.0,
            'expectancy_per_trade': 0.0,
            'omega_ratio_zero': 0.0,
            'mean_reversion_score': 0.0,
            'mean_reversion_optimization_target': 0.0,
            'worst_trade_pct': 0.0
        }

        if len(self.trades) == 0:
            return default_metrics

        try:
            # Obtener retornos como numpy array
            if 'return_pct' in self.trades.columns:
                trade_returns = np.array(self.trades['return_pct'].values, dtype=float)
            else:
                if self.initial_capital > 0:
                    trade_returns = np.array(self.trades['pnl'].values, dtype=float) / self.initial_capital * 100
                else:
                    return default_metrics

            # Filtrar NaN e Inf
            trade_returns = trade_returns[~np.isnan(trade_returns) & ~np.isinf(trade_returns)]

            if len(trade_returns) == 0:
                return default_metrics

            num_trades = len(trade_returns)

            # --- Adjusted Profit Factor ---
            gains = float(np.sum(trade_returns[trade_returns > 0]))
            losses = float(np.abs(np.sum(trade_returns[trade_returns < 0])))

            if losses > 0:
                base_pf = gains / losses
            else:
                base_pf = 10.0 if gains > 0 else 0.0  # Cap at 10 instead of inf

            # Bonus por consistencia (muchos trades)
            consistency_bonus = min(num_trades / 200.0, 1.2)
            adjusted_profit_factor = base_pf * consistency_bonus

            # --- Expectancy per Trade ---
            winners = trade_returns[trade_returns > 0]
            losers = trade_returns[trade_returns < 0]

            win_rate = len(winners) / num_trades if num_trades > 0 else 0.0
            avg_win = float(np.mean(winners)) if len(winners) > 0 else 0.0
            avg_loss = float(np.abs(np.mean(losers))) if len(losers) > 0 else 0.0

            expectancy_per_trade = (win_rate * avg_win) - ((1.0 - win_rate) * avg_loss)

            # --- Omega Ratio (threshold=0) ---
            omega_gains = float(np.sum(trade_returns[trade_returns > 0]))
            omega_losses = float(np.abs(np.sum(trade_returns[trade_returns <= 0])))

            if omega_losses > 0:
                omega_ratio_zero = omega_gains / omega_losses
            else:
                omega_ratio_zero = 10.0 if omega_gains > 0 else 0.0  # Cap at 10

            # --- Worst Trade ---
            worst_trade_pct = float(np.abs(np.min(trade_returns))) if len(trade_returns) > 0 else 0.0

            # --- Recovery Factor (usando equity curve) ---
            equity_start = float(self.equity_curve.iloc[0])
            equity_end = float(self.equity_curve.iloc[-1])

            if equity_start > 0:
                total_return = (equity_end / equity_start) - 1.0
            else:
                total_return = 0.0

            # Calcular max drawdown de forma segura
            equity_cummax = self.equity_curve.cummax()
            # Evitar division por cero
            equity_cummax_safe = equity_cummax.replace(0, np.nan)
            drawdown_series = (self.equity_curve / equity_cummax_safe) - 1.0
            drawdown_series = drawdown_series.fillna(0)
            max_dd = float(np.abs(drawdown_series.min()))

            if max_dd > 0 and not np.isnan(max_dd) and not np.isinf(max_dd):
                recovery_factor_mr = total_return / max_dd
            else:
                recovery_factor_mr = 0.0

            # --- Mean Reversion Score ---
            # Componente de consistencia (corregido precedencia de operadores)
            win_rate_component = 0.5 * min(win_rate / 0.65, 1.0)
            if base_pf > 0 and base_pf < 100:
                pf_component = 0.5 * min(max(base_pf - 1.0, 0) / 0.8, 1.0)
            else:
                pf_component = 0.5 if base_pf >= 100 else 0.0

            consistency_score = win_rate_component + pf_component

            # Componente de riesgo
            dd_risk_component = 0.5 * max(1.0 - max_dd / 0.30, 0.0) if max_dd < 10 else 0.0
            worst_trade_component = 0.5 * max(1.0 - worst_trade_pct / 10.0, 0.0)
            risk_score = dd_risk_component + worst_trade_component

            # Componente de volumen
            volume_score = min(np.log10(max(num_trades, 1)) / np.log10(300), 1.0)

            # Componente de retorno (clamped a [0, 1])
            return_score = max(min(total_return / 0.50, 1.0), 0.0)

            # Score final
            mean_reversion_score = (
                0.30 * consistency_score +
                0.25 * risk_score +
                0.25 * return_score +
                0.20 * volume_score
            )

            # --- Mean Reversion Optimization Target ---
            min_trades_required = 50
            max_allowed_dd = 0.35

            if num_trades < min_trades_required:
                mean_reversion_optimization_target = 0.0
            elif max_dd > max_allowed_dd:
                mean_reversion_optimization_target = 0.0
            elif expectancy_per_trade <= 0:
                mean_reversion_optimization_target = 0.0
            else:
                # Score ponderado con componentes clamped
                omega_component = min(omega_ratio_zero / 2.0, 1.0)
                recovery_component = min(max(recovery_factor_mr, 0) / 3.0, 1.0)
                expectancy_component = min(max(expectancy_per_trade, 0) / 2.0, 1.0)
                dd_component = max(1.0 - max_dd / max_allowed_dd, 0.0)

                mean_reversion_optimization_target = (
                    0.30 * omega_component +
                    0.25 * recovery_component +
                    0.25 * expectancy_component +
                    0.20 * dd_component
                )

            # Sanitizar todos los valores antes de retornar
            def sanitize(val, default=0.0):
                if val is None or np.isnan(val) or np.isinf(val):
                    return default
                return float(val)

            return {
                'adjusted_profit_factor': sanitize(adjusted_profit_factor),
                'expectancy_per_trade': sanitize(expectancy_per_trade),
                'omega_ratio_zero': sanitize(omega_ratio_zero),
                'mean_reversion_score': sanitize(mean_reversion_score),
                'mean_reversion_optimization_target': sanitize(mean_reversion_optimization_target),
                'worst_trade_pct': sanitize(worst_trade_pct)
            }

        except Exception as e:
            # En caso de cualquier error, retornar valores por defecto
            return default_metrics

    def _mrqs_metric(self, metrics: Dict) -> Dict:
        """
        Mean Reversion Quality Score (MRQS)

        Metrica compuesta optimizada para estrategias de mean reversion
        en mercados de alta volatilidad como Bitcoin futuros.

        Componentes:
        - Sortino Ratio (40%): Penaliza volatilidad negativa
        - Calmar Ratio (25%): Return / Max Drawdown - critico para futuros
        - Win Rate ajustado (15%): Penaliza WR sospechosamente alto (>85%)
        - Profit Factor (10%): Con cap para evitar outliers
        - Expectancy (10%): Valida edge matematico real

        Penalizaciones:
        - Pocos trades (<30): Reduce score significativamente
        - WR muy alto (>85%): Indica probable overfitting
        """
        try:
            # Extraer metricas base (ya calculadas)
            sortino = metrics.get('sortino_ratio', 0) or 0
            total_return = metrics.get('total_return_pct', 0) or 0
            max_dd = abs(metrics.get('max_drawdown_pct', 100) or 100)
            win_rate = (metrics.get('win_rate_pct', 0) or 0) / 100
            profit_factor = metrics.get('profit_factor', 0) or 0
            total_trades = metrics.get('total_trades', 0) or 0

            # Obtener avg_win y avg_loss de trades
            if len(self.trades) > 0:
                winners = self.trades[self.trades['pnl'] > 0]
                losers = self.trades[self.trades['pnl'] < 0]

                if 'return_pct' in self.trades.columns:
                    avg_win = float(winners['return_pct'].mean()) if len(winners) > 0 else 0
                    avg_loss = abs(float(losers['return_pct'].mean())) if len(losers) > 0 else 1
                else:
                    avg_win = float(winners['pnl'].mean()) / self.initial_capital * 100 if len(winners) > 0 else 0
                    avg_loss = abs(float(losers['pnl'].mean())) / self.initial_capital * 100 if len(losers) > 0 else 1
            else:
                avg_win = 0
                avg_loss = 1

            # 1. Sortino Score (40% peso)
            # Mejor que Sharpe para MR porque tiene distribucion asimetrica
            sortino_score = min(sortino / 2.0, 1.5)  # Cap en 1.5

            # 2. Calmar Score (25% peso)
            # Critico para futuros - una liquidacion borra todo
            calmar = (total_return / max_dd) if max_dd > 0 else 0
            calmar_score = min(calmar / 1.5, 1.5)  # Buen Calmar > 1.5

            # 3. Win Rate ajustado (15% peso)
            # MR tipicamente tiene WR alto (60-75%)
            # WR > 85% es sospechoso de overfitting
            if win_rate < 0.35:
                wr_score = win_rate * 2  # Penaliza bajo WR
            elif win_rate > 0.85:
                wr_score = 1.7 - win_rate  # Penaliza WR sospechosamente alto
            else:
                wr_score = 0.7 + (win_rate - 0.35)  # Rango optimo 35-85%

            # 4. Profit Factor con cap (10% peso)
            # PF > 2 es excelente, cap en 1.0 para el score
            if profit_factor == float('inf') or profit_factor > 10:
                pf_score = 1.0
            else:
                pf_score = min(profit_factor / 2.0, 1.0)

            # 5. Expectancy ajustada (10% peso)
            # E = (WR * avg_win) - ((1-WR) * avg_loss)
            expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
            exp_score = min(max(expectancy / 2.0, -0.5), 1.0)

            # 6. Penalizacion por pocos trades
            # Minimo 30 trades para significancia estadistica en 4H
            if total_trades < 15:
                trade_penalty = 0.3  # Penalizacion severa
            elif total_trades < 30:
                trade_penalty = 0.6 + (total_trades - 15) * 0.027
            else:
                trade_penalty = 1.0

            # Score final ponderado
            raw_score = (
                sortino_score * 0.40 +
                calmar_score * 0.25 +
                wr_score * 0.15 +
                pf_score * 0.10 +
                exp_score * 0.10
            )

            mrqs = raw_score * trade_penalty

            # Sanitizar
            if np.isnan(mrqs) or np.isinf(mrqs):
                mrqs = 0.0

            return {
                'mrqs': float(mrqs),
                'mrqs_raw': float(raw_score),
                'mrqs_trade_penalty': float(trade_penalty)
            }

        except Exception:
            return {
                'mrqs': 0.0,
                'mrqs_raw': 0.0,
                'mrqs_trade_penalty': 0.0
            }

    def print_report(self):
        """Imprime un reporte completo y formateado"""
        metrics = self.calculate_all_metrics()

        print("\n" + "="*70)
        print(" "*20 + "REPORTE DE PERFORMANCE")
        print("="*70)

        # RENTABILIDAD
        print("\n[RENTABILIDAD]")
        print(f"  Capital Inicial:        ${metrics['final_capital'] - metrics['total_pnl']:,.2f}")
        print(f"  Capital Final:          ${metrics['final_capital']:,.2f}")
        print(f"  P&L Total:              ${metrics['total_pnl']:,.2f}")
        print(f"  Retorno Total:          {metrics['total_return_pct']:.2f}%")
        print(f"  CAGR:                   {metrics['cagr_pct']:.2f}%")
        print(f"  Expectancy:             ${metrics['expectancy']:.2f}")

        # RIESGO
        print("\n[RIESGO]")
        print(f"  Max Drawdown:           {metrics['max_drawdown_pct']:.2f}%")
        print(f"  Duracion Max DD:        {metrics['max_drawdown_duration']} periodos")
        print(f"  Volatilidad Anual:      {metrics['volatility_pct']:.2f}%")
        print(f"  VaR 95%:                ${metrics['value_at_risk_95']:,.2f}")

        # EFICIENCIA
        print("\n[EFICIENCIA - Risk-Adjusted]")
        print(f"  Sharpe Ratio:           {metrics['sharpe_ratio']:.2f} {self._rating_sharpe(metrics['sharpe_ratio'])}")
        print(f"  Sortino Ratio:          {metrics['sortino_ratio']:.2f}")
        print(f"  Calmar Ratio:           {metrics['calmar_ratio']:.2f} {self._rating_calmar(metrics['calmar_ratio'])}")
        print(f"  Omega Ratio:            {metrics['omega_ratio']:.2f}")

        # CONSISTENCIA
        print("\n[CONSISTENCIA]")
        print(f"  Win Rate:               {metrics['win_rate_pct']:.2f}% {self._rating_winrate(metrics['win_rate_pct'])}")
        print(f"  Profit Factor:          {metrics['profit_factor']:.2f} {self._rating_pf(metrics['profit_factor'])}")
        print(f"  Risk/Reward:            {metrics['risk_reward_ratio']:.2f}")
        print(f"  Recovery Factor:        {metrics['recovery_factor']:.2f}")

        # OPERATIVAS
        print("\n[METRICAS OPERATIVAS]")
        print(f"  Total Trades:           {metrics['total_trades']}")
        print(f"  Trades Ganadores:       {metrics['winning_trades']}")
        print(f"  Trades Perdedores:      {metrics['losing_trades']}")
        print(f"  Duracion Prom:          {metrics['avg_trade_duration_hours']:.1f} horas")
        print(f"  Max Racha Ganadora:     {metrics['max_consecutive_wins']}")
        print(f"  Max Racha Perdedora:    {metrics['max_consecutive_losses']}")

        # MEAN REVERSION
        print("\n[MEAN REVERSION - Targets de Optimizacion]")
        print(f"  Adjusted Profit Factor: {metrics['adjusted_profit_factor']:.2f}")
        print(f"  Expectancy per Trade:   {metrics['expectancy_per_trade']:.4f}%")
        print(f"  Omega Ratio (th=0):     {metrics['omega_ratio_zero']:.2f}")
        print(f"  Worst Trade:            -{metrics['worst_trade_pct']:.2f}%")
        print(f"  MR Score:               {metrics['mean_reversion_score']:.4f} {self._rating_mr_score(metrics['mean_reversion_score'])}")
        print(f"  MR Optimization Target: {metrics['mean_reversion_optimization_target']:.4f}")

        # MRQS (Bitcoin Futuros)
        print("\n[MRQS - Mean Reversion Quality Score (BTC Futuros)]")
        print(f"  MRQS:                   {metrics['mrqs']:.4f} {self._rating_mrqs(metrics['mrqs'])}")
        print(f"  MRQS Raw Score:         {metrics['mrqs_raw']:.4f}")
        print(f"  Trade Penalty:          {metrics['mrqs_trade_penalty']:.2f}x")

        # VEREDICTO FINAL
        print("\n" + "="*70)
        print(self._final_verdict(metrics))
        print("="*70 + "\n")

    def _rating_sharpe(self, sharpe: float) -> str:
        if sharpe < 0: return "[Malo]"
        elif sharpe < 1: return "[Suboptimo]"
        elif sharpe < 2: return "[Bueno]"
        elif sharpe < 3: return "[Muy Bueno]"
        else: return "[Excelente]"

    def _rating_calmar(self, calmar: float) -> str:
        if calmar < 1: return "[Bajo]"
        elif calmar < 2: return "[Aceptable]"
        elif calmar < 3: return "[Bueno]"
        else: return "[Excelente]"

    def _rating_winrate(self, wr: float) -> str:
        if wr < 40: return "[Bajo]"
        elif wr < 50: return "[Aceptable]"
        elif wr < 60: return "[Bueno]"
        else: return "[Excelente]"

    def _rating_pf(self, pf: float) -> str:
        if pf < 1: return "[Perdedora]"
        elif pf < 1.5: return "[Marginal]"
        elif pf < 2: return "[Bueno]"
        elif pf < 3: return "[Muy Bueno]"
        else: return "[Excelente]"

    def _rating_mr_score(self, score: float) -> str:
        if score < 0.3: return "[Bajo]"
        elif score < 0.5: return "[Moderado]"
        elif score < 0.7: return "[Bueno]"
        elif score < 0.85: return "[Muy Bueno]"
        else: return "[Excelente]"

    def _rating_mrqs(self, mrqs: float) -> str:
        """Rating para MRQS (Mean Reversion Quality Score)"""
        if mrqs < 0.3: return "[Bajo - No operar]"
        elif mrqs < 0.5: return "[Marginal]"
        elif mrqs < 0.7: return "[Aceptable]"
        elif mrqs < 0.9: return "[Bueno]"
        elif mrqs < 1.1: return "[Muy Bueno]"
        else: return "[Excelente]"

    def _final_verdict(self, metrics: Dict) -> str:
        """Veredicto final sobre la viabilidad de la estrategia"""
        score = 0

        # Criterios de evaluacion
        if metrics['sharpe_ratio'] > 1: score += 1
        if metrics['sharpe_ratio'] > 2: score += 1
        if metrics['profit_factor'] > 1.5: score += 1
        if metrics['win_rate_pct'] > 50: score += 1
        if metrics['max_drawdown_pct'] > -30: score += 1
        if metrics['total_trades'] >= 30: score += 1
        if metrics['calmar_ratio'] > 1: score += 1

        if score >= 6:
            return "VEREDICTO: ESTRATEGIA VIABLE - Excelentes metricas"
        elif score >= 4:
            return "VEREDICTO: ESTRATEGIA PROMETEDORA - Requiere optimizacion"
        elif score >= 2:
            return "VEREDICTO: ESTRATEGIA MARGINAL - Necesita mejoras significativas"
        else:
            return "VEREDICTO: ESTRATEGIA NO VIABLE - Redisenar completamente"


class PerformanceVisualizer:
    """
    Clase para crear visualizaciones avanzadas

    Ejemplo de uso:
    ---------------
    visualizer = PerformanceVisualizer(analyzer)
    visualizer.plot_comprehensive_dashboard()
    visualizer.plot_rolling_metrics(window=50)
    """

    def __init__(self, analyzer: PerformanceAnalyzer):
        """
        Args:
            analyzer: Instancia de PerformanceAnalyzer
        """
        self.analyzer = analyzer
        self.equity_curve = analyzer.equity_curve
        self.trades = analyzer.trades
        self.returns = analyzer.returns
        self.initial_capital = analyzer.initial_capital

    def plot_comprehensive_dashboard(self, figsize=(20, 12)):
        """
        Dashboard completo con 6 graficos de analisis
        """
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. Equity Curve con Drawdown
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_equity_with_drawdown(ax1)

        # 2. Distribucion de Returns
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_returns_distribution(ax2)

        # 3. P&L por Trade
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_pnl_per_trade(ax3)

        # 4. Rolling Sharpe Ratio
        ax4 = fig.add_subplot(gs[1, 2])
        self._plot_rolling_sharpe(ax4)

        # 5. Monthly Returns Heatmap
        ax5 = fig.add_subplot(gs[2, :2])
        self._plot_monthly_returns_heatmap(ax5)

        # 6. Underwater Plot (Drawdown)
        ax6 = fig.add_subplot(gs[2, 2])
        self._plot_underwater(ax6)

        plt.suptitle('Dashboard Completo de Performance', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()

        return fig

    def _plot_equity_with_drawdown(self, ax):
        """Curva de equity con zonas de drawdown"""
        equity = self.equity_curve.values

        # Equity curve
        ax.plot(equity, linewidth=2, label='Equity Curve', color='#2E86AB')
        ax.axhline(y=self.initial_capital, color='red', linestyle='--',
                   alpha=0.5, linewidth=1, label='Capital Inicial')

        # Resaltar drawdowns
        rolling_max = pd.Series(equity).expanding().max()
        drawdown_pct = (pd.Series(equity) - rolling_max) / rolling_max * 100

        # Rellenar areas de drawdown
        ax.fill_between(range(len(equity)), rolling_max, equity,
                        where=(equity < rolling_max),
                        color='red', alpha=0.2, label='Drawdown')

        ax.set_title('Curva de Equity con Drawdowns', fontsize=12, fontweight='bold')
        ax.set_xlabel('Periodo')
        ax.set_ylabel('Capital ($)')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        # Formato de eje Y
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

    def _plot_returns_distribution(self, ax):
        """Distribucion de retornos con estadisticas"""
        if len(self.trades) == 0:
            ax.text(0.5, 0.5, 'No hay datos de trades',
                   ha='center', va='center', transform=ax.transAxes)
            return

        returns = self.trades['return_pct'].values

        # Histograma
        n, bins, patches = ax.hist(returns, bins=30, edgecolor='black',
                                   alpha=0.7, color='#A23B72')

        # Colorear barras segun ganancia/perdida
        for i, patch in enumerate(patches):
            if bins[i] < 0:
                patch.set_facecolor('#E63946')
            else:
                patch.set_facecolor('#06A77D')

        # Linea vertical en cero
        ax.axvline(x=0, color='black', linestyle='--', linewidth=2, alpha=0.7)

        # Media
        mean_return = returns.mean()
        ax.axvline(x=mean_return, color='blue', linestyle='-',
                  linewidth=2, label=f'Media: {mean_return:.2f}%')

        # Agregar texto con estadisticas
        skew = stats.skew(returns)
        kurt = stats.kurtosis(returns)
        textstr = f'Skewness: {skew:.2f}\nKurtosis: {kurt:.2f}'
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
               fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.set_title('Distribucion de Retornos', fontsize=11, fontweight='bold')
        ax.set_xlabel('Retorno (%)')
        ax.set_ylabel('Frecuencia')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    def _plot_pnl_per_trade(self, ax):
        """P&L acumulado por trade"""
        if len(self.trades) == 0:
            ax.text(0.5, 0.5, 'No hay datos de trades',
                   ha='center', va='center', transform=ax.transAxes)
            return

        cumulative_pnl = self.trades['pnl'].cumsum()
        trade_numbers = range(1, len(cumulative_pnl) + 1)

        # Linea principal
        ax.plot(trade_numbers, cumulative_pnl.values,
               linewidth=2, color='#06A77D', marker='o', markersize=3)

        # Linea de cero
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1)

        # Resaltar area positiva y negativa
        ax.fill_between(trade_numbers, 0, cumulative_pnl.values,
                       where=(cumulative_pnl.values >= 0),
                       color='green', alpha=0.2)
        ax.fill_between(trade_numbers, 0, cumulative_pnl.values,
                       where=(cumulative_pnl.values < 0),
                       color='red', alpha=0.2)

        ax.set_title('P&L Acumulado por Trade', fontsize=11, fontweight='bold')
        ax.set_xlabel('Numero de Trade')
        ax.set_ylabel('P&L Acumulado ($)')
        ax.grid(True, alpha=0.3)

        # Formato de eje Y
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

    def _plot_rolling_sharpe(self, ax, window=30):
        """Sharpe Ratio rodante"""
        if len(self.returns) < window:
            ax.text(0.5, 0.5, f'Datos insuficientes (min {window})',
                   ha='center', va='center', transform=ax.transAxes)
            return

        # Calcular Sharpe rodante
        rolling_sharpe = (self.returns.rolling(window=window).mean() /
                         self.returns.rolling(window=window).std()) * np.sqrt(252)

        ax.plot(rolling_sharpe.values, linewidth=2, color='#F18F01')

        # Lineas de referencia
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
        ax.axhline(y=1, color='orange', linestyle='--', alpha=0.5, linewidth=1, label='Sharpe = 1')
        ax.axhline(y=2, color='green', linestyle='--', alpha=0.5, linewidth=1, label='Sharpe = 2')

        # Colorear area segun calidad
        ax.fill_between(range(len(rolling_sharpe)), 0, rolling_sharpe.values,
                       where=(rolling_sharpe.values >= 1),
                       color='green', alpha=0.2)
        ax.fill_between(range(len(rolling_sharpe)), 0, rolling_sharpe.values,
                       where=(rolling_sharpe.values < 1) & (rolling_sharpe.values >= 0),
                       color='yellow', alpha=0.2)
        ax.fill_between(range(len(rolling_sharpe)), 0, rolling_sharpe.values,
                       where=(rolling_sharpe.values < 0),
                       color='red', alpha=0.2)

        ax.set_title(f'Rolling Sharpe Ratio (ventana={window})', fontsize=11, fontweight='bold')
        ax.set_xlabel('Periodo')
        ax.set_ylabel('Sharpe Ratio')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

    def _plot_monthly_returns_heatmap(self, ax):
        """Heatmap de retornos mensuales"""
        if len(self.trades) == 0 or 'exit_time' not in self.trades.columns:
            ax.text(0.5, 0.5, 'No hay datos temporales',
                   ha='center', va='center', transform=ax.transAxes)
            return

        # Preparar datos mensuales
        trades_copy = self.trades.copy()
        trades_copy['year_month'] = trades_copy['exit_time'].dt.to_period('M')
        monthly_returns = trades_copy.groupby('year_month')['return_pct'].sum()

        if len(monthly_returns) == 0:
            ax.text(0.5, 0.5, 'Datos insuficientes',
                   ha='center', va='center', transform=ax.transAxes)
            return

        # Crear matriz para heatmap
        monthly_returns.index = monthly_returns.index.to_timestamp()
        monthly_returns_df = monthly_returns.to_frame()
        monthly_returns_df['Year'] = monthly_returns_df.index.year
        monthly_returns_df['Month'] = monthly_returns_df.index.month

        # Pivot para heatmap
        heatmap_data = monthly_returns_df.pivot_table(
            values='return_pct',
            index='Year',
            columns='Month',
            aggfunc='sum'
        )

        # Crear heatmap
        sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlGn',
                   center=0, cbar_kws={'label': 'Retorno (%)'},
                   linewidths=0.5, ax=ax)

        ax.set_title('Retornos Mensuales (%)', fontsize=11, fontweight='bold')
        ax.set_xlabel('Mes')
        ax.set_ylabel('Ano')

        # Nombres de meses
        month_names = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun',
                      'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
        # Obtener las posiciones de los ticks del eje X
        xticks = [label.get_text() for label in ax.get_xticklabels()]
        ax.set_xticklabels([month_names[int(float(i))-1] if i and float(i) <= 12 else ''
                           for i in xticks], rotation=0)

    def _plot_underwater(self, ax):
        """Grafico underwater (drawdown en el tiempo)"""
        rolling_max = self.equity_curve.expanding().max()
        drawdown = (self.equity_curve - rolling_max) / rolling_max * 100

        ax.fill_between(range(len(drawdown)), drawdown.values, 0,
                       color='red', alpha=0.5)
        ax.plot(drawdown.values, color='darkred', linewidth=1)

        # Linea de cero
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)

        # Marcar el drawdown maximo
        max_dd_idx = drawdown.idxmin()
        max_dd_val = drawdown.min()
        ax.plot(max_dd_idx, max_dd_val, 'ro', markersize=8,
               label=f'Max DD: {max_dd_val:.2f}%')

        ax.set_title('Underwater Plot (Drawdown)', fontsize=11, fontweight='bold')
        ax.set_xlabel('Periodo')
        ax.set_ylabel('Drawdown (%)')
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([drawdown.min() * 1.1, 5])

    def plot_rolling_metrics(self, window=50, figsize=(15, 10)):
        """
        Graficos de metricas rodantes en el tiempo
        """
        if len(self.returns) < window:
            print(f"Error: Se necesitan al menos {window} periodos de datos")
            return

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # 1. Rolling Return
        rolling_return = self.returns.rolling(window=window).mean() * 252 * 100
        axes[0, 0].plot(rolling_return.values, linewidth=2, color='#2E86AB')
        axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[0, 0].set_title(f'Retorno Anualizado Rodante (ventana={window})',
                           fontsize=11, fontweight='bold')
        axes[0, 0].set_xlabel('Periodo')
        axes[0, 0].set_ylabel('Retorno Anualizado (%)')
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Rolling Volatility
        rolling_vol = self.returns.rolling(window=window).std() * np.sqrt(252) * 100
        axes[0, 1].plot(rolling_vol.values, linewidth=2, color='#E63946')
        axes[0, 1].set_title(f'Volatilidad Rodante (ventana={window})',
                           fontsize=11, fontweight='bold')
        axes[0, 1].set_xlabel('Periodo')
        axes[0, 1].set_ylabel('Volatilidad Anualizada (%)')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Rolling Sharpe
        rolling_sharpe = (self.returns.rolling(window=window).mean() /
                         self.returns.rolling(window=window).std()) * np.sqrt(252)
        axes[1, 0].plot(rolling_sharpe.values, linewidth=2, color='#06A77D')
        axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[1, 0].axhline(y=1, color='orange', linestyle='--', alpha=0.5)
        axes[1, 0].axhline(y=2, color='green', linestyle='--', alpha=0.5)
        axes[1, 0].set_title(f'Sharpe Ratio Rodante (ventana={window})',
                           fontsize=11, fontweight='bold')
        axes[1, 0].set_xlabel('Periodo')
        axes[1, 0].set_ylabel('Sharpe Ratio')
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Rolling Max Drawdown
        rolling_max_dd = []
        for i in range(window, len(self.equity_curve)):
            window_data = self.equity_curve.iloc[i-window:i]
            rolling_max = window_data.expanding().max()
            dd = ((window_data - rolling_max) / rolling_max * 100).min()
            rolling_max_dd.append(dd)

        axes[1, 1].plot(rolling_max_dd, linewidth=2, color='#F18F01')
        axes[1, 1].axhline(y=-20, color='orange', linestyle='--', alpha=0.5, label='DD -20%')
        axes[1, 1].axhline(y=-30, color='red', linestyle='--', alpha=0.5, label='DD -30%')
        axes[1, 1].set_title(f'Max Drawdown Rodante (ventana={window})',
                           fontsize=11, fontweight='bold')
        axes[1, 1].set_xlabel('Periodo')
        axes[1, 1].set_ylabel('Max Drawdown (%)')
        axes[1, 1].legend(loc='lower right', fontsize=8)
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.suptitle('Metricas Rodantes en el Tiempo', fontsize=14,
                    fontweight='bold', y=1.00)
        plt.show()

        return fig

    def plot_trade_analysis(self, figsize=(15, 8)):
        """
        Analisis detallado de trades individuales
        """
        if len(self.trades) == 0:
            print("No hay datos de trades para analizar")
            return

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # 1. Wins vs Losses
        wins = self.trades[self.trades['pnl'] > 0]['pnl']
        losses = self.trades[self.trades['pnl'] < 0]['pnl']

        axes[0, 0].hist([wins, losses], bins=20, label=['Wins', 'Losses'],
                       color=['green', 'red'], alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Distribucion de Wins vs Losses',
                           fontsize=11, fontweight='bold')
        axes[0, 0].set_xlabel('P&L ($)')
        axes[0, 0].set_ylabel('Frecuencia')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. P&L por Trade (barras)
        colors = ['green' if x > 0 else 'red' for x in self.trades['pnl']]
        axes[0, 1].bar(range(len(self.trades)), self.trades['pnl'],
                      color=colors, alpha=0.7, edgecolor='black')
        axes[0, 1].axhline(y=0, color='black', linestyle='-', linewidth=1)
        axes[0, 1].set_title('P&L por Trade Individual',
                           fontsize=11, fontweight='bold')
        axes[0, 1].set_xlabel('Trade #')
        axes[0, 1].set_ylabel('P&L ($)')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Duracion vs P&L
        if 'duration' in self.trades.columns:
            scatter_colors = ['green' if x > 0 else 'red' for x in self.trades['pnl']]
            axes[1, 0].scatter(self.trades['duration'], self.trades['pnl'],
                             c=scatter_colors, alpha=0.6, edgecolors='black', s=50)
            axes[1, 0].axhline(y=0, color='black', linestyle='--', linewidth=1)
            axes[1, 0].set_title('Duracion vs P&L', fontsize=11, fontweight='bold')
            axes[1, 0].set_xlabel('Duracion (horas)')
            axes[1, 0].set_ylabel('P&L ($)')
            axes[1, 0].grid(True, alpha=0.3)

        # 4. Evolucion de Win Rate
        cumulative_wins = (self.trades['pnl'] > 0).cumsum()
        cumulative_total = pd.Series(range(1, len(self.trades) + 1))
        win_rate_evolution = (cumulative_wins / cumulative_total) * 100

        axes[1, 1].plot(win_rate_evolution.values, linewidth=2, color='#2E86AB')
        axes[1, 1].axhline(y=50, color='orange', linestyle='--', alpha=0.5, label='50%')
        axes[1, 1].set_title('Evolucion del Win Rate', fontsize=11, fontweight='bold')
        axes[1, 1].set_xlabel('Trade #')
        axes[1, 1].set_ylabel('Win Rate (%)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim([0, 100])

        plt.tight_layout()
        plt.suptitle('Analisis Detallado de Trades', fontsize=14,
                    fontweight='bold', y=1.00)
        plt.show()

        return fig

    def plot_risk_analysis(self, figsize=(15, 8)):
        """
        Analisis detallado de riesgo
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # 1. Distribucion de Drawdowns
        rolling_max = self.equity_curve.expanding().max()
        drawdown = (self.equity_curve - rolling_max) / rolling_max * 100

        axes[0, 0].hist(drawdown[drawdown < 0].values, bins=30,
                       color='red', alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Distribucion de Drawdowns',
                           fontsize=11, fontweight='bold')
        axes[0, 0].set_xlabel('Drawdown (%)')
        axes[0, 0].set_ylabel('Frecuencia')
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Q-Q Plot (normalidad de retornos)
        stats.probplot(self.returns.dropna(), dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('Q-Q Plot (Test de Normalidad)',
                           fontsize=11, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Rolling Beta (si hay benchmark)
        # Por ahora mostramos volatilidad acumulada
        cumulative_vol = self.returns.expanding().std() * np.sqrt(252) * 100
        axes[1, 0].plot(cumulative_vol.values, linewidth=2, color='#E63946')
        axes[1, 0].set_title('Volatilidad Acumulada', fontsize=11, fontweight='bold')
        axes[1, 0].set_xlabel('Periodo')
        axes[1, 0].set_ylabel('Volatilidad Anualizada (%)')
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Value at Risk historico
        var_values = []
        window = 30
        for i in range(window, len(self.returns)):
            window_returns = self.returns.iloc[i-window:i]
            var_95 = np.percentile(window_returns, 5)
            var_values.append(var_95 * 100)

        axes[1, 1].plot(var_values, linewidth=2, color='#F18F01')
        axes[1, 1].set_title(f'Value at Risk Rodante (95%, ventana={window})',
                           fontsize=11, fontweight='bold')
        axes[1, 1].set_xlabel('Periodo')
        axes[1, 1].set_ylabel('VaR (%)')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.suptitle('Analisis de Riesgo', fontsize=14,
                    fontweight='bold', y=1.00)
        plt.show()

        return fig
