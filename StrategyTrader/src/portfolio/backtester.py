"""
Motor de backtesting para portfolios multi-activo.

Permite:
- Backtest simultaneo de multiples activos
- Rebalanceo automatico durante el backtest
- Tracking de equity curve del portfolio
- Calculo de metricas de rendimiento
"""

import time
from datetime import datetime
from typing import Dict, List, Optional, Callable, Tuple
import numpy as np
import pandas as pd

from .models import (
    PortfolioConfig,
    PortfolioState,
    PortfolioPosition,
    PortfolioTradeRecord,
    PortfolioResult,
    PortfolioMetrics,
    RebalanceEvent,
    RebalanceReason,
    AllocationMethod,
    PortfolioSplitResult,
    PortfolioWalkForwardResult,
)
from .allocator import PortfolioAllocator, AllocationResult
from .rebalancer import PortfolioRebalancer


class PortfolioBacktester:
    """
    Motor de backtesting para portfolios multi-activo.

    Ejemplo de uso:
        config = PortfolioConfig(
            initial_capital=10000,
            symbols=['BTC/USDT', 'ETH/USDT', 'SOL/USDT'],
            allocation_method=AllocationMethod.RISK_PARITY,
            rebalance_frequency=RebalanceFrequency.MONTHLY,
        )

        backtester = PortfolioBacktester(config)
        backtester.load_data({
            'BTC/USDT': btc_df,
            'ETH/USDT': eth_df,
            'SOL/USDT': sol_df,
        })
        result = backtester.run()
    """

    def __init__(self, config: PortfolioConfig):
        """
        Args:
            config: Configuracion del portfolio
        """
        self.config = config
        self.allocator = PortfolioAllocator(
            risk_free_rate=config.risk_free_rate,
            min_weight=config.min_position_weight,
            max_weight=config.max_position_weight,
        )
        self.rebalancer = PortfolioRebalancer(config)

        # Data
        self.data: Dict[str, pd.DataFrame] = {}
        self.aligned_data: Optional[pd.DataFrame] = None
        self.returns_data: Optional[pd.DataFrame] = None

        # State
        self.state: Optional[PortfolioState] = None

        # Results tracking
        self.equity_curve: List[float] = []
        self.returns: List[float] = []
        self.drawdown_curve: List[float] = []
        self.timestamps: List[datetime] = []
        self.weight_history: Dict[str, List[float]] = {}
        self.trade_history: List[PortfolioTradeRecord] = []
        self.rebalance_history: List[RebalanceEvent] = []

        # Metrics
        self.peak_equity: float = 0.0
        self.total_commission: float = 0.0
        self.total_slippage: float = 0.0

    def load_data(self, data: Dict[str, pd.DataFrame]) -> None:
        """
        Carga datos OHLCV para cada simbolo.

        Args:
            data: Dict symbol -> DataFrame con columnas ['open', 'high', 'low', 'close', 'volume']
        """
        self.data = data

        # Validar que todos los simbolos tengan datos
        for symbol in self.config.symbols:
            if symbol not in data:
                raise ValueError(f"Missing data for symbol: {symbol}")

            df = data[symbol]
            required_cols = ['open', 'high', 'low', 'close']
            for col in required_cols:
                if col not in df.columns:
                    raise ValueError(f"Missing column '{col}' for {symbol}")

        # Alinear datos por timestamp
        self._align_data()

    def _align_data(self) -> None:
        """Alinea todos los datos al mismo indice temporal"""
        # Encontrar interseccion de indices
        common_index = None

        for symbol, df in self.data.items():
            if common_index is None:
                common_index = df.index
            else:
                common_index = common_index.intersection(df.index)

        if common_index is None or len(common_index) == 0:
            raise ValueError("No overlapping data between symbols")

        # Crear DataFrame alineado con precios de cierre
        aligned_dict = {}
        for symbol in self.config.symbols:
            df = self.data[symbol]
            aligned_dict[symbol] = df.loc[common_index, 'close']

        self.aligned_data = pd.DataFrame(aligned_dict)

        # Calcular retornos
        self.returns_data = self.aligned_data.pct_change().dropna()

    def run(
        self,
        warmup_period: int = 30,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> PortfolioResult:
        """
        Ejecuta el backtest.

        Args:
            warmup_period: Periodos iniciales para calcular allocation
            progress_callback: Funcion opcional para reportar progreso

        Returns:
            PortfolioResult con todos los resultados
        """
        start_time = time.time()

        if self.aligned_data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        # Reset state
        self._reset()

        # Obtener pesos iniciales
        initial_weights = self._calculate_initial_weights(warmup_period)
        self.state.target_weights = initial_weights

        # Inicializar posiciones
        initial_prices = self.aligned_data.iloc[warmup_period].to_dict()
        self._initialize_positions(initial_prices)

        # Iterar sobre cada bar
        total_bars = len(self.aligned_data) - warmup_period
        start_idx = warmup_period

        for i, (timestamp, row) in enumerate(self.aligned_data.iloc[start_idx:].iterrows()):
            # Callback de progreso
            if progress_callback:
                progress_callback(i + 1, total_bars)

            # Precios actuales
            prices = row.to_dict()

            # Actualizar precios de posiciones
            self._update_prices(prices)

            # Actualizar state
            self._update_state(timestamp)

            # Verificar rebalanceo
            decision = self.rebalancer.should_rebalance(self.state, timestamp)

            if decision.should_rebalance:
                # Recalcular weights si es necesario
                if self.config.allocation_method != AllocationMethod.CUSTOM:
                    lookback_returns = self.returns_data.iloc[max(0, i - 60):i + start_idx]
                    if len(lookback_returns) >= 20:
                        new_weights = self._recalculate_weights(lookback_returns)
                        self.state.target_weights = new_weights

                # Ejecutar rebalanceo
                trades = self.rebalancer.calculate_rebalance_trades(
                    self.state, prices, self.state.target_weights
                )

                if trades:
                    event = self.rebalancer.execute_rebalance(
                        self.state, trades, prices, decision.reason
                    )
                    self.rebalance_history.append(event)
                    self.total_commission += event.total_commission

                    # Registrar trades
                    for trade in trades:
                        record = PortfolioTradeRecord(
                            timestamp=timestamp,
                            symbol=trade.symbol,
                            side=trade.side,
                            quantity=trade.quantity,
                            price=prices.get(trade.symbol, trade.estimated_price),
                            value=trade.estimated_value,
                            commission=trade.estimated_commission,
                            reason="rebalance",
                        )
                        self.trade_history.append(record)

            # Registrar historiales
            self._record_history(timestamp)

        # Calcular metricas finales
        metrics = self._calculate_metrics()

        # Construir resultado
        result = PortfolioResult(
            config=self.config,
            metrics=metrics,
            equity_curve=self.equity_curve,
            returns=self.returns,
            drawdown_curve=self.drawdown_curve,
            timestamps=self.timestamps,
            weight_history=self.weight_history,
            trade_history=self.trade_history,
            rebalance_history=self.rebalance_history,
            final_state=self.state,
            start_date=self.timestamps[0] if self.timestamps else None,
            end_date=self.timestamps[-1] if self.timestamps else None,
            execution_time=time.time() - start_time,
        )

        return result

    def _reset(self) -> None:
        """Reinicia el estado del backtester"""
        self.state = PortfolioState(
            total_equity=self.config.initial_capital,
            cash=self.config.initial_capital,
            invested_value=0.0,
            target_weights={s: 1.0 / len(self.config.symbols) for s in self.config.symbols},
        )

        self.equity_curve = []
        self.returns = []
        self.drawdown_curve = []
        self.timestamps = []
        self.weight_history = {s: [] for s in self.config.symbols}
        self.trade_history = []
        self.rebalance_history = []

        self.peak_equity = self.config.initial_capital
        self.total_commission = 0.0
        self.total_slippage = 0.0

        self.rebalancer.reset()

    def _calculate_initial_weights(self, warmup_period: int) -> Dict[str, float]:
        """Calcula los pesos iniciales usando el periodo de warmup"""
        if self.returns_data is None or len(self.returns_data) < warmup_period:
            # Equal weight si no hay suficientes datos
            n = len(self.config.symbols)
            return {s: 1.0 / n for s in self.config.symbols}

        warmup_returns = self.returns_data.iloc[:warmup_period]

        result = self.allocator.calculate_weights(
            returns=warmup_returns.values,
            symbols=self.config.symbols,
            method=self.config.allocation_method,
            target_weights=self.config.target_weights if self.config.allocation_method == AllocationMethod.CUSTOM else None,
        )

        return result.weights

    def _recalculate_weights(self, returns: pd.DataFrame) -> Dict[str, float]:
        """Recalcula los pesos con datos recientes"""
        result = self.allocator.calculate_weights(
            returns=returns.values,
            symbols=self.config.symbols,
            method=self.config.allocation_method,
        )
        return result.weights

    def _initialize_positions(self, prices: Dict[str, float]) -> None:
        """Inicializa las posiciones segun los pesos target"""
        for symbol in self.config.symbols:
            weight = self.state.target_weights.get(symbol, 0.0)
            price = prices.get(symbol, 0.0)

            if weight <= 0 or price <= 0:
                continue

            # Valor a invertir
            value = self.state.total_equity * weight

            # Aplicar slippage
            exec_price = price * (1 + self.config.slippage)
            commission = value * self.config.commission

            # Cantidad
            quantity = (value - commission) / exec_price

            self.state.positions[symbol] = PortfolioPosition(
                symbol=symbol,
                quantity=quantity,
                entry_price=exec_price,
                current_price=price,
            )

            self.state.cash -= value
            self.total_commission += commission

            # Registrar trade
            record = PortfolioTradeRecord(
                symbol=symbol,
                side="buy",
                quantity=quantity,
                price=exec_price,
                value=value,
                commission=commission,
                reason="initial",
            )
            self.trade_history.append(record)

        self.state.invested_value = sum(p.value for p in self.state.positions.values())
        self.state.update_weights()

        # Marcar rebalanceo inicial
        self.rebalancer.last_rebalance_date = datetime.now()

    def _update_prices(self, prices: Dict[str, float]) -> None:
        """Actualiza los precios actuales de las posiciones"""
        for symbol, position in self.state.positions.items():
            if symbol in prices:
                position.current_price = prices[symbol]

    def _update_state(self, timestamp: datetime) -> None:
        """Actualiza el estado del portfolio"""
        self.state.timestamp = timestamp
        self.state.invested_value = sum(p.value for p in self.state.positions.values())
        self.state.total_equity = self.state.cash + self.state.invested_value
        self.state.unrealized_pnl = sum(p.pnl for p in self.state.positions.values())
        self.state.update_weights()

        # Actualizar peak y drawdown
        if self.state.total_equity > self.peak_equity:
            self.peak_equity = self.state.total_equity

    def _record_history(self, timestamp: datetime) -> None:
        """Registra el estado actual en los historiales"""
        # Equity
        self.equity_curve.append(self.state.total_equity)
        self.timestamps.append(timestamp)

        # Returns
        if len(self.equity_curve) > 1:
            ret = (self.equity_curve[-1] / self.equity_curve[-2]) - 1
        else:
            ret = 0.0
        self.returns.append(ret)

        # Drawdown
        dd = (self.peak_equity - self.state.total_equity) / self.peak_equity if self.peak_equity > 0 else 0.0
        self.drawdown_curve.append(dd)

        # Weights
        for symbol in self.config.symbols:
            weight = self.state.current_weights.get(symbol, 0.0)
            self.weight_history[symbol].append(weight)

    def _calculate_metrics(self) -> PortfolioMetrics:
        """Calcula todas las metricas del backtest"""
        metrics = PortfolioMetrics()

        if not self.equity_curve or not self.returns:
            return metrics

        returns_array = np.array(self.returns)
        equity_array = np.array(self.equity_curve)

        # Rentabilidad
        metrics.total_return = equity_array[-1] - self.config.initial_capital
        metrics.total_return_pct = (equity_array[-1] / self.config.initial_capital - 1) * 100

        # Dias
        metrics.total_days = len(self.equity_curve)
        metrics.trading_days = metrics.total_days  # Simplificado

        # Anualizado
        years = metrics.trading_days / 252.0
        if years > 0:
            metrics.cagr = (equity_array[-1] / self.config.initial_capital) ** (1 / years) - 1
            metrics.annualized_return = metrics.cagr

        # Volatilidad
        if len(returns_array) > 1:
            metrics.volatility = float(np.std(returns_array))
            metrics.annualized_volatility = metrics.volatility * np.sqrt(252)

        # Drawdown
        dd_array = np.array(self.drawdown_curve)
        metrics.max_drawdown = float(np.max(dd_array)) if len(dd_array) > 0 else 0.0

        # Calcular duracion del max drawdown
        if metrics.max_drawdown > 0:
            in_drawdown = dd_array > 0
            dd_lengths = []
            current_length = 0
            for is_dd in in_drawdown:
                if is_dd:
                    current_length += 1
                else:
                    if current_length > 0:
                        dd_lengths.append(current_length)
                    current_length = 0
            if current_length > 0:
                dd_lengths.append(current_length)
            metrics.max_drawdown_duration = max(dd_lengths) if dd_lengths else 0

        # VaR y CVaR
        if len(returns_array) > 20:
            metrics.var_95 = float(np.percentile(returns_array, 5))
            metrics.cvar_95 = float(np.mean(returns_array[returns_array <= metrics.var_95]))

        # Ratios
        if metrics.annualized_volatility > 0:
            excess_return = metrics.annualized_return - self.config.risk_free_rate
            metrics.sharpe_ratio = excess_return / metrics.annualized_volatility

        # Sortino (usando solo retornos negativos)
        negative_returns = returns_array[returns_array < 0]
        if len(negative_returns) > 0:
            downside_std = float(np.std(negative_returns)) * np.sqrt(252)
            if downside_std > 0:
                excess_return = metrics.annualized_return - self.config.risk_free_rate
                metrics.sortino_ratio = excess_return / downside_std

        # Calmar
        if metrics.max_drawdown > 0:
            metrics.calmar_ratio = metrics.annualized_return / metrics.max_drawdown

        # Portfolio metrics
        metrics = self._calculate_portfolio_specific_metrics(metrics)

        # Costos
        metrics.total_commission = self.total_commission
        metrics.total_slippage = self.total_slippage
        metrics.rebalancing_cost = self.total_commission + self.total_slippage
        metrics.num_rebalances = len(self.rebalance_history)

        # Turnover
        if len(self.rebalance_history) > 0 and years > 0:
            total_turnover = sum(e.total_value_traded for e in self.rebalance_history)
            avg_equity = np.mean(equity_array)
            metrics.turnover = (total_turnover / avg_equity) / years if avg_equity > 0 else 0.0

        return metrics

    def _calculate_portfolio_specific_metrics(
        self,
        metrics: PortfolioMetrics
    ) -> PortfolioMetrics:
        """Calcula metricas especificas de portfolio"""
        if self.returns_data is None or len(self.returns_data) < 2:
            return metrics

        # Matriz de correlacion
        corr_matrix = self.returns_data.corr()
        n = len(self.config.symbols)

        # Correlacion promedio (excluyendo diagonal)
        if n > 1:
            corr_values = corr_matrix.values
            upper_triangle = corr_values[np.triu_indices(n, k=1)]
            metrics.avg_correlation = float(np.mean(np.abs(upper_triangle)))

        # Concentration ratio (HHI)
        if self.state and self.state.current_weights:
            weights = list(self.state.current_weights.values())
            metrics.concentration_ratio = sum(w ** 2 for w in weights)

            # Effective N
            if metrics.concentration_ratio > 0:
                metrics.effective_n = 1.0 / metrics.concentration_ratio

        # Diversification ratio
        if len(self.returns_data) >= 20 and self.state:
            # Vol ponderada promedio
            vols = self.returns_data.std() * np.sqrt(252)
            weights = np.array([self.state.current_weights.get(s, 0) for s in self.config.symbols])

            weighted_avg_vol = np.sum(weights * vols)

            # Vol del portfolio
            cov_matrix = self.returns_data.cov() * 252
            portfolio_vol = np.sqrt(weights @ cov_matrix.values @ weights)

            if portfolio_vol > 0:
                metrics.diversification_ratio = weighted_avg_vol / portfolio_vol

        # Contribution to return
        if len(self.weight_history) > 0 and len(self.returns_data) > 0:
            for symbol in self.config.symbols:
                if symbol in self.weight_history and len(self.weight_history[symbol]) > 0:
                    avg_weight = np.mean(self.weight_history[symbol])
                    symbol_return = (self.returns_data[symbol].mean() * 252) if symbol in self.returns_data else 0.0
                    metrics.contribution_to_return[symbol] = avg_weight * symbol_return

        return metrics

    def get_equity_curve(self) -> pd.Series:
        """Retorna la equity curve como Serie de pandas"""
        return pd.Series(self.equity_curve, index=self.timestamps, name="equity")

    def get_returns(self) -> pd.Series:
        """Retorna los retornos como Serie de pandas"""
        return pd.Series(self.returns, index=self.timestamps, name="returns")

    def get_drawdown_curve(self) -> pd.Series:
        """Retorna la curva de drawdown como Serie de pandas"""
        return pd.Series(self.drawdown_curve, index=self.timestamps, name="drawdown")

    def get_weight_history(self) -> pd.DataFrame:
        """Retorna el historial de pesos como DataFrame"""
        return pd.DataFrame(self.weight_history, index=self.timestamps)

    def walk_forward_backtest(
        self,
        n_splits: int = 5,
        train_pct: float = 0.6,
        anchored: bool = True,
        gap: int = 0,
        warmup_period: int = 30,
        stability_threshold: float = 0.3,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> PortfolioWalkForwardResult:
        """
        Ejecuta walk-forward analysis para el portfolio.

        En cada split:
        1. Optimiza los pesos del portfolio en el periodo de training
        2. Aplica esos pesos fijos en el periodo de test
        3. Compara rendimiento IS vs OOS

        Args:
            n_splits: Numero de splits walk-forward
            train_pct: Porcentaje de datos para training en el primer split
            anchored: Si True, ventana expandible. Si False, ventana fija.
            gap: Filas de separacion entre train y test
            warmup_period: Periodos para calcular allocation inicial
            stability_threshold: Umbral de CV para considerar peso inestable
            progress_callback: Callback para reportar progreso (split, total, phase)

        Returns:
            PortfolioWalkForwardResult con todos los resultados
        """
        start_time = time.time()

        if self.aligned_data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        n_samples = len(self.aligned_data)

        # Calculate split boundaries
        splits = self._calculate_walk_forward_splits(
            n_samples=n_samples,
            n_splits=n_splits,
            train_pct=train_pct,
            anchored=anchored,
            gap=gap,
        )

        result = PortfolioWalkForwardResult(
            config=self.config,
            n_splits=n_splits,
            anchored=anchored,
        )

        # Track weights per split for stability analysis
        weights_per_split: Dict[str, List[float]] = {
            s: [] for s in self.config.symbols
        }

        # Process each split
        for split_idx, (train_start, train_end, test_start, test_end) in enumerate(splits):
            if progress_callback:
                progress_callback(split_idx + 1, n_splits, "processing")

            # Get data subsets
            train_data = {
                symbol: df.iloc[train_start:train_end].copy()
                for symbol, df in self.data.items()
            }
            test_data = {
                symbol: df.iloc[test_start:test_end].copy()
                for symbol, df in self.data.items()
            }

            # Get timestamps for split info
            train_start_ts = self.aligned_data.index[train_start]
            train_end_ts = self.aligned_data.index[train_end - 1]
            test_start_ts = self.aligned_data.index[test_start]
            test_end_ts = self.aligned_data.index[min(test_end - 1, n_samples - 1)]

            # Optimize weights on training data
            optimized_weights = self._optimize_weights_on_data(
                train_data, warmup_period
            )

            # Track weights for stability
            for symbol, weight in optimized_weights.items():
                weights_per_split[symbol].append(weight)

            # Backtest on train period (IS)
            train_result = self._backtest_subset(
                train_data, optimized_weights, warmup_period
            )

            # Backtest on test period (OOS) with fixed weights
            test_result = self._backtest_subset(
                test_data, optimized_weights, min(warmup_period, len(test_data[self.config.symbols[0]]) // 2)
            )

            # Calculate degradation
            train_sharpe = train_result.metrics.sharpe_ratio
            test_sharpe = test_result.metrics.sharpe_ratio
            train_return = train_result.metrics.total_return_pct
            test_return = test_result.metrics.total_return_pct

            if abs(train_sharpe) > 0.001:
                degradation = (train_sharpe - test_sharpe) / abs(train_sharpe)
            else:
                degradation = 0.0

            # Create split result
            split_result = PortfolioSplitResult(
                split_idx=split_idx,
                train_start=train_start_ts.to_pydatetime() if hasattr(train_start_ts, 'to_pydatetime') else train_start_ts,  # type: ignore
                train_end=train_end_ts.to_pydatetime() if hasattr(train_end_ts, 'to_pydatetime') else train_end_ts,  # type: ignore
                test_start=test_start_ts.to_pydatetime() if hasattr(test_start_ts, 'to_pydatetime') else test_start_ts,  # type: ignore
                test_end=test_end_ts.to_pydatetime() if hasattr(test_end_ts, 'to_pydatetime') else test_end_ts,  # type: ignore
                train_rows=train_end - train_start,
                test_rows=test_end - test_start,
                optimized_weights=optimized_weights,
                train_metrics=train_result.metrics,
                test_metrics=test_result.metrics,
                train_sharpe=train_sharpe,
                test_sharpe=test_sharpe,
                train_return=train_return,
                test_return=test_return,
                degradation_pct=degradation,
                train_equity=train_result.equity_curve,
                test_equity=test_result.equity_curve,
            )

            result.splits.append(split_result)

            # Accumulate OOS equity
            if test_result.equity_curve:
                # Normalize to connect with previous
                if result.combined_oos_equity:
                    scale_factor = result.combined_oos_equity[-1] / test_result.equity_curve[0]
                    scaled_equity = [e * scale_factor for e in test_result.equity_curve]
                else:
                    scaled_equity = test_result.equity_curve

                result.combined_oos_equity.extend(scaled_equity)
                result.combined_oos_returns.extend(test_result.returns)
                result.combined_oos_timestamps.extend(test_result.timestamps)

        # Calculate aggregated metrics
        if result.splits:
            result.avg_train_sharpe = float(np.mean([s.train_sharpe for s in result.splits]))
            result.avg_test_sharpe = float(np.mean([s.test_sharpe for s in result.splits]))
            result.avg_train_return = float(np.mean([s.train_return for s in result.splits]))
            result.avg_test_return = float(np.mean([s.test_return for s in result.splits]))
            result.avg_degradation = float(np.mean([s.degradation_pct for s in result.splits]))

            # Positive OOS ratio (splits where test Sharpe > 0)
            positive_oos = sum(1 for s in result.splits if s.test_sharpe > 0)
            result.positive_oos_ratio = positive_oos / len(result.splits)

            # Consistency ratio (splits where test Sharpe >= train Sharpe * 0.5)
            consistent = sum(
                1 for s in result.splits
                if s.test_sharpe >= s.train_sharpe * 0.5 or s.train_sharpe <= 0
            )
            result.consistency_ratio = consistent / len(result.splits)

            # Robustness score
            result.robustness_score = self._calculate_robustness_score(
                positive_oos_ratio=result.positive_oos_ratio,
                consistency_ratio=result.consistency_ratio,
                avg_degradation=result.avg_degradation,
            )

        # Calculate weight stability
        result.weight_stability = {}
        result.unstable_allocations = []

        for symbol, weights in weights_per_split.items():
            if len(weights) > 1:
                mean_weight = float(np.mean(weights))
                std_weight = float(np.std(weights))

                if mean_weight > 0.01:  # Only for meaningful allocations
                    cv = std_weight / mean_weight
                    result.weight_stability[symbol] = cv

                    if cv > stability_threshold:
                        result.unstable_allocations.append(symbol)

        # Calculate metrics on combined OOS
        if result.combined_oos_equity:
            result.combined_oos_metrics = self._calculate_metrics_from_equity(
                equity_curve=result.combined_oos_equity,
                returns=result.combined_oos_returns,
            )

        result.execution_time = time.time() - start_time

        return result

    def _calculate_walk_forward_splits(
        self,
        n_samples: int,
        n_splits: int,
        train_pct: float,
        anchored: bool,
        gap: int,
    ) -> List[Tuple[int, int, int, int]]:
        """Calculate train/test boundaries for each split.

        Returns:
            List of (train_start, train_end, test_start, test_end) tuples.
        """
        splits = []

        initial_train_size = int(n_samples * train_pct)
        remaining = n_samples - initial_train_size - (gap * n_splits)
        test_size_per_split = remaining // n_splits

        rolling_train_size = initial_train_size

        for split_idx in range(n_splits):
            if anchored:
                train_start = 0
                train_end = initial_train_size + (split_idx * test_size_per_split)
            else:
                train_start = split_idx * test_size_per_split
                train_end = train_start + rolling_train_size

            test_start = train_end + gap
            test_end = test_start + test_size_per_split

            if test_end > n_samples:
                test_end = n_samples

            if test_start >= n_samples:
                break

            splits.append((train_start, train_end, test_start, test_end))

        return splits

    def _optimize_weights_on_data(
        self,
        data: Dict[str, pd.DataFrame],
        warmup_period: int,
    ) -> Dict[str, float]:
        """Optimize portfolio weights using the given data."""
        # Align and calculate returns
        common_index = None
        for symbol, df in data.items():
            if common_index is None:
                common_index = df.index
            else:
                common_index = common_index.intersection(df.index)

        if common_index is None or len(common_index) < warmup_period:
            # Equal weights if not enough data
            n = len(self.config.symbols)
            return {s: 1.0 / n for s in self.config.symbols}

        aligned = pd.DataFrame({
            symbol: data[symbol].loc[common_index, 'close']
            for symbol in self.config.symbols
        })

        returns = aligned.pct_change().dropna()

        if len(returns) < 20:
            n = len(self.config.symbols)
            return {s: 1.0 / n for s in self.config.symbols}

        # Use allocator to calculate optimal weights
        result = self.allocator.calculate_weights(
            returns=returns.values,
            symbols=self.config.symbols,
            method=self.config.allocation_method,
            target_weights=self.config.target_weights if self.config.allocation_method == AllocationMethod.CUSTOM else None,
        )

        return result.weights

    def _backtest_subset(
        self,
        data: Dict[str, pd.DataFrame],
        weights: Dict[str, float],
        warmup_period: int,
    ) -> PortfolioResult:
        """Run backtest on a data subset with fixed weights."""
        # Create a temporary backtester
        temp_config = PortfolioConfig(
            initial_capital=self.config.initial_capital,
            symbols=self.config.symbols,
            timeframe=self.config.timeframe,
            allocation_method=AllocationMethod.CUSTOM,
            target_weights=weights,
            rebalance_frequency=self.config.rebalance_frequency,
            rebalance_threshold=self.config.rebalance_threshold,
            min_trade_value=self.config.min_trade_value,
            commission=self.config.commission,
            slippage=self.config.slippage,
            max_position_weight=self.config.max_position_weight,
            min_position_weight=self.config.min_position_weight,
            max_positions=self.config.max_positions,
            risk_free_rate=self.config.risk_free_rate,
            target_volatility=self.config.target_volatility,
        )

        temp_backtester = PortfolioBacktester(temp_config)
        temp_backtester.load_data(data)

        # Adjust warmup if data is too small
        data_len = min(len(df) for df in data.values())
        actual_warmup = min(warmup_period, data_len // 3)
        actual_warmup = max(5, actual_warmup)

        if data_len <= actual_warmup + 10:
            # Not enough data for meaningful backtest
            return PortfolioResult(config=temp_config)

        return temp_backtester.run(warmup_period=actual_warmup)

    def _calculate_robustness_score(
        self,
        positive_oos_ratio: float,
        consistency_ratio: float,
        avg_degradation: float,
    ) -> float:
        """Calculate robustness score (0-1).

        Components:
        - 40% positive OOS ratio
        - 30% consistency ratio
        - 30% low degradation
        """
        # Degradation component: 1 when degradation is 0, 0 when degradation >= 1
        degradation_component = max(0, 1 - abs(avg_degradation))

        score = (
            0.4 * positive_oos_ratio +
            0.3 * consistency_ratio +
            0.3 * degradation_component
        )

        return min(1.0, max(0.0, score))

    def _calculate_metrics_from_equity(
        self,
        equity_curve: List[float],
        returns: List[float],
    ) -> PortfolioMetrics:
        """Calculate metrics from equity curve and returns."""
        metrics = PortfolioMetrics()

        if not equity_curve or len(equity_curve) < 2:
            return metrics

        returns_array = np.array(returns)
        equity_array = np.array(equity_curve)

        initial = equity_curve[0]
        final = equity_curve[-1]

        # Returns
        metrics.total_return = final - initial
        metrics.total_return_pct = ((final / initial) - 1) * 100 if initial > 0 else 0.0

        # Days
        metrics.total_days = len(equity_curve)
        metrics.trading_days = metrics.total_days

        # Annualized
        years = metrics.trading_days / 252.0
        if years > 0:
            metrics.cagr = (final / initial) ** (1 / years) - 1 if initial > 0 else 0.0
            metrics.annualized_return = metrics.cagr

        # Volatility
        if len(returns_array) > 1:
            metrics.volatility = float(np.std(returns_array))
            metrics.annualized_volatility = metrics.volatility * np.sqrt(252)

        # Drawdown
        peak = np.maximum.accumulate(equity_array)
        drawdown = (peak - equity_array) / peak
        metrics.max_drawdown = float(np.max(drawdown))

        # Sharpe
        if metrics.annualized_volatility > 0:
            excess_return = metrics.annualized_return - self.config.risk_free_rate
            metrics.sharpe_ratio = excess_return / metrics.annualized_volatility

        # Sortino
        negative_returns = returns_array[returns_array < 0]
        if len(negative_returns) > 0:
            downside_std = float(np.std(negative_returns)) * np.sqrt(252)
            if downside_std > 0:
                excess_return = metrics.annualized_return - self.config.risk_free_rate
                metrics.sortino_ratio = excess_return / downside_std

        # Calmar
        if metrics.max_drawdown > 0:
            metrics.calmar_ratio = metrics.annualized_return / metrics.max_drawdown

        return metrics
