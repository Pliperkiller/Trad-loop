"""
Orquestador principal del modulo de portfolio.

Integra todos los componentes:
- Asset allocation
- Rebalanceo
- Backtesting
- Metricas
"""

from typing import Dict, List, Optional
import pandas as pd

from .models import (
    PortfolioConfig,
    PortfolioState,
    PortfolioResult,
    PortfolioMetrics,
    AllocationMethod,
    RebalanceFrequency,
)
from .allocator import PortfolioAllocator, AllocationResult
from .rebalancer import PortfolioRebalancer
from .backtester import PortfolioBacktester
from .metrics import PortfolioMetricsCalculator


class PortfolioManager:
    """
    Gestor principal de portfolios multi-activo.

    Ejemplo de uso:
        config = PortfolioConfig(
            initial_capital=10000,
            symbols=['BTC/USDT', 'ETH/USDT', 'SOL/USDT'],
            allocation_method=AllocationMethod.RISK_PARITY,
        )

        manager = PortfolioManager(config)
        result = manager.backtest(data_dict)
        manager.print_summary()
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
        self.backtester: Optional[PortfolioBacktester] = None
        self.metrics_calculator: Optional[PortfolioMetricsCalculator] = None

        # Resultados
        self.last_result: Optional[PortfolioResult] = None
        self.returns_data: Optional[pd.DataFrame] = None

    def backtest(
        self,
        data: Dict[str, pd.DataFrame],
        warmup_period: int = 30
    ) -> PortfolioResult:
        """
        Ejecuta un backtest del portfolio.

        Args:
            data: Dict symbol -> DataFrame OHLCV
            warmup_period: Periodos de warmup para allocation

        Returns:
            PortfolioResult con resultados completos
        """
        self.backtester = PortfolioBacktester(self.config)
        self.backtester.load_data(data)
        result = self.backtester.run(warmup_period=warmup_period)

        self.last_result = result
        self.returns_data = self.backtester.returns_data

        return result

    def optimize_weights(
        self,
        returns: Optional[pd.DataFrame] = None,
        method: Optional[AllocationMethod] = None
    ) -> AllocationResult:
        """
        Calcula los pesos optimos.

        Args:
            returns: DataFrame de retornos (opcional, usa cached)
            method: Metodo de allocation (opcional, usa config)

        Returns:
            AllocationResult con pesos calculados
        """
        if returns is None:
            returns = self.returns_data

        if returns is None:
            raise ValueError("No returns data available. Run backtest first or provide returns.")

        if method is None:
            method = self.config.allocation_method

        return self.allocator.calculate_weights(
            returns=returns.values,
            symbols=self.config.symbols,
            method=method,
        )

    def set_target_weights(self, weights: Dict[str, float]) -> None:
        """
        Establece pesos objetivo personalizados.

        Args:
            weights: Dict symbol -> weight
        """
        # Validar que sumen ~1
        total = sum(weights.values())
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Weights must sum to 1.0, got {total}")

        self.config.target_weights = weights
        self.config.allocation_method = AllocationMethod.CUSTOM

    def get_metrics(self) -> Optional[PortfolioMetrics]:
        """
        Obtiene las metricas del ultimo backtest.

        Returns:
            PortfolioMetrics o None si no hay backtest
        """
        if self.last_result:
            return self.last_result.metrics
        return None

    def get_current_weights(self) -> Dict[str, float]:
        """
        Obtiene los pesos actuales del portfolio.

        Returns:
            Dict symbol -> weight
        """
        if self.last_result and self.last_result.final_state:
            return self.last_result.final_state.current_weights
        return {s: 1.0 / len(self.config.symbols) for s in self.config.symbols}

    def get_weight_drift(self) -> Dict[str, float]:
        """
        Obtiene el drift de cada peso vs target.

        Returns:
            Dict symbol -> drift
        """
        current = self.get_current_weights()
        target = self.config.target_weights or {
            s: 1.0 / len(self.config.symbols) for s in self.config.symbols
        }

        return {s: current.get(s, 0) - target.get(s, 0) for s in self.config.symbols}

    def get_correlation_matrix(self) -> Optional[pd.DataFrame]:
        """
        Obtiene la matriz de correlacion.

        Returns:
            DataFrame con matriz de correlacion
        """
        if self.returns_data is not None:
            return self.returns_data.corr()
        return None

    def get_efficient_frontier(
        self,
        n_points: int = 50
    ) -> List[Dict]:
        """
        Calcula la frontera eficiente.

        Args:
            n_points: Numero de puntos en la frontera

        Returns:
            Lista de dicts con (return, volatility, weights)
        """
        if self.returns_data is None:
            return []

        frontier = self.allocator.get_efficient_frontier(
            returns=self.returns_data.values,
            symbols=self.config.symbols,
            n_points=n_points,
        )

        return [
            {"return": ret, "volatility": vol, "weights": weights}
            for ret, vol, weights in frontier
        ]

    def calculate_portfolio_stats(
        self,
        weights: Optional[Dict[str, float]] = None
    ) -> Dict:
        """
        Calcula estadisticas del portfolio para pesos dados.

        Args:
            weights: Pesos a evaluar (default: actuales)

        Returns:
            Dict con estadisticas
        """
        if weights is None:
            weights = self.get_current_weights()

        if self.returns_data is None:
            return {}

        calc = PortfolioMetricsCalculator(
            returns=self.returns_data,
            weights=weights,
            risk_free_rate=self.config.risk_free_rate,
        )

        return calc.generate_report()

    def compare_allocations(
        self,
        methods: Optional[List[AllocationMethod]] = None
    ) -> pd.DataFrame:
        """
        Compara diferentes metodos de allocation.

        Args:
            methods: Lista de metodos a comparar

        Returns:
            DataFrame con comparacion
        """
        if self.returns_data is None:
            raise ValueError("No returns data. Run backtest first.")

        if methods is None:
            methods = [
                AllocationMethod.EQUAL_WEIGHT,
                AllocationMethod.RISK_PARITY,
                AllocationMethod.MIN_VARIANCE,
                AllocationMethod.MAX_SHARPE,
            ]

        results = []

        for method in methods:
            allocation = self.allocator.calculate_weights(
                returns=self.returns_data.values,
                symbols=self.config.symbols,
                method=method,
            )

            results.append({
                "method": method.value,
                "expected_return": allocation.expected_return,
                "expected_volatility": allocation.expected_volatility,
                "sharpe_ratio": allocation.sharpe_ratio,
                "success": allocation.success,
                **{f"weight_{s}": allocation.weights.get(s, 0) for s in self.config.symbols}
            })

        return pd.DataFrame(results)

    def print_summary(self) -> None:
        """Imprime un resumen del ultimo backtest"""
        if not self.last_result:
            print("No backtest results available. Run backtest() first.")
            return

        m = self.last_result.metrics
        print("\n" + "=" * 60)
        print("PORTFOLIO BACKTEST SUMMARY")
        print("=" * 60)

        print(f"\nConfiguration:")
        print(f"  Symbols: {', '.join(self.config.symbols)}")
        print(f"  Initial Capital: ${self.config.initial_capital:,.2f}")
        print(f"  Allocation Method: {self.config.allocation_method.value}")
        print(f"  Rebalance Frequency: {self.config.rebalance_frequency.value}")

        print(f"\nPerformance:")
        print(f"  Total Return: {m.total_return_pct:+.2f}%")
        print(f"  CAGR: {m.cagr * 100:+.2f}%")
        print(f"  Volatility (ann.): {m.annualized_volatility * 100:.2f}%")
        print(f"  Max Drawdown: {m.max_drawdown * 100:.2f}%")

        print(f"\nRisk Ratios:")
        print(f"  Sharpe Ratio: {m.sharpe_ratio:.2f}")
        print(f"  Sortino Ratio: {m.sortino_ratio:.2f}")
        print(f"  Calmar Ratio: {m.calmar_ratio:.2f}")

        print(f"\nPortfolio Metrics:")
        print(f"  Diversification Ratio: {m.diversification_ratio:.2f}")
        print(f"  Effective N: {m.effective_n:.1f}")
        print(f"  Avg Correlation: {m.avg_correlation:.2f}")

        print(f"\nCosts:")
        print(f"  Total Commission: ${m.total_commission:.2f}")
        print(f"  Turnover (ann.): {m.turnover * 100:.1f}%")
        print(f"  Rebalances: {m.num_rebalances}")

        if self.last_result.final_state:
            print(f"\nFinal Weights:")
            for symbol, weight in sorted(
                self.last_result.final_state.current_weights.items(),
                key=lambda x: x[1],
                reverse=True
            ):
                print(f"  {symbol}: {weight * 100:.1f}%")

        print("\n" + "=" * 60)

    def get_equity_curve(self) -> Optional[pd.Series]:
        """Retorna la equity curve del ultimo backtest"""
        if self.backtester:
            return self.backtester.get_equity_curve()
        return None

    def get_returns(self) -> Optional[pd.Series]:
        """Retorna los retornos del portfolio"""
        if self.backtester:
            return self.backtester.get_returns()
        return None

    def get_weight_history(self) -> Optional[pd.DataFrame]:
        """Retorna el historial de pesos"""
        if self.backtester:
            return self.backtester.get_weight_history()
        return None


class PortfolioManagerFactory:
    """Factory para crear managers preconfigurados"""

    @staticmethod
    def create_equal_weight(
        symbols: List[str],
        initial_capital: float = 10000,
        rebalance_frequency: RebalanceFrequency = RebalanceFrequency.MONTHLY
    ) -> PortfolioManager:
        """Crea manager con equal weight"""
        config = PortfolioConfig(
            initial_capital=initial_capital,
            symbols=symbols,
            allocation_method=AllocationMethod.EQUAL_WEIGHT,
            rebalance_frequency=rebalance_frequency,
        )
        return PortfolioManager(config)

    @staticmethod
    def create_risk_parity(
        symbols: List[str],
        initial_capital: float = 10000,
        rebalance_frequency: RebalanceFrequency = RebalanceFrequency.MONTHLY
    ) -> PortfolioManager:
        """Crea manager con risk parity"""
        config = PortfolioConfig(
            initial_capital=initial_capital,
            symbols=symbols,
            allocation_method=AllocationMethod.RISK_PARITY,
            rebalance_frequency=rebalance_frequency,
        )
        return PortfolioManager(config)

    @staticmethod
    def create_max_sharpe(
        symbols: List[str],
        initial_capital: float = 10000,
        rebalance_frequency: RebalanceFrequency = RebalanceFrequency.MONTHLY
    ) -> PortfolioManager:
        """Crea manager optimizado para max Sharpe"""
        config = PortfolioConfig(
            initial_capital=initial_capital,
            symbols=symbols,
            allocation_method=AllocationMethod.MAX_SHARPE,
            rebalance_frequency=rebalance_frequency,
        )
        return PortfolioManager(config)

    @staticmethod
    def create_custom(
        symbols: List[str],
        weights: Dict[str, float],
        initial_capital: float = 10000,
        rebalance_frequency: RebalanceFrequency = RebalanceFrequency.MONTHLY
    ) -> PortfolioManager:
        """Crea manager con pesos personalizados"""
        config = PortfolioConfig(
            initial_capital=initial_capital,
            symbols=symbols,
            allocation_method=AllocationMethod.CUSTOM,
            target_weights=weights,
            rebalance_frequency=rebalance_frequency,
        )
        return PortfolioManager(config)
