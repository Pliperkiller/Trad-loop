"""
Calculador de metricas especificas de portfolio.

Incluye:
- Metricas de diversificacion
- Contribuciones a retorno y riesgo
- Tracking error y alpha
- Analisis de correlacion
"""

from dataclasses import dataclass, field
from typing import Dict, Optional
import numpy as np
import pandas as pd


@dataclass
class DiversificationMetrics:
    """Metricas de diversificacion del portfolio"""
    diversification_ratio: float = 0.0
    concentration_hhi: float = 0.0
    effective_n: float = 0.0
    max_weight: float = 0.0
    min_weight: float = 0.0
    weight_std: float = 0.0


@dataclass
class CorrelationMetrics:
    """Metricas de correlacion del portfolio"""
    avg_correlation: float = 0.0
    max_correlation: float = 0.0
    min_correlation: float = 0.0
    highly_correlated_pairs: int = 0
    correlation_matrix: Optional[np.ndarray] = None


@dataclass
class ContributionMetrics:
    """Metricas de contribucion por activo"""
    return_contribution: Dict[str, float] = field(default_factory=dict)
    risk_contribution: Dict[str, float] = field(default_factory=dict)
    marginal_contribution: Dict[str, float] = field(default_factory=dict)


class PortfolioMetricsCalculator:
    """
    Calculador avanzado de metricas de portfolio.

    Ejemplo de uso:
        calculator = PortfolioMetricsCalculator(
            returns=returns_df,
            weights=weights_dict,
            risk_free_rate=0.02
        )
        div_metrics = calculator.calculate_diversification()
        contrib = calculator.calculate_contributions()
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        weights: Dict[str, float],
        risk_free_rate: float = 0.02,
        benchmark_returns: Optional[pd.Series] = None,
    ):
        """
        Args:
            returns: DataFrame de retornos (timestamps x symbols)
            weights: Dict symbol -> weight actual
            risk_free_rate: Tasa libre de riesgo anualizada
            benchmark_returns: Serie de retornos del benchmark (opcional)
        """
        self.returns = returns
        self.weights = weights
        self.symbols = list(weights.keys())
        self.risk_free_rate = risk_free_rate
        self.benchmark_returns = benchmark_returns

        # Pre-calcular estadisticas
        self.mean_returns: pd.Series = returns.mean() * 252  # Anualizado
        self.cov_matrix: pd.DataFrame = returns.cov() * 252
        self.corr_matrix: pd.DataFrame = returns.corr()
        self.volatilities: pd.Series = returns.std() * np.sqrt(252)

        # Weights como array
        self.weights_array = np.array([weights.get(s, 0) for s in self.symbols])

    def calculate_diversification(self) -> DiversificationMetrics:
        """
        Calcula metricas de diversificacion.

        Returns:
            DiversificationMetrics con todas las metricas
        """
        metrics = DiversificationMetrics()

        weights = self.weights_array
        n = len(weights)

        if n == 0 or np.sum(weights) == 0:
            return metrics

        # Diversification Ratio: Vol ponderada / Vol portfolio
        weighted_avg_vol = np.sum(weights * self.volatilities.values)
        portfolio_vol = np.sqrt(weights @ self.cov_matrix.values @ weights)

        if portfolio_vol > 0:
            metrics.diversification_ratio = weighted_avg_vol / portfolio_vol

        # Concentration (HHI)
        metrics.concentration_hhi = np.sum(weights ** 2)

        # Effective N
        if metrics.concentration_hhi > 0:
            metrics.effective_n = 1.0 / metrics.concentration_hhi

        # Weight stats
        positive_weights = weights[weights > 0]
        if len(positive_weights) > 0:
            metrics.max_weight = float(np.max(positive_weights))
            metrics.min_weight = float(np.min(positive_weights))
            metrics.weight_std = float(np.std(positive_weights))

        return metrics

    def calculate_correlation_metrics(self) -> CorrelationMetrics:
        """
        Calcula metricas de correlacion.

        Returns:
            CorrelationMetrics con todas las metricas
        """
        metrics = CorrelationMetrics()

        corr = self.corr_matrix.values
        n = len(self.symbols)

        if n < 2:
            return metrics

        # Obtener triangulo superior (sin diagonal)
        upper_indices = np.triu_indices(n, k=1)
        correlations = corr[upper_indices]

        if len(correlations) > 0:
            metrics.avg_correlation = float(np.mean(correlations))
            metrics.max_correlation = float(np.max(correlations))
            metrics.min_correlation = float(np.min(correlations))

            # Pares altamente correlacionados (>0.7)
            metrics.highly_correlated_pairs = int(np.sum(correlations > 0.7))

        metrics.correlation_matrix = corr

        return metrics

    def calculate_contributions(self) -> ContributionMetrics:
        """
        Calcula las contribuciones de cada activo al retorno y riesgo.

        Returns:
            ContributionMetrics con contribuciones por activo
        """
        metrics = ContributionMetrics()

        weights = self.weights_array
        n = len(self.symbols)

        if n == 0 or np.sum(weights) == 0:
            return metrics

        # Contribution to Return: w_i * r_i
        for i, symbol in enumerate(self.symbols):
            ret = self.mean_returns.get(symbol, 0)
            metrics.return_contribution[symbol] = weights[i] * ret

        # Contribution to Risk
        # RC_i = w_i * (Cov * w)_i / sigma_p
        cov_w = self.cov_matrix.values @ weights
        portfolio_vol = np.sqrt(weights @ self.cov_matrix.values @ weights)

        if portfolio_vol > 0:
            for i, symbol in enumerate(self.symbols):
                risk_contrib = weights[i] * cov_w[i] / portfolio_vol
                metrics.risk_contribution[symbol] = risk_contrib

                # Marginal contribution
                metrics.marginal_contribution[symbol] = cov_w[i] / portfolio_vol

        return metrics

    def calculate_portfolio_return(self) -> float:
        """Calcula el retorno esperado del portfolio"""
        return float(np.dot(self.weights_array, np.array(self.mean_returns.values)))

    def calculate_portfolio_volatility(self) -> float:
        """Calcula la volatilidad del portfolio"""
        return float(np.sqrt(
            self.weights_array @ self.cov_matrix.values @ self.weights_array
        ))

    def calculate_sharpe_ratio(self) -> float:
        """Calcula el Sharpe Ratio del portfolio"""
        ret = self.calculate_portfolio_return()
        vol = self.calculate_portfolio_volatility()

        if vol > 0:
            return (ret - self.risk_free_rate) / vol
        return 0.0

    def calculate_tracking_error(self) -> float:
        """
        Calcula el tracking error vs benchmark.

        Returns:
            Tracking error anualizado
        """
        if self.benchmark_returns is None:
            return 0.0

        # Calcular retornos del portfolio
        portfolio_returns = (self.returns * list(self.weights.values())).sum(axis=1)

        # Alinear con benchmark
        common_idx = portfolio_returns.index.intersection(self.benchmark_returns.index)
        if len(common_idx) < 2:
            return 0.0

        port_ret = portfolio_returns.loc[common_idx]
        bench_ret = self.benchmark_returns.loc[common_idx]

        # Tracking error = std(portfolio - benchmark) * sqrt(252)
        tracking_diff = port_ret - bench_ret
        return float(tracking_diff.std() * np.sqrt(252))

    def calculate_information_ratio(self) -> float:
        """
        Calcula el Information Ratio vs benchmark.

        Returns:
            Information Ratio
        """
        if self.benchmark_returns is None:
            return 0.0

        tracking_error = self.calculate_tracking_error()
        if tracking_error == 0:
            return 0.0

        # Calcular retornos del portfolio
        portfolio_returns = (self.returns * list(self.weights.values())).sum(axis=1)

        # Alpha = ret_portfolio - ret_benchmark
        common_idx = portfolio_returns.index.intersection(self.benchmark_returns.index)
        if len(common_idx) < 2:
            return 0.0

        port_ret = portfolio_returns.loc[common_idx].mean() * 252
        bench_ret = self.benchmark_returns.loc[common_idx].mean() * 252

        alpha = port_ret - bench_ret
        return alpha / tracking_error

    def calculate_beta(self) -> float:
        """
        Calcula el beta del portfolio vs benchmark.

        Returns:
            Beta del portfolio
        """
        if self.benchmark_returns is None:
            return 1.0

        # Calcular retornos del portfolio
        portfolio_returns = (self.returns * list(self.weights.values())).sum(axis=1)

        # Alinear
        common_idx = portfolio_returns.index.intersection(self.benchmark_returns.index)
        if len(common_idx) < 2:
            return 1.0

        port_ret = portfolio_returns.loc[common_idx]
        bench_ret = self.benchmark_returns.loc[common_idx]

        # Beta = Cov(p, b) / Var(b)
        covariance = np.cov(port_ret, bench_ret)[0, 1]
        variance = np.var(bench_ret)

        if variance > 0:
            return covariance / variance
        return 1.0

    def calculate_alpha(self) -> float:
        """
        Calcula el alpha (Jensen's Alpha) vs benchmark.

        Returns:
            Alpha anualizado
        """
        if self.benchmark_returns is None:
            return 0.0

        # Calcular retornos del portfolio
        portfolio_returns = (self.returns * list(self.weights.values())).sum(axis=1)

        # Alinear
        common_idx = portfolio_returns.index.intersection(self.benchmark_returns.index)
        if len(common_idx) < 2:
            return 0.0

        port_ret = portfolio_returns.loc[common_idx].mean() * 252
        bench_ret = self.benchmark_returns.loc[common_idx].mean() * 252

        beta = self.calculate_beta()

        # Alpha = Rp - [Rf + beta * (Rb - Rf)]
        expected_return = self.risk_free_rate + beta * (bench_ret - self.risk_free_rate)
        return port_ret - expected_return

    def calculate_var(self, confidence: float = 0.95) -> float:
        """
        Calcula el Value at Risk del portfolio.

        Args:
            confidence: Nivel de confianza (default 95%)

        Returns:
            VaR como retorno negativo
        """
        # Calcular retornos del portfolio
        portfolio_returns = (self.returns * list(self.weights.values())).sum(axis=1)

        percentile = (1 - confidence) * 100
        return float(np.percentile(portfolio_returns, percentile))

    def calculate_cvar(self, confidence: float = 0.95) -> float:
        """
        Calcula el Conditional VaR (Expected Shortfall).

        Args:
            confidence: Nivel de confianza (default 95%)

        Returns:
            CVaR como retorno negativo
        """
        var = self.calculate_var(confidence)

        # Calcular retornos del portfolio
        portfolio_returns = (self.returns * list(self.weights.values())).sum(axis=1)

        # CVaR es el promedio de retornos por debajo del VaR
        returns_below_var = portfolio_returns[portfolio_returns <= var]

        if len(returns_below_var) > 0:
            return float(returns_below_var.mean())
        return var

    def get_efficient_weights(
        self,
        target_return: Optional[float] = None,
        target_volatility: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Sugiere pesos eficientes dado un objetivo.

        Args:
            target_return: Retorno objetivo (anualizado)
            target_volatility: Volatilidad objetivo (anualizada)

        Returns:
            Dict con pesos sugeridos
        """
        from scipy import optimize

        n = len(self.symbols)

        # Pre-compute numpy arrays for optimization
        mean_ret_arr = np.array(self.mean_returns.values)
        cov_arr = np.array(self.cov_matrix.values)

        if target_return is not None:
            # Minimizar volatilidad para el target return
            def objective_vol(w: np.ndarray) -> float:
                return float(np.sqrt(w @ cov_arr @ w))

            constraints = [
                {"type": "eq", "fun": lambda w: np.sum(w) - 1},
                {"type": "eq", "fun": lambda w: np.dot(w, mean_ret_arr) - target_return}
            ]
            objective = objective_vol

        elif target_volatility is not None:
            # Maximizar retorno para la target volatility
            def objective_ret(w: np.ndarray) -> float:
                return float(-np.dot(w, mean_ret_arr))

            constraints = [
                {"type": "eq", "fun": lambda w: np.sum(w) - 1},
                {"type": "eq", "fun": lambda w: np.sqrt(w @ cov_arr @ w) - target_volatility}
            ]
            objective = objective_ret

        else:
            # Maximizar Sharpe
            rf = self.risk_free_rate

            def objective_sharpe(w: np.ndarray) -> float:
                ret = np.dot(w, mean_ret_arr)
                vol = np.sqrt(w @ cov_arr @ w)
                if vol < 1e-10:
                    return 1e10
                return float(-(ret - rf) / vol)

            constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
            objective = objective_sharpe

        bounds = [(0, 1) for _ in range(n)]
        initial = np.ones(n) / n

        result = optimize.minimize(
            objective,
            initial,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        if result.success:
            weights = result.x / result.x.sum()
            return {s: float(w) for s, w in zip(self.symbols, weights)}

        return self.weights

    def get_risk_parity_weights(self) -> Dict[str, float]:
        """
        Calcula pesos de Risk Parity.

        Returns:
            Dict con pesos de risk parity
        """
        from scipy import optimize

        n = len(self.symbols)

        def risk_parity_objective(w):
            w = w / w.sum()
            vol = np.sqrt(w @ self.cov_matrix.values @ w)

            if vol < 1e-10:
                return 1e10

            marginal = self.cov_matrix.values @ w
            risk_contrib = w * marginal / vol

            target = vol / n
            return np.sum((risk_contrib - target) ** 2)

        bounds = [(0.01, 1) for _ in range(n)]
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        initial = np.ones(n) / n

        result = optimize.minimize(
            risk_parity_objective,
            initial,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        if result.success:
            weights = result.x / result.x.sum()
            return {s: float(w) for s, w in zip(self.symbols, weights)}

        return self.weights

    def generate_report(self) -> Dict:
        """
        Genera un reporte completo de metricas.

        Returns:
            Dict con todas las metricas
        """
        div_metrics = self.calculate_diversification()
        corr_metrics = self.calculate_correlation_metrics()
        contrib = self.calculate_contributions()

        return {
            "portfolio_metrics": {
                "expected_return": self.calculate_portfolio_return(),
                "volatility": self.calculate_portfolio_volatility(),
                "sharpe_ratio": self.calculate_sharpe_ratio(),
                "var_95": self.calculate_var(0.95),
                "cvar_95": self.calculate_cvar(0.95),
            },
            "diversification": {
                "diversification_ratio": div_metrics.diversification_ratio,
                "concentration_hhi": div_metrics.concentration_hhi,
                "effective_n": div_metrics.effective_n,
                "max_weight": div_metrics.max_weight,
                "min_weight": div_metrics.min_weight,
            },
            "correlation": {
                "avg_correlation": corr_metrics.avg_correlation,
                "max_correlation": corr_metrics.max_correlation,
                "min_correlation": corr_metrics.min_correlation,
                "highly_correlated_pairs": corr_metrics.highly_correlated_pairs,
            },
            "contributions": {
                "return_contribution": contrib.return_contribution,
                "risk_contribution": contrib.risk_contribution,
            },
            "benchmark": {
                "tracking_error": self.calculate_tracking_error(),
                "information_ratio": self.calculate_information_ratio(),
                "beta": self.calculate_beta(),
                "alpha": self.calculate_alpha(),
            } if self.benchmark_returns is not None else None,
            "weights": self.weights,
        }
