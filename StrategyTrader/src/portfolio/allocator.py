"""
Algoritmos de asset allocation para portfolio multi-activo.

Metodos disponibles:
- Equal Weight: Pesos iguales para todos los activos
- Risk Parity: Pesos basados en contribucion igual al riesgo
- Mean-Variance: Optimizacion de Markowitz
- Min Variance: Portfolio de minima varianza
- Max Sharpe: Maximo ratio de Sharpe
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy import optimize

from .models import AllocationMethod


@dataclass
class AllocationResult:
    """Resultado del calculo de allocation"""
    weights: Dict[str, float]
    method: AllocationMethod
    expected_return: float = 0.0
    expected_volatility: float = 0.0
    sharpe_ratio: float = 0.0
    success: bool = True
    message: str = ""
    iterations: int = 0
    metadata: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "weights": self.weights,
            "method": self.method.value,
            "expected_return": self.expected_return,
            "expected_volatility": self.expected_volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "success": self.success,
            "message": self.message,
            "iterations": self.iterations,
            "metadata": self.metadata,
        }


class PortfolioAllocator:
    """
    Calculador de asset allocation usando diferentes metodos.

    Ejemplo de uso:
        allocator = PortfolioAllocator()
        result = allocator.calculate_weights(
            returns=returns_df,
            method=AllocationMethod.RISK_PARITY
        )
    """

    def __init__(
        self,
        risk_free_rate: float = 0.02,
        min_weight: float = 0.0,
        max_weight: float = 1.0,
    ):
        """
        Args:
            risk_free_rate: Tasa libre de riesgo anualizada
            min_weight: Peso minimo por activo
            max_weight: Peso maximo por activo
        """
        self.risk_free_rate = risk_free_rate
        self.min_weight = min_weight
        self.max_weight = max_weight

    def calculate_weights(
        self,
        returns: np.ndarray,
        symbols: List[str],
        method: AllocationMethod,
        target_return: Optional[float] = None,
        target_weights: Optional[Dict[str, float]] = None,
    ) -> AllocationResult:
        """
        Calcula los pesos optimos segun el metodo especificado.

        Args:
            returns: Array de retornos (n_periods x n_assets)
            symbols: Lista de simbolos en el mismo orden que las columnas
            method: Metodo de allocation
            target_return: Retorno objetivo (para mean-variance)
            target_weights: Pesos personalizados (para custom)

        Returns:
            AllocationResult con los pesos calculados
        """
        if len(symbols) == 0:
            return AllocationResult(
                weights={},
                method=method,
                success=False,
                message="No symbols provided"
            )

        if len(symbols) == 1:
            return AllocationResult(
                weights={symbols[0]: 1.0},
                method=method,
                success=True,
                message="Single asset portfolio"
            )

        # Convertir a numpy array si es necesario
        returns_array = np.array(returns)

        if returns_array.shape[0] < 2:
            return AllocationResult(
                weights={s: 1.0 / len(symbols) for s in symbols},
                method=method,
                success=False,
                message="Insufficient data points"
            )

        # Dispatch al metodo correspondiente
        if method == AllocationMethod.EQUAL_WEIGHT:
            return self._equal_weight(symbols)
        elif method == AllocationMethod.RISK_PARITY:
            return self._risk_parity(returns_array, symbols)
        elif method == AllocationMethod.MEAN_VARIANCE:
            return self._mean_variance(returns_array, symbols, target_return)
        elif method == AllocationMethod.MIN_VARIANCE:
            return self._min_variance(returns_array, symbols)
        elif method == AllocationMethod.MAX_SHARPE:
            return self._max_sharpe(returns_array, symbols)
        elif method == AllocationMethod.CUSTOM:
            return self._custom(symbols, target_weights)
        else:
            return AllocationResult(
                weights={s: 1.0 / len(symbols) for s in symbols},
                method=method,
                success=False,
                message=f"Unknown method: {method}"
            )

    def _equal_weight(self, symbols: List[str]) -> AllocationResult:
        """Portfolio de pesos iguales"""
        n = len(symbols)
        weight = 1.0 / n
        weights = {s: weight for s in symbols}

        return AllocationResult(
            weights=weights,
            method=AllocationMethod.EQUAL_WEIGHT,
            success=True,
            message=f"Equal weight allocation: {weight:.4f} per asset"
        )

    def _risk_parity(
        self,
        returns: np.ndarray,
        symbols: List[str]
    ) -> AllocationResult:
        """
        Risk Parity: Cada activo contribuye igual al riesgo total.

        La contribucion al riesgo de cada activo es:
        RC_i = w_i * (Cov * w)_i / sqrt(w' * Cov * w)

        Objetivo: RC_i = RC_j para todo i, j
        """
        n = len(symbols)

        # Calcular matriz de covarianza
        cov_matrix = np.cov(returns, rowvar=False)

        # Asegurar que la matriz sea definida positiva
        cov_matrix = self._ensure_positive_definite(cov_matrix)

        # Funcion objetivo: minimizar diferencia de contribuciones al riesgo
        def risk_parity_objective(weights: np.ndarray) -> float:
            weights = weights / weights.sum()  # Normalizar
            portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)

            if portfolio_vol < 1e-10:
                return 1e10

            # Contribuciones marginales al riesgo
            marginal_contrib = cov_matrix @ weights
            risk_contrib = weights * marginal_contrib / portfolio_vol

            # Objetivo: minimizar desviacion de contribuciones iguales
            target_contrib = portfolio_vol / n
            return np.sum((risk_contrib - target_contrib) ** 2)

        # Optimizar
        initial_weights = np.ones(n) / n
        bounds = [(self.min_weight, self.max_weight) for _ in range(n)]
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

        result = optimize.minimize(
            risk_parity_objective,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-10}
        )

        if result.success:
            weights = result.x / result.x.sum()
            weights_dict = {s: float(w) for s, w in zip(symbols, weights)}

            # Calcular metricas esperadas
            mean_returns = np.mean(returns, axis=0)
            exp_return = float(np.dot(weights, mean_returns))
            exp_vol = float(np.sqrt(weights @ cov_matrix @ weights))
            sharpe = (exp_return * 252 - self.risk_free_rate) / (exp_vol * np.sqrt(252)) if exp_vol > 0 else 0

            return AllocationResult(
                weights=weights_dict,
                method=AllocationMethod.RISK_PARITY,
                expected_return=exp_return * 252,  # Anualizado
                expected_volatility=exp_vol * np.sqrt(252),
                sharpe_ratio=sharpe,
                success=True,
                message="Risk parity optimization successful",
                iterations=result.nit,
            )
        else:
            # Fallback a equal weight
            return AllocationResult(
                weights={s: 1.0 / n for s in symbols},
                method=AllocationMethod.RISK_PARITY,
                success=False,
                message=f"Optimization failed: {result.message}",
                iterations=result.nit,
            )

    def _mean_variance(
        self,
        returns: np.ndarray,
        symbols: List[str],
        target_return: Optional[float] = None
    ) -> AllocationResult:
        """
        Mean-Variance (Markowitz): Minimizar varianza para un retorno objetivo.

        Si no se especifica target_return, se usa el retorno promedio de los activos.
        """
        n = len(symbols)

        # Calcular estadisticas
        mean_returns = np.mean(returns, axis=0) * 252  # Anualizado
        cov_matrix = np.cov(returns, rowvar=False) * 252  # Anualizada
        cov_matrix = self._ensure_positive_definite(cov_matrix)

        # Si no hay target, usar promedio
        if target_return is None:
            target_return = np.mean(mean_returns)

        # Funcion objetivo: minimizar varianza
        def variance_objective(weights: np.ndarray) -> float:
            return float(weights @ cov_matrix @ weights)

        # Restricciones
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},  # Sum to 1
            {"type": "eq", "fun": lambda w: np.dot(w, mean_returns) - target_return},  # Target return
        ]

        # Optimizar
        initial_weights = np.ones(n) / n
        bounds = [(self.min_weight, self.max_weight) for _ in range(n)]

        result = optimize.minimize(
            variance_objective,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-10}
        )

        if result.success:
            weights = result.x / result.x.sum()
            weights_dict = {s: float(w) for s, w in zip(symbols, weights)}

            exp_return = float(np.dot(weights, mean_returns))
            exp_vol = float(np.sqrt(weights @ cov_matrix @ weights))
            sharpe = (exp_return - self.risk_free_rate) / exp_vol if exp_vol > 0 else 0

            return AllocationResult(
                weights=weights_dict,
                method=AllocationMethod.MEAN_VARIANCE,
                expected_return=exp_return,
                expected_volatility=exp_vol,
                sharpe_ratio=sharpe,
                success=True,
                message=f"Mean-variance optimization successful (target return: {target_return:.4f})",
                iterations=result.nit,
                metadata={"target_return": float(target_return) if target_return else 0.0}
            )
        else:
            # Intentar sin restriccion de retorno
            return self._min_variance(returns, symbols)

    def _min_variance(
        self,
        returns: np.ndarray,
        symbols: List[str]
    ) -> AllocationResult:
        """
        Minimum Variance Portfolio: Portfolio de menor riesgo posible.
        """
        n = len(symbols)

        # Calcular estadisticas
        mean_returns = np.mean(returns, axis=0) * 252
        cov_matrix = np.cov(returns, rowvar=False) * 252
        cov_matrix = self._ensure_positive_definite(cov_matrix)

        # Funcion objetivo: minimizar varianza
        def variance_objective(weights: np.ndarray) -> float:
            return float(weights @ cov_matrix @ weights)

        # Restricciones: solo que sumen 1
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

        # Optimizar
        initial_weights = np.ones(n) / n
        bounds = [(self.min_weight, self.max_weight) for _ in range(n)]

        result = optimize.minimize(
            variance_objective,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-10}
        )

        if result.success:
            weights = result.x / result.x.sum()
            weights_dict = {s: float(w) for s, w in zip(symbols, weights)}

            exp_return = float(np.dot(weights, mean_returns))
            exp_vol = float(np.sqrt(weights @ cov_matrix @ weights))
            sharpe = (exp_return - self.risk_free_rate) / exp_vol if exp_vol > 0 else 0

            return AllocationResult(
                weights=weights_dict,
                method=AllocationMethod.MIN_VARIANCE,
                expected_return=exp_return,
                expected_volatility=exp_vol,
                sharpe_ratio=sharpe,
                success=True,
                message="Minimum variance optimization successful",
                iterations=result.nit,
            )
        else:
            return AllocationResult(
                weights={s: 1.0 / n for s in symbols},
                method=AllocationMethod.MIN_VARIANCE,
                success=False,
                message=f"Optimization failed: {result.message}",
                iterations=result.nit,
            )

    def _max_sharpe(
        self,
        returns: np.ndarray,
        symbols: List[str]
    ) -> AllocationResult:
        """
        Maximum Sharpe Ratio Portfolio: Maximizar el ratio de Sharpe.
        """
        n = len(symbols)

        # Calcular estadisticas
        mean_returns = np.mean(returns, axis=0) * 252
        cov_matrix = np.cov(returns, rowvar=False) * 252
        cov_matrix = self._ensure_positive_definite(cov_matrix)

        # Funcion objetivo: minimizar -Sharpe (equivalente a maximizar Sharpe)
        def neg_sharpe_objective(weights: np.ndarray) -> float:
            weights = weights / weights.sum()
            portfolio_return = np.dot(weights, mean_returns)
            portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)

            if portfolio_vol < 1e-10:
                return 1e10

            sharpe = (portfolio_return - self.risk_free_rate) / portfolio_vol
            return -sharpe

        # Restricciones
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

        # Optimizar
        initial_weights = np.ones(n) / n
        bounds = [(self.min_weight, self.max_weight) for _ in range(n)]

        result = optimize.minimize(
            neg_sharpe_objective,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-10}
        )

        if result.success:
            weights = result.x / result.x.sum()
            weights_dict = {s: float(w) for s, w in zip(symbols, weights)}

            exp_return = float(np.dot(weights, mean_returns))
            exp_vol = float(np.sqrt(weights @ cov_matrix @ weights))
            sharpe = (exp_return - self.risk_free_rate) / exp_vol if exp_vol > 0 else 0

            return AllocationResult(
                weights=weights_dict,
                method=AllocationMethod.MAX_SHARPE,
                expected_return=exp_return,
                expected_volatility=exp_vol,
                sharpe_ratio=sharpe,
                success=True,
                message="Maximum Sharpe optimization successful",
                iterations=result.nit,
            )
        else:
            return AllocationResult(
                weights={s: 1.0 / n for s in symbols},
                method=AllocationMethod.MAX_SHARPE,
                success=False,
                message=f"Optimization failed: {result.message}",
                iterations=result.nit,
            )

    def _custom(
        self,
        symbols: List[str],
        target_weights: Optional[Dict[str, float]] = None
    ) -> AllocationResult:
        """Pesos personalizados"""
        if target_weights is None:
            return AllocationResult(
                weights={s: 1.0 / len(symbols) for s in symbols},
                method=AllocationMethod.CUSTOM,
                success=False,
                message="No target weights provided, using equal weight"
            )

        # Normalizar pesos
        total = sum(target_weights.values())
        if total <= 0:
            return AllocationResult(
                weights={s: 1.0 / len(symbols) for s in symbols},
                method=AllocationMethod.CUSTOM,
                success=False,
                message="Invalid target weights"
            )

        weights = {s: target_weights.get(s, 0.0) / total for s in symbols}

        return AllocationResult(
            weights=weights,
            method=AllocationMethod.CUSTOM,
            success=True,
            message="Custom weights applied"
        )

    def _ensure_positive_definite(self, matrix: np.ndarray) -> np.ndarray:
        """Asegura que la matriz sea definida positiva"""
        # Eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)

        # Reemplazar eigenvalues negativos con un valor pequeno
        min_eigenvalue = 1e-10
        eigenvalues = np.maximum(eigenvalues, min_eigenvalue)

        # Reconstruir matriz
        return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    def get_efficient_frontier(
        self,
        returns: np.ndarray,
        symbols: List[str],
        n_points: int = 50
    ) -> List[Tuple[float, float, Dict[str, float]]]:
        """
        Calcula la frontera eficiente de Markowitz.

        Returns:
            Lista de tuplas (retorno, volatilidad, pesos)
        """
        # Calcular estadisticas
        mean_returns = np.mean(returns, axis=0) * 252
        min_return = np.min(mean_returns)
        max_return = np.max(mean_returns)

        frontier = []

        for target_return in np.linspace(min_return, max_return, n_points):
            result = self._mean_variance(returns, symbols, target_return)
            if result.success:
                frontier.append((
                    result.expected_return,
                    result.expected_volatility,
                    result.weights
                ))

        return frontier


class PortfolioAllocatorFactory:
    """Factory para crear allocators preconfigurados"""

    @staticmethod
    def create_conservative() -> PortfolioAllocator:
        """Allocator conservador con limites estrictos"""
        return PortfolioAllocator(
            risk_free_rate=0.02,
            min_weight=0.05,  # Minimo 5%
            max_weight=0.30,  # Maximo 30%
        )

    @staticmethod
    def create_moderate() -> PortfolioAllocator:
        """Allocator moderado"""
        return PortfolioAllocator(
            risk_free_rate=0.02,
            min_weight=0.02,
            max_weight=0.50,
        )

    @staticmethod
    def create_aggressive() -> PortfolioAllocator:
        """Allocator agresivo sin restricciones de peso"""
        return PortfolioAllocator(
            risk_free_rate=0.02,
            min_weight=0.0,
            max_weight=1.0,
        )
