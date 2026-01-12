"""
Simulacion Monte Carlo para stress testing.

Genera multiples escenarios alterando el orden/distribucion de retornos
para evaluar la robustez de una estrategia.
"""

from dataclasses import dataclass
from typing import List, Optional, Callable
import numpy as np

from .models import MonteCarloConfig, MonteCarloResult


class MonteCarloSimulator:
    """
    Simulador Monte Carlo para estrategias de trading.

    Metodos disponibles:
    - Shuffle: Barajar orden de retornos
    - Bootstrap: Muestreo con reemplazo
    - Block Bootstrap: Muestreo por bloques (preserva autocorrelacion)

    Ejemplo de uso:
        simulator = MonteCarloSimulator(config)
        result = simulator.run(returns)
        print(f"Probabilidad de ganancia: {result.prob_profit:.1%}")
    """

    def __init__(self, config: Optional[MonteCarloConfig] = None):
        """
        Args:
            config: Configuracion de la simulacion
        """
        self.config = config or MonteCarloConfig()

        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)

    def run(
        self,
        returns: np.ndarray,
        initial_capital: float = 10000.0,
        risk_free_rate: float = 0.02,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> MonteCarloResult:
        """
        Ejecuta la simulacion Monte Carlo.

        Args:
            returns: Array de retornos historicos
            initial_capital: Capital inicial
            risk_free_rate: Tasa libre de riesgo anualizada
            progress_callback: Callback de progreso (current, total)

        Returns:
            MonteCarloResult con estadisticas de la simulacion
        """
        returns = np.array(returns)
        n_sims = self.config.n_simulations

        # Arrays para almacenar resultados
        final_returns = np.zeros(n_sims)
        max_drawdowns = np.zeros(n_sims)
        sharpe_ratios = np.zeros(n_sims)

        for i in range(n_sims):
            if progress_callback:
                progress_callback(i + 1, n_sims)

            # Generar secuencia de retornos alterada
            if self.config.block_size:
                sim_returns = self._block_bootstrap(returns, self.config.block_size)
            else:
                sim_returns = self._shuffle_returns(returns)

            # Calcular equity curve
            equity = self._calculate_equity(sim_returns, initial_capital)

            # Calcular metricas
            final_returns[i] = (equity[-1] / initial_capital - 1) * 100  # Porcentaje
            max_drawdowns[i] = self._calculate_max_drawdown(equity) * 100
            sharpe_ratios[i] = self._calculate_sharpe(sim_returns, risk_free_rate)

        # Construir resultado
        result = self._build_result(final_returns, max_drawdowns, sharpe_ratios)

        return result

    def run_with_strategy(
        self,
        strategy_func: Callable[[np.ndarray], np.ndarray],
        price_data: np.ndarray,
        n_simulations: Optional[int] = None,
        initial_capital: float = 10000.0,
    ) -> MonteCarloResult:
        """
        Ejecuta Monte Carlo re-ejecutando la estrategia con datos alterados.

        Args:
            strategy_func: Funcion que recibe precios y retorna retornos
            price_data: Datos de precios historicos
            n_simulations: Numero de simulaciones (default: config)
            initial_capital: Capital inicial

        Returns:
            MonteCarloResult
        """
        n_sims = n_simulations or self.config.n_simulations

        final_returns = np.zeros(n_sims)
        max_drawdowns = np.zeros(n_sims)
        sharpe_ratios = np.zeros(n_sims)

        # Calcular retornos de precios
        price_returns = np.diff(price_data) / price_data[:-1]

        for i in range(n_sims):
            # Generar precios alterados
            shuffled_returns = self._shuffle_returns(price_returns)
            simulated_prices = self._returns_to_prices(shuffled_returns, price_data[0])

            # Ejecutar estrategia con precios simulados
            strategy_returns = strategy_func(simulated_prices)

            # Calcular metricas
            equity = self._calculate_equity(strategy_returns, initial_capital)
            final_returns[i] = (equity[-1] / initial_capital - 1) * 100
            max_drawdowns[i] = self._calculate_max_drawdown(equity) * 100
            sharpe_ratios[i] = self._calculate_sharpe(strategy_returns, 0.02)

        return self._build_result(final_returns, max_drawdowns, sharpe_ratios)

    def _shuffle_returns(self, returns: np.ndarray) -> np.ndarray:
        """Barajar retornos aleatoriamente"""
        shuffled = returns.copy()
        np.random.shuffle(shuffled)
        return shuffled

    def _block_bootstrap(self, returns: np.ndarray, block_size: int) -> np.ndarray:
        """
        Bootstrap por bloques para preservar autocorrelacion.

        Args:
            returns: Retornos originales
            block_size: Tamano de cada bloque

        Returns:
            Array de retornos remuestreados
        """
        n = len(returns)
        n_blocks = int(np.ceil(n / block_size))

        # Crear indices de bloques
        block_starts = np.random.randint(0, n - block_size + 1, n_blocks)

        # Construir secuencia
        result = []
        for start in block_starts:
            result.extend(returns[start:start + block_size])

        return np.array(result[:n])

    def _calculate_equity(
        self,
        returns: np.ndarray,
        initial_capital: float
    ) -> np.ndarray:
        """Calcula equity curve desde retornos"""
        cum_returns = np.cumprod(1 + returns)
        return initial_capital * cum_returns

    def _calculate_max_drawdown(self, equity: np.ndarray) -> float:
        """Calcula maximo drawdown de una equity curve"""
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        return float(np.max(drawdown))

    def _calculate_sharpe(
        self,
        returns: np.ndarray,
        risk_free_rate: float = 0.02
    ) -> float:
        """Calcula Sharpe ratio anualizado"""
        if len(returns) < 2:
            return 0.0

        mean_return = np.mean(returns) * 252
        std_return = np.std(returns) * np.sqrt(252)

        if std_return < 1e-10:
            return 0.0

        return (mean_return - risk_free_rate) / std_return

    def _returns_to_prices(
        self,
        returns: np.ndarray,
        initial_price: float
    ) -> np.ndarray:
        """Convierte retornos a precios"""
        cum_returns = np.cumprod(1 + returns)
        prices = np.concatenate([[initial_price], initial_price * cum_returns])
        return prices

    def _build_result(
        self,
        final_returns: np.ndarray,
        max_drawdowns: np.ndarray,
        sharpe_ratios: np.ndarray
    ) -> MonteCarloResult:
        """Construye el resultado desde los arrays de simulacion"""
        result = MonteCarloResult(
            n_simulations=len(final_returns),

            # Retornos
            mean_return=float(np.mean(final_returns)),
            median_return=float(np.median(final_returns)),
            std_return=float(np.std(final_returns)),
            min_return=float(np.min(final_returns)),
            max_return=float(np.max(final_returns)),

            # Percentiles
            percentile_5=float(np.percentile(final_returns, 5)),
            percentile_25=float(np.percentile(final_returns, 25)),
            percentile_75=float(np.percentile(final_returns, 75)),
            percentile_95=float(np.percentile(final_returns, 95)),

            # Probabilidades
            prob_profit=float(np.mean(final_returns > 0)),
            prob_loss_10pct=float(np.mean(final_returns < -10)),
            prob_loss_20pct=float(np.mean(final_returns < -20)),
            prob_double=float(np.mean(final_returns > 100)),

            # Drawdowns
            mean_max_drawdown=float(np.mean(max_drawdowns)),
            median_max_drawdown=float(np.median(max_drawdowns)),
            worst_max_drawdown=float(np.max(max_drawdowns)),

            # Sharpe
            mean_sharpe=float(np.mean(sharpe_ratios)),
            median_sharpe=float(np.median(sharpe_ratios)),
            std_sharpe=float(np.std(sharpe_ratios)),

            # Raw data
            all_final_returns=final_returns.tolist(),
            all_max_drawdowns=max_drawdowns.tolist(),
            all_sharpe_ratios=sharpe_ratios.tolist(),

            # VaR y CVaR
            var_95=float(np.percentile(final_returns, 5)),
            var_99=float(np.percentile(final_returns, 1)),
            cvar_95=float(np.mean(final_returns[final_returns <= np.percentile(final_returns, 5)])),
            cvar_99=float(np.mean(final_returns[final_returns <= np.percentile(final_returns, 1)])),
        )

        return result


class MonteCarloAnalyzer:
    """Analisis adicional de resultados Monte Carlo"""

    @staticmethod
    def get_confidence_interval(
        result: MonteCarloResult,
        confidence: float = 0.95
    ) -> tuple:
        """
        Obtiene intervalo de confianza para el retorno.

        Args:
            result: Resultado de Monte Carlo
            confidence: Nivel de confianza (0-1)

        Returns:
            Tupla (lower, upper)
        """
        alpha = (1 - confidence) / 2
        returns = np.array(result.all_final_returns)

        lower = np.percentile(returns, alpha * 100)
        upper = np.percentile(returns, (1 - alpha) * 100)

        return (float(lower), float(upper))

    @staticmethod
    def get_risk_of_ruin(
        result: MonteCarloResult,
        ruin_threshold: float = -50.0
    ) -> float:
        """
        Calcula probabilidad de ruina (perder X% del capital).

        Args:
            result: Resultado de Monte Carlo
            ruin_threshold: Umbral de ruina en porcentaje (ej: -50 = perder 50%)

        Returns:
            Probabilidad de ruina (0-1)
        """
        returns = np.array(result.all_final_returns)
        return float(np.mean(returns <= ruin_threshold))

    @staticmethod
    def get_expected_shortfall(
        result: MonteCarloResult,
        percentile: float = 5.0
    ) -> float:
        """
        Calcula Expected Shortfall (CVaR).

        Args:
            result: Resultado de Monte Carlo
            percentile: Percentil de corte

        Returns:
            Expected shortfall
        """
        returns = np.array(result.all_final_returns)
        threshold = np.percentile(returns, percentile)
        return float(np.mean(returns[returns <= threshold]))

    @staticmethod
    def get_probability_target(
        result: MonteCarloResult,
        target_return: float
    ) -> float:
        """
        Calcula probabilidad de alcanzar un retorno objetivo.

        Args:
            result: Resultado de Monte Carlo
            target_return: Retorno objetivo en porcentaje

        Returns:
            Probabilidad de alcanzar el objetivo
        """
        returns = np.array(result.all_final_returns)
        return float(np.mean(returns >= target_return))

    @staticmethod
    def generate_summary(result: MonteCarloResult) -> dict:
        """Genera resumen ejecutivo"""
        # Clasificar riesgo
        if result.prob_loss_20pct > 0.20:
            risk_level = "EXTREME"
        elif result.prob_loss_10pct > 0.20:
            risk_level = "HIGH"
        elif result.prob_loss_10pct > 0.10:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        # Clasificar robustez
        sharpe_consistency = result.std_sharpe / abs(result.mean_sharpe) if result.mean_sharpe != 0 else float('inf')

        if result.prob_profit > 0.80 and sharpe_consistency < 0.5:
            robustness = "ROBUST"
        elif result.prob_profit > 0.60:
            robustness = "MODERATE"
        else:
            robustness = "FRAGILE"

        return {
            "risk_level": risk_level,
            "robustness": robustness,
            "expected_return": f"{result.mean_return:.1f}%",
            "expected_range": f"{result.percentile_5:.1f}% to {result.percentile_95:.1f}%",
            "probability_of_profit": f"{result.prob_profit:.1%}",
            "worst_case_drawdown": f"{result.worst_max_drawdown:.1f}%",
            "key_insight": _generate_insight(result),
        }


def _generate_insight(result: MonteCarloResult) -> str:
    """Genera insight principal del resultado"""
    if result.prob_profit < 0.50:
        return "ALERTA: La estrategia tiene mas probabilidad de perder que de ganar"
    elif result.prob_loss_20pct > 0.10:
        return f"CUIDADO: {result.prob_loss_20pct:.0%} de probabilidad de perder >20%"
    elif result.mean_sharpe < 0.5:
        return "El ratio riesgo/retorno es bajo. Considerar optimizar la estrategia"
    elif result.std_sharpe > result.mean_sharpe:
        return "Alta variabilidad en performance. Los resultados dependen mucho de la suerte"
    else:
        return f"Estrategia consistente con {result.prob_profit:.0%} prob. de ganancia"
