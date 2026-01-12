"""
Analisis de escenarios para stress testing.

Simula eventos historicos extremos (crashes, alta volatilidad, etc.)
para evaluar como se comportaria la estrategia.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict
import numpy as np

from .models import (
    ScenarioConfig,
    ScenarioResult,
    ScenarioType,
    PREDEFINED_SCENARIOS,
)


@dataclass
class ScenarioData:
    """Datos para simular un escenario"""
    returns: np.ndarray
    volatility: float
    duration_days: int


class ScenarioAnalyzer:
    """
    Analizador de escenarios de estres.

    Permite simular como se comportaria una estrategia
    bajo diferentes condiciones de mercado extremas.

    Ejemplo de uso:
        analyzer = ScenarioAnalyzer()
        result = analyzer.run_scenario(
            returns=strategy_returns,
            scenario_type=ScenarioType.COVID_CRASH
        )
    """

    def __init__(self, initial_capital: float = 10000.0):
        """
        Args:
            initial_capital: Capital inicial para simulaciones
        """
        self.initial_capital = initial_capital

    def run_scenario(
        self,
        returns: np.ndarray,
        scenario_type: ScenarioType = ScenarioType.COVID_CRASH,
        config: Optional[ScenarioConfig] = None
    ) -> ScenarioResult:
        """
        Ejecuta un escenario de estres.

        Args:
            returns: Retornos historicos de la estrategia
            scenario_type: Tipo de escenario predefinido
            config: Configuracion personalizada

        Returns:
            ScenarioResult con los resultados
        """
        returns = np.array(returns)

        # Obtener parametros del escenario
        if scenario_type == ScenarioType.CUSTOM and config:
            params = {
                "name": "Custom Scenario",
                "drawdown": config.custom_drawdown,
                "duration_days": config.custom_duration_days,
                "recovery_days": config.custom_recovery_days,
                "volatility_mult": config.custom_volatility_mult,
            }
        else:
            params = PREDEFINED_SCENARIOS.get(scenario_type, PREDEFINED_SCENARIOS[ScenarioType.COVID_CRASH])

        # Generar retornos del escenario
        scenario_returns = self._generate_scenario_returns(returns, params)

        # Calcular metricas
        result = self._analyze_scenario(
            returns=scenario_returns,
            scenario_type=scenario_type,
            scenario_name=params["name"],
            normal_returns=returns,
        )

        return result

    def run_all_scenarios(
        self,
        returns: np.ndarray
    ) -> List[ScenarioResult]:
        """
        Ejecuta todos los escenarios predefinidos.

        Args:
            returns: Retornos historicos de la estrategia

        Returns:
            Lista de ScenarioResult
        """
        results = []

        for scenario_type in ScenarioType:
            if scenario_type == ScenarioType.CUSTOM:
                continue

            result = self.run_scenario(returns, scenario_type)
            results.append(result)

        return results

    def run_custom_scenario(
        self,
        returns: np.ndarray,
        drawdown: float,
        duration_days: int,
        recovery_days: int = 0,
        volatility_mult: float = 1.0,
        name: str = "Custom Scenario"
    ) -> ScenarioResult:
        """
        Ejecuta un escenario personalizado.

        Args:
            returns: Retornos historicos
            drawdown: Drawdown a simular (0-1)
            duration_days: Duracion del drawdown
            recovery_days: Dias para recuperar
            volatility_mult: Multiplicador de volatilidad
            name: Nombre del escenario

        Returns:
            ScenarioResult
        """
        params = {
            "name": name,
            "drawdown": drawdown,
            "duration_days": duration_days,
            "recovery_days": recovery_days,
            "volatility_mult": volatility_mult,
        }

        scenario_returns = self._generate_scenario_returns(returns, params)

        return self._analyze_scenario(
            returns=scenario_returns,
            scenario_type=ScenarioType.CUSTOM,
            scenario_name=name,
            normal_returns=returns,
        )

    def _generate_scenario_returns(
        self,
        base_returns: np.ndarray,
        params: Dict
    ) -> np.ndarray:
        """
        Genera retornos simulados para un escenario.

        Modifica los retornos base para simular el escenario.
        """
        drawdown = params["drawdown"]
        duration_days = params["duration_days"]
        recovery_days = params.get("recovery_days", 0)
        volatility_mult = params.get("volatility_mult", 1.0)

        # Calcular estadisticas base
        base_std = np.std(base_returns)
        base_mean = np.mean(base_returns)

        # Numero total de dias del escenario (para referencia)
        _ = duration_days + (recovery_days if recovery_days != float('inf') else 0)

        # Generar retornos del escenario
        scenario_returns = []

        # Fase de caida
        if duration_days > 0:
            # Calcular retorno diario necesario para el drawdown
            daily_drawdown = 1 - (1 - drawdown) ** (1 / duration_days)

            for _ in range(duration_days):
                # Retorno base negativo + volatilidad aumentada
                base_ret = -daily_drawdown
                noise = np.random.normal(0, base_std * volatility_mult)
                scenario_returns.append(base_ret + noise)

        # Fase de recuperacion
        if recovery_days > 0 and recovery_days != float('inf'):
            # Calcular retorno diario para recuperar
            recovery_needed = 1 / (1 - drawdown) - 1
            daily_recovery = (1 + recovery_needed) ** (1 / recovery_days) - 1

            for _ in range(int(recovery_days)):
                base_ret = daily_recovery
                noise = np.random.normal(0, base_std * volatility_mult * 0.5)
                scenario_returns.append(base_ret + noise)

        # Si no hay recuperacion, continuar con retornos negativos
        elif recovery_days == float('inf'):
            for _ in range(90):  # 3 meses adicionales de bear market
                base_ret = base_mean * -0.5
                noise = np.random.normal(0, base_std * volatility_mult * 0.8)
                scenario_returns.append(base_ret + noise)

        return np.array(scenario_returns)

    def _analyze_scenario(
        self,
        returns: np.ndarray,
        scenario_type: ScenarioType,
        scenario_name: str,
        normal_returns: np.ndarray
    ) -> ScenarioResult:
        """Analiza los resultados de un escenario"""
        # Calcular equity curve
        equity = self._calculate_equity(returns)

        # Metricas basicas
        initial = self.initial_capital
        final = equity[-1] if len(equity) > 0 else initial
        total_return = final - initial
        total_return_pct = (total_return / initial) * 100

        # Drawdown
        max_dd, dd_duration = self._calculate_drawdown_stats(equity)

        # Volatilidad
        vol = np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0
        normal_vol = np.std(normal_returns) * np.sqrt(252) if len(normal_returns) > 0 else 1
        vol_ratio = vol / normal_vol if normal_vol > 0 else 1

        # Peores periodos
        worst_day = np.min(returns) * 100 if len(returns) > 0 else 0
        worst_week = self._calculate_worst_period(returns, 5) * 100

        # Sharpe durante estres
        mean_ret = np.mean(returns) * 252 if len(returns) > 0 else 0
        sharpe = mean_ret / vol if vol > 0 else 0

        # Sobrevivencia
        survived = final > initial * 0.2  # Sobrevive si conserva >20%
        margin_call = np.min(equity) < initial * 0.1  # Margin call si cae <10%

        # Calcular dias de recuperacion (si hubo recuperacion)
        recovery_days = self._calculate_recovery_days(equity)

        return ScenarioResult(
            scenario_type=scenario_type,
            scenario_name=scenario_name,
            initial_equity=initial,
            final_equity=final,
            total_return=total_return,
            total_return_pct=total_return_pct,
            max_drawdown=max_dd * initial,
            max_drawdown_pct=max_dd * 100,
            drawdown_duration_days=dd_duration,
            recovery_days=recovery_days,
            volatility=vol,
            volatility_vs_normal=vol_ratio,
            worst_day_return=worst_day,
            worst_week_return=worst_week,
            sharpe_during_stress=sharpe,
            survived=survived,
            margin_call=margin_call,
        )

    def _calculate_equity(self, returns: np.ndarray) -> np.ndarray:
        """Calcula equity curve"""
        if len(returns) == 0:
            return np.array([self.initial_capital])

        cum_returns = np.cumprod(1 + returns)
        equity = np.concatenate([[self.initial_capital], self.initial_capital * cum_returns])
        return equity

    def _calculate_drawdown_stats(self, equity: np.ndarray) -> tuple:
        """Calcula estadisticas de drawdown"""
        if len(equity) < 2:
            return 0.0, 0

        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak

        max_dd = float(np.max(drawdown))

        # Duracion del max drawdown
        in_drawdown = drawdown > 0
        dd_lengths = []
        current = 0

        for is_dd in in_drawdown:
            if is_dd:
                current += 1
            else:
                if current > 0:
                    dd_lengths.append(current)
                current = 0

        if current > 0:
            dd_lengths.append(current)

        max_duration = max(dd_lengths) if dd_lengths else 0

        return max_dd, max_duration

    def _calculate_worst_period(self, returns: np.ndarray, period: int) -> float:
        """Calcula el peor retorno de N dias"""
        if len(returns) < period:
            return np.sum(returns) if len(returns) > 0 else 0

        rolling_returns = []
        for i in range(len(returns) - period + 1):
            period_return = np.prod(1 + returns[i:i + period]) - 1
            rolling_returns.append(period_return)

        return float(np.min(rolling_returns))

    def _calculate_recovery_days(self, equity: np.ndarray) -> int:
        """Calcula dias hasta recuperar el capital inicial"""
        if len(equity) < 2:
            return 0

        initial = equity[0]

        # Encontrar punto mas bajo
        trough_idx = np.argmin(equity)

        # Buscar recuperacion desde el punto mas bajo
        for i in range(trough_idx, len(equity)):
            if equity[i] >= initial:
                return i - trough_idx

        return -1  # No recupero


class ScenarioComparator:
    """Comparador de resultados de escenarios"""

    @staticmethod
    def rank_by_survival(results: List[ScenarioResult]) -> List[ScenarioResult]:
        """Ordena escenarios por supervivencia (peores primero)"""
        return sorted(results, key=lambda r: (not r.survived, -r.max_drawdown_pct))

    @staticmethod
    def get_worst_case(results: List[ScenarioResult]) -> ScenarioResult:
        """Obtiene el peor escenario"""
        return max(results, key=lambda r: r.max_drawdown_pct)

    @staticmethod
    def get_survival_rate(results: List[ScenarioResult]) -> float:
        """Calcula tasa de supervivencia entre escenarios"""
        if not results:
            return 0.0
        return sum(1 for r in results if r.survived) / len(results)

    @staticmethod
    def generate_summary(results: List[ScenarioResult]) -> Dict:
        """Genera resumen de todos los escenarios"""
        if not results:
            return {}

        survival_rate = ScenarioComparator.get_survival_rate(results)
        worst = ScenarioComparator.get_worst_case(results)

        avg_drawdown = np.mean([r.max_drawdown_pct for r in results])
        max_drawdown = max(r.max_drawdown_pct for r in results)

        margin_calls = sum(1 for r in results if r.margin_call)

        return {
            "total_scenarios": len(results),
            "survival_rate": f"{survival_rate:.0%}",
            "scenarios_survived": sum(1 for r in results if r.survived),
            "margin_calls": margin_calls,
            "worst_scenario": worst.scenario_name,
            "worst_drawdown": f"{worst.max_drawdown_pct:.1f}%",
            "average_drawdown": f"{avg_drawdown:.1f}%",
            "max_drawdown": f"{max_drawdown:.1f}%",
            "risk_assessment": _assess_risk(survival_rate, max_drawdown),
        }


def _assess_risk(survival_rate: float, max_drawdown: float) -> str:
    """Evalua el nivel de riesgo"""
    if survival_rate < 0.5 or max_drawdown > 80:
        return "EXTREME - Estrategia muy vulnerable a eventos de estres"
    elif survival_rate < 0.7 or max_drawdown > 60:
        return "HIGH - Riesgo significativo en escenarios adversos"
    elif survival_rate < 0.85 or max_drawdown > 40:
        return "MEDIUM - Vulnerabilidad moderada a eventos extremos"
    else:
        return "LOW - Estrategia resiliente a la mayoria de escenarios"
