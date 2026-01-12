"""
Orquestador principal de stress testing.

Integra todos los componentes de stress testing:
- Monte Carlo simulation
- Scenario analysis
- Sensitivity analysis
"""

import time
from typing import Dict, List, Optional, Callable, Any
import numpy as np

from .models import (
    StressTestConfig,
    StressTestReport,
    MonteCarloConfig,
    ScenarioConfig,
    SensitivityConfig,
    ScenarioType,
)
from .monte_carlo import MonteCarloSimulator, MonteCarloAnalyzer
from .scenario_analysis import ScenarioAnalyzer, ScenarioComparator
from .sensitivity import (
    SensitivityAnalyzer,
    RobustnessAnalyzer,
    ParameterRange,
)


class StressTester:
    """
    Orquestador principal de stress testing.

    Combina Monte Carlo, analisis de escenarios y sensibilidad
    para evaluar la robustez de una estrategia.

    Ejemplo de uso:
        tester = StressTester()
        report = tester.run_full_analysis(
            returns=strategy_returns,
            strategy_func=run_strategy,
            parameters={'sma_period': 20, 'rsi_threshold': 30}
        )
        tester.print_report(report)
    """

    def __init__(self, config: Optional[StressTestConfig] = None):
        """
        Args:
            config: Configuracion de stress testing
        """
        self.config = config or StressTestConfig()

        # Inicializar componentes
        self.monte_carlo = MonteCarloSimulator(self.config.monte_carlo)
        self.scenario_analyzer = ScenarioAnalyzer(self.config.initial_capital)
        self.sensitivity_analyzer = SensitivityAnalyzer(self.config.sensitivity)

    def run_full_analysis(
        self,
        returns: np.ndarray,
        strategy_func: Optional[Callable[[Dict[str, Any]], Dict[str, float]]] = None,
        parameters: Optional[Dict[str, float]] = None,
        include_monte_carlo: bool = True,
        include_scenarios: bool = True,
        include_sensitivity: bool = True,
    ) -> StressTestReport:
        """
        Ejecuta analisis completo de stress testing.

        Args:
            returns: Retornos historicos de la estrategia
            strategy_func: Funcion de estrategia (para sensibilidad)
            parameters: Parametros base (para sensibilidad)
            include_monte_carlo: Incluir Monte Carlo
            include_scenarios: Incluir escenarios
            include_sensitivity: Incluir sensibilidad

        Returns:
            StressTestReport completo
        """
        start_time = time.time()
        returns = np.array(returns)

        report = StressTestReport(config=self.config)

        # 1. Monte Carlo
        if include_monte_carlo:
            report.monte_carlo = self.monte_carlo.run(
                returns=returns,
                initial_capital=self.config.initial_capital,
                risk_free_rate=self.config.risk_free_rate,
            )

        # 2. Analisis de escenarios
        if include_scenarios:
            report.scenarios = self.scenario_analyzer.run_all_scenarios(returns)

        # 3. Analisis de sensibilidad
        if include_sensitivity and strategy_func and parameters:
            param_ranges = {
                name: ParameterRange(
                    name=name,
                    base_value=value,
                    min_value=value * (1 - self.config.sensitivity.variation_pct),
                    max_value=value * (1 + self.config.sensitivity.variation_pct),
                    n_steps=self.config.sensitivity.n_steps,
                )
                for name, value in parameters.items()
            }

            report.sensitivity = self.sensitivity_analyzer.analyze_multiple_parameters(
                strategy_func=strategy_func,
                parameters=param_ranges,
                metric=self.config.sensitivity.metric_to_optimize,
            )

        # 4. Calcular score de robustez y generar resumen
        report = self._finalize_report(report)

        report.execution_time = time.time() - start_time

        return report

    def run_monte_carlo(
        self,
        returns: np.ndarray,
        n_simulations: Optional[int] = None
    ):
        """
        Ejecuta solo Monte Carlo.

        Args:
            returns: Retornos historicos
            n_simulations: Numero de simulaciones

        Returns:
            MonteCarloResult
        """
        if n_simulations:
            self.monte_carlo.config.n_simulations = n_simulations

        return self.monte_carlo.run(
            returns=returns,
            initial_capital=self.config.initial_capital,
            risk_free_rate=self.config.risk_free_rate,
        )

    def run_scenario(
        self,
        returns: np.ndarray,
        scenario_type: ScenarioType = ScenarioType.COVID_CRASH
    ):
        """
        Ejecuta un escenario especifico.

        Args:
            returns: Retornos historicos
            scenario_type: Tipo de escenario

        Returns:
            ScenarioResult
        """
        return self.scenario_analyzer.run_scenario(
            returns=returns,
            scenario_type=scenario_type,
        )

    def run_all_scenarios(self, returns: np.ndarray):
        """
        Ejecuta todos los escenarios predefinidos.

        Args:
            returns: Retornos historicos

        Returns:
            Lista de ScenarioResult
        """
        return self.scenario_analyzer.run_all_scenarios(returns)

    def run_sensitivity(
        self,
        strategy_func: Callable[[Dict[str, Any]], Dict[str, float]],
        parameters: Dict[str, float],
        variation_pct: float = 0.20
    ):
        """
        Ejecuta analisis de sensibilidad.

        Args:
            strategy_func: Funcion de estrategia
            parameters: Parametros a analizar
            variation_pct: Variacion porcentual

        Returns:
            Lista de SensitivityResult
        """
        param_ranges = {
            name: ParameterRange(
                name=name,
                base_value=value,
                min_value=value * (1 - variation_pct),
                max_value=value * (1 + variation_pct),
                n_steps=self.config.sensitivity.n_steps,
            )
            for name, value in parameters.items()
        }

        return self.sensitivity_analyzer.analyze_multiple_parameters(
            strategy_func=strategy_func,
            parameters=param_ranges,
        )

    def _finalize_report(self, report: StressTestReport) -> StressTestReport:
        """Calcula scores finales y genera resumen"""
        scores = []
        risks = []
        recommendations = []

        # Score de Monte Carlo
        if report.monte_carlo:
            mc = report.monte_carlo

            # Score basado en prob de ganancia y Sharpe
            mc_score = (mc.prob_profit * 50) + (min(mc.mean_sharpe, 2) / 2 * 50)
            scores.append(mc_score)

            if mc.prob_loss_20pct > 0.10:
                risks.append(f"Alto riesgo de perdida >20% ({mc.prob_loss_20pct:.0%} prob.)")

            if mc.mean_sharpe < 0.5:
                recommendations.append("Considerar mejorar ratio riesgo/retorno")

        # Score de escenarios
        if report.scenarios:
            survival_rate = ScenarioComparator.get_survival_rate(report.scenarios)
            scenario_score = survival_rate * 100
            scores.append(scenario_score)

            if survival_rate < 0.70:
                risks.append(f"Baja tasa de supervivencia ({survival_rate:.0%}) en escenarios de estres")

            worst = ScenarioComparator.get_worst_case(report.scenarios)
            if worst.max_drawdown_pct > 50:
                risks.append(f"Drawdown extremo en {worst.scenario_name}: {worst.max_drawdown_pct:.0f}%")

        # Score de sensibilidad
        if report.sensitivity:
            sens_score = RobustnessAnalyzer.calculate_robustness_score(report.sensitivity)
            scores.append(sens_score)

            fragile = RobustnessAnalyzer.identify_fragile_parameters(report.sensitivity)
            if fragile:
                risks.append(f"Parametros fragiles: {', '.join(fragile)}")
                recommendations.append(f"Revisar configuracion de: {', '.join(fragile)}")

        # Score global
        if scores:
            report.overall_robustness_score = float(np.mean(scores))
        else:
            report.overall_robustness_score = 0.0

        # Rating de riesgo
        if report.overall_robustness_score >= 80:
            report.risk_rating = "LOW"
        elif report.overall_robustness_score >= 60:
            report.risk_rating = "MEDIUM"
        elif report.overall_robustness_score >= 40:
            report.risk_rating = "HIGH"
        else:
            report.risk_rating = "EXTREME"

        report.key_risks = risks
        report.recommendations = recommendations

        return report

    def print_report(self, report: StressTestReport) -> None:
        """Imprime reporte de stress testing"""
        print("\n" + "=" * 70)
        print("                    STRESS TESTING REPORT")
        print("=" * 70)

        # Resumen ejecutivo
        print(f"\n{'RESUMEN EJECUTIVO':^70}")
        print("-" * 70)
        print(f"  Score de Robustez: {report.overall_robustness_score:.1f}/100")
        print(f"  Rating de Riesgo:  {report.risk_rating}")
        print(f"  Tiempo de ejecucion: {report.execution_time:.2f}s")

        # Monte Carlo
        if report.monte_carlo:
            mc = report.monte_carlo
            print(f"\n{'MONTE CARLO SIMULATION':^70}")
            print("-" * 70)
            print(f"  Simulaciones: {mc.n_simulations}")
            print(f"  Retorno esperado: {mc.mean_return:+.1f}%")
            print(f"  Rango (5-95%): {mc.percentile_5:+.1f}% a {mc.percentile_95:+.1f}%")
            print(f"  Prob. ganancia: {mc.prob_profit:.1%}")
            print(f"  Prob. perdida >20%: {mc.prob_loss_20pct:.1%}")
            print(f"  Max DD promedio: {mc.mean_max_drawdown:.1f}%")
            print(f"  Sharpe promedio: {mc.mean_sharpe:.2f}")

        # Escenarios
        if report.scenarios:
            print(f"\n{'ANALISIS DE ESCENARIOS':^70}")
            print("-" * 70)

            survival = ScenarioComparator.get_survival_rate(report.scenarios)
            print(f"  Tasa de supervivencia: {survival:.0%}")
            print(f"  Escenarios evaluados: {len(report.scenarios)}")
            print()

            for s in sorted(report.scenarios, key=lambda x: -x.max_drawdown_pct)[:5]:
                status = "OK" if s.survived else "FAIL"
                print(f"  [{status}] {s.scenario_name[:30]:30} | DD: {s.max_drawdown_pct:5.1f}% | Ret: {s.total_return_pct:+6.1f}%")

        # Sensibilidad
        if report.sensitivity:
            print(f"\n{'ANALISIS DE SENSIBILIDAD':^70}")
            print("-" * 70)

            sens_score = RobustnessAnalyzer.calculate_robustness_score(report.sensitivity)
            fragile = RobustnessAnalyzer.identify_fragile_parameters(report.sensitivity)

            print(f"  Score de robustez: {sens_score:.1f}/100")
            print(f"  Parametros fragiles: {len(fragile)}/{len(report.sensitivity)}")
            print()

            for r in report.sensitivity:
                status = "ROBUST" if r.is_robust else "FRAGILE"
                print(f"  [{status:7}] {r.parameter_name:20} | Base: {r.base_value:8.2f} | Optimo: {r.optimal_value:8.2f}")

        # Riesgos y recomendaciones
        if report.key_risks:
            print(f"\n{'RIESGOS IDENTIFICADOS':^70}")
            print("-" * 70)
            for risk in report.key_risks:
                print(f"  ! {risk}")

        if report.recommendations:
            print(f"\n{'RECOMENDACIONES':^70}")
            print("-" * 70)
            for rec in report.recommendations:
                print(f"  > {rec}")

        print("\n" + "=" * 70)


class StressTesterFactory:
    """Factory para crear stress testers preconfigurados"""

    @staticmethod
    def create_quick(initial_capital: float = 10000.0) -> StressTester:
        """Stress tester rapido (menos simulaciones)"""
        config = StressTestConfig(
            initial_capital=initial_capital,
            monte_carlo=MonteCarloConfig(n_simulations=200),
            sensitivity=SensitivityConfig(n_steps=3),
        )
        return StressTester(config)

    @staticmethod
    def create_standard(initial_capital: float = 10000.0) -> StressTester:
        """Stress tester estandar"""
        config = StressTestConfig(
            initial_capital=initial_capital,
            monte_carlo=MonteCarloConfig(n_simulations=1000),
            sensitivity=SensitivityConfig(n_steps=5),
        )
        return StressTester(config)

    @staticmethod
    def create_thorough(initial_capital: float = 10000.0) -> StressTester:
        """Stress tester exhaustivo"""
        config = StressTestConfig(
            initial_capital=initial_capital,
            monte_carlo=MonteCarloConfig(n_simulations=5000),
            sensitivity=SensitivityConfig(n_steps=10, variation_pct=0.30),
        )
        return StressTester(config)
