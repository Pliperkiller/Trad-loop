"""
Modulo de stress testing para estrategias de trading.

Proporciona herramientas para:
- Monte Carlo simulation: Evaluar distribucion de resultados posibles
- Scenario analysis: Simular eventos historicos extremos
- Sensitivity analysis: Medir impacto de cambios en parametros
"""

from .models import (
    StressTestType,
    ScenarioType,
    MonteCarloConfig,
    ScenarioConfig,
    SensitivityConfig,
    StressTestConfig,
    MonteCarloResult,
    ScenarioResult,
    SensitivityResult,
    StressTestReport,
    PREDEFINED_SCENARIOS,
)
from .monte_carlo import MonteCarloSimulator, MonteCarloAnalyzer
from .scenario_analysis import ScenarioAnalyzer, ScenarioComparator
from .sensitivity import (
    SensitivityAnalyzer,
    RobustnessAnalyzer,
    ParameterRange,
)
from .stress_tester import StressTester, StressTesterFactory

__all__ = [
    # Enums
    "StressTestType",
    "ScenarioType",
    # Configs
    "MonteCarloConfig",
    "ScenarioConfig",
    "SensitivityConfig",
    "StressTestConfig",
    # Results
    "MonteCarloResult",
    "ScenarioResult",
    "SensitivityResult",
    "StressTestReport",
    # Constants
    "PREDEFINED_SCENARIOS",
    # Monte Carlo
    "MonteCarloSimulator",
    "MonteCarloAnalyzer",
    # Scenarios
    "ScenarioAnalyzer",
    "ScenarioComparator",
    # Sensitivity
    "SensitivityAnalyzer",
    "RobustnessAnalyzer",
    "ParameterRange",
    # Main
    "StressTester",
    "StressTesterFactory",
]
