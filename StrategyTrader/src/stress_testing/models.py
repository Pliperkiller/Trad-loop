"""
Modelos de datos para el modulo de stress testing.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, List, Any
import numpy as np


class StressTestType(Enum):
    """Tipos de stress test"""
    MONTE_CARLO = "monte_carlo"
    SCENARIO = "scenario"
    SENSITIVITY = "sensitivity"
    BOOTSTRAP = "bootstrap"


class ScenarioType(Enum):
    """Tipos de escenarios predefinidos"""
    COVID_CRASH = "covid_crash"  # Marzo 2020
    LUNA_CRASH = "luna_crash"  # Mayo 2022
    FTX_COLLAPSE = "ftx_collapse"  # Noviembre 2022
    FLASH_CRASH = "flash_crash"  # Caida rapida y recuperacion
    BEAR_MARKET = "bear_market"  # Mercado bajista prolongado
    HIGH_VOLATILITY = "high_volatility"  # Volatilidad extrema
    LOW_VOLATILITY = "low_volatility"  # Volatilidad muy baja
    CUSTOM = "custom"


@dataclass
class MonteCarloConfig:
    """Configuracion para simulacion Monte Carlo"""
    n_simulations: int = 1000  # Numero de simulaciones
    confidence_levels: List[float] = field(default_factory=lambda: [0.95, 0.99])
    shuffle_returns: bool = True  # Barajar retornos
    block_size: Optional[int] = None  # Tamano de bloques (para bootstrap)
    preserve_correlation: bool = False  # Preservar estructura temporal
    random_seed: Optional[int] = None

    def __post_init__(self):
        if self.n_simulations < 100:
            raise ValueError("n_simulations debe ser al menos 100")


@dataclass
class ScenarioConfig:
    """Configuracion para analisis de escenarios"""
    scenario_type: ScenarioType = ScenarioType.COVID_CRASH

    # Para escenarios personalizados
    custom_drawdown: float = 0.0  # Drawdown a simular (ej: 0.30 = 30%)
    custom_duration_days: int = 0  # Duracion del drawdown
    custom_recovery_days: int = 0  # Dias para recuperar
    custom_volatility_mult: float = 1.0  # Multiplicador de volatilidad

    # Parametros del escenario
    apply_to_portfolio: bool = True  # Aplicar a todo el portfolio


@dataclass
class SensitivityConfig:
    """Configuracion para analisis de sensibilidad"""
    parameters: Dict[str, List[float]] = field(default_factory=dict)  # param -> [valores a probar]
    variation_pct: float = 0.20  # Variacion porcentual si no se especifican valores
    n_steps: int = 5  # Numero de pasos por parametro
    metric_to_optimize: str = "sharpe_ratio"  # Metrica objetivo


@dataclass
class StressTestConfig:
    """Configuracion general de stress testing"""
    monte_carlo: MonteCarloConfig = field(default_factory=MonteCarloConfig)
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    sensitivity: SensitivityConfig = field(default_factory=SensitivityConfig)

    # General
    initial_capital: float = 10000.0
    risk_free_rate: float = 0.02

    def to_dict(self) -> dict:
        return {
            "initial_capital": self.initial_capital,
            "risk_free_rate": self.risk_free_rate,
            "monte_carlo": {
                "n_simulations": self.monte_carlo.n_simulations,
                "confidence_levels": self.monte_carlo.confidence_levels,
            },
            "scenario": {
                "scenario_type": self.scenario.scenario_type.value,
            },
            "sensitivity": {
                "parameters": self.sensitivity.parameters,
                "variation_pct": self.sensitivity.variation_pct,
            },
        }


@dataclass
class MonteCarloResult:
    """Resultado de simulacion Monte Carlo"""
    n_simulations: int = 0

    # Distribucion de retornos finales
    mean_return: float = 0.0
    median_return: float = 0.0
    std_return: float = 0.0
    min_return: float = 0.0
    max_return: float = 0.0

    # Percentiles
    percentile_5: float = 0.0
    percentile_25: float = 0.0
    percentile_75: float = 0.0
    percentile_95: float = 0.0

    # Probabilidades
    prob_profit: float = 0.0  # P(retorno > 0)
    prob_loss_10pct: float = 0.0  # P(retorno < -10%)
    prob_loss_20pct: float = 0.0  # P(retorno < -20%)
    prob_double: float = 0.0  # P(retorno > 100%)

    # Distribucion de drawdowns
    mean_max_drawdown: float = 0.0
    median_max_drawdown: float = 0.0
    worst_max_drawdown: float = 0.0

    # Distribucion de Sharpe
    mean_sharpe: float = 0.0
    median_sharpe: float = 0.0
    std_sharpe: float = 0.0

    # Datos raw
    all_final_returns: List[float] = field(default_factory=list)
    all_max_drawdowns: List[float] = field(default_factory=list)
    all_sharpe_ratios: List[float] = field(default_factory=list)

    # VaR y CVaR
    var_95: float = 0.0
    var_99: float = 0.0
    cvar_95: float = 0.0
    cvar_99: float = 0.0

    def to_dict(self) -> dict:
        return {
            "n_simulations": self.n_simulations,
            "returns": {
                "mean": self.mean_return,
                "median": self.median_return,
                "std": self.std_return,
                "min": self.min_return,
                "max": self.max_return,
            },
            "percentiles": {
                "5th": self.percentile_5,
                "25th": self.percentile_25,
                "75th": self.percentile_75,
                "95th": self.percentile_95,
            },
            "probabilities": {
                "profit": self.prob_profit,
                "loss_10pct": self.prob_loss_10pct,
                "loss_20pct": self.prob_loss_20pct,
                "double": self.prob_double,
            },
            "drawdown": {
                "mean": self.mean_max_drawdown,
                "median": self.median_max_drawdown,
                "worst": self.worst_max_drawdown,
            },
            "sharpe": {
                "mean": self.mean_sharpe,
                "median": self.median_sharpe,
                "std": self.std_sharpe,
            },
            "risk_metrics": {
                "var_95": self.var_95,
                "var_99": self.var_99,
                "cvar_95": self.cvar_95,
                "cvar_99": self.cvar_99,
            },
        }


@dataclass
class ScenarioResult:
    """Resultado de un escenario de stress"""
    scenario_type: ScenarioType = ScenarioType.CUSTOM
    scenario_name: str = ""

    # Performance durante el escenario
    initial_equity: float = 0.0
    final_equity: float = 0.0
    total_return: float = 0.0
    total_return_pct: float = 0.0

    # Drawdown
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    drawdown_duration_days: int = 0
    recovery_days: int = 0

    # Volatilidad
    volatility: float = 0.0
    volatility_vs_normal: float = 0.0  # Ratio vs periodo normal

    # Peores dias
    worst_day_return: float = 0.0
    worst_week_return: float = 0.0

    # Metricas durante estres
    sharpe_during_stress: float = 0.0

    # Sobrevivencia
    survived: bool = True
    margin_call: bool = False

    def to_dict(self) -> dict:
        return {
            "scenario_type": self.scenario_type.value,
            "scenario_name": self.scenario_name,
            "performance": {
                "initial_equity": self.initial_equity,
                "final_equity": self.final_equity,
                "total_return": self.total_return,
                "total_return_pct": self.total_return_pct,
            },
            "drawdown": {
                "max_drawdown": self.max_drawdown,
                "max_drawdown_pct": self.max_drawdown_pct,
                "duration_days": self.drawdown_duration_days,
                "recovery_days": self.recovery_days,
            },
            "volatility": {
                "value": self.volatility,
                "vs_normal": self.volatility_vs_normal,
            },
            "worst_periods": {
                "worst_day": self.worst_day_return,
                "worst_week": self.worst_week_return,
            },
            "survival": {
                "survived": self.survived,
                "margin_call": self.margin_call,
            },
        }


@dataclass
class SensitivityResult:
    """Resultado de analisis de sensibilidad"""
    parameter_name: str = ""
    base_value: float = 0.0

    # Valores probados y resultados
    tested_values: List[float] = field(default_factory=list)
    metric_values: List[float] = field(default_factory=list)  # Resultados para cada valor

    # Analisis
    optimal_value: float = 0.0
    optimal_metric: float = 0.0

    # Robustez
    is_robust: bool = True  # True si funciona en rango amplio
    working_range: tuple = (0.0, 0.0)  # Rango donde funciona
    sensitivity_score: float = 0.0  # 0-1, mayor = mas sensible

    # Estadisticas
    metric_mean: float = 0.0
    metric_std: float = 0.0
    metric_min: float = 0.0
    metric_max: float = 0.0

    def to_dict(self) -> dict:
        return {
            "parameter": self.parameter_name,
            "base_value": self.base_value,
            "tested_values": self.tested_values,
            "metric_values": self.metric_values,
            "optimal": {
                "value": self.optimal_value,
                "metric": self.optimal_metric,
            },
            "robustness": {
                "is_robust": self.is_robust,
                "working_range": self.working_range,
                "sensitivity_score": self.sensitivity_score,
            },
            "statistics": {
                "mean": self.metric_mean,
                "std": self.metric_std,
                "min": self.metric_min,
                "max": self.metric_max,
            },
        }


@dataclass
class StressTestReport:
    """Reporte completo de stress testing"""
    timestamp: datetime = field(default_factory=datetime.now)
    config: Optional[StressTestConfig] = None

    # Resultados
    monte_carlo: Optional[MonteCarloResult] = None
    scenarios: List[ScenarioResult] = field(default_factory=list)
    sensitivity: List[SensitivityResult] = field(default_factory=list)

    # Resumen ejecutivo
    overall_robustness_score: float = 0.0  # 0-100
    risk_rating: str = "UNKNOWN"  # LOW, MEDIUM, HIGH, EXTREME
    key_risks: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    # Metadata
    execution_time: float = 0.0

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "monte_carlo": self.monte_carlo.to_dict() if self.monte_carlo else None,
            "scenarios": [s.to_dict() for s in self.scenarios],
            "sensitivity": [s.to_dict() for s in self.sensitivity],
            "summary": {
                "robustness_score": self.overall_robustness_score,
                "risk_rating": self.risk_rating,
                "key_risks": self.key_risks,
                "recommendations": self.recommendations,
            },
            "execution_time": self.execution_time,
        }


# Escenarios predefinidos
PREDEFINED_SCENARIOS = {
    ScenarioType.COVID_CRASH: {
        "name": "COVID-19 Crash (Marzo 2020)",
        "drawdown": 0.35,
        "duration_days": 33,
        "recovery_days": 140,
        "volatility_mult": 4.0,
    },
    ScenarioType.LUNA_CRASH: {
        "name": "Luna/UST Crash (Mayo 2022)",
        "drawdown": 0.60,
        "duration_days": 14,
        "recovery_days": float("inf"),  # No recuperado
        "volatility_mult": 6.0,
    },
    ScenarioType.FTX_COLLAPSE: {
        "name": "FTX Collapse (Nov 2022)",
        "drawdown": 0.25,
        "duration_days": 7,
        "recovery_days": 60,
        "volatility_mult": 3.0,
    },
    ScenarioType.FLASH_CRASH: {
        "name": "Flash Crash",
        "drawdown": 0.15,
        "duration_days": 1,
        "recovery_days": 3,
        "volatility_mult": 10.0,
    },
    ScenarioType.BEAR_MARKET: {
        "name": "Bear Market Prolongado",
        "drawdown": 0.50,
        "duration_days": 365,
        "recovery_days": 500,
        "volatility_mult": 1.5,
    },
    ScenarioType.HIGH_VOLATILITY: {
        "name": "Alta Volatilidad",
        "drawdown": 0.20,
        "duration_days": 30,
        "recovery_days": 45,
        "volatility_mult": 3.0,
    },
    ScenarioType.LOW_VOLATILITY: {
        "name": "Baja Volatilidad",
        "drawdown": 0.05,
        "duration_days": 90,
        "recovery_days": 30,
        "volatility_mult": 0.3,
    },
}
