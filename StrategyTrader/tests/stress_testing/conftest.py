"""Fixtures para tests del modulo de stress testing"""

import pytest
import numpy as np
from typing import Dict, Any

from src.stress_testing.models import (
    MonteCarloConfig,
    ScenarioConfig,
    SensitivityConfig,
    StressTestConfig,
    ScenarioType,
)


@pytest.fixture
def sample_returns() -> np.ndarray:
    """Retornos sinteticos para testing"""
    np.random.seed(42)
    # Simular 252 dias de trading (1 año)
    returns = np.random.normal(0.0005, 0.02, 252)
    return returns


@pytest.fixture
def trending_returns() -> np.ndarray:
    """Retornos con tendencia alcista"""
    np.random.seed(42)
    returns = np.random.normal(0.002, 0.015, 252)
    return returns


@pytest.fixture
def volatile_returns() -> np.ndarray:
    """Retornos altamente volatiles"""
    np.random.seed(42)
    returns = np.random.normal(0.0, 0.05, 252)
    return returns


@pytest.fixture
def losing_returns() -> np.ndarray:
    """Retornos negativos consistentes"""
    np.random.seed(42)
    returns = np.random.normal(-0.003, 0.02, 252)
    return returns


@pytest.fixture
def short_returns() -> np.ndarray:
    """Retornos cortos para tests rapidos"""
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, 30)
    return returns


@pytest.fixture
def monte_carlo_config() -> MonteCarloConfig:
    """Configuracion de Monte Carlo para tests"""
    return MonteCarloConfig(
        n_simulations=100,
        confidence_levels=[0.90, 0.95],
        block_size=5,
    )


@pytest.fixture
def scenario_config() -> ScenarioConfig:
    """Configuracion de escenarios para tests"""
    return ScenarioConfig(
        scenario_type=ScenarioType.COVID_CRASH,
        custom_drawdown=0.30,
        custom_duration_days=20,
        custom_recovery_days=30,
        custom_volatility_mult=2.0,
    )


@pytest.fixture
def sensitivity_config() -> SensitivityConfig:
    """Configuracion de sensibilidad para tests"""
    return SensitivityConfig(
        variation_pct=0.20,
        n_steps=5,
        metric_to_optimize="sharpe_ratio",
    )


@pytest.fixture
def stress_test_config(
    monte_carlo_config: MonteCarloConfig,
    sensitivity_config: SensitivityConfig,
) -> StressTestConfig:
    """Configuracion completa de stress testing"""
    return StressTestConfig(
        initial_capital=10000.0,
        risk_free_rate=0.02,
        monte_carlo=monte_carlo_config,
        sensitivity=sensitivity_config,
    )


@pytest.fixture
def mock_strategy_func():
    """Funcion de estrategia mock para tests de sensibilidad"""
    def strategy(params: Dict[str, Any]) -> Dict[str, float]:
        # Simular metricas basadas en parametros
        sma_period = params.get("sma_period", 20)
        rsi_threshold = params.get("rsi_threshold", 30)

        # Valores optimos simulados: sma_period=25, rsi_threshold=35
        sma_distance = abs(sma_period - 25) / 25
        rsi_distance = abs(rsi_threshold - 35) / 35

        sharpe = 1.5 - sma_distance - rsi_distance
        total_return = 0.15 - sma_distance * 0.1 - rsi_distance * 0.1
        max_drawdown = 0.10 + sma_distance * 0.05 + rsi_distance * 0.05

        return {
            "sharpe_ratio": max(sharpe, 0),
            "total_return": total_return,
            "max_drawdown": max_drawdown,
            "win_rate": 0.55 - sma_distance * 0.1,
        }

    return strategy


@pytest.fixture
def base_parameters() -> Dict[str, float]:
    """Parametros base para tests de sensibilidad"""
    return {
        "sma_period": 20,
        "rsi_threshold": 30,
    }


@pytest.fixture
def fragile_strategy_func():
    """Estrategia fragil (muy sensible a parametros)"""
    def strategy(params: Dict[str, Any]) -> Dict[str, float]:
        sma_period = params.get("sma_period", 20)

        # Solo funciona bien en rango muy estrecho
        if 18 <= sma_period <= 22:
            sharpe = 1.5
        else:
            sharpe = -0.5

        return {
            "sharpe_ratio": sharpe,
            "total_return": sharpe * 0.1,
            "max_drawdown": 0.20 if sharpe < 0 else 0.10,
        }

    return strategy


@pytest.fixture
def robust_strategy_func():
    """Estrategia robusta (poco sensible a parametros)"""
    def strategy(params: Dict[str, Any]) -> Dict[str, float]:
        # Funciona bien con cualquier parametro
        return {
            "sharpe_ratio": 1.2,
            "total_return": 0.12,
            "max_drawdown": 0.08,
            "win_rate": 0.55,
        }

    return strategy
