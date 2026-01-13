"""
Tests para el modulo de optimizacion de estrategias
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from src.optimizer import StrategyOptimizer, OptimizationVisualizer
from src.optimizers.optimization_types import (
    ParameterSpace,
    OptimizationResult,
    WalkForwardResult
)
from src.strategy import MovingAverageCrossoverStrategy, StrategyConfig


class TestParameterSpace:
    """Tests para ParameterSpace"""

    def test_int_parameter(self):
        """Test parametro entero"""
        param = ParameterSpace(
            name='period',
            param_type='int',
            low=5,
            high=50,
            step=5
        )

        assert param.name == 'period'
        assert param.param_type == 'int'
        assert param.low == 5
        assert param.high == 50

    def test_float_parameter(self):
        """Test parametro flotante"""
        param = ParameterSpace(
            name='threshold',
            param_type='float',
            low=0.0,
            high=1.0
        )

        assert param.name == 'threshold'
        assert param.param_type == 'float'

    def test_categorical_parameter(self):
        """Test parametro categorico"""
        param = ParameterSpace(
            name='indicator',
            param_type='categorical',
            values=['sma', 'ema', 'wma']
        )

        assert param.name == 'indicator'
        assert param.values == ['sma', 'ema', 'wma']


class TestOptimizationResult:
    """Tests para OptimizationResult"""

    def test_result_creation(self):
        """Test creacion de resultado"""
        result = OptimizationResult(
            best_params={'period': 20},
            best_score=1.5,
            all_results=pd.DataFrame({'score': [1.0, 1.2, 1.5]}),
            optimization_time=10.5,
            method='Test',
            iterations=100
        )

        assert result.best_score == 1.5
        assert result.method == 'Test'
        assert result.iterations == 100

    def test_print_summary(self, capsys):
        """Test impresion de resumen"""
        result = OptimizationResult(
            best_params={'period': 20},
            best_score=1.5,
            all_results=pd.DataFrame({'score': [1.0, 1.2, 1.5]}),
            optimization_time=10.5,
            method='Test Method',
            iterations=100
        )

        result.print_summary()

        captured = capsys.readouterr()
        assert 'Test Method' in captured.out
        assert '1.5' in captured.out


class TestWalkForwardResult:
    """Tests para WalkForwardResult"""

    def test_result_creation(self):
        """Test creacion de resultado WalkForward"""
        splits = [
            {'split': 1, 'train_score': 1.5, 'oos_score': 1.2, 'best_params': {}},
            {'split': 2, 'train_score': 1.4, 'oos_score': 1.1, 'best_params': {}}
        ]

        result = WalkForwardResult(
            splits_results=splits,
            aggregated_metrics={'avg_oos_score': 1.15},
            out_of_sample_equity=[10000, 10100, 10200],
            robustness_score=0.75,
            optimization_time=60.0,
            n_splits=2,
            train_size=0.6,
            optimization_method='random'
        )

        assert result.robustness_score == 0.75
        assert result.n_splits == 2
        assert len(result.splits_results) == 2

    def test_print_summary(self, capsys):
        """Test impresion de resumen WalkForward"""
        splits = [
            {'split': 1, 'train_score': 1.5, 'oos_score': 1.2, 'best_params': {}},
        ]

        result = WalkForwardResult(
            splits_results=splits,
            aggregated_metrics={'avg_oos_score': 1.2},
            out_of_sample_equity=[10000],
            robustness_score=0.8,
            optimization_time=30.0,
            n_splits=1,
            train_size=0.6,
            optimization_method='random'
        )

        result.print_summary()

        captured = capsys.readouterr()
        assert 'WALK FORWARD' in captured.out
        assert 'ROBUSTA' in captured.out  # Score > 0.7


class TestStrategyOptimizer:
    """Tests para StrategyOptimizer"""

    @pytest.fixture
    def optimizer_setup(self, sample_ohlcv_data):
        """Setup para optimizer tests"""
        config = StrategyConfig(
            symbol='TEST/USDT',
            timeframe='1h',
            initial_capital=10000.0,
            risk_per_trade=2.0,
            max_positions=1,
            commission=0.1,
            slippage=0.05
        )

        optimizer = StrategyOptimizer(
            strategy_class=MovingAverageCrossoverStrategy,
            data=sample_ohlcv_data,
            config_template=config,
            objective_metric='sharpe_ratio'
        )

        return optimizer

    def test_optimizer_initialization(self, optimizer_setup):
        """Test inicializacion del optimizador"""
        optimizer = optimizer_setup

        assert optimizer.strategy_class == MovingAverageCrossoverStrategy
        assert optimizer.objective_metric == 'sharpe_ratio'
        assert len(optimizer.parameter_space) == 0

    def test_add_parameter(self, optimizer_setup):
        """Test agregar parametros"""
        optimizer = optimizer_setup

        optimizer.add_parameter('fast_period', 'int', low=5, high=20)
        optimizer.add_parameter('slow_period', 'int', low=20, high=50)

        assert len(optimizer.parameter_space) == 2
        assert optimizer.parameter_space[0].name == 'fast_period'
        assert optimizer.parameter_space[1].name == 'slow_period'

    def test_evaluate_parameters(self, optimizer_setup):
        """Test evaluacion de parametros"""
        optimizer = optimizer_setup

        params = {'fast_period': 10, 'slow_period': 30, 'rsi_period': 14}
        score = optimizer._evaluate_parameters(params)

        # Score debe ser un numero (puede ser -inf si falla)
        assert isinstance(score, (int, float))

    def test_evaluate_parameters_detailed(self, optimizer_setup):
        """Test evaluacion detallada de parametros"""
        optimizer = optimizer_setup

        params = {'fast_period': 10, 'slow_period': 30, 'rsi_period': 14}
        result = optimizer._evaluate_parameters_detailed(params)

        assert 'score' in result
        assert 'fast_period' in result

    def test_results_cache(self, optimizer_setup):
        """Test que los resultados se cachean"""
        optimizer = optimizer_setup

        params = {'fast_period': 10, 'slow_period': 30, 'rsi_period': 14}

        # Primera evaluacion
        score1 = optimizer._evaluate_parameters(params)

        # Segunda evaluacion (debe venir del cache)
        score2 = optimizer._evaluate_parameters(params)

        assert score1 == score2
        assert len(optimizer.results_cache) == 1


class TestRandomSearch:
    """Tests para Random Search"""

    @pytest.fixture
    def optimizer_with_params(self, sample_ohlcv_data):
        """Optimizer con parametros configurados"""
        config = StrategyConfig(
            symbol='TEST/USDT',
            timeframe='1h',
            initial_capital=10000.0,
            risk_per_trade=2.0,
            max_positions=1,
            commission=0.1,
            slippage=0.05
        )

        optimizer = StrategyOptimizer(
            strategy_class=MovingAverageCrossoverStrategy,
            data=sample_ohlcv_data,
            config_template=config,
            objective_metric='sharpe_ratio'
        )

        optimizer.add_parameter('fast_period', 'int', low=5, high=15)
        optimizer.add_parameter('slow_period', 'int', low=20, high=40)

        return optimizer

    def test_random_search_runs(self, optimizer_with_params):
        """Test que random search se ejecuta"""
        result = optimizer_with_params.random_search(n_iter=5, verbose=False)

        assert isinstance(result, OptimizationResult)
        assert result.method == 'Random Search'
        assert result.iterations == 5

    def test_random_search_returns_best(self, optimizer_with_params):
        """Test que random search retorna mejores parametros"""
        result = optimizer_with_params.random_search(n_iter=10, verbose=False)

        # Debe tener best_params
        assert 'fast_period' in result.best_params
        assert 'slow_period' in result.best_params

        # Parametros deben estar en rango
        assert 5 <= result.best_params['fast_period'] <= 15
        assert 20 <= result.best_params['slow_period'] <= 40

    def test_random_search_all_results(self, optimizer_with_params):
        """Test que random search guarda todos los resultados"""
        n_iter = 10
        result = optimizer_with_params.random_search(n_iter=n_iter, verbose=False)

        assert len(result.all_results) == n_iter
        assert 'score' in result.all_results.columns


class TestGridSearch:
    """Tests para Grid Search"""

    @pytest.fixture
    def optimizer_grid(self, sample_ohlcv_data):
        """Optimizer para grid search (pocos parametros)"""
        config = StrategyConfig(
            symbol='TEST/USDT',
            timeframe='1h',
            initial_capital=10000.0,
            risk_per_trade=2.0,
            max_positions=1,
            commission=0.1,
            slippage=0.05
        )

        optimizer = StrategyOptimizer(
            strategy_class=MovingAverageCrossoverStrategy,
            data=sample_ohlcv_data,
            config_template=config,
            objective_metric='sharpe_ratio'
        )

        # Solo 2 valores por parametro para test rapido
        optimizer.add_parameter('fast_period', 'int', low=5, high=10, step=5)
        optimizer.add_parameter('slow_period', 'int', low=20, high=30, step=10)

        return optimizer

    def test_grid_search_runs(self, optimizer_grid):
        """Test que grid search se ejecuta"""
        result = optimizer_grid.grid_search(verbose=False)

        assert isinstance(result, OptimizationResult)
        assert result.method == 'Grid Search'

    def test_grid_search_exhaustive(self, optimizer_grid):
        """Test que grid search prueba todas las combinaciones"""
        result = optimizer_grid.grid_search(verbose=False)

        # 2 valores de fast_period x 2 valores de slow_period = 4 combinaciones
        assert len(result.all_results) == 4


class TestWalkForwardOptimization:
    """Tests para Walk Forward Optimization"""

    @pytest.fixture
    def optimizer_wf(self, sample_ohlcv_data):
        """Optimizer para walk forward"""
        config = StrategyConfig(
            symbol='TEST/USDT',
            timeframe='1h',
            initial_capital=10000.0,
            risk_per_trade=2.0,
            max_positions=1,
            commission=0.1,
            slippage=0.05
        )

        optimizer = StrategyOptimizer(
            strategy_class=MovingAverageCrossoverStrategy,
            data=sample_ohlcv_data,
            config_template=config,
            objective_metric='sharpe_ratio'
        )

        optimizer.add_parameter('fast_period', 'int', low=5, high=15)
        optimizer.add_parameter('slow_period', 'int', low=20, high=40)

        return optimizer

    def test_walk_forward_runs(self, optimizer_wf):
        """Test que walk forward se ejecuta"""
        result = optimizer_wf.walk_forward_optimization(
            optimization_method='random',
            n_splits=2,
            train_size=0.6,
            n_iter=5,
            verbose=False
        )

        assert isinstance(result, WalkForwardResult)
        assert result.n_splits == 2
        assert result.train_size == 0.6

    def test_walk_forward_splits(self, optimizer_wf):
        """Test que walk forward genera splits correctos"""
        result = optimizer_wf.walk_forward_optimization(
            optimization_method='random',
            n_splits=3,
            train_size=0.6,
            n_iter=5,
            verbose=False
        )

        # Debe tener resultados por split
        assert len(result.splits_results) <= 3

    def test_walk_forward_robustness_score(self, optimizer_wf):
        """Test que robustness score esta en rango"""
        result = optimizer_wf.walk_forward_optimization(
            optimization_method='random',
            n_splits=2,
            train_size=0.6,
            n_iter=5,
            verbose=False
        )

        assert 0.0 <= result.robustness_score <= 1.0

    def test_walk_forward_aggregated_metrics(self, optimizer_wf):
        """Test que se calculan metricas agregadas"""
        result = optimizer_wf.walk_forward_optimization(
            optimization_method='random',
            n_splits=2,
            train_size=0.6,
            n_iter=5,
            verbose=False
        )

        # Debe tener metricas agregadas
        assert isinstance(result.aggregated_metrics, dict)


class TestOptimizationVisualizer:
    """Tests para OptimizationVisualizer"""

    def test_visualizer_initialization(self):
        """Test inicializacion del visualizador"""
        result = OptimizationResult(
            best_params={'period': 20},
            best_score=1.5,
            all_results=pd.DataFrame({'score': [1.0, 1.2, 1.5]}),
            optimization_time=10.5,
            method='Test',
            iterations=100
        )

        visualizer = OptimizationVisualizer(result)

        assert visualizer.result == result


class TestOptimizerEdgeCases:
    """Tests para casos limite del optimizador"""

    def test_empty_parameter_space(self, sample_ohlcv_data):
        """Test con espacio de parametros vacio"""
        config = StrategyConfig(
            symbol='TEST/USDT',
            timeframe='1h',
            initial_capital=10000.0,
            risk_per_trade=2.0,
            max_positions=1,
            commission=0.1,
            slippage=0.05
        )

        optimizer = StrategyOptimizer(
            strategy_class=MovingAverageCrossoverStrategy,
            data=sample_ohlcv_data,
            config_template=config,
            objective_metric='sharpe_ratio'
        )

        # Sin parametros definidos, random search deberia manejar el caso
        # o lanzar error apropiado
        assert len(optimizer.parameter_space) == 0

    def test_invalid_parameter_evaluation(self, sample_ohlcv_data):
        """Test evaluacion con parametros invalidos"""
        config = StrategyConfig(
            symbol='TEST/USDT',
            timeframe='1h',
            initial_capital=10000.0,
            risk_per_trade=2.0,
            max_positions=1,
            commission=0.1,
            slippage=0.05
        )

        optimizer = StrategyOptimizer(
            strategy_class=MovingAverageCrossoverStrategy,
            data=sample_ohlcv_data,
            config_template=config,
            objective_metric='sharpe_ratio'
        )

        # Parametros que podrian causar problemas
        params = {'fast_period': 100, 'slow_period': 5}  # fast > slow
        score = optimizer._evaluate_parameters(params)

        # Debe retornar -inf o manejar el error
        assert isinstance(score, (int, float))

    def test_very_small_data(self):
        """Test con datos muy pequenos"""
        small_data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [101, 102, 103],
            'low': [99, 100, 101],
            'close': [100.5, 101.5, 102.5],
            'volume': [1000, 1000, 1000]
        }, index=pd.date_range('2024-01-01', periods=3, freq='1h'))

        config = StrategyConfig(
            symbol='TEST/USDT',
            timeframe='1h',
            initial_capital=10000.0,
            risk_per_trade=2.0,
            max_positions=1,
            commission=0.1,
            slippage=0.05
        )

        optimizer = StrategyOptimizer(
            strategy_class=MovingAverageCrossoverStrategy,
            data=small_data,
            config_template=config,
            objective_metric='sharpe_ratio'
        )

        optimizer.add_parameter('fast_period', 'int', low=1, high=2)

        # No debe fallar con datos minimos
        result = optimizer.random_search(n_iter=2, verbose=False)
        assert isinstance(result, OptimizationResult)
