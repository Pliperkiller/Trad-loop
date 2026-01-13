"""
Tests para el modulo de analisis de performance
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.performance import PerformanceAnalyzer, PerformanceVisualizer


class TestPerformanceAnalyzer:
    """Tests para PerformanceAnalyzer"""

    def test_analyzer_initialization(self, sample_equity_curve, sample_trades):
        """Test inicializacion del analizador"""
        analyzer = PerformanceAnalyzer(
            equity_curve=sample_equity_curve,
            trades=sample_trades,
            initial_capital=10000
        )

        assert analyzer.initial_capital == 10000
        assert len(analyzer.equity_curve) == len(sample_equity_curve)
        assert len(analyzer.trades) == len(sample_trades)

    def test_calculate_all_metrics(self, sample_equity_curve, sample_trades):
        """Test calculo de todas las metricas"""
        analyzer = PerformanceAnalyzer(
            equity_curve=sample_equity_curve,
            trades=sample_trades,
            initial_capital=10000
        )

        metrics = analyzer.calculate_all_metrics()

        # Verificar que contiene las metricas principales
        assert 'total_return_pct' in metrics
        assert 'max_drawdown_pct' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'win_rate_pct' in metrics
        assert 'profit_factor' in metrics
        assert 'total_trades' in metrics

    def test_profitability_metrics(self, sample_equity_curve, sample_trades):
        """Test metricas de rentabilidad"""
        initial_capital = 10000
        analyzer = PerformanceAnalyzer(
            equity_curve=sample_equity_curve,
            trades=sample_trades,
            initial_capital=initial_capital
        )

        metrics = analyzer._profitability_metrics()

        # Total return debe calcularse correctamente
        expected_return = ((sample_equity_curve[-1] - initial_capital) / initial_capital) * 100
        assert abs(metrics['total_return_pct'] - expected_return) < 0.01

        # Final capital debe coincidir
        assert metrics['final_capital'] == sample_equity_curve[-1]

    def test_risk_metrics(self, sample_equity_curve, sample_trades):
        """Test metricas de riesgo"""
        analyzer = PerformanceAnalyzer(
            equity_curve=sample_equity_curve,
            trades=sample_trades,
            initial_capital=10000
        )

        metrics = analyzer._risk_metrics()

        # Max drawdown debe ser negativo o cero
        assert metrics['max_drawdown_pct'] <= 0

        # Volatilidad debe ser positiva
        assert metrics['volatility_pct'] >= 0

    def test_risk_adjusted_metrics(self, sample_equity_curve, sample_trades):
        """Test metricas ajustadas por riesgo"""
        analyzer = PerformanceAnalyzer(
            equity_curve=sample_equity_curve,
            trades=sample_trades,
            initial_capital=10000
        )

        metrics = analyzer._risk_adjusted_metrics()

        # Debe contener los ratios principales
        assert 'sharpe_ratio' in metrics
        assert 'sortino_ratio' in metrics
        assert 'calmar_ratio' in metrics
        assert 'omega_ratio' in metrics

    def test_consistency_metrics(self, sample_equity_curve, sample_trades):
        """Test metricas de consistencia"""
        analyzer = PerformanceAnalyzer(
            equity_curve=sample_equity_curve,
            trades=sample_trades,
            initial_capital=10000
        )

        metrics = analyzer._consistency_metrics()

        # Win rate debe estar entre 0 y 100
        assert 0 <= metrics['win_rate_pct'] <= 100

        # Profit factor debe ser positivo (si hay trades perdedores)
        assert metrics['profit_factor'] >= 0

    def test_operational_metrics(self, sample_equity_curve, sample_trades):
        """Test metricas operativas"""
        analyzer = PerformanceAnalyzer(
            equity_curve=sample_equity_curve,
            trades=sample_trades,
            initial_capital=10000
        )

        metrics = analyzer._operational_metrics()

        # Total trades debe coincidir
        assert metrics['total_trades'] == len(sample_trades)

        # Winning + Losing debe ser <= Total
        assert metrics['winning_trades'] + metrics['losing_trades'] <= metrics['total_trades']

    def test_print_report(self, sample_equity_curve, sample_trades, capsys):
        """Test impresion de reporte"""
        analyzer = PerformanceAnalyzer(
            equity_curve=sample_equity_curve,
            trades=sample_trades,
            initial_capital=10000
        )

        # No debe lanzar excepciones
        analyzer.print_report()

        # Verificar que se imprimio algo
        captured = capsys.readouterr()
        assert 'REPORTE DE PERFORMANCE' in captured.out
        assert 'RENTABILIDAD' in captured.out
        assert 'RIESGO' in captured.out

    def test_empty_trades(self, sample_equity_curve):
        """Test con lista de trades vacia"""
        empty_trades = pd.DataFrame(columns=[
            'entry_time', 'exit_time', 'entry_price', 'exit_price',
            'quantity', 'pnl', 'return_pct', 'reason'
        ])

        analyzer = PerformanceAnalyzer(
            equity_curve=sample_equity_curve,
            trades=empty_trades,
            initial_capital=10000
        )

        metrics = analyzer.calculate_all_metrics()

        # Debe manejar trades vacios sin errores
        assert metrics['total_trades'] == 0
        assert metrics['win_rate_pct'] == 0

    def test_single_trade(self, sample_equity_curve):
        """Test con un solo trade"""
        single_trade = pd.DataFrame([{
            'entry_time': datetime(2024, 1, 1, 10, 0),
            'exit_time': datetime(2024, 1, 1, 14, 0),
            'entry_price': 100.0,
            'exit_price': 110.0,
            'quantity': 10.0,
            'pnl': 100.0,
            'return_pct': 10.0,
            'reason': 'Take Profit'
        }])

        analyzer = PerformanceAnalyzer(
            equity_curve=sample_equity_curve,
            trades=single_trade,
            initial_capital=10000
        )

        metrics = analyzer.calculate_all_metrics()

        assert metrics['total_trades'] == 1
        assert metrics['winning_trades'] == 1
        assert metrics['win_rate_pct'] == 100.0

    def test_all_winning_trades(self, sample_equity_curve):
        """Test con todos los trades ganadores"""
        winning_trades = pd.DataFrame([
            {
                'entry_time': datetime(2024, 1, i, 10, 0),
                'exit_time': datetime(2024, 1, i, 14, 0),
                'entry_price': 100.0,
                'exit_price': 110.0,
                'quantity': 10.0,
                'pnl': 100.0,
                'return_pct': 10.0,
                'reason': 'Take Profit'
            }
            for i in range(1, 6)
        ])

        analyzer = PerformanceAnalyzer(
            equity_curve=sample_equity_curve,
            trades=winning_trades,
            initial_capital=10000
        )

        metrics = analyzer.calculate_all_metrics()

        assert metrics['win_rate_pct'] == 100.0
        assert metrics['profit_factor'] == float('inf')

    def test_all_losing_trades(self, sample_equity_curve):
        """Test con todos los trades perdedores"""
        losing_trades = pd.DataFrame([
            {
                'entry_time': datetime(2024, 1, i, 10, 0),
                'exit_time': datetime(2024, 1, i, 14, 0),
                'entry_price': 100.0,
                'exit_price': 90.0,
                'quantity': 10.0,
                'pnl': -100.0,
                'return_pct': -10.0,
                'reason': 'Stop Loss'
            }
            for i in range(1, 6)
        ])

        analyzer = PerformanceAnalyzer(
            equity_curve=sample_equity_curve,
            trades=losing_trades,
            initial_capital=10000
        )

        metrics = analyzer.calculate_all_metrics()

        assert metrics['win_rate_pct'] == 0.0
        assert metrics['profit_factor'] == 0.0

    def test_drawdown_calculation(self):
        """Test calculo de drawdown"""
        # Equity curve con drawdown conocido
        # Sube a 12000, cae a 9000 (25% DD), sube a 11000
        equity = [10000, 11000, 12000, 10000, 9000, 10000, 11000]
        trades = pd.DataFrame([{
            'entry_time': datetime(2024, 1, 1),
            'exit_time': datetime(2024, 1, 2),
            'entry_price': 100, 'exit_price': 110,
            'quantity': 10, 'pnl': 100, 'return_pct': 10.0, 'reason': 'TP'
        }])

        analyzer = PerformanceAnalyzer(
            equity_curve=equity,
            trades=trades,
            initial_capital=10000
        )

        metrics = analyzer._risk_metrics()

        # Max DD debe ser -25% (de 12000 a 9000)
        expected_dd = ((9000 - 12000) / 12000) * 100
        assert abs(metrics['max_drawdown_pct'] - expected_dd) < 0.1


class TestPerformanceVisualizer:
    """Tests para PerformanceVisualizer"""

    def test_visualizer_initialization(self, sample_equity_curve, sample_trades):
        """Test inicializacion del visualizador"""
        analyzer = PerformanceAnalyzer(
            equity_curve=sample_equity_curve,
            trades=sample_trades,
            initial_capital=10000
        )

        visualizer = PerformanceVisualizer(analyzer)

        assert visualizer.analyzer == analyzer
        assert len(visualizer.equity_curve) == len(sample_equity_curve)

    def test_plot_comprehensive_dashboard(self, sample_equity_curve, sample_trades):
        """Test que el dashboard se genera sin errores"""
        analyzer = PerformanceAnalyzer(
            equity_curve=sample_equity_curve,
            trades=sample_trades,
            initial_capital=10000
        )

        visualizer = PerformanceVisualizer(analyzer)

        # Debe retornar una figura
        import matplotlib
        matplotlib.use('Agg')  # Backend sin GUI para tests

        fig = visualizer.plot_comprehensive_dashboard()

        assert fig is not None

        # Limpiar
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_rolling_metrics(self, sample_equity_curve, sample_trades):
        """Test graficos de metricas rodantes"""
        # Necesitamos suficientes datos
        np.random.seed(42)
        long_equity = list(10000 * np.cumprod(1 + np.random.normal(0.001, 0.02, 200)))

        analyzer = PerformanceAnalyzer(
            equity_curve=long_equity,
            trades=sample_trades,
            initial_capital=10000
        )

        visualizer = PerformanceVisualizer(analyzer)

        import matplotlib
        matplotlib.use('Agg')

        fig = visualizer.plot_rolling_metrics(window=30)

        assert fig is not None

        import matplotlib.pyplot as plt
        plt.close('all')

    def test_plot_trade_analysis(self, sample_equity_curve, sample_trades):
        """Test analisis de trades"""
        analyzer = PerformanceAnalyzer(
            equity_curve=sample_equity_curve,
            trades=sample_trades,
            initial_capital=10000
        )

        visualizer = PerformanceVisualizer(analyzer)

        import matplotlib
        matplotlib.use('Agg')

        fig = visualizer.plot_trade_analysis()

        assert fig is not None

        import matplotlib.pyplot as plt
        plt.close('all')

    def test_plot_risk_analysis(self, sample_equity_curve, sample_trades):
        """Test analisis de riesgo"""
        # Necesitamos suficientes datos
        np.random.seed(42)
        long_equity = list(10000 * np.cumprod(1 + np.random.normal(0.001, 0.02, 200)))

        analyzer = PerformanceAnalyzer(
            equity_curve=long_equity,
            trades=sample_trades,
            initial_capital=10000
        )

        visualizer = PerformanceVisualizer(analyzer)

        import matplotlib
        matplotlib.use('Agg')

        fig = visualizer.plot_risk_analysis()

        assert fig is not None

        import matplotlib.pyplot as plt
        plt.close('all')

    def test_visualizer_empty_trades(self, sample_equity_curve):
        """Test visualizador con trades vacios"""
        empty_trades = pd.DataFrame(columns=[
            'entry_time', 'exit_time', 'entry_price', 'exit_price',
            'quantity', 'pnl', 'return_pct', 'reason'
        ])

        analyzer = PerformanceAnalyzer(
            equity_curve=sample_equity_curve,
            trades=empty_trades,
            initial_capital=10000
        )

        visualizer = PerformanceVisualizer(analyzer)

        import matplotlib
        matplotlib.use('Agg')

        # No debe fallar con trades vacios
        fig = visualizer.plot_comprehensive_dashboard()
        assert fig is not None

        import matplotlib.pyplot as plt
        plt.close('all')


class TestPerformanceEdgeCases:
    """Tests para casos limite de performance"""

    def test_constant_equity(self):
        """Test con equity constante (sin variacion)"""
        constant_equity = [10000] * 100
        trades = pd.DataFrame([{
            'entry_time': datetime(2024, 1, 1),
            'exit_time': datetime(2024, 1, 2),
            'entry_price': 100, 'exit_price': 100,
            'quantity': 10, 'pnl': 0, 'return_pct': 0.0, 'reason': 'Flat'
        }])

        analyzer = PerformanceAnalyzer(
            equity_curve=constant_equity,
            trades=trades,
            initial_capital=10000
        )

        metrics = analyzer.calculate_all_metrics()

        # Sin variacion, volatilidad y sharpe deben ser 0
        assert metrics['volatility_pct'] == 0
        assert metrics['total_return_pct'] == 0

    def test_very_short_equity(self):
        """Test con equity muy corta"""
        short_equity = [10000, 10100]
        trades = pd.DataFrame([{
            'entry_time': datetime(2024, 1, 1),
            'exit_time': datetime(2024, 1, 2),
            'entry_price': 100, 'exit_price': 101,
            'quantity': 100, 'pnl': 100, 'return_pct': 1.0, 'reason': 'TP'
        }])

        analyzer = PerformanceAnalyzer(
            equity_curve=short_equity,
            trades=trades,
            initial_capital=10000
        )

        # No debe fallar con datos minimos
        metrics = analyzer.calculate_all_metrics()
        assert metrics['total_return_pct'] == 1.0

    def test_negative_pnl_trades(self):
        """Test con trades que resultan en perdida total"""
        losing_equity = [10000, 9000, 8000, 7000, 6000]
        trades = pd.DataFrame([
            {
                'entry_time': datetime(2024, 1, i),
                'exit_time': datetime(2024, 1, i+1),
                'entry_price': 100, 'exit_price': 90,
                'quantity': 10, 'pnl': -100, 'return_pct': -10.0, 'reason': 'SL'
            }
            for i in range(1, 5)
        ])

        analyzer = PerformanceAnalyzer(
            equity_curve=losing_equity,
            trades=trades,
            initial_capital=10000
        )

        metrics = analyzer.calculate_all_metrics()

        assert metrics['total_return_pct'] < 0
        assert metrics['win_rate_pct'] == 0
