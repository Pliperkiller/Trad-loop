"""
Ejemplo Completo de Workflow
Demuestra el uso completo del sistema: datos → estrategia → backtest → análisis → optimización
"""

import pandas as pd
import numpy as np
from datetime import datetime
import sys
sys.path.append('..')

from src.strategy import MovingAverageCrossoverStrategy, StrategyConfig
from src.performance import PerformanceAnalyzer, PerformanceVisualizer
from src.optimizer import StrategyOptimizer


def generate_sample_data(n_periods=1000):
    """Genera datos de muestra para demostración"""
    dates = pd.date_range(start='2023-01-01', periods=n_periods, freq='1H')
    
    # Simular precios con tendencia y ruido
    np.random.seed(42)
    trend = np.linspace(0, 50, n_periods)
    noise = np.random.randn(n_periods).cumsum() * 2
    close_prices = 100 + trend + noise
    
    data = pd.DataFrame({
        'open': close_prices + np.random.randn(n_periods) * 0.5,
        'high': close_prices + np.abs(np.random.randn(n_periods)) * 1.5,
        'low': close_prices - np.abs(np.random.randn(n_periods)) * 1.5,
        'close': close_prices,
        'volume': np.random.randint(1000, 10000, n_periods)
    }, index=dates)
    
    # Asegurar que high y low sean correctos
    data['high'] = data[['open', 'close', 'high']].max(axis=1)
    data['low'] = data[['open', 'close', 'low']].min(axis=1)
    
    return data


def main():
    print("="*70)
    print("STRATEGY TRADER - WORKFLOW COMPLETO")
    print("="*70)
    
    # PASO 1: Generar/Cargar Datos
    print("\n[PASO 1] Cargando datos...")
    data = generate_sample_data(n_periods=1000)
    print(f"  Datos cargados: {len(data)} períodos")
    print(f"  Rango: {data.index[0]} a {data.index[-1]}")
    
    # PASO 2: Configurar Estrategia
    print("\n[PASO 2] Configurando estrategia...")
    config = StrategyConfig(
        symbol='BTC/USD',
        timeframe='1H',
        initial_capital=10000,
        risk_per_trade=2.0,
        max_positions=3,
        commission=0.1,
        slippage=0.05
    )
    
    strategy = MovingAverageCrossoverStrategy(
        config=config,
        fast_period=10,
        slow_period=30,
        rsi_period=14
    )
    print("  Estrategia: Moving Average Crossover + RSI")
    print(f"  Capital inicial: ${config.initial_capital:,.2f}")
    
    # PASO 3: Ejecutar Backtest
    print("\n[PASO 3] Ejecutando backtest...")
    strategy.load_data(data)
    strategy.backtest()
    
    metrics = strategy.get_performance_metrics()
    print(f"  Trades totales: {metrics.get('total_trades', 0)}")
    print(f"  Win rate: {metrics.get('win_rate', 0):.2f}%")
    print(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
    print(f"  Retorno total: {metrics.get('total_return_pct', 0):.2f}%")
    
    # PASO 4: Análisis de Performance
    print("\n[PASO 4] Analizando performance...")
    
    analyzer = PerformanceAnalyzer(
        equity_curve=strategy.equity_curve,
        trades=pd.DataFrame(strategy.closed_trades),
        initial_capital=config.initial_capital
    )
    
    print("\n  REPORTE DETALLADO:")
    analyzer.print_report()
    
    # PASO 5: Visualizaciones
    print("\n[PASO 5] Generando visualizaciones...")
    
    visualizer = PerformanceVisualizer(analyzer)
    
    print("  Generando dashboard completo...")
    visualizer.plot_comprehensive_dashboard()
    
    print("  Generando análisis de métricas rodantes...")
    visualizer.plot_rolling_metrics(window=50)
    
    print("  Generando análisis de trades...")
    visualizer.plot_trade_analysis()
    
    # PASO 6: Optimización de Parámetros
    print("\n[PASO 6] Optimizando parámetros...")
    
    optimizer = StrategyOptimizer(
        strategy_class=MovingAverageCrossoverStrategy,
        data=data,
        config_template=config,
        objective_metric='sharpe_ratio',
        n_jobs=1
    )
    
    # Definir espacio de parámetros
    optimizer.add_parameter('fast_period', 'int', low=5, high=20, step=5)
    optimizer.add_parameter('slow_period', 'int', low=20, high=50, step=10)
    optimizer.add_parameter('rsi_period', 'int', low=10, high=20, step=5)
    
    # Optimización con Random Search (rápido)
    print("\n  Ejecutando Random Search...")
    result = optimizer.random_search(n_iter=30, verbose=True)
    result.print_summary()
    
    # PASO 7: Validación Walk Forward (Opcional)
    print("\n[PASO 7] Validación Walk Forward...")
    print("  Esta puede tomar varios minutos...")
    
    wf_result = optimizer.walk_forward_optimization(
        optimization_method='random',
        n_splits=3,
        train_size=0.6,
        n_iter=20  # Reducido para demo
    )
    
    # PASO 8: Recomendaciones Finales
    print("\n" + "="*70)
    print("RESUMEN Y RECOMENDACIONES")
    print("="*70)
    
    print("\n[Parámetros Originales]")
    print(f"  Fast Period: 10, Slow Period: 30, RSI Period: 14")
    print(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
    
    print("\n[Parámetros Optimizados]")
    for param, value in result.best_params.items():
        print(f"  {param}: {value}")
    print(f"  Sharpe Ratio: {result.best_score:.2f}")
    
    mejora = ((result.best_score - metrics.get('sharpe_ratio', 0)) / 
              abs(metrics.get('sharpe_ratio', 0.001)) * 100)
    print(f"\n  Mejora: {mejora:.1f}%")
    
    print("\n[Próximos Pasos Recomendados]")
    print("  1. Ejecutar Walk Forward más extenso (n_splits=5)")
    print("  2. Probar con datos de diferentes períodos")
    print("  3. Validar con datos out-of-sample")
    print("  4. Implementar en paper trading")
    print("  5. Monitorear performance en tiempo real")
    
    print("\n" + "="*70)
    print("WORKFLOW COMPLETADO EXITOSAMENTE")
    print("="*70)


if __name__ == "__main__":
    main()
