"""
Ejemplo: Comparación de Métodos de Optimización
Compara Grid Search, Random Search, Bayesian y Genetic Algorithm
"""

import pandas as pd
import numpy as np
import time
import sys
sys.path.append('..')

from src.strategy import MovingAverageCrossoverStrategy, StrategyConfig
from src.optimizer import StrategyOptimizer


def generate_sample_data(n_periods=800):
    """Genera datos de muestra"""
    dates = pd.date_range(start='2024-01-01', periods=n_periods, freq='1H')
    np.random.seed(42)
    
    close_prices = 100 + np.random.randn(n_periods).cumsum()
    
    data = pd.DataFrame({
        'open': close_prices + np.random.randn(n_periods) * 0.5,
        'high': close_prices + np.abs(np.random.randn(n_periods)),
        'low': close_prices - np.abs(np.random.randn(n_periods)),
        'close': close_prices,
        'volume': np.random.randint(1000, 10000, n_periods)
    }, index=dates)
    
    data['high'] = data[['open', 'close', 'high']].max(axis=1)
    data['low'] = data[['open', 'close', 'low']].min(axis=1)
    
    return data


def compare_optimization_methods():
    """Compara diferentes métodos de optimización"""
    print("="*70)
    print("COMPARACION DE METODOS DE OPTIMIZACION")
    print("="*70)
    
    # Preparar datos
    print("\n[1] Preparando datos...")
    data = generate_sample_data()
    print(f"  Datos: {len(data)} períodos")
    
    # Configuración
    config = StrategyConfig(
        symbol='BTC/USD',
        timeframe='1H',
        initial_capital=10000,
        risk_per_trade=2.0,
        max_positions=3,
        commission=0.1,
        slippage=0.05
    )
    
    # Crear optimizador
    optimizer = StrategyOptimizer(
        strategy_class=MovingAverageCrossoverStrategy,
        data=data,
        config_template=config,
        objective_metric='sharpe_ratio',
        n_jobs=1
    )
    
    # Definir espacio de parámetros (reducido para demo)
    optimizer.add_parameter('fast_period', 'int', low=5, high=15, step=5)
    optimizer.add_parameter('slow_period', 'int', low=20, high=40, step=10)
    
    results = {}
    
    # METODO 1: Grid Search
    print("\n[2] Grid Search...")
    print("  Evaluará TODAS las combinaciones posibles")
    start = time.time()
    result_grid = optimizer.grid_search(verbose=False)
    results['Grid Search'] = {
        'result': result_grid,
        'time': time.time() - start
    }
    print(f"  Completado en {results['Grid Search']['time']:.2f}s")
    print(f"  Mejor Sharpe: {result_grid.best_score:.4f}")
    
    # METODO 2: Random Search
    print("\n[3] Random Search...")
    print("  Evaluará 20 combinaciones aleatorias")
    start = time.time()
    result_random = optimizer.random_search(n_iter=20, verbose=False)
    results['Random Search'] = {
        'result': result_random,
        'time': time.time() - start
    }
    print(f"  Completado en {results['Random Search']['time']:.2f}s")
    print(f"  Mejor Sharpe: {result_random.best_score:.4f}")
    
    # METODO 3: Bayesian Optimization (si está disponible)
    try:
        from skopt import gp_minimize
        print("\n[4] Bayesian Optimization...")
        print("  Evaluará 20 combinaciones de forma inteligente")
        start = time.time()
        result_bayes = optimizer.bayesian_optimization(n_calls=20, verbose=False)
        results['Bayesian Opt'] = {
            'result': result_bayes,
            'time': time.time() - start
        }
        print(f"  Completado en {results['Bayesian Opt']['time']:.2f}s")
        print(f"  Mejor Sharpe: {result_bayes.best_score:.4f}")
    except ImportError:
        print("\n[4] Bayesian Optimization no disponible")
        print("  Instalar con: pip install scikit-optimize")
    
    # METODO 4: Genetic Algorithm
    print("\n[5] Genetic Algorithm...")
    print("  Población: 10, Generaciones: 3")
    start = time.time()
    result_genetic = optimizer.genetic_algorithm(
        population_size=10,
        max_generations=3,
        verbose=False
    )
    results['Genetic Algo'] = {
        'result': result_genetic,
        'time': time.time() - start
    }
    print(f"  Completado en {results['Genetic Algo']['time']:.2f}s")
    print(f"  Mejor Sharpe: {result_genetic.best_score:.4f}")
    
    # TABLA COMPARATIVA
    print("\n" + "="*70)
    print("TABLA COMPARATIVA")
    print("="*70)
    print(f"\n{'Método':<20} {'Iteraciones':<15} {'Tiempo (s)':<15} {'Mejor Score':<15}")
    print("-"*70)
    
    for method_name, data in results.items():
        result = data['result']
        print(f"{method_name:<20} {result.iterations:<15} "
              f"{data['time']:<15.2f} {result.best_score:<15.4f}")
    
    # GANADOR
    print("\n" + "="*70)
    best_method = max(results.items(), key=lambda x: x[1]['result'].best_score)
    print(f"GANADOR: {best_method[0]}")
    print(f"  Mejor Score: {best_method[1]['result'].best_score:.4f}")
    print(f"  Parametros: {best_method[1]['result'].best_params}")
    
    # RECOMENDACIONES
    print("\n" + "="*70)
    print("RECOMENDACIONES DE USO")
    print("="*70)
    
    print("\n[Grid Search]")
    print("  Usar cuando: 2-3 parámetros, espacio pequeño")
    print("  Ventaja: Garantiza encontrar el óptimo")
    print("  Desventaja: Muy lento con muchos parámetros")
    
    print("\n[Random Search]")
    print("  Usar cuando: Exploración rápida inicial")
    print("  Ventaja: Rápido, simple")
    print("  Desventaja: No hay garantías")
    
    print("\n[Bayesian Optimization]")
    print("  Usar cuando: Presupuesto limitado de evaluaciones")
    print("  Ventaja: Muy eficiente")
    print("  Desventaja: Requiere librería adicional")
    
    print("\n[Genetic Algorithm]")
    print("  Usar cuando: Espacios complejos, 4-8 parámetros")
    print("  Ventaja: Robusto contra óptimos locales")
    print("  Desventaja: Requiere muchas evaluaciones")
    
    print("\n" + "="*70)
    print("WORKFLOW RECOMENDADO:")
    print("="*70)
    print("  1. Random Search (100 iter) - Exploración inicial")
    print("  2. Bayesian Opt (50 calls) - Refinamiento")
    print("  3. Walk Forward - Validación final")
    print("="*70)


if __name__ == "__main__":
    compare_optimization_methods()
