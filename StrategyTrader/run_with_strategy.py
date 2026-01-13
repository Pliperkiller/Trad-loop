"""
Script para ejecutar la API con una estrategia de ejemplo registrada.

Uso:
    python run_with_strategy.py

Esto:
1. Carga datos CSV de ejemplo
2. Crea y ejecuta una estrategia de backtest
3. Registra la estrategia en la API
4. Inicia el servidor en http://localhost:8000
"""

import pandas as pd
import uvicorn
from pathlib import Path

# Importar componentes de Trad-loop
from src.strategy import MovingAverageCrossoverStrategy, StrategyConfig
from src.api import app, register_strategy


def load_csv_data(csv_path: str) -> pd.DataFrame:
    """Carga datos OHLCV desde un archivo CSV"""
    df = pd.read_csv(csv_path)

    # Detectar columna de timestamp
    time_cols = ['timestamp', 'time', 'date', 'datetime', 'Date', 'Timestamp']
    time_col = None
    for col in time_cols:
        if col in df.columns:
            time_col = col
            break

    if time_col is None:
        # Usar primera columna como índice si no se encuentra
        time_col = df.columns[0]

    # Convertir a datetime y usar como índice
    df[time_col] = pd.to_datetime(df[time_col])
    df.set_index(time_col, inplace=True)

    # Normalizar nombres de columnas a minúsculas
    df.columns = df.columns.str.lower()

    # Asegurar que existan las columnas requeridas
    required = ['open', 'high', 'low', 'close', 'volume']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Columna '{col}' no encontrada en el CSV")

    return df


def create_and_run_strategy(data: pd.DataFrame, strategy_id: str) -> None:
    """Crea una estrategia, ejecuta backtest y la registra en la API"""

    # Configuración de la estrategia
    config = StrategyConfig(
        symbol='BTC/USD',
        timeframe='1H',
        initial_capital=10000.0,
        risk_per_trade=2.0,      # 2% de riesgo por trade
        max_positions=3,          # Máximo 3 posiciones abiertas
        commission=0.1,           # 0.1% de comisión
        slippage=0.05             # 0.05% de slippage
    )

    # Crear estrategia de cruce de medias móviles
    strategy = MovingAverageCrossoverStrategy(
        config=config,
        fast_period=10,   # EMA rápida de 10 períodos
        slow_period=30,   # EMA lenta de 30 períodos
        rsi_period=14     # RSI de 14 períodos
    )

    # Cargar datos y ejecutar backtest
    strategy.load_data(data)
    strategy.backtest()

    # Mostrar resultados
    metrics = strategy.get_performance_metrics()
    print(f"\n{'='*50}")
    print(f"Estrategia: {strategy_id}")
    print(f"{'='*50}")
    print(f"Total trades:    {metrics.get('total_trades', 0)}")
    print(f"Win rate:        {metrics.get('win_rate', 0):.1f}%")
    print(f"Profit factor:   {metrics.get('profit_factor', 0):.2f}")
    print(f"Total return:    {metrics.get('total_return_pct', 0):.2f}%")
    print(f"Max drawdown:    {metrics.get('max_drawdown_pct', 0):.2f}%")
    print(f"Sharpe ratio:    {metrics.get('sharpe_ratio', 0):.2f}")
    print(f"Capital final:   ${metrics.get('final_capital', 0):,.2f}")
    print(f"{'='*50}\n")

    # Registrar en la API
    register_strategy(strategy_id, strategy)
    print(f"Estrategia '{strategy_id}' registrada en la API")


def main():
    # Buscar archivo CSV de ejemplo
    csv_paths = [
        Path("../csv_charts/btc_futures_1h_2025.csv"),
        Path("csv_charts/btc_futures_1h_2025.csv"),
        Path("../csv_charts"),
    ]

    csv_file = None
    for path in csv_paths:
        if path.exists():
            if path.is_dir():
                # Buscar primer CSV en el directorio
                csvs = list(path.glob("*.csv"))
                if csvs:
                    csv_file = csvs[0]
                    break
            else:
                csv_file = path
                break

    if csv_file is None:
        print("No se encontró archivo CSV de ejemplo.")
        print("Creando datos sintéticos para demostración...")

        # Crear datos sintéticos
        import numpy as np
        dates = pd.date_range(start='2024-01-01', periods=500, freq='1H')
        np.random.seed(42)

        price = 40000
        prices = []
        for _ in range(500):
            price = price * (1 + np.random.randn() * 0.01)
            prices.append(price)

        data = pd.DataFrame({
            'open': prices,
            'high': [p * (1 + abs(np.random.randn()) * 0.005) for p in prices],
            'low': [p * (1 - abs(np.random.randn()) * 0.005) for p in prices],
            'close': [p * (1 + np.random.randn() * 0.002) for p in prices],
            'volume': [np.random.randint(100, 10000) for _ in prices],
        }, index=dates)
    else:
        print(f"Cargando datos desde: {csv_file}")
        data = load_csv_data(str(csv_file))

    print(f"Datos cargados: {len(data)} barras")
    print(f"Rango: {data.index[0]} a {data.index[-1]}")

    # Crear y registrar estrategia
    create_and_run_strategy(data, "demo-strategy")

    # Iniciar servidor API
    print("\nIniciando servidor API...")
    print("Endpoints disponibles:")
    print("  - http://localhost:8000/docs          (Documentación)")
    print("  - http://localhost:8000/api/v1/strategies")
    print("  - http://localhost:8000/api/v1/trades/demo-strategy")
    print("  - http://localhost:8000/api/v1/performance/demo-strategy")
    print("\nEn fyGraphr, usa 'demo-strategy' como Strategy ID\n")

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
