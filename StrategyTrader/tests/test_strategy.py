"""
Tests para el modulo de estrategias de trading
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from src.strategy import (
    TechnicalIndicators,
    TradingStrategy,
    MovingAverageCrossoverStrategy,
    StrategyConfig,
    TradeSignal,
    Position
)


class TestTechnicalIndicators:
    """Tests para la clase TechnicalIndicators"""

    def test_sma_calculation(self, sample_ohlcv_data):
        """Test que SMA se calcula correctamente"""
        close = sample_ohlcv_data['close']
        period = 20

        sma = TechnicalIndicators.sma(close, period)

        # Los primeros (period-1) valores deben ser NaN
        assert sma.iloc[:period-1].isna().all()

        # A partir del periodo, debe haber valores
        assert sma.iloc[period:].notna().all()

        # Verificar calculo manual para un punto
        expected = close.iloc[:period].mean()
        assert abs(sma.iloc[period-1] - expected) < 0.0001

    def test_ema_calculation(self, sample_ohlcv_data):
        """Test que EMA se calcula correctamente"""
        close = sample_ohlcv_data['close']
        period = 20

        ema = TechnicalIndicators.ema(close, period)

        # EMA debe tener valores para todos los puntos
        assert ema.notna().all()

        # EMA debe ser diferente de SMA (excepto quizas el primer punto)
        sma = TechnicalIndicators.sma(close, period)
        assert not ema.equals(sma)

    def test_rsi_range(self, sample_ohlcv_data):
        """Test que RSI esta siempre entre 0 y 100"""
        close = sample_ohlcv_data['close']
        period = 14

        rsi = TechnicalIndicators.rsi(close, period)

        # Filtrar NaN
        rsi_valid = rsi.dropna()

        # RSI debe estar entre 0 y 100
        assert (rsi_valid >= 0).all()
        assert (rsi_valid <= 100).all()

    def test_rsi_extreme_values(self):
        """Test RSI en condiciones extremas"""
        # Solo subidas -> RSI debe acercarse a 100
        rising = pd.Series([100 + i for i in range(50)])
        rsi_up = TechnicalIndicators.rsi(rising, 14)
        assert rsi_up.iloc[-1] > 70

        # Solo bajadas -> RSI debe acercarse a 0
        falling = pd.Series([100 - i for i in range(50)])
        rsi_down = TechnicalIndicators.rsi(falling, 14)
        assert rsi_down.iloc[-1] < 30

    def test_macd_components(self, sample_ohlcv_data):
        """Test que MACD retorna 3 componentes"""
        close = sample_ohlcv_data['close']

        macd_line, signal_line, histogram = TechnicalIndicators.macd(close)

        # Debe retornar 3 series
        assert isinstance(macd_line, pd.Series)
        assert isinstance(signal_line, pd.Series)
        assert isinstance(histogram, pd.Series)

        # Todas deben tener la misma longitud
        assert len(macd_line) == len(close)
        assert len(signal_line) == len(close)
        assert len(histogram) == len(close)

        # Histograma = MACD - Signal
        diff = (macd_line - signal_line - histogram).dropna()
        assert (abs(diff) < 0.0001).all()

    def test_bollinger_bands_order(self, sample_ohlcv_data):
        """Test que las bandas de Bollinger estan en orden correcto"""
        close = sample_ohlcv_data['close']

        upper, middle, lower = TechnicalIndicators.bollinger_bands(close)

        # Filtrar NaN
        valid_idx = upper.notna() & middle.notna() & lower.notna()

        # Upper > Middle > Lower
        assert (upper[valid_idx] >= middle[valid_idx]).all()
        assert (middle[valid_idx] >= lower[valid_idx]).all()

    def test_atr_positive(self, sample_ohlcv_data):
        """Test que ATR es siempre positivo"""
        high = sample_ohlcv_data['high']
        low = sample_ohlcv_data['low']
        close = sample_ohlcv_data['close']

        atr = TechnicalIndicators.atr(high, low, close)

        # ATR debe ser positivo
        atr_valid = atr.dropna()
        assert (atr_valid >= 0).all()

    def test_stochastic_range(self, sample_ohlcv_data):
        """Test que Stochastic esta entre 0 y 100"""
        high = sample_ohlcv_data['high']
        low = sample_ohlcv_data['low']
        close = sample_ohlcv_data['close']

        k_pct, d_pct = TechnicalIndicators.stochastic(high, low, close)

        # Filtrar NaN
        k_valid = k_pct.dropna()
        d_valid = d_pct.dropna()

        # Debe estar entre 0 y 100
        assert (k_valid >= 0).all() and (k_valid <= 100).all()
        assert (d_valid >= 0).all() and (d_valid <= 100).all()


class TestStrategyConfig:
    """Tests para StrategyConfig"""

    def test_config_creation(self):
        """Test creacion de configuracion"""
        config = StrategyConfig(
            symbol='BTC/USDT',
            timeframe='1h',
            initial_capital=10000,
            risk_per_trade=2.0,
            max_positions=3,
            commission=0.1,
            slippage=0.05
        )

        assert config.symbol == 'BTC/USDT'
        assert config.initial_capital == 10000
        assert config.risk_per_trade == 2.0


class TestTradeSignal:
    """Tests para TradeSignal"""

    def test_signal_creation(self):
        """Test creacion de senal de trading"""
        signal = TradeSignal(
            timestamp=datetime.now(),
            signal='BUY',
            price=100.0,
            confidence=0.8,
            indicators={'rsi': 45.0, 'macd': 0.5}
        )

        assert signal.signal == 'BUY'
        assert signal.price == 100.0
        assert signal.confidence == 0.8


class TestPosition:
    """Tests para Position"""

    def test_position_creation(self):
        """Test creacion de posicion"""
        position = Position(
            entry_time=datetime.now(),
            entry_price=100.0,
            quantity=10.0,
            position_type='LONG',
            stop_loss=95.0,
            take_profit=110.0
        )

        assert position.entry_price == 100.0
        assert position.quantity == 10.0
        assert position.position_type == 'LONG'


class TestMovingAverageCrossoverStrategy:
    """Tests para la estrategia de cruce de medias moviles"""

    def test_strategy_initialization(self, sample_strategy_config):
        """Test inicializacion de estrategia"""
        strategy = MovingAverageCrossoverStrategy(
            sample_strategy_config,
            fast_period=10,
            slow_period=30,
            rsi_period=14
        )

        assert strategy.fast_period == 10
        assert strategy.slow_period == 30
        assert strategy.capital == sample_strategy_config.initial_capital

    def test_load_data(self, sample_strategy_config, sample_ohlcv_data):
        """Test carga de datos"""
        strategy = MovingAverageCrossoverStrategy(sample_strategy_config)
        strategy.load_data(sample_ohlcv_data)

        assert strategy.data is not None
        assert len(strategy.data) == len(sample_ohlcv_data)

    def test_load_data_invalid(self, sample_strategy_config):
        """Test que datos invalidos lanzan error"""
        strategy = MovingAverageCrossoverStrategy(sample_strategy_config)

        # DataFrame sin columnas requeridas
        invalid_data = pd.DataFrame({'price': [1, 2, 3]})

        with pytest.raises(ValueError):
            strategy.load_data(invalid_data)

    def test_calculate_indicators(self, sample_strategy_config, sample_ohlcv_data):
        """Test calculo de indicadores"""
        strategy = MovingAverageCrossoverStrategy(sample_strategy_config)
        strategy.load_data(sample_ohlcv_data)
        strategy.calculate_indicators()

        # Verificar que se crearon los indicadores
        assert 'ema_fast' in strategy.data.columns
        assert 'ema_slow' in strategy.data.columns
        assert 'rsi' in strategy.data.columns

    def test_generate_signals(self, sample_strategy_config, sample_ohlcv_data):
        """Test generacion de senales"""
        strategy = MovingAverageCrossoverStrategy(sample_strategy_config)
        strategy.load_data(sample_ohlcv_data)
        strategy.calculate_indicators()

        signals = strategy.generate_signals()

        # Debe ser una Serie con la misma longitud
        assert isinstance(signals, pd.Series)
        assert len(signals) == len(sample_ohlcv_data)

        # Solo debe contener BUY, SELL o NaN
        valid_signals = signals.dropna()
        assert all(s in ['BUY', 'SELL'] for s in valid_signals)

    def test_backtest_runs(self, sample_strategy_config, sample_ohlcv_data):
        """Test que backtest se ejecuta sin errores"""
        strategy = MovingAverageCrossoverStrategy(sample_strategy_config)
        strategy.load_data(sample_ohlcv_data)

        # No debe lanzar excepciones
        strategy.backtest()

        # Equity curve debe tener valores
        assert len(strategy.equity_curve) > 0

    def test_backtest_no_data_error(self, sample_strategy_config):
        """Test que backtest sin datos lanza error"""
        strategy = MovingAverageCrossoverStrategy(sample_strategy_config)

        with pytest.raises(ValueError):
            strategy.backtest()

    def test_performance_metrics(self, sample_strategy_config, trending_data):
        """Test calculo de metricas de performance"""
        strategy = MovingAverageCrossoverStrategy(
            sample_strategy_config,
            fast_period=5,
            slow_period=15
        )
        strategy.load_data(trending_data)
        strategy.backtest()

        metrics = strategy.get_performance_metrics()

        # Si hubo trades, debe haber metricas
        if strategy.closed_trades:
            assert 'total_trades' in metrics
            assert 'win_rate' in metrics
            assert 'sharpe_ratio' in metrics
            assert 'max_drawdown_pct' in metrics

    def test_position_sizing(self, sample_strategy_config, sample_ohlcv_data):
        """Test calculo de tamano de posicion"""
        strategy = MovingAverageCrossoverStrategy(sample_strategy_config)
        strategy.load_data(sample_ohlcv_data)

        price = 100.0
        stop_loss = 95.0

        position_size = strategy.calculate_position_size(price, stop_loss)

        # Position size debe ser positivo
        assert position_size > 0

        # Riesgo maximo = capital * risk_per_trade%
        max_risk = strategy.capital * (sample_strategy_config.risk_per_trade / 100)
        actual_risk = position_size * (price - stop_loss)
        assert actual_risk <= max_risk * 1.01  # Pequena tolerancia

    def test_open_close_position(self, sample_strategy_config, sample_ohlcv_data):
        """Test apertura y cierre de posiciones"""
        strategy = MovingAverageCrossoverStrategy(sample_strategy_config)
        strategy.load_data(sample_ohlcv_data)

        initial_capital = strategy.capital

        # Crear senal de compra
        signal = TradeSignal(
            timestamp=datetime.now(),
            signal='LONG',
            price=100.0,
            confidence=1.0,
            indicators={}
        )

        # Abrir posicion
        strategy.open_position(signal, stop_loss=95.0, take_profit=110.0)

        # Debe haber una posicion abierta
        assert len(strategy.positions) == 1

        # Capital debe haber disminuido
        assert strategy.capital < initial_capital

        # Cerrar posicion
        position = strategy.positions[0]
        strategy.close_position(position, exit_price=105.0, exit_time=datetime.now(), reason='Test')

        # No debe haber posiciones abiertas
        assert len(strategy.positions) == 0

        # Debe haber un trade cerrado
        assert len(strategy.closed_trades) == 1

    def test_max_positions_limit(self, sample_strategy_config, sample_ohlcv_data):
        """Test que no se excede el limite de posiciones"""
        sample_strategy_config.max_positions = 1
        strategy = MovingAverageCrossoverStrategy(sample_strategy_config)
        strategy.load_data(sample_ohlcv_data)

        signal = TradeSignal(
            timestamp=datetime.now(),
            signal='LONG',
            price=100.0,
            confidence=1.0,
            indicators={}
        )

        # Abrir primera posicion
        strategy.open_position(signal, stop_loss=95.0, take_profit=110.0)
        assert len(strategy.positions) == 1

        # Intentar abrir segunda posicion
        strategy.open_position(signal, stop_loss=95.0, take_profit=110.0)

        # Solo debe haber 1 posicion (max_positions=1)
        assert len(strategy.positions) == 1


class TestStrategyEdgeCases:
    """Tests para casos limite"""

    def test_empty_data(self, sample_strategy_config):
        """Test con DataFrame vacio"""
        strategy = MovingAverageCrossoverStrategy(sample_strategy_config)

        empty_data = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])

        strategy.load_data(empty_data)
        # No debe fallar con datos vacios

    def test_single_bar(self, sample_strategy_config):
        """Test con un solo bar de datos"""
        strategy = MovingAverageCrossoverStrategy(sample_strategy_config)

        single_bar = pd.DataFrame({
            'open': [100],
            'high': [101],
            'low': [99],
            'close': [100.5],
            'volume': [1000]
        }, index=pd.date_range('2024-01-01', periods=1, freq='1h'))

        strategy.load_data(single_bar)
        strategy.backtest()

        # No debe haber trades con un solo bar
        assert len(strategy.closed_trades) == 0

    def test_constant_prices(self, sample_strategy_config):
        """Test con precios constantes"""
        strategy = MovingAverageCrossoverStrategy(sample_strategy_config)

        n = 100
        constant_data = pd.DataFrame({
            'open': [100] * n,
            'high': [101] * n,
            'low': [99] * n,
            'close': [100] * n,
            'volume': [1000] * n
        }, index=pd.date_range('2024-01-01', periods=n, freq='1h'))

        strategy.load_data(constant_data)
        strategy.backtest()

        # Con precios constantes, no deberia haber cruces de EMAs
        # Por lo tanto, pocos o ningun trade
