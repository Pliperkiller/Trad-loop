"""
Sistema de Trading con Indicadores Técnicos
Plantilla base para estrategias de inversión
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class TradeSignal:
    """Estructura para señales de trading"""
    timestamp: datetime
    signal: str  # 'BUY', 'SELL', 'HOLD'
    price: float
    confidence: float  # 0 a 1
    indicators: Dict[str, float]
    
@dataclass
class Position:
    """Estructura para posiciones abiertas"""
    entry_time: datetime
    entry_price: float
    quantity: float
    position_type: str  # 'LONG' o 'SHORT'
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

@dataclass
class StrategyConfig:
    """Configuración de la estrategia"""
    symbol: str
    timeframe: str
    initial_capital: float
    risk_per_trade: float  # Porcentaje del capital
    max_positions: int
    commission: float  # Porcentaje de comisión
    slippage: float  # Slippage estimado en porcentaje


class TechnicalIndicators:
    """Clase con métodos estáticos para calcular indicadores técnicos"""
    
    @staticmethod
    def sma(data: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average"""
        return data.rolling(window=period).mean()
    
    @staticmethod
    def ema(data: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average"""
        return data.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD (Moving Average Convergence Divergence)"""
        ema_fast = data.ewm(span=fast, adjust=False).mean()
        ema_slow = data.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def bollinger_bands(data: pd.Series, period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Bandas de Bollinger"""
        sma = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, sma, lower_band
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average True Range (volatilidad)"""
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(window=period).mean()
    
    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> Tuple[pd.Series, pd.Series]:
        """Oscilador Estocástico"""
        lowest_low = low.rolling(window=period).min()
        highest_high = high.rolling(window=period).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=3).mean()
        return k_percent, d_percent


class TradingStrategy(ABC):
    """Clase base abstracta para todas las estrategias de trading"""
    
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.data: Optional[pd.DataFrame] = None
        self.positions: List[Position] = []
        self.closed_trades: List[Dict] = []
        self.capital = config.initial_capital
        self.equity_curve: List[float] = [config.initial_capital]
        
    def load_data(self, data: pd.DataFrame):
        """Carga y valida los datos de mercado"""
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"El DataFrame debe contener: {required_columns}")
        
        self.data = data.copy()
        self.data.index = pd.to_datetime(self.data.index)
        
    @abstractmethod
    def calculate_indicators(self):
        """Calcula todos los indicadores técnicos necesarios"""
        pass
    
    @abstractmethod
    def generate_signals(self) -> pd.Series:
        """Genera señales de compra/venta basadas en los indicadores"""
        pass
    
    def calculate_position_size(self, price: float, stop_loss: float) -> float:
        """Calcula el tamaño de posición basado en riesgo"""
        risk_amount = self.capital * (self.config.risk_per_trade / 100)
        risk_per_unit = abs(price - stop_loss)
        
        if risk_per_unit == 0:
            return 0
        
        position_size = risk_amount / risk_per_unit
        max_position_value = self.capital * 0.95
        position_size = min(position_size, max_position_value / price)
        
        return position_size
    
    def open_position(self, signal: TradeSignal, stop_loss: float, take_profit: float):
        """Abre una nueva posición"""
        if len(self.positions) >= self.config.max_positions:
            return
        
        position_size = self.calculate_position_size(signal.price, stop_loss)
        
        if position_size > 0:
            position = Position(
                entry_time=signal.timestamp,
                entry_price=signal.price,
                quantity=position_size,
                position_type=signal.signal,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            self.positions.append(position)
            
            cost = position_size * signal.price * (1 + self.config.commission / 100)
            self.capital -= cost
    
    def close_position(self, position: Position, exit_price: float, exit_time: datetime, reason: str):
        """Cierra una posición existente"""
        if position.position_type == 'LONG':
            pnl = (exit_price - position.entry_price) * position.quantity
        else:
            pnl = (position.entry_price - exit_price) * position.quantity
        
        commission = (position.entry_price + exit_price) * position.quantity * (self.config.commission / 100)
        net_pnl = pnl - commission
        
        self.closed_trades.append({
            'entry_time': position.entry_time,
            'exit_time': exit_time,
            'entry_price': position.entry_price,
            'exit_price': exit_price,
            'quantity': position.quantity,
            'position_type': position.position_type,
            'pnl': net_pnl,
            'return_pct': (net_pnl / (position.entry_price * position.quantity)) * 100,
            'reason': reason
        })
        
        self.capital += (position.quantity * exit_price * (1 - self.config.commission / 100))
        self.positions.remove(position)
    
    def backtest(self):
        """Ejecuta el backtest de la estrategia"""
        if self.data is None:
            raise ValueError("Primero debes cargar los datos con load_data()")
        
        self.calculate_indicators()
        signals = self.generate_signals()
        
        for i in range(len(self.data)):
            current_bar = self.data.iloc[i]
            current_time = self.data.index[i]
            current_price = current_bar['close']
            
            for position in self.positions.copy():
                if position.stop_loss and current_price <= position.stop_loss:
                    self.close_position(position, position.stop_loss, current_time, 'Stop Loss')
                elif position.take_profit and current_price >= position.take_profit:
                    self.close_position(position, position.take_profit, current_time, 'Take Profit')
            
            if pd.notna(signals.iloc[i]):
                signal_type = signals.iloc[i]
                
                if signal_type == 'BUY' and len(self.positions) < self.config.max_positions:
                    signal = TradeSignal(
                        timestamp=current_time,
                        signal='LONG',
                        price=current_price,
                        confidence=1.0,
                        indicators={}
                    )
                    stop_loss = current_price * 0.98
                    take_profit = current_price * 1.04
                    self.open_position(signal, stop_loss, take_profit)
                
                elif signal_type == 'SELL' and len(self.positions) > 0:
                    for position in self.positions.copy():
                        self.close_position(position, current_price, current_time, 'Signal Exit')
            
            total_equity = self.capital
            for position in self.positions:
                total_equity += position.quantity * current_price
            self.equity_curve.append(total_equity)
    
    def get_performance_metrics(self) -> Dict:
        """Calcula métricas de rendimiento"""
        if not self.closed_trades:
            return {}
        
        trades_df = pd.DataFrame(self.closed_trades)
        
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] < 0])
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
        
        profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades)) if losing_trades > 0 else float('inf')
        
        total_return = ((self.equity_curve[-1] - self.config.initial_capital) / self.config.initial_capital) * 100
        
        equity_series = pd.Series(self.equity_curve)
        rolling_max = equity_series.expanding().max()
        drawdown = (equity_series - rolling_max) / rolling_max * 100
        max_drawdown = drawdown.min()
        
        returns = equity_series.pct_change().dropna()
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if len(returns) > 0 and returns.std() > 0 else 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_return_pct': total_return,
            'max_drawdown_pct': max_drawdown,
            'final_capital': self.equity_curve[-1],
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'sharpe_ratio': sharpe_ratio
        }


class MovingAverageCrossoverStrategy(TradingStrategy):
    """
    Estrategia de cruce de medias móviles con filtro RSI
    """
    
    def __init__(self, config: StrategyConfig, fast_period: int = 10, slow_period: int = 30, rsi_period: int = 14):
        super().__init__(config)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.rsi_period = rsi_period
        
    def calculate_indicators(self):
        """Calcula EMAs y RSI"""
        self.data['ema_fast'] = TechnicalIndicators.ema(self.data['close'], self.fast_period)
        self.data['ema_slow'] = TechnicalIndicators.ema(self.data['close'], self.slow_period)
        self.data['rsi'] = TechnicalIndicators.rsi(self.data['close'], self.rsi_period)
        
    def generate_signals(self) -> pd.Series:
        """Genera señales basadas en cruces de EMAs y RSI"""
        signals = pd.Series(index=self.data.index, dtype=object)
        
        self.data['ema_cross'] = np.where(
            self.data['ema_fast'] > self.data['ema_slow'], 1, -1
        )
        self.data['ema_cross_signal'] = self.data['ema_cross'].diff()
        
        for i in range(1, len(self.data)):
            if (self.data['ema_cross_signal'].iloc[i] == 2 and
                30 < self.data['rsi'].iloc[i] < 70):
                signals.iloc[i] = 'BUY'
            
            elif (self.data['ema_cross_signal'].iloc[i] == -2 or
                  self.data['rsi'].iloc[i] > 80):
                signals.iloc[i] = 'SELL'
        
        return signals
