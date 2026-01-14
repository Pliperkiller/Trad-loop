"""
Clases base para el sistema de trading.
Contiene dataclasses y la clase abstracta TradingStrategy.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


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


def validate_ohlcv_data(df: pd.DataFrame, strict: bool = True) -> Tuple[bool, List[str]]:
    """
    Valida la integridad de datos OHLCV.

    Args:
        df: DataFrame con datos OHLCV
        strict: Si True, cualquier error hace fallar la validación

    Returns:
        Tuple[bool, List[str]]: (es_válido, lista_de_errores)
    """
    errors = []

    # Verificar columnas requeridas
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        errors.append(f"Columnas faltantes: {missing}")
        return False, errors  # No podemos continuar sin columnas

    # Verificar DataFrame vacío
    if len(df) == 0:
        errors.append("DataFrame vacío")
        return False, errors

    # Verificar integridad OHLC
    invalid_high_low = (df['high'] < df['low']).sum()
    if invalid_high_low > 0:
        errors.append(f"Datos inválidos: high < low en {invalid_high_low} filas")

    invalid_high = ((df['high'] < df['open']) | (df['high'] < df['close'])).sum()
    if invalid_high > 0:
        errors.append(f"Datos inválidos: high no es el máximo en {invalid_high} filas")

    invalid_low = ((df['low'] > df['open']) | (df['low'] > df['close'])).sum()
    if invalid_low > 0:
        errors.append(f"Datos inválidos: low no es el mínimo en {invalid_low} filas")

    # Verificar valores no positivos en precios
    for col in ['open', 'high', 'low', 'close']:
        non_positive = (df[col] <= 0).sum()
        if non_positive > 0:
            errors.append(f"Valores no positivos en '{col}': {non_positive} filas")

    # Verificar volumen negativo
    negative_volume = (df['volume'] < 0).sum()
    if negative_volume > 0:
        errors.append(f"Volumen negativo en {negative_volume} filas")

    # Verificar NaN
    nan_counts = df[required_columns].isna().sum()
    nan_cols = nan_counts[nan_counts > 0]
    if len(nan_cols) > 0:
        errors.append(f"Valores NaN encontrados: {nan_cols.to_dict()}")

    # Verificar Inf
    numeric_cols = df[required_columns].select_dtypes(include=[np.number])
    inf_counts = np.isinf(numeric_cols).sum()
    inf_cols = inf_counts[inf_counts > 0]
    if len(inf_cols) > 0:
        errors.append(f"Valores Inf encontrados: {inf_cols.to_dict()}")

    is_valid = len(errors) == 0 if strict else not any("Columnas faltantes" in e or "DataFrame vacío" in e for e in errors)

    return is_valid, errors


def sanitize_ohlcv_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia datos OHLCV removiendo filas problemáticas.

    Args:
        df: DataFrame con datos OHLCV

    Returns:
        DataFrame limpio
    """
    df = df.copy()
    required_columns = ['open', 'high', 'low', 'close', 'volume']

    # Remover filas con NaN en columnas requeridas
    initial_len = len(df)
    df = df.dropna(subset=required_columns)

    # Remover filas con valores no positivos en precios
    for col in ['open', 'high', 'low', 'close']:
        df = df[df[col] > 0]

    # Remover filas con volumen negativo
    df = df[df['volume'] >= 0]

    # Remover filas con Inf
    numeric_cols = df[required_columns].select_dtypes(include=[np.number]).columns
    df = df[~np.isinf(df[numeric_cols]).any(axis=1)]

    # Corregir integridad OHLC: ajustar high/low si están mal
    df['high'] = df[['open', 'high', 'low', 'close']].max(axis=1)
    df['low'] = df[['open', 'high', 'low', 'close']].min(axis=1)

    removed = initial_len - len(df)
    if removed > 0:
        logger.warning(f"sanitize_ohlcv_data: removidas {removed} filas problemáticas")

    return df


class TradingStrategy(ABC):
    """
    Clase base abstracta para todas las estrategias de trading.

    Soporta dependency injection opcional para:
    - data_validator: Validador de datos OHLCV
    - position_sizer: Calculador de tamaño de posición

    Ejemplo con DI:
        from src.interfaces import DefaultDataValidator, FixedFractionalSizer

        strategy = MyStrategy(
            config,
            data_validator=DefaultDataValidator(strict=False),
            position_sizer=FixedFractionalSizer(risk_pct=1.5)
        )
    """

    def __init__(
        self,
        config: StrategyConfig,
        data_validator: Optional[object] = None,
        position_sizer: Optional[object] = None
    ):
        """
        Args:
            config: Configuración de la estrategia
            data_validator: Validador de datos opcional (debe tener métodos validate/sanitize)
            position_sizer: Position sizer opcional (debe tener método calculate_size)
        """
        self.config = config
        self.data: Optional[pd.DataFrame] = None
        self.positions: List[Position] = []
        self.closed_trades: List[Dict] = []
        self.capital = config.initial_capital
        self.equity_curve: List[float] = [config.initial_capital]

        # Dependency injection (opcional)
        self._data_validator = data_validator
        self._position_sizer = position_sizer

    def load_data(self, data: pd.DataFrame, validate: bool = True, auto_sanitize: bool = False):
        """
        Carga y valida los datos de mercado.

        Args:
            data: DataFrame con datos OHLCV
            validate: Si True, valida la integridad de los datos
            auto_sanitize: Si True, limpia automáticamente datos problemáticos

        Raises:
            ValueError: Si los datos no pasan la validación
        """
        if validate:
            # Usar validador inyectado si existe
            if self._data_validator is not None and hasattr(self._data_validator, 'validate'):
                result = self._data_validator.validate(data)
                if hasattr(result, 'is_valid'):
                    if not result.is_valid:
                        errors_str = '; '.join(result.errors) if hasattr(result, 'errors') else str(result)
                        raise ValueError(f"Datos OHLCV inválidos: {errors_str}")
                    elif hasattr(result, 'warnings') and result.warnings:
                        logger.warning(f"Advertencias en datos OHLCV: {'; '.join(result.warnings)}")
            else:
                # Fallback a validación por defecto
                is_valid, errors = validate_ohlcv_data(data, strict=not auto_sanitize)
                if not is_valid:
                    raise ValueError(f"Datos OHLCV inválidos: {'; '.join(errors)}")
                elif errors:
                    logger.warning(f"Advertencias en datos OHLCV: {'; '.join(errors)}")

        if auto_sanitize:
            # Usar sanitizer inyectado si existe
            if self._data_validator is not None and hasattr(self._data_validator, 'sanitize'):
                data = self._data_validator.sanitize(data)
            else:
                data = sanitize_ohlcv_data(data)

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

    def calculate_position_size(self, price: float, stop_loss: float, volatility: Optional[float] = None) -> float:
        """
        Calcula el tamaño de posición basado en riesgo.

        Args:
            price: Precio de entrada
            stop_loss: Precio de stop loss
            volatility: Volatilidad opcional (ATR) para position sizers basados en vol

        Returns:
            Tamaño de posición (0 si los parámetros son inválidos)
        """
        # Usar position sizer inyectado si existe
        if self._position_sizer is not None and hasattr(self._position_sizer, 'calculate_size'):
            try:
                return self._position_sizer.calculate_size(
                    capital=self.capital,
                    price=price,
                    stop_loss=stop_loss,
                    volatility=volatility
                )
            except Exception as e:
                logger.warning(f"calculate_position_size: error en position_sizer inyectado: {e}")
                # Fallback a cálculo por defecto

        # Cálculo por defecto
        # Validaciones de entrada
        if price <= 0:
            logger.warning(f"calculate_position_size: price inválido ({price})")
            return 0

        if stop_loss < 0:
            logger.warning(f"calculate_position_size: stop_loss inválido ({stop_loss})")
            return 0

        if self.capital <= 0:
            logger.warning(f"calculate_position_size: capital insuficiente ({self.capital})")
            return 0

        risk_amount = self.capital * (self.config.risk_per_trade / 100)
        risk_per_unit = abs(price - stop_loss)

        # Protección contra división por cero o riesgo muy pequeño
        MIN_RISK_PER_UNIT = 1e-10
        if risk_per_unit < MIN_RISK_PER_UNIT:
            logger.warning(f"calculate_position_size: risk_per_unit muy pequeño ({risk_per_unit})")
            return 0

        position_size = risk_amount / risk_per_unit

        # Verificar overflow
        if not np.isfinite(position_size):
            logger.warning(f"calculate_position_size: overflow en cálculo de posición")
            return 0

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

        # Calcular return_pct con protección contra división por cero
        position_value = position.entry_price * position.quantity
        if position_value > 0:
            return_pct = (net_pnl / position_value) * 100
        else:
            return_pct = 0.0
            logger.warning(f"close_position: position_value es cero o negativo ({position_value})")

        self.closed_trades.append({
            'entry_time': position.entry_time,
            'exit_time': exit_time,
            'entry_price': position.entry_price,
            'exit_price': exit_price,
            'quantity': position.quantity,
            'position_type': position.position_type,
            'pnl': net_pnl,
            'return_pct': return_pct,
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

        # Profit factor con protección completa contra división por cero
        if losing_trades > 0 and avg_loss != 0:
            total_loss = abs(avg_loss * losing_trades)
            if total_loss > 0:
                profit_factor = abs(avg_win * winning_trades) / total_loss
            else:
                profit_factor = float('inf') if winning_trades > 0 else 0.0
        else:
            profit_factor = float('inf') if winning_trades > 0 else 0.0

        # Total return con protección contra initial_capital = 0
        if self.config.initial_capital > 0:
            total_return = ((self.equity_curve[-1] - self.config.initial_capital) / self.config.initial_capital) * 100
        else:
            total_return = 0.0
            logger.warning("get_performance_metrics: initial_capital es cero")

        equity_series = pd.Series(self.equity_curve)
        rolling_max = equity_series.expanding().max()

        # Drawdown con protección contra rolling_max = 0
        with np.errstate(divide='ignore', invalid='ignore'):
            drawdown = np.where(
                rolling_max > 0,
                (equity_series - rolling_max) / rolling_max * 100,
                0.0
            )
        max_drawdown = np.nanmin(drawdown) if len(drawdown) > 0 else 0.0

        returns = equity_series.pct_change().dropna()
        std_returns = returns.std()
        if len(returns) > 0 and std_returns > 0 and np.isfinite(std_returns):
            sharpe_ratio = (returns.mean() / std_returns) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0

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
