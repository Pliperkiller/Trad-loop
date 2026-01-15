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
    maker_fee: float = 0.1  # Fee para órdenes limit (%)
    taker_fee: float = 0.1  # Fee para órdenes market (%)
    slippage: float = 0.05  # Slippage estimado en porcentaje

    @property
    def commission(self) -> float:
        """Backward compatibility: retorna taker_fee como default"""
        return self.taker_fee

    def get_fee(self, is_maker: bool = False) -> float:
        """Retorna el fee según tipo de orden"""
        return self.maker_fee if is_maker else self.taker_fee


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

        # Validar y configurar el índice como DatetimeIndex
        if isinstance(self.data.index, pd.DatetimeIndex):
            # Ya es DatetimeIndex, verificar que no tenga fechas de 1970 (señal de índice incorrecto)
            if len(self.data) > 0:
                first_timestamp = pd.Timestamp(self.data.index[0])
                if first_timestamp.year < 2000:
                    logger.warning(
                        f"DataFrame tiene índice DatetimeIndex pero con fechas sospechosas ({first_timestamp}). "
                        "Verificar que los datos OHLCV tienen timestamps correctos."
                    )
        elif 'timestamp' in self.data.columns:
            # Si hay columna timestamp, usarla como índice
            logger.debug("Estableciendo columna 'timestamp' como índice del DataFrame")
            self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
            self.data = self.data.set_index('timestamp')
        elif isinstance(self.data.index, pd.RangeIndex):
            # RangeIndex numérico - esto causará fechas de 1970
            raise ValueError(
                "El DataFrame tiene índice numérico (RangeIndex) sin columna 'timestamp'. "
                "Los datos OHLCV deben tener timestamps válidos como índice o columna 'timestamp'. "
                "Verifica que los datos se obtuvieron correctamente del exchange."
            )
        else:
            # Intentar convertir, pero con validación
            try:
                self.data.index = pd.to_datetime(self.data.index)
                # Verificar resultado
                if len(self.data) > 0:
                    first_timestamp = pd.Timestamp(self.data.index[0])
                    if first_timestamp.year < 2000:
                        raise ValueError(
                            f"La conversión del índice produjo fechas inválidas ({first_timestamp}). "
                            "Verifica que los datos OHLCV tienen timestamps correctos."
                        )
            except ValueError:
                raise  # Re-lanzar ValueError de la validación de fecha
            except Exception as e:
                raise ValueError(
                    f"No se pudo convertir el índice del DataFrame a datetime: {e}. "
                    "Los datos OHLCV deben tener timestamps válidos."
                )

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

            # Entry siempre usa taker fee (orden market)
            entry_fee = self.config.get_fee(is_maker=False)
            cost = position_size * signal.price * (1 + entry_fee / 100)
            self.capital -= cost

    def close_position(self, position: Position, exit_price: float, exit_time: datetime, reason: str):
        """Cierra una posición existente"""
        if position.position_type == 'LONG':
            pnl = (exit_price - position.entry_price) * position.quantity
        else:
            pnl = (position.entry_price - exit_price) * position.quantity

        # Stop Loss y Take Profit son órdenes limit (maker), Signal Exit es market (taker)
        is_exit_maker = reason in ('Stop Loss', 'Take Profit')
        entry_fee = self.config.get_fee(is_maker=False)  # Entry siempre market
        exit_fee = self.config.get_fee(is_maker=is_exit_maker)

        # Comisión total: entry + exit
        entry_commission = position.entry_price * position.quantity * (entry_fee / 100)
        exit_commission = exit_price * position.quantity * (exit_fee / 100)
        commission = entry_commission + exit_commission
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
            'reason': reason,
            # Detalles de comisiones
            'commission_total': commission,
            'commission_entry': entry_commission,
            'commission_entry_type': 'taker',  # Entry siempre es market/taker
            'commission_exit': exit_commission,
            'commission_exit_type': 'maker' if is_exit_maker else 'taker',
        })

        self.capital += (position.quantity * exit_price * (1 - exit_fee / 100))
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
            # Retornar métricas en 0 en lugar de dict vacío para evitar -inf en optimizador
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate_pct': 0.0,
                'profit_factor': 0.0,
                'total_return_pct': 0.0,
                'max_drawdown_pct': 0.0,
                'final_capital': self.config.initial_capital,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'calmar_ratio': 0.0,
                'omega_ratio': 0.0,
                'recovery_factor': 0.0,
                'adjusted_profit_factor': 0.0,
                'expectancy_per_trade': 0.0,
                'omega_ratio_zero': 0.0,
                'mean_reversion_score': 0.0,
                'mean_reversion_optimization_target': 0.0,
                'mrqs': 0.0,
                'mrqs_raw': 0.0,
                'mrqs_trade_penalty': 0.0,
                'buy_hold_return_pct': 0.0,
                'strategy_vs_hold_pct': 0.0,
            }

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

        # Buy & Hold return (benchmark) - comparar primer vs último precio
        buy_hold_return = 0.0
        if self.data is not None and len(self.data) > 0 and 'close' in self.data.columns:
            first_price = float(self.data['close'].iloc[0])
            last_price = float(self.data['close'].iloc[-1])
            if first_price > 0:
                buy_hold_return = ((last_price - first_price) / first_price) * 100

        # Alpha vs Buy & Hold (strategy outperformance)
        strategy_vs_hold = total_return - buy_hold_return

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

        # Sortino Ratio (solo downside risk)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else 0.0
        if len(returns) > 0 and downside_std > 0 and np.isfinite(downside_std):
            sortino_ratio = (returns.mean() / downside_std) * np.sqrt(252)
        else:
            sortino_ratio = 0.0

        # Calmar Ratio (CAGR / Max Drawdown)
        days = len(equity_series)
        years = days / 252 if days > 0 else 0
        if years > 0 and abs(max_drawdown) > 0:
            cagr = (((equity_series.iloc[-1] / self.config.initial_capital) ** (1 / years)) - 1) * 100
            calmar_ratio = cagr / abs(max_drawdown)
        else:
            calmar_ratio = 0.0

        # Omega Ratio (ganancias sobre threshold vs pérdidas)
        threshold = 0.0
        gains = returns[returns > threshold] - threshold
        losses = threshold - returns[returns < threshold]
        omega_ratio = gains.sum() / losses.sum() if losses.sum() > 0 else 0.0

        # Recovery Factor (ganancia neta / max drawdown)
        net_profit = equity_series.iloc[-1] - self.config.initial_capital
        max_dd_value = abs(max_drawdown) * self.config.initial_capital / 100
        recovery_factor = net_profit / max_dd_value if max_dd_value > 0 else 0.0

        # Mean Reversion metrics usando retornos de trades
        trade_returns = trades_df['return_pct'].values if 'return_pct' in trades_df.columns else np.array([])

        # Adjusted Profit Factor
        if total_trades >= 10 and profit_factor != float('inf'):
            adjusted_profit_factor = profit_factor * (1 - 1/np.sqrt(total_trades))
        else:
            adjusted_profit_factor = 0.0

        # Expectancy per Trade
        win_rate_decimal = win_rate / 100
        expectancy_per_trade = (win_rate_decimal * avg_win) - ((1 - win_rate_decimal) * abs(avg_loss))

        # Omega Ratio Zero (threshold=0, usando trade returns)
        if len(trade_returns) > 0:
            omega_gains = float(np.sum(trade_returns[trade_returns > 0]))
            omega_losses = float(np.abs(np.sum(trade_returns[trade_returns <= 0])))
            omega_ratio_zero = omega_gains / omega_losses if omega_losses > 0 else 10.0
        else:
            omega_ratio_zero = 0.0

        # Mean Reversion Score
        win_rate_component = 0.5 * min(win_rate_decimal / 0.65, 1.0)
        pf_for_score = profit_factor if profit_factor != float('inf') else 10.0
        pf_component = 0.5 * min(max(pf_for_score - 1.0, 0) / 0.8, 1.0) if pf_for_score < 100 else 0.5
        consistency_score = win_rate_component + pf_component
        dd_risk_component = 0.5 * max(1.0 - abs(max_drawdown) / 30.0, 0.0)
        recovery_component = 0.5 * min(max(recovery_factor, 0) / 2.0, 1.0)
        mean_reversion_score = 0.5 * consistency_score + 0.5 * (dd_risk_component + recovery_component) / 2

        # Mean Reversion Optimization Target
        max_allowed_dd = 25.0
        if abs(max_drawdown) > max_allowed_dd or total_trades < 10:
            mean_reversion_optimization_target = 0.0
        else:
            omega_comp = min(omega_ratio_zero / 2.0, 1.0)
            recovery_comp = min(max(recovery_factor, 0) / 3.0, 1.0)
            expectancy_comp = min(max(expectancy_per_trade, 0) / 2.0, 1.0)
            dd_comp = max(1.0 - abs(max_drawdown) / max_allowed_dd, 0.0)
            mean_reversion_optimization_target = 0.30 * omega_comp + 0.25 * recovery_comp + 0.25 * expectancy_comp + 0.20 * dd_comp

        # MRQS (Mean Reversion Quality Score) - optimizado para crypto futuros
        # Componentes ponderados
        sortino_score = min(sortino_ratio / 2.0, 1.5) if np.isfinite(sortino_ratio) else 0.0
        calmar_for_mrqs = (total_return / abs(max_drawdown)) if abs(max_drawdown) > 0 else 0.0
        calmar_score = min(calmar_for_mrqs / 1.5, 1.5)

        # Win rate ajustado (penaliza WR muy alto que indica overfitting)
        if win_rate_decimal < 0.35:
            wr_score = win_rate_decimal * 2
        elif win_rate_decimal > 0.85:
            wr_score = 1.7 - win_rate_decimal
        else:
            wr_score = 0.7 + (win_rate_decimal - 0.35)

        pf_for_mrqs = profit_factor if profit_factor != float('inf') and profit_factor < 10 else 10.0
        pf_score = min(pf_for_mrqs / 2.0, 1.0)

        # Expectancy score
        exp_score = min(max(expectancy_per_trade / 2.0, -0.5), 1.0)

        # Trade penalty (mínimo 30 trades para 4H)
        if total_trades < 15:
            mrqs_trade_penalty = 0.3
        elif total_trades < 30:
            mrqs_trade_penalty = 0.6 + (total_trades - 15) * 0.027
        else:
            mrqs_trade_penalty = 1.0

        mrqs_raw = (
            sortino_score * 0.40 +
            calmar_score * 0.25 +
            wr_score * 0.15 +
            pf_score * 0.10 +
            exp_score * 0.10
        )
        mrqs = mrqs_raw * mrqs_trade_penalty
        if not np.isfinite(mrqs):
            mrqs = 0.0

        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate_pct': win_rate,
            'profit_factor': profit_factor if profit_factor != float('inf') else 999.0,
            'total_return_pct': total_return,
            'max_drawdown_pct': max_drawdown,
            'final_capital': self.equity_curve[-1],
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'omega_ratio': omega_ratio,
            'recovery_factor': recovery_factor,
            'adjusted_profit_factor': adjusted_profit_factor,
            'expectancy_per_trade': expectancy_per_trade,
            'omega_ratio_zero': omega_ratio_zero,
            'mean_reversion_score': mean_reversion_score,
            'mean_reversion_optimization_target': mean_reversion_optimization_target,
            'mrqs': mrqs,
            'mrqs_raw': mrqs_raw,
            'mrqs_trade_penalty': mrqs_trade_penalty,
            'buy_hold_return_pct': buy_hold_return,
            'strategy_vs_hold_pct': strategy_vs_hold,
        }
