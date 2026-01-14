"""
Protocolos (interfaces) para componentes de StrategyTrader.

Estos protocolos definen contratos que las clases pueden implementar
sin necesidad de heredar de una clase base específica (duck typing estructural).

Uso:
    from src.interfaces import IDataValidator, IMetricsCalculator

    class MyValidator:
        def validate(self, data: pd.DataFrame) -> ValidationResult:
            ...

    # MyValidator implementa IDataValidator implícitamente si tiene el método correcto
"""

from typing import (
    Protocol,
    Dict,
    List,
    Any,
    Optional,
    Tuple,
    Callable,
    TypeVar,
    runtime_checkable,
)
from dataclasses import dataclass
from datetime import datetime
import pandas as pd


# ============================================================================
# Data Types
# ============================================================================

@dataclass
class ValidationResult:
    """Resultado de validación de datos."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]

    def __bool__(self) -> bool:
        return self.is_valid


@dataclass
class Signal:
    """Señal de trading genérica."""
    timestamp: datetime
    symbol: str
    action: str  # 'BUY', 'SELL', 'HOLD'
    price: float
    confidence: float
    metadata: Dict[str, Any]


@dataclass
class PositionInfo:
    """Información de una posición."""
    id: str
    symbol: str
    side: str  # 'LONG', 'SHORT'
    entry_price: float
    quantity: float
    current_price: float
    unrealized_pnl: float
    entry_time: datetime


@dataclass
class TradeResult:
    """Resultado de un trade cerrado."""
    id: str
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    return_pct: float
    entry_time: datetime
    exit_time: datetime
    exit_reason: str


@dataclass
class PerformanceMetrics:
    """Métricas de rendimiento."""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    total_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    avg_win: float
    avg_loss: float
    expectancy: float


# ============================================================================
# Core Interfaces
# ============================================================================

@runtime_checkable
class IDataValidator(Protocol):
    """
    Interface para validadores de datos OHLCV.

    Implementaciones pueden agregar validaciones específicas
    (ej: validar gaps, detectar splits, etc.)
    """

    def validate(self, data: pd.DataFrame) -> ValidationResult:
        """
        Valida un DataFrame OHLCV.

        Args:
            data: DataFrame con columnas ['open', 'high', 'low', 'close', 'volume']

        Returns:
            ValidationResult con is_valid, errors y warnings
        """
        ...

    def sanitize(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Limpia datos removiendo filas problemáticas.

        Args:
            data: DataFrame OHLCV

        Returns:
            DataFrame limpio
        """
        ...


@runtime_checkable
class IMetricsCalculator(Protocol):
    """
    Interface para calculadores de métricas de rendimiento.

    Permite inyectar diferentes calculadores según necesidades
    (ej: métricas estándar vs métricas avanzadas para HFT).
    """

    def calculate(
        self,
        trades: List[TradeResult],
        equity_curve: List[float],
        initial_capital: float
    ) -> PerformanceMetrics:
        """
        Calcula métricas de rendimiento.

        Args:
            trades: Lista de trades cerrados
            equity_curve: Curva de equity
            initial_capital: Capital inicial

        Returns:
            PerformanceMetrics con todas las métricas calculadas
        """
        ...


@runtime_checkable
class IPositionSizer(Protocol):
    """
    Interface para calculadores de tamaño de posición.

    Permite implementar diferentes estrategias de sizing:
    - Fixed fractional
    - Kelly criterion
    - Volatility-based
    - Equal weight
    """

    def calculate_size(
        self,
        capital: float,
        price: float,
        stop_loss: float,
        volatility: Optional[float] = None
    ) -> float:
        """
        Calcula el tamaño de posición.

        Args:
            capital: Capital disponible
            price: Precio de entrada
            stop_loss: Precio de stop loss
            volatility: Volatilidad opcional (para sizing basado en vol)

        Returns:
            Tamaño de posición (cantidad)
        """
        ...


@runtime_checkable
class ISignalGenerator(Protocol):
    """
    Interface para generadores de señales.

    Separa la lógica de generación de señales de la ejecución,
    permitiendo composición de múltiples generadores.
    """

    def generate(self, data: pd.DataFrame) -> pd.Series:
        """
        Genera señales de trading.

        Args:
            data: DataFrame OHLCV con indicadores

        Returns:
            Series con señales ('BUY', 'SELL', None)
        """
        ...

    def get_current_signal(self, data: pd.DataFrame) -> Optional[Signal]:
        """
        Obtiene la señal actual (última fila).

        Args:
            data: DataFrame OHLCV con indicadores

        Returns:
            Signal si hay señal activa, None si no
        """
        ...


@runtime_checkable
class IRiskManager(Protocol):
    """
    Interface para gestores de riesgo.

    Valida si un trade debe ejecutarse basado en:
    - Límites de posición
    - Drawdown máximo
    - Correlación con posiciones existentes
    - Exposure por símbolo/sector
    """

    def can_open_position(
        self,
        symbol: str,
        side: str,
        size: float,
        price: float,
        current_positions: List[PositionInfo],
        current_equity: float
    ) -> Tuple[bool, str]:
        """
        Verifica si se puede abrir una posición.

        Args:
            symbol: Símbolo del instrumento
            side: 'LONG' o 'SHORT'
            size: Tamaño propuesto
            price: Precio de entrada
            current_positions: Posiciones abiertas
            current_equity: Equity actual

        Returns:
            Tuple (puede_abrir, razón)
        """
        ...

    def calculate_stop_loss(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        atr: Optional[float] = None
    ) -> float:
        """
        Calcula el stop loss para una posición.

        Args:
            symbol: Símbolo
            side: 'LONG' o 'SHORT'
            entry_price: Precio de entrada
            atr: ATR opcional para stop dinámico

        Returns:
            Precio de stop loss
        """
        ...


# ============================================================================
# Data Provider Interfaces
# ============================================================================

@runtime_checkable
class IOHLCVFetcher(Protocol):
    """Interface para obtener datos OHLCV."""

    def fetch(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime
    ) -> pd.DataFrame:
        """
        Obtiene datos OHLCV.

        Args:
            symbol: Par de trading (ej: "BTC/USDT")
            timeframe: Temporalidad (ej: "1h", "1d")
            start: Fecha inicio
            end: Fecha fin

        Returns:
            DataFrame con columnas OHLCV
        """
        ...


@runtime_checkable
class IDataProvider(Protocol):
    """
    Interface para proveedores de datos completos.

    Extiende IOHLCVFetcher con funcionalidad adicional
    como streaming, caché, etc.
    """

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime
    ) -> pd.DataFrame:
        """Obtiene datos OHLCV históricos."""
        ...

    def get_supported_symbols(self) -> List[str]:
        """Lista símbolos soportados."""
        ...

    def get_supported_timeframes(self) -> List[str]:
        """Lista temporalidades soportadas."""
        ...


# ============================================================================
# Trading Interfaces
# ============================================================================

@runtime_checkable
class IOrderExecutor(Protocol):
    """
    Interface para ejecutores de órdenes.

    Abstrae la ejecución de órdenes para permitir:
    - Paper trading
    - Live trading
    - Backtesting
    """

    def execute_market_order(
        self,
        symbol: str,
        side: str,
        quantity: float
    ) -> Dict[str, Any]:
        """
        Ejecuta orden de mercado.

        Args:
            symbol: Símbolo
            side: 'BUY' o 'SELL'
            quantity: Cantidad

        Returns:
            Resultado de la ejecución
        """
        ...

    def execute_limit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float
    ) -> Dict[str, Any]:
        """Ejecuta orden límite."""
        ...

    def cancel_order(self, order_id: str) -> bool:
        """Cancela una orden."""
        ...


@runtime_checkable
class IPositionManager(Protocol):
    """
    Interface para gestores de posiciones.

    Maneja el ciclo de vida de posiciones:
    apertura, actualización, cierre.
    """

    def open_position(
        self,
        symbol: str,
        side: str,
        quantity: float,
        entry_price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Optional[PositionInfo]:
        """Abre una nueva posición."""
        ...

    def close_position(
        self,
        position_id: str,
        exit_price: float,
        reason: str
    ) -> Optional[TradeResult]:
        """Cierra una posición existente."""
        ...

    def update_prices(self, prices: Dict[str, float]) -> None:
        """Actualiza precios y verifica SL/TP."""
        ...

    def get_positions(self) -> List[PositionInfo]:
        """Obtiene todas las posiciones abiertas."""
        ...

    def get_position(self, position_id: str) -> Optional[PositionInfo]:
        """Obtiene una posición por ID."""
        ...


# ============================================================================
# Optimization Interfaces
# ============================================================================

# Type variable para resultados genéricos de optimización
T = TypeVar('T')


@runtime_checkable
class IObjectiveFunction(Protocol):
    """
    Interface para funciones objetivo de optimización.

    Permite definir diferentes criterios de optimización:
    - Sharpe ratio
    - Profit factor
    - Combinaciones personalizadas
    """

    def evaluate(self, params: Dict[str, Any]) -> float:
        """
        Evalúa un conjunto de parámetros.

        Args:
            params: Diccionario de parámetros

        Returns:
            Score (mayor es mejor)
        """
        ...


@runtime_checkable
class IOptimizer(Protocol):
    """
    Interface para optimizadores.

    Define el contrato común para diferentes algoritmos:
    - Grid search
    - Random search
    - Bayesian
    - Genetic
    """

    def optimize(
        self,
        objective: IObjectiveFunction,
        parameter_space: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Ejecuta la optimización.

        Args:
            objective: Función objetivo a maximizar
            parameter_space: Espacio de parámetros
            **kwargs: Opciones específicas del optimizador

        Returns:
            Diccionario con best_params, best_score, all_results
        """
        ...
