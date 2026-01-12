"""
Configuracion para Paper Trading

Define las opciones de configuracion del sistema:
- Capital inicial y gestion de riesgo
- Costos de transaccion (comisiones, slippage)
- Opciones de simulacion
- Persistencia de datos
"""

from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum


class SlippageModel(Enum):
    """Modelos de slippage disponibles"""
    NONE = "none"               # Sin slippage
    FIXED = "fixed"             # Porcentaje fijo
    VOLATILITY = "volatility"   # Basado en volatilidad (ATR)
    ORDERBOOK = "orderbook"     # Basado en order book (mas realista)


class CommissionModel(Enum):
    """Modelos de comision"""
    PERCENTAGE = "percentage"   # Porcentaje del valor
    FIXED = "fixed"             # Monto fijo por trade
    TIERED = "tiered"           # Por volumen (maker/taker)


@dataclass
class PaperTradingConfig:
    """
    Configuracion completa para Paper Trading.

    Example:
        config = PaperTradingConfig(
            initial_balance=10000,
            commission_rate=0.001,
            symbols=["BTC/USDT", "ETH/USDT"]
        )

    Attributes:
        initial_balance: Capital inicial en USD
        symbols: Lista de simbolos a monitorear
        default_timeframe: Temporalidad por defecto

        commission_model: Modelo de comision
        commission_rate: Tasa de comision (0.001 = 0.1%)
        maker_fee: Fee para ordenes maker
        taker_fee: Fee para ordenes taker

        slippage_model: Modelo de slippage
        fixed_slippage: Slippage fijo (0.0005 = 0.05%)
        max_slippage: Slippage maximo permitido

        max_position_size: Tamano maximo de posicion (% del capital)
        max_positions: Numero maximo de posiciones simultaneas
        max_drawdown_pct: Drawdown maximo antes de alertar
        risk_per_trade: Riesgo por trade (% del capital)

        latency_ms: Latencia simulada en milisegundos
        partial_fills: Permitir ejecuciones parciales
        simulate_rejects: Simular rechazos aleatorios

        save_trades: Guardar trades en archivo
        trades_file: Ruta del archivo de trades
        log_level: Nivel de logging

        exchange: Exchange a usar (binance, kraken, etc.)
        testnet: Usar testnet del exchange
    """
    # Capital
    initial_balance: float = 10000.0
    symbols: List[str] = field(default_factory=lambda: ["BTC/USDT"])
    default_timeframe: str = "1m"

    # Comisiones
    commission_model: CommissionModel = CommissionModel.PERCENTAGE
    commission_rate: float = 0.001  # 0.1% (Binance standard)
    maker_fee: float = 0.001
    taker_fee: float = 0.001

    # Slippage
    slippage_model: SlippageModel = SlippageModel.VOLATILITY
    fixed_slippage: float = 0.0005  # 0.05%
    max_slippage: float = 0.01      # 1% maximo

    # Gestion de riesgo
    max_position_size: float = 0.25     # 25% del capital
    max_positions: int = 5
    max_drawdown_pct: float = 0.20      # 20% drawdown alerta
    risk_per_trade: float = 0.02        # 2% riesgo por trade
    use_stop_loss: bool = True
    default_stop_loss_pct: float = 0.02  # 2% stop loss
    use_take_profit: bool = True
    default_take_profit_pct: float = 0.04  # 4% take profit

    # Simulacion
    latency_ms: int = 50        # Latencia simulada
    partial_fills: bool = False
    simulate_rejects: bool = False
    reject_probability: float = 0.01  # 1% probabilidad rechazo

    # Persistencia
    save_trades: bool = True
    trades_file: str = "paper_trades.json"
    save_state_interval: int = 60  # Guardar estado cada N segundos
    log_level: str = "INFO"

    # Exchange
    exchange: str = "binance"
    testnet: bool = False
    api_key: Optional[str] = None
    api_secret: Optional[str] = None

    def validate(self) -> List[str]:
        """
        Valida la configuracion y retorna lista de errores.

        Returns:
            Lista de mensajes de error (vacia si es valida)
        """
        errors = []

        if self.initial_balance <= 0:
            errors.append("initial_balance debe ser mayor que 0")

        if not self.symbols:
            errors.append("Debe especificar al menos un symbol")

        if not 0 <= self.commission_rate <= 0.1:
            errors.append("commission_rate debe estar entre 0 y 0.1 (10%)")

        if not 0 <= self.fixed_slippage <= 0.1:
            errors.append("fixed_slippage debe estar entre 0 y 0.1 (10%)")

        if not 0 < self.max_position_size <= 1:
            errors.append("max_position_size debe estar entre 0 y 1")

        if self.max_positions < 1:
            errors.append("max_positions debe ser al menos 1")

        if not 0 < self.risk_per_trade <= 0.1:
            errors.append("risk_per_trade debe estar entre 0 y 0.1 (10%)")

        if self.latency_ms < 0:
            errors.append("latency_ms no puede ser negativo")

        return errors

    def is_valid(self) -> bool:
        """Verifica si la configuracion es valida"""
        return len(self.validate()) == 0

    def get_commission(self, order_value: float, is_maker: bool = False) -> float:
        """
        Calcula la comision para un valor de orden dado.

        Args:
            order_value: Valor total de la orden
            is_maker: Si es orden maker (limit) o taker (market)

        Returns:
            Monto de comision
        """
        if self.commission_model == CommissionModel.PERCENTAGE:
            rate = self.maker_fee if is_maker else self.taker_fee
            return order_value * rate
        elif self.commission_model == CommissionModel.FIXED:
            return self.commission_rate  # En este caso es monto fijo
        else:
            # Tiered - simplificado, usar taker por defecto
            return order_value * self.taker_fee

    def get_max_position_value(self) -> float:
        """Retorna el valor maximo permitido por posicion"""
        return self.initial_balance * self.max_position_size

    def to_dict(self) -> dict:
        """Convierte la configuracion a diccionario"""
        return {
            "initial_balance": self.initial_balance,
            "symbols": self.symbols,
            "default_timeframe": self.default_timeframe,
            "commission_model": self.commission_model.value,
            "commission_rate": self.commission_rate,
            "slippage_model": self.slippage_model.value,
            "fixed_slippage": self.fixed_slippage,
            "max_position_size": self.max_position_size,
            "max_positions": self.max_positions,
            "risk_per_trade": self.risk_per_trade,
            "exchange": self.exchange,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PaperTradingConfig":
        """Crea configuracion desde diccionario"""
        config = cls()

        if "initial_balance" in data:
            config.initial_balance = data["initial_balance"]
        if "symbols" in data:
            config.symbols = data["symbols"]
        if "default_timeframe" in data:
            config.default_timeframe = data["default_timeframe"]
        if "commission_rate" in data:
            config.commission_rate = data["commission_rate"]
        if "fixed_slippage" in data:
            config.fixed_slippage = data["fixed_slippage"]
        if "max_position_size" in data:
            config.max_position_size = data["max_position_size"]
        if "max_positions" in data:
            config.max_positions = data["max_positions"]
        if "risk_per_trade" in data:
            config.risk_per_trade = data["risk_per_trade"]
        if "exchange" in data:
            config.exchange = data["exchange"]

        return config


# Configuraciones predefinidas para diferentes escenarios
CONSERVATIVE_CONFIG = PaperTradingConfig(
    initial_balance=10000,
    max_position_size=0.10,  # 10%
    max_positions=3,
    risk_per_trade=0.01,  # 1%
    default_stop_loss_pct=0.015,  # 1.5%
    default_take_profit_pct=0.03,  # 3%
)

MODERATE_CONFIG = PaperTradingConfig(
    initial_balance=10000,
    max_position_size=0.20,  # 20%
    max_positions=5,
    risk_per_trade=0.02,  # 2%
    default_stop_loss_pct=0.02,  # 2%
    default_take_profit_pct=0.04,  # 4%
)

AGGRESSIVE_CONFIG = PaperTradingConfig(
    initial_balance=10000,
    max_position_size=0.30,  # 30%
    max_positions=8,
    risk_per_trade=0.03,  # 3%
    default_stop_loss_pct=0.03,  # 3%
    default_take_profit_pct=0.06,  # 6%
)
