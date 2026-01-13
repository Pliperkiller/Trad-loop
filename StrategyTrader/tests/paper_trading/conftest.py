"""
Fixtures compartidos para tests de ordenes avanzadas.

Proporciona configuraciones, datos de mercado y ordenes de ejemplo
para todos los tests del modulo paper_trading.
"""

import pytest
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from paper_trading.orders.enums import OrderSide, OrderStatus, TriggerDirection
from paper_trading.orders.risk_control import TrailingStopOrder, BracketOrder
from paper_trading.orders.execution_algos import TWAPOrder, VWAPOrder, IcebergOrder, HiddenOrder
from paper_trading.orders.dynamic_orders import FOKOrder, IOCOrder, ReduceOnlyOrder, PostOnlyOrder, ScaleOrder
from paper_trading.orders.conditional_orders import IfTouchedOrder, OCOOrder, OTOCOOrder
from paper_trading.simulators.trailing_simulator import TrailingStopSimulator
from paper_trading.simulators.algo_simulator import AlgoOrderSimulator, TWAPConfig, VWAPConfig, IcebergConfig
from paper_trading.simulators.composite_simulator import CompositeOrderSimulator


# ============================================================================
# Market State Fixtures
# ============================================================================

@pytest.fixture
def btc_price():
    """Precio base de BTC para tests"""
    return 50000.0


@pytest.fixture
def market_state_btc(btc_price):
    """Estado de mercado BTC estandar"""
    return {
        "symbol": "BTC/USDT",
        "current_price": btc_price,
        "bid": btc_price - 5,
        "ask": btc_price + 5,
        "spread": 0.0002,
        "volume_24h": 1_000_000_000,
        "volatility": 0.02,
    }


@pytest.fixture
def volatile_market_state(btc_price):
    """Estado de mercado con alta volatilidad"""
    return {
        "symbol": "BTC/USDT",
        "current_price": btc_price,
        "bid": btc_price - 50,
        "ask": btc_price + 50,
        "spread": 0.002,
        "volume_24h": 500_000_000,
        "volatility": 0.08,
    }


@pytest.fixture
def price_sequence_uptrend(btc_price):
    """Secuencia de precios en tendencia alcista"""
    return [btc_price + i * 100 for i in range(20)]


@pytest.fixture
def price_sequence_downtrend(btc_price):
    """Secuencia de precios en tendencia bajista"""
    return [btc_price - i * 100 for i in range(20)]


@pytest.fixture
def price_sequence_volatile(btc_price):
    """Secuencia de precios volatil"""
    import random
    random.seed(42)
    prices = [btc_price]
    for _ in range(50):
        change = random.uniform(-500, 500)
        prices.append(prices[-1] + change)
    return prices


@pytest.fixture
def price_sequence_with_spike(btc_price):
    """Secuencia con spike y reversal"""
    return [
        btc_price,
        btc_price + 200,
        btc_price + 500,
        btc_price + 1000,  # Spike
        btc_price + 800,
        btc_price + 500,
        btc_price + 200,
        btc_price - 200,   # Reversal
        btc_price - 500,
    ]


# ============================================================================
# Trailing Stop Fixtures
# ============================================================================

@pytest.fixture
def trailing_stop_long(btc_price):
    """Trailing stop para posicion long (2%)"""
    return TrailingStopOrder(
        symbol="BTC/USDT",
        side=OrderSide.SELL,
        quantity=0.1,
        trail_percent=0.02,
        initial_price=btc_price
    )


@pytest.fixture
def trailing_stop_short(btc_price):
    """Trailing stop para posicion short (2%)"""
    return TrailingStopOrder(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        quantity=0.1,
        trail_percent=0.02,
        initial_price=btc_price
    )


@pytest.fixture
def trailing_stop_fixed_amount(btc_price):
    """Trailing stop con cantidad fija"""
    return TrailingStopOrder(
        symbol="BTC/USDT",
        side=OrderSide.SELL,
        quantity=0.1,
        trail_amount=1000,
        initial_price=btc_price
    )


@pytest.fixture
def trailing_stop_with_activation(btc_price):
    """Trailing stop con precio de activacion"""
    return TrailingStopOrder(
        symbol="BTC/USDT",
        side=OrderSide.SELL,
        quantity=0.1,
        trail_percent=0.02,
        activation_price=btc_price + 1000,
        initial_price=btc_price
    )


@pytest.fixture
def trailing_simulator():
    """Simulador de trailing stops"""
    return TrailingStopSimulator()


# ============================================================================
# Bracket Order Fixtures
# ============================================================================

@pytest.fixture
def bracket_order_long(btc_price):
    """Bracket order para entrada long"""
    return BracketOrder(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        quantity=0.1,
        entry_price=btc_price,
        stop_loss_price=btc_price - 1000,
        take_profit_price=btc_price + 2000
    )


@pytest.fixture
def bracket_order_short(btc_price):
    """Bracket order para entrada short"""
    return BracketOrder(
        symbol="BTC/USDT",
        side=OrderSide.SELL,
        quantity=0.1,
        entry_price=btc_price,
        stop_loss_price=btc_price + 1000,
        take_profit_price=btc_price - 2000
    )


@pytest.fixture
def bracket_order_market_entry(btc_price):
    """Bracket order con entrada de mercado"""
    from paper_trading.orders.enums import OrderType
    return BracketOrder(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        quantity=0.1,
        entry_price=None,
        entry_type=OrderType.MARKET,
        stop_loss_price=btc_price - 1000,
        take_profit_price=btc_price + 2000
    )


@pytest.fixture
def composite_simulator():
    """Simulador de ordenes compuestas"""
    return CompositeOrderSimulator()


# ============================================================================
# Execution Algorithm Fixtures
# ============================================================================

@pytest.fixture
def twap_order(btc_price):
    """Orden TWAP basica"""
    return TWAPOrder(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        total_quantity=10.0,
        duration_seconds=3600,
        slice_count=60
    )


@pytest.fixture
def twap_config():
    """Configuracion TWAP"""
    return TWAPConfig(
        total_quantity=10.0,
        duration_seconds=3600,
        slice_count=60,
        size_variation=0.1
    )


@pytest.fixture
def vwap_order(btc_price):
    """Orden VWAP basica"""
    return VWAPOrder(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        total_quantity=10.0,
        duration_seconds=3600,
        max_participation=0.05
    )


@pytest.fixture
def vwap_config():
    """Configuracion VWAP"""
    return VWAPConfig(
        total_quantity=10.0,
        duration_seconds=3600,
        max_participation=0.05
    )


@pytest.fixture
def iceberg_order(btc_price):
    """Orden Iceberg basica"""
    return IcebergOrder(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        total_quantity=100.0,
        display_quantity=10.0,
        price=btc_price
    )


@pytest.fixture
def iceberg_config(btc_price):
    """Configuracion Iceberg"""
    return IcebergConfig(
        total_quantity=100.0,
        display_quantity=10.0,
        price=btc_price
    )


@pytest.fixture
def hidden_order(btc_price):
    """Orden Hidden basica"""
    return HiddenOrder(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        quantity=10.0,
        price=btc_price
    )


@pytest.fixture
def algo_simulator():
    """Simulador de ordenes algoritmicas"""
    return AlgoOrderSimulator()


# ============================================================================
# Dynamic Order Fixtures
# ============================================================================

@pytest.fixture
def fok_order(btc_price):
    """Orden Fill-or-Kill"""
    return FOKOrder(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        quantity=1.0,
        price=btc_price
    )


@pytest.fixture
def ioc_order(btc_price):
    """Orden Immediate-or-Cancel"""
    return IOCOrder(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        quantity=1.0,
        price=btc_price
    )


@pytest.fixture
def reduce_only_order(btc_price):
    """Orden Reduce-Only con posicion"""
    return ReduceOnlyOrder(
        symbol="BTC/USDT",
        side=OrderSide.SELL,
        quantity=0.5,
        position_quantity=1.0,
        price=btc_price
    )


@pytest.fixture
def reduce_only_order_no_position(btc_price):
    """Orden Reduce-Only sin posicion"""
    return ReduceOnlyOrder(
        symbol="BTC/USDT",
        side=OrderSide.SELL,
        quantity=0.5,
        position_quantity=0.0,
        price=btc_price
    )


@pytest.fixture
def post_only_order(btc_price):
    """Orden Post-Only"""
    return PostOnlyOrder(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        quantity=1.0,
        price=btc_price - 100  # Por debajo del mercado para ser maker
    )


@pytest.fixture
def scale_in_order(btc_price):
    """Orden Scale-In"""
    from paper_trading.orders.enums import OrderType
    return ScaleOrder(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        total_quantity=1.0,
        levels=[btc_price, btc_price - 500, btc_price - 1000],
        quantities=[0.25, 0.25, 0.50],
        order_type=OrderType.SCALE_IN
    )


@pytest.fixture
def scale_out_order(btc_price):
    """Orden Scale-Out"""
    from paper_trading.orders.enums import OrderType
    return ScaleOrder(
        symbol="BTC/USDT",
        side=OrderSide.SELL,
        total_quantity=1.0,
        levels=[btc_price + 500, btc_price + 1000, btc_price + 2000],
        quantities=[0.30, 0.30, 0.40],
        order_type=OrderType.SCALE_OUT
    )


# ============================================================================
# Conditional Order Fixtures
# ============================================================================

@pytest.fixture
def if_touched_order_above(btc_price):
    """Orden If-Touched activada por precio arriba"""
    return IfTouchedOrder(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        quantity=0.1,
        trigger_price=btc_price + 1000,
        trigger_direction=TriggerDirection.ABOVE
    )


@pytest.fixture
def if_touched_order_below(btc_price):
    """Orden If-Touched activada por precio abajo"""
    return IfTouchedOrder(
        symbol="BTC/USDT",
        side=OrderSide.SELL,
        quantity=0.1,
        trigger_price=btc_price - 1000,
        trigger_direction=TriggerDirection.BELOW
    )


@pytest.fixture
def oco_order_long(btc_price):
    """OCO para posicion long (SL + TP)"""
    return OCOOrder(
        symbol="BTC/USDT",
        quantity=0.1,
        side=OrderSide.SELL,
        stop_price=btc_price - 1000,
        limit_price=btc_price + 2000
    )


@pytest.fixture
def oco_order_short(btc_price):
    """OCO para posicion short (SL + TP)"""
    return OCOOrder(
        symbol="BTC/USDT",
        quantity=0.1,
        side=OrderSide.BUY,
        stop_price=btc_price + 1000,
        limit_price=btc_price - 2000
    )


@pytest.fixture
def otoco_order_long(btc_price):
    """OTOCO para entrada long"""
    return OTOCOOrder(
        symbol="BTC/USDT",
        entry_side=OrderSide.BUY,
        entry_quantity=0.1,
        entry_price=btc_price,
        stop_loss_price=btc_price - 1000,
        take_profit_price=btc_price + 2000
    )


@pytest.fixture
def otoco_order_short(btc_price):
    """OTOCO para entrada short"""
    return OTOCOOrder(
        symbol="BTC/USDT",
        entry_side=OrderSide.SELL,
        entry_quantity=0.1,
        entry_price=btc_price,
        stop_loss_price=btc_price + 1000,
        take_profit_price=btc_price - 2000
    )


# ============================================================================
# Fixtures for existing tests (test_config.py, test_models.py, etc.)
# ============================================================================

@pytest.fixture
def default_config():
    """Configuracion por defecto para paper trading"""
    from paper_trading.config import PaperTradingConfig
    return PaperTradingConfig()


@pytest.fixture
def sample_order(btc_price):
    """Orden de ejemplo para tests"""
    from paper_trading.models import Order, OrderType as ModelsOrderType
    from paper_trading.models import OrderSide as ModelsOrderSide
    return Order(
        symbol="BTC/USDT",
        side=ModelsOrderSide.BUY,
        type=ModelsOrderType.MARKET,
        quantity=0.1
    )


@pytest.fixture
def sample_limit_order(btc_price):
    """Orden limite de ejemplo"""
    from paper_trading.models import Order, OrderType as ModelsOrderType
    from paper_trading.models import OrderSide as ModelsOrderSide
    return Order(
        symbol="BTC/USDT",
        side=ModelsOrderSide.BUY,
        type=ModelsOrderType.LIMIT,
        quantity=0.1,
        price=49000
    )


@pytest.fixture
def sample_position(btc_price):
    """Posicion de ejemplo"""
    from paper_trading.models import PaperPosition, PositionSide
    return PaperPosition(
        symbol="BTC/USDT",
        side=PositionSide.LONG,
        entry_price=50000,
        quantity=0.1,
        stop_loss=49000,
        take_profit=52000
    )


@pytest.fixture
def sample_candle():
    """Candle de ejemplo"""
    from paper_trading.models import RealtimeCandle
    return RealtimeCandle(
        symbol="BTC/USDT",
        timeframe="1m",
        open=50000,
        high=50500,
        low=49800,
        close=50200,
        volume=100.0,
        timestamp=datetime.now(),
        is_closed=True
    )


@pytest.fixture
def sample_trade(btc_price):
    """Trade de ejemplo"""
    from paper_trading.models import TradeRecord, PositionSide
    return TradeRecord(
        symbol="BTC/USDT",
        side=PositionSide.LONG,
        entry_price=50000,
        exit_price=51000,
        quantity=0.1,
        pnl=100.0,
        return_pct=2.0,
        commission=1.0,
        entry_time=datetime.now() - timedelta(hours=1),
        exit_time=datetime.now(),
        exit_reason="Take Profit"
    )


@pytest.fixture
def order_simulator_config():
    """Configuracion para OrderSimulator"""
    from paper_trading.config import PaperTradingConfig
    return PaperTradingConfig(
        initial_balance=10000,
        symbols=["BTC/USDT", "ETH/USDT"],
        commission_rate=0.001
    )


@pytest.fixture
def position_manager_config():
    """Configuracion para PositionManager"""
    from paper_trading.config import PaperTradingConfig
    return PaperTradingConfig(
        initial_balance=10000,
        symbols=["BTC/USDT"],
        max_positions=5,
        risk_per_trade=0.02
    )
