"""
Mapper de ordenes para Interactive Brokers.

Convierte BrokerOrder a ordenes IB y viceversa.
"""

from typing import Dict, Any, Optional
from datetime import datetime

try:
    from ib_insync import (
        MarketOrder, LimitOrder, StopOrder, StopLimitOrder,
        Order as IBOrder, Trade, OrderStatus as IBOrderStatus
    )
    IB_AVAILABLE = True
except ImportError:
    MarketOrder = LimitOrder = StopOrder = StopLimitOrder = None
    IBOrder = Trade = IBOrderStatus = None
    IB_AVAILABLE = False

from ...core.enums import OrderType, OrderSide, OrderStatus, TimeInForce, PositionSide
from ...core.models import BrokerOrder, BrokerPosition, ExecutionReport


class IBKROrderMapper:
    """
    Mapper para convertir ordenes entre formato interno e IB.
    """

    def __init__(self):
        if not IB_AVAILABLE:
            raise ImportError(
                "ib_insync library is not installed. "
                "Install it with: pip install ib_insync"
            )

    def to_ib_order(self, order: BrokerOrder) -> "IBOrder":
        """
        Convertir BrokerOrder a orden IB.

        Args:
            order: Orden a convertir

        Returns:
            Orden IB
        """
        action = "BUY" if order.side == OrderSide.BUY else "SELL"

        # Crear orden basica segun tipo
        if order.order_type == OrderType.MARKET:
            ib_order = MarketOrder(action, order.quantity)

        elif order.order_type == OrderType.LIMIT:
            ib_order = LimitOrder(action, order.quantity, order.price or 0)

        elif order.order_type == OrderType.STOP_LOSS:
            stop_price = order.stop_price or order.price or 0
            ib_order = StopOrder(action, order.quantity, stop_price)

        elif order.order_type == OrderType.STOP_LIMIT:
            limit_price = order.price or 0
            stop_price = order.stop_price or 0
            ib_order = StopLimitOrder(
                action, order.quantity, limit_price, stop_price
            )

        elif order.order_type == OrderType.TRAILING_STOP:
            ib_order = IBOrder()
            ib_order.action = action
            ib_order.totalQuantity = order.quantity
            ib_order.orderType = "TRAIL"

            if order.trail_percent:
                ib_order.trailingPercent = order.trail_percent * 100
            elif order.trail_amount:
                ib_order.auxPrice = order.trail_amount

        elif order.order_type == OrderType.TAKE_PROFIT:
            # IB no tiene take profit nativo, usar limit
            ib_order = LimitOrder(action, order.quantity, order.price or 0)

        else:
            # Default: limit
            ib_order = LimitOrder(action, order.quantity, order.price or 0)

        # Aplicar time in force
        self._apply_time_in_force(ib_order, order)

        # Aplicar client order ID
        if order.client_order_id:
            ib_order.orderRef = order.client_order_id

        return ib_order

    def _apply_time_in_force(self, ib_order: "IBOrder", order: BrokerOrder):
        """Aplicar time in force a la orden IB"""
        if order.time_in_force == TimeInForce.FOK:
            ib_order.tif = "FOK"
        elif order.time_in_force == TimeInForce.IOC:
            ib_order.tif = "IOC"
        elif order.time_in_force == TimeInForce.GTD:
            ib_order.tif = "GTD"
            if order.expire_time:
                ib_order.goodTillDate = order.expire_time.strftime("%Y%m%d %H:%M:%S")
        elif order.time_in_force == TimeInForce.DAY:
            ib_order.tif = "DAY"
        else:
            ib_order.tif = "GTC"

    def from_ib_trade(
        self,
        trade: "Trade",
        original_order: Optional[BrokerOrder] = None
    ) -> BrokerOrder:
        """
        Convertir Trade IB a BrokerOrder.

        Args:
            trade: Trade IB
            original_order: Orden original (opcional)

        Returns:
            BrokerOrder
        """
        ib_order = trade.order
        order_status = trade.orderStatus

        # Determinar lado
        side = OrderSide.BUY if ib_order.action == "BUY" else OrderSide.SELL

        # Determinar tipo
        order_type = self._from_ib_order_type(ib_order.orderType)

        # Determinar estado
        status = self._from_ib_status(order_status.status)

        order = BrokerOrder(
            id=str(ib_order.orderId),
            client_order_id=ib_order.orderRef,
            symbol="",  # Se debe llenar externamente
            side=side,
            order_type=order_type,
            quantity=float(ib_order.totalQuantity),
            price=ib_order.lmtPrice if hasattr(ib_order, 'lmtPrice') else None,
            stop_price=ib_order.auxPrice if hasattr(ib_order, 'auxPrice') else None,
            status=status,
            filled_quantity=float(order_status.filled),
            average_price=float(order_status.avgFillPrice),
        )

        # Copiar parametros de orden original si existe
        if original_order:
            order.symbol = original_order.symbol
            order.trail_percent = original_order.trail_percent
            order.trail_amount = original_order.trail_amount
            order.take_profit = original_order.take_profit
            order.stop_loss = original_order.stop_loss

        return order

    def _from_ib_order_type(self, ib_type: str) -> OrderType:
        """Convertir tipo IB a OrderType"""
        mapping = {
            "MKT": OrderType.MARKET,
            "LMT": OrderType.LIMIT,
            "STP": OrderType.STOP_LOSS,
            "STP LMT": OrderType.STOP_LIMIT,
            "TRAIL": OrderType.TRAILING_STOP,
            "TRAIL LIMIT": OrderType.TRAILING_STOP,
        }
        return mapping.get(ib_type.upper(), OrderType.LIMIT)

    def _from_ib_status(self, ib_status: str) -> OrderStatus:
        """Convertir status IB a OrderStatus"""
        mapping = {
            "Submitted": OrderStatus.OPEN,
            "Filled": OrderStatus.FILLED,
            "Cancelled": OrderStatus.CANCELLED,
            "Inactive": OrderStatus.REJECTED,
            "PendingSubmit": OrderStatus.PENDING,
            "PreSubmitted": OrderStatus.PENDING,
            "PendingCancel": OrderStatus.OPEN,
            "ApiCancelled": OrderStatus.CANCELLED,
            "ApiPending": OrderStatus.PENDING,
        }
        return mapping.get(ib_status, OrderStatus.PENDING)

    def to_execution_report(
        self,
        trade: "Trade",
        original_order: Optional[BrokerOrder] = None
    ) -> ExecutionReport:
        """
        Crear ExecutionReport desde Trade IB.

        Args:
            trade: Trade IB
            original_order: Orden original

        Returns:
            ExecutionReport
        """
        order_status = trade.orderStatus
        status = self._from_ib_status(order_status.status)

        # Calcular comision de fills
        commission = sum(f.commission for f in trade.fills if f.commission)

        return ExecutionReport(
            order_id=str(trade.order.orderId),
            client_order_id=trade.order.orderRef,
            status=status,
            filled_quantity=float(order_status.filled),
            remaining_quantity=float(order_status.remaining),
            average_price=float(order_status.avgFillPrice),
            commission=commission,
            timestamp=datetime.now(),
            original_order=original_order,
        )

    def from_ib_position(
        self,
        position: Any,
        current_price: float = 0
    ) -> BrokerPosition:
        """
        Convertir posicion IB a BrokerPosition.

        Args:
            position: Posicion IB (PortfolioItem o Position)
            current_price: Precio actual (opcional)

        Returns:
            BrokerPosition
        """
        # Determinar cantidad y lado
        qty = float(position.position)
        side = PositionSide.LONG if qty > 0 else PositionSide.SHORT

        # Obtener precio promedio
        avg_cost = float(position.avgCost) if hasattr(position, 'avgCost') else 0

        # Calcular PnL
        if current_price and avg_cost:
            if side == PositionSide.LONG:
                unrealized_pnl = (current_price - avg_cost) * abs(qty)
            else:
                unrealized_pnl = (avg_cost - current_price) * abs(qty)
        else:
            unrealized_pnl = float(getattr(position, 'unrealizedPNL', 0) or 0)

        return BrokerPosition(
            symbol=position.contract.symbol,
            side=side,
            quantity=abs(qty),
            entry_price=avg_cost,
            current_price=current_price or avg_cost,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=float(getattr(position, 'realizedPNL', 0) or 0),
        )

    def create_bracket_order(
        self,
        parent_order: BrokerOrder,
        take_profit: Optional[float] = None,
        stop_loss: Optional[float] = None
    ) -> tuple:
        """
        Crear orden bracket con parent, take profit y stop loss.

        Args:
            parent_order: Orden principal
            take_profit: Precio de take profit
            stop_loss: Precio de stop loss

        Returns:
            Tupla de (parent_ib_order, tp_ib_order, sl_ib_order)
        """
        # Orden principal
        parent = self.to_ib_order(parent_order)
        parent.transmit = False  # No transmitir hasta que todas esten listas

        action = "SELL" if parent_order.side == OrderSide.BUY else "BUY"
        orders = [parent]

        # Take profit
        if take_profit:
            tp_order = LimitOrder(action, parent_order.quantity, take_profit)
            tp_order.parentId = parent.orderId
            tp_order.transmit = False
            orders.append(tp_order)

        # Stop loss
        if stop_loss:
            sl_order = StopOrder(action, parent_order.quantity, stop_loss)
            sl_order.parentId = parent.orderId
            sl_order.transmit = True  # Ultima orden transmite todas
            orders.append(sl_order)

        # Si no hay SL, el TP debe transmitir
        if take_profit and not stop_loss:
            orders[-1].transmit = True

        return tuple(orders)
