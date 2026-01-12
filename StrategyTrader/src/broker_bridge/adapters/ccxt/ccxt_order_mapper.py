"""
Mapper de ordenes para CCXT.

Convierte BrokerOrder a parametros CCXT y viceversa.
"""

from typing import Dict, Any, Optional
from datetime import datetime

from ...core.enums import OrderType, OrderSide, OrderStatus, TimeInForce
from ...core.models import BrokerOrder, ExecutionReport


class CCXTOrderMapper:
    """
    Mapper para convertir ordenes entre formato interno y CCXT.
    """

    def __init__(self, exchange_id: str):
        """
        Inicializar mapper.

        Args:
            exchange_id: ID del exchange (ej: 'binance', 'bybit')
        """
        self.exchange_id = exchange_id.lower()

    def to_ccxt_order_type(self, order_type: OrderType) -> str:
        """
        Convertir OrderType interno a tipo CCXT.

        Args:
            order_type: Tipo de orden interno

        Returns:
            Tipo de orden CCXT
        """
        mapping = {
            OrderType.MARKET: "market",
            OrderType.LIMIT: "limit",
            OrderType.STOP_LOSS: "stop_loss",
            OrderType.STOP_LIMIT: "stop_loss_limit",
            OrderType.TAKE_PROFIT: "take_profit",
            OrderType.TRAILING_STOP: "trailing_stop_market",
        }
        return mapping.get(order_type, "limit")

    def to_ccxt_side(self, side: OrderSide) -> str:
        """
        Convertir OrderSide a lado CCXT.

        Args:
            side: Lado de la orden

        Returns:
            Lado CCXT ('buy' o 'sell')
        """
        return side.value.lower()

    def to_ccxt_params(self, order: BrokerOrder) -> Dict[str, Any]:
        """
        Generar parametros extra para CCXT.

        Args:
            order: Orden a convertir

        Returns:
            Diccionario de parametros para CCXT
        """
        params: Dict[str, Any] = {}

        # Time in force
        params.update(self._get_time_in_force_params(order))

        # Reduce only
        if order.reduce_only:
            params.update(self._get_reduce_only_params())

        # Post only
        if order.post_only:
            params.update(self._get_post_only_params())

        # Trailing stop
        if order.order_type == OrderType.TRAILING_STOP:
            params.update(self._get_trailing_params(order))

        # Bracket order (SL/TP)
        if order.stop_loss or order.take_profit:
            params.update(self._get_bracket_params(order))

        # Iceberg
        if order.order_type == OrderType.ICEBERG and order.display_quantity:
            params.update(self._get_iceberg_params(order))

        return params

    def _get_time_in_force_params(self, order: BrokerOrder) -> Dict[str, Any]:
        """Obtener parametros de time in force"""
        params = {}

        if order.time_in_force == TimeInForce.FOK:
            params["timeInForce"] = "FOK"
        elif order.time_in_force == TimeInForce.IOC:
            params["timeInForce"] = "IOC"
        elif order.time_in_force == TimeInForce.GTC:
            params["timeInForce"] = "GTC"

        return params

    def _get_reduce_only_params(self) -> Dict[str, Any]:
        """Obtener parametros para reduce only"""
        if self.exchange_id in ["binanceusdm", "binancecoinm"]:
            return {"reduceOnly": True}
        elif self.exchange_id == "bybit":
            return {"reduce_only": True}
        elif self.exchange_id == "okx":
            return {"reduceOnly": True}
        else:
            return {"reduceOnly": True}

    def _get_post_only_params(self) -> Dict[str, Any]:
        """Obtener parametros para post only"""
        if self.exchange_id == "binance":
            return {"timeInForce": "GTX"}
        elif self.exchange_id in ["binanceusdm", "binancecoinm"]:
            return {"timeInForce": "GTX"}
        elif self.exchange_id == "bybit":
            return {"time_in_force": "PostOnly"}
        else:
            return {"postOnly": True}

    def _get_trailing_params(self, order: BrokerOrder) -> Dict[str, Any]:
        """Obtener parametros para trailing stop"""
        params = {}

        if self.exchange_id == "binance":
            # Binance usa bps (basis points)
            if order.trail_percent:
                params["trailingDelta"] = int(order.trail_percent * 10000)
            if order.activation_price:
                params["activationPrice"] = order.activation_price

        elif self.exchange_id in ["binanceusdm", "binancecoinm"]:
            if order.trail_percent:
                params["callbackRate"] = order.trail_percent * 100
            if order.activation_price:
                params["activatePrice"] = order.activation_price

        elif self.exchange_id == "bybit":
            if order.trail_percent:
                params["trailingStop"] = str(order.trail_percent * 100)
            params["triggerDirection"] = 1 if order.side == OrderSide.SELL else 2

        elif self.exchange_id == "okx":
            if order.trail_percent:
                params["callbackRatio"] = str(order.trail_percent)
            if order.activation_price:
                params["activePx"] = str(order.activation_price)

        elif self.exchange_id == "kucoin":
            if order.trail_percent:
                params["trailingDelta"] = str(order.trail_percent * 100)

        return params

    def _get_bracket_params(self, order: BrokerOrder) -> Dict[str, Any]:
        """Obtener parametros para bracket order (SL/TP)"""
        params = {}

        if self.exchange_id == "bybit":
            if order.stop_loss:
                params["stopLoss"] = str(order.stop_loss)
            if order.take_profit:
                params["takeProfit"] = str(order.take_profit)

        elif self.exchange_id == "okx":
            if order.stop_loss:
                params["slTriggerPx"] = str(order.stop_loss)
                params["slOrdPx"] = "-1"  # Market
            if order.take_profit:
                params["tpTriggerPx"] = str(order.take_profit)
                params["tpOrdPx"] = "-1"  # Market

        elif self.exchange_id in ["binanceusdm", "binancecoinm"]:
            # Binance futures no soporta bracket nativo,
            # se debe manejar con ordenes separadas
            pass

        return params

    def _get_iceberg_params(self, order: BrokerOrder) -> Dict[str, Any]:
        """Obtener parametros para iceberg order"""
        params = {}

        if self.exchange_id == "binance":
            params["icebergQty"] = order.display_quantity

        elif self.exchange_id == "okx":
            params["szLimit"] = str(order.display_quantity)

        elif self.exchange_id == "kucoin":
            params["visibleSize"] = str(order.display_quantity)

        elif self.exchange_id == "htx":
            params["source"] = "spot-api"
            # HTX maneja iceberg de forma diferente

        return params

    def from_ccxt_order(
        self,
        ccxt_order: Dict[str, Any],
        original_order: Optional[BrokerOrder] = None
    ) -> BrokerOrder:
        """
        Convertir respuesta CCXT a BrokerOrder.

        Args:
            ccxt_order: Diccionario de orden CCXT
            original_order: Orden original (opcional)

        Returns:
            BrokerOrder con los datos de CCXT
        """
        # Determinar tipo de orden
        ccxt_type = ccxt_order.get("type", "limit")
        order_type = self._from_ccxt_order_type(ccxt_type)

        # Determinar lado
        side = OrderSide.BUY if ccxt_order.get("side") == "buy" else OrderSide.SELL

        # Determinar estado
        status = self._from_ccxt_status(ccxt_order.get("status"))

        order = BrokerOrder(
            id=str(ccxt_order.get("id", "")),
            client_order_id=ccxt_order.get("clientOrderId"),
            symbol=ccxt_order.get("symbol", ""),
            side=side,
            order_type=order_type,
            quantity=float(ccxt_order.get("amount", 0)),
            price=ccxt_order.get("price"),
            stop_price=ccxt_order.get("stopPrice"),
            status=status,
            filled_quantity=float(ccxt_order.get("filled", 0)),
            average_price=float(ccxt_order.get("average", 0) or 0),
        )

        # Copiar parametros avanzados de la orden original
        if original_order:
            order.trail_percent = original_order.trail_percent
            order.trail_amount = original_order.trail_amount
            order.take_profit = original_order.take_profit
            order.stop_loss = original_order.stop_loss
            order.reduce_only = original_order.reduce_only
            order.post_only = original_order.post_only
            order.time_in_force = original_order.time_in_force

        return order

    def _from_ccxt_order_type(self, ccxt_type: str) -> OrderType:
        """Convertir tipo CCXT a OrderType"""
        mapping = {
            "market": OrderType.MARKET,
            "limit": OrderType.LIMIT,
            "stop_loss": OrderType.STOP_LOSS,
            "stop_loss_limit": OrderType.STOP_LIMIT,
            "take_profit": OrderType.TAKE_PROFIT,
            "take_profit_limit": OrderType.TAKE_PROFIT,
            "trailing_stop_market": OrderType.TRAILING_STOP,
            "stop": OrderType.STOP_LOSS,
            "stop_market": OrderType.STOP_LOSS,
        }
        return mapping.get(ccxt_type.lower(), OrderType.LIMIT)

    def _from_ccxt_status(self, ccxt_status: Optional[str]) -> OrderStatus:
        """Convertir status CCXT a OrderStatus"""
        if not ccxt_status:
            return OrderStatus.PENDING

        mapping = {
            "open": OrderStatus.OPEN,
            "closed": OrderStatus.FILLED,
            "canceled": OrderStatus.CANCELLED,
            "cancelled": OrderStatus.CANCELLED,
            "expired": OrderStatus.EXPIRED,
            "rejected": OrderStatus.REJECTED,
        }
        return mapping.get(ccxt_status.lower(), OrderStatus.PENDING)

    def to_execution_report(
        self,
        ccxt_result: Dict[str, Any],
        original_order: Optional[BrokerOrder] = None
    ) -> ExecutionReport:
        """
        Crear ExecutionReport desde respuesta CCXT.

        Args:
            ccxt_result: Resultado de orden CCXT
            original_order: Orden original

        Returns:
            ExecutionReport con los datos
        """
        status = self._from_ccxt_status(ccxt_result.get("status"))

        # Calcular comision
        fee = ccxt_result.get("fee", {})
        commission = float(fee.get("cost", 0)) if fee else 0

        return ExecutionReport(
            order_id=str(ccxt_result.get("id", "")),
            client_order_id=ccxt_result.get("clientOrderId"),
            status=status,
            filled_quantity=float(ccxt_result.get("filled", 0)),
            remaining_quantity=float(ccxt_result.get("remaining", 0) or 0),
            average_price=float(ccxt_result.get("average", 0) or 0),
            commission=commission,
            timestamp=datetime.now(),
            original_order=original_order,
        )
