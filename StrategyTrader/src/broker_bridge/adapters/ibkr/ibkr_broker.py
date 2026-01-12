"""
Implementacion del adaptador para Interactive Brokers.

Proporciona acceso a mercados tradicionales (stocks, indices,
forex, futuros, opciones) a traves de IB/TWS.
"""

import asyncio
from typing import Optional, List, Dict, Any
from datetime import datetime

try:
    from ib_insync import IB, Contract, util
    IB_AVAILABLE = True
except ImportError:
    IB = Contract = util = None
    IB_AVAILABLE = False

from ...core.enums import (
    BrokerType, AssetClass, OrderType, OrderSide, OrderStatus, PositionSide
)
from ...core.models import (
    BrokerCapabilities, BrokerOrder, BrokerPosition,
    ExecutionReport, AccountInfo
)
from ...core.interfaces import IBrokerAdapter
from ...core.exceptions import (
    BrokerConnectionError, OrderError, OrderNotFoundError,
    SymbolNotFoundError, BrokerNotConnectedError,
    ContractQualificationError
)
from .ibkr_contracts import IBKRContractFactory
from .ibkr_order_mapper import IBKROrderMapper


class IBKRBroker(IBrokerAdapter):
    """
    Adaptador para Interactive Brokers via ib_insync.

    Soporta stocks, indices, forex, futuros y opciones.

    Puertos por defecto:
    - TWS Paper: 7497
    - TWS Live: 7496
    - Gateway Paper: 4002
    - Gateway Live: 4001
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7497,
        client_id: int = 1,
        readonly: bool = False,
        timeout: float = 60.0
    ):
        """
        Inicializar adaptador IBKR.

        Args:
            host: Host de TWS/Gateway
            port: Puerto de conexion
            client_id: ID de cliente (unico por conexion)
            readonly: Modo solo lectura (sin trading)
            timeout: Timeout para operaciones en segundos
        """
        if not IB_AVAILABLE:
            raise ImportError(
                "ib_insync library is not installed. "
                "Install it with: pip install ib_insync"
            )

        self._host = host
        self._port = port
        self._client_id = client_id
        self._readonly = readonly
        self._timeout = timeout

        self._ib = IB()
        self._connected = False

        self._contract_factory = IBKRContractFactory()
        self._order_mapper = IBKROrderMapper()

        # Cache de contratos cualificados
        self._contracts: Dict[str, Contract] = {}

        # Tracking de ordenes
        self._orders: Dict[str, BrokerOrder] = {}

    @property
    def broker_type(self) -> BrokerType:
        return BrokerType.IBKR

    @property
    def broker_id(self) -> str:
        return "ibkr"

    @property
    def is_connected(self) -> bool:
        return self._ib.isConnected()

    def _ensure_connected(self):
        """Verificar que estamos conectados"""
        if not self.is_connected:
            raise BrokerNotConnectedError("ibkr")

    # ==================== Connection ====================

    async def connect(self) -> bool:
        """Conectar a TWS/Gateway"""
        try:
            await self._ib.connectAsync(
                self._host,
                self._port,
                clientId=self._client_id,
                readonly=self._readonly,
                timeout=self._timeout
            )
            self._connected = True
            return True

        except asyncio.TimeoutError:
            raise BrokerConnectionError(
                f"Connection timeout to IBKR at {self._host}:{self._port}",
                broker_id="ibkr",
                host=self._host,
                port=self._port
            )
        except Exception as e:
            raise BrokerConnectionError(
                f"Failed to connect to IBKR: {e}",
                broker_id="ibkr",
                host=self._host,
                port=self._port
            )

    async def disconnect(self) -> None:
        """Desconectar de TWS/Gateway"""
        self._ib.disconnect()
        self._connected = False
        self._contracts.clear()

    def get_capabilities(self, exchange_id: Optional[str] = None) -> BrokerCapabilities:
        """Obtener capacidades de IBKR"""
        return BrokerCapabilities(
            broker_type=BrokerType.IBKR,
            exchange_id="ibkr",
            # Basicos
            supports_market=True,
            supports_limit=True,
            supports_stop_loss=True,
            supports_stop_limit=True,
            supports_take_profit=True,
            # Avanzados
            supports_trailing_stop=True,
            supports_bracket=True,
            supports_oco=True,
            supports_twap=False,
            supports_vwap=False,
            supports_iceberg=True,
            supports_hidden=True,
            # Time in force
            supports_gtc=True,
            supports_fok=True,
            supports_ioc=True,
            supports_gtd=True,
            supports_day=True,
            # Features
            supports_reduce_only=False,
            supports_post_only=False,
            supports_hedging=False,
            supports_leverage=True,
            supports_margin=True,
            # Asset classes
            asset_classes=[
                AssetClass.STOCK,
                AssetClass.INDEX,
                AssetClass.FOREX,
                AssetClass.FUTURES,
                AssetClass.OPTIONS
            ],
            max_orders_per_second=50.0,
        )

    # ==================== Contract Management ====================

    async def _get_contract(
        self,
        symbol: str,
        asset_class: Optional[AssetClass] = None
    ) -> Contract:
        """
        Obtener contrato cualificado.

        Args:
            symbol: Simbolo del instrumento
            asset_class: Clase de activo (opcional, se detecta automaticamente)

        Returns:
            Contrato IB cualificado
        """
        # Verificar cache
        if symbol in self._contracts:
            return self._contracts[symbol]

        # Parsear simbolo
        if asset_class is None:
            base_symbol, asset_class, kwargs = self._contract_factory.parse_symbol(symbol)
        else:
            base_symbol = symbol
            kwargs = {}

        # Crear contrato
        contract = self._contract_factory.create_contract(
            base_symbol, asset_class, **kwargs
        )

        # Cualificar contrato
        try:
            qualified = await self._ib.qualifyContractsAsync(contract)
            if not qualified:
                raise ContractQualificationError(
                    symbol, "ibkr", "No matching contract found"
                )
            self._contracts[symbol] = qualified[0]
            return qualified[0]

        except Exception as e:
            raise ContractQualificationError(symbol, "ibkr", str(e))

    # ==================== Orders ====================

    async def submit_order(self, order: BrokerOrder) -> ExecutionReport:
        """Enviar orden"""
        self._ensure_connected()

        try:
            # Obtener contrato
            contract = await self._get_contract(order.symbol)

            # Convertir a orden IB
            ib_order = self._order_mapper.to_ib_order(order)

            # Enviar orden
            trade = self._ib.placeOrder(contract, ib_order)

            # Esperar confirmacion inicial
            await asyncio.sleep(0.2)

            # Guardar referencia
            order.id = str(trade.order.orderId)
            self._orders[order.id] = order

            return self._order_mapper.to_execution_report(trade, order)

        except ContractQualificationError:
            raise
        except Exception as e:
            raise OrderError(str(e), order.id, "ibkr")

    async def cancel_order(
        self,
        order_id: str,
        symbol: Optional[str] = None
    ) -> bool:
        """Cancelar orden"""
        self._ensure_connected()

        try:
            # Buscar trade
            for trade in self._ib.openTrades():
                if str(trade.order.orderId) == order_id:
                    self._ib.cancelOrder(trade.order)
                    return True

            return False

        except Exception as e:
            raise OrderError(str(e), order_id, "ibkr")

    async def modify_order(
        self,
        order_id: str,
        modifications: Dict[str, Any],
        symbol: Optional[str] = None
    ) -> ExecutionReport:
        """Modificar orden"""
        self._ensure_connected()

        try:
            # Buscar trade
            for trade in self._ib.openTrades():
                if str(trade.order.orderId) == order_id:
                    ib_order = trade.order

                    # Aplicar modificaciones
                    if 'price' in modifications:
                        ib_order.lmtPrice = modifications['price']
                    if 'quantity' in modifications:
                        ib_order.totalQuantity = modifications['quantity']
                    if 'stop_price' in modifications:
                        ib_order.auxPrice = modifications['stop_price']

                    # Reenviar
                    self._ib.placeOrder(trade.contract, ib_order)
                    await asyncio.sleep(0.2)

                    return self._order_mapper.to_execution_report(trade)

            raise OrderNotFoundError(order_id=order_id)

        except OrderNotFoundError:
            raise
        except Exception as e:
            raise OrderError(str(e), order_id, "ibkr")

    async def get_order(
        self,
        order_id: str,
        symbol: Optional[str] = None
    ) -> Optional[BrokerOrder]:
        """Obtener estado de orden"""
        self._ensure_connected()

        # Buscar en trades abiertos
        for trade in self._ib.openTrades():
            if str(trade.order.orderId) == order_id:
                original = self._orders.get(order_id)
                return self._order_mapper.from_ib_trade(trade, original)

        # Buscar en ordenes guardadas
        return self._orders.get(order_id)

    async def get_open_orders(
        self,
        symbol: Optional[str] = None
    ) -> List[BrokerOrder]:
        """Obtener ordenes abiertas"""
        self._ensure_connected()

        result = []
        for trade in self._ib.openTrades():
            if symbol and trade.contract.symbol != symbol:
                continue

            original = self._orders.get(str(trade.order.orderId))
            order = self._order_mapper.from_ib_trade(trade, original)
            order.symbol = trade.contract.symbol
            result.append(order)

        return result

    async def get_order_history(
        self,
        symbol: Optional[str] = None,
        limit: int = 100
    ) -> List[BrokerOrder]:
        """Obtener historial de ordenes"""
        self._ensure_connected()

        # IB no tiene un metodo simple para historial
        # Retornamos las ordenes que tenemos en cache
        orders = list(self._orders.values())

        if symbol:
            orders = [o for o in orders if o.symbol == symbol]

        return orders[:limit]

    # ==================== Positions ====================

    async def get_positions(
        self,
        symbol: Optional[str] = None
    ) -> List[BrokerPosition]:
        """Obtener posiciones"""
        self._ensure_connected()

        positions = self._ib.positions()

        result = []
        for pos in positions:
            if pos.position == 0:
                continue

            if symbol and pos.contract.symbol != symbol:
                continue

            # Obtener precio actual
            ticker = self._ib.reqMktData(pos.contract)
            await asyncio.sleep(0.5)
            current_price = ticker.last or ticker.close or pos.avgCost

            result.append(self._order_mapper.from_ib_position(pos, current_price))

        return result

    async def close_position(
        self,
        symbol: str,
        quantity: Optional[float] = None
    ) -> ExecutionReport:
        """Cerrar posicion"""
        self._ensure_connected()

        # Obtener posicion
        positions = await self.get_positions(symbol)
        if not positions:
            raise OrderError(f"No position found for {symbol}", broker_id="ibkr")

        position = positions[0]
        close_qty = quantity or position.quantity

        # Determinar lado
        close_side = OrderSide.SELL if position.side == PositionSide.LONG else OrderSide.BUY

        # Crear orden de cierre
        order = BrokerOrder(
            symbol=symbol,
            side=close_side,
            order_type=OrderType.MARKET,
            quantity=close_qty,
        )

        return await self.submit_order(order)

    # ==================== Account ====================

    async def get_balance(self) -> Dict[str, float]:
        """Obtener balance"""
        self._ensure_connected()

        account_values = self._ib.accountValues()

        result = {}
        for av in account_values:
            if av.tag == "CashBalance":
                result[av.currency] = float(av.value)
            elif av.tag == "TotalCashBalance" and av.currency == "BASE":
                result["TOTAL"] = float(av.value)

        return result

    async def get_account_info(self) -> AccountInfo:
        """Obtener informacion de cuenta"""
        self._ensure_connected()

        account_values = self._ib.accountValues()
        account_summary = self._ib.accountSummary()

        # Extraer valores
        balances = {}
        total_balance = 0
        available_balance = 0
        margin_used = 0

        for av in account_values:
            if av.tag == "CashBalance":
                balances[av.currency] = float(av.value)
            elif av.tag == "NetLiquidation":
                total_balance = float(av.value)
            elif av.tag == "AvailableFunds":
                available_balance = float(av.value)
            elif av.tag == "MaintMarginReq":
                margin_used = float(av.value)

        # Obtener account ID
        account_id = ""
        for av in account_values:
            if av.account:
                account_id = av.account
                break

        return AccountInfo(
            broker_type=BrokerType.IBKR,
            account_id=account_id,
            total_balance=total_balance,
            available_balance=available_balance,
            margin_used=margin_used,
            balances=balances,
        )

    # ==================== Market Data ====================

    async def get_ticker(self, symbol: str) -> Dict[str, float]:
        """Obtener ticker"""
        self._ensure_connected()

        try:
            contract = await self._get_contract(symbol)
            ticker = self._ib.reqMktData(contract)

            # Esperar datos
            await asyncio.sleep(0.5)

            return {
                "bid": float(ticker.bid or 0),
                "ask": float(ticker.ask or 0),
                "last": float(ticker.last or ticker.close or 0),
                "volume": float(ticker.volume or 0),
                "high": float(ticker.high or 0),
                "low": float(ticker.low or 0),
            }

        except ContractQualificationError:
            raise SymbolNotFoundError(symbol, "ibkr")

    async def get_orderbook(
        self,
        symbol: str,
        limit: int = 20
    ) -> Dict[str, List]:
        """Obtener order book"""
        self._ensure_connected()

        try:
            contract = await self._get_contract(symbol)

            # Solicitar market depth
            self._ib.reqMktDepth(contract, numRows=limit)
            await asyncio.sleep(0.5)

            ticker = self._ib.ticker(contract)

            bids = []
            asks = []

            if ticker and ticker.domBids:
                bids = [[d.price, d.size] for d in ticker.domBids[:limit]]
            if ticker and ticker.domAsks:
                asks = [[d.price, d.size] for d in ticker.domAsks[:limit]]

            return {"bids": bids, "asks": asks}

        except ContractQualificationError:
            raise SymbolNotFoundError(symbol, "ibkr")

    # ==================== Context Manager ====================

    async def __aenter__(self):
        """Soporte para async context manager"""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Limpiar al salir del context"""
        await self.disconnect()
