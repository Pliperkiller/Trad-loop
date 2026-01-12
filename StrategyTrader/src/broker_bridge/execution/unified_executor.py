"""
Executor unificado para broker_bridge.

Orquesta la ejecucion de ordenes a traves de multiples brokers,
manejando ruteo automatico y fallback a simulacion.
"""

from typing import Optional, Dict, List, Any
from datetime import datetime

from ..core.enums import BrokerType
from ..core.models import BrokerOrder, BrokerPosition, ExecutionReport, AccountInfo
from ..core.interfaces import IBrokerAdapter
from ..core.exceptions import (
    BrokerNotRegisteredError, BrokerNotConnectedError, BrokerError
)
from .symbol_router import SymbolRouter
from .fallback_simulator import FallbackSimulator


class UnifiedExecutor:
    """
    Executor unificado para multiples brokers.

    Maneja:
    - Registro y conexion de brokers
    - Ruteo automatico de ordenes por simbolo
    - Fallback a simulacion para ordenes no soportadas
    - Agregacion de posiciones y balances
    """

    def __init__(self, auto_route: bool = True):
        """
        Inicializar executor.

        Args:
            auto_route: Rutear automaticamente por simbolo
        """
        self._brokers: Dict[BrokerType, IBrokerAdapter] = {}
        self._fallback_simulators: Dict[BrokerType, FallbackSimulator] = {}
        self._router = SymbolRouter()
        self._auto_route = auto_route

        # Tracking
        self._orders: Dict[str, BrokerOrder] = {}
        self._executions: List[ExecutionReport] = []

    # ==================== Broker Management ====================

    def register_broker(self, broker: IBrokerAdapter):
        """
        Registrar un broker.

        Args:
            broker: Instancia del broker a registrar
        """
        self._brokers[broker.broker_type] = broker
        self._fallback_simulators[broker.broker_type] = FallbackSimulator(broker)

    def unregister_broker(self, broker_type: BrokerType):
        """
        Desregistrar un broker.

        Args:
            broker_type: Tipo de broker a desregistrar
        """
        self._brokers.pop(broker_type, None)
        self._fallback_simulators.pop(broker_type, None)

    def get_broker(self, broker_type: BrokerType) -> Optional[IBrokerAdapter]:
        """
        Obtener broker registrado.

        Args:
            broker_type: Tipo de broker

        Returns:
            Broker o None si no esta registrado
        """
        return self._brokers.get(broker_type)

    def get_registered_brokers(self) -> List[BrokerType]:
        """Obtener lista de brokers registrados"""
        return list(self._brokers.keys())

    # ==================== Connection ====================

    async def connect_all(self) -> Dict[BrokerType, bool]:
        """
        Conectar todos los brokers registrados.

        Returns:
            Diccionario {BrokerType: success}
        """
        results = {}
        for broker_type, broker in self._brokers.items():
            try:
                results[broker_type] = await broker.connect()
            except Exception as e:
                results[broker_type] = False
        return results

    async def disconnect_all(self):
        """Desconectar todos los brokers"""
        for broker in self._brokers.values():
            try:
                await broker.disconnect()
            except Exception:
                pass

    async def connect_broker(self, broker_type: BrokerType) -> bool:
        """
        Conectar un broker especifico.

        Args:
            broker_type: Tipo de broker

        Returns:
            True si la conexion fue exitosa
        """
        broker = self._brokers.get(broker_type)
        if not broker:
            raise BrokerNotRegisteredError(broker_type.value)

        return await broker.connect()

    async def disconnect_broker(self, broker_type: BrokerType):
        """
        Desconectar un broker especifico.

        Args:
            broker_type: Tipo de broker
        """
        broker = self._brokers.get(broker_type)
        if broker:
            await broker.disconnect()

    # ==================== Order Execution ====================

    async def submit_order(
        self,
        order: BrokerOrder,
        broker_type: Optional[BrokerType] = None,
        use_fallback: bool = True
    ) -> ExecutionReport:
        """
        Enviar orden.

        Args:
            order: Orden a enviar
            broker_type: Tipo de broker (opcional, se detecta automaticamente)
            use_fallback: Usar fallback a simulacion si no es soportado

        Returns:
            ExecutionReport con el resultado
        """
        # Determinar broker
        if broker_type is None and self._auto_route:
            broker_type, _ = self._router.route(order.symbol)

        if broker_type is None:
            raise BrokerError("Could not determine broker for order")

        # Verificar que el broker esta registrado
        if broker_type not in self._brokers:
            raise BrokerNotRegisteredError(broker_type.value)

        broker = self._brokers[broker_type]

        # Verificar conexion
        if not broker.is_connected:
            raise BrokerNotConnectedError(broker.broker_id)

        # Obtener capacidades
        capabilities = broker.get_capabilities()

        # Ejecutar con o sin fallback
        if use_fallback:
            fallback = self._fallback_simulators[broker_type]
            report = await fallback.execute_with_fallback(order, capabilities)
        else:
            report = await broker.submit_order(order)

        # Guardar tracking
        self._orders[report.order_id] = order
        self._executions.append(report)

        return report

    async def cancel_order(
        self,
        order_id: str,
        broker_type: Optional[BrokerType] = None
    ) -> bool:
        """
        Cancelar orden.

        Args:
            order_id: ID de la orden
            broker_type: Tipo de broker (opcional)

        Returns:
            True si se cancelo
        """
        # Si conocemos la orden, podemos determinar el broker
        if order_id in self._orders and broker_type is None:
            order = self._orders[order_id]
            broker_type, _ = self._router.route(order.symbol)

        # Intentar cancelar en cada broker si no conocemos el tipo
        if broker_type:
            broker = self._brokers.get(broker_type)
            if broker:
                # Primero intentar cancelar simulada
                fallback = self._fallback_simulators.get(broker_type)
                if fallback and fallback.cancel_simulated_order(order_id):
                    return True

                # Luego cancelar en broker real
                return await broker.cancel_order(order_id)

        # Intentar en todos los brokers
        for bt, broker in self._brokers.items():
            try:
                # Intentar simulada primero
                fallback = self._fallback_simulators.get(bt)
                if fallback and fallback.cancel_simulated_order(order_id):
                    return True

                if await broker.cancel_order(order_id):
                    return True
            except Exception:
                continue

        return False

    async def get_order(
        self,
        order_id: str,
        broker_type: Optional[BrokerType] = None
    ) -> Optional[BrokerOrder]:
        """
        Obtener estado de orden.

        Args:
            order_id: ID de la orden
            broker_type: Tipo de broker (opcional)

        Returns:
            BrokerOrder o None
        """
        # Verificar cache local
        if order_id in self._orders:
            return self._orders[order_id]

        # Buscar en brokers
        brokers_to_check = (
            [self._brokers[broker_type]] if broker_type and broker_type in self._brokers
            else self._brokers.values()
        )

        for broker in brokers_to_check:
            try:
                order = await broker.get_order(order_id)
                if order:
                    return order
            except Exception:
                continue

        return None

    async def get_open_orders(
        self,
        symbol: Optional[str] = None,
        broker_type: Optional[BrokerType] = None
    ) -> Dict[BrokerType, List[BrokerOrder]]:
        """
        Obtener ordenes abiertas.

        Args:
            symbol: Filtrar por simbolo (opcional)
            broker_type: Filtrar por broker (opcional)

        Returns:
            Diccionario {BrokerType: [ordenes]}
        """
        result: Dict[BrokerType, List[BrokerOrder]] = {}

        brokers = (
            {broker_type: self._brokers[broker_type]}
            if broker_type and broker_type in self._brokers
            else self._brokers
        )

        for bt, broker in brokers.items():
            if broker.is_connected:
                try:
                    orders = await broker.get_open_orders(symbol)
                    result[bt] = orders
                except Exception:
                    result[bt] = []

        return result

    # ==================== Positions ====================

    async def get_all_positions(
        self,
        symbol: Optional[str] = None
    ) -> Dict[BrokerType, List[BrokerPosition]]:
        """
        Obtener posiciones de todos los brokers.

        Args:
            symbol: Filtrar por simbolo (opcional)

        Returns:
            Diccionario {BrokerType: [posiciones]}
        """
        result: Dict[BrokerType, List[BrokerPosition]] = {}

        for broker_type, broker in self._brokers.items():
            if broker.is_connected:
                try:
                    positions = await broker.get_positions(symbol)
                    result[broker_type] = positions
                except Exception:
                    result[broker_type] = []

        return result

    async def get_positions(
        self,
        broker_type: BrokerType,
        symbol: Optional[str] = None
    ) -> List[BrokerPosition]:
        """
        Obtener posiciones de un broker.

        Args:
            broker_type: Tipo de broker
            symbol: Filtrar por simbolo (opcional)

        Returns:
            Lista de posiciones
        """
        broker = self._brokers.get(broker_type)
        if not broker:
            raise BrokerNotRegisteredError(broker_type.value)

        return await broker.get_positions(symbol)

    async def close_position(
        self,
        symbol: str,
        quantity: Optional[float] = None,
        broker_type: Optional[BrokerType] = None
    ) -> ExecutionReport:
        """
        Cerrar posicion.

        Args:
            symbol: Simbolo
            quantity: Cantidad a cerrar (None = todo)
            broker_type: Broker (opcional, se detecta)

        Returns:
            ExecutionReport
        """
        if broker_type is None:
            broker_type, _ = self._router.route(symbol)

        broker = self._brokers.get(broker_type)
        if not broker:
            raise BrokerNotRegisteredError(broker_type.value)

        return await broker.close_position(symbol, quantity)

    # ==================== Account ====================

    async def get_all_balances(self) -> Dict[BrokerType, Dict[str, float]]:
        """
        Obtener balances de todos los brokers.

        Returns:
            Diccionario {BrokerType: {asset: balance}}
        """
        result: Dict[BrokerType, Dict[str, float]] = {}

        for broker_type, broker in self._brokers.items():
            if broker.is_connected:
                try:
                    balance = await broker.get_balance()
                    result[broker_type] = balance
                except Exception:
                    result[broker_type] = {}

        return result

    async def get_balance(
        self,
        broker_type: BrokerType
    ) -> Dict[str, float]:
        """
        Obtener balance de un broker.

        Args:
            broker_type: Tipo de broker

        Returns:
            Diccionario {asset: balance}
        """
        broker = self._brokers.get(broker_type)
        if not broker:
            raise BrokerNotRegisteredError(broker_type.value)

        return await broker.get_balance()

    async def get_all_account_info(self) -> Dict[BrokerType, AccountInfo]:
        """
        Obtener informacion de cuenta de todos los brokers.

        Returns:
            Diccionario {BrokerType: AccountInfo}
        """
        result: Dict[BrokerType, AccountInfo] = {}

        for broker_type, broker in self._brokers.items():
            if broker.is_connected:
                try:
                    info = await broker.get_account_info()
                    result[broker_type] = info
                except Exception:
                    pass

        return result

    # ==================== Market Data ====================

    async def get_ticker(
        self,
        symbol: str,
        broker_type: Optional[BrokerType] = None
    ) -> Dict[str, float]:
        """
        Obtener ticker.

        Args:
            symbol: Simbolo
            broker_type: Broker (opcional, se detecta)

        Returns:
            Diccionario con datos del ticker
        """
        if broker_type is None:
            broker_type, _ = self._router.route(symbol)

        broker = self._brokers.get(broker_type)
        if not broker:
            raise BrokerNotRegisteredError(broker_type.value)

        return await broker.get_ticker(symbol)

    async def get_orderbook(
        self,
        symbol: str,
        limit: int = 20,
        broker_type: Optional[BrokerType] = None
    ) -> Dict[str, List]:
        """
        Obtener order book.

        Args:
            symbol: Simbolo
            limit: Profundidad
            broker_type: Broker (opcional, se detecta)

        Returns:
            Diccionario con bids y asks
        """
        if broker_type is None:
            broker_type, _ = self._router.route(symbol)

        broker = self._brokers.get(broker_type)
        if not broker:
            raise BrokerNotRegisteredError(broker_type.value)

        return await broker.get_orderbook(symbol, limit)

    # ==================== Utilities ====================

    def get_router(self) -> SymbolRouter:
        """Obtener router de simbolos"""
        return self._router

    def get_execution_history(self) -> List[ExecutionReport]:
        """Obtener historial de ejecuciones"""
        return list(self._executions)

    def get_order_cache(self) -> Dict[str, BrokerOrder]:
        """Obtener cache de ordenes"""
        return dict(self._orders)

    async def process_price_updates(
        self,
        prices: Dict[str, float],
        timestamp: Optional[datetime] = None
    ) -> List[ExecutionReport]:
        """
        Procesar actualizaciones de precio para ordenes simuladas.

        Args:
            prices: Diccionario {symbol: price}
            timestamp: Timestamp

        Returns:
            Lista de ejecuciones de ordenes simuladas
        """
        executions = []
        timestamp = timestamp or datetime.now()

        for broker_type, fallback in self._fallback_simulators.items():
            for symbol, price in prices.items():
                execs = await fallback.process_price_update(symbol, price, timestamp)
                executions.extend(execs)

        return executions

    # ==================== Context Manager ====================

    async def __aenter__(self):
        """Soporte para async context manager"""
        await self.connect_all()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Limpiar al salir del context"""
        await self.disconnect_all()
