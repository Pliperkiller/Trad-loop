"""
Interfaces para el modulo broker_bridge.

Define la interfaz abstracta IBrokerAdapter que todos
los adaptadores de broker deben implementar.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any

from .enums import BrokerType
from .models import (
    BrokerCapabilities,
    BrokerOrder,
    BrokerPosition,
    ExecutionReport,
    AccountInfo
)


class IBrokerAdapter(ABC):
    """
    Interfaz abstracta para adaptadores de broker.

    Define el contrato que todos los brokers (CCXT, IBKR, etc.)
    deben implementar para integrarse con el sistema.
    """

    # ==================== Properties ====================

    @property
    @abstractmethod
    def broker_type(self) -> BrokerType:
        """Tipo de broker"""
        pass

    @property
    @abstractmethod
    def broker_id(self) -> str:
        """Identificador unico del broker (ej: 'binance', 'ibkr')"""
        pass

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Estado de conexion"""
        pass

    # ==================== Connection ====================

    @abstractmethod
    async def connect(self) -> bool:
        """
        Conectar al broker.

        Returns:
            True si la conexion fue exitosa

        Raises:
            BrokerConnectionError: Si no se puede conectar
            AuthenticationError: Si las credenciales son invalidas
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """
        Desconectar del broker.

        Limpia recursos y cierra la conexion de forma ordenada.
        """
        pass

    @abstractmethod
    def get_capabilities(self, exchange_id: Optional[str] = None) -> BrokerCapabilities:
        """
        Obtener capacidades del broker.

        Args:
            exchange_id: ID del exchange especifico (opcional)

        Returns:
            BrokerCapabilities con las capacidades soportadas
        """
        pass

    # ==================== Orders ====================

    @abstractmethod
    async def submit_order(self, order: BrokerOrder) -> ExecutionReport:
        """
        Enviar orden al broker.

        Args:
            order: Orden a enviar

        Returns:
            ExecutionReport con el resultado de la orden

        Raises:
            OrderError: Si hay error al enviar la orden
            InsufficientFundsError: Si no hay fondos suficientes
            UnsupportedOrderTypeError: Si el tipo de orden no es soportado
        """
        pass

    @abstractmethod
    async def cancel_order(
        self,
        order_id: str,
        symbol: Optional[str] = None
    ) -> bool:
        """
        Cancelar orden.

        Args:
            order_id: ID de la orden a cancelar
            symbol: Simbolo de la orden (requerido por algunos brokers)

        Returns:
            True si la orden fue cancelada exitosamente

        Raises:
            OrderNotFoundError: Si la orden no existe
        """
        pass

    @abstractmethod
    async def modify_order(
        self,
        order_id: str,
        modifications: Dict[str, Any],
        symbol: Optional[str] = None
    ) -> ExecutionReport:
        """
        Modificar orden existente.

        Args:
            order_id: ID de la orden a modificar
            modifications: Diccionario con los campos a modificar
            symbol: Simbolo de la orden (requerido por algunos brokers)

        Returns:
            ExecutionReport con el resultado

        Raises:
            OrderNotFoundError: Si la orden no existe
            OrderError: Si no se puede modificar
        """
        pass

    @abstractmethod
    async def get_order(
        self,
        order_id: str,
        symbol: Optional[str] = None
    ) -> Optional[BrokerOrder]:
        """
        Obtener estado de una orden.

        Args:
            order_id: ID de la orden
            symbol: Simbolo de la orden (requerido por algunos brokers)

        Returns:
            BrokerOrder con el estado actual, o None si no existe
        """
        pass

    @abstractmethod
    async def get_open_orders(
        self,
        symbol: Optional[str] = None
    ) -> List[BrokerOrder]:
        """
        Obtener ordenes abiertas.

        Args:
            symbol: Filtrar por simbolo (opcional)

        Returns:
            Lista de ordenes abiertas
        """
        pass

    @abstractmethod
    async def get_order_history(
        self,
        symbol: Optional[str] = None,
        limit: int = 100
    ) -> List[BrokerOrder]:
        """
        Obtener historial de ordenes.

        Args:
            symbol: Filtrar por simbolo (opcional)
            limit: Numero maximo de ordenes a retornar

        Returns:
            Lista de ordenes historicas
        """
        pass

    # ==================== Positions ====================

    @abstractmethod
    async def get_positions(
        self,
        symbol: Optional[str] = None
    ) -> List[BrokerPosition]:
        """
        Obtener posiciones abiertas.

        Args:
            symbol: Filtrar por simbolo (opcional)

        Returns:
            Lista de posiciones
        """
        pass

    @abstractmethod
    async def close_position(
        self,
        symbol: str,
        quantity: Optional[float] = None
    ) -> ExecutionReport:
        """
        Cerrar posicion.

        Args:
            symbol: Simbolo de la posicion
            quantity: Cantidad a cerrar (None = cerrar todo)

        Returns:
            ExecutionReport con el resultado

        Raises:
            PositionNotFoundError: Si no hay posicion
        """
        pass

    # ==================== Account ====================

    @abstractmethod
    async def get_balance(self) -> Dict[str, float]:
        """
        Obtener balance de la cuenta.

        Returns:
            Diccionario {asset: balance_disponible}
        """
        pass

    @abstractmethod
    async def get_account_info(self) -> AccountInfo:
        """
        Obtener informacion completa de la cuenta.

        Returns:
            AccountInfo con toda la informacion de la cuenta
        """
        pass

    # ==================== Market Data ====================

    @abstractmethod
    async def get_ticker(self, symbol: str) -> Dict[str, float]:
        """
        Obtener ticker actual.

        Args:
            symbol: Simbolo a consultar

        Returns:
            Diccionario con bid, ask, last, volume

        Raises:
            SymbolNotFoundError: Si el simbolo no existe
        """
        pass

    @abstractmethod
    async def get_orderbook(
        self,
        symbol: str,
        limit: int = 20
    ) -> Dict[str, List]:
        """
        Obtener order book.

        Args:
            symbol: Simbolo a consultar
            limit: Niveles de profundidad

        Returns:
            Diccionario con 'bids' y 'asks'
        """
        pass

    # ==================== Utility ====================

    async def ping(self) -> bool:
        """
        Verificar conectividad con el broker.

        Returns:
            True si el broker responde
        """
        try:
            await self.get_balance()
            return True
        except Exception:
            return False

    def __repr__(self) -> str:
        status = "connected" if self.is_connected else "disconnected"
        return f"<{self.__class__.__name__}({self.broker_id}) [{status}]>"
