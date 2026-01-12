"""
Excepciones para el modulo broker_bridge.

Define excepciones especificas para errores de conexion,
ordenes, autenticacion y otros errores de broker.
"""

from typing import Optional


class BrokerError(Exception):
    """
    Error base para todos los errores de broker.

    Todas las excepciones especificas heredan de esta clase.
    """

    def __init__(self, message: str, broker_id: Optional[str] = None):
        super().__init__(message)
        self.broker_id = broker_id


class BrokerConnectionError(BrokerError):
    """
    Error de conexion al broker.

    Se lanza cuando no se puede establecer o mantener
    la conexion con el broker.
    """

    def __init__(
        self,
        message: str,
        broker_id: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None
    ):
        super().__init__(message, broker_id)
        self.host = host
        self.port = port


class AuthenticationError(BrokerError):
    """
    Error de autenticacion.

    Se lanza cuando las credenciales son invalidas
    o la autenticacion falla por cualquier razon.
    """

    def __init__(
        self,
        message: str = "Authentication failed",
        broker_id: Optional[str] = None
    ):
        super().__init__(message, broker_id)


class OrderError(BrokerError):
    """
    Error relacionado con ordenes.

    Error base para todos los errores de ordenes.
    """

    def __init__(
        self,
        message: str,
        order_id: Optional[str] = None,
        broker_id: Optional[str] = None
    ):
        super().__init__(message, broker_id)
        self.order_id = order_id


class InsufficientFundsError(OrderError):
    """
    Error de fondos insuficientes.

    Se lanza cuando no hay suficiente balance
    para ejecutar la orden.
    """

    def __init__(
        self,
        message: str = "Insufficient funds",
        order_id: Optional[str] = None,
        required: Optional[float] = None,
        available: Optional[float] = None
    ):
        super().__init__(message, order_id)
        self.required = required
        self.available = available


class OrderNotFoundError(OrderError):
    """
    Error de orden no encontrada.

    Se lanza cuando se intenta operar con una orden
    que no existe.
    """

    def __init__(
        self,
        message: str = "Order not found",
        order_id: Optional[str] = None
    ):
        super().__init__(message, order_id)


class OrderRejectedError(OrderError):
    """
    Error de orden rechazada.

    Se lanza cuando el broker rechaza la orden.
    """

    def __init__(
        self,
        message: str,
        order_id: Optional[str] = None,
        rejection_reason: Optional[str] = None
    ):
        super().__init__(message, order_id)
        self.rejection_reason = rejection_reason


class UnsupportedOrderTypeError(OrderError):
    """
    Error de tipo de orden no soportado.

    Se lanza cuando se intenta crear un tipo de orden
    que el broker no soporta.
    """

    def __init__(
        self,
        order_type: str,
        broker_id: Optional[str] = None,
        order_id: Optional[str] = None
    ):
        message = f"Order type '{order_type}' is not supported"
        if broker_id:
            message += f" by {broker_id}"
        super().__init__(message, order_id, broker_id)
        self.order_type = order_type


class InvalidOrderError(OrderError):
    """
    Error de orden invalida.

    Se lanza cuando los parametros de la orden son invalidos.
    """

    def __init__(
        self,
        message: str,
        order_id: Optional[str] = None,
        field: Optional[str] = None
    ):
        super().__init__(message, order_id)
        self.field = field


class RateLimitError(BrokerError):
    """
    Error de rate limit.

    Se lanza cuando se excede el limite de peticiones al broker.
    """

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: Optional[float] = None,
        broker_id: Optional[str] = None
    ):
        super().__init__(message, broker_id)
        self.retry_after = retry_after  # Segundos hasta poder reintentar


class PositionError(BrokerError):
    """
    Error relacionado con posiciones.
    """

    def __init__(
        self,
        message: str,
        symbol: Optional[str] = None,
        broker_id: Optional[str] = None
    ):
        super().__init__(message, broker_id)
        self.symbol = symbol


class PositionNotFoundError(PositionError):
    """
    Error de posicion no encontrada.
    """

    def __init__(
        self,
        symbol: str,
        broker_id: Optional[str] = None
    ):
        message = f"Position not found for {symbol}"
        super().__init__(message, symbol, broker_id)


class MarketClosedError(BrokerError):
    """
    Error de mercado cerrado.

    Se lanza cuando se intenta operar en un mercado
    que esta cerrado.
    """

    def __init__(
        self,
        message: str = "Market is closed",
        symbol: Optional[str] = None,
        broker_id: Optional[str] = None
    ):
        super().__init__(message, broker_id)
        self.symbol = symbol


class SymbolNotFoundError(BrokerError):
    """
    Error de simbolo no encontrado.
    """

    def __init__(
        self,
        symbol: str,
        broker_id: Optional[str] = None
    ):
        message = f"Symbol '{symbol}' not found"
        super().__init__(message, broker_id)
        self.symbol = symbol


class ContractQualificationError(BrokerError):
    """
    Error al calificar un contrato (IBKR).

    Se lanza cuando no se puede obtener informacion
    del contrato del broker.
    """

    def __init__(
        self,
        symbol: str,
        broker_id: Optional[str] = None,
        details: Optional[str] = None
    ):
        message = f"Could not qualify contract for '{symbol}'"
        if details:
            message += f": {details}"
        super().__init__(message, broker_id)
        self.symbol = symbol
        self.details = details


class TimeoutError(BrokerError):
    """
    Error de timeout.

    Se lanza cuando una operacion tarda demasiado.
    """

    def __init__(
        self,
        message: str = "Operation timed out",
        timeout_seconds: Optional[float] = None,
        broker_id: Optional[str] = None
    ):
        super().__init__(message, broker_id)
        self.timeout_seconds = timeout_seconds


class BrokerNotConnectedError(BrokerError):
    """
    Error de broker no conectado.

    Se lanza cuando se intenta operar sin estar conectado.
    """

    def __init__(
        self,
        broker_id: Optional[str] = None
    ):
        message = "Broker is not connected"
        if broker_id:
            message = f"Broker '{broker_id}' is not connected"
        super().__init__(message, broker_id)


class BrokerNotRegisteredError(BrokerError):
    """
    Error de broker no registrado.

    Se lanza cuando se intenta usar un broker que no esta registrado.
    """

    def __init__(
        self,
        broker_type: str
    ):
        message = f"Broker type '{broker_type}' is not registered"
        super().__init__(message)
        self.broker_type = broker_type
