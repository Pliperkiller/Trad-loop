"""
Enumeraciones para ordenes avanzadas.

Define todos los tipos de ordenes soportados y sus parametros asociados.
"""

from enum import Enum


class OrderType(Enum):
    """
    Tipos de orden soportados.

    Incluye ordenes basicas y avanzadas para trading profesional.
    """
    # Ordenes basicas (existentes)
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    STOP_LIMIT = "stop_limit"

    # Ordenes de control de riesgo
    STOP_MARKET = "stop_market"
    TAKE_PROFIT_LIMIT = "take_profit_limit"
    TAKE_PROFIT_MARKET = "take_profit_market"
    TRAILING_STOP = "trailing_stop"
    TRAILING_STOP_LIMIT = "trailing_stop_limit"
    BRACKET = "bracket"

    # Algoritmos de ejecucion
    TWAP = "twap"  # Time-Weighted Average Price
    VWAP = "vwap"  # Volume-Weighted Average Price
    ICEBERG = "iceberg"
    HIDDEN = "hidden"

    # Ordenes de gestion dinamica
    FOK = "fill_or_kill"
    IOC = "immediate_or_cancel"
    REDUCE_ONLY = "reduce_only"
    POST_ONLY = "post_only"
    SCALE_IN = "scale_in"
    SCALE_OUT = "scale_out"

    # Ordenes condicionales
    IF_TOUCHED = "if_touched"
    OCO = "one_cancels_other"
    OTOCO = "one_triggers_oco"


class TimeInForce(Enum):
    """
    Tiempo de vigencia de una orden.

    Define cuanto tiempo permanece activa una orden en el mercado.
    """
    GTC = "good_till_cancel"      # Activa hasta cancelacion manual
    IOC = "immediate_or_cancel"   # Ejecuta inmediatamente o cancela
    FOK = "fill_or_kill"          # Todo o nada, inmediato
    GTD = "good_till_date"        # Activa hasta fecha especifica
    DAY = "day"                   # Valida solo durante la sesion
    GTX = "good_till_crossing"    # Post-only, cancela si cruza


class TriggerType(Enum):
    """
    Tipo de precio que activa ordenes condicionales.

    Define que precio se usa para evaluar condiciones de trigger.
    """
    LAST_PRICE = "last_price"     # Ultimo precio de transaccion
    MARK_PRICE = "mark_price"     # Precio de marca (futuros)
    INDEX_PRICE = "index_price"   # Precio del indice subyacente
    BID_PRICE = "bid_price"       # Mejor precio de compra
    ASK_PRICE = "ask_price"       # Mejor precio de venta


class TriggerDirection(Enum):
    """
    Direccion del trigger para ordenes condicionales.

    Define si el trigger se activa cuando el precio sube o baja.
    """
    ABOVE = "above"  # Trigger cuando precio >= valor
    BELOW = "below"  # Trigger cuando precio <= valor


class OrderSide(Enum):
    """Lado de la orden (compra/venta)"""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """
    Estados posibles de una orden.

    Representa el ciclo de vida completo de una orden.
    """
    PENDING = "pending"                    # Orden creada, no enviada
    SUBMITTED = "submitted"                # Enviada al mercado
    ACCEPTED = "accepted"                  # Aceptada por el exchange
    PARTIALLY_FILLED = "partially_filled"  # Ejecutada parcialmente
    FILLED = "filled"                      # Completamente ejecutada
    CANCELLED = "cancelled"                # Cancelada por usuario
    REJECTED = "rejected"                  # Rechazada por el sistema
    EXPIRED = "expired"                    # Expirada por tiempo
    TRIGGERED = "triggered"                # Orden condicional activada


class PositionSide(Enum):
    """Tipo de posicion"""
    LONG = "long"
    SHORT = "short"


class CompositeOrderStatus(Enum):
    """
    Estado de ordenes compuestas (Bracket, OCO, OTOCO).

    Representa el estado del grupo de ordenes como unidad.
    """
    PENDING = "pending"            # Esperando activacion
    ACTIVE = "active"              # Activa y monitoreando
    PARTIALLY_ACTIVE = "partial"   # Algunas ordenes activas
    COMPLETED = "completed"        # Todas las ordenes completadas
    CANCELLED = "cancelled"        # Grupo cancelado
