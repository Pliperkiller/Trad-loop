"""
Implementacion del adaptador CCXT.

Proporciona acceso a 100+ exchanges de criptomonedas
a traves de la libreria CCXT.
"""

import os
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime

try:
    import ccxt.async_support as ccxt
    CCXT_AVAILABLE = True
except ImportError:
    ccxt = None
    CCXT_AVAILABLE = False

logger = logging.getLogger(__name__)

from ...core.enums import (
    BrokerType, OrderType, OrderSide, OrderStatus, PositionSide
)
from ...core.models import (
    BrokerCapabilities, BrokerOrder, BrokerPosition,
    ExecutionReport, AccountInfo
)
from ...core.interfaces import IBrokerAdapter
from ...core.exceptions import (
    BrokerConnectionError, AuthenticationError, OrderError,
    InsufficientFundsError, OrderNotFoundError, RateLimitError,
    SymbolNotFoundError, BrokerNotConnectedError
)
from .ccxt_capabilities import get_exchange_capabilities
from .ccxt_order_mapper import CCXTOrderMapper


class CCXTBroker(IBrokerAdapter):
    """
    Adaptador para exchanges de criptomonedas via CCXT.

    Soporta 100+ exchanges incluyendo Binance, Bybit, OKX, Kraken, etc.
    """

    def __init__(
        self,
        exchange_id: str,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        password: Optional[str] = None,
        testnet: bool = False,
        **kwargs
    ):
        """
        Inicializar adaptador CCXT.

        Las credenciales se pueden proporcionar directamente o via variables de entorno:
        - {EXCHANGE_ID}_API_KEY (ej: BINANCE_API_KEY)
        - {EXCHANGE_ID}_API_SECRET (ej: BINANCE_API_SECRET)
        - {EXCHANGE_ID}_PASSWORD (ej: OKX_PASSWORD)

        Args:
            exchange_id: ID del exchange (ej: 'binance', 'bybit', 'okx')
            api_key: API key del exchange (opcional si se usa env var)
            api_secret: API secret del exchange (opcional si se usa env var)
            password: Passphrase (requerido por algunos exchanges como OKX)
            testnet: Usar testnet/sandbox si esta disponible
            **kwargs: Parametros adicionales para CCXT
        """
        if not CCXT_AVAILABLE:
            raise ImportError(
                "CCXT library is not installed. "
                "Install it with: pip install ccxt"
            )

        self._exchange_id = exchange_id.lower()
        self._testnet = testnet
        self._exchange: Optional[ccxt.Exchange] = None
        self._connected = False

        # Resolver credenciales desde env vars si no se proporcionan
        env_prefix = self._exchange_id.upper()
        resolved_api_key = api_key or os.environ.get(f"{env_prefix}_API_KEY")
        resolved_api_secret = api_secret or os.environ.get(f"{env_prefix}_API_SECRET")
        resolved_password = password or os.environ.get(f"{env_prefix}_PASSWORD")

        # Validar credenciales
        if not resolved_api_key or not resolved_api_secret:
            raise ValueError(
                f"API credentials required for {exchange_id}. "
                f"Provide api_key/api_secret or set {env_prefix}_API_KEY "
                f"and {env_prefix}_API_SECRET environment variables."
            )

        # Advertir si las credenciales se pasan directamente (menos seguro)
        if api_key or api_secret:
            logger.warning(
                f"API credentials for {exchange_id} passed directly. "
                f"Consider using environment variables ({env_prefix}_API_KEY, "
                f"{env_prefix}_API_SECRET) for better security."
            )

        # Configuracion - NO almacenar credenciales en texto plano en el objeto
        # Solo pasarlas al exchange cuando se conecta
        self._config = {
            "apiKey": resolved_api_key,
            "secret": resolved_api_secret,
            "enableRateLimit": True,
            **kwargs
        }

        if resolved_password:
            self._config["password"] = resolved_password

        if testnet:
            self._config["sandbox"] = True

        # Mapper y capacidades
        self._mapper = CCXTOrderMapper(self._exchange_id)
        self._capabilities = get_exchange_capabilities(self._exchange_id)

        # Registrar que se inicializó (sin revelar credenciales)
        logger.info(
            f"CCXTBroker initialized for {exchange_id} "
            f"(testnet={testnet}, credentials={'env' if not api_key else 'direct'})"
        )

    @property
    def broker_type(self) -> BrokerType:
        return BrokerType.CCXT

    @property
    def broker_id(self) -> str:
        return self._exchange_id

    @property
    def is_connected(self) -> bool:
        return self._connected and self._exchange is not None

    def _ensure_connected(self):
        """Verificar que estamos conectados"""
        if not self.is_connected:
            raise BrokerNotConnectedError(self._exchange_id)

    # ==================== Connection ====================

    async def connect(self) -> bool:
        """Conectar al exchange"""
        try:
            # Obtener clase del exchange
            if not hasattr(ccxt, self._exchange_id):
                raise BrokerConnectionError(
                    f"Exchange '{self._exchange_id}' not supported by CCXT",
                    broker_id=self._exchange_id
                )

            exchange_class = getattr(ccxt, self._exchange_id)
            self._exchange = exchange_class(self._config)

            # Cargar mercados para verificar conexion
            await self._exchange.load_markets()
            self._connected = True
            return True

        except ccxt.AuthenticationError as e:
            self._connected = False
            raise AuthenticationError(str(e), self._exchange_id)
        except ccxt.NetworkError as e:
            self._connected = False
            raise BrokerConnectionError(str(e), self._exchange_id)
        except Exception as e:
            self._connected = False
            raise BrokerConnectionError(
                f"Failed to connect to {self._exchange_id}: {e}",
                broker_id=self._exchange_id
            )

    async def disconnect(self) -> None:
        """Desconectar del exchange"""
        if self._exchange:
            await self._exchange.close()
            self._exchange = None
        self._connected = False

    def get_capabilities(self, exchange_id: Optional[str] = None) -> BrokerCapabilities:
        """Obtener capacidades del exchange"""
        return self._capabilities

    # ==================== Orders ====================

    async def submit_order(self, order: BrokerOrder) -> ExecutionReport:
        """Enviar orden al exchange"""
        self._ensure_connected()

        try:
            # Convertir a parametros CCXT
            ccxt_type = self._mapper.to_ccxt_order_type(order.order_type)
            side = self._mapper.to_ccxt_side(order.side)
            params = self._mapper.to_ccxt_params(order)

            # Ejecutar orden
            result = await self._exchange.create_order(
                symbol=order.symbol,
                type=ccxt_type,
                side=side,
                amount=order.quantity,
                price=order.price,
                params=params
            )

            return self._mapper.to_execution_report(result, order)

        except ccxt.InsufficientFunds as e:
            raise InsufficientFundsError(str(e), order.id)
        except ccxt.OrderNotFound as e:
            raise OrderNotFoundError(str(e), order.id)
        except ccxt.RateLimitExceeded as e:
            raise RateLimitError(str(e), broker_id=self._exchange_id)
        except ccxt.BadSymbol as e:
            raise SymbolNotFoundError(order.symbol, self._exchange_id)
        except Exception as e:
            raise OrderError(str(e), order.id, self._exchange_id)

    async def cancel_order(
        self,
        order_id: str,
        symbol: Optional[str] = None
    ) -> bool:
        """Cancelar orden"""
        self._ensure_connected()

        try:
            await self._exchange.cancel_order(order_id, symbol)
            return True
        except ccxt.OrderNotFound:
            return False
        except Exception as e:
            raise OrderError(str(e), order_id, self._exchange_id)

    async def modify_order(
        self,
        order_id: str,
        modifications: Dict[str, Any],
        symbol: Optional[str] = None
    ) -> ExecutionReport:
        """
        Modificar orden existente.

        Nota: La mayoria de exchanges crypto no soportan modificacion
        nativa, se cancela y recrea la orden.
        """
        self._ensure_connected()

        try:
            # Intentar editar si el exchange lo soporta
            if hasattr(self._exchange, 'edit_order'):
                result = await self._exchange.edit_order(
                    order_id,
                    symbol,
                    modifications.get('type', 'limit'),
                    modifications.get('side', 'buy'),
                    modifications.get('amount'),
                    modifications.get('price'),
                )
                return self._mapper.to_execution_report(result)

            # Fallback: cancelar y recrear
            await self.cancel_order(order_id, symbol)

            # Crear nueva orden con modificaciones
            new_order = BrokerOrder(
                symbol=symbol or "",
                side=OrderSide(modifications.get('side', 'buy')),
                order_type=OrderType(modifications.get('type', 'limit')),
                quantity=modifications.get('amount', 0),
                price=modifications.get('price'),
            )
            return await self.submit_order(new_order)

        except Exception as e:
            raise OrderError(str(e), order_id, self._exchange_id)

    async def get_order(
        self,
        order_id: str,
        symbol: Optional[str] = None
    ) -> Optional[BrokerOrder]:
        """Obtener estado de una orden"""
        self._ensure_connected()

        try:
            result = await self._exchange.fetch_order(order_id, symbol)
            return self._mapper.from_ccxt_order(result)
        except ccxt.OrderNotFound:
            return None
        except Exception as e:
            raise OrderError(str(e), order_id, self._exchange_id)

    async def get_open_orders(
        self,
        symbol: Optional[str] = None
    ) -> List[BrokerOrder]:
        """Obtener ordenes abiertas"""
        self._ensure_connected()

        try:
            results = await self._exchange.fetch_open_orders(symbol)
            return [self._mapper.from_ccxt_order(r) for r in results]
        except Exception as e:
            raise OrderError(str(e), broker_id=self._exchange_id)

    async def get_order_history(
        self,
        symbol: Optional[str] = None,
        limit: int = 100
    ) -> List[BrokerOrder]:
        """Obtener historial de ordenes"""
        self._ensure_connected()

        try:
            results = await self._exchange.fetch_closed_orders(symbol, limit=limit)
            return [self._mapper.from_ccxt_order(r) for r in results]
        except Exception as e:
            raise OrderError(str(e), broker_id=self._exchange_id)

    # ==================== Positions ====================

    async def get_positions(
        self,
        symbol: Optional[str] = None
    ) -> List[BrokerPosition]:
        """Obtener posiciones (solo para futures)"""
        self._ensure_connected()

        try:
            symbols = [symbol] if symbol else None
            positions = await self._exchange.fetch_positions(symbols)

            result = []
            for pos in positions:
                # Filtrar posiciones vacias
                contracts = pos.get("contracts", 0) or 0
                if contracts == 0:
                    continue

                side = PositionSide.LONG if pos.get("side") == "long" else PositionSide.SHORT

                result.append(BrokerPosition(
                    symbol=pos.get("symbol", ""),
                    side=side,
                    quantity=abs(float(contracts)),
                    entry_price=float(pos.get("entryPrice", 0) or 0),
                    current_price=float(pos.get("markPrice", 0) or 0),
                    unrealized_pnl=float(pos.get("unrealizedPnl", 0) or 0),
                    realized_pnl=float(pos.get("realizedPnl", 0) or 0),
                    leverage=float(pos.get("leverage", 1) or 1),
                    liquidation_price=pos.get("liquidationPrice"),
                ))

            return result

        except Exception as e:
            # Spot exchanges no tienen posiciones
            return []

    async def close_position(
        self,
        symbol: str,
        quantity: Optional[float] = None
    ) -> ExecutionReport:
        """Cerrar posicion"""
        self._ensure_connected()

        # Obtener posicion actual
        positions = await self.get_positions(symbol)
        if not positions:
            raise OrderError(
                f"No position found for {symbol}",
                broker_id=self._exchange_id
            )

        position = positions[0]
        close_qty = quantity or position.quantity

        # Crear orden de cierre
        close_side = OrderSide.SELL if position.side == PositionSide.LONG else OrderSide.BUY

        order = BrokerOrder(
            symbol=symbol,
            side=close_side,
            order_type=OrderType.MARKET,
            quantity=close_qty,
            reduce_only=True,
        )

        return await self.submit_order(order)

    # ==================== Account ====================

    async def get_balance(self) -> Dict[str, float]:
        """Obtener balance disponible"""
        self._ensure_connected()

        try:
            balance = await self._exchange.fetch_balance()

            result = {}
            for asset, data in balance.items():
                if isinstance(data, dict):
                    free = data.get("free", 0)
                    if free and float(free) > 0:
                        result[asset] = float(free)

            return result

        except Exception as e:
            raise OrderError(str(e), broker_id=self._exchange_id)

    async def get_account_info(self) -> AccountInfo:
        """Obtener informacion de cuenta"""
        self._ensure_connected()

        try:
            balance = await self._exchange.fetch_balance()

            # Calcular totales
            total = float(balance.get("total", {}).get("USDT", 0) or 0)
            free = float(balance.get("free", {}).get("USDT", 0) or 0)
            used = float(balance.get("used", {}).get("USDT", 0) or 0)

            # Extraer balances
            balances = {}
            for asset, data in balance.items():
                if isinstance(data, dict) and data.get("free", 0):
                    balances[asset] = float(data["free"])

            return AccountInfo(
                broker_type=BrokerType.CCXT,
                account_id=self._exchange_id,
                total_balance=total,
                available_balance=free,
                margin_used=used,
                balances=balances,
            )

        except Exception as e:
            raise OrderError(str(e), broker_id=self._exchange_id)

    # ==================== Market Data ====================

    async def get_ticker(self, symbol: str) -> Dict[str, float]:
        """Obtener ticker actual"""
        self._ensure_connected()

        try:
            ticker = await self._exchange.fetch_ticker(symbol)
            return {
                "bid": float(ticker.get("bid", 0) or 0),
                "ask": float(ticker.get("ask", 0) or 0),
                "last": float(ticker.get("last", 0) or 0),
                "volume": float(ticker.get("baseVolume", 0) or 0),
                "high": float(ticker.get("high", 0) or 0),
                "low": float(ticker.get("low", 0) or 0),
                "change": float(ticker.get("percentage", 0) or 0),
            }
        except ccxt.BadSymbol:
            raise SymbolNotFoundError(symbol, self._exchange_id)

    async def get_orderbook(
        self,
        symbol: str,
        limit: int = 20
    ) -> Dict[str, List]:
        """Obtener order book"""
        self._ensure_connected()

        try:
            orderbook = await self._exchange.fetch_order_book(symbol, limit)
            return {
                "bids": orderbook.get("bids", []),
                "asks": orderbook.get("asks", []),
            }
        except ccxt.BadSymbol:
            raise SymbolNotFoundError(symbol, self._exchange_id)

    # ==================== Context Manager ====================

    async def __aenter__(self):
        """Soporte para async context manager"""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Limpiar al salir del context"""
        await self.disconnect()
