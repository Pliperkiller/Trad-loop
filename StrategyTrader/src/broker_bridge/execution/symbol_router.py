"""
Router de simbolos para broker_bridge.

Determina automaticamente que broker debe manejar
cada simbolo basado en patrones y reglas.
"""

import re
from typing import Tuple, Set, List, Optional

from ..core.enums import BrokerType, AssetClass


class SymbolRouter:
    """
    Router de simbolos a brokers.

    Determina automaticamente el broker y clase de activo
    apropiados para cada simbolo.
    """

    # Patrones de crypto
    DEFAULT_CRYPTO_PATTERNS = [
        r".*/(USDT|BUSD|USD|USDC|BTC|ETH)$",  # BTC/USDT, ETH/BTC
        r"^(BTC|ETH|BNB|SOL|XRP|ADA|AVAX|DOGE|DOT|MATIC)",  # Prefijos crypto
    ]

    # Monedas forex conocidas
    DEFAULT_FOREX_CURRENCIES = {
        "EUR", "GBP", "JPY", "CHF", "AUD", "CAD",
        "NZD", "SEK", "NOK", "DKK", "SGD", "HKD", "USD"
    }

    # Simbolos de indices conocidos
    DEFAULT_INDEX_SYMBOLS = {
        "SPX", "NDX", "DJI", "RUT", "VIX",
        "ESTX50", "DAX", "FTSE", "N225", "HSI"
    }

    # Patrones de futuros
    DEFAULT_FUTURES_PATTERNS = [
        r"^([A-Z]{2,4})(\d{4})$",  # ES2403, NQ2406
        r"^([A-Z]{2,4})([FGHJKMNQUVXZ])(\d{2})$",  # ESH24, NQM24
    ]

    def __init__(self):
        """Inicializar router con valores por defecto"""
        self._crypto_patterns: List[str] = list(self.DEFAULT_CRYPTO_PATTERNS)
        self._forex_currencies: Set[str] = set(self.DEFAULT_FOREX_CURRENCIES)
        self._index_symbols: Set[str] = set(self.DEFAULT_INDEX_SYMBOLS)
        self._futures_patterns: List[str] = list(self.DEFAULT_FUTURES_PATTERNS)

        # Overrides manuales: symbol -> (BrokerType, AssetClass)
        self._overrides: dict = {}

    def route(self, symbol: str) -> Tuple[BrokerType, AssetClass]:
        """
        Determinar broker y asset class para un simbolo.

        Args:
            symbol: Simbolo a rutear

        Returns:
            Tupla de (BrokerType, AssetClass)
        """
        symbol = symbol.strip().upper()

        # Verificar overrides
        if symbol in self._overrides:
            return self._overrides[symbol]

        # Forex (verificar antes de crypto porque EUR/USD podria matchear patrones crypto)
        if self._is_forex(symbol):
            return BrokerType.IBKR, AssetClass.FOREX

        # Crypto
        if self._is_crypto(symbol):
            return BrokerType.CCXT, AssetClass.CRYPTO

        # Index
        if self._is_index(symbol):
            return BrokerType.IBKR, AssetClass.INDEX

        # Futures
        if self._is_futures(symbol):
            return BrokerType.IBKR, AssetClass.FUTURES

        # Options (formato OCC)
        if self._is_options(symbol):
            return BrokerType.IBKR, AssetClass.OPTIONS

        # Default: Stock via IBKR
        return BrokerType.IBKR, AssetClass.STOCK

    def _is_crypto(self, symbol: str) -> bool:
        """Verificar si es un simbolo crypto"""
        for pattern in self._crypto_patterns:
            if re.match(pattern, symbol):
                return True
        return False

    def _is_forex(self, symbol: str) -> bool:
        """Verificar si es un par forex"""
        if "/" not in symbol:
            return False

        parts = symbol.split("/")
        if len(parts) != 2:
            return False

        base, quote = parts

        # Ambas deben ser monedas conocidas
        if base in self._forex_currencies and quote in self._forex_currencies:
            # Excluir crypto
            crypto_bases = {"BTC", "ETH", "BNB", "SOL", "XRP"}
            if base not in crypto_bases and quote not in crypto_bases:
                return True

        return False

    def _is_index(self, symbol: str) -> bool:
        """Verificar si es un simbolo de indice"""
        # Limpiar sufijos comunes
        base = symbol.split("/")[0] if "/" in symbol else symbol
        return base in self._index_symbols

    def _is_futures(self, symbol: str) -> bool:
        """Verificar si es un simbolo de futuros"""
        for pattern in self._futures_patterns:
            if re.match(pattern, symbol):
                return True
        return False

    def _is_options(self, symbol: str) -> bool:
        """Verificar si es un simbolo de opciones (formato OCC)"""
        # Formato OCC: AAPL240315C00175000
        pattern = r"^[A-Z]+\d{6}[CP]\d{8}$"
        return bool(re.match(pattern, symbol))

    # ==================== Configuration ====================

    def add_crypto_pattern(self, pattern: str):
        """
        Agregar patron para detectar crypto.

        Args:
            pattern: Expresion regular
        """
        self._crypto_patterns.append(pattern)

    def add_forex_currency(self, currency: str):
        """
        Agregar moneda forex.

        Args:
            currency: Codigo de moneda (3 letras)
        """
        self._forex_currencies.add(currency.upper())

    def add_index_symbol(self, symbol: str):
        """
        Agregar simbolo de indice.

        Args:
            symbol: Simbolo del indice
        """
        self._index_symbols.add(symbol.upper())

    def add_futures_pattern(self, pattern: str):
        """
        Agregar patron para detectar futuros.

        Args:
            pattern: Expresion regular
        """
        self._futures_patterns.append(pattern)

    def set_override(
        self,
        symbol: str,
        broker_type: BrokerType,
        asset_class: AssetClass
    ):
        """
        Establecer override manual para un simbolo.

        Args:
            symbol: Simbolo
            broker_type: Tipo de broker a usar
            asset_class: Clase de activo
        """
        self._overrides[symbol.upper()] = (broker_type, asset_class)

    def remove_override(self, symbol: str):
        """
        Remover override de un simbolo.

        Args:
            symbol: Simbolo
        """
        self._overrides.pop(symbol.upper(), None)

    def get_overrides(self) -> dict:
        """Obtener todos los overrides"""
        return dict(self._overrides)

    # ==================== Bulk Operations ====================

    def route_multiple(
        self,
        symbols: List[str]
    ) -> dict:
        """
        Rutear multiples simbolos.

        Args:
            symbols: Lista de simbolos

        Returns:
            Diccionario {symbol: (BrokerType, AssetClass)}
        """
        return {s: self.route(s) for s in symbols}

    def group_by_broker(
        self,
        symbols: List[str]
    ) -> dict:
        """
        Agrupar simbolos por broker.

        Args:
            symbols: Lista de simbolos

        Returns:
            Diccionario {BrokerType: [symbols]}
        """
        groups: dict = {}
        for symbol in symbols:
            broker_type, _ = self.route(symbol)
            if broker_type not in groups:
                groups[broker_type] = []
            groups[broker_type].append(symbol)
        return groups

    def group_by_asset_class(
        self,
        symbols: List[str]
    ) -> dict:
        """
        Agrupar simbolos por clase de activo.

        Args:
            symbols: Lista de simbolos

        Returns:
            Diccionario {AssetClass: [symbols]}
        """
        groups: dict = {}
        for symbol in symbols:
            _, asset_class = self.route(symbol)
            if asset_class not in groups:
                groups[asset_class] = []
            groups[asset_class].append(symbol)
        return groups
