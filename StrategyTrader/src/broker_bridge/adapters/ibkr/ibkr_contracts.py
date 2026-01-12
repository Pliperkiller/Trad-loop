"""
Factory de contratos para Interactive Brokers.

Genera contratos IB a partir de simbolos y detecta
automaticamente el tipo de activo.
"""

from typing import Tuple, Dict, Any, Optional
import re

try:
    from ib_insync import Stock, Forex, Future, Option, Index, Contract
    IB_AVAILABLE = True
except ImportError:
    Stock = Forex = Future = Option = Index = Contract = None
    IB_AVAILABLE = False

from ...core.enums import AssetClass


class IBKRContractFactory:
    """
    Factory para crear contratos IB.

    Soporta stocks, forex, indices, futuros y opciones.
    """

    # Simbolos conocidos de indices
    INDEX_SYMBOLS = {
        "SPX", "NDX", "DJI", "RUT", "VIX",
        "ESTX50", "DAX", "FTSE", "N225", "HSI"
    }

    # Monedas forex conocidas
    FOREX_CURRENCIES = {
        "EUR", "GBP", "JPY", "CHF", "AUD", "CAD",
        "NZD", "SEK", "NOK", "DKK", "SGD", "HKD"
    }

    # Patrones de futuros comunes
    FUTURES_PATTERNS = [
        r"^([A-Z]{2,4})(\d{4})$",  # ES2403, NQ2406
        r"^([A-Z]{2,4})([FGHJKMNQUVXZ])(\d{2})$",  # ESH24, NQM24
    ]

    def __init__(self):
        if not IB_AVAILABLE:
            raise ImportError(
                "ib_insync library is not installed. "
                "Install it with: pip install ib_insync"
            )

    def create_contract(
        self,
        symbol: str,
        asset_class: AssetClass,
        exchange: str = "SMART",
        currency: str = "USD",
        **kwargs
    ) -> "Contract":
        """
        Crear contrato IB.

        Args:
            symbol: Simbolo del instrumento
            asset_class: Clase de activo
            exchange: Exchange (default: SMART para routing automatico)
            currency: Moneda (default: USD)
            **kwargs: Parametros adicionales (expiry, strike, right)

        Returns:
            Contrato IB

        Raises:
            ValueError: Si el asset class no es soportado
        """
        if asset_class == AssetClass.STOCK:
            return Stock(symbol, exchange, currency)

        elif asset_class == AssetClass.FOREX:
            # Formato: EUR/USD -> EURUSD
            pair = symbol.replace("/", "")
            return Forex(pair)

        elif asset_class == AssetClass.INDEX:
            return Index(symbol, exchange, currency)

        elif asset_class == AssetClass.FUTURES:
            expiry = kwargs.get("expiry", "")
            multiplier = kwargs.get("multiplier", "")
            contract = Future(
                symbol=symbol,
                lastTradeDateOrContractMonth=expiry,
                exchange=exchange,
                currency=currency
            )
            if multiplier:
                contract.multiplier = multiplier
            return contract

        elif asset_class == AssetClass.OPTIONS:
            expiry = kwargs.get("expiry", "")
            strike = kwargs.get("strike", 0)
            right = kwargs.get("right", "C")  # C=Call, P=Put
            return Option(
                symbol=symbol,
                lastTradeDateOrContractMonth=expiry,
                strike=strike,
                right=right,
                exchange=exchange,
                currency=currency
            )

        else:
            raise ValueError(f"Unsupported asset class: {asset_class}")

    def parse_symbol(self, symbol: str) -> Tuple[str, AssetClass, Dict[str, Any]]:
        """
        Parsear simbolo y detectar asset class.

        Args:
            symbol: Simbolo a parsear

        Returns:
            Tupla de (simbolo_base, asset_class, kwargs)
        """
        # Limpiar simbolo
        symbol = symbol.strip().upper()

        # Crypto - no soportado en IBKR
        if self._is_crypto(symbol):
            raise ValueError(
                f"Crypto symbol '{symbol}' not supported by IBKR. "
                "Use CCXT adapter instead."
            )

        # Forex: EUR/USD, GBP/JPY
        if "/" in symbol:
            parts = symbol.split("/")
            if len(parts) == 2:
                base, quote = parts
                if base in self.FOREX_CURRENCIES or quote in self.FOREX_CURRENCIES:
                    return symbol, AssetClass.FOREX, {}

        # Index
        if symbol in self.INDEX_SYMBOLS:
            return symbol, AssetClass.INDEX, {}

        # Futures: ES2403, NQH24
        for pattern in self.FUTURES_PATTERNS:
            match = re.match(pattern, symbol)
            if match:
                return self._parse_futures_symbol(symbol, match)

        # Options: AAPL240315C00175000
        option_result = self._parse_option_symbol(symbol)
        if option_result:
            return option_result

        # Default: Stock
        return symbol, AssetClass.STOCK, {}

    def _is_crypto(self, symbol: str) -> bool:
        """Verificar si es un simbolo crypto"""
        crypto_bases = {"BTC", "ETH", "XRP", "SOL", "ADA", "DOGE", "AVAX", "DOT"}
        crypto_quotes = {"USDT", "BUSD", "USDC"}

        if "/" in symbol:
            parts = symbol.split("/")
            if len(parts) == 2:
                base, quote = parts
                if base in crypto_bases or quote in crypto_quotes:
                    return True

        return any(c in symbol for c in crypto_bases)

    def _parse_futures_symbol(
        self,
        symbol: str,
        match: re.Match
    ) -> Tuple[str, AssetClass, Dict[str, Any]]:
        """Parsear simbolo de futuros"""
        groups = match.groups()

        if len(groups) == 2:
            # Formato: ES2403
            base = groups[0]
            expiry = "20" + groups[1]
        else:
            # Formato: ESH24
            base = groups[0]
            month_code = groups[1]
            year = groups[2]

            # Convertir codigo de mes a numero
            month_map = {
                "F": "01", "G": "02", "H": "03", "J": "04",
                "K": "05", "M": "06", "N": "07", "Q": "08",
                "U": "09", "V": "10", "X": "11", "Z": "12"
            }
            month = month_map.get(month_code, "01")
            expiry = f"20{year}{month}"

        return base, AssetClass.FUTURES, {"expiry": expiry}

    def _parse_option_symbol(
        self,
        symbol: str
    ) -> Optional[Tuple[str, AssetClass, Dict[str, Any]]]:
        """
        Parsear simbolo de opciones.

        Formato OCC: AAPL240315C00175000
        """
        # Patron OCC: SYMBOL + YYMMDD + C/P + STRIKE*1000
        pattern = r"^([A-Z]+)(\d{6})([CP])(\d{8})$"
        match = re.match(pattern, symbol)

        if not match:
            return None

        base = match.group(1)
        expiry_raw = match.group(2)
        right = match.group(3)
        strike_raw = match.group(4)

        # Parsear fecha (YYMMDD -> YYYYMMDD)
        expiry = "20" + expiry_raw

        # Parsear strike (los ultimos 3 digitos son decimales)
        strike = int(strike_raw) / 1000

        return base, AssetClass.OPTIONS, {
            "expiry": expiry,
            "strike": strike,
            "right": right
        }

    @staticmethod
    def get_futures_months() -> Dict[str, str]:
        """Obtener mapeo de codigos de mes para futuros"""
        return {
            "F": "January", "G": "February", "H": "March",
            "J": "April", "K": "May", "M": "June",
            "N": "July", "Q": "August", "U": "September",
            "V": "October", "X": "November", "Z": "December"
        }
