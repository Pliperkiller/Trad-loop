"""
Tests para SymbolRouter.
"""

import pytest
import sys
from pathlib import Path

src_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_path))

from broker_bridge.core.enums import BrokerType, AssetClass
from broker_bridge.execution.symbol_router import SymbolRouter


class TestSymbolRouter:
    """Tests para SymbolRouter"""

    @pytest.fixture
    def router(self):
        """Router por defecto"""
        return SymbolRouter()

    # ==================== Crypto Detection ====================

    def test_route_crypto_btc_usdt(self, router):
        """Test ruteo de BTC/USDT a CCXT"""
        broker_type, asset_class = router.route("BTC/USDT")

        assert broker_type == BrokerType.CCXT
        assert asset_class == AssetClass.CRYPTO

    def test_route_crypto_eth_btc(self, router):
        """Test ruteo de ETH/BTC a CCXT"""
        broker_type, asset_class = router.route("ETH/BTC")

        assert broker_type == BrokerType.CCXT
        assert asset_class == AssetClass.CRYPTO

    def test_route_crypto_sol_usdt(self, router):
        """Test ruteo de SOL/USDT a CCXT"""
        broker_type, asset_class = router.route("SOL/USDT")

        assert broker_type == BrokerType.CCXT
        assert asset_class == AssetClass.CRYPTO

    def test_route_crypto_with_busd(self, router):
        """Test ruteo de par con BUSD a CCXT"""
        broker_type, asset_class = router.route("BNB/BUSD")

        assert broker_type == BrokerType.CCXT
        assert asset_class == AssetClass.CRYPTO

    # ==================== Forex Detection ====================

    def test_route_forex_eur_usd(self, router):
        """Test ruteo de EUR/USD a IBKR Forex"""
        broker_type, asset_class = router.route("EUR/USD")

        assert broker_type == BrokerType.IBKR
        assert asset_class == AssetClass.FOREX

    def test_route_forex_gbp_jpy(self, router):
        """Test ruteo de GBP/JPY a IBKR Forex"""
        broker_type, asset_class = router.route("GBP/JPY")

        assert broker_type == BrokerType.IBKR
        assert asset_class == AssetClass.FOREX

    def test_route_forex_aud_cad(self, router):
        """Test ruteo de AUD/CAD a IBKR Forex"""
        broker_type, asset_class = router.route("AUD/CAD")

        assert broker_type == BrokerType.IBKR
        assert asset_class == AssetClass.FOREX

    # ==================== Index Detection ====================

    def test_route_index_spx(self, router):
        """Test ruteo de SPX a IBKR Index"""
        broker_type, asset_class = router.route("SPX")

        assert broker_type == BrokerType.IBKR
        assert asset_class == AssetClass.INDEX

    def test_route_index_ndx(self, router):
        """Test ruteo de NDX a IBKR Index"""
        broker_type, asset_class = router.route("NDX")

        assert broker_type == BrokerType.IBKR
        assert asset_class == AssetClass.INDEX

    def test_route_index_vix(self, router):
        """Test ruteo de VIX a IBKR Index"""
        broker_type, asset_class = router.route("VIX")

        assert broker_type == BrokerType.IBKR
        assert asset_class == AssetClass.INDEX

    # ==================== Futures Detection ====================

    def test_route_futures_es2403(self, router):
        """Test ruteo de ES2403 a IBKR Futures"""
        broker_type, asset_class = router.route("ES2403")

        assert broker_type == BrokerType.IBKR
        assert asset_class == AssetClass.FUTURES

    def test_route_futures_nq2406(self, router):
        """Test ruteo de NQ2406 a IBKR Futures"""
        broker_type, asset_class = router.route("NQ2406")

        assert broker_type == BrokerType.IBKR
        assert asset_class == AssetClass.FUTURES

    def test_route_futures_esh24(self, router):
        """Test ruteo de ESH24 a IBKR Futures"""
        broker_type, asset_class = router.route("ESH24")

        assert broker_type == BrokerType.IBKR
        assert asset_class == AssetClass.FUTURES

    # ==================== Options Detection ====================

    def test_route_options_format(self, router):
        """Test ruteo de opcion formato OCC"""
        # AAPL March 15, 2024, Call, Strike $175
        broker_type, asset_class = router.route("AAPL240315C00175000")

        assert broker_type == BrokerType.IBKR
        assert asset_class == AssetClass.OPTIONS

    # ==================== Stock Detection ====================

    def test_route_stock_aapl(self, router):
        """Test ruteo de AAPL a IBKR Stock"""
        broker_type, asset_class = router.route("AAPL")

        assert broker_type == BrokerType.IBKR
        assert asset_class == AssetClass.STOCK

    def test_route_stock_msft(self, router):
        """Test ruteo de MSFT a IBKR Stock"""
        broker_type, asset_class = router.route("MSFT")

        assert broker_type == BrokerType.IBKR
        assert asset_class == AssetClass.STOCK

    def test_route_stock_googl(self, router):
        """Test ruteo de GOOGL a IBKR Stock"""
        broker_type, asset_class = router.route("GOOGL")

        assert broker_type == BrokerType.IBKR
        assert asset_class == AssetClass.STOCK

    # ==================== Override Tests ====================

    def test_set_override(self, router):
        """Test establecer override"""
        router.set_override("SPECIAL", BrokerType.PAPER, AssetClass.STOCK)
        broker_type, asset_class = router.route("SPECIAL")

        assert broker_type == BrokerType.PAPER
        assert asset_class == AssetClass.STOCK

    def test_remove_override(self, router):
        """Test remover override"""
        router.set_override("TEST", BrokerType.PAPER, AssetClass.STOCK)
        router.remove_override("TEST")

        broker_type, asset_class = router.route("TEST")

        # Default: stock via IBKR
        assert broker_type == BrokerType.IBKR
        assert asset_class == AssetClass.STOCK

    def test_get_overrides(self, router):
        """Test obtener overrides"""
        router.set_override("SYM1", BrokerType.CCXT, AssetClass.CRYPTO)
        router.set_override("SYM2", BrokerType.IBKR, AssetClass.STOCK)

        overrides = router.get_overrides()

        assert "SYM1" in overrides
        assert "SYM2" in overrides

    # ==================== Configuration Tests ====================

    def test_add_crypto_pattern(self, router):
        """Test agregar patron crypto"""
        router.add_crypto_pattern(r"^CUSTOM.*$")
        broker_type, asset_class = router.route("CUSTOM123")

        assert broker_type == BrokerType.CCXT
        assert asset_class == AssetClass.CRYPTO

    def test_add_forex_currency(self, router):
        """Test agregar moneda forex"""
        router.add_forex_currency("MXN")
        broker_type, asset_class = router.route("USD/MXN")

        assert broker_type == BrokerType.IBKR
        assert asset_class == AssetClass.FOREX

    def test_add_index_symbol(self, router):
        """Test agregar simbolo de indice"""
        router.add_index_symbol("MYINDEX")
        broker_type, asset_class = router.route("MYINDEX")

        assert broker_type == BrokerType.IBKR
        assert asset_class == AssetClass.INDEX

    # ==================== Bulk Operations ====================

    def test_route_multiple(self, router):
        """Test rutear multiples simbolos"""
        symbols = ["BTC/USDT", "AAPL", "EUR/USD", "SPX"]
        result = router.route_multiple(symbols)

        assert result["BTC/USDT"] == (BrokerType.CCXT, AssetClass.CRYPTO)
        assert result["AAPL"] == (BrokerType.IBKR, AssetClass.STOCK)
        assert result["EUR/USD"] == (BrokerType.IBKR, AssetClass.FOREX)
        assert result["SPX"] == (BrokerType.IBKR, AssetClass.INDEX)

    def test_group_by_broker(self, router):
        """Test agrupar por broker"""
        symbols = ["BTC/USDT", "ETH/USDT", "AAPL", "MSFT"]
        groups = router.group_by_broker(symbols)

        assert BrokerType.CCXT in groups
        assert BrokerType.IBKR in groups
        assert len(groups[BrokerType.CCXT]) == 2
        assert len(groups[BrokerType.IBKR]) == 2

    def test_group_by_asset_class(self, router):
        """Test agrupar por clase de activo"""
        symbols = ["BTC/USDT", "EUR/USD", "AAPL", "SPX"]
        groups = router.group_by_asset_class(symbols)

        assert AssetClass.CRYPTO in groups
        assert AssetClass.FOREX in groups
        assert AssetClass.STOCK in groups
        assert AssetClass.INDEX in groups

    # ==================== Edge Cases ====================

    def test_case_insensitive(self, router):
        """Test que es case insensitive"""
        broker_type1, _ = router.route("btc/usdt")
        broker_type2, _ = router.route("BTC/USDT")

        assert broker_type1 == broker_type2

    def test_whitespace_handling(self, router):
        """Test manejo de espacios"""
        broker_type, asset_class = router.route("  AAPL  ")

        assert broker_type == BrokerType.IBKR
        assert asset_class == AssetClass.STOCK
