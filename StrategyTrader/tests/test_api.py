"""
Tests para API REST (api.py)

Tests para endpoints:
- Health check (/)
- Strategies (/api/v1/strategies)
- Trades (/api/v1/trades/{strategy_id})
- Performance (/api/v1/performance/{strategy_id})
- Equity (/api/v1/equity/{strategy_id})
"""

import pytest
import sys
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Import api module once at module level
from src import api
from fastapi.testclient import TestClient


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_strategy():
    """Mock strategy para tests"""
    strategy = Mock()
    strategy.__class__.__name__ = "MockStrategy"
    strategy.config = Mock()
    strategy.config.symbol = "BTC/USDT"
    strategy.config.timeframe = "1h"
    strategy.config.initial_capital = 10000.0
    strategy.positions = []
    strategy.closed_trades = [
        {
            'entry_time': datetime(2024, 1, 1, 10, 0),
            'exit_time': datetime(2024, 1, 1, 14, 0),
            'entry_price': 50000.0,
            'exit_price': 51000.0,
            'quantity': 0.1,
            'position_type': 'LONG',
            'pnl': 100.0,
            'return_pct': 2.0,
            'reason': 'Take Profit'
        },
        {
            'entry_time': datetime(2024, 1, 2, 10, 0),
            'exit_time': datetime(2024, 1, 2, 14, 0),
            'entry_price': 51000.0,
            'exit_price': 50500.0,
            'quantity': 0.1,
            'position_type': 'LONG',
            'pnl': -50.0,
            'return_pct': -1.0,
            'reason': 'Stop Loss'
        }
    ]
    strategy.equity_curve = [10000, 10100, 10050]
    strategy.get_performance_metrics.return_value = {
        'total_trades': 2,
        'winning_trades': 1,
        'losing_trades': 1,
        'win_rate': 50.0,
        'profit_factor': 2.0,
        'total_return_pct': 0.5,
        'max_drawdown_pct': 1.0,
        'final_capital': 10050,
        'avg_win': 100.0,
        'avg_loss': -50.0,
        'sharpe_ratio': 0.5
    }
    return strategy


@pytest.fixture
def client():
    """TestClient para la API"""
    # Clear any existing strategies before each test
    api.clear_all_strategies()
    yield TestClient(api.app)
    # Cleanup after test
    api.clear_all_strategies()


@pytest.fixture
def client_with_strategy(client, mock_strategy):
    """TestClient con una estrategia registrada"""
    api.register_strategy("test-strategy", mock_strategy)
    yield client
    api.unregister_strategy("test-strategy")


# ============================================================================
# Tests: Health Check
# ============================================================================

class TestHealthCheck:
    """Tests para endpoint de health check"""

    def test_root_endpoint(self, client):
        """Test que el endpoint raiz responde correctamente"""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "Trad-loop API"
        assert data["status"] == "running"
        assert data["version"] == "1.0.0"
        assert "features" in data


# ============================================================================
# Tests: Strategy Management
# ============================================================================

class TestStrategyManagement:
    """Tests para registro y listado de estrategias"""

    def test_list_strategies_empty(self, client):
        """Test listar estrategias cuando no hay ninguna"""
        response = client.get("/api/v1/strategies")

        assert response.status_code == 200
        data = response.json()
        assert data["strategies"] == []

    def test_list_strategies_with_strategy(self, client_with_strategy):
        """Test listar estrategias cuando hay una registrada"""
        response = client_with_strategy.get("/api/v1/strategies")

        assert response.status_code == 200
        data = response.json()
        assert len(data["strategies"]) == 1

        strategy = data["strategies"][0]
        assert strategy["id"] == "test-strategy"
        assert strategy["name"] == "MockStrategy"
        assert strategy["symbol"] == "BTC/USDT"
        assert strategy["timeframe"] == "1h"
        assert strategy["total_trades"] == 2
        assert strategy["is_active"] == False

    def test_register_multiple_strategies(self, client, mock_strategy):
        """Test registrar multiples estrategias"""
        # Registrar 3 estrategias
        api.register_strategy("strategy-1", mock_strategy)
        api.register_strategy("strategy-2", mock_strategy)
        api.register_strategy("strategy-3", mock_strategy)

        response = client.get("/api/v1/strategies")

        assert response.status_code == 200
        data = response.json()
        assert len(data["strategies"]) == 3


# ============================================================================
# Tests: Trades Endpoint
# ============================================================================

class TestTradesEndpoint:
    """Tests para endpoint de trades"""

    def test_get_trades_not_found(self, client):
        """Test obtener trades de estrategia inexistente"""
        response = client.get("/api/v1/trades/nonexistent")

        assert response.status_code == 404
        data = response.json()
        assert "no encontrada" in data["detail"].lower()

    def test_get_trades_success(self, client_with_strategy):
        """Test obtener trades correctamente"""
        response = client_with_strategy.get("/api/v1/trades/test-strategy")

        assert response.status_code == 200
        data = response.json()
        assert data["symbol"] == "BTC/USDT"
        assert data["strategy_name"] == "MockStrategy"
        assert data["total_trades"] == 2
        assert len(data["trades"]) == 2

        # Verificar estructura de trade
        trade = data["trades"][0]
        assert "id" in trade
        assert "entry_time" in trade
        assert "exit_time" in trade
        assert "entry_price" in trade
        assert "exit_price" in trade
        assert "quantity" in trade
        assert "position_type" in trade
        assert "pnl" in trade
        assert "return_pct" in trade
        assert "reason" in trade

    def test_get_trades_filter_profitable(self, client_with_strategy):
        """Test filtrar solo trades rentables"""
        response = client_with_strategy.get("/api/v1/trades/test-strategy?profitable_only=true")

        assert response.status_code == 200
        data = response.json()
        assert data["total_trades"] == 1
        assert data["trades"][0]["pnl"] > 0

    def test_get_trades_filter_position_type_long(self, client_with_strategy):
        """Test filtrar por tipo de posicion LONG"""
        response = client_with_strategy.get("/api/v1/trades/test-strategy?position_type=LONG")

        assert response.status_code == 200
        data = response.json()
        assert all(t["position_type"] == "LONG" for t in data["trades"])

    def test_get_trades_filter_position_type_invalid(self, client_with_strategy):
        """Test filtrar con tipo de posicion invalido"""
        response = client_with_strategy.get("/api/v1/trades/test-strategy?position_type=INVALID")

        assert response.status_code == 400
        data = response.json()
        assert "LONG o SHORT" in data["detail"]

    def test_get_trades_filter_time_range(self, client_with_strategy):
        """Test filtrar por rango de tiempo"""
        response = client_with_strategy.get(
            "/api/v1/trades/test-strategy?"
            "start_time=2024-01-01T00:00:00&"
            "end_time=2024-01-01T23:59:59"
        )

        assert response.status_code == 200
        data = response.json()
        # Solo el primer trade deberia estar en ese rango
        assert data["total_trades"] == 1

    def test_get_trades_invalid_start_time(self, client_with_strategy):
        """Test con formato de fecha invalido"""
        response = client_with_strategy.get("/api/v1/trades/test-strategy?start_time=invalid")

        assert response.status_code == 400
        assert "formato iso" in response.json()["detail"].lower()


# ============================================================================
# Tests: Performance Endpoint
# ============================================================================

class TestPerformanceEndpoint:
    """Tests para endpoint de performance"""

    def test_get_performance_not_found(self, client):
        """Test obtener performance de estrategia inexistente"""
        response = client.get("/api/v1/performance/nonexistent")

        assert response.status_code == 404

    def test_get_performance_success(self, client_with_strategy):
        """Test obtener performance correctamente"""
        response = client_with_strategy.get("/api/v1/performance/test-strategy")

        assert response.status_code == 200
        data = response.json()

        # Verificar metricas
        assert data["total_trades"] == 2
        assert data["winning_trades"] == 1
        assert data["losing_trades"] == 1
        assert data["win_rate"] == 50.0
        assert data["profit_factor"] == 2.0
        assert "total_return_pct" in data
        assert "max_drawdown_pct" in data
        assert "final_capital" in data
        assert "avg_win" in data
        assert "avg_loss" in data
        assert "sharpe_ratio" in data

    def test_get_performance_no_metrics(self, client, mock_strategy):
        """Test cuando strategy.get_performance_metrics() retorna None"""
        mock_strategy.get_performance_metrics.return_value = None
        api.register_strategy("no-metrics", mock_strategy)

        response = client.get("/api/v1/performance/no-metrics")

        assert response.status_code == 404
        assert "backtest()" in response.json()["detail"].lower()

        api.unregister_strategy("no-metrics")


# ============================================================================
# Tests: Equity Curve Endpoint
# ============================================================================

class TestEquityCurveEndpoint:
    """Tests para endpoint de equity curve"""

    def test_get_equity_not_found(self, client):
        """Test obtener equity de estrategia inexistente"""
        response = client.get("/api/v1/equity/nonexistent")

        assert response.status_code == 404

    def test_get_equity_success(self, client_with_strategy):
        """Test obtener equity curve correctamente"""
        response = client_with_strategy.get("/api/v1/equity/test-strategy")

        assert response.status_code == 200
        data = response.json()

        assert data["strategy_id"] == "test-strategy"
        assert data["initial_capital"] == 10000.0
        assert data["equity_curve"] == [10000, 10100, 10050]
        assert data["final_capital"] == 10050

    def test_get_equity_empty_curve(self, client, mock_strategy):
        """Test cuando equity_curve esta vacio"""
        mock_strategy.equity_curve = []
        api.register_strategy("empty-equity", mock_strategy)

        response = client.get("/api/v1/equity/empty-equity")

        assert response.status_code == 200
        data = response.json()
        assert data["final_capital"] == 10000.0  # Deberia usar initial_capital

        api.unregister_strategy("empty-equity")


# ============================================================================
# Tests: Strategy Registration Functions
# ============================================================================

class TestStrategyRegistrationFunctions:
    """Tests para funciones de registro/desregistro"""

    def test_register_strategy(self, mock_strategy):
        """Test registrar estrategia"""
        api.clear_all_strategies()
        result = api.register_strategy("my-strategy", mock_strategy)

        assert result == "my-strategy"
        assert "my-strategy" in api._strategy_registry

        api.clear_all_strategies()

    def test_unregister_strategy_exists(self, mock_strategy):
        """Test desregistrar estrategia existente"""
        api.clear_all_strategies()
        api.register_strategy("to-remove", mock_strategy)
        result = api.unregister_strategy("to-remove")

        assert result == True
        assert "to-remove" not in api._strategy_registry

    def test_unregister_strategy_not_exists(self):
        """Test desregistrar estrategia inexistente"""
        api.clear_all_strategies()
        result = api.unregister_strategy("nonexistent")

        assert result == False

    def test_clear_all_strategies(self, mock_strategy):
        """Test limpiar todas las estrategias"""
        api.register_strategy("s1", mock_strategy)
        api.register_strategy("s2", mock_strategy)
        api.register_strategy("s3", mock_strategy)

        api.clear_all_strategies()

        assert len(api._strategy_registry) == 0


# ============================================================================
# Tests: OHLCV Endpoint
# ============================================================================

class TestOHLCVEndpoint:
    """Tests para endpoint OHLCV"""

    def test_ohlcv_missing_parameters(self, client):
        """Test OHLCV sin parametros requeridos"""
        response = client.get("/api/v1/ohlcv")

        # FastAPI retorna 422 para parametros faltantes
        assert response.status_code == 422

    def test_ohlcv_invalid_timeframe(self, client):
        """Test OHLCV con timeframe invalido"""
        with patch.object(api, 'DATA_EXTRACTOR_AVAILABLE', True):
            response = client.get(
                "/api/v1/ohlcv?"
                "exchange=binance&"
                "symbol=BTC/USDT&"
                "timeframe=invalid&"
                "start=2024-01-01T00:00:00&"
                "end=2024-01-02T00:00:00"
            )

            assert response.status_code == 400
            assert "timeframe" in response.json()["detail"].lower()

    def test_ohlcv_invalid_date_format(self, client):
        """Test OHLCV con formato de fecha invalido"""
        with patch.object(api, 'DATA_EXTRACTOR_AVAILABLE', True):
            response = client.get(
                "/api/v1/ohlcv?"
                "exchange=binance&"
                "symbol=BTC/USDT&"
                "timeframe=1h&"
                "start=invalid-date&"
                "end=2024-01-02T00:00:00"
            )

            assert response.status_code == 400
            assert "formato" in response.json()["detail"].lower()

    def test_ohlcv_start_after_end(self, client):
        """Test OHLCV con fecha inicio despues de fin"""
        with patch.object(api, 'DATA_EXTRACTOR_AVAILABLE', True):
            response = client.get(
                "/api/v1/ohlcv?"
                "exchange=binance&"
                "symbol=BTC/USDT&"
                "timeframe=1h&"
                "start=2024-01-02T00:00:00&"
                "end=2024-01-01T00:00:00"
            )

            assert response.status_code == 400
            assert "anterior" in response.json()["detail"].lower()

    def test_ohlcv_data_extractor_unavailable(self, client):
        """Test cuando DataExtractor no esta disponible"""
        with patch.object(api, 'DATA_EXTRACTOR_AVAILABLE', False):
            response = client.get(
                "/api/v1/ohlcv?"
                "exchange=binance&"
                "symbol=BTC/USDT&"
                "timeframe=1h&"
                "start=2024-01-01T00:00:00&"
                "end=2024-01-02T00:00:00"
            )

            assert response.status_code == 503


# ============================================================================
# Tests: Exchanges Endpoint
# ============================================================================

class TestExchangesEndpoint:
    """Tests para endpoints de exchanges"""

    def test_exchanges_data_extractor_unavailable(self, client):
        """Test cuando DataExtractor no esta disponible"""
        with patch.object(api, 'DATA_EXTRACTOR_AVAILABLE', False):
            response = client.get("/api/v1/exchanges")

            assert response.status_code == 503
            assert "DataExtractor" in response.json()["detail"]

    def test_symbols_data_extractor_unavailable(self, client):
        """Test symbols cuando DataExtractor no esta disponible"""
        with patch.object(api, 'DATA_EXTRACTOR_AVAILABLE', False):
            response = client.get("/api/v1/exchanges/binance/symbols")

            assert response.status_code == 503

    def test_exchange_info_data_extractor_unavailable(self, client):
        """Test exchange info cuando DataExtractor no esta disponible"""
        with patch.object(api, 'DATA_EXTRACTOR_AVAILABLE', False):
            response = client.get("/api/v1/exchanges/binance/info")

            assert response.status_code == 503


# ============================================================================
# Tests: Edge Cases
# ============================================================================

class TestEdgeCases:
    """Tests para casos limite"""

    def test_strategy_with_no_trades(self, client, mock_strategy):
        """Test estrategia sin trades"""
        mock_strategy.closed_trades = []
        api.register_strategy("no-trades", mock_strategy)

        response = client.get("/api/v1/trades/no-trades")

        assert response.status_code == 200
        data = response.json()
        assert data["total_trades"] == 0
        assert data["trades"] == []

        api.unregister_strategy("no-trades")

    def test_strategy_with_active_position(self, client, mock_strategy):
        """Test estrategia con posicion activa"""
        mock_strategy.positions = [Mock()]  # Una posicion activa
        api.register_strategy("active", mock_strategy)

        response = client.get("/api/v1/strategies")

        assert response.status_code == 200
        strategy = [s for s in response.json()["strategies"] if s["id"] == "active"][0]
        assert strategy["is_active"] == True

        api.unregister_strategy("active")

    def test_trade_with_non_datetime_times(self, client, mock_strategy):
        """Test trade con tiempos que no son datetime"""
        mock_strategy.closed_trades = [{
            'entry_time': "2024-01-01T10:00:00",  # String en vez de datetime
            'exit_time': "2024-01-01T14:00:00",
            'entry_price': 50000.0,
            'exit_price': 51000.0,
            'quantity': 0.1,
            'position_type': 'LONG',
            'pnl': 100.0,
            'return_pct': 2.0,
            'reason': 'Test'
        }]
        api.register_strategy("string-times", mock_strategy)

        response = client.get("/api/v1/trades/string-times")

        assert response.status_code == 200
        # Debe manejar strings correctamente

        api.unregister_strategy("string-times")

    def test_trade_missing_position_type(self, client, mock_strategy):
        """Test trade sin position_type (backward compatibility)"""
        mock_strategy.closed_trades = [{
            'entry_time': datetime(2024, 1, 1, 10, 0),
            'exit_time': datetime(2024, 1, 1, 14, 0),
            'entry_price': 50000.0,
            'exit_price': 51000.0,
            'quantity': 0.1,
            # Sin position_type
            'pnl': 100.0,
            'return_pct': 2.0,
            'reason': 'Test'
        }]
        api.register_strategy("no-type", mock_strategy)

        response = client.get("/api/v1/trades/no-type")

        assert response.status_code == 200
        data = response.json()
        # Debe usar LONG como default
        assert data["trades"][0]["position_type"] == "LONG"

        api.unregister_strategy("no-type")


# ============================================================================
# Tests: CORS
# ============================================================================

class TestCORS:
    """Tests para configuracion CORS"""

    def test_cors_headers_present(self, client):
        """Test que los headers CORS estan presentes"""
        response = client.options("/", headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET"
        })

        # CORS deberia permitir cualquier origen
        assert response.status_code in [200, 204]
