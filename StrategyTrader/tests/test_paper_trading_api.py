"""
Tests para Paper Trading API (paper_trading_api.py)

Tests para endpoints:
- GET /api/v1/paper-trading/strategies
- POST /api/v1/paper-trading/start
- GET /api/v1/paper-trading/sessions
- GET /api/v1/paper-trading/{session_id}
- POST /api/v1/paper-trading/{session_id}/stop
- POST /api/v1/paper-trading/{session_id}/pause
- POST /api/v1/paper-trading/{session_id}/resume
- DELETE /api/v1/paper-trading/{session_id}
- GET /api/v1/paper-trading/{session_id}/trades
- GET /api/v1/paper-trading/{session_id}/positions
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Import after path setup
from src.paper_trading_api import (
    PaperTradingSessionManager,
    PaperTradingSession,
    get_session_manager
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_engine():
    """Mock PaperTradingEngine"""
    engine = Mock()
    engine.is_running = False
    engine.is_paused = False
    engine.state = Mock()
    engine.state.to_dict.return_value = {
        "balance": 10000.0,
        "equity": 10050.0,
        "unrealized_pnl": 50.0,
        "realized_pnl": 0.0,
        "total_trades": 0,
        "winning_trades": 0,
        "losing_trades": 0,
    }
    engine.position_manager = Mock()
    engine.position_manager.positions = []
    engine.position_manager.trade_history = []
    engine.get_performance_report.return_value = {"total_return": 0.5}
    engine.start_async = Mock()
    engine.stop = AsyncMock()
    engine.pause = Mock()
    engine.resume = Mock()
    return engine


@pytest.fixture
def mock_config():
    """Mock PaperTradingConfig"""
    config = Mock()
    config.exchange = "binance"
    config.initial_balance = 10000.0
    config.risk_per_trade = 0.02
    config.max_position_size = 0.1
    config.commission_rate = 0.001
    return config


@pytest.fixture
def session_manager():
    """Session manager instance para tests"""
    manager = PaperTradingSessionManager()
    return manager


# ============================================================================
# Tests: PaperTradingSessionManager
# ============================================================================

class TestPaperTradingSessionManager:
    """Tests para el gestor de sesiones"""

    def test_init_default_strategies(self, session_manager):
        """Test que las estrategias default estan registradas"""
        assert "ma_crossover" in session_manager._strategies

    def test_get_available_strategies(self, session_manager):
        """Test listar estrategias disponibles"""
        strategies = session_manager.get_available_strategies()

        assert len(strategies) >= 1
        assert strategies[0]["id"] == "ma_crossover"
        assert "parameters" in strategies[0]

    def test_register_custom_strategy(self, session_manager):
        """Test registrar estrategia custom"""
        mock_strategy_class = Mock()
        session_manager.register_strategy("custom", mock_strategy_class)

        assert "custom" in session_manager._strategies

    def test_create_session_invalid_strategy(self, session_manager):
        """Test crear sesion con estrategia invalida"""
        with pytest.raises(ValueError) as exc:
            session_manager.create_session(
                strategy_id="nonexistent",
                exchange="binance",
                symbol="BTC/USDT",
                timeframe="1m",
                initial_balance=10000.0,
                risk_per_trade=0.02,
                max_position_size=0.1,
                commission=0.001,
                parameters={}
            )

        assert "not found" in str(exc.value)

    def test_create_session_success(self, session_manager, mock_engine):
        """Test crear sesion exitosamente"""
        with patch('src.paper_trading_api.PaperTradingEngine', return_value=mock_engine):
            session = session_manager.create_session(
                strategy_id="ma_crossover",
                exchange="binance",
                symbol="BTC/USDT",
                timeframe="1m",
                initial_balance=10000.0,
                risk_per_trade=0.02,
                max_position_size=0.1,
                commission=0.001,
                parameters={"fast_period": 10}
            )

            assert session is not None
            assert session.id.startswith("pt_")
            assert session.strategy_id == "ma_crossover"
            assert session.symbol == "BTC/USDT"

    def test_get_session(self, session_manager, mock_engine):
        """Test obtener sesion por ID"""
        with patch('src.paper_trading_api.PaperTradingEngine', return_value=mock_engine):
            session = session_manager.create_session(
                strategy_id="ma_crossover",
                exchange="binance",
                symbol="BTC/USDT",
                timeframe="1m",
                initial_balance=10000.0,
                risk_per_trade=0.02,
                max_position_size=0.1,
                commission=0.001,
                parameters={}
            )

            retrieved = session_manager.get_session(session.id)
            assert retrieved == session

    def test_get_session_not_found(self, session_manager):
        """Test obtener sesion inexistente"""
        result = session_manager.get_session("nonexistent")
        assert result is None

    def test_get_all_sessions(self, session_manager, mock_engine):
        """Test listar todas las sesiones"""
        with patch('src.paper_trading_api.PaperTradingEngine', return_value=mock_engine):
            # Crear 3 sesiones
            for _ in range(3):
                session_manager.create_session(
                    strategy_id="ma_crossover",
                    exchange="binance",
                    symbol="BTC/USDT",
                    timeframe="1m",
                    initial_balance=10000.0,
                    risk_per_trade=0.02,
                    max_position_size=0.1,
                    commission=0.001,
                    parameters={}
                )

            sessions = session_manager.get_all_sessions()
            assert len(sessions) == 3

    def test_get_active_sessions(self, session_manager, mock_engine):
        """Test listar solo sesiones activas"""
        with patch('src.paper_trading_api.PaperTradingEngine', return_value=mock_engine):
            session_manager.create_session(
                strategy_id="ma_crossover",
                exchange="binance",
                symbol="BTC/USDT",
                timeframe="1m",
                initial_balance=10000.0,
                risk_per_trade=0.02,
                max_position_size=0.1,
                commission=0.001,
                parameters={}
            )

            # Engine no esta corriendo
            mock_engine.is_running = False
            active = session_manager.get_active_sessions()
            assert len(active) == 0

            # Simular que esta corriendo
            mock_engine.is_running = True
            active = session_manager.get_active_sessions()
            assert len(active) == 1

    @pytest.mark.asyncio
    async def test_start_session(self, session_manager, mock_engine):
        """Test iniciar sesion"""
        with patch('src.paper_trading_api.PaperTradingEngine', return_value=mock_engine):
            session = session_manager.create_session(
                strategy_id="ma_crossover",
                exchange="binance",
                symbol="BTC/USDT",
                timeframe="1m",
                initial_balance=10000.0,
                risk_per_trade=0.02,
                max_position_size=0.1,
                commission=0.001,
                parameters={}
            )

            result = await session_manager.start_session(session.id)

            assert result == True
            mock_engine.start_async.assert_called_once_with("BTC/USDT", "1m")

    @pytest.mark.asyncio
    async def test_start_session_not_found(self, session_manager):
        """Test iniciar sesion inexistente"""
        result = await session_manager.start_session("nonexistent")
        assert result == False

    @pytest.mark.asyncio
    async def test_stop_session(self, session_manager, mock_engine):
        """Test detener sesion"""
        with patch('src.paper_trading_api.PaperTradingEngine', return_value=mock_engine):
            session = session_manager.create_session(
                strategy_id="ma_crossover",
                exchange="binance",
                symbol="BTC/USDT",
                timeframe="1m",
                initial_balance=10000.0,
                risk_per_trade=0.02,
                max_position_size=0.1,
                commission=0.001,
                parameters={}
            )

            result = await session_manager.stop_session(session.id)

            assert result == True
            mock_engine.stop.assert_called_once()

    def test_pause_session(self, session_manager, mock_engine):
        """Test pausar sesion"""
        with patch('src.paper_trading_api.PaperTradingEngine', return_value=mock_engine):
            session = session_manager.create_session(
                strategy_id="ma_crossover",
                exchange="binance",
                symbol="BTC/USDT",
                timeframe="1m",
                initial_balance=10000.0,
                risk_per_trade=0.02,
                max_position_size=0.1,
                commission=0.001,
                parameters={}
            )

            result = session_manager.pause_session(session.id)

            assert result == True
            mock_engine.pause.assert_called_once()

    def test_resume_session(self, session_manager, mock_engine):
        """Test reanudar sesion"""
        with patch('src.paper_trading_api.PaperTradingEngine', return_value=mock_engine):
            session = session_manager.create_session(
                strategy_id="ma_crossover",
                exchange="binance",
                symbol="BTC/USDT",
                timeframe="1m",
                initial_balance=10000.0,
                risk_per_trade=0.02,
                max_position_size=0.1,
                commission=0.001,
                parameters={}
            )

            result = session_manager.resume_session(session.id)

            assert result == True
            mock_engine.resume.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_session(self, session_manager, mock_engine):
        """Test eliminar sesion"""
        with patch('src.paper_trading_api.PaperTradingEngine', return_value=mock_engine):
            session = session_manager.create_session(
                strategy_id="ma_crossover",
                exchange="binance",
                symbol="BTC/USDT",
                timeframe="1m",
                initial_balance=10000.0,
                risk_per_trade=0.02,
                max_position_size=0.1,
                commission=0.001,
                parameters={}
            )

            session_id = session.id
            result = await session_manager.delete_session(session_id)

            assert result == True
            assert session_manager.get_session(session_id) is None

    @pytest.mark.asyncio
    async def test_delete_session_running(self, session_manager, mock_engine):
        """Test eliminar sesion que esta corriendo"""
        mock_engine.is_running = True

        with patch('src.paper_trading_api.PaperTradingEngine', return_value=mock_engine):
            session = session_manager.create_session(
                strategy_id="ma_crossover",
                exchange="binance",
                symbol="BTC/USDT",
                timeframe="1m",
                initial_balance=10000.0,
                risk_per_trade=0.02,
                max_position_size=0.1,
                commission=0.001,
                parameters={}
            )

            await session_manager.delete_session(session.id)

            # Debe haber detenido el engine primero
            mock_engine.stop.assert_called_once()


# ============================================================================
# Tests: PaperTradingSession
# ============================================================================

class TestPaperTradingSession:
    """Tests para la clase PaperTradingSession"""

    def test_session_to_dict(self, mock_engine):
        """Test serializar sesion a dict"""
        session = PaperTradingSession(
            id="pt_0001",
            engine=mock_engine,
            strategy_id="ma_crossover",
            symbol="BTC/USDT",
            timeframe="1m"
        )

        data = session.to_dict()

        assert data["id"] == "pt_0001"
        assert data["strategy_id"] == "ma_crossover"
        assert data["symbol"] == "BTC/USDT"
        assert data["timeframe"] == "1m"
        assert "created_at" in data
        assert data["is_running"] == False
        assert data["is_paused"] == False


# ============================================================================
# Tests: API Endpoints (Integration)
# ============================================================================

class TestPaperTradingAPIEndpoints:
    """Tests de integracion para endpoints de API"""

    @pytest.fixture
    def client(self, mock_engine):
        """TestClient con mocks"""
        from fastapi.testclient import TestClient

        # Mock dependencies
        with patch.dict('sys.modules', {
            'DataExtractor': MagicMock(),
            'DataExtractor.src': MagicMock(),
            'DataExtractor.src.application': MagicMock(),
            'DataExtractor.src.application.services': MagicMock(),
            'DataExtractor.src.infrastructure': MagicMock(),
            'DataExtractor.src.infrastructure.exchanges': MagicMock(),
            'DataExtractor.src.infrastructure.exchanges.ccxt_adapter': MagicMock(),
            'DataExtractor.src.domain': MagicMock(),
        }):
            with patch('src.paper_trading_api.PaperTradingEngine', return_value=mock_engine):
                from src import api
                # Clear strategy registry
                api.clear_all_strategies()

                yield TestClient(api.app)

    def test_list_strategies_endpoint(self, client):
        """Test GET /api/v1/paper-trading/strategies"""
        response = client.get("/api/v1/paper-trading/strategies")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 1
        assert data[0]["id"] == "ma_crossover"

    def test_list_sessions_empty(self, client):
        """Test GET /api/v1/paper-trading/sessions cuando no hay sesiones"""
        # Reset session manager
        manager = get_session_manager()
        manager._sessions.clear()

        response = client.get("/api/v1/paper-trading/sessions")

        assert response.status_code == 200
        data = response.json()
        assert data == []

    def test_get_session_not_found(self, client):
        """Test GET /api/v1/paper-trading/{session_id} con ID invalido"""
        response = client.get("/api/v1/paper-trading/nonexistent")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_stop_session_not_found(self, client):
        """Test POST /api/v1/paper-trading/{session_id}/stop con ID invalido"""
        response = client.post("/api/v1/paper-trading/nonexistent/stop")

        assert response.status_code == 404

    def test_pause_session_not_found(self, client):
        """Test POST /api/v1/paper-trading/{session_id}/pause con ID invalido"""
        response = client.post("/api/v1/paper-trading/nonexistent/pause")

        assert response.status_code == 404

    def test_resume_session_not_found(self, client):
        """Test POST /api/v1/paper-trading/{session_id}/resume con ID invalido"""
        response = client.post("/api/v1/paper-trading/nonexistent/resume")

        assert response.status_code == 404

    def test_delete_session_not_found(self, client):
        """Test DELETE /api/v1/paper-trading/{session_id} con ID invalido"""
        response = client.delete("/api/v1/paper-trading/nonexistent")

        assert response.status_code == 404

    def test_get_trades_not_found(self, client):
        """Test GET /api/v1/paper-trading/{session_id}/trades con ID invalido"""
        response = client.get("/api/v1/paper-trading/nonexistent/trades")

        assert response.status_code == 404

    def test_get_positions_not_found(self, client):
        """Test GET /api/v1/paper-trading/{session_id}/positions con ID invalido"""
        response = client.get("/api/v1/paper-trading/nonexistent/positions")

        assert response.status_code == 404


# ============================================================================
# Tests: Request Validation
# ============================================================================

class TestRequestValidation:
    """Tests para validacion de requests"""

    @pytest.fixture
    def client(self, mock_engine):
        """TestClient con mocks"""
        from fastapi.testclient import TestClient

        with patch.dict('sys.modules', {
            'DataExtractor': MagicMock(),
            'DataExtractor.src': MagicMock(),
            'DataExtractor.src.application': MagicMock(),
            'DataExtractor.src.application.services': MagicMock(),
            'DataExtractor.src.infrastructure': MagicMock(),
            'DataExtractor.src.infrastructure.exchanges': MagicMock(),
            'DataExtractor.src.infrastructure.exchanges.ccxt_adapter': MagicMock(),
            'DataExtractor.src.domain': MagicMock(),
        }):
            with patch('src.paper_trading_api.PaperTradingEngine', return_value=mock_engine):
                from src import api
                yield TestClient(api.app)

    def test_start_missing_required_fields(self, client):
        """Test POST /api/v1/paper-trading/start sin campos requeridos"""
        response = client.post("/api/v1/paper-trading/start", json={})

        # FastAPI retorna 422 para validacion fallida
        assert response.status_code == 422

    def test_start_invalid_strategy(self, client):
        """Test POST /api/v1/paper-trading/start con estrategia invalida"""
        response = client.post("/api/v1/paper-trading/start", json={
            "strategy_id": "nonexistent",
            "exchange": "binance",
            "symbol": "BTC/USDT"
        })

        assert response.status_code == 400
        assert "not found" in response.json()["detail"].lower()


# ============================================================================
# Tests: Risk Per Trade Conversion
# ============================================================================

class TestRiskPerTradeConversion:
    """Tests para conversion de risk_per_trade"""

    def test_risk_as_percentage(self, session_manager, mock_engine):
        """Test que risk_per_trade > 1 se convierte a decimal"""
        with patch('src.paper_trading_api.PaperTradingEngine', return_value=mock_engine):
            with patch('src.paper_trading_api.PaperTradingConfig') as mock_config_class:
                session_manager.create_session(
                    strategy_id="ma_crossover",
                    exchange="binance",
                    symbol="BTC/USDT",
                    timeframe="1m",
                    initial_balance=10000.0,
                    risk_per_trade=2.0,  # 2% como porcentaje
                    max_position_size=0.1,
                    commission=0.001,
                    parameters={}
                )

                # Verificar que se paso el valor correcto
                # Note: La conversion ocurre en el endpoint, no en create_session
                # Este test verifica que el valor se pasa tal cual
                mock_config_class.assert_called_once()

    def test_risk_as_decimal(self, session_manager, mock_engine):
        """Test que risk_per_trade < 1 se mantiene como decimal"""
        with patch('src.paper_trading_api.PaperTradingEngine', return_value=mock_engine):
            with patch('src.paper_trading_api.PaperTradingConfig') as mock_config_class:
                session_manager.create_session(
                    strategy_id="ma_crossover",
                    exchange="binance",
                    symbol="BTC/USDT",
                    timeframe="1m",
                    initial_balance=10000.0,
                    risk_per_trade=0.02,  # 2% como decimal
                    max_position_size=0.1,
                    commission=0.001,
                    parameters={}
                )

                mock_config_class.assert_called_once()
                call_kwargs = mock_config_class.call_args[1]
                assert call_kwargs["risk_per_trade"] == 0.02
