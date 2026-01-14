"""
Registry de estrategias para la API.

Separa la lógica de registro/gestión de estrategias del módulo principal de API.

Este módulo puede usarse independientemente o importarse desde api.py.
"""

from typing import Dict, Any, List, Optional
import threading
import logging

logger = logging.getLogger(__name__)


class StrategyRegistry:
    """
    Registro thread-safe de estrategias de trading.

    Permite registrar, desregistrar y consultar estrategias
    para exposición via API.
    """

    def __init__(self):
        self._strategies: Dict[str, Any] = {}
        self._lock = threading.RLock()

    def register(self, strategy_id: str, strategy_instance: Any) -> str:
        """
        Registra una estrategia.

        Args:
            strategy_id: Identificador único
            strategy_instance: Instancia de la estrategia

        Returns:
            El ID de la estrategia registrada
        """
        with self._lock:
            self._strategies[strategy_id] = strategy_instance
            logger.debug(f"Estrategia '{strategy_id}' registrada")
            return strategy_id

    def unregister(self, strategy_id: str) -> bool:
        """
        Desregistra una estrategia.

        Args:
            strategy_id: ID de la estrategia a remover

        Returns:
            True si se removió, False si no existía
        """
        with self._lock:
            if strategy_id in self._strategies:
                del self._strategies[strategy_id]
                logger.debug(f"Estrategia '{strategy_id}' desregistrada")
                return True
            return False

    def get(self, strategy_id: str) -> Optional[Any]:
        """
        Obtiene una estrategia por ID.

        Args:
            strategy_id: ID de la estrategia

        Returns:
            Instancia de la estrategia o None
        """
        with self._lock:
            return self._strategies.get(strategy_id)

    def get_all(self) -> Dict[str, Any]:
        """
        Obtiene todas las estrategias.

        Returns:
            Diccionario {id: strategy}
        """
        with self._lock:
            return self._strategies.copy()

    def list_ids(self) -> List[str]:
        """
        Lista los IDs de estrategias registradas.

        Returns:
            Lista de IDs
        """
        with self._lock:
            return list(self._strategies.keys())

    def clear(self) -> None:
        """Elimina todas las estrategias."""
        with self._lock:
            self._strategies.clear()
            logger.debug("Todas las estrategias eliminadas")

    def __contains__(self, strategy_id: str) -> bool:
        with self._lock:
            return strategy_id in self._strategies

    def __len__(self) -> int:
        with self._lock:
            return len(self._strategies)


# Instancia global del registry
_registry = StrategyRegistry()


def get_registry() -> StrategyRegistry:
    """Obtiene el registry global."""
    return _registry


def register_strategy(strategy_id: str, strategy_instance: Any) -> str:
    """Atajo para registrar en el registry global."""
    return _registry.register(strategy_id, strategy_instance)


def unregister_strategy(strategy_id: str) -> bool:
    """Atajo para desregistrar del registry global."""
    return _registry.unregister(strategy_id)


def clear_all_strategies() -> None:
    """Atajo para limpiar el registry global."""
    _registry.clear()


def get_strategy(strategy_id: str) -> Optional[Any]:
    """Atajo para obtener del registry global."""
    return _registry.get(strategy_id)
