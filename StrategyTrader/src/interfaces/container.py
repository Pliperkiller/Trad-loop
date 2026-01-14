"""
Contenedor de Dependency Injection para StrategyTrader.

Permite configurar y resolver dependencias de manera centralizada.

Uso básico:
    from src.interfaces.container import Container

    # Configurar
    container = Container()
    container.register_singleton('validator', DefaultDataValidator(strict=True))
    container.register_factory('sizer', lambda: FixedFractionalSizer(risk_pct=2.0))

    # Usar
    validator = container.resolve('validator')

Uso con configuración predeterminada:
    container = Container.with_defaults()
    validator = container.resolve('validator')
"""

from typing import (
    Dict,
    Any,
    Optional,
    Callable,
    TypeVar,
    Generic,
    Type,
    overload,
)
from dataclasses import dataclass, field
import threading
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class ServiceDescriptor:
    """Describe cómo resolver un servicio."""
    factory: Callable[[], Any]
    singleton: bool = False
    instance: Optional[Any] = None


class Container:
    """
    Contenedor de Dependency Injection.

    Soporta:
    - Singletons: una única instancia compartida
    - Factories: nueva instancia en cada resolución
    - Scoped: instancias por scope (ej: por request)
    """

    def __init__(self):
        self._services: Dict[str, ServiceDescriptor] = {}
        self._lock = threading.Lock()

    def register_singleton(self, name: str, instance: Any) -> 'Container':
        """
        Registra una instancia singleton.

        Args:
            name: Nombre del servicio
            instance: Instancia a registrar

        Returns:
            self para encadenar llamadas
        """
        with self._lock:
            self._services[name] = ServiceDescriptor(
                factory=lambda i=instance: i,
                singleton=True,
                instance=instance
            )
        return self

    def register_factory(
        self,
        name: str,
        factory: Callable[[], T],
        singleton: bool = False
    ) -> 'Container':
        """
        Registra una factory.

        Args:
            name: Nombre del servicio
            factory: Callable que crea la instancia
            singleton: Si True, solo se llama una vez

        Returns:
            self para encadenar llamadas
        """
        with self._lock:
            self._services[name] = ServiceDescriptor(
                factory=factory,
                singleton=singleton
            )
        return self

    def register_type(
        self,
        name: str,
        service_type: Type[T],
        singleton: bool = False,
        **kwargs
    ) -> 'Container':
        """
        Registra un tipo para instanciar.

        Args:
            name: Nombre del servicio
            service_type: Clase a instanciar
            singleton: Si True, singleton
            **kwargs: Argumentos para el constructor

        Returns:
            self para encadenar llamadas
        """
        def factory():
            return service_type(**kwargs)

        return self.register_factory(name, factory, singleton)

    def resolve(self, name: str) -> Any:
        """
        Resuelve un servicio por nombre.

        Args:
            name: Nombre del servicio

        Returns:
            Instancia del servicio

        Raises:
            KeyError: Si el servicio no está registrado
        """
        with self._lock:
            if name not in self._services:
                raise KeyError(f"Servicio '{name}' no registrado")

            descriptor = self._services[name]

            if descriptor.singleton and descriptor.instance is not None:
                return descriptor.instance

            instance = descriptor.factory()

            if descriptor.singleton:
                descriptor.instance = instance

            return instance

    def try_resolve(self, name: str) -> Optional[Any]:
        """
        Intenta resolver un servicio, retorna None si no existe.

        Args:
            name: Nombre del servicio

        Returns:
            Instancia o None
        """
        try:
            return self.resolve(name)
        except KeyError:
            return None

    def has_service(self, name: str) -> bool:
        """Verifica si un servicio está registrado."""
        with self._lock:
            return name in self._services

    def unregister(self, name: str) -> bool:
        """
        Elimina un servicio registrado.

        Returns:
            True si se eliminó, False si no existía
        """
        with self._lock:
            if name in self._services:
                del self._services[name]
                return True
            return False

    def clear(self) -> None:
        """Elimina todos los servicios registrados."""
        with self._lock:
            self._services.clear()

    @classmethod
    def with_defaults(cls) -> 'Container':
        """
        Crea un contenedor con configuración por defecto.

        Registra implementaciones estándar para todos los servicios.

        Returns:
            Container configurado
        """
        from .implementations import (
            DefaultDataValidator,
            DefaultMetricsCalculator,
            DefaultRiskManager,
            FixedFractionalSizer,
        )

        container = cls()

        # Validador de datos
        container.register_type(
            'validator',
            DefaultDataValidator,
            singleton=True,
            strict=True
        )

        # Calculador de métricas
        container.register_type(
            'metrics_calculator',
            DefaultMetricsCalculator,
            singleton=True,
            risk_free_rate=0.0,
            periods_per_year=252
        )

        # Position sizer
        container.register_type(
            'position_sizer',
            FixedFractionalSizer,
            singleton=True,
            risk_pct=2.0,
            max_position_pct=95.0
        )

        # Risk manager
        container.register_type(
            'risk_manager',
            DefaultRiskManager,
            singleton=True,
            max_positions=5,
            max_drawdown_pct=25.0
        )

        return container

    def __repr__(self) -> str:
        services = list(self._services.keys())
        return f"Container({len(services)} services: {services})"


# Contenedor global por defecto
_default_container: Optional[Container] = None
_container_lock = threading.Lock()


def get_container() -> Container:
    """
    Obtiene el contenedor global.

    Si no existe, crea uno con configuración por defecto.

    Returns:
        Container global
    """
    global _default_container
    with _container_lock:
        if _default_container is None:
            _default_container = Container.with_defaults()
        return _default_container


def set_container(container: Container) -> None:
    """
    Establece el contenedor global.

    Args:
        container: Contenedor a usar globalmente
    """
    global _default_container
    with _container_lock:
        _default_container = container


def reset_container() -> None:
    """Resetea el contenedor global a None."""
    global _default_container
    with _container_lock:
        _default_container = None


def resolve(name: str) -> Any:
    """
    Atajo para resolver desde el contenedor global.

    Args:
        name: Nombre del servicio

    Returns:
        Instancia del servicio
    """
    return get_container().resolve(name)
