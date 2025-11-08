# Arquitectura del Sistema

## Vision General

Market Data Extractor esta construido siguiendo los principios de **Clean Architecture** propuestos por Robert C. Martin. Esta arquitectura permite:

- Independencia de frameworks
- Testabilidad
- Independencia de la UI
- Independencia de bases de datos
- Independencia de cualquier agente externo

## Capas de la Arquitectura

### 1. Domain (Dominio)

**Ubicacion**: `src/domain/`

Esta es la capa mas interna y contiene las reglas de negocio de la aplicacion. No depende de ninguna otra capa.

#### Entidades

- **Candle** (`candle.py`): Representa una vela (candlestick) con OHLCV (Open, High, Low, Close, Volume)
- **MarketConfig** (`market_config.py`): Configuracion para la extraccion de datos
- **MarketType** (enum): Tipos de mercado (SPOT, FUTURES, MARGIN)
- **Timeframe** (enum): Temporalidades disponibles (1m, 5m, 15m, etc.)

#### Repositorios (Interfaces)

- **IMarketRepository** (`market_repository.py`): Define el contrato que deben cumplir todas las implementaciones de exchanges

```python
class IMarketRepository(ABC):
    @abstractmethod
    def get_candles(self, symbol, timeframe, start_date, end_date, ...):
        pass
```

### 2. Infrastructure (Infraestructura)

**Ubicacion**: `src/infrastructure/`

Contiene las implementaciones concretas de las interfaces definidas en el dominio.

#### Exchanges

- **BaseExchange** (`base_exchange.py`): Clase base abstracta con funcionalidad comun
  - Manejo de rate limiting
  - Callbacks de progreso
  - Normalizacion de simbolos

- **BinanceExchange** (`binance_exchange.py`): Implementacion para Binance
  - Soporta SPOT y FUTURES
  - API publica de Binance
  - Limite de 1000 velas por request

- **KrakenExchange** (`kraken_exchange.py`): Implementacion para Kraken
  - Solo soporta SPOT
  - API publica de Kraken
  - Limite de 720 velas por request

#### Exportadores

- **CSVExporter** (`csv_exporter.py`): Exporta datos a formato CSV

### 3. Application (Aplicacion)

**Ubicacion**: `src/application/`

Contiene los casos de uso y la logica de aplicacion.

#### Casos de Uso

- **ExtractMarketDataUseCase** (`extract_market_data.py`):
  - Orquesta la extraccion de datos
  - Valida la configuracion
  - Maneja el flujo completo de extraccion y exportacion

#### Servicios

- **DataExtractionService** (`data_extraction_service.py`):
  - Fachada para la aplicacion
  - Registra todos los exchanges disponibles
  - Proporciona una interfaz simple para la capa de presentacion

### 4. Presentation (Presentacion)

**Ubicacion**: `src/presentation/`

Contiene la interfaz grafica de usuario.

#### GUI

- **MainWindow** (`main_window.py`): Ventana principal de la aplicacion
  - Formulario de configuracion
  - Botones de control
  - Integracion con el servicio de aplicacion

- **LogWidget** (`log_widget.py`): Widget personalizado para mostrar logs
  - Timestamps automaticos
  - Niveles de log (INFO, WARNING, ERROR, SUCCESS)
  - Limite de lineas para evitar consumo de memoria

- **app.py**: Punto de entrada de la presentacion

## Flujo de Dependencias

```
Presentation (UI)
       |
       v
   Application (Use Cases)
       |
       v
  Infrastructure (Implementations)
       |
       v
    Domain (Entities & Interfaces)
```

Las flechas indican la direccion de las dependencias. El dominio no depende de nada, y las capas externas dependen de las internas.

## Principios Aplicados

### 1. Dependency Inversion Principle (DIP)

Las capas externas dependen de abstracciones (interfaces) definidas en capas internas, no de implementaciones concretas.

Ejemplo:
```python
# Application depende de la interfaz, no de la implementacion
class ExtractMarketDataUseCase:
    def __init__(self, repository: IMarketRepository):  # Interface
        self.repository = repository
```

### 2. Single Responsibility Principle (SRP)

Cada clase tiene una unica razon para cambiar.

- `Candle`: Solo representa una vela
- `BinanceExchange`: Solo se comunica con Binance
- `CSVExporter`: Solo exporta a CSV

### 3. Open/Closed Principle (OCP)

El sistema esta abierto para extension pero cerrado para modificacion.

Para agregar un nuevo exchange:
1. Crear una nueva clase que implemente `IMarketRepository`
2. Registrarla en `DataExtractionService`
3. No se modifica codigo existente

### 4. Interface Segregation Principle (ISP)

Las interfaces son especificas y no contienen metodos innecesarios.

## Agregar Nuevo Exchange

### Paso 1: Crear la implementacion

```python
# src/infrastructure/exchanges/coinbase_exchange.py
from .base_exchange import BaseExchange

class CoinbaseExchange(BaseExchange):
    def get_exchange_name(self) -> str:
        return "Coinbase"

    def get_candles(self, symbol, timeframe, start_date, end_date, ...):
        # Implementacion especifica para Coinbase
        pass

    # Implementar otros metodos abstractos...
```

### Paso 2: Registrar en el servicio

```python
# src/application/services/data_extraction_service.py
from ...infrastructure import CoinbaseExchange

class DataExtractionService:
    def __init__(self):
        self._exchanges = {
            'Binance': BinanceExchange(),
            'Kraken': KrakenExchange(),
            'Coinbase': CoinbaseExchange(),  # Nuevo exchange
        }
```

### Paso 3: Actualizar exports

```python
# src/infrastructure/exchanges/__init__.py
from .coinbase_exchange import CoinbaseExchange

__all__ = [..., 'CoinbaseExchange']
```

## Manejo de Errores

- Los errores se propagan hacia arriba
- Cada capa maneja los errores apropiados a su nivel
- La UI presenta mensajes amigables al usuario
- Los logs registran detalles tecnicos

## Threading

La UI ejecuta extracciones en hilos separados para mantener la interfaz responsiva:

```python
thread = threading.Thread(target=self._run_extraction, daemon=True)
thread.start()
```

Los callbacks de progreso se sincronizan con el hilo principal usando `root.after()`.

## Patrones de Diseno Utilizados

### 1. Repository Pattern

`IMarketRepository` define el patron Repository para acceso a datos de mercado.

### 2. Facade Pattern

`DataExtractionService` actua como fachada simplificando el acceso a los casos de uso.

### 3. Template Method

`BaseExchange` define el template method con funcionalidad comun que las subclases especializan.

### 4. Dependency Injection

Las dependencias se inyectan via constructor:

```python
def __init__(self, repository: IMarketRepository):
    self.repository = repository
```

### 5. Strategy Pattern

Los diferentes exchanges son estrategias intercambiables que implementan la misma interfaz.

## Testing

La arquitectura facilita el testing:

- **Domain**: Se puede testear en aislamiento
- **Application**: Se pueden inyectar mocks de repositories
- **Infrastructure**: Se pueden testear contra APIs reales o mocks
- **Presentation**: Se puede testear la logica separada de la UI

## Consideraciones de Rendimiento

- **Rate Limiting**: Implementado en `BaseExchange` para respetar limites de APIs
- **Batch Processing**: Las APIs se consultan en lotes para optimizar requests
- **Threading**: La extraccion se ejecuta en hilos separados
- **Progress Callbacks**: Permiten feedback sin bloquear la UI

## Seguridad

- No se almacenan credenciales (se usan APIs publicas)
- Validacion de inputs en multiples capas
- Manejo seguro de rutas de archivos
- Sin ejecucion de codigo arbitrario

## Escalabilidad

El sistema esta diseñado para escalar en multiples dimensiones:

1. **Nuevos Exchanges**: Implementar `IMarketRepository`
2. **Nuevos Formatos de Exportacion**: Crear nuevos exportadores
3. **Nuevas Fuentes de UI**: La logica es independiente de tkinter
4. **Nuevos Casos de Uso**: Agregar en la capa de aplicacion

## Conclusion

Esta arquitectura proporciona:

- Codigo mantenible y testeable
- Separacion clara de responsabilidades
- Facilidad para agregar nuevas funcionalidades
- Independencia de frameworks y librerias externas
- Base solida para crecimiento futuro del proyecto
