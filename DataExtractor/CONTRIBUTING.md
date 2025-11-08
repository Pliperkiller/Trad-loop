# Guia de Contribucion

Gracias por tu interes en contribuir a Market Data Extractor. Este documento proporciona guias y mejores practicas para contribuir al proyecto.

## Tabla de Contenidos

- [Codigo de Conducta](#codigo-de-conducta)
- [Como Contribuir](#como-contribuir)
- [Configuracion del Entorno de Desarrollo](#configuracion-del-entorno-de-desarrollo)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Principios de Arquitectura](#principios-de-arquitectura)
- [Guia de Estilo](#guia-de-estilo)
- [Proceso de Pull Request](#proceso-de-pull-request)
- [Reportar Bugs](#reportar-bugs)
- [Solicitar Funcionalidades](#solicitar-funcionalidades)

## Codigo de Conducta

Este proyecto se adhiere a un codigo de conducta profesional y respetuoso. Al participar, se espera que mantengas este codigo.

- Se respetuoso y profesional
- Acepta criticas constructivas
- Enfocate en lo que es mejor para la comunidad
- Muestra empatia hacia otros miembros

## Como Contribuir

Hay varias formas de contribuir:

1. **Reportar Bugs**: Identifica y reporta problemas
2. **Sugerir Mejoras**: Propone nuevas funcionalidades
3. **Documentacion**: Mejora o corrige la documentacion
4. **Codigo**: Implementa nuevas funcionalidades o corrige bugs
5. **Pruebas**: Agrega o mejora tests

## Configuracion del Entorno de Desarrollo

### 1. Fork y Clone

```bash
git clone https://github.com/tu-usuario/DataExtractor.git
cd DataExtractor
```

### 2. Crear Entorno Virtual

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# o
venv\Scripts\activate  # Windows
```

### 3. Instalar Dependencias

```bash
pip install -r requirements.txt
```

### 4. Instalar Dependencias de Desarrollo (opcional)

```bash
pip install pytest pytest-cov black flake8 mypy
```

## Estructura del Proyecto

El proyecto sigue Clean Architecture:

```
src/
├── domain/           # Entidades y contratos (no dependencias)
├── infrastructure/   # Implementaciones de APIs
├── application/      # Casos de uso y servicios
└── presentation/     # Interfaz de usuario
```

**Regla de Oro**: Las dependencias siempre apuntan hacia adentro. El dominio no debe depender de nada.

## Principios de Arquitectura

### 1. Separacion de Responsabilidades

Cada capa tiene un proposito especifico:

- **Domain**: Define el negocio (entidades, interfaces)
- **Infrastructure**: Implementa detalles tecnicos (APIs, I/O)
- **Application**: Orquesta casos de uso
- **Presentation**: Maneja la UI

### 2. Inversion de Dependencias

Las capas externas dependen de interfaces definidas en capas internas:

```python
# CORRECTO
class UseCase:
    def __init__(self, repository: IRepository):  # Interface
        self.repository = repository

# INCORRECTO
class UseCase:
    def __init__(self, repository: BinanceAPI):  # Implementacion concreta
        self.repository = repository
```

### 3. Single Responsibility

Cada clase debe tener una sola razon para cambiar:

```python
# CORRECTO - Una responsabilidad
class CSVExporter:
    def export_candles(self, candles: List[Candle], path: str):
        # Solo exporta a CSV
        pass

# INCORRECTO - Multiples responsabilidades
class DataHandler:
    def fetch_data(self): pass
    def export_csv(self): pass
    def send_email(self): pass
```

## Guia de Estilo

### Python

Seguimos PEP 8 con algunas excepciones:

- Longitud maxima de linea: 100 caracteres (codigo), 72 (docstrings)
- Usar comillas simples para strings
- Docstrings en formato Google

### Ejemplo de Docstring

```python
def get_candles(self, symbol: str, timeframe: Timeframe) -> List[Candle]:
    """
    Obtiene velas historicas del exchange.

    Args:
        symbol: Par de trading (ej: BTC/USDT)
        timeframe: Temporalidad de las velas

    Returns:
        Lista de objetos Candle

    Raises:
        ValueError: Si el simbolo es invalido
        ConnectionError: Si hay problemas de red
    """
    pass
```

### Nombres

- **Clases**: PascalCase (`BinanceExchange`, `MarketConfig`)
- **Funciones/Metodos**: snake_case (`get_candles`, `validate_symbol`)
- **Constantes**: UPPER_SNAKE_CASE (`BASE_URL`, `MAX_RETRIES`)
- **Variables**: snake_case (`candle_data`, `exchange_name`)

### Imports

Organiza imports en tres grupos separados por linea en blanco:

```python
# 1. Libreria estandar
import os
from datetime import datetime

# 2. Librerias de terceros
import requests
from tkinter import ttk

# 3. Modulos locales
from ..domain import Candle
from .base_exchange import BaseExchange
```

## Agregar Nuevo Exchange

### Paso 1: Crear Implementacion

Crea un nuevo archivo en `src/infrastructure/exchanges/`:

```python
# tu_exchange.py
from typing import List
from .base_exchange import BaseExchange
from ...domain import Candle, MarketType, Timeframe

class TuExchange(BaseExchange):
    def get_exchange_name(self) -> str:
        return "TuExchange"

    def get_candles(self, ...) -> List[Candle]:
        # Implementacion
        pass

    # Implementar todos los metodos abstractos...
```

### Paso 2: Agregar Tests

```python
# tests/test_tu_exchange.py
def test_get_candles():
    exchange = TuExchange()
    candles = exchange.get_candles(...)
    assert len(candles) > 0
```

### Paso 3: Registrar en Servicio

```python
# src/application/services/data_extraction_service.py
self._exchanges = {
    'Binance': BinanceExchange(),
    'Kraken': KrakenExchange(),
    'TuExchange': TuExchange(),
}
```

### Paso 4: Actualizar Documentacion

- Agrega a README.md
- Actualiza CHANGELOG.md
- Documenta APIs especificas si es necesario

## Proceso de Pull Request

1. **Crea una Branch**
   ```bash
   git checkout -b feature/nombre-funcionalidad
   ```

2. **Haz tus Cambios**
   - Escribe codigo limpio
   - Agrega tests si es aplicable
   - Actualiza documentacion

3. **Formatea el Codigo**
   ```bash
   black src/
   flake8 src/
   ```

4. **Commit**
   ```bash
   git add .
   git commit -m "Descripcion clara del cambio"
   ```

5. **Push**
   ```bash
   git push origin feature/nombre-funcionalidad
   ```

6. **Abre Pull Request**
   - Describe los cambios claramente
   - Referencia issues relacionados
   - Espera revision

### Formato de Commit Messages

```
tipo: Descripcion breve (50 chars max)

Descripcion detallada si es necesario (72 chars por linea).
Explica el QUE y el POR QUE, no el COMO.

Fixes #123
```

Tipos comunes:
- `feat`: Nueva funcionalidad
- `fix`: Correccion de bug
- `docs`: Cambios en documentacion
- `style`: Formato, no cambia codigo
- `refactor`: Refactorizacion de codigo
- `test`: Agregar o modificar tests
- `chore`: Mantenimiento

## Reportar Bugs

Al reportar un bug, incluye:

1. **Descripcion**: Que paso y que esperabas
2. **Pasos para Reproducir**: Como recrear el problema
3. **Entorno**: SO, version de Python, etc.
4. **Logs**: Mensajes de error relevantes
5. **Screenshots**: Si es relevante para UI

Ejemplo:

```markdown
**Descripcion**
La aplicacion se cierra al seleccionar Kraken con temporalidad 2h

**Pasos para Reproducir**
1. Abrir aplicacion
2. Seleccionar Exchange: Kraken
3. Seleccionar Temporalidad: 2h
4. Click en Extraer Datos

**Entorno**
- SO: Windows 10
- Python: 3.9.5
- Version: 1.0.0

**Logs**
[ERROR] KeyError: Timeframe.TWO_HOURS not supported
```

## Solicitar Funcionalidades

Para solicitar una nueva funcionalidad:

1. **Busca Issues Existentes**: Verifica que no este ya solicitada
2. **Describe el Caso de Uso**: Por que es necesaria
3. **Propone una Solucion**: Como podria implementarse
4. **Alternativas**: Otras formas de resolver el problema

## Preguntas Frecuentes

### Como agrego un nuevo formato de exportacion?

1. Crea una clase exportadora en `src/infrastructure/`
2. Implementa metodo `export_candles(candles, path)`
3. Integra en el caso de uso
4. Actualiza la UI si es necesario

### Como cambio la UI a PyQt en lugar de tkinter?

1. Crea nueva implementacion en `src/presentation/`
2. La logica de aplicacion no deberia cambiar
3. Solo reemplaza la capa de presentacion

### Como agrego autenticacion para APIs privadas?

1. Agrega configuracion en entidades de dominio
2. Modifica la interfaz del repositorio si es necesario
3. Implementa en el exchange especifico
4. Agrega UI para credenciales

## Recursos

- [Clean Architecture - Robert C. Martin](https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html)
- [PEP 8 - Style Guide for Python](https://pep8.org/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [Semantic Versioning](https://semver.org/)

## Contacto

Para preguntas sobre contribuciones, abre un issue o discusion en el repositorio.

Gracias por contribuir a Market Data Extractor.
