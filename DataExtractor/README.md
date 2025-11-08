# Market Data Extractor

Aplicacion de escritorio para extraer datos historicos de velas (candlestick) de los principales mercados financieros con APIs publicas.

## Caracteristicas

- Interfaz grafica intuitiva desarrollada con tkinter (sin dependencias externas)
- Soporte para multiples exchanges: Binance, Kraken
- Extraccion de datos de mercados SPOT y FUTURES
- Multiples temporalidades: 1m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d, 3d, 1w, 1M
- Selector de fechas integrado sin dependencias externas
- Sistema de logs en tiempo real para seguimiento del proceso
- Exportacion de datos a formato CSV
- Arquitectura limpia y escalable para facil expansion
- Solo requiere Python 3.8+ y la libreria requests

## Arquitectura

El proyecto sigue los principios de Clean Architecture, organizando el codigo en capas bien definidas:

```
src/
├── domain/           # Entidades y contratos (reglas de negocio)
├── infrastructure/   # Implementaciones de APIs y servicios externos
├── application/      # Casos de uso y logica de aplicacion
└── presentation/     # Interfaz grafica de usuario
```

Para mas detalles, consulta [ARCHITECTURE.md](docs/ARCHITECTURE.md)

## Requisitos

- Python 3.8 o superior
- Sistema operativo: Windows, Linux, macOS
- Conexion a Internet para obtener datos de los exchanges

## Instalacion

### 1. Clonar o descargar el proyecto

```bash
cd DataExtractor
```

### 2. Crear un entorno virtual (recomendado)

```bash
python -m venv venv
```

### 3. Activar el entorno virtual

**Windows:**
```bash
venv\Scripts\activate
```

**Linux/macOS:**
```bash
source venv/bin/activate
```

### 4. Instalar dependencias

```bash
pip install -r requirements.txt
```

## Uso

### Ejecutar la aplicacion

```bash
python main.py
```

### Uso de la interfaz

1. **Seleccionar Exchange**: Elige el exchange del cual quieres obtener datos (Binance, Kraken)

2. **Tipo de Mercado**: Selecciona el tipo de mercado:
   - SPOT: Mercado al contado
   - FUTURES: Mercado de futuros (solo Binance)
   - MARGIN: Mercado de margen (disponibilidad segun exchange)

3. **Simbolo**: Ingresa el par de trading (ej: BTC/USDT, ETH/USDT)

4. **Temporalidad**: Selecciona el intervalo de tiempo para las velas

5. **Fechas**: Selecciona la fecha de inicio y fin usando el calendario
   - La fecha de fin se establece automaticamente a la fecha/hora actual

6. **Ruta de Salida**: Haz clic en "Examinar..." para seleccionar donde guardar el archivo CSV

7. **Extraer Datos**: Haz clic en el boton "Extraer Datos" para iniciar el proceso

8. **Logs**: Observa el progreso en el area de logs

Para mas informacion detallada, consulta [USER_GUIDE.md](docs/USER_GUIDE.md)

## Estructura del Proyecto

```
DataExtractor/
├── src/
│   ├── domain/
│   │   ├── entities/
│   │   │   ├── candle.py
│   │   │   └── market_config.py
│   │   └── repositories/
│   │       └── market_repository.py
│   ├── infrastructure/
│   │   ├── exchanges/
│   │   │   ├── base_exchange.py
│   │   │   ├── binance_exchange.py
│   │   │   └── kraken_exchange.py
│   │   └── csv_exporter.py
│   ├── application/
│   │   ├── use_cases/
│   │   │   └── extract_market_data.py
│   │   └── services/
│   │       └── data_extraction_service.py
│   └── presentation/
│       ├── gui/
│       │   ├── main_window.py
│       │   └── widgets/
│       │       └── log_widget.py
│       └── app.py
├── docs/
│   ├── ARCHITECTURE.md
│   └── USER_GUIDE.md
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Formato del CSV

Los archivos CSV exportados contienen las siguientes columnas:

- `timestamp`: Fecha y hora de la vela (formato ISO 8601)
- `open`: Precio de apertura
- `high`: Precio maximo
- `low`: Precio minimo
- `close`: Precio de cierre
- `volume`: Volumen negociado
- `symbol`: Simbolo del par de trading

## Agregar Nuevos Exchanges

El sistema esta diseñado para facilitar la adicion de nuevos exchanges:

1. Crear una nueva clase en `src/infrastructure/exchanges/`
2. Heredar de `BaseExchange`
3. Implementar los metodos abstractos de `IMarketRepository`
4. Registrar el exchange en `DataExtractionService`

Ver [ARCHITECTURE.md](docs/ARCHITECTURE.md) para mas detalles.

## Desarrollo

### Ejecutar tests (si estan disponibles)

```bash
pytest
```

### Formatear codigo

```bash
black src/
```

### Verificar estilo

```bash
flake8 src/
```

## Limitaciones Conocidas

- Los exchanges tienen limites de rate limiting que se respetan automaticamente
- Binance limita a 1000 velas por request
- Kraken limita a 720 velas por request
- La extraccion de grandes volumenes de datos puede tomar tiempo

## Contribuir

Este proyecto sigue principios de Clean Architecture y buenas practicas de desarrollo. Al contribuir:

1. Respeta la separacion de capas
2. Mantén las dependencias apuntando hacia el interior (domain no debe depender de nada)
3. Escribe codigo limpio y documentado
4. Sigue las convenciones de nombres de Python (PEP 8)

## Licencia

Este proyecto es de uso libre para propositos educativos y personales.

## Soporte

Si encuentras problemas al ejecutar la aplicacion:

1. Consulta la [Guia de Solucion de Problemas](TROUBLESHOOTING.md)
2. Ejecuta el script de prueba: `python test_imports.py`
3. Revisa los logs en la aplicacion para detalles del error

Para reportar problemas o solicitar nuevas funcionalidades, por favor crea un issue en el repositorio.

## Version

Version actual: 1.0.1
