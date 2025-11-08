# Changelog

Todos los cambios notables en este proyecto seran documentados en este archivo.

El formato esta basado en [Keep a Changelog](https://keepachangelog.com/es-ES/1.0.0/),
y este proyecto adhiere a [Semantic Versioning](https://semver.org/lang/es/).

## [1.0.1] - 2024-11-07

### Corregido
- Error "Pack.pack_info() takes 1 positional argument but 2 were given" causado por incompatibilidad con tkcalendar
- La aplicacion ahora arranca correctamente sin errores de dependencias

### Cambiado
- Reemplazado tkcalendar con selector de fechas personalizado (DateSelector)
- Eliminada dependencia externa de tkcalendar del requirements.txt
- Ahora solo requiere Python stdlib y requests (mas liviano y compatible)

### Agregado
- Widget DateSelector personalizado usando solo tkinter/ttk
- Script de prueba test_imports.py para verificar dependencias
- Guia completa de solucion de problemas (TROUBLESHOOTING.md)
- Boton "Hoy" en los selectores de fecha para establecer rapidamente la fecha actual

### Mejorado
- Mayor compatibilidad con diferentes versiones de Python 3.8+
- Reduccion de dependencias externas mejora la estabilidad
- Instalacion mas rapida y ligera

## [1.0.0] - 2024-11-07

### Agregado

#### Core Features
- Aplicacion de escritorio con interfaz grafica (tkinter)
- Extraccion de datos historicos de velas (OHLCV)
- Soporte para exchange Binance (SPOT y FUTURES)
- Soporte para exchange Kraken (SPOT)
- Exportacion a formato CSV
- Sistema de logs en tiempo real

#### Interfaz de Usuario
- Selector de exchange (Binance, Kraken)
- Selector de tipo de mercado (spot, futures, margin)
- Campo de entrada para simbolo del par
- Selector de temporalidad (1m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d, 3d, 1w, 1M)
- Selectores de fecha con calendario desplegable
- Selector de ruta de archivo de salida
- Area de logs con timestamps
- Botones de control (Extraer, Cancelar, Limpiar Logs)

#### Arquitectura
- Implementacion de Clean Architecture
- Capa de dominio con entidades y contratos
- Capa de infraestructura con implementaciones de exchanges
- Capa de aplicacion con casos de uso y servicios
- Capa de presentacion con GUI

#### Documentacion
- README.md completo
- Guia de usuario detallada (USER_GUIDE.md)
- Documentacion de arquitectura (ARCHITECTURE.md)
- Guia de inicio rapido (QUICKSTART.md)
- Changelog para seguimiento de versiones

#### Scripts y Herramientas
- Script de instalacion para Windows (install.bat)
- Script de instalacion para Linux/macOS (install.sh)
- Script de ejecucion para Windows (run.bat)
- Script de ejecucion para Linux/macOS (run.sh)
- Archivo .gitignore configurado
- requirements.txt con todas las dependencias

#### Funcionalidades Tecnicas
- Rate limiting automatico para respetar limites de APIs
- Manejo de multiples requests para rangos grandes de datos
- Callbacks de progreso en tiempo real
- Validacion de simbolos antes de extraccion
- Ejecucion en hilos separados para UI responsiva
- Manejo robusto de errores
- Normalizacion automatica de simbolos

### Caracteristicas Tecnicas

- Python 3.8+ compatible
- Sin dependencias de bases de datos
- APIs publicas sin autenticacion requerida
- Arquitectura extensible para agregar nuevos exchanges
- Codigo documentado con docstrings

## [Futuro] - Planificado

### Por Agregar

- Soporte para mas exchanges (Coinbase, Bybit, OKX, etc.)
- Exportacion a formatos adicionales (JSON, Parquet, Excel)
- Indicadores tecnicos incorporados
- Graficos en tiempo real
- Descarga paralela para acelerar extracciones grandes
- Planificacion de tareas automaticas
- Modo CLI (linea de comandos)
- Sistema de notificaciones al completar
- Configuracion de preferencias guardada
- Historial de extracciones recientes
- Validacion avanzada de datos
- Compresion de archivos CSV
- Modo oscuro en la interfaz

### Mejoras Consideradas

- Cache de datos para evitar re-descargas
- Soporte para proxy
- Reintentos automaticos en caso de error
- Pausa y reanudacion de extracciones
- Estimacion de tiempo restante
- Tests unitarios y de integracion
- Empaquetado como ejecutable standalone (.exe, .app)
- Internacionalizacion (i18n) para multiples idiomas

---

## Formato de Versiones

- **MAJOR**: Cambios incompatibles en la API
- **MINOR**: Nueva funcionalidad compatible con versiones anteriores
- **PATCH**: Correcciones de bugs compatibles con versiones anteriores

## Tipos de Cambios

- **Agregado**: Para nuevas funcionalidades
- **Cambiado**: Para cambios en funcionalidades existentes
- **Obsoleto**: Para funcionalidades que seran eliminadas
- **Eliminado**: Para funcionalidades eliminadas
- **Corregido**: Para correcciones de bugs
- **Seguridad**: Para vulnerabilidades de seguridad
