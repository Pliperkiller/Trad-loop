# Quick Start Guide

## Instalacion Rapida

### Windows

1. Doble clic en `install.bat`
2. Espera a que se instalen las dependencias
3. Doble clic en `run.bat` para ejecutar la aplicacion

### Linux/macOS

1. Abre una terminal en la carpeta del proyecto
2. Ejecuta: `./install.sh`
3. Ejecuta: `./run.sh` para iniciar la aplicacion

## Uso Basico

1. **Selecciona un Exchange**: Binance o Kraken
2. **Tipo de Mercado**: Elige spot o futures
3. **Simbolo**: Ingresa el par, por ejemplo: BTC/USDT
4. **Temporalidad**: Selecciona 1h (1 hora) para empezar
5. **Fechas**: Usa las fechas por defecto (ultimo mes)
6. **Ruta de Salida**: Haz clic en "Examinar..." y elige donde guardar
7. **Extraer**: Haz clic en "Extraer Datos"

## Primer Ejemplo

Para descargar datos de Bitcoin de Binance:

- Exchange: `Binance`
- Tipo de Mercado: `spot`
- Simbolo: `BTC/USDT`
- Temporalidad: `1h`
- Fecha Inicio: (hace 30 dias)
- Fecha Fin: (hoy)
- Guardar como: `bitcoin_datos.csv`

Tiempo estimado: 1-2 minutos

## Documentacion Completa

- [README.md](README.md) - Informacion general del proyecto
- [docs/USER_GUIDE.md](docs/USER_GUIDE.md) - Guia completa de usuario
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) - Documentacion tecnica

## Soporte

Si encuentras problemas:
1. Verifica que Python 3.8+ este instalado
2. Asegurate de tener conexion a Internet
3. Revisa los logs en la aplicacion para detalles del error
