# Guia de Usuario - Market Data Extractor

## Introduccion

Market Data Extractor es una aplicacion de escritorio que permite descargar datos historicos de velas (candlesticks) desde exchanges de criptomonedas como Binance y Kraken, y guardarlos en formato CSV para su posterior analisis.

## Instalacion

### Requisitos Previos

- Python 3.8 o superior instalado en tu sistema
- Conexion a Internet

### Pasos de Instalacion

1. Descarga o clona el proyecto en tu computadora

2. Abre una terminal o linea de comandos en la carpeta del proyecto

3. (Opcional pero recomendado) Crea un entorno virtual:
   ```bash
   python -m venv venv
   ```

4. Activa el entorno virtual:
   - Windows: `venv\Scripts\activate`
   - Linux/macOS: `source venv/bin/activate`

5. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

## Iniciar la Aplicacion

Ejecuta el siguiente comando en la terminal:

```bash
python main.py
```

Se abrira la ventana principal de la aplicacion.

## Interfaz de Usuario

La interfaz principal consta de los siguientes elementos:

### 1. Exchange

Lista desplegable para seleccionar el exchange del cual deseas obtener datos.

**Exchanges disponibles:**
- **Binance**: Uno de los exchanges mas grandes del mundo
- **Kraken**: Exchange reconocido con sede en USA

### 2. Tipo de Mercado

Selecciona el tipo de mercado para la extraccion:

- **spot**: Mercado al contado (compra/venta directa)
- **futures**: Mercado de futuros (solo Binance)
- **margin**: Mercado de margen (disponibilidad segun exchange)

**Nota**: No todos los exchanges soportan todos los tipos de mercado.

### 3. Simbolo

Ingresa el par de trading que deseas descargar.

**Formato**: BASE/QUOTE

**Ejemplos:**
- `BTC/USDT` - Bitcoin contra Tether
- `ETH/USDT` - Ethereum contra Tether
- `BTC/USD` - Bitcoin contra Dolar (principalmente para Kraken)
- `ETH/BTC` - Ethereum contra Bitcoin

**Tip**: Asegurate de que el simbolo exista en el exchange seleccionado. La aplicacion validara el simbolo antes de iniciar la descarga.

### 4. Temporalidad

Selecciona el intervalo de tiempo para cada vela.

**Temporalidades comunes:**
- `1m` - 1 minuto
- `5m` - 5 minutos
- `15m` - 15 minutos
- `30m` - 30 minutos
- `1h` - 1 hora
- `4h` - 4 horas
- `1d` - 1 dia
- `1w` - 1 semana

**Nota**: Diferentes exchanges pueden soportar diferentes temporalidades.

### 5. Fecha Inicio

Usa el selector de fecha para elegir desde cuando deseas obtener datos.

**Consideraciones:**
- Los exchanges tienen limites en cuanto a que tan atras pueden ir los datos historicos
- Fechas muy antiguas pueden no tener datos disponibles para ciertos pares

### 6. Fecha Fin

Usa el selector de fecha para elegir hasta cuando deseas obtener datos.

**Default**: La fecha/hora actual se establece automaticamente como fecha de fin.

**Tip**: Para obtener datos hasta el momento actual, simplemente usa la fecha de hoy.

### 7. Ruta de Salida

Especifica donde quieres guardar el archivo CSV resultante.

**Como usar:**
1. Haz clic en el boton "Examinar..."
2. Navega a la carpeta donde deseas guardar el archivo
3. Ingresa un nombre para el archivo
4. Haz clic en "Guardar"

**Nombre sugerido**: La aplicacion sugerira un nombre basado en:
- Simbolo del par
- Temporalidad
- Rango de fechas

Ejemplo: `BTC_USDT_1h_20240101_20240131.csv`

### 8. Botones de Control

#### Extraer Datos
Inicia el proceso de extraccion con la configuracion especificada.

**Proceso:**
1. Valida la configuracion
2. Verifica que el simbolo exista
3. Descarga los datos del exchange
4. Guarda los datos en el archivo CSV

#### Cancelar
Cancela la extraccion en progreso (en desarrollo).

#### Limpiar Logs
Limpia el area de logs para mejor visualizacion.

### 9. Area de Logs

Muestra mensajes en tiempo real sobre el progreso de la extraccion.

**Niveles de log:**
- `[INFO]`: Informacion general
- `[WARNING]`: Advertencias
- `[ERROR]`: Errores
- `[SUCCESS]`: Operaciones exitosas

**Informacion mostrada:**
- Timestamp de cada mensaje
- Estado de la validacion
- Progreso de descarga
- Numero de velas descargadas
- Errores o problemas encontrados
- Confirmacion de exportacion exitosa

## Ejemplo de Uso Paso a Paso

### Caso: Descargar datos de BTC/USDT de Binance

1. **Seleccionar Exchange**: Binance

2. **Tipo de Mercado**: spot

3. **Simbolo**: BTC/USDT

4. **Temporalidad**: 1h (velas de 1 hora)

5. **Fecha Inicio**: 2024-01-01

6. **Fecha Fin**: 2024-01-31

7. **Ruta de Salida**:
   - Clic en "Examinar..."
   - Navegar a `C:\Users\TuUsuario\Documents\Trading\`
   - Nombre: `bitcoin_enero_2024.csv`
   - Guardar

8. **Extraer Datos**: Clic en el boton "Extraer Datos"

9. **Observar Logs**: Monitorea el progreso en el area de logs

10. **Resultado**: El archivo CSV estara disponible en la ruta especificada

## Formato del Archivo CSV

El archivo CSV generado contiene las siguientes columnas:

| Columna | Descripcion | Ejemplo |
|---------|-------------|---------|
| timestamp | Fecha y hora de apertura de la vela | 2024-01-01T00:00:00 |
| open | Precio de apertura | 42150.50 |
| high | Precio maximo alcanzado | 42300.75 |
| low | Precio minimo alcanzado | 42050.25 |
| close | Precio de cierre | 42200.00 |
| volume | Volumen negociado | 125.45 |
| symbol | Simbolo del par | BTC/USDT |

### Ejemplo de Contenido

```csv
timestamp,open,high,low,close,volume,symbol
2024-01-01T00:00:00,42150.50,42300.75,42050.25,42200.00,125.45,BTC/USDT
2024-01-01T01:00:00,42200.00,42250.50,42100.00,42180.25,98.32,BTC/USDT
2024-01-01T02:00:00,42180.25,42280.00,42150.00,42260.50,110.78,BTC/USDT
```

## Solucionar Problemas Comunes

### Error: "El simbolo no es valido"

**Causa**: El par de trading no existe en el exchange seleccionado.

**Solucion**:
- Verifica que el simbolo este escrito correctamente
- Consulta la lista de pares disponibles en el sitio web del exchange
- Algunos exchanges usan formatos diferentes (ej: Kraken usa XBT en lugar de BTC)

### Error: "No se obtuvieron datos del exchange"

**Causa**: No hay datos disponibles para el rango de fechas especificado.

**Soluciones**:
- Verifica que las fechas sean correctas
- Intenta con un rango de fechas mas reciente
- Algunos pares nuevos no tienen datos historicos antiguos

### Error de Conexion

**Causa**: Problemas de red o el exchange esta inaccesible.

**Soluciones**:
- Verifica tu conexion a Internet
- Intenta nuevamente mas tarde
- El exchange podria estar en mantenimiento

### La Aplicacion se Congela

**Causa**: Extraccion de grandes volumenes de datos.

**Solucion**:
- Es normal para rangos de fechas muy amplios
- Observa los logs para ver el progreso
- Considera dividir en rangos mas pequeños

### Archivo CSV no se Crea

**Causa**: Problemas de permisos o ruta invalida.

**Soluciones**:
- Verifica que tengas permisos de escritura en la carpeta destino
- Elige una ruta diferente
- No uses caracteres especiales en el nombre del archivo

## Limites y Consideraciones

### Rate Limiting

Los exchanges imponen limites en la cantidad de requests por segundo. La aplicacion respeta estos limites automaticamente, por lo que:

- La descarga puede tomar tiempo para grandes volumenes
- No intentes ejecutar multiples extracciones simultaneas
- Los retrasos son normales y esperados

### Limites de Datos por Request

- **Binance**: Maximo 1000 velas por request
- **Kraken**: Maximo 720 velas por request

La aplicacion maneja esto automaticamente haciendo multiples requests si es necesario.

### Rangos de Fechas Amplios

Para rangos muy amplios (mas de 1 año con temporalidad de minutos):

- La extraccion tomara mas tiempo
- Se haran muchos requests al exchange
- Considera dividir en multiples extracciones

## Consejos y Mejores Practicas

1. **Empieza con Rangos Pequenos**: Prueba primero con rangos de fechas cortos para familiarizarte con la aplicacion.

2. **Verifica el Simbolo**: Antes de extracciones largas, prueba con un rango corto para verificar que el simbolo sea correcto.

3. **Nomenclatura de Archivos**: Usa nombres descriptivos que incluyan simbolo, temporalidad y fechas.

4. **Organizacion**: Crea carpetas por exchange, simbolo o estrategia para organizar tus datos.

5. **Respaldos**: Guarda respaldos de datos importantes, especialmente si son costosos de volver a descargar.

6. **Temporalidad Apropiada**:
   - Para analisis intradiario: 1m, 5m, 15m
   - Para analisis diario: 1h, 4h
   - Para analisis de largo plazo: 1d, 1w

7. **Revision de Logs**: Siempre revisa los logs para confirmar que la extraccion fue exitosa.

## Usar los Datos

Una vez exportados, los datos CSV pueden ser utilizados en:

- **Python**: pandas, numpy para analisis
- **Excel**: Importar CSV para visualizacion
- **Plataformas de Trading**: Backtesting de estrategias
- **R**: Analisis estadistico
- **Herramientas de BI**: Power BI, Tableau

### Ejemplo en Python

```python
import pandas as pd

# Cargar el CSV
df = pd.read_csv('bitcoin_enero_2024.csv')

# Convertir timestamp a datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Establecer como indice
df.set_index('timestamp', inplace=True)

# Analisis basico
print(df.describe())
print(f"Precio maximo: {df['high'].max()}")
print(f"Precio minimo: {df['low'].min()}")
```

## Soporte

Si encuentras problemas o tienes preguntas:

1. Revisa esta guia y la documentacion de arquitectura
2. Verifica los logs para mensajes de error especificos
3. Consulta la documentacion oficial de los exchanges
4. Reporta issues en el repositorio del proyecto

## Actualizaciones Futuras

El proyecto esta diseñado para ser extensible. Futuras versiones podrian incluir:

- Mas exchanges (Coinbase, Bybit, etc.)
- Exportacion a otros formatos (JSON, Parquet)
- Indicadores tecnicos incorporados
- Graficos en tiempo real
- Descarga paralela
- Planificacion de tareas automaticas
