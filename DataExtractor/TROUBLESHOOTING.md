# Guia de Solucion de Problemas

Esta guia te ayudara a resolver problemas comunes al ejecutar Market Data Extractor.

## Problemas de Instalacion

### Error: "python no esta instalado o no esta en el PATH"

**Causa**: Python no esta instalado o no esta configurado correctamente.

**Solucion**:
1. Descarga e instala Python 3.8 o superior desde https://www.python.org/
2. Durante la instalacion, marca la opcion "Add Python to PATH"
3. Reinicia la terminal y ejecuta nuevamente el script de instalacion

### Error al crear el entorno virtual

**Causa**: El modulo venv no esta instalado.

**Solucion**:

**Windows**:
```bash
python -m pip install --user virtualenv
```

**Linux/Ubuntu**:
```bash
sudo apt-get install python3-venv
```

**macOS**:
```bash
pip3 install virtualenv
```

### Error al instalar dependencias

**Causa**: Problemas de red o pip desactualizado.

**Solucion**:
1. Actualiza pip:
   ```bash
   python -m pip install --upgrade pip
   ```

2. Intenta instalar manualmente:
   ```bash
   pip install requests
   ```

3. Si persiste el error, verifica tu conexion a Internet

## Problemas al Ejecutar

### Error: "Pack.pack_info() takes 1 positional argument but 2 were given"

**Causa**: Este error ya fue corregido en la version actual. Ocurria con versiones antiguas que usaban tkcalendar.

**Solucion**:
1. Asegurate de tener la version mas reciente del codigo
2. El selector de fechas ahora es nativo de tkinter y no requiere dependencias externas
3. Si instalaste antes, reinstala:
   ```bash
   # Windows
   install.bat

   # Linux/macOS
   ./install.sh
   ```

### Error: "ModuleNotFoundError: No module named 'requests'"

**Causa**: Las dependencias no estan instaladas.

**Solucion**:
```bash
pip install -r requirements.txt
```

### Error: "No module named 'tkinter'"

**Causa**: Tkinter no esta instalado (raro en Windows, mas comun en Linux).

**Solucion**:

**Linux/Ubuntu**:
```bash
sudo apt-get install python3-tk
```

**Fedora**:
```bash
sudo dnf install python3-tkinter
```

**macOS**: Tkinter deberia venir con Python, pero si falta:
```bash
brew install python-tk
```

### La ventana no se abre o se cierra inmediatamente

**Causa**: Error en el codigo o en las importaciones.

**Solucion**:
1. Ejecuta el script de prueba:
   ```bash
   python test_imports.py
   ```

2. Si hay errores, revisa los mensajes y corrige las dependencias faltantes

3. Ejecuta directamente para ver el error:
   ```bash
   python main.py
   ```

## Problemas de Extraccion de Datos

### Error: "El simbolo no es valido"

**Causa**: El par de trading no existe en el exchange seleccionado.

**Solucion**:
1. Verifica el simbolo en el sitio web del exchange
2. Binance usa formato sin barra: `BTCUSDT` pero la aplicacion acepta `BTC/USDT`
3. Kraken puede usar notaciones diferentes: `XBT` en lugar de `BTC`

Ejemplos validos:
- Binance: `BTC/USDT`, `ETH/USDT`, `BNB/USDT`
- Kraken: `XBT/USD`, `ETH/USD`, `BTC/EUR`

### Error: "No se obtuvieron datos del exchange"

**Posibles causas**:
1. El rango de fechas no tiene datos disponibles
2. El par es muy nuevo y no hay datos historicos
3. Problemas de conexion

**Solucion**:
1. Intenta con un rango de fechas mas reciente
2. Verifica tu conexion a Internet
3. Prueba con un par popular como `BTC/USDT`
4. Revisa los logs para detalles especificos

### Error: "Connection timeout" o "Connection refused"

**Causa**: Problemas de red o el exchange esta inaccesible.

**Solucion**:
1. Verifica tu conexion a Internet
2. Intenta acceder al sitio web del exchange en tu navegador
3. El exchange puede estar en mantenimiento, intenta mas tarde
4. Verifica si hay un firewall bloqueando la conexion

### Error al guardar el archivo CSV

**Causa**: Permisos insuficientes o ruta invalida.

**Solucion**:
1. Elige una carpeta donde tengas permisos de escritura
2. En Windows, evita carpetas del sistema como `C:\Windows`
3. Usa rutas como `C:\Users\TuUsuario\Documents\`
4. No uses caracteres especiales en el nombre del archivo

### La aplicacion se congela durante la extraccion

**Causa**: Esto es normal para extracciones grandes.

**Explicacion**:
- La extraccion se ejecuta en un hilo separado
- Los logs deben mostrar progreso
- Grandes volumenes de datos toman tiempo

**Si realmente esta congelada**:
1. Verifica si los logs se siguen actualizando
2. Si no hay actualizacion por mas de 5 minutos, cierra y reinicia
3. Intenta con un rango de fechas mas pequeno

## Problemas de Rendimiento

### La extraccion es muy lenta

**Causa**: Rate limiting de la API.

**Explicacion**:
- Los exchanges limitan la cantidad de requests por segundo
- La aplicacion respeta estos limites automaticamente
- Es normal que extracciones grandes tomen tiempo

**Tiempos aproximados**:
- 1 mes de datos 1h: 1-2 minutos
- 1 año de datos 1h: 5-10 minutos
- 1 mes de datos 1m: 10-20 minutos

### Uso alto de memoria

**Causa**: Muchas velas en memoria antes de exportar.

**Solucion**:
- Esto es normal y se limpia al completar
- Si es critico, divide en multiples extracciones mas pequenas

## Problemas en Sistemas Especificos

### Windows: "No se puede ejecutar scripts en este sistema"

**Causa**: Politica de ejecucion de PowerShell.

**Solucion**:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Luego ejecuta nuevamente `.\run.bat`

### Linux/macOS: "Permission denied"

**Causa**: Los scripts no tienen permisos de ejecucion.

**Solucion**:
```bash
chmod +x install.sh run.sh
./install.sh
```

### WSL (Windows Subsystem for Linux)

**Nota**: Tkinter puede tener problemas en WSL1.

**Solucion**:
1. Usa WSL2 si es posible
2. O ejecuta directamente desde Windows PowerShell/CMD
3. Instala un servidor X11 como VcXsrv si necesitas GUI en WSL

## Verificacion del Sistema

Para verificar que todo esta correctamente instalado:

```bash
# Verificar Python
python --version

# Verificar pip
pip --version

# Verificar imports
python test_imports.py

# Ejecutar aplicacion
python main.py
```

## Obtener Ayuda Adicional

Si ninguna de estas soluciones funciona:

1. **Revisa los logs**: El area de logs en la aplicacion muestra detalles del error
2. **Ejecuta test_imports.py**: Te dira exactamente que modulo falta
3. **Verifica versiones**:
   ```bash
   python --version
   pip list
   ```
4. **Reporta el problema**: Crea un issue con:
   - Tu sistema operativo y version
   - Version de Python
   - Mensaje de error completo
   - Pasos para reproducir

## Preguntas Frecuentes

**P: Puedo usar Python 2.7?**
R: No, la aplicacion requiere Python 3.8 o superior.

**P: Necesito credenciales de API?**
R: No, la aplicacion usa solo APIs publicas sin autenticacion.

**P: Puedo descargar datos en tiempo real?**
R: No, esta aplicacion solo descarga datos historicos. Los datos mas recientes pueden tener un retraso de minutos.

**P: Los datos descargados son gratis?**
R: Si, usamos APIs publicas gratuitas de los exchanges.

**P: Hay limite en la cantidad de datos que puedo descargar?**
R: Solo los limites de rate limiting de las APIs. Puedes descargar todos los datos historicos disponibles.

**P: Puedo usar esto para trading automatico?**
R: Esta aplicacion es solo para descargar datos historicos. Para trading automatico necesitarias otra solucion.

## Informacion de Contacto

Para mas ayuda:
- Revisa README.md para documentacion general
- Revisa USER_GUIDE.md para instrucciones detalladas
- Revisa ARCHITECTURE.md para detalles tecnicos
