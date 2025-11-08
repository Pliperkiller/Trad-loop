@echo off
REM Script de instalacion para Windows
REM Market Data Extractor

echo ============================================
echo Market Data Extractor - Instalacion
echo ============================================
echo.

REM Verificar Python
echo [INFO] Verificando instalacion de Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python no esta instalado o no esta en el PATH
    echo [ERROR] Por favor instala Python 3.8 o superior desde https://www.python.org/
    pause
    exit /b 1
)

python --version
echo [SUCCESS] Python encontrado
echo.

REM Crear entorno virtual
echo [INFO] Creando entorno virtual...
if exist "venv\" (
    echo [WARNING] El entorno virtual ya existe. Eliminando...
    rmdir /s /q venv
)

python -m venv venv
if errorlevel 1 (
    echo [ERROR] No se pudo crear el entorno virtual
    pause
    exit /b 1
)
echo [SUCCESS] Entorno virtual creado
echo.

REM Activar entorno virtual
echo [INFO] Activando entorno virtual...
call venv\Scripts\activate.bat

REM Actualizar pip
echo [INFO] Actualizando pip...
python -m pip install --upgrade pip

REM Instalar dependencias
echo [INFO] Instalando dependencias...
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Error al instalar dependencias
    pause
    exit /b 1
)
echo [SUCCESS] Dependencias instaladas exitosamente
echo.

echo ============================================
echo Instalacion completada exitosamente
echo ============================================
echo.
echo Para ejecutar la aplicacion, usa: run.bat
echo O ejecuta: python main.py
echo.
pause
