@echo off
REM Script de inicializacion para Windows
REM Market Data Extractor

echo ============================================
echo Market Data Extractor
echo ============================================
echo.

REM Verificar si existe el entorno virtual
if not exist "venv\" (
    echo [INFO] No se encontro entorno virtual. Creando...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] No se pudo crear el entorno virtual
        echo [ERROR] Verifica que Python este instalado correctamente
        pause
        exit /b 1
    )
    echo [SUCCESS] Entorno virtual creado exitosamente
    echo.
)

REM Activar entorno virtual
echo [INFO] Activando entorno virtual...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] No se pudo activar el entorno virtual
    pause
    exit /b 1
)

REM Verificar si las dependencias estan instaladas
echo [INFO] Verificando dependencias...
python -c "import requests" 2>nul
if errorlevel 1 (
    echo [INFO] Instalando dependencias...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] No se pudieron instalar las dependencias
        pause
        exit /b 1
    )
    echo [SUCCESS] Dependencias instaladas exitosamente
    echo.
)

REM Ejecutar la aplicacion
echo [INFO] Iniciando Market Data Extractor...
echo.
python main.py

REM Desactivar entorno virtual al cerrar
deactivate
