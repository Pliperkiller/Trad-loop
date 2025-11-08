@echo off
REM Script de actualizacion rapida para Windows
REM Market Data Extractor v1.0.1

echo ============================================
echo Market Data Extractor - Actualizacion v1.0.1
echo ============================================
echo.

echo [INFO] Correccion aplicada:
echo   - Eliminado tkcalendar (causaba error de compatibilidad)
echo   - Nuevo selector de fechas integrado
echo   - Solo requiere Python y requests
echo.

REM Verificar si existe el entorno virtual
if not exist "venv\" (
    echo [ERROR] No se encontro entorno virtual.
    echo [INFO] Ejecuta install.bat primero
    pause
    exit /b 1
)

REM Activar entorno virtual
echo [INFO] Activando entorno virtual...
call venv\Scripts\activate.bat

REM Desinstalar tkcalendar si existe
echo [INFO] Limpiando dependencias antiguas...
pip uninstall tkcalendar -y 2>nul

REM Reinstalar dependencias
echo [INFO] Instalando dependencias actualizadas...
pip install -r requirements.txt

echo.
echo ============================================
echo Actualizacion completada
echo ============================================
echo.
echo La aplicacion ahora deberia funcionar correctamente.
echo Ejecuta run.bat para iniciar la aplicacion.
echo.
pause
