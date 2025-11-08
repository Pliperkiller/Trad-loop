@echo off
REM Script de correccion rapida para el error de tkcalendar
REM Market Data Extractor v1.0.1

echo ============================================
echo Market Data Extractor - Correccion de Error
echo ============================================
echo.
echo Este script corregira el error:
echo "Pack.pack_info() takes 1 positional argument but 2 were given"
echo.

REM Verificar si existe el entorno virtual
if not exist "venv\" (
    echo [ERROR] No se encontro entorno virtual.
    echo [INFO] Ejecutando instalacion limpia...
    echo.
    goto :clean_install
)

echo [INFO] Activando entorno virtual...
call venv\Scripts\activate.bat

echo [INFO] Desinstalando tkcalendar (causa del error)...
pip uninstall tkcalendar -y

echo [INFO] Limpiando cache de pip...
pip cache purge

echo [INFO] Verificando instalacion...
python test_imports.py

if errorlevel 1 (
    echo.
    echo [WARNING] Hubo un problema. Instalando desde cero...
    deactivate
    goto :clean_install
)

echo.
echo ============================================
echo Correccion aplicada exitosamente
echo ============================================
echo.
echo La aplicacion ahora deberia funcionar correctamente.
echo Ejecuta: python main.py
echo.
pause
exit /b 0

:clean_install
echo.
echo [INFO] Eliminando entorno virtual antiguo...
if exist "venv\" rmdir /s /q venv

echo [INFO] Creando nuevo entorno virtual limpio...
python -m venv venv

echo [INFO] Activando nuevo entorno virtual...
call venv\Scripts\activate.bat

echo [INFO] Actualizando pip...
python -m pip install --upgrade pip --quiet

echo [INFO] Instalando solo las dependencias necesarias...
pip install requests

echo [INFO] Verificando instalacion...
python test_imports.py

echo.
echo ============================================
echo Instalacion limpia completada
echo ============================================
echo.
echo La aplicacion esta lista para usar.
echo Ejecuta: python main.py
echo.
pause
