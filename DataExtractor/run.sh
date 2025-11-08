#!/bin/bash
# Script de inicializacion para Linux/macOS
# Market Data Extractor

echo "============================================"
echo "Market Data Extractor"
echo "============================================"
echo ""

# Verificar si existe el entorno virtual
if [ ! -d "venv" ]; then
    echo "[INFO] No se encontro entorno virtual. Creando..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "[ERROR] No se pudo crear el entorno virtual"
        echo "[ERROR] Verifica que Python 3 este instalado correctamente"
        exit 1
    fi
    echo "[SUCCESS] Entorno virtual creado exitosamente"
    echo ""
fi

# Activar entorno virtual
echo "[INFO] Activando entorno virtual..."
source venv/bin/activate
if [ $? -ne 0 ]; then
    echo "[ERROR] No se pudo activar el entorno virtual"
    exit 1
fi

# Verificar si las dependencias estan instaladas
echo "[INFO] Verificando dependencias..."
python -c "import requests" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "[INFO] Instalando dependencias..."
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "[ERROR] No se pudieron instalar las dependencias"
        exit 1
    fi
    echo "[SUCCESS] Dependencias instaladas exitosamente"
    echo ""
fi

# Ejecutar la aplicacion
echo "[INFO] Iniciando Market Data Extractor..."
echo ""
python main.py

# Desactivar entorno virtual al cerrar
deactivate
