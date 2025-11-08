#!/bin/bash
# Script de instalacion para Linux/macOS
# Market Data Extractor

echo "============================================"
echo "Market Data Extractor - Instalacion"
echo "============================================"
echo ""

# Verificar Python
echo "[INFO] Verificando instalacion de Python..."
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 no esta instalado"
    echo "[ERROR] Por favor instala Python 3.8 o superior"
    exit 1
fi

python3 --version
echo "[SUCCESS] Python encontrado"
echo ""

# Crear entorno virtual
echo "[INFO] Creando entorno virtual..."
if [ -d "venv" ]; then
    echo "[WARNING] El entorno virtual ya existe. Eliminando..."
    rm -rf venv
fi

python3 -m venv venv
if [ $? -ne 0 ]; then
    echo "[ERROR] No se pudo crear el entorno virtual"
    exit 1
fi
echo "[SUCCESS] Entorno virtual creado"
echo ""

# Activar entorno virtual
echo "[INFO] Activando entorno virtual..."
source venv/bin/activate

# Actualizar pip
echo "[INFO] Actualizando pip..."
pip install --upgrade pip

# Instalar dependencias
echo "[INFO] Instalando dependencias..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "[ERROR] Error al instalar dependencias"
    exit 1
fi
echo "[SUCCESS] Dependencias instaladas exitosamente"
echo ""

# Dar permisos de ejecucion al script run.sh
chmod +x run.sh

echo "============================================"
echo "Instalacion completada exitosamente"
echo "============================================"
echo ""
echo "Para ejecutar la aplicacion, usa: ./run.sh"
echo "O ejecuta: python main.py"
echo ""
