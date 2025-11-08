#!/bin/bash

# Script para inicializar repositorio Git

echo "Inicializando repositorio Git..."

# Inicializar Git
git init

# Añadir todos los archivos
git add .

# Primer commit
git commit -m "Initial commit: Strategy Trader v1.0.0

- Sistema completo de trading algorítmico
- Framework modular de estrategias
- Análisis de performance con 30+ métricas
- 4 métodos de optimización de parámetros
- Visualizaciones avanzadas
- Ejemplos y documentación completa
"

echo ""
echo "Repositorio inicializado exitosamente!"
echo ""
echo "Próximos pasos:"
echo "1. Crear repositorio en GitHub: https://github.com/new"
echo "2. Ejecutar:"
echo "   git remote add origin https://github.com/TU-USUARIO/strategy-trader.git"
echo "   git branch -M main"
echo "   git push -u origin main"

