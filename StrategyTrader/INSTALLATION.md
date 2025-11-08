# Guía de Instalación

## Requisitos Previos

- Python 3.8 o superior
- pip
- Git (opcional)

## Opción 1: Instalación desde GitHub

```bash
# Clonar repositorio
git clone https://github.com/tu-usuario/strategy-trader.git
cd strategy-trader

# Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias básicas
pip install -r requirements.txt

# Instalar con optimización Bayesiana (opcional)
pip install -r requirements-full.txt

# Verificar instalación
python -c "from src import TradingStrategy; print('Instalación exitosa!')"
```

## Opción 2: Instalación como Paquete

```bash
pip install -e .
```

## Dependencias

### Básicas (requirements.txt)
- pandas >= 1.5.0
- numpy >= 1.21.0
- matplotlib >= 3.5.0
- seaborn >= 0.12.0
- scipy >= 1.9.0

### Completas (requirements-full.txt)
Incluye todo lo anterior más:
- scikit-optimize >= 0.9.0 (para Bayesian Optimization)
- joblib >= 1.2.0

## Verificación

```python
# Test rápido
from src.strategy import TradingStrategy, TechnicalIndicators
from src.performance import PerformanceAnalyzer
from src.optimizer import StrategyOptimizer

print("Todas las importaciones exitosas!")
```

## Solución de Problemas

### Error: "ModuleNotFoundError: No module named 'src'"

Solución: Ejecuta Python desde el directorio raíz del proyecto

```bash
cd strategy-trader
python examples/complete_workflow.py
```

### Error: "ImportError: cannot import name 'gp_minimize'"

Solución: Instala scikit-optimize

```bash
pip install scikit-optimize
```

### Problemas con Matplotlib en Mac

```bash
pip install matplotlib --upgrade
```

## Próximos Pasos

1. Lee [QUICKSTART.md](QUICKSTART.md)
2. Ejecuta ejemplos en `examples/`
3. Crea tu primera estrategia
