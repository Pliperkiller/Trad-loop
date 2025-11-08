#!/usr/bin/env python3
"""
Script para verificar el entorno y dependencias instaladas.
"""

import sys
import importlib.util

print("=" * 60)
print("Verificacion del Entorno")
print("=" * 60)
print(f"\nVersion de Python: {sys.version}")
print(f"Ejecutable: {sys.executable}")
print("\n" + "-" * 60)
print("Verificando dependencias:")
print("-" * 60)

# Verificar requests (requerida)
try:
    import requests
    print(f"[OK] requests instalado - version: {requests.__version__}")
except ImportError:
    print("[ERROR] requests NO instalado (REQUERIDO)")

# Verificar tkcalendar (NO deberia estar)
try:
    import tkcalendar
    print(f"[WARNING] tkcalendar instalado - version: {tkcalendar.__version__}")
    print("         ESTE ES EL PROBLEMA - tkcalendar debe ser desinstalado")
except ImportError:
    print("[OK] tkcalendar NO instalado (correcto)")

# Verificar tkinter (deberia estar)
try:
    import tkinter
    print(f"[OK] tkinter disponible - version tk: {tkinter.TkVersion}")
except ImportError:
    print("[ERROR] tkinter NO disponible (REQUERIDO)")

print("\n" + "-" * 60)
print("Verificando modulos del proyecto:")
print("-" * 60)

# Verificar imports del proyecto
modules = [
    'src.domain',
    'src.infrastructure',
    'src.application',
    'src.presentation.gui.widgets.date_selector',
    'src.presentation.gui.widgets.log_widget'
]

for module in modules:
    try:
        spec = importlib.util.find_spec(module)
        if spec is not None:
            print(f"[OK] {module}")
        else:
            print(f"[ERROR] {module} - no encontrado")
    except Exception as e:
        print(f"[ERROR] {module} - {e}")

print("\n" + "=" * 60)
print("Solucion:")
print("=" * 60)

try:
    import tkcalendar
    print("\nPROBLEMA DETECTADO: tkcalendar esta instalado")
    print("\nPara solucionarlo, ejecuta:")
    print("  pip uninstall tkcalendar -y")
    print("  python main.py")
except ImportError:
    print("\nEl entorno esta configurado correctamente.")
    print("Si aun hay errores, ejecuta: python main.py")
