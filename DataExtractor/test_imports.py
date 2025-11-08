#!/usr/bin/env python3
"""
Script de prueba para verificar que todos los imports funcionen correctamente.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    print("Probando imports del dominio...")
    from src.domain import Candle, MarketConfig, MarketType, Timeframe
    print("  [OK] Domain imports")

    print("Probando imports de infraestructura...")
    from src.infrastructure import BinanceExchange, KrakenExchange, CSVExporter
    print("  [OK] Infrastructure imports")

    print("Probando imports de aplicacion...")
    from src.application import DataExtractionService
    print("  [OK] Application imports")

    print("Probando imports de presentacion (widgets)...")
    from src.presentation.gui.widgets import LogWidget, DateSelector
    print("  [OK] Widgets imports")

    print("Probando imports de presentacion (GUI principal)...")
    from src.presentation.gui import MainWindow
    print("  [OK] GUI imports")

    print("\n" + "="*50)
    print("TODOS LOS IMPORTS FUNCIONAN CORRECTAMENTE")
    print("="*50)
    print("\nLa aplicacion esta lista para ejecutarse.")
    print("Usa: python main.py")

except ImportError as e:
    print(f"\n[ERROR] Fallo al importar: {e}")
    sys.exit(1)
except Exception as e:
    print(f"\n[ERROR] Error inesperado: {e}")
    sys.exit(1)
