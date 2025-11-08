#!/usr/bin/env python3
"""
Market Data Extractor - Punto de entrada principal

Este script inicializa y ejecuta la aplicacion de extraccion de datos
historicos de mercados financieros.
"""

import sys
import os
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.presentation import run_app


def main():
    """Funcion principal que inicia la aplicacion."""
    try:
        run_app()
    except KeyboardInterrupt:
        print("\nAplicacion cerrada por el usuario")
        sys.exit(0)
    except Exception as e:
        print(f"Error fatal: {e}")
        print("\nTraceback completo:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
