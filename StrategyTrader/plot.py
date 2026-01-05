#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EJEMPLOS DE USO DEL IDENTIFICADOR DE SOPORTES Y RESISTENCIAS

Este script muestra cómo usar fácilmente el identificador de S/R
para diferentes criptomonedas, timeframes y fuentes de datos.
"""

from identificador_sr import analizar_cripto_auto, SoporteResistenciaIdentificadorMejorado, ultima_vela_cerrada

# ============================================================================
# EJEMPLOS RÁPIDOS - USA ESTOS PARA EMPEZAR
# ============================================================================

def ejemplo_1_bitcoin_yahoo_spot():
    """
    EJEMPLO 1: Bitcoin desde Yahoo Finance (Spot)
    - Fuente: Yahoo Finance
    - Mercado: Spot
    - Timeframe: 1 hora
    - Lookback: 30 días
    """
    print("\n" + "="*70)
    print("  EJEMPLO 1: Bitcoin desde Yahoo Finance (Spot)")
    print("="*70)

    sr = analizar_cripto_auto(
        symbol='BTC-USD',           # Símbolo en Yahoo Finance
        timeframe='1h',             # Velas de 1 hora
        lookback_days=30,           # 30 días hacia atrás
        graficar=True,              # Mostrar gráfico
        estilo='yahoo',             # Estilo Yahoo
        exportar_csv=True,          # Exportar a CSV
        fuente='yahoo',             # Usar Yahoo Finance
        market_type='spot'          # Mercado spot
    )

    return sr


def ejemplo_2_bitcoin_binance_spot():
    """
    EJEMPLO 2: Bitcoin desde Binance (Spot)
    - Fuente: Binance
    - Mercado: Spot
    - Timeframe: 4 horas
    - Lookback: 20 días
    """
    print("\n" + "="*70)
    print("  EJEMPLO 2: Bitcoin desde Binance Spot")
    print("="*70)

    sr = analizar_cripto_auto(
        symbol='BTCUSDT',           # Símbolo en Binance (sin guión)
        timeframe='4h',             # Velas de 4 horas
        lookback_days=20,           # 20 días hacia atrás
        graficar=True,              # Mostrar gráfico
        estilo='charles',           # Estilo Charles
        exportar_csv=True,          # Exportar a CSV
        fuente='binance',           # Usar Binance
        market_type='spot'          # Mercado spot
    )

    return sr


def ejemplo_3_bitcoin_binance_futuros():
    """
    EJEMPLO 3: Bitcoin desde Binance (Futuros USDT-M)
    - Fuente: Binance
    - Mercado: Futuros
    - Timeframe: 1 hora
    - Lookback: 15 días
    """
    print("\n" + "="*70)
    print("  EJEMPLO 3: Bitcoin desde Binance Futuros")
    print("="*70)

    sr = analizar_cripto_auto(
        symbol='BTCUSDT',           # Símbolo en Binance Futures
        timeframe='1h',             # Velas de 1 hora
        lookback_days=15,           # 15 días hacia atrás
        graficar=True,              # Mostrar gráfico
        estilo='charles',           # Estilo Charles
        exportar_csv=True,          # Exportar a CSV
        fuente='binance',           # Usar Binance
        market_type='futures'       # ⭐ Mercado de FUTUROS
    )

    return sr


def ejemplo_4_ethereum_binance_futuros():
    """
    EJEMPLO 4: Ethereum desde Binance (Futuros)
    - Fuente: Binance
    - Mercado: Futuros
    - Timeframe: 15 minutos
    - Lookback: 7 días
    """
    print("\n" + "="*70)
    print("  EJEMPLO 4: Ethereum desde Binance Futuros (15m)")
    print("="*70)

    sr = analizar_cripto_auto(
        symbol='ETHUSDT',           # Ethereum
        timeframe='15m',            # Velas de 15 minutos
        lookback_days=7,            # 7 días hacia atrás
        graficar=True,
        estilo='charles',
        exportar_csv=True,
        fuente='binance',
        market_type='futures'       # Futuros
    )

    return sr


def ejemplo_5_personalizado():
    """
    EJEMPLO 5: Uso personalizado (control manual)

    Si quieres más control sobre el proceso, puedes
    usar la clase directamente en lugar de la función
    automática.
    """
    print("\n" + "="*70)
    print("  EJEMPLO 5: Uso Personalizado (Control Manual)")
    print("="*70)

    # Crear instancia manualmente
    sr = SoporteResistenciaIdentificadorMejorado(
        symbol='SOLUSDT',           # Solana
        timeframe='1h',
        lookback_days=14,
        fuente='binance',
        market_type='futures'       # Futuros
    )

    # Paso 1: Descargar datos
    sr.descargar_datos()

    # Paso 2: Ver información del timeframe
    sr.obtener_info_timeframe()

    # Paso 3: Consolidar niveles de S/R
    sr.consolidar_niveles()

    # Paso 4: Exportar niveles
    sr.exportar_niveles(formato='csv')

    # Paso 5: Graficar
    sr.graficar_auto(estilo='charles', leyenda_externa=True)

    return sr


def ejemplo_6_multiples_cryptos_futuros():
    """
    EJEMPLO 6: Analizar múltiples criptos en Futuros

    Analiza varios pares en un loop
    """
    print("\n" + "="*70)
    print("  EJEMPLO 6: Análisis Múltiple de Futuros")
    print("="*70)

    # Lista de símbolos a analizar
    simbolos = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']

    resultados = {}

    for symbol in simbolos:
        print(f"\n📊 Analizando {symbol}...")

        sr = analizar_cripto_auto(
            symbol=symbol,
            timeframe='1h',
            lookback_days=10,
            graficar=True,          # Genera un gráfico por cada símbolo
            estilo='charles',
            exportar_csv=True,
            fuente='binance',
            market_type='futures'   # Futuros
        )

        resultados[symbol] = sr

    return resultados


def ejemplo_7_comparar_spot_vs_futuros():
    """
    EJEMPLO 7: Comparar Spot vs Futuros

    Analiza el mismo par en ambos mercados para comparar
    """
    print("\n" + "="*70)
    print("  EJEMPLO 7: Comparar BTC Spot vs Futuros")
    print("="*70)

    # Bitcoin en Spot
    print("\n📊 Analizando BTC en SPOT...")
    sr_spot = analizar_cripto_auto(
        symbol='BTCUSDT',
        timeframe='1h',
        lookback_days=20,
        graficar=True,
        estilo='charles',
        exportar_csv=False,
        fuente='binance',
        market_type='spot'
    )

    # Bitcoin en Futuros
    print("\n📊 Analizando BTC en FUTUROS...")
    sr_futures = analizar_cripto_auto(
        symbol='BTCUSDT',
        timeframe='1h',
        lookback_days=30,
        graficar=True,
        estilo='charles',
        exportar_csv=False,
        fuente='binance',
        market_type='futures'
    )

    # Comparar niveles
    print("\n" + "="*70)
    print("COMPARACIÓN DE NIVELES:")
    print(f"  Resistencias SPOT: {len(sr_spot.niveles['resistencias'])}")
    print(f"  Resistencias FUTUROS: {len(sr_futures.niveles['resistencias'])}")
    print(f"  Soportes SPOT: {len(sr_spot.niveles['soportes'])}")
    print(f"  Soportes FUTUROS: {len(sr_futures.niveles['soportes'])}")
    print("="*70)

    return sr_spot, sr_futures


# ============================================================================
# MENÚ INTERACTIVO
# ============================================================================

def mostrar_menu():
    """Muestra el menú de ejemplos disponibles"""
    print("\n" + "="*70)
    print("  IDENTIFICADOR DE SOPORTES Y RESISTENCIAS - MENÚ DE EJEMPLOS")
    print("="*70)
    print("\nElige un ejemplo para ejecutar:\n")
    print("  1. Bitcoin desde Yahoo Finance (Spot, 1h)")
    print("  2. Bitcoin desde Binance Spot (4h)")
    print("  3. Bitcoin desde Binance Futuros (1h) ⭐")
    print("  4. Ethereum desde Binance Futuros (15m) ⭐")
    print("  5. Uso Personalizado con Control Manual")
    print("  6. Análisis Múltiple de Futuros ⭐")
    print("  7. Comparar Spot vs Futuros ⭐")
    print("  0. Salir")
    print("\n" + "="*70)


def ejecutar_menu():
    """Ejecuta el menú interactivo"""
    while True:
        mostrar_menu()

        try:
            opcion = input("\nSelecciona una opción (0-7): ").strip()

            if opcion == '0':
                print("\n✅ ¡Hasta luego!")
                break
            elif opcion == '1':
                ejemplo_1_bitcoin_yahoo_spot()
            elif opcion == '2':
                ejemplo_2_bitcoin_binance_spot()
            elif opcion == '3':
                ejemplo_3_bitcoin_binance_futuros()
            elif opcion == '4':
                ejemplo_4_ethereum_binance_futuros()
            elif opcion == '5':
                ejemplo_5_personalizado()
            elif opcion == '6':
                ejemplo_6_multiples_cryptos_futuros()
            elif opcion == '7':
                ejemplo_7_comparar_spot_vs_futuros()
            else:
                print("\n❌ Opción inválida. Por favor, elige un número del 0 al 7.")

            input("\n\nPresiona ENTER para continuar...")

        except KeyboardInterrupt:
            print("\n\n✅ ¡Hasta luego!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            input("\nPresiona ENTER para continuar...")


# ============================================================================
# EJECUCIÓN PRINCIPAL
# ============================================================================

if __name__ == "__main__":

    # Opción 1: Ejecutar un ejemplo específico directamente
    # Descomenta el que quieras ejecutar:

    # ejemplo_1_bitcoin_yahoo_spot()
    # ejemplo_2_bitcoin_binance_spot()
    # ejemplo_3_bitcoin_binance_futuros()
    # ejemplo_4_ethereum_binance_futuros()
    # ejemplo_5_personalizado()
    # ejemplo_6_multiples_cryptos_futuros()
    # ejemplo_7_comparar_spot_vs_futuros()

    # Opción 2: Ejecutar el menú interactivo
    #ejecutar_menu()


    # ============================================================================
    # GUÍA RÁPIDA DE USO
    # ============================================================================
    """
    GUÍA RÁPIDA:

    1. USO MÁS SIMPLE (una sola línea):

    from identificador_sr import analizar_cripto_auto

    sr = analizar_cripto_auto('BTCUSDT', '1h', 20, fuente='binance', market_type='futures')


    2. PARÁMETROS PRINCIPALES:

    - symbol:
        * Yahoo: 'BTC-USD', 'ETH-USD', etc.
        * Binance: 'BTCUSDT', 'ETHUSDT', etc.

    - timeframe:
        '1m', '5m', '15m', '30m', '1h', '2h', '4h', '1d', '1wk'

    - lookback_days:
        Cuántos días hacia atrás analizar (ej: 20, 30, 45)

    - fuente:
        'yahoo' o 'binance'

    - market_type: ⭐ NUEVO
        'spot' o 'futures' (solo para Binance)


    3. EJEMPLOS QUICK:

    # Bitcoin Spot desde Yahoo
    sr = analizar_cripto_auto('BTC-USD', '1h', 30, fuente='yahoo')

    # Bitcoin Spot desde Binance
    sr = analizar_cripto_auto('BTCUSDT', '4h', 20, fuente='binance', market_type='spot')

    # Bitcoin Futuros desde Binance ⭐
    sr = analizar_cripto_auto('BTCUSDT', '1h', 15, fuente='binance', market_type='futures')

    # Ethereum Futuros ⭐
    sr = analizar_cripto_auto('ETHUSDT', '15m', 7, fuente='binance', market_type='futures')


    4. ACCEDER A LOS NIVELES:

    # Ver resistencias
    print(sr.niveles['resistencias'])

    # Ver soportes
    print(sr.niveles['soportes'])

    # Precio actual
    precio = sr.df['close'].iloc[-1]


    5. PERSONALIZAR GRÁFICO:

    sr.graficar_auto(estilo='charles', leyenda_externa=True)

    # Estilos disponibles: 'charles', 'yahoo', 'mike', 'brasil', etc.


    6. EXPORTAR A CSV:

    sr.exportar_niveles(formato='csv')

    # Se crea un archivo CSV con todos los niveles


    ¡Disfruta el análisis! 🚀
    """

    analizar_cripto_auto(
        symbol='BTCUSDT',           # Símbolo en Binance (sin guión)
        timeframe='1h',             # Velas de 4 horas
        lookback_days=200,           # 20 días hacia atrás
        graficar=True,              # Mostrar gráfico
        estilo='charles',           # Estilo Charles
        exportar_csv=True,          # Exportar a CSV
        fuente='binance',           # Usar Binance
        market_type='futures'          # Mercado
    )