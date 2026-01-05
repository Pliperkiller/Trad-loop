#!/usr/bin/env python3
"""
CLI para extracción de datos de mercado.

Este script permite extraer datos históricos de criptomonedas
desde la línea de comandos sin necesidad de usar la GUI.
"""

import sys
import os
import argparse
from datetime import datetime

# Añadir el directorio actual al path para poder importar los módulos
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.application import DataExtractionService
from src.domain import MarketConfig, MarketType, Timeframe


class Colors:
    """Códigos ANSI para colorear el output de la terminal."""
    RESET = '\033[0m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'


def print_info(message: str):
    """Imprime mensaje informativo."""
    print(f"{Colors.CYAN}ℹ {message}{Colors.RESET}")


def print_success(message: str):
    """Imprime mensaje de éxito."""
    print(f"{Colors.GREEN}✓ {message}{Colors.RESET}")


def print_error(message: str):
    """Imprime mensaje de error."""
    print(f"{Colors.RED}✗ {message}{Colors.RESET}", file=sys.stderr)


def print_warning(message: str):
    """Imprime mensaje de advertencia."""
    print(f"{Colors.YELLOW}⚠ {message}{Colors.RESET}")


def progress_callback(current: int, total: int, message: str):
    """Callback para mostrar progreso en la terminal."""
    percentage = (current / total * 100) if total > 0 else 0
    bar_length = 40
    filled_length = int(bar_length * current / total) if total > 0 else 0
    bar = '█' * filled_length + '░' * (bar_length - filled_length)
    print(f"\r{Colors.BLUE}[{bar}] {percentage:.1f}% ({current}/{total}) {message}{Colors.RESET}", end='', flush=True)
    if current >= total:
        print()  # Nueva línea al terminar


def parse_date(date_str: str) -> datetime:
    """
    Parsea una fecha en formato YYYY-MM-DD.

    Args:
        date_str: Fecha en formato YYYY-MM-DD

    Returns:
        Objeto datetime

    Raises:
        argparse.ArgumentTypeError: Si el formato es inválido
    """
    try:
        return datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        raise argparse.ArgumentTypeError(f"Formato de fecha inválido: {date_str}. Use YYYY-MM-DD")


def create_parser() -> argparse.ArgumentParser:
    """Crea el parser de argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description='Extrae datos históricos de criptomonedas desde exchanges.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Bitcoin hora a hora del 2025 en Binance
  %(prog)s --exchange Binance --symbol BTC/USDT --timeframe 1h \\
    --start-date 2025-01-01 --end-date 2025-01-04 \\
    --output bitcoin_hourly_2025.csv

  # Ethereum diario del último mes en Kraken
  %(prog)s -e Kraken -s ETH/USD -t 1d \\
    -sd 2024-12-01 -ed 2025-01-01 \\
    -o ethereum_daily.csv
        """
    )

    parser.add_argument(
        '-e', '--exchange',
        type=str,
        required=True,
        choices=['Binance', 'Kraken'],
        help='Exchange del cual extraer datos'
    )

    parser.add_argument(
        '-s', '--symbol',
        type=str,
        required=True,
        help='Símbolo del par (ej: BTC/USDT, ETH/USD)'
    )

    parser.add_argument(
        '-t', '--timeframe',
        type=str,
        required=True,
        choices=['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w'],
        help='Temporalidad de las velas'
    )

    parser.add_argument(
        '-sd', '--start-date',
        type=parse_date,
        required=True,
        metavar='YYYY-MM-DD',
        help='Fecha de inicio (formato: YYYY-MM-DD)'
    )

    parser.add_argument(
        '-ed', '--end-date',
        type=parse_date,
        required=True,
        metavar='YYYY-MM-DD',
        help='Fecha de fin (formato: YYYY-MM-DD)'
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        required=True,
        help='Ruta del archivo CSV de salida'
    )

    parser.add_argument(
        '-m', '--market-type',
        type=str,
        default='spot',
        choices=['spot', 'future'],
        help='Tipo de mercado (default: spot)'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Mostrar información detallada'
    )

    return parser


def validate_config(service: DataExtractionService, args: argparse.Namespace) -> bool:
    """
    Valida la configuración antes de extraer datos.

    Args:
        service: Servicio de extracción
        args: Argumentos parseados

    Returns:
        True si la configuración es válida
    """
    # Validar fechas
    if args.start_date >= args.end_date:
        print_error("La fecha de inicio debe ser anterior a la fecha de fin")
        return False

    # Validar que el exchange soporte el timeframe
    timeframe = Timeframe(args.timeframe)
    supported_timeframes = service.get_supported_timeframes(args.exchange)
    if timeframe not in supported_timeframes:
        print_error(f"El exchange {args.exchange} no soporta la temporalidad {args.timeframe}")
        print_info(f"Temporalidades soportadas: {', '.join([tf.value for tf in supported_timeframes])}")
        return False

    # Validar que el exchange soporte el tipo de mercado
    market_type = MarketType(args.market_type)
    supported_market_types = service.get_supported_market_types(args.exchange)
    if market_type not in supported_market_types:
        print_error(f"El exchange {args.exchange} no soporta el tipo de mercado {args.market_type}")
        print_info(f"Tipos de mercado soportados: {', '.join([mt.value for mt in supported_market_types])}")
        return False

    return True


def main():
    """Función principal del CLI."""
    parser = create_parser()
    args = parser.parse_args()

    # Banner
    print(f"\n{Colors.CYAN}{'='*60}")
    print(f"  Market Data Extractor CLI")
    print(f"{'='*60}{Colors.RESET}\n")

    if args.verbose:
        print_info(f"Exchange: {args.exchange}")
        print_info(f"Símbolo: {args.symbol}")
        print_info(f"Temporalidad: {args.timeframe}")
        print_info(f"Tipo de mercado: {args.market_type}")
        print_info(f"Período: {args.start_date.strftime('%Y-%m-%d')} a {args.end_date.strftime('%Y-%m-%d')}")
        print_info(f"Archivo de salida: {args.output}\n")

    # Inicializar servicio
    service = DataExtractionService()

    # Validar configuración
    if not validate_config(service, args):
        sys.exit(1)

    # Crear configuración
    config = MarketConfig(
        exchange=args.exchange,
        symbol=args.symbol,
        market_type=MarketType(args.market_type),
        timeframe=Timeframe(args.timeframe),
        start_date=datetime.combine(args.start_date, datetime.min.time()),
        end_date=datetime.combine(args.end_date, datetime.max.time()),
        output_path=args.output
    )

    # Ejecutar extracción
    print_info("Iniciando extracción de datos...")

    try:
        success, message = service.extract_market_data(
            config,
            progress_callback=progress_callback if not args.verbose else None
        )

        print()  # Nueva línea después de la barra de progreso

        if success:
            print_success(message)
            print_info(f"Datos guardados en: {args.output}")
            sys.exit(0)
        else:
            print_error(message)
            sys.exit(1)

    except KeyboardInterrupt:
        print()
        print_warning("Extracción cancelada por el usuario")
        sys.exit(130)
    except Exception as e:
        print()
        print_error(f"Error inesperado: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
