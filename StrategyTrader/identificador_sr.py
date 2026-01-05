#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Identificador Automático de Soportes y Resistencias
Timeframes soportados: 1m, 5m, 15m, 30m, 1h, 2h, 4h, 1d, 1wk
Fuentes: Yahoo Finance, Binance
"""

# ============================================================================
# IMPORTS
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import mplfinance as mpf
from datetime import datetime, timedelta, timezone
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

def ultima_vela_cerrada(timeframe: str):
    now = datetime.now(timezone.utc)

    if timeframe.endswith('m'):
        minutes = int(timeframe.replace('m', ''))
        delta = timedelta(minutes=minutes)
    elif timeframe.endswith('h'):
        hours = int(timeframe.replace('h', ''))
        delta = timedelta(hours=hours)
    elif timeframe == '1d':
        delta = timedelta(days=1)
    elif timeframe == '1wk':
        delta = timedelta(weeks=1)
    else:
        raise ValueError("Timeframe no soportado")

    # Redondear hacia abajo al cierre de vela
    epoch = int(now.timestamp())
    candle_seconds = int(delta.total_seconds())
    closed_ts = epoch - (epoch % candle_seconds)

    return datetime.fromtimestamp(closed_ts, tz=timezone.utc)

# Imports para Binance (se instalarán si no están disponibles)
try:
    from binance.client import Client
    from binance.enums import *
    BINANCE_DISPONIBLE = True
except ImportError:
    BINANCE_DISPONIBLE = False
    print("⚠️  Módulo 'python-binance' no instalado.")
    print("   Para usar Binance, instala con: pip install python-binance")

# ============================================================================
# CLASE PRINCIPAL: IDENTIFICADOR DE SOPORTES Y RESISTENCIAS
# ============================================================================

class SoporteResistenciaIdentificador:
    def __init__(self, symbol='BTC-USD', timeframe='1h', lookback_days=20, fuente='yahoo', market_type='spot'):
        """
        Inicializa el identificador de S/R

        Args:
            symbol: Par a analizar (ej: 'BTC-USD' para Yahoo, 'BTCUSDT' para Binance)
            timeframe: Temporalidad ('1h', '4h', '1d')
            lookback_days: Días hacia atrás a analizar
            fuente: 'yahoo' o 'binance'
            market_type: 'spot' o 'futures' (solo aplica para Binance)
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.lookback_days = lookback_days
        self.fuente = fuente.lower()
        self.market_type = market_type.lower()
        self.df = None
        self.niveles = {
            'soportes': [],
            'resistencias': []
        }

        # Validar fuente
        if self.fuente == 'binance' and not BINANCE_DISPONIBLE:
            print("⚠️  Binance no disponible. Cambiando a Yahoo Finance...")
            self.fuente = 'yahoo'
            self.market_type = 'spot'

        # Yahoo solo soporta spot
        if self.fuente == 'yahoo' and self.market_type == 'futures':
            print("⚠️  Yahoo Finance solo soporta mercado spot. Cambiando a spot...")
            self.market_type = 'spot'
    
    # ------------------------------------------------------------------------
    # FASE 0: DESCARGA DE DATOS
    # ------------------------------------------------------------------------
    
    def descargar_datos(self):
        """Descarga datos históricos desde la fuente especificada"""
        
        if self.fuente == 'binance':
            return self._descargar_datos_binance()
        else:
            return self._descargar_datos_yahoo()
    
    def _descargar_datos_yahoo(self):
        """Descarga datos históricos de Yahoo Finance"""
        print(f"📊 Descargando datos de {self.symbol} desde Yahoo Finance...")
        
        end_date = ultima_vela_cerrada(self.timeframe)
        start_date = end_date - timedelta(days=self.lookback_days)
        
        # Descargar datos
        ticker = yf.Ticker(self.symbol)
        self.df = ticker.history(
            start=start_date,
            end=end_date,
            interval=self.timeframe
        )
        
        # Renombrar columnas a minúsculas
        self.df.columns = [col.lower() for col in self.df.columns]
        
        print(f"✅ Descargados {len(self.df)} registros desde Yahoo Finance")
        print(f"   Desde: {self.df.index[0]}")
        print(f"   Hasta: {self.df.index[-1]}")
        print(f"   Precio actual: ${self.df['close'].iloc[-1]:,.2f}")
        
        return self.df
    
    def _descargar_datos_binance(self):
        """Descarga datos históricos de Binance (Spot o Futures)"""
        market_display = "Spot" if self.market_type == "spot" else "Futuros"
        print(f"📊 Descargando datos de {self.symbol} desde Binance {market_display}...")

        if not BINANCE_DISPONIBLE:
            raise ImportError("python-binance no está instalado. Usa: pip install python-binance")

        # Cliente de Binance (sin API key para datos públicos)
        client = Client("", "")

        # Mapeo de timeframes
        binance_intervals = {
            '1m': Client.KLINE_INTERVAL_1MINUTE,
            '5m': Client.KLINE_INTERVAL_5MINUTE,
            '15m': Client.KLINE_INTERVAL_15MINUTE,
            '30m': Client.KLINE_INTERVAL_30MINUTE,
            '1h': Client.KLINE_INTERVAL_1HOUR,
            '2h': Client.KLINE_INTERVAL_2HOUR,
            '4h': Client.KLINE_INTERVAL_4HOUR,
            '1d': Client.KLINE_INTERVAL_1DAY,
            '1wk': Client.KLINE_INTERVAL_1WEEK,
        }

        interval = binance_intervals.get(self.timeframe)
        if not interval:
            raise ValueError(f"Timeframe '{self.timeframe}' no soportado para Binance")

        end_date = ultima_vela_cerrada(self.timeframe)
        start_date = end_date - timedelta(days=self.lookback_days)

        # Descargar klines según el tipo de mercado
        if self.market_type == 'futures':
            # Usar API de Futuros USDT-M
            klines = client.futures_historical_klines(
                self.symbol,
                interval,
                start_date.strftime("%d %b %Y %H:%M:%S"),
                end_date.strftime("%d %b %Y %H:%M:%S")
            )
        else:
            # Usar API de Spot
            klines = client.get_historical_klines(
                self.symbol,
                interval,
                start_date.strftime("%d %b %Y %H:%M:%S"),
                end_date.strftime("%d %b %Y %H:%M:%S")
            )

        # Convertir a DataFrame
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])

        # Procesar datos
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        # Convertir a numérico
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Mantener solo columnas necesarias
        self.df = df[['open', 'high', 'low', 'close', 'volume']].copy()

        print(f"✅ Descargados {len(self.df)} registros desde Binance {market_display}")
        print(f"   Desde: {self.df.index[0]}")
        print(f"   Hasta: {self.df.index[-1]}")
        print(f"   Precio actual: ${self.df['close'].iloc[-1]:,.2f}")

        return self.df
    
    # ------------------------------------------------------------------------
    # FASE 1: IDENTIFICAR CONSOLIDACIONES HORIZONTALES
    # ------------------------------------------------------------------------
    
    def identificar_consolidaciones(self, window=10, threshold_pct=2.5):
        """
        Identifica zonas de consolidación horizontal
        
        Args:
            window: Número de velas a analizar por ventana
            threshold_pct: % máximo de rango para considerar consolidación
        
        Returns:
            Lista de zonas de consolidación
        """
        print(f"\n🔍 FASE 1: Buscando consolidaciones (window={window}, threshold={threshold_pct}%)...")
        
        consolidaciones = []
        
        for i in range(len(self.df) - window):
            ventana = self.df.iloc[i:i+window]
            
            high_max = ventana['high'].max()
            low_min = ventana['low'].min()
            
            # Calcular rango porcentual
            rango_pct = ((high_max - low_min) / low_min) * 100
            
            if rango_pct <= threshold_pct:
                # Es una consolidación válida
                score = self._calcular_score_consolidacion(ventana, high_max, low_min)
                
                consolidaciones.append({
                    'tipo': 'consolidacion',
                    'soporte': low_min,
                    'resistencia': high_max,
                    'rango_pct': rango_pct,
                    'inicio': ventana.index[0],
                    'fin': ventana.index[-1],
                    'duracion_velas': window,
                    'score': score
                })
        
        # Eliminar duplicados muy cercanos
        consolidaciones = self._eliminar_duplicados(consolidaciones, threshold=0.005)
        
        print(f"   ✅ Encontradas {len(consolidaciones)} consolidaciones válidas")
        
        return consolidaciones
    
    # ------------------------------------------------------------------------
    # FASE 2: IDENTIFICAR HIGH/LOW DIARIOS
    # ------------------------------------------------------------------------
    
    def identificar_highs_lows_diarios(self, dias=7):
        """
        Identifica máximos y mínimos de cada día
        
        Args:
            dias: Número de días hacia atrás a analizar
        """
        print(f"\n🔍 FASE 2: Identificando High/Low de últimos {dias} días...")
        
        highs_lows = []
        
        # Agrupar por día
        df_daily = self.df.copy()
        df_daily['date'] = df_daily.index.date
        
        for date in df_daily['date'].unique()[-dias:]:
            dia_data = df_daily[df_daily['date'] == date]
            
            if len(dia_data) > 0:
                high_del_dia = dia_data['high'].max()
                low_del_dia = dia_data['low'].min()
                
                # Score más alto para días recientes
                dias_antiguedad = (datetime.now().date() - date).days
                score = max(5 - dias_antiguedad, 1)
                
                highs_lows.append({
                    'tipo': 'high_diario',
                    'nivel': high_del_dia,
                    'fecha': date,
                    'score': score + 2  # Bonus por ser high
                })
                
                highs_lows.append({
                    'tipo': 'low_diario',
                    'nivel': low_del_dia,
                    'fecha': date,
                    'score': score + 2  # Bonus por ser low
                })
        
        print(f"   ✅ Identificados {len(highs_lows)} niveles diarios")
        
        return highs_lows
    
    # ------------------------------------------------------------------------
    # FASE 3: IDENTIFICAR NÚMEROS REDONDOS
    # ------------------------------------------------------------------------
    
    def identificar_numeros_redondos(self, rango_pct=5):
        """
        Identifica números redondos cercanos al precio actual
        
        Args:
            rango_pct: % arriba/abajo del precio actual a considerar
        """
        print(f"\n🔍 FASE 3: Identificando números redondos (±{rango_pct}%)...")
        
        precio_actual = self.df['close'].iloc[-1]
        rango_superior = precio_actual * (1 + rango_pct/100)
        rango_inferior = precio_actual * (1 - rango_pct/100)
        
        numeros_redondos = []
        
        # Determinar el step según el precio
        if precio_actual > 10000:
            step = 1000  # Cada $1,000
        elif precio_actual > 1000:
            step = 100   # Cada $100
        else:
            step = 10    # Cada $10
        
        # Generar números redondos
        inicio = int(rango_inferior // step) * step
        fin = int(rango_superior // step) * step + step
        
        for numero in range(inicio, fin, step):
            if rango_inferior <= numero <= rango_superior:
                # Score más alto si está muy cerca del precio actual
                distancia_pct = abs((numero - precio_actual) / precio_actual) * 100
                score = max(5 - int(distancia_pct), 1)
                
                numeros_redondos.append({
                    'tipo': 'numero_redondo',
                    'nivel': numero,
                    'score': score + 1  # Bonus por ser número redondo
                })
        
        print(f"   ✅ Identificados {len(numeros_redondos)} números redondos")
        
        return numeros_redondos
    
    # ------------------------------------------------------------------------
    # FASE 4: IDENTIFICAR SUPPLY/DEMAND ZONES
    # ------------------------------------------------------------------------
    
    def identificar_supply_demand(self, impulso_pct=2.0, volumen_multiplier=1.5):
        """
        Identifica zonas de supply/demand basadas en impulsos
        
        Args:
            impulso_pct: % mínimo de movimiento para considerar impulso
            volumen_multiplier: Multiplicador de volumen promedio
        """
        print(f"\n🔍 FASE 4: Identificando Supply/Demand zones...")
        
        zonas = []
        volumen_promedio = self.df['volume'].mean()
        
        for i in range(3, len(self.df)):
            vela = self.df.iloc[i]
            
            # Calcular movimiento de la vela
            movimiento_pct = abs((vela['close'] - vela['open']) / vela['open']) * 100
            
            # Verificar si es impulso
            if movimiento_pct >= impulso_pct and vela['volume'] > volumen_promedio * volumen_multiplier:
                
                # Obtener velas previas (zona de acumulación/distribución)
                zona_previas = self.df.iloc[i-3:i]
                zona_high = zona_previas['high'].max()
                zona_low = zona_previas['low'].min()
                
                if vela['close'] > vela['open']:
                    # Impulso alcista = DEMAND ZONE
                    zonas.append({
                        'tipo': 'demand_zone',
                        'soporte': zona_low,
                        'resistencia': zona_high,
                        'impulso_pct': movimiento_pct,
                        'volumen': vela['volume'],
                        'fecha': vela.name,
                        'score': 6  # Score alto por ser zona institucional
                    })
                else:
                    # Impulso bajista = SUPPLY ZONE
                    zonas.append({
                        'tipo': 'supply_zone',
                        'soporte': zona_low,
                        'resistencia': zona_high,
                        'impulso_pct': movimiento_pct,
                        'volumen': vela['volume'],
                        'fecha': vela.name,
                        'score': 6
                    })
        
        print(f"   ✅ Identificadas {len(zonas)} zonas de supply/demand")
        
        return zonas
    
    # ------------------------------------------------------------------------
    # FASE 5: CONSOLIDAR Y VALIDAR NIVELES
    # ------------------------------------------------------------------------
    
    def consolidar_niveles(self):
        """Consolida todos los niveles identificados y los valida"""
        print(f"\n🔍 FASE 5: Consolidando y validando niveles...")
        
        # Ejecutar todas las fases
        consolidaciones = self.identificar_consolidaciones()
        highs_lows = self.identificar_highs_lows_diarios()
        numeros_redondos = self.identificar_numeros_redondos()
        supply_demand = self.identificar_supply_demand()
        
        precio_actual = self.df['close'].iloc[-1]
        
        # Procesar consolidaciones
        for consol in consolidaciones:
            self._agregar_nivel(consol['resistencia'], 'resistencia', consol['score'], consol['tipo'])
            self._agregar_nivel(consol['soporte'], 'soporte', consol['score'], consol['tipo'])
        
        # Procesar high/low diarios
        for hl in highs_lows:
            tipo_sr = 'resistencia' if 'high' in hl['tipo'] else 'soporte'
            self._agregar_nivel(hl['nivel'], tipo_sr, hl['score'], hl['tipo'])
        
        # Procesar números redondos
        for nr in numeros_redondos:
            tipo_sr = 'resistencia' if nr['nivel'] > precio_actual else 'soporte'
            self._agregar_nivel(nr['nivel'], tipo_sr, nr['score'], nr['tipo'])
        
        # Procesar supply/demand
        for sd in supply_demand:
            if 'demand' in sd['tipo']:
                self._agregar_nivel(sd['soporte'], 'soporte', sd['score'], sd['tipo'])
            else:
                self._agregar_nivel(sd['resistencia'], 'resistencia', sd['score'], sd['tipo'])
        
        # Validar y limpiar
        self._validar_niveles()
        self._limpiar_niveles_cercanos()
        self._limitar_cantidad()
        
        print(f"\n✅ CONSOLIDACIÓN COMPLETA:")
        print(f"   📈 Resistencias: {len(self.niveles['resistencias'])}")
        print(f"   📉 Soportes: {len(self.niveles['soportes'])}")
    
    # ------------------------------------------------------------------------
    # HELPERS INTERNOS
    # ------------------------------------------------------------------------
    
    def _calcular_score_consolidacion(self, ventana, high, low):
        """Calcula score de una consolidación"""
        score = 0
        
        # Duración (más velas = mejor)
        if len(ventana) >= 10:
            score += 3
        elif len(ventana) >= 6:
            score += 2
        else:
            score += 1
        
        # Volumen (decreciente en consolidación es bueno)
        vol_inicio = ventana['volume'].iloc[:3].mean()
        vol_fin = ventana['volume'].iloc[-3:].mean()
        if vol_fin < vol_inicio:
            score += 1
        
        return score
    
    def _agregar_nivel(self, nivel, tipo, score, origen):
        """Agrega un nivel a la lista correspondiente"""
        self.niveles[f"{tipo}s"].append({
            'nivel': nivel,
            'score': score,
            'origen': origen,
            'toques': self._contar_toques(nivel)
        })
    
    def _contar_toques(self, nivel, threshold_pct=0.3):
        """Cuenta cuántas veces el precio tocó un nivel"""
        toques = 0
        threshold = nivel * (threshold_pct / 100)
        
        for _, row in self.df.iterrows():
            if abs(row['high'] - nivel) <= threshold or abs(row['low'] - nivel) <= threshold:
                toques += 1
        
        return toques
    
    def _validar_niveles(self):
        """Añade score por toques y actualiza scores finales"""
        for tipo in ['soportes', 'resistencias']:
            for nivel in self.niveles[tipo]:
                # Bonus por toques
                if nivel['toques'] >= 3:
                    nivel['score'] += 3
                elif nivel['toques'] == 2:
                    nivel['score'] += 2
    
    def _eliminar_duplicados(self, lista, threshold=0.005):
        """Elimina niveles muy cercanos entre sí"""
        if not lista:
            return lista
        
        resultado = []
        lista_ordenada = sorted(lista, key=lambda x: x.get('soporte', x.get('nivel', 0)))
        
        for item in lista_ordenada:
            nivel_actual = item.get('soporte', item.get('nivel'))
            
            es_duplicado = False
            for item_resultado in resultado:
                nivel_resultado = item_resultado.get('soporte', item_resultado.get('nivel'))
                
                if abs(nivel_actual - nivel_resultado) / nivel_resultado < threshold:
                    es_duplicado = True
                    break
            
            if not es_duplicado:
                resultado.append(item)
        
        return resultado
    
    def _limpiar_niveles_cercanos(self, threshold_pct=0.5):
        """Elimina niveles muy cercanos, manteniendo el de mayor score"""
        for tipo in ['soportes', 'resistencias']:
            niveles = self.niveles[tipo]
            
            # Ordenar por nivel
            niveles.sort(key=lambda x: x['nivel'])
            
            # Filtrar
            resultado = []
            i = 0
            while i < len(niveles):
                nivel_actual = niveles[i]
                j = i + 1
                
                # Buscar niveles cercanos
                while j < len(niveles):
                    nivel_siguiente = niveles[j]
                    distancia_pct = abs(nivel_siguiente['nivel'] - nivel_actual['nivel']) / nivel_actual['nivel'] * 100
                    
                    if distancia_pct < threshold_pct:
                        # Mantener el de mayor score
                        if nivel_siguiente['score'] > nivel_actual['score']:
                            nivel_actual = nivel_siguiente
                        j += 1
                    else:
                        break
                
                resultado.append(nivel_actual)
                i = j
            
            self.niveles[tipo] = resultado
    
    def _limitar_cantidad(self, max_por_tipo=5):
        """Limita la cantidad de niveles a los top N por score"""
        for tipo in ['soportes', 'resistencias']:
            niveles = self.niveles[tipo]
            niveles.sort(key=lambda x: x['score'], reverse=True)
            self.niveles[tipo] = niveles[:max_por_tipo]
    
    # ------------------------------------------------------------------------
    # EXPORTAR NIVELES
    # ------------------------------------------------------------------------
    
    def exportar_niveles(self, formato='csv'):
        """Exporta los niveles a CSV o mostrar en pantalla"""

        market_display = "SPOT" if self.market_type == "spot" else "FUTUROS"
        print(f"\n{'='*60}")
        print(f"NIVELES IDENTIFICADOS - {self.symbol} ({self.timeframe}) [{self.fuente.upper()} {market_display}]")
        print(f"{'='*60}")
        
        print(f"\n📈 RESISTENCIAS:")
        print(f"{'#':<3} {'Nivel':<12} {'Score':<7} {'Toques':<8} {'Origen':<20}")
        print(f"{'-'*60}")
        
        for i, r in enumerate(sorted(self.niveles['resistencias'], 
                                    key=lambda x: x['nivel']), 1):
            print(f"{i:<3} ${r['nivel']:<11,.2f} {r['score']:<7} {r['toques']:<8} {r['origen']:<20}")
        
        print(f"\n📉 SOPORTES:")
        print(f"{'#':<3} {'Nivel':<12} {'Score':<7} {'Toques':<8} {'Origen':<20}")
        print(f"{'-'*60}")
        
        for i, s in enumerate(sorted(self.niveles['soportes'], 
                                    key=lambda x: x['nivel'], reverse=True), 1):
            print(f"{i:<3} ${s['nivel']:<11,.2f} {s['score']:<7} {s['toques']:<8} {s['origen']:<20}")
        
        # Exportar a CSV si se solicita
        if formato == 'csv':
            # Resistencias
            df_resistencias = pd.DataFrame(self.niveles['resistencias'])
            df_resistencias['tipo'] = 'resistencia'
            
            # Soportes
            df_soportes = pd.DataFrame(self.niveles['soportes'])
            df_soportes['tipo'] = 'soporte'
            
            # Combinar
            df_todos = pd.concat([df_resistencias, df_soportes], ignore_index=True)
            df_todos = df_todos.sort_values('nivel', ascending=False)
            
            filename = f"{self.symbol}_{self.timeframe}_{self.fuente}_niveles_{datetime.now().strftime('%Y%m%d')}.csv"
            df_todos.to_csv(filename, index=False)
            print(f"\n✅ Niveles exportados a: {filename}")


# ============================================================================
# CLASE MEJORADA CON GRÁFICOS Y AUTO-CÁLCULO
# ============================================================================

class SoporteResistenciaIdentificadorMejorado(SoporteResistenciaIdentificador):
    """Versión mejorada con gráficos profesionales y cálculo automático de velas"""
    
    def graficar(self, ultimas_velas=200, estilo='charles', mostrar_leyenda_externa=True):
        """
        Grafica el chart con los niveles de S/R - VERSIÓN MEJORADA CON ETIQUETAS
        
        Args:
            ultimas_velas: Número de velas a mostrar
            estilo: Estilo del gráfico ('charles', 'yahoo', 'mike', etc.)
            mostrar_leyenda_externa: Si True, crea tabla externa. Si False, leyenda en el gráfico
        """
        print(f"\n📊 Generando gráfico...")
        
        # Preparar datos
        df_plot = self.df.tail(ultimas_velas).copy()
        precio_actual = df_plot['close'].iloc[-1]
        
        # Crear figura con mejor distribución
        if mostrar_leyenda_externa:
            # Con tabla externa
            fig = plt.figure(figsize=(20, 12))
            gs = fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[3, 1],
                                hspace=0.3, wspace=0.1)
            ax_candles = fig.add_subplot(gs[0, 0])
            ax_volume = fig.add_subplot(gs[1, 0], sharex=ax_candles)
            ax_table = fig.add_subplot(gs[:, 1])
            ax_table.axis('off')
        else:
            # Sin tabla externa (leyenda mejorada en gráfico)
            fig, axes = plt.subplots(2, 1, figsize=(18, 10), 
                                    gridspec_kw={'height_ratios': [3, 1]})
            ax_candles = axes[0]
            ax_volume = axes[1]
        
        # Preparar líneas horizontales para resistencias
        hlines_resistencias = []
        labels_resistencias = []
        colors_resistencias = []
        
        # Gradiente de colores para resistencias (más oscuro = más importante)
        color_palette_resistencias = ['#8B0000', '#B22222', '#CD5C5C', '#F08080', '#FFA07A']
        
        # Mapeo de orígenes legibles
        origen_map = {
            'consolidacion': 'Consolidación',
            'high_diario': 'High Diario',
            'numero_redondo': 'Número Redondo',
            'supply_zone': 'Supply Zone',
            'low_diario': 'Low Diario',
            'demand_zone': 'Demand Zone'
        }
        
        for i, r in enumerate(sorted(self.niveles['resistencias'], 
                                    key=lambda x: x['nivel']), 1):
            hlines_resistencias.append(r['nivel'])
            
            origen_legible = origen_map.get(r['origen'], r['origen'])
            distancia_pct = ((r['nivel'] - precio_actual) / precio_actual) * 100
            
            labels_resistencias.append({
                'num': i,
                'nivel': r['nivel'],
                'origen': origen_legible,
                'score': r['score'],
                'toques': r['toques'],
                'distancia': distancia_pct
            })
            
            color_idx = min(i-1, len(color_palette_resistencias)-1)
            colors_resistencias.append(color_palette_resistencias[color_idx])
        
        # Preparar líneas horizontales para soportes
        hlines_soportes = []
        labels_soportes = []
        colors_soportes = []
        
        color_palette_soportes = ['#006400', '#228B22', '#32CD32', '#90EE90', '#98FB98']
        
        for i, s in enumerate(sorted(self.niveles['soportes'], 
                                    key=lambda x: x['nivel'], reverse=True), 1):
            hlines_soportes.append(s['nivel'])
            
            origen_legible = origen_map.get(s['origen'], s['origen'])
            distancia_pct = ((s['nivel'] - precio_actual) / precio_actual) * 100
            
            labels_soportes.append({
                'num': i,
                'nivel': s['nivel'],
                'origen': origen_legible,
                'score': s['score'],
                'toques': s['toques'],
                'distancia': distancia_pct
            })
            
            color_idx = min(i-1, len(color_palette_soportes)-1)
            colors_soportes.append(color_palette_soportes[color_idx])
        
        # Configurar líneas con colores individuales
        hlines = dict(
            hlines=hlines_resistencias + hlines_soportes,
            colors=colors_resistencias + colors_soportes,
            linestyle='--',
            linewidths=2,
            alpha=0.8
        )
        
        # Graficar con mplfinance
        mpf.plot(
            df_plot,
            type='candle',
            style=estilo,
            ax=ax_candles,
            volume=ax_volume,
            hlines=hlines,
            datetime_format='%Y-%m-%d %H:%M',
            xrotation=15
        )
        
        # ========================================================================
        # AGREGAR ETIQUETAS DE TEXTO EN LAS LÍNEAS
        # ========================================================================
        
        # Obtener límites del eje X
        xlim = ax_candles.get_xlim()
        x_pos_label = xlim[0] + (xlim[1] - xlim[0]) * 0.02  # 2% desde la izquierda
        
        # ETIQUETAS PARA RESISTENCIAS
        for i, (label, color) in enumerate(zip(labels_resistencias, colors_resistencias)):
            nivel = label['nivel']
            
            # Texto de la etiqueta
            texto_etiqueta = f"R{label['num']}: ${nivel:,.0f}"
            
            # Agregar etiqueta en el gráfico
            ax_candles.text(
                x_pos_label,
                nivel,
                texto_etiqueta,
                color='white',
                fontsize=10,
                fontweight='bold',
                verticalalignment='center',
                horizontalalignment='left',
                bbox=dict(
                    boxstyle='round,pad=0.5',
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.9,
                    linewidth=2
                ),
                zorder=10  # Asegurar que esté por encima de las velas
            )
        
        # ETIQUETAS PARA SOPORTES
        for i, (label, color) in enumerate(zip(labels_soportes, colors_soportes)):
            nivel = label['nivel']
            
            # Texto de la etiqueta
            texto_etiqueta = f"S{label['num']}: ${nivel:,.0f}"
            
            # Agregar etiqueta en el gráfico
            ax_candles.text(
                x_pos_label,
                nivel,
                texto_etiqueta,
                color='white',
                fontsize=10,
                fontweight='bold',
                verticalalignment='center',
                horizontalalignment='left',
                bbox=dict(
                    boxstyle='round,pad=0.5',
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.9,
                    linewidth=2
                ),
                zorder=10
            )
        
        # ========================================================================
        
        # Título mejorado
        fuente_display = self.fuente.upper()
        market_display = "SPOT" if self.market_type == "spot" else "FUTUROS"
        ax_candles.set_title(
            f'{self.symbol} - {self.timeframe.upper()} [{fuente_display} {market_display}] | Precio Actual: ${precio_actual:,.2f}\n'
            f'Análisis de Soportes y Resistencias | {len(self.niveles["resistencias"])} Resistencias, '
            f'{len(self.niveles["soportes"])} Soportes',
            fontsize=16,
            fontweight='bold',
            pad=20
        )
        
        # Configurar grid
        ax_candles.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        ax_volume.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        
        if mostrar_leyenda_externa:
            self._crear_tabla_externa(ax_table, labels_resistencias, labels_soportes, 
                                    colors_resistencias, colors_soportes, precio_actual)
        else:
            self._crear_leyenda_interna(ax_candles, labels_resistencias, labels_soportes,
                                        colors_resistencias, colors_soportes)
        
        # Ajustar layout
        plt.tight_layout()
        
        # Guardar
        filename = f"{self.symbol}_{self.timeframe}_{self.fuente}_SR_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(filename, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"✅ Gráfico guardado: {filename}")
        
        plt.show()

    def _crear_tabla_externa(self, ax, labels_resistencias, labels_soportes, 
                            colors_r, colors_s, precio_actual):
        """Crea tabla externa profesional con los niveles"""
        
        ax.text(0.5, 0.98, 'NIVELES CLAVE', 
            ha='center', va='top', fontsize=14, fontweight='bold',
            transform=ax.transAxes)
        
        ax.text(0.5, 0.94, f'Precio: ${precio_actual:,.2f}',
            ha='center', va='top', fontsize=11, 
            transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        # RESISTENCIAS
        y_pos = 0.88
        ax.text(0.05, y_pos, '📈 RESISTENCIAS', 
            fontsize=12, fontweight='bold', color='red',
            transform=ax.transAxes)
        y_pos -= 0.04
        
        # Headers
        ax.text(0.05, y_pos, '#', fontsize=8, fontweight='bold', transform=ax.transAxes)
        ax.text(0.15, y_pos, 'Nivel', fontsize=8, fontweight='bold', transform=ax.transAxes)
        ax.text(0.35, y_pos, 'Dist%', fontsize=8, fontweight='bold', transform=ax.transAxes)
        ax.text(0.50, y_pos, 'Score', fontsize=8, fontweight='bold', transform=ax.transAxes)
        ax.text(0.65, y_pos, 'Origen', fontsize=8, fontweight='bold', transform=ax.transAxes)
        y_pos -= 0.02
        
        ax.plot([0.05, 0.95], [y_pos, y_pos], 'k-', linewidth=0.5, transform=ax.transAxes)
        y_pos -= 0.03
        
        for i, (label, color) in enumerate(zip(labels_resistencias, colors_r)):
            ax.text(0.05, y_pos, f"R{label['num']}", fontsize=9, 
                color=color, fontweight='bold', transform=ax.transAxes)
            ax.text(0.15, y_pos, f"${label['nivel']:,.0f}", fontsize=9, 
                color=color, transform=ax.transAxes)
            ax.text(0.35, y_pos, f"+{label['distancia']:.1f}%", fontsize=8, 
                color='darkred', transform=ax.transAxes)
            
            stars = '★' * min(label['score'], 5)
            ax.text(0.50, y_pos, f"{stars}", fontsize=9, 
                color='gold', transform=ax.transAxes)
            
            origen_corto = label['origen'][:12] + '...' if len(label['origen']) > 12 else label['origen']
            ax.text(0.65, y_pos, origen_corto, fontsize=8, 
                color='gray', transform=ax.transAxes)
            
            y_pos -= 0.04
        
        y_pos -= 0.03
        
        # SOPORTES
        ax.text(0.05, y_pos, '📉 SOPORTES', 
            fontsize=12, fontweight='bold', color='green',
            transform=ax.transAxes)
        y_pos -= 0.04
        
        ax.text(0.05, y_pos, '#', fontsize=8, fontweight='bold', transform=ax.transAxes)
        ax.text(0.15, y_pos, 'Nivel', fontsize=8, fontweight='bold', transform=ax.transAxes)
        ax.text(0.35, y_pos, 'Dist%', fontsize=8, fontweight='bold', transform=ax.transAxes)
        ax.text(0.50, y_pos, 'Score', fontsize=8, fontweight='bold', transform=ax.transAxes)
        ax.text(0.65, y_pos, 'Origen', fontsize=8, fontweight='bold', transform=ax.transAxes)
        y_pos -= 0.02
        
        ax.plot([0.05, 0.95], [y_pos, y_pos], 'k-', linewidth=0.5, transform=ax.transAxes)
        y_pos -= 0.03
        
        for i, (label, color) in enumerate(zip(labels_soportes, colors_s)):
            ax.text(0.05, y_pos, f"S{label['num']}", fontsize=9, 
                color=color, fontweight='bold', transform=ax.transAxes)
            ax.text(0.15, y_pos, f"${label['nivel']:,.0f}", fontsize=9, 
                color=color, transform=ax.transAxes)
            ax.text(0.35, y_pos, f"{label['distancia']:.1f}%", fontsize=8, 
                color='darkgreen', transform=ax.transAxes)
            
            stars = '★' * min(label['score'], 5)
            ax.text(0.50, y_pos, f"{stars}", fontsize=9, 
                color='gold', transform=ax.transAxes)
            
            origen_corto = label['origen'][:12] + '...' if len(label['origen']) > 12 else label['origen']
            ax.text(0.65, y_pos, origen_corto, fontsize=8, 
                color='gray', transform=ax.transAxes)
            
            y_pos -= 0.04
        
        y_pos = 0.05
        ax.text(0.5, y_pos, 'Score: ★ = Débil | ★★★ = Medio | ★★★★★ = Fuerte',
            ha='center', fontsize=8, style='italic', color='gray',
            transform=ax.transAxes)

    def _crear_leyenda_interna(self, ax, labels_resistencias, labels_soportes,
                              colors_r, colors_s):
        """Crea leyenda mejorada dentro del gráfico"""
        
        y_pos = 0.98
        for i, (label, color) in enumerate(zip(labels_resistencias, colors_r)):
            text = f"R{label['num']}: ${label['nivel']:,.0f} ({label['origen'][:8]}, {'★'*min(label['score'],5)})"
            ax.text(1.01, y_pos, text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                color=color, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            edgecolor=color, alpha=0.9, linewidth=1.5))
            y_pos -= 0.045
        
        y_pos -= 0.02
        
        for i, (label, color) in enumerate(zip(labels_soportes, colors_s)):
            text = f"S{label['num']}: ${label['nivel']:,.0f} ({label['origen'][:8]}, {'★'*min(label['score'],5)})"
            ax.text(1.01, y_pos, text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                color=color, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            edgecolor=color, alpha=0.9, linewidth=1.5))
            y_pos -= 0.045
    
    def _calcular_velas_por_timeframe(self):
        """Calcula el número de velas a graficar basado en timeframe y lookback_days"""
        
        velas_por_dia = {
            '1m': 1440, '5m': 288, '15m': 96, '30m': 48,
            '1h': 24, '2h': 12, '4h': 6, '1d': 1,
            '1wk': 1/7, '1mo': 1/30
        }
        
        vpd = velas_por_dia.get(self.timeframe, 24)
        velas_totales = int(self.lookback_days * vpd)
        velas_disponibles = len(self.df)
        velas_a_graficar = min(velas_totales, velas_disponibles)
        
        velas_minimas = {
            '1m': 200, '5m': 150, '15m': 100, '30m': 80,
            '1h': 60, '2h': 50, '4h': 40, '1d': 30,
            '1wk': 20, '1mo': 12
        }
        
        minimo = velas_minimas.get(self.timeframe, 60)
        velas_a_graficar = max(velas_a_graficar, minimo)
        
        return velas_a_graficar
    
    def graficar_auto(self, estilo='charles', porcentaje_datos=100, leyenda_externa=True):
        """Grafica automáticamente calculando las velas óptimas"""
        
        velas_calculadas = self._calcular_velas_por_timeframe()
        velas_ajustadas = int(velas_calculadas * (porcentaje_datos / 100))
        
        print(f"\n📊 Configuración automática del gráfico:")
        print(f"   Timeframe: {self.timeframe}")
        print(f"   Lookback days: {self.lookback_days}")
        print(f"   Fuente: {self.fuente.upper()}")
        print(f"   Velas calculadas: {velas_calculadas}")
        print(f"   Velas a graficar: {velas_ajustadas}")
        print(f"   Velas disponibles: {len(self.df)}")
        print(f"   Leyenda: {'Externa (Tabla)' if leyenda_externa else 'Interna (Gráfico)'}")
        
        self.graficar(ultimas_velas=velas_ajustadas, estilo=estilo, 
                     mostrar_leyenda_externa=leyenda_externa)
    
    def obtener_info_timeframe(self):
        """Muestra información detallada sobre el timeframe y datos"""
        
        velas_totales = len(self.df)
        velas_a_graficar = self._calcular_velas_por_timeframe()
        
        inicio_grafico = self.df.index[-velas_a_graficar]
        fin_grafico = self.df.index[-1]
        tiempo_real = fin_grafico - inicio_grafico
        
        print(f"\n{'='*60}")
        print(f"INFORMACIÓN DEL TIMEFRAME")
        print(f"{'='*60}")
        print(f"Par: {self.symbol}")
        print(f"Timeframe: {self.timeframe}")
        print(f"Fuente: {self.fuente.upper()}")
        print(f"Lookback solicitado: {self.lookback_days} días")
        print(f"\nDATOS DESCARGADOS:")
        print(f"  - Velas totales: {velas_totales}")
        print(f"  - Período: {self.df.index[0]} a {self.df.index[-1]}")
        print(f"  - Duración real: {(self.df.index[-1] - self.df.index[0]).days} días")
        print(f"\nGRÁFICO GENERADO:")
        print(f"  - Velas a graficar: {velas_a_graficar}")
        print(f"  - Período gráfico: {inicio_grafico} a {fin_grafico}")
        print(f"  - Duración gráfico: {tiempo_real.days} días, {tiempo_real.seconds//3600} horas")
        print(f"{'='*60}\n")


# ============================================================================
# FUNCIONES DE USO SIMPLIFICADO
# ============================================================================

def analizar_cripto_auto(symbol='BTC-USD', timeframe='1h', lookback_days=20,
                         graficar=True, estilo='charles', exportar_csv=True,
                         fuente='yahoo', market_type='spot'):
    """
    Función all-in-one con cálculo automático de velas

    Args:
        symbol: Par a analizar
            - Para Yahoo: 'BTC-USD', 'ETH-USD', etc.
            - Para Binance Spot: 'BTCUSDT', 'ETHUSDT', etc.
            - Para Binance Futures: 'BTCUSDT', 'ETHUSDT', etc.
        timeframe: Temporalidad ('1m', '5m', '15m', '30m', '1h', '2h', '4h', '1d', '1wk')
        lookback_days: Días hacia atrás a analizar
        graficar: Si True, genera gráfico
        estilo: Estilo del gráfico ('charles', 'yahoo', 'mike')
        exportar_csv: Si True, exporta niveles a CSV
        fuente: 'yahoo' o 'binance'
        market_type: 'spot' o 'futures' (solo para Binance)
    """

    market_display = "SPOT" if market_type == "spot" else "FUTUROS"
    print("="*60)
    print(f"  ANÁLISIS AUTOMÁTICO DE SOPORTES Y RESISTENCIAS")
    print(f"  Par: {symbol} | Timeframe: {timeframe} | Lookback: {lookback_days} días")
    print(f"  Fuente: {fuente.upper()} | Mercado: {market_display}")
    print("="*60)

    sr = SoporteResistenciaIdentificadorMejorado(
        symbol=symbol,
        timeframe=timeframe,
        lookback_days=lookback_days,
        fuente=fuente,
        market_type=market_type
    )
    
    sr.descargar_datos()
    sr.obtener_info_timeframe()
    sr.consolidar_niveles()
    sr.exportar_niveles(formato='csv' if exportar_csv else 'console')
    
    if graficar:
        sr.graficar_auto(estilo=estilo)
    
    return sr


# ============================================================================
# EJECUCIÓN PRINCIPAL
# ============================================================================

if __name__ == "__main__":

    print("\n" + "="*60)
    print("EJEMPLO 1: BTC desde Yahoo Finance Spot (1H, 45 días)")
    print("="*60)

    sr_btc_yahoo = analizar_cripto_auto(
        symbol='BTC-USD',
        timeframe='1h',
        lookback_days=45,
        graficar=True,
        estilo='yahoo',
        fuente='yahoo',
        market_type='spot'
    )

    print("\n" + "="*60)
    print("EJEMPLO 2: BTC desde Binance Spot (4H, 30 días)")
    print("="*60)

    # Para Binance, el símbolo debe ser 'BTCUSDT' (sin guión)
    sr_btc_binance_spot = analizar_cripto_auto(
        symbol='BTCUSDT',
        timeframe='4h',
        lookback_days=30,
        graficar=True,
        estilo='charles',
        fuente='binance',
        market_type='spot'
    )

    print("\n" + "="*60)
    print("EJEMPLO 3: BTC desde Binance Futuros (1H, 20 días)")
    print("="*60)

    sr_btc_binance_futures = analizar_cripto_auto(
        symbol='BTCUSDT',
        timeframe='1h',
        lookback_days=20,
        graficar=True,
        estilo='charles',
        fuente='binance',
        market_type='futures'
    )

    print("\n✅ Análisis completado!")