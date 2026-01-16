# Guía de Métricas de Performance

El sistema incluye 36+ métricas de análisis cuantitativo organizadas en categorías.

## Métricas de Rentabilidad

### Total Return
**Qué mide**: Ganancia o pérdida total en porcentaje

**Fórmula**: `(Capital Final - Capital Inicial) / Capital Inicial × 100`

**Interpretación**:
- Positivo = ganancia
- Negativo = pérdida
- No considera el tiempo ni el riesgo

### CAGR (Compound Annual Growth Rate)
**Qué mide**: Tasa de crecimiento anualizada compuesta

**Fórmula**: `((Capital Final / Capital Inicial)^(1/años) - 1) × 100`

**Interpretación**:
- 20%+ anual: Excelente
- 10-15% anual: Muy bueno
- 5-10% anual: Aceptable
- < 5% anual: Considerar alternativas

### Expectancy (Expectativa Matemática)
**Qué mide**: Ganancia promedio esperada por operación

**Fórmula**: `(Win Rate × Avg Win) - (Loss Rate × Avg Loss)`

**Interpretación**:
- Debe ser positivo
- Indica si la estrategia tiene ventaja estadística
- Ejemplo: $50 = esperas ganar $50 por trade

---

## Métricas de Riesgo

### Maximum Drawdown (MDD)
**Qué mide**: Mayor caída desde un pico hasta un valle

**Fórmula**: `max((Peak - Valley) / Peak × 100)`

**Interpretación**:
- **< 10%**: Excelente (muy raro)
- **10-20%**: Muy bueno
- **20-30%**: Aceptable
- **> 40%**: Peligroso

**Importante**: Si MDD = 50%, necesitas +100% para recuperarte

### Volatility (Volatilidad)
**Qué mide**: Desviación estándar de los retornos

**Fórmula**: `std(retornos_diarios) × sqrt(252)` (anualizada)

**Interpretación**:
- Menor volatilidad = movimientos más predecibles
- Alta volatilidad = mayor incertidumbre
- BTC típicamente: 60-80% anual
- Acciones: 20-30% anual

### Value at Risk (VaR)
**Qué mide**: Pérdida máxima esperada con X% de confianza

**Ejemplo**: VaR 95% = $1,000 significa:
- 95% de probabilidad de no perder más de $1,000 en un período

### Conditional VaR (CVaR / Expected Shortfall)
**Qué mide**: Pérdida promedio cuando se supera el VaR

**Interpretación**:
- Más conservador que VaR
- Captura el riesgo de cola
- Siempre >= VaR

---

## Métricas Ajustadas por Riesgo

### Sharpe Ratio (La Más Importante)
**Qué mide**: Retorno ajustado por riesgo

**Fórmula**: `(Retorno - Tasa Libre Riesgo) / Volatilidad`

**Interpretación**:
- **< 0**: Peor que no hacer nada
- **0-1**: Subóptimo
- **1-2**: Bueno
- **2-3**: Muy bueno
- **> 3**: Excelente (muy raro)

**Ejemplo**: Sharpe = 1.5 significa obtienes 1.5 unidades de retorno por cada unidad de riesgo

### Sortino Ratio
**Qué mide**: Similar a Sharpe pero solo considera volatilidad negativa

**Ventaja**: No penaliza volatilidad positiva (ganancias grandes)

**Interpretación**: Valores similares a Sharpe

### Calmar Ratio
**Qué mide**: CAGR / Maximum Drawdown

**Interpretación**:
- **> 3**: Excelente
- **2-3**: Muy bueno
- **1-2**: Aceptable
- **< 1**: Malo

**Ejemplo**: Calmar = 2.5 significa ganas 2.5% anual por cada 1% de drawdown máximo

### Omega Ratio
**Qué mide**: Ratio de ganancias ponderadas sobre pérdidas ponderadas

**Fórmula**: `Integral de retornos sobre threshold / Integral de retornos bajo threshold`

**Interpretación**:
- **> 1**: Estrategia rentable
- **> 2**: Muy bueno
- **> 3**: Excelente

---

## Métricas de Consistencia

### Win Rate (Tasa de Acierto)
**Fórmula**: `Trades Ganadores / Total Trades × 100`

**Interpretación**:
- **> 60%**: Excelente
- **50-60%**: Bueno
- **40-50%**: Puede ser viable si Profit Factor es alto
- **< 40%**: Requiere ganancias muy grandes

### Profit Factor
**Qué mide**: Relación entre ganancias brutas y pérdidas brutas

**Fórmula**: `Suma Ganancias / Suma Pérdidas`

**Interpretación**:
- **< 1.0**: Estrategia perdedora
- **1.0-1.5**: Marginal
- **1.5-2.0**: Bueno
- **2.0-3.0**: Muy bueno
- **> 3.0**: Excelente

**Ejemplo**: PF = 2.0 significa ganas $2 por cada $1 que pierdes

### Risk/Reward Ratio
**Fórmula**: `Avg Win / Avg Loss`

**Interpretación**:
- **> 2**: Muy bueno
- **1.5-2**: Bueno
- **< 1**: Necesitas win rate muy alto

**Relación con Win Rate**:
```
Win Rate = 50% → R/R debe ser > 1
Win Rate = 40% → R/R debe ser > 1.5
Win Rate = 60% → R/R puede ser < 1
```

### Recovery Factor
**Qué mide**: Total Return / Maximum Drawdown

**Interpretación**:
- **> 3**: Excelente
- **2-3**: Muy bueno
- **1-2**: Aceptable
- **< 1**: Malo

---

## Métricas de Mean Reversion

Estas métricas son específicas para estrategias de mean reversion y miden qué tan bien la estrategia captura movimientos de reversión a la media.

### MRQS (Mean Reversion Quality Score)
**Qué mide**: Score compuesto de calidad de mean reversion

**Componentes**:
- Target Hit Rate
- Average Excursion
- Time to Target
- Drawdown durante trade

**Interpretación**:
- **> 0.7**: Excelente captura de mean reversion
- **0.5-0.7**: Bueno
- **0.3-0.5**: Aceptable
- **< 0.3**: Necesita mejoras

### Target Hit Rate
**Qué mide**: Porcentaje de trades que alcanzan el target de reversión

**Fórmula**: `Trades que alcanzan target / Total trades × 100`

**Interpretación**:
- **> 70%**: Excelente
- **50-70%**: Bueno
- **< 50%**: Revisar señales de entrada

### Average Excursion
**Qué mide**: Desviación promedio del precio vs. target durante el trade

**Interpretación**:
- Menor es mejor
- Indica eficiencia de entrada
- Excursión alta = entradas prematuras

### Maximum Favorable Excursion (MFE)
**Qué mide**: Máxima ganancia no realizada durante un trade

**Uso**: Optimizar take profits

### Maximum Adverse Excursion (MAE)
**Qué mide**: Máxima pérdida no realizada durante un trade

**Uso**: Optimizar stop losses

---

## Métricas Adicionales

### Consecutive Wins/Losses
**Qué mide**: Rachas máximas de trades ganadores/perdedores

**Uso**: Evaluar consistencia y riesgo de ruina

### Average Trade Duration
**Qué mide**: Tiempo promedio que dura un trade

**Uso**: Determinar si coincide con el estilo de trading deseado

### Trades Per Month
**Qué mide**: Frecuencia de operaciones

**Interpretación**:
- Muy bajo: Puede tener problemas de validez estadística
- Muy alto: Comisiones pueden impactar resultados

### Ulcer Index
**Qué mide**: Profundidad y duración de drawdowns

**Fórmula**: `sqrt(mean(drawdown^2))`

**Interpretación**:
- Menor es mejor
- Captura el "dolor" de los drawdowns

---

## Checklist de Viabilidad

Una estrategia se considera **VIABLE** si cumple:

```
Sharpe Ratio > 1.0
Profit Factor > 1.5
Max Drawdown < 30%
Win Rate > 40% (o Risk/Reward > 2)
Total Trades > 30 (validez estadística)
Calmar Ratio > 1.0
CAGR > Tasa libre de riesgo + 5%
Recovery Factor > 2.0
```

**Si cumples 6 o más → Estrategia viable**
**Si cumples 4-5 → Requiere optimización**
**Si cumples menos de 4 → Necesita rediseño**

---

## Cómo Usar las Métricas

### Flujo de Análisis

1. **Primera Revisión**: Mira Total Return y Win Rate
2. **Evaluación de Riesgo**: Verifica Max Drawdown
3. **Eficiencia**: Calcula Sharpe Ratio
4. **Consistencia**: Analiza Profit Factor
5. **Mean Reversion** (si aplica): Revisa MRQS y Target Hit Rate
6. **Decisión**: Usa el checklist de viabilidad

### Ejemplo Práctico

```
Estrategia A:
- Total Return: 50%
- Win Rate: 65%
- Sharpe: 0.8
- Max DD: 45%
→ RECHAZAR: Sharpe bajo y DD muy alto

Estrategia B:
- Total Return: 35%
- Win Rate: 52%
- Sharpe: 1.8
- Max DD: 18%
→ ACEPTAR: Buen balance riesgo/retorno
```

---

## Trampas Comunes

### 1. Overfitting
**Señal**: Métricas "demasiado buenas" (Sharpe > 4, Win Rate > 80%)
**Solución**: Walk Forward Validation

### 2. Pocos Trades
**Señal**: Total Trades < 30
**Solución**: Necesitas más datos históricos

### 3. Ignorar Drawdown
**Señal**: Buen retorno pero MDD > 40%
**Solución**: Revisar gestión de riesgo

### 4. Solo Mirar Win Rate
**Señal**: Win Rate alto pero Profit Factor < 1.5
**Solución**: Revisar tamaño de pérdidas

### 5. Ignorar Costos
**Señal**: Buenos resultados en backtest, malos en vivo
**Solución**: Incluir comisiones y slippage realistas

---

## Código de Uso

```python
from src.performance import PerformanceAnalyzer

# Crear analizador
analyzer = PerformanceAnalyzer(
    equity_curve=strategy.equity_curve,
    trades=pd.DataFrame(strategy.closed_trades),
    initial_capital=10000
)

# Obtener todas las métricas
metrics = analyzer.calculate_all_metrics()

# Imprimir reporte
analyzer.print_report()

# Acceder a métricas específicas
sharpe = metrics['sharpe_ratio']
mrqs = metrics.get('mrqs', None)  # Solo si aplica mean reversion
```

---

## Recursos Adicionales

Ver ejemplos en:
- `examples/complete_workflow.py` - Análisis completo
- Código: `src/performance.py` - Implementación de métricas
