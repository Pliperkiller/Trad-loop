# Indicadores - Manual de Componente

## Descripcion

El modulo **indicators** proporciona una coleccion completa de indicadores tecnicos y fundamentales para analisis de mercado.

## Arquitectura

```
indicators/
├── base.py                # Clases base y protocolos
├── technical/             # Indicadores tecnicos
│   ├── momentum.py        # RSI, MACD, Stochastic, etc.
│   ├── trend.py           # SMA, EMA, ADX, Supertrend
│   ├── volatility.py      # Bollinger, ATR, Keltner
│   ├── volume.py          # OBV, VWAP, MFI
│   ├── pivots.py          # Pivot Points, Fibonacci
│   └── ichimoku.py        # Ichimoku Cloud
├── fundamental/           # Datos fundamentales
│   ├── base_client.py     # Cliente API base
│   ├── coingecko.py       # CoinGecko
│   ├── glassnode.py       # Glassnode
│   ├── defillama.py       # DeFi Llama
│   ├── alternative_me.py  # Fear & Greed
│   └── santiment.py       # Santiment
└── utils/
    ├── cache.py           # Cache de datos
    └── validators.py      # Validacion
```

## Indicadores Tecnicos

### Momentum

```python
from src.indicators.technical import momentum

# RSI (Relative Strength Index)
rsi = momentum.rsi(close, period=14)

# MACD
macd_result = momentum.macd(close, fast=12, slow=26, signal=9)
# macd_result.macd, macd_result.signal, macd_result.histogram

# Stochastic
stoch = momentum.stochastic(high, low, close, k_period=14, d_period=3)
# stoch.k, stoch.d

# KDJ (variante asiatica del Stochastic)
kdj = momentum.kdj(high, low, close, period=9)
# kdj.k, kdj.d, kdj.j

# CCI (Commodity Channel Index)
cci = momentum.cci(high, low, close, period=20)

# Williams %R
williams_r = momentum.williams_r(high, low, close, period=14)

# ROC (Rate of Change)
roc = momentum.roc(close, period=10)

# Momentum
mom = momentum.momentum(close, period=10)
```

### Trend

```python
from src.indicators.technical import trend

# Medias Moviles
sma = trend.sma(close, period=20)
ema = trend.ema(close, period=20)
dema = trend.dema(close, period=20)  # Double EMA
tema = trend.tema(close, period=20)  # Triple EMA
wma = trend.wma(close, period=20)    # Weighted MA

# ADX (Average Directional Index)
adx_result = trend.adx(high, low, close, period=14)
# adx_result.adx, adx_result.plus_di, adx_result.minus_di

# TRIX
trix = trend.trix(close, period=15)

# Supertrend
supertrend = trend.supertrend(high, low, close, period=10, multiplier=3)
# supertrend.trend, supertrend.direction

# Parabolic SAR
psar = trend.parabolic_sar(high, low, af_start=0.02, af_max=0.2)

# Aroon
aroon = trend.aroon(high, low, period=25)
# aroon.up, aroon.down, aroon.oscillator
```

### Volatilidad

```python
from src.indicators.technical import volatility

# Bollinger Bands
bb = volatility.bollinger_bands(close, period=20, std_dev=2.0)
# bb.upper, bb.middle, bb.lower, bb.bandwidth, bb.percent_b

# ATR (Average True Range)
atr = volatility.atr(high, low, close, period=14)

# Keltner Channels
keltner = volatility.keltner_channels(high, low, close, period=20, multiplier=2.0)
# keltner.upper, keltner.middle, keltner.lower

# Donchian Channels
donchian = volatility.donchian_channels(high, low, period=20)
# donchian.upper, donchian.middle, donchian.lower

# Historical Volatility
hvol = volatility.historical_volatility(close, period=20, annualize=True)

# Chaikin Volatility
chaikin_vol = volatility.chaikin_volatility(high, low, period=10)
```

### Volumen

```python
from src.indicators.technical import volume

# OBV (On-Balance Volume)
obv = volume.obv(close, vol)

# VWAP (Volume Weighted Average Price)
vwap = volume.vwap(high, low, close, vol)

# MFI (Money Flow Index)
mfi = volume.mfi(high, low, close, vol, period=14)

# ADL (Accumulation/Distribution Line)
adl = volume.adl(high, low, close, vol)

# Chaikin Money Flow
cmf = volume.cmf(high, low, close, vol, period=20)

# Force Index
force = volume.force_index(close, vol, period=13)

# Volume Profile (agrupado por precio)
profile = volume.volume_profile(close, vol, bins=20)
```

### Pivot Points

```python
from src.indicators.technical import pivots

# Pivot Points clasicos
pivot = pivots.pivot_points(high, low, close)
# pivot.pivot, pivot.r1, pivot.r2, pivot.r3, pivot.s1, pivot.s2, pivot.s3

# Fibonacci Pivot Points
fib_pivot = pivots.fibonacci_pivots(high, low, close)

# Woodie Pivot Points
woodie = pivots.woodie_pivots(high, low, close, open_price)

# Camarilla Pivot Points
camarilla = pivots.camarilla_pivots(high, low, close)

# Soporte y Resistencia automaticos
sr_levels = pivots.auto_support_resistance(high, low, close, lookback=100)
```

### Ichimoku Cloud

```python
from src.indicators.technical.ichimoku import IchimokuCloud

ichimoku = IchimokuCloud(
    tenkan_period=9,
    kijun_period=26,
    senkou_b_period=52,
    displacement=26
)

result = ichimoku.calculate(high, low, close)

# Componentes
tenkan = result.tenkan_sen      # Conversion Line
kijun = result.kijun_sen        # Base Line
senkou_a = result.senkou_span_a # Leading Span A
senkou_b = result.senkou_span_b # Leading Span B
chikou = result.chikou_span     # Lagging Span

# Senales
cloud_color = result.cloud_color  # 'green' o 'red'
price_vs_cloud = result.price_position  # 'above', 'below', 'inside'
tk_cross = result.tk_cross_signal  # 'bullish', 'bearish', None
```

## Indicadores Fundamentales

### CoinGecko

```python
from src.indicators.fundamental.coingecko import CoinGeckoClient

client = CoinGeckoClient(api_key="tu_api_key")  # Opcional

# Market data
market_data = await client.get_market_data("bitcoin")
print(f"Market Cap: ${market_data['market_cap']:,.0f}")
print(f"24h Volume: ${market_data['volume_24h']:,.0f}")
print(f"Dominance: {market_data['market_cap_dominance']:.2f}%")

# Global data
global_data = await client.get_global_data()
print(f"Total Market Cap: ${global_data['total_market_cap']:,.0f}")
print(f"BTC Dominance: {global_data['btc_dominance']:.2f}%")

# Trending
trending = await client.get_trending()
```

### Glassnode

```python
from src.indicators.fundamental.glassnode import GlassnodeClient

client = GlassnodeClient(api_key="tu_api_key")

# On-chain metrics
sopr = await client.get_sopr("BTC")  # Spent Output Profit Ratio
nupl = await client.get_nupl("BTC")  # Net Unrealized Profit/Loss
mvrv = await client.get_mvrv("BTC")  # Market Value to Realized Value

# Supply metrics
supply = await client.get_supply_metrics("BTC")
print(f"Circulating: {supply['circulating']:,.0f}")
print(f"Illiquid: {supply['illiquid']:,.0f}")

# Address metrics
addresses = await client.get_address_metrics("BTC")
print(f"Active Addresses: {addresses['active']:,.0f}")
print(f"New Addresses: {addresses['new']:,.0f}")
```

### DeFi Llama

```python
from src.indicators.fundamental.defillama import DefiLlamaClient

client = DefiLlamaClient()

# TVL total
tvl = await client.get_total_tvl()
print(f"Total TVL: ${tvl:,.0f}")

# TVL por protocolo
protocol_tvl = await client.get_protocol_tvl("aave")

# TVL por chain
chain_tvl = await client.get_chain_tvl("ethereum")

# Protocolos
protocols = await client.get_protocols()
```

### Fear & Greed Index

```python
from src.indicators.fundamental.alternative_me import AlternativeMeClient

client = AlternativeMeClient()

# Indice actual
fng = await client.get_fear_greed()
print(f"Value: {fng['value']}")
print(f"Classification: {fng['classification']}")
# 0-25: Extreme Fear, 26-46: Fear, 47-54: Neutral
# 55-75: Greed, 76-100: Extreme Greed

# Historico
history = await client.get_fear_greed_history(limit=30)
```

### Santiment

```python
from src.indicators.fundamental.santiment import SantimentClient

client = SantimentClient(api_key="tu_api_key")

# Social volume
social = await client.get_social_volume("bitcoin")
print(f"Social Volume: {social['volume']}")
print(f"Sentiment: {social['sentiment']}")

# Developer activity
dev = await client.get_dev_activity("ethereum")
print(f"Dev Activity: {dev['activity']}")

# Exchange flow
flow = await client.get_exchange_flow("bitcoin")
print(f"Inflow: {flow['inflow']}")
print(f"Outflow: {flow['outflow']}")
```

## Clases Base

### IndicatorResult

```python
from src.indicators.base import IndicatorResult

# Los indicadores multi-valor retornan IndicatorResult
class BollingerBandsResult(IndicatorResult):
    upper: pd.Series
    middle: pd.Series
    lower: pd.Series
    bandwidth: pd.Series
    percent_b: pd.Series
```

## Cache y Optimizacion

```python
from src.indicators.utils.cache import IndicatorCache

# Cache automatico para indicadores costosos
cache = IndicatorCache(max_size=1000, ttl=3600)

@cache.cached
def expensive_indicator(data, params):
    # Calculo costoso
    return result
```

## Uso con Strategy

```python
from src.strategy import TradingStrategy
from src.indicators.technical import momentum, trend, volatility

class MiEstrategia(TradingStrategy):
    def calculate_indicators(self):
        # Momentum
        self.data['rsi'] = momentum.rsi(self.data['close'], 14)
        macd_result = momentum.macd(self.data['close'])
        self.data['macd'] = macd_result.macd
        self.data['macd_signal'] = macd_result.signal

        # Trend
        self.data['sma_20'] = trend.sma(self.data['close'], 20)
        self.data['sma_50'] = trend.sma(self.data['close'], 50)
        adx_result = trend.adx(
            self.data['high'],
            self.data['low'],
            self.data['close']
        )
        self.data['adx'] = adx_result.adx

        # Volatility
        bb = volatility.bollinger_bands(self.data['close'])
        self.data['bb_upper'] = bb.upper
        self.data['bb_lower'] = bb.lower

    def generate_signals(self):
        signals = []
        for i in range(len(self.data)):
            row = self.data.iloc[i]

            # Condiciones de compra
            if (row['rsi'] < 30 and
                row['macd'] > row['macd_signal'] and
                row['close'] < row['bb_lower'] and
                row['adx'] > 25):
                signals.append(TradeSignal(
                    timestamp=self.data.index[i],
                    signal='BUY',
                    confidence=0.9
                ))
            # ... mas condiciones
        return signals
```

## Tests

```bash
pytest tests/indicators/ -v
```
