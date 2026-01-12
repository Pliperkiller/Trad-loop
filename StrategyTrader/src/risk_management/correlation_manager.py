"""
Gestion de correlacion entre posiciones.

Analiza y controla la correlacion entre activos para
evitar concentracion de riesgo.
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
import numpy as np

from .models import CorrelationData, RiskLimit, DrawdownAction
from .config import CorrelationConfig


@dataclass
class AssetReturns:
    """Retornos de un activo"""
    symbol: str
    returns: List[float]
    last_updated: datetime = field(default_factory=datetime.now)


class CorrelationManager:
    """
    Gestor de correlacion entre activos.

    Calcula y monitorea la correlacion entre pares de activos
    para evitar concentracion de riesgo.
    """

    def __init__(self, config: CorrelationConfig):
        self.config = config
        self.returns_data: Dict[str, AssetReturns] = {}
        self.correlation_matrix: Dict[Tuple[str, str], CorrelationData] = {}
        self.last_matrix_update: Optional[datetime] = None

    def update_returns(self, symbol: str, returns: List[float]):
        """Actualiza los retornos de un activo"""
        self.returns_data[symbol] = AssetReturns(
            symbol=symbol,
            returns=returns[-self.config.lookback_days:],
            last_updated=datetime.now(),
        )

    def add_return(self, symbol: str, daily_return: float):
        """Agrega un retorno diario para un activo"""
        if symbol not in self.returns_data:
            self.returns_data[symbol] = AssetReturns(symbol=symbol, returns=[])

        data = self.returns_data[symbol]
        data.returns.append(daily_return)

        # Mantener solo los ultimos N dias
        if len(data.returns) > self.config.lookback_days:
            data.returns = data.returns[-self.config.lookback_days:]

        data.last_updated = datetime.now()

    def calculate_correlation(self, symbol_a: str, symbol_b: str) -> Optional[float]:
        """
        Calcula la correlacion entre dos activos.

        Returns:
            float: Correlacion de Pearson (-1 a 1), o None si no hay datos suficientes
        """
        if symbol_a not in self.returns_data or symbol_b not in self.returns_data:
            return None

        returns_a = self.returns_data[symbol_a].returns
        returns_b = self.returns_data[symbol_b].returns

        # Necesitamos al menos 10 datos
        min_len = min(len(returns_a), len(returns_b))
        if min_len < 10:
            return None

        # Usar los ultimos N puntos comunes
        r_a = np.array(returns_a[-min_len:])
        r_b = np.array(returns_b[-min_len:])

        # Calcular correlacion de Pearson
        if np.std(r_a) == 0 or np.std(r_b) == 0:
            return 0.0

        correlation = np.corrcoef(r_a, r_b)[0, 1]

        # Manejar NaN
        if np.isnan(correlation):
            return 0.0

        return float(correlation)

    def update_correlation_matrix(self, symbols: Optional[List[str]] = None):
        """
        Actualiza la matriz de correlacion para todos los simbolos.

        Args:
            symbols: Lista de simbolos a calcular. Si es None, usa todos los disponibles.
        """
        if symbols is None:
            symbols = list(self.returns_data.keys())

        for i, sym_a in enumerate(symbols):
            for sym_b in symbols[i + 1:]:
                correlation = self.calculate_correlation(sym_a, sym_b)
                if correlation is not None:
                    key = (sym_a, sym_b)
                    self.correlation_matrix[key] = CorrelationData(
                        symbol_a=sym_a,
                        symbol_b=sym_b,
                        correlation=correlation,
                        period_days=self.config.lookback_days,
                    )

        self.last_matrix_update = datetime.now()

    def get_correlation(self, symbol_a: str, symbol_b: str) -> Optional[float]:
        """Obtiene la correlacion almacenada entre dos activos"""
        # Ordenar para encontrar la key correcta
        key1 = (symbol_a, symbol_b)
        key2 = (symbol_b, symbol_a)

        if key1 in self.correlation_matrix:
            return self.correlation_matrix[key1].correlation
        elif key2 in self.correlation_matrix:
            return self.correlation_matrix[key2].correlation

        return None

    def get_highly_correlated_pairs(self) -> List[CorrelationData]:
        """Obtiene los pares con alta correlacion"""
        threshold = self.config.high_correlation_threshold
        return [
            data for data in self.correlation_matrix.values()
            if abs(data.correlation) >= threshold
        ]

    def get_diversifying_pairs(self) -> List[CorrelationData]:
        """Obtiene los pares con correlacion negativa (diversifican)"""
        return [
            data for data in self.correlation_matrix.values()
            if data.is_diversifying
        ]

    def get_portfolio_correlation(self, positions: List[Tuple[str, float]]) -> float:
        """
        Calcula la correlacion promedio ponderada del portfolio.

        Args:
            positions: Lista de (simbolo, peso) donde peso es la proporcion del portfolio

        Returns:
            float: Correlacion promedio ponderada
        """
        if len(positions) < 2:
            return 0.0

        total_weight = 0.0
        weighted_correlation = 0.0

        for i, (sym_a, weight_a) in enumerate(positions):
            for sym_b, weight_b in positions[i + 1:]:
                corr = self.get_correlation(sym_a, sym_b)
                if corr is not None:
                    pair_weight = weight_a * weight_b
                    weighted_correlation += corr * pair_weight
                    total_weight += pair_weight

        if total_weight == 0:
            return 0.0

        return weighted_correlation / total_weight

    def check_correlation_limits(
        self,
        current_positions: List[str],
        new_symbol: str,
    ) -> Tuple[bool, List[str]]:
        """
        Verifica si agregar un nuevo activo excede los limites de correlacion.

        Args:
            current_positions: Simbolos de posiciones actuales
            new_symbol: Simbolo que se quiere agregar

        Returns:
            Tuple[bool, List[str]]: (permitido, razones si no)
        """
        if not self.config.enabled:
            return True, []

        reasons = []

        # Contar posiciones altamente correlacionadas
        high_corr_count = 0
        threshold = self.config.high_correlation_threshold

        for pos_symbol in current_positions:
            corr = self.get_correlation(pos_symbol, new_symbol)
            if corr is not None and abs(corr) >= threshold:
                high_corr_count += 1
                if self.config.correlation_penalty:
                    reasons.append(
                        f"Alta correlacion con {pos_symbol}: {corr:.2f}"
                    )

        # Verificar limite de posiciones correlacionadas
        if high_corr_count >= self.config.max_correlated_positions:
            return False, [
                f"Excede maximo de posiciones correlacionadas: "
                f"{high_corr_count} >= {self.config.max_correlated_positions}"
            ]

        return True, reasons

    def get_correlation_penalty(
        self,
        current_positions: List[str],
        new_symbol: str,
    ) -> float:
        """
        Calcula el factor de penalizacion por correlacion.

        Returns:
            float: Multiplicador (1.0 = sin penalizacion, 0.5 = 50% reduccion)
        """
        if not self.config.enabled or not self.config.correlation_penalty:
            return 1.0

        max_correlation = 0.0
        threshold = self.config.high_correlation_threshold

        for pos_symbol in current_positions:
            corr = self.get_correlation(pos_symbol, new_symbol)
            if corr is not None:
                max_correlation = max(max_correlation, abs(corr))

        if max_correlation >= threshold:
            # Penalizacion proporcional a la correlacion
            penalty_strength = (max_correlation - threshold) / (1 - threshold)
            return 1.0 - (penalty_strength * (1 - self.config.penalty_factor))

        return 1.0

    def get_max_correlation_pair(self) -> Optional[Tuple[str, str, float]]:
        """Obtiene el par con mayor correlacion"""
        if not self.correlation_matrix:
            return None

        max_corr_data = max(
            self.correlation_matrix.values(),
            key=lambda x: abs(x.correlation)
        )

        return (
            max_corr_data.symbol_a,
            max_corr_data.symbol_b,
            max_corr_data.correlation,
        )

    def get_average_correlation(self) -> float:
        """Obtiene la correlacion promedio de todos los pares"""
        if not self.correlation_matrix:
            return 0.0

        correlations = [abs(data.correlation) for data in self.correlation_matrix.values()]
        return sum(correlations) / len(correlations)

    def get_correlation_report(self) -> Dict:
        """Genera un reporte de correlacion"""
        return {
            "total_pairs": len(self.correlation_matrix),
            "average_correlation": self.get_average_correlation(),
            "max_correlation_pair": self.get_max_correlation_pair(),
            "highly_correlated_count": len(self.get_highly_correlated_pairs()),
            "diversifying_count": len(self.get_diversifying_pairs()),
            "last_update": self.last_matrix_update.isoformat() if self.last_matrix_update else None,
        }

    def needs_update(self) -> bool:
        """Verifica si la matriz necesita actualizarse"""
        if self.last_matrix_update is None:
            return True

        hours_since_update = (datetime.now() - self.last_matrix_update).total_seconds() / 3600
        return hours_since_update >= self.config.update_frequency_hours

    def check_limits(self, current_positions: List[str]) -> List[RiskLimit]:
        """Verifica limites de correlacion"""
        breached = []

        # Verificar pares altamente correlacionados
        high_corr_pairs = self.get_highly_correlated_pairs()

        # Filtrar solo los pares donde ambos activos tienen posicion
        active_high_corr = [
            pair for pair in high_corr_pairs
            if pair.symbol_a in current_positions and pair.symbol_b in current_positions
        ]

        if len(active_high_corr) > 0:
            breached.append(RiskLimit(
                name="High Correlation Pairs",
                limit_type="correlation",
                threshold=self.config.high_correlation_threshold,
                current_value=len(active_high_corr),
                action=DrawdownAction.REDUCE_SIZE,
                is_breached=True,
            ))

        return breached

    def reset(self):
        """Reinicia todos los datos"""
        self.returns_data.clear()
        self.correlation_matrix.clear()
        self.last_matrix_update = None
