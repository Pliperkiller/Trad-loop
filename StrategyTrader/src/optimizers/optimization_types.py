"""
Tipos y clases de datos compartidas para el módulo de optimización
"""

import pandas as pd
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class ParameterSpace:
    """Define el espacio de búsqueda para un parámetro"""
    name: str
    param_type: str  # 'int', 'float', 'categorical'
    low: Optional[float] = None
    high: Optional[float] = None
    values: Optional[List[Any]] = None
    step: Optional[float] = None


@dataclass
class OptimizationResult:
    """Resultado de la optimización"""
    best_params: Dict[str, Any]
    best_score: float
    all_results: pd.DataFrame
    optimization_time: float
    method: str
    iterations: int

    def print_summary(self):
        """Imprime resumen de la optimización"""
        print(f"\nMétodo: {self.method}")
        print(f"Mejor Score: {self.best_score:.4f}")
        print(f"Mejores Parámetros: {self.best_params}")
        print(f"Iteraciones: {self.iterations}")
        print(f"Tiempo: {self.optimization_time:.2f}s")
