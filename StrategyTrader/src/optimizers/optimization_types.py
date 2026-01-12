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
        print(f"\nMetodo: {self.method}")
        print(f"Mejor Score: {self.best_score:.4f}")
        print(f"Mejores Parametros: {self.best_params}")
        print(f"Iteraciones: {self.iterations}")
        print(f"Tiempo: {self.optimization_time:.2f}s")


@dataclass
class WalkForwardResult:
    """Resultado de Walk Forward Optimization"""
    splits_results: List[Dict[str, Any]]  # Resultados por cada split
    aggregated_metrics: Dict[str, float]  # Metricas agregadas
    out_of_sample_equity: List[float]  # Equity curve combinada OOS
    robustness_score: float  # Score de robustez (0-1)
    optimization_time: float
    n_splits: int
    train_size: float
    optimization_method: str
    # Additional attributes for validation integration
    best_params: Optional[Dict[str, Any]] = None  # Best params across all splits
    consistency_ratio: float = 0.0  # Ratio of positive OOS splits
    avg_is_score: float = 0.0  # Average in-sample score
    avg_oos_score: float = 0.0  # Average out-of-sample score
    parameter_stability: Optional[Dict[str, Any]] = None  # Parameter stability analysis

    def print_summary(self):
        """Imprime resumen del Walk Forward"""
        print("\n" + "="*70)
        print("RESUMEN WALK FORWARD OPTIMIZATION")
        print("="*70)
        print(f"\nConfiguracion:")
        print(f"  Splits: {self.n_splits}")
        print(f"  Train Size: {self.train_size*100:.0f}%")
        print(f"  Metodo Optimizacion: {self.optimization_method}")
        print(f"  Tiempo Total: {self.optimization_time:.2f}s")

        print(f"\n[METRICAS AGREGADAS OUT-OF-SAMPLE]")
        for metric, value in self.aggregated_metrics.items():
            if 'pct' in metric or 'rate' in metric:
                print(f"  {metric}: {value:.2f}%")
            elif 'ratio' in metric:
                print(f"  {metric}: {value:.2f}")
            else:
                print(f"  {metric}: {value:.4f}")

        print(f"\n[ROBUSTEZ]")
        print(f"  Score de Robustez: {self.robustness_score:.2f}")
        if self.robustness_score >= 0.7:
            print("  Veredicto: ESTRATEGIA ROBUSTA")
        elif self.robustness_score >= 0.5:
            print("  Veredicto: ESTRATEGIA ACEPTABLE")
        else:
            print("  Veredicto: ESTRATEGIA FRAGIL - Posible overfitting")

        print(f"\n[RESULTADOS POR SPLIT]")
        for i, split in enumerate(self.splits_results):
            status = "OK" if split.get('oos_score', 0) > 0 else "FAIL"
            print(f"  Split {i+1}: Train Score={split.get('train_score', 0):.4f}, "
                  f"OOS Score={split.get('oos_score', 0):.4f} [{status}]")

        print("="*70)
