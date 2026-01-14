"""
Utilidades compartidas para los optimizadores.

Este módulo contiene funciones auxiliares usadas por múltiples
algoritmos de optimización (bayesian, genetic, etc.).
"""

import numpy as np
from typing import Optional, List, Dict, Any

from .optimization_types import ParameterSpace


def snap_to_step(
    value: float,
    low: float,
    high: float,
    step: Optional[float],
    param_type: str
) -> float:
    """
    Ajusta un valor continuo al grid definido por step.

    Útil para convertir valores continuos del espacio de búsqueda
    a valores discretos que respetan el step definido en el parámetro.

    Args:
        value: Valor continuo a ajustar
        low: Límite inferior del rango
        high: Límite superior del rango
        step: Tamaño del paso (None o <= 0 significa sin restricción)
        param_type: Tipo del parámetro ('int' o 'float')

    Returns:
        Valor ajustado al step más cercano dentro de [low, high]

    Examples:
        >>> snap_to_step(7.3, 0, 10, 2, 'int')
        8
        >>> snap_to_step(7.3, 0, 10, 2, 'float')
        8.0
        >>> snap_to_step(7.3, 0, 10, None, 'float')
        7.3
    """
    if step is None or step <= 0:
        if param_type == 'int':
            return int(round(np.clip(value, low, high)))
        return float(np.clip(value, low, high))

    # Calcular el número de pasos desde low
    steps_from_low = round((value - low) / step)

    # Calcular el valor ajustado
    snapped = low + (steps_from_low * step)

    # Asegurar que está dentro de los límites
    snapped = np.clip(snapped, low, high)

    if param_type == 'int':
        return int(round(snapped))
    return round(float(snapped), 10)


def convert_point_to_params(
    point: List,
    parameter_space: List[ParameterSpace]
) -> Dict[str, Any]:
    """
    Convierte un punto del espacio de búsqueda a un diccionario de parámetros.

    Aplica snap_to_step y conversiones de tipo apropiadas según
    la definición de cada parámetro.

    Args:
        point: Lista de valores continuos del espacio de búsqueda
        parameter_space: Lista de definiciones de parámetros

    Returns:
        Diccionario {nombre_param: valor_convertido}

    Example:
        >>> params = [
        ...     ParameterSpace('period', 'int', low=5, high=50, step=5),
        ...     ParameterSpace('threshold', 'float', low=0.0, high=1.0, step=0.1),
        ... ]
        >>> convert_point_to_params([12.3, 0.47], params)
        {'period': 10, 'threshold': 0.5}
    """
    params = {}
    for i, param in enumerate(parameter_space):
        value = point[i]

        if param.param_type == 'categorical' and param.values:
            # Para categóricos, usar el índice redondeado
            idx = int(np.clip(round(value), 0, len(param.values) - 1))
            params[param.name] = param.values[idx]
        elif param.param_type == 'int':
            low = param.low if param.low is not None else 0
            high = param.high if param.high is not None else 100
            params[param.name] = snap_to_step(value, low, high, param.step, 'int')
        else:
            # float o cualquier otro tipo numérico
            low = param.low if param.low is not None else 0
            high = param.high if param.high is not None else 1
            params[param.name] = snap_to_step(value, low, high, param.step, 'float')

    return params


def clip_to_bounds(
    value: float,
    low: Optional[float],
    high: Optional[float]
) -> float:
    """
    Recorta un valor a los límites especificados.

    Args:
        value: Valor a recortar
        low: Límite inferior (None = sin límite)
        high: Límite superior (None = sin límite)

    Returns:
        Valor recortado
    """
    if low is not None and value < low:
        return low
    if high is not None and value > high:
        return high
    return value
