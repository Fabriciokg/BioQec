"""Primitivas auditáveis para o baseline MWPM com recalibração periódica.

A estimação das probabilidades pode ser fornecida por um estimador de
correlações ou de janela deslizante. Este módulo fixa a agenda, a contração e a
conversão para pesos sem permitir ajuste no conjunto de teste.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class PeriodicRecalibrationConfig:
    period: int
    calibration_window: int
    shrinkage_to_initial: float = 0.1
    eps: float = 1e-9

    def __post_init__(self) -> None:
        if self.period < 1 or self.calibration_window < 1:
            raise ValueError("period e calibration_window devem ser positivos")
        if not 0 <= self.shrinkage_to_initial <= 1:
            raise ValueError("shrinkage_to_initial deve pertencer a [0, 1]")
        if not 0 < self.eps < 0.5:
            raise ValueError("eps deve pertencer a (0, 0.5)")


def should_recalibrate(cycle: int, config: PeriodicRecalibrationConfig) -> bool:
    if cycle < 0:
        raise ValueError("cycle deve ser não negativo")
    return cycle > 0 and cycle % config.period == 0


def shrink_probabilities(
    estimated: np.ndarray,
    initial: np.ndarray,
    config: PeriodicRecalibrationConfig,
) -> np.ndarray:
    estimated = np.asarray(estimated, dtype=float)
    initial = np.asarray(initial, dtype=float)
    if estimated.shape != initial.shape:
        raise ValueError("estimated e initial devem ter o mesmo shape")
    if np.any(~np.isfinite(estimated)) or np.any(~np.isfinite(initial)):
        raise ValueError("probabilidades devem ser finitas")
    gamma = config.shrinkage_to_initial
    result = (1.0 - gamma) * estimated + gamma * initial
    return np.clip(result, config.eps, 1.0 - config.eps)


def probabilities_to_weights(probabilities: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    p = np.clip(np.asarray(probabilities, dtype=float), eps, 1.0 - eps)
    return np.log((1.0 - p) / p)
