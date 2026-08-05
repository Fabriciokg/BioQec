"""Monitor causal simples de mudança baseado em densidade de detecções e CUSUM."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class CUSUMConfig:
    reference_mean: float
    reference_std: float
    allowance: float = 0.5
    threshold: float = 5.0

    def __post_init__(self) -> None:
        if self.reference_std <= 0.0:
            raise ValueError("reference_std deve ser positivo")
        if self.allowance < 0.0:
            raise ValueError("allowance deve ser não negativa")
        if self.threshold <= 0.0:
            raise ValueError("threshold deve ser positivo")


@dataclass(frozen=True, slots=True)
class CUSUMResult:
    scores: np.ndarray
    alarms: np.ndarray


def syndrome_density(detections: np.ndarray) -> float:
    detections = np.asarray(detections, dtype=np.bool_)
    if detections.ndim != 2 or detections.size == 0:
        raise ValueError("detections deve ser uma matriz 2D não vazia")
    return float(np.mean(detections))


def fit_reference(values: np.ndarray, min_std: float = 1e-8) -> tuple[float, float]:
    values = np.asarray(values, dtype=float).ravel()
    if values.size < 2:
        raise ValueError("são necessários ao menos dois valores de referência")
    mean = float(np.mean(values))
    std = max(float(np.std(values, ddof=1)), min_std)
    return mean, std


def upper_cusum(values: np.ndarray, cfg: CUSUMConfig, reset_after_alarm: bool = True) -> CUSUMResult:
    """Executa CUSUM unilateral superior em ordem temporal."""

    values = np.asarray(values, dtype=float).ravel()
    scores = np.zeros(values.size, dtype=float)
    alarms = np.zeros(values.size, dtype=bool)
    score = 0.0
    for index, value in enumerate(values):
        z = (value - cfg.reference_mean) / cfg.reference_std
        score = max(0.0, score + z - cfg.allowance)
        scores[index] = score
        if score >= cfg.threshold:
            alarms[index] = True
            if reset_after_alarm:
                score = 0.0
    return CUSUMResult(scores=scores, alarms=alarms)
