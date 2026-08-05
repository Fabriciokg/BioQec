"""Extração causal das oito características especificadas no artigo BioQEC.

Nenhuma função deste módulo acessa observações posteriores ao índice ``t``.
As estatísticas retornam também máscaras de disponibilidade para componentes
que dependem de janela mínima, leitura analógica ou distribuição preditiva.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


FEATURE_NAMES = (
    "r_X",
    "r_Z",
    "r_measurement_pair",
    "r_leak",
    "c_spatial",
    "c_temporal",
    "u_analog",
    "u_model",
)


@dataclass(frozen=True, slots=True)
class CausalFeatures:
    values: np.ndarray
    available: np.ndarray
    start: int
    stop: int
    names: tuple[str, ...] = FEATURE_NAMES

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=float)
        available = np.asarray(self.available, dtype=bool)
        if values.shape != (8,) or available.shape != (8,):
            raise ValueError("values e available devem ter shape (8,)")
        values.setflags(write=False)
        available.setflags(write=False)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "available", available)

    def as_dict(self) -> dict[str, float | None]:
        return {
            name: float(value) if present else None
            for name, value, present in zip(self.names, self.values, self.available)
        }


def _validate_binary_matrix(detections: np.ndarray) -> np.ndarray:
    arr = np.asarray(detections)
    if arr.ndim != 2 or arr.shape[0] < 1 or arr.shape[1] < 1:
        raise ValueError("detections deve ter shape (tempo, verificadores)")
    if not np.all((arr == 0) | (arr == 1)):
        raise ValueError("detections deve ser binária")
    return arr.astype(float, copy=False)


def _normalized_entropy(probabilities: np.ndarray, eps: float) -> float:
    p = np.asarray(probabilities, dtype=float).ravel()
    if p.size < 2 or np.any(p < 0) or not np.isfinite(p).all():
        raise ValueError("predictive_probs deve ser uma distribuição finita")
    total = float(np.sum(p))
    if total <= 0:
        raise ValueError("predictive_probs deve ter soma positiva")
    p = p / total
    return float(-np.sum(p * np.log(p + eps)) / np.log(p.size))


def extract_causal_features(
    detections: np.ndarray,
    check_types: Sequence[str],
    adjacency: Iterable[tuple[int, int]],
    *,
    t: int | None = None,
    window: int = 16,
    leakage: np.ndarray | None = None,
    analog_prob_one: np.ndarray | None = None,
    predictive_probs: np.ndarray | None = None,
    eps: float = 1e-8,
) -> CausalFeatures:
    """Calcula o vetor bruto de oito componentes usando somente dados até ``t``.

    ``detections`` deve estar organizado como ``(tempo, verificadores)``.
    ``analog_prob_one`` contém probabilidades calibradas de o bit medido ser 1,
    não amplitudes analógicas brutas. ``predictive_probs`` pode ter shape
    ``(tempo, classes)`` ou ``(classes,)`` e deve corresponder à previsão antes
    da prova de leitura e antes do rótulo final.
    """

    d = _validate_binary_matrix(detections)
    n_time, n_checks = d.shape
    if len(check_types) != n_checks:
        raise ValueError("check_types deve ter um rótulo por verificador")
    if window < 1:
        raise ValueError("window deve ser positivo")
    if eps <= 0:
        raise ValueError("eps deve ser positivo")

    t = n_time - 1 if t is None else int(t)
    if not 0 <= t < n_time:
        raise ValueError("t fora do intervalo observado")
    start = max(0, t - window + 1)
    stop = t + 1
    x = d[start:stop]
    width = x.shape[0]

    types = np.asarray([str(v).upper() for v in check_types])
    ix = np.flatnonzero(types == "X")
    iz = np.flatnonzero(types == "Z")
    if ix.size == 0 or iz.size == 0:
        raise ValueError("check_types deve conter verificadores X e Z")

    values = np.zeros(8, dtype=float)
    available = np.ones(8, dtype=bool)
    values[0] = float(np.mean(x[:, ix]))
    values[1] = float(np.mean(x[:, iz]))

    if width >= 2:
        values[2] = float(np.mean(x[1:] * x[:-1]))
    else:
        available[2] = False
        values[2] = np.nan

    if leakage is None:
        available[3] = False
        values[3] = np.nan
    else:
        leak = np.asarray(leakage, dtype=float)
        if leak.shape != d.shape or np.any((leak < 0) | (leak > 1)):
            raise ValueError("leakage deve ter o mesmo shape e valores em [0, 1]")
        values[3] = float(np.mean(leak[start:stop]))

    means = np.mean(x, axis=0)
    spatial_terms: list[float] = []
    for i, j in adjacency:
        if not (0 <= i < n_checks and 0 <= j < n_checks) or i == j:
            raise ValueError("adjacency contém uma aresta inválida")
        cov = float(np.mean((x[:, i] - means[i]) * (x[:, j] - means[j])))
        denom = float(np.sqrt((means[i] * (1 - means[i]) + eps) * (means[j] * (1 - means[j]) + eps)))
        spatial_terms.append(cov / denom)
    if spatial_terms:
        values[4] = float(np.mean(spatial_terms))
    else:
        available[4] = False
        values[4] = np.nan

    if width >= 2:
        temporal_terms = []
        for i in range(n_checks):
            cov = float(np.mean((x[1:, i] - means[i]) * (x[:-1, i] - means[i])))
            temporal_terms.append(cov / (means[i] * (1 - means[i]) + eps))
        values[5] = float(np.mean(temporal_terms))
    else:
        available[5] = False
        values[5] = np.nan

    if analog_prob_one is None:
        available[6] = False
        values[6] = np.nan
    else:
        analog = np.asarray(analog_prob_one, dtype=float)
        if analog.shape != d.shape or np.any((analog < 0) | (analog > 1)):
            raise ValueError("analog_prob_one deve ter o mesmo shape e valores em [0, 1]")
        ambiguity = 1.0 - 2.0 * np.abs(analog[start:stop] - 0.5)
        values[6] = float(np.mean(ambiguity))

    if predictive_probs is None:
        available[7] = False
        values[7] = np.nan
    else:
        probs = np.asarray(predictive_probs, dtype=float)
        current = probs if probs.ndim == 1 else probs[t]
        values[7] = _normalized_entropy(current, eps)

    return CausalFeatures(values=values, available=available, start=start, stop=stop)


def robust_standardize(
    features: CausalFeatures,
    location: np.ndarray,
    scale: np.ndarray,
    eps: float = 1e-8,
) -> CausalFeatures:
    """Padroniza componentes disponíveis com parâmetros congelados de treino."""

    location = np.asarray(location, dtype=float)
    scale = np.asarray(scale, dtype=float)
    if location.shape != (8,) or scale.shape != (8,) or np.any(scale <= 0):
        raise ValueError("location e scale devem ter shape (8,), com scale positiva")
    values = features.values.copy()
    mask = features.available
    values[mask] = (values[mask] - location[mask]) / (scale[mask] + eps)
    return CausalFeatures(values=values, available=mask, start=features.start, stop=features.stop)
