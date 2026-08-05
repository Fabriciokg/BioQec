"""Métricas estatísticas básicas para os experimentos BioQEC."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from statistics import NormalDist

import numpy as np


@dataclass(frozen=True, slots=True)
class BinomialEstimate:
    events: int
    trials: int
    rate: float
    ci_low: float
    ci_high: float


def wilson_interval(events: int, trials: int, confidence: float = 0.95) -> BinomialEstimate:
    if trials < 1:
        raise ValueError("trials deve ser maior ou igual a 1")
    if events < 0 or events > trials:
        raise ValueError("events deve pertencer a 0,...,trials")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence deve pertencer a (0, 1)")

    p_hat = events / trials
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    denominator = 1.0 + z**2 / trials
    center = (p_hat + z**2 / (2.0 * trials)) / denominator
    half = z * sqrt(
        p_hat * (1.0 - p_hat) / trials + z**2 / (4.0 * trials**2)
    ) / denominator
    ci_low = 0.0 if events == 0 else max(0.0, center - half)
    ci_high = 1.0 if events == trials else min(1.0, center + half)
    return BinomialEstimate(
        events=events,
        trials=trials,
        rate=p_hat,
        ci_low=ci_low,
        ci_high=ci_high,
    )


def summarize_failures(failures: np.ndarray, confidence: float = 0.95) -> BinomialEstimate:
    failures = np.asarray(failures, dtype=bool).ravel()
    if failures.size == 0:
        raise ValueError("failures não pode ser vazio")
    return wilson_interval(int(np.sum(failures)), int(failures.size), confidence)


def effective_error_per_round(logical_error_rate: float, rounds: int) -> float:
    if not 0.0 <= logical_error_rate <= 1.0:
        raise ValueError("logical_error_rate deve pertencer a [0, 1]")
    if rounds < 1:
        raise ValueError("rounds deve ser maior ou igual a 1")
    return float(1.0 - (1.0 - logical_error_rate) ** (1.0 / rounds))


def brier_score(probabilities: np.ndarray, outcomes: np.ndarray) -> float:
    probabilities = np.asarray(probabilities, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    if probabilities.shape != outcomes.shape:
        raise ValueError("probabilities e outcomes devem ter o mesmo shape")
    if probabilities.size == 0:
        raise ValueError("os vetores não podem ser vazios")
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("probabilidades devem pertencer a [0, 1]")
    if np.any((outcomes < 0.0) | (outcomes > 1.0)):
        raise ValueError("outcomes deve ser binário ou probabilístico em [0, 1]")
    return float(np.mean((probabilities - outcomes) ** 2))
