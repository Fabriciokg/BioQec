"""Seleção multiobjetivo sem soma ponderada de métricas incompatíveis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True, slots=True)
class CandidateMetrics:
    name: str
    logical_risk: float
    cvar: float
    brier: float
    adaptation_time: float
    latency_p99: float
    cost: float
    escalation_rate: float
    instability: float
    guardrail_violations: int = 0

    @property
    def objectives(self) -> np.ndarray:
        return np.asarray(
            [self.logical_risk, self.cvar, self.brier, self.adaptation_time],
            dtype=float,
        )


@dataclass(frozen=True, slots=True)
class OperationalLimits:
    latency_p99: float
    cost: float
    escalation_rate: float
    instability: float


def is_feasible(candidate: CandidateMetrics, limits: OperationalLimits) -> bool:
    return bool(
        candidate.guardrail_violations == 0
        and candidate.latency_p99 <= limits.latency_p99
        and candidate.cost <= limits.cost
        and candidate.escalation_rate <= limits.escalation_rate
        and candidate.instability <= limits.instability
    )


def dominates(
    candidate: CandidateMetrics,
    reference: CandidateMetrics,
    tolerances: Iterable[float] = (0.0, 0.0, 0.0, 0.0),
) -> bool:
    eps = np.asarray(tuple(tolerances), dtype=float)
    if eps.shape != (4,) or np.any(eps < 0):
        raise ValueError("tolerances deve conter quatro valores não negativos")
    c = candidate.objectives
    r = reference.objectives
    return bool(np.all(c <= r + eps) and np.any(c < r - eps))


def pareto_front(
    candidates: Iterable[CandidateMetrics],
    limits: OperationalLimits,
    tolerances: Iterable[float] = (0.0, 0.0, 0.0, 0.0),
) -> list[CandidateMetrics]:
    feasible = [candidate for candidate in candidates if is_feasible(candidate, limits)]
    return [
        candidate
        for candidate in feasible
        if not any(
            dominates(other, candidate, tolerances)
            for other in feasible
            if other.name != candidate.name
        )
    ]


def select_lexicographic(
    candidates: Iterable[CandidateMetrics],
    order: tuple[str, ...] = ("logical_risk", "cvar", "brier", "adaptation_time"),
) -> CandidateMetrics:
    """Seleciona um candidato da fronteira por ordem pré-registrada.

    A função não normaliza nem soma os objetivos. Os nomes devem apontar para
    campos numéricos de ``CandidateMetrics`` e a ordem precisa ser fixada antes
    de abrir o conjunto confirmatório.
    """

    values = list(candidates)
    if not values:
        raise ValueError("candidates não pode ser vazio")
    allowed = {"logical_risk", "cvar", "brier", "adaptation_time"}
    if not order or len(set(order)) != len(order) or not set(order).issubset(allowed):
        raise ValueError("order contém campo inválido ou repetido")
    return min(values, key=lambda candidate: tuple(getattr(candidate, field) for field in order))
