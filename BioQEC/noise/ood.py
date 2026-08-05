"""Rótulos OOD definidos pelo gerador, independentemente do BioQEC.

O módulo cobre os quatro critérios declarados no artigo: mecanismo não visto,
parâmetro fora do suporte, composição retida e envelope temporal retido.
Nenhuma função recebe distância à memória ou limiar de novidade do método.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

from noise.nonstationary import NoiseRegime


class OODCategory(Enum):
    IN_DISTRIBUTION = "in_distribution"
    UNSEEN_MECHANISM = "unseen_mechanism"
    OUTSIDE_PARAMETER_SUPPORT = "outside_parameter_support"
    UNSEEN_COMPOSITION = "unseen_composition"
    OUTSIDE_TEMPORAL_ENVELOPE = "outside_temporal_envelope"


@dataclass(frozen=True, slots=True)
class TrainingSupport:
    noise_models: frozenset[str]
    p_min: float
    p_max: float
    max_bias_eta: float = 1.0
    max_spatial_delta: float = 0.0
    seen_compositions: frozenset[frozenset[str]] = field(default_factory=frozenset)
    max_drift_rate: float | None = None
    max_burst_duration: int | None = None
    max_persistence: int | None = None

    def __post_init__(self) -> None:
        if not self.noise_models:
            raise ValueError("noise_models não pode ser vazio")
        if not 0 < self.p_min <= self.p_max < 1:
            raise ValueError("use 0 < p_min <= p_max < 1")
        if self.max_bias_eta < 1:
            raise ValueError("max_bias_eta deve ser >= 1")
        if self.max_spatial_delta < 0:
            raise ValueError("max_spatial_delta deve ser não negativo")
        if self.max_drift_rate is not None and self.max_drift_rate < 0:
            raise ValueError("max_drift_rate deve ser não negativo")
        for value, name in (
            (self.max_burst_duration, "max_burst_duration"),
            (self.max_persistence, "max_persistence"),
        ):
            if value is not None and value < 1:
                raise ValueError(f"{name} deve ser positivo")


@dataclass(frozen=True, slots=True)
class TrajectoryDescriptor:
    """Metadados congelados pelo gerador antes do treinamento."""

    regimes: tuple[NoiseRegime, ...]
    active_mechanisms: frozenset[str]
    drift_rate: float | None = None
    burst_duration: int | None = None
    persistence: int | None = None

    def __post_init__(self) -> None:
        if not self.regimes:
            raise ValueError("regimes não pode ser vazio")
        if not self.active_mechanisms:
            raise ValueError("active_mechanisms não pode ser vazio")
        if self.drift_rate is not None and self.drift_rate < 0:
            raise ValueError("drift_rate deve ser não negativo")
        if self.burst_duration is not None and self.burst_duration < 1:
            raise ValueError("burst_duration deve ser positivo")
        if self.persistence is not None and self.persistence < 1:
            raise ValueError("persistence deve ser positiva")


def classify_regime(regime: NoiseRegime, support: TrainingSupport) -> OODCategory:
    """Classifica um regime isolado pelos critérios O1 e O2."""

    if regime.noise_model not in support.noise_models:
        return OODCategory.UNSEEN_MECHANISM
    if not support.p_min <= regime.p <= support.p_max:
        return OODCategory.OUTSIDE_PARAMETER_SUPPORT
    if regime.bias_eta > support.max_bias_eta or regime.spatial_delta > support.max_spatial_delta:
        return OODCategory.OUTSIDE_PARAMETER_SUPPORT
    return OODCategory.IN_DISTRIBUTION


def assess_trajectory(
    descriptor: TrajectoryDescriptor,
    support: TrainingSupport,
) -> tuple[OODCategory, ...]:
    """Retorna todas as causas OOD de uma trajetória, em ordem estável.

    A classificação é inteiramente determinada pelo manifesto do gerador. Um
    resultado vazio é convertido em ``IN_DISTRIBUTION``.
    """

    causes: set[OODCategory] = set()
    for regime in descriptor.regimes:
        category = classify_regime(regime, support)
        if category is not OODCategory.IN_DISTRIBUTION:
            causes.add(category)

    if not descriptor.active_mechanisms.issubset(support.noise_models):
        causes.add(OODCategory.UNSEEN_MECHANISM)

    if (
        len(descriptor.active_mechanisms) > 1
        and descriptor.active_mechanisms not in support.seen_compositions
    ):
        causes.add(OODCategory.UNSEEN_COMPOSITION)

    temporal_outside = (
        support.max_drift_rate is not None
        and descriptor.drift_rate is not None
        and descriptor.drift_rate > support.max_drift_rate
    ) or (
        support.max_burst_duration is not None
        and descriptor.burst_duration is not None
        and descriptor.burst_duration > support.max_burst_duration
    ) or (
        support.max_persistence is not None
        and descriptor.persistence is not None
        and descriptor.persistence > support.max_persistence
    )
    if temporal_outside:
        causes.add(OODCategory.OUTSIDE_TEMPORAL_ENVELOPE)

    if not causes:
        return (OODCategory.IN_DISTRIBUTION,)
    order = {category: index for index, category in enumerate(OODCategory)}
    return tuple(sorted(causes, key=order.__getitem__))


def is_ood(categories: Iterable[OODCategory]) -> bool:
    values = tuple(categories)
    if not values:
        raise ValueError("categories não pode ser vazio")
    return any(value is not OODCategory.IN_DISTRIBUTION for value in values)
