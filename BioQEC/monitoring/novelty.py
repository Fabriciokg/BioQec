"""Coordenação entre detecção de mudança e classificação de novidade.

O CUSUM abre um único episódio. A distância à memória classifica o episódio
após um período de graça; não cria alarmes concorrentes.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ChangeState(Enum):
    NORMAL = "normal"
    PENDING = "pending"
    KNOWN = "known"
    NOVEL = "novel"


@dataclass(frozen=True, slots=True)
class NoveltyConfig:
    cusum_threshold: float
    reset_threshold: float
    grace_cycles: int = 3
    novelty_threshold: float = 9.0
    novelty_window: int = 4
    novelty_fraction: float = 0.75
    reset_cycles: int = 4

    def __post_init__(self) -> None:
        if self.cusum_threshold <= self.reset_threshold:
            raise ValueError("cusum_threshold deve exceder reset_threshold")
        if min(self.grace_cycles, self.novelty_window, self.reset_cycles) < 1:
            raise ValueError("janelas devem ser positivas")
        if not 0 < self.novelty_fraction <= 1:
            raise ValueError("novelty_fraction deve pertencer a (0, 1]")


@dataclass(frozen=True, slots=True)
class NoveltyStep:
    state: ChangeState
    event_id: int | None
    change_started: bool
    classified_now: bool
    novelty_score: float


class ChangeNoveltyCoordinator:
    """Máquina de estados causal para episódios de mudança."""

    def __init__(self, config: NoveltyConfig):
        self.config = config
        self.state = ChangeState.NORMAL
        self.event_id: int | None = None
        self._next_event_id = 1
        self._age = 0
        self._distances: list[float] = []
        self._reset_count = 0

    def step(self, cusum_score: float, raw_distance: float) -> NoveltyStep:
        if raw_distance < 0:
            raise ValueError("raw_distance deve ser não negativa")
        started = False
        classified = False

        if self.state is ChangeState.NORMAL and cusum_score >= self.config.cusum_threshold:
            self.state = ChangeState.PENDING
            self.event_id = self._next_event_id
            self._next_event_id += 1
            self._age = 0
            self._distances = []
            self._reset_count = 0
            started = True

        if self.state is not ChangeState.NORMAL:
            self._age += 1
            self._distances.append(float(raw_distance))
            recent = self._distances[-self.config.novelty_window :]
            novelty_score = sum(v > self.config.novelty_threshold for v in recent) / len(recent)
            if self.state is ChangeState.PENDING and self._age >= self.config.grace_cycles and len(recent) >= self.config.novelty_window:
                self.state = (
                    ChangeState.NOVEL
                    if novelty_score >= self.config.novelty_fraction
                    else ChangeState.KNOWN
                )
                classified = True

            if cusum_score < self.config.reset_threshold:
                self._reset_count += 1
            else:
                self._reset_count = 0
            if self._reset_count >= self.config.reset_cycles:
                self.state = ChangeState.NORMAL
                self.event_id = None
                self._age = 0
                self._distances = []
                self._reset_count = 0
                novelty_score = 0.0
        else:
            novelty_score = 0.0

        return NoveltyStep(
            state=self.state,
            event_id=self.event_id,
            change_started=started,
            classified_now=classified,
            novelty_score=float(novelty_score),
        )
