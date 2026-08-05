"""Geradores de regimes de ruído não estacionários para o BioQEC."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

import numpy as np


class RegimeType(Enum):
    STATIONARY = "stationary"
    DRIFT = "drift"
    ABRUPT = "abrupt"
    RECURRENCE = "recurrence"
    OOD = "ood"


@dataclass(frozen=True, slots=True)
class NoiseRegime:
    """Parâmetros nominais de um regime de ruído."""

    name: str
    p: float
    noise_model: str = "depolarizing"
    bias_eta: float = 1.0
    spatial_delta: float = 0.0

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("name não pode ser vazio")
        if not 0.0 < self.p < 1.0:
            raise ValueError(f"p={self.p} inválido; use 0 < p < 1")
        if self.bias_eta <= 0.0:
            raise ValueError("bias_eta deve ser positivo")
        if self.spatial_delta < 0.0:
            raise ValueError("spatial_delta deve ser não negativo")


@dataclass(frozen=True, slots=True)
class NoiseTrajectory:
    """Trajetória latente de ruído ao longo de ``T`` ciclos."""

    T: int
    p_sequence: np.ndarray
    regime_sequence: np.ndarray
    change_points: tuple[int, ...]
    regime_type: RegimeType
    regimes: tuple[NoiseRegime, ...]
    seed: int
    label: str = ""

    def __post_init__(self) -> None:
        p_sequence = np.asarray(self.p_sequence, dtype=float)
        regime_sequence = np.asarray(self.regime_sequence, dtype=int)
        if self.T < 1:
            raise ValueError("T deve ser maior ou igual a 1")
        if p_sequence.shape != (self.T,):
            raise ValueError("p_sequence deve ter shape (T,)")
        if regime_sequence.shape != (self.T,):
            raise ValueError("regime_sequence deve ter shape (T,)")
        if np.any(~np.isfinite(p_sequence)) or np.any((p_sequence <= 0) | (p_sequence >= 1)):
            raise ValueError("p_sequence deve conter probabilidades finitas em (0, 1)")
        if not self.regimes:
            raise ValueError("regimes não pode ser vazio")
        if np.any(regime_sequence < 0) or np.any(regime_sequence >= len(self.regimes)):
            raise ValueError("regime_sequence contém índices inválidos")
        if any(cp <= 0 or cp >= self.T for cp in self.change_points):
            raise ValueError("cada ponto de mudança deve pertencer a 1,...,T-1")
        if tuple(sorted(set(self.change_points))) != self.change_points:
            raise ValueError("change_points deve estar ordenado e sem repetições")
        if self.seed < 0:
            raise ValueError("seed deve ser não negativa")
        p_sequence.setflags(write=False)
        regime_sequence.setflags(write=False)
        object.__setattr__(self, "p_sequence", p_sequence)
        object.__setattr__(self, "regime_sequence", regime_sequence)


@dataclass(frozen=True, slots=True)
class TrajectoryWindow:
    """Resumo causal de uma janela para execução exploratória com Stim."""

    index: int
    start: int
    stop: int
    p: float
    regime_index: int
    noise_model: str
    mixed_regime: bool
    contains_change: bool

    @property
    def rounds(self) -> int:
        return self.stop - self.start


def _validate_transition(T: int, start: float, end: float | None = None) -> None:
    if T < 2:
        raise ValueError("T deve ser maior ou igual a 2 para trajetórias com mudança")
    if not 0.0 < start < 1.0:
        raise ValueError("a fração de mudança deve pertencer a (0, 1)")
    if end is not None and not start < end < 1.0:
        raise ValueError("use 0 < início < fim < 1")


def make_stationary(regime: NoiseRegime, T: int, seed: int = 0) -> NoiseTrajectory:
    if T < 1:
        raise ValueError("T deve ser maior ou igual a 1")
    return NoiseTrajectory(
        T=T,
        p_sequence=np.full(T, regime.p, dtype=float),
        regime_sequence=np.zeros(T, dtype=int),
        change_points=(),
        regime_type=RegimeType.STATIONARY,
        regimes=(regime,),
        seed=seed,
        label=f"stationary_{regime.name}_T{T}",
    )


def make_abrupt_change(
    regime_A: NoiseRegime,
    regime_B: NoiseRegime,
    T: int,
    tc_frac: float = 0.5,
    seed: int = 0,
) -> NoiseTrajectory:
    _validate_transition(T, tc_frac)
    tc = min(T - 1, max(1, int(T * tc_frac)))
    p_seq = np.r_[np.full(tc, regime_A.p), np.full(T - tc, regime_B.p)]
    reg_seq = np.r_[np.zeros(tc, dtype=int), np.ones(T - tc, dtype=int)]
    return NoiseTrajectory(
        T=T,
        p_sequence=p_seq,
        regime_sequence=reg_seq,
        change_points=(tc,),
        regime_type=RegimeType.ABRUPT,
        regimes=(regime_A, regime_B),
        seed=seed,
        label=f"abrupt_{regime_A.name}_to_{regime_B.name}_tc{tc}_T{T}",
    )


def make_drift(
    regime_A: NoiseRegime,
    regime_B: NoiseRegime,
    T: int,
    drift_start_frac: float = 0.2,
    drift_end_frac: float = 0.7,
    smooth: bool = True,
    seed: int = 0,
) -> NoiseTrajectory:
    _validate_transition(T, drift_start_frac, drift_end_frac)
    t0 = max(1, int(T * drift_start_frac))
    t1 = min(T - 1, int(T * drift_end_frac))
    if t1 <= t0:
        raise ValueError("a deriva precisa ocupar ao menos um ciclo")

    p_seq = np.empty(T, dtype=float)
    reg_seq = np.zeros(T, dtype=int)
    for t in range(T):
        if t < t0:
            p_seq[t] = regime_A.p
        elif t >= t1:
            p_seq[t] = regime_B.p
            reg_seq[t] = 1
        else:
            r = (t - t0) / (t1 - t0)
            g = 3.0 * r**2 - 2.0 * r**3 if smooth else r
            p_seq[t] = (1.0 - g) * regime_A.p + g * regime_B.p
            reg_seq[t] = int(g >= 0.5)

    return NoiseTrajectory(
        T=T,
        p_sequence=np.clip(p_seq, 1e-12, 1.0 - 1e-12),
        regime_sequence=reg_seq,
        change_points=(t0, t1),
        regime_type=RegimeType.DRIFT,
        regimes=(regime_A, regime_B),
        seed=seed,
        label=f"drift_{regime_A.name}_to_{regime_B.name}_{t0}_{t1}_T{T}",
    )


def make_recurrence(
    regime_A: NoiseRegime,
    regime_B: NoiseRegime,
    T: int,
    tc1_frac: float = 0.33,
    tc2_frac: float = 0.67,
    seed: int = 0,
) -> NoiseTrajectory:
    _validate_transition(T, tc1_frac, tc2_frac)
    tc1 = max(1, int(T * tc1_frac))
    tc2 = min(T - 1, int(T * tc2_frac))
    if tc2 <= tc1:
        raise ValueError("os pontos de recorrência devem ser distintos")

    p_seq = np.r_[
        np.full(tc1, regime_A.p),
        np.full(tc2 - tc1, regime_B.p),
        np.full(T - tc2, regime_A.p),
    ]
    reg_seq = np.r_[
        np.zeros(tc1, dtype=int),
        np.ones(tc2 - tc1, dtype=int),
        np.zeros(T - tc2, dtype=int),
    ]
    return NoiseTrajectory(
        T=T,
        p_sequence=p_seq,
        regime_sequence=reg_seq,
        change_points=(tc1, tc2),
        regime_type=RegimeType.RECURRENCE,
        regimes=(regime_A, regime_B),
        seed=seed,
        label=f"recurrence_{regime_A.name}_{regime_B.name}_{tc1}_{tc2}_T{T}",
    )


def make_ood(
    regime_A: NoiseRegime,
    T: int,
    tc_frac: float = 0.5,
    ood_multiplier: float = 3.0,
    seed: int = 0,
) -> NoiseTrajectory:
    _validate_transition(T, tc_frac)
    if ood_multiplier <= 1.0:
        raise ValueError("ood_multiplier deve ser maior que 1")
    tc = min(T - 1, max(1, int(T * tc_frac)))
    p_ood = min(regime_A.p * ood_multiplier, 0.45)
    regime_ood = NoiseRegime(
        name="ood",
        p=p_ood,
        noise_model="measurement_heavy",
    )
    p_seq = np.r_[np.full(tc, regime_A.p), np.full(T - tc, p_ood)]
    reg_seq = np.r_[np.zeros(tc, dtype=int), np.ones(T - tc, dtype=int)]
    return NoiseTrajectory(
        T=T,
        p_sequence=p_seq,
        regime_sequence=reg_seq,
        change_points=(tc,),
        regime_type=RegimeType.OOD,
        regimes=(regime_A, regime_ood),
        seed=seed,
        label=f"ood_from_{regime_A.name}_x{ood_multiplier:g}_T{T}",
    )


def split_trajectory(
    trajectory: NoiseTrajectory,
    window_size: int,
    p_aggregation: Literal["mean", "median"] = "mean",
) -> list[TrajectoryWindow]:
    """Converte uma trajetória em janelas independentes para o protótipo.

    Esta função não cria um único circuito contínuo com p variando a cada ciclo.
    Cada janela será simulada como um circuito de memória independente, o que é
    apropriado para validação exploratória do gerador e dos pesos do decodificador.
    """

    if window_size < 1:
        raise ValueError("window_size deve ser maior ou igual a 1")
    if p_aggregation not in {"mean", "median"}:
        raise ValueError("p_aggregation deve ser 'mean' ou 'median'")
    windows: list[TrajectoryWindow] = []
    change_points = set(trajectory.change_points)
    for index, start in enumerate(range(0, trajectory.T, window_size)):
        stop = min(start + window_size, trajectory.T)
        p_slice = trajectory.p_sequence[start:stop]
        regime_slice = trajectory.regime_sequence[start:stop]
        counts = np.bincount(regime_slice, minlength=len(trajectory.regimes))
        regime_index = int(np.argmax(counts))
        p = float(np.mean(p_slice) if p_aggregation == "mean" else np.median(p_slice))
        windows.append(
            TrajectoryWindow(
                index=index,
                start=start,
                stop=stop,
                p=p,
                regime_index=regime_index,
                noise_model=trajectory.regimes[regime_index].noise_model,
                mixed_regime=bool(np.unique(regime_slice).size > 1),
                contains_change=any(start <= cp < stop for cp in change_points),
            )
        )
    return windows


def default_regimes() -> dict[str, NoiseRegime]:
    return {
        "low": NoiseRegime("low", p=1e-3),
        "medium": NoiseRegime("medium", p=3e-3),
        "high": NoiseRegime("high", p=7e-3),
        "biased": NoiseRegime("biased", p=3e-3, noise_model="biased_Z", bias_eta=10.0),
        "meas": NoiseRegime("meas", p=3e-3, noise_model="measurement_heavy"),
    }
