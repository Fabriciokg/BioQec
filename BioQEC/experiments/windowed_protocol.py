"""Protocolo exploratório em janelas independentes para ruído não estacionário.

O Stim gera circuitos com parâmetros de ruído estáticos. Este módulo aproxima
uma trajetória p(t) por janelas e simula cada janela como um circuito de memória
independente. Isso permite testar o gerador, o descasamento de pesos e a lógica
de monitoramento sem alegar continuidade física entre janelas.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from circuits.surface_code import CircuitConfig, build_memory_circuit, sample_syndromes
from decoders.mwpm import build_matching, decode_batch
from monitoring.change_detection import syndrome_density
from noise.nonstationary import NoiseRegime, NoiseTrajectory, split_trajectory


@dataclass(frozen=True, slots=True)
class WindowBenchmarkResult:
    window: int
    start: int
    stop: int
    rounds: int
    p_actual: float
    actual_noise_model: str
    mixed_regime: bool
    contains_change: bool
    detection_density: float
    static_ler: float
    oracle_ler: float
    static_us_per_shot: float
    oracle_us_per_shot: float
    n_shots: int
    static_failures: int
    oracle_failures: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_windowed_benchmark(
    trajectory: NoiseTrajectory,
    calibration_regime: NoiseRegime,
    distance: int = 5,
    basis: str = "X",
    window_size: int = 5,
    shots_per_window: int = 2_000,
    enable_correlations: bool = False,
) -> list[WindowBenchmarkResult]:
    """Compara MWPM estático e MWPM-oráculo sobre as mesmas síndromes.

    O decoder estático usa o regime de calibração em todas as janelas. O
    decoder-oráculo usa os parâmetros nominais da janela apenas como limite de
    referência; ele não representa uma solução operacional do BioQEC.
    """

    if basis not in {"X", "Z"}:
        raise ValueError("basis deve ser 'X' ou 'Z'")
    if shots_per_window < 1:
        raise ValueError("shots_per_window deve ser positivo")

    results: list[WindowBenchmarkResult] = []
    for window in split_trajectory(trajectory, window_size=window_size):
        sample_seed = trajectory.seed + 10_007 * (window.index + 1)
        actual_cfg = CircuitConfig(
            distance=distance,
            rounds=window.rounds,
            basis=basis,  # type: ignore[arg-type]
            noise_model=window.noise_model,  # type: ignore[arg-type]
            p=window.p,
            seed=sample_seed,
        )
        static_cfg = CircuitConfig(
            distance=distance,
            rounds=window.rounds,
            basis=basis,  # type: ignore[arg-type]
            noise_model=calibration_regime.noise_model,  # type: ignore[arg-type]
            p=calibration_regime.p,
            seed=sample_seed,
        )

        actual_circuit = build_memory_circuit(actual_cfg)
        detections, observables = sample_syndromes(
            actual_circuit,
            n_shots=shots_per_window,
            seed=sample_seed,
        )

        static_matching = build_matching(static_cfg_to_circuit(static_cfg), enable_correlations)
        oracle_matching = build_matching(actual_circuit, enable_correlations)
        static_result = decode_batch(
            static_matching,
            detections,
            observables,
            enable_correlations,
        )
        oracle_result = decode_batch(
            oracle_matching,
            detections,
            observables,
            enable_correlations,
        )

        results.append(
            WindowBenchmarkResult(
                window=window.index,
                start=window.start,
                stop=window.stop,
                rounds=window.rounds,
                p_actual=window.p,
                actual_noise_model=window.noise_model,
                mixed_regime=window.mixed_regime,
                contains_change=window.contains_change,
                detection_density=syndrome_density(detections),
                static_ler=static_result.logical_error_rate,
                oracle_ler=oracle_result.logical_error_rate,
                static_us_per_shot=static_result.microseconds_per_shot,
                oracle_us_per_shot=oracle_result.microseconds_per_shot,
                n_shots=shots_per_window,
                static_failures=int(np.sum(static_result.logical_failures)),
                oracle_failures=int(np.sum(oracle_result.logical_failures)),
            )
        )
    return results


def static_cfg_to_circuit(cfg: CircuitConfig) -> Any:
    """Ponto isolado para facilitar mocking nos testes."""

    return build_memory_circuit(cfg)
