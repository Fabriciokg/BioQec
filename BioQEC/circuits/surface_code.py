"""Geração de circuitos de memória para códigos de superfície rotacionados.

Compatível com Stim 1.16. O módulo mantém a configuração separada do
amostrador para facilitar experimentos pareados e auditoria de sementes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TYPE_CHECKING

import numpy as np

try:
    import stim
except ImportError:  # Permite importar a configuração antes de instalar Stim.
    stim = None  # type: ignore[assignment]

if TYPE_CHECKING:
    import stim as stim_types
else:
    stim_types = Any


NoiseModel = Literal["depolarizing", "biased_Z", "measurement_heavy"]


@dataclass(frozen=True, slots=True)
class CircuitConfig:
    """Configuração imutável de um circuito de memória."""

    distance: int
    rounds: int
    basis: Literal["X", "Z"]
    noise_model: NoiseModel = "depolarizing"
    p: float = 1e-3
    seed: int = 42

    def __post_init__(self) -> None:
        if self.distance < 3 or self.distance % 2 == 0:
            raise ValueError("distance deve ser ímpar e maior ou igual a 3")
        if self.rounds < 1:
            raise ValueError("rounds deve ser maior ou igual a 1")
        if self.basis not in {"X", "Z"}:
            raise ValueError("basis deve ser 'X' ou 'Z'")
        if self.noise_model not in {
            "depolarizing",
            "biased_Z",
            "measurement_heavy",
        }:
            raise ValueError(f"Modelo de ruído desconhecido: {self.noise_model}")
        if not 0.0 <= self.p <= 1.0:
            raise ValueError("p deve pertencer ao intervalo [0, 1]")
        if self.seed < 0:
            raise ValueError("seed deve ser não negativa")
        # Alguns modelos multiplicam p. A validação antecipada evita passar
        # probabilidades inválidas para o Stim.
        _build_noise_args(self)

    @property
    def label(self) -> str:
        return (
            f"d{self.distance}_T{self.rounds}_{self.basis}_"
            f"{self.noise_model}_p{self.p:.2e}_s{self.seed}"
        )


def _require_stim() -> Any:
    if stim is None:
        raise ImportError(
            "Stim não está instalado. Execute `pip install stim==1.16.0`."
        )
    return stim


def _probability(value: float, name: str) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name}={value} não pertence a [0, 1]")
    return value


def _build_noise_args(cfg: CircuitConfig) -> dict[str, float]:
    """Traduz o modelo nominal para os argumentos de ``Circuit.generated``.

    ``biased_Z`` é uma aproximação operacional. O gerador padrão do Stim não
    recebe um canal Pauli assimétrico por localização; por isso, o viés é
    representado por taxas diferentes de preparação e medição. Experimentos
    que exigem um canal Z/X exato devem utilizar um circuito customizado.
    """

    p = float(cfg.p)
    if cfg.noise_model == "depolarizing":
        args = {
            "after_clifford_depolarization": p,
            "after_reset_flip_probability": p,
            "before_measure_flip_probability": p,
            "before_round_data_depolarization": p,
        }
    elif cfg.noise_model == "biased_Z":
        p_z = p * 10.0 / 12.0
        p_xy = p / 12.0
        args = {
            "after_clifford_depolarization": p,
            "after_reset_flip_probability": p_xy,
            "before_measure_flip_probability": p_z,
            "before_round_data_depolarization": p,
        }
    elif cfg.noise_model == "measurement_heavy":
        args = {
            "after_clifford_depolarization": p * 0.5,
            "after_reset_flip_probability": p,
            "before_measure_flip_probability": p * 2.0,
            "before_round_data_depolarization": p * 0.5,
        }
    else:  # Proteção adicional para chamadas não tipadas.
        raise ValueError(f"Modelo de ruído desconhecido: {cfg.noise_model}")

    return {name: _probability(value, name) for name, value in args.items()}


def build_memory_circuit(cfg: CircuitConfig) -> stim_types.Circuit:
    """Constrói um circuito de memória lógica rotacionado."""

    stim_module = _require_stim()
    task = f"surface_code:rotated_memory_{cfg.basis.lower()}"
    return stim_module.Circuit.generated(
        task,
        distance=cfg.distance,
        rounds=cfg.rounds,
        **_build_noise_args(cfg),
    )


def get_detector_error_model(
    circuit: stim_types.Circuit,
    decompose: bool = True,
) -> stim_types.DetectorErrorModel:
    """Extrai o modelo de erro de detectores do circuito."""

    return circuit.detector_error_model(
        decompose_errors=decompose,
        approximate_disjoint_errors=True,
    )


def sample_syndromes(
    circuit: stim_types.Circuit,
    n_shots: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Amostra eventos de detecção e observáveis lógicos em lote."""

    if n_shots < 1:
        raise ValueError("n_shots deve ser maior ou igual a 1")
    if seed < 0:
        raise ValueError("seed deve ser não negativa")

    sampler = circuit.compile_detector_sampler(seed=seed)
    detections, observables = sampler.sample(
        shots=n_shots,
        separate_observables=True,
    )
    return np.asarray(detections, dtype=np.bool_), np.asarray(
        observables, dtype=np.bool_
    )


def circuit_metadata(circuit: stim_types.Circuit) -> dict[str, int]:
    """Retorna metadados estruturais úteis para auditoria."""

    dem = get_detector_error_model(circuit)
    flat_dem = dem.flattened()
    n_dem_errors = sum(1 for instruction in flat_dem if instruction.type == "error")
    return {
        "n_qubits": int(circuit.num_qubits),
        "n_detectors": int(circuit.num_detectors),
        "n_observables": int(circuit.num_observables),
        "n_measurements": int(circuit.num_measurements),
        "n_dem_errors": int(n_dem_errors),
    }
