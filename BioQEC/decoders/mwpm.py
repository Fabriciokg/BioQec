"""Decodificador de referência MWPM baseado em PyMatching."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, TYPE_CHECKING

import numpy as np

try:
    import pymatching
except ImportError:
    pymatching = None  # type: ignore[assignment]

if TYPE_CHECKING:
    import stim
else:
    stim = Any


@dataclass(frozen=True, slots=True)
class DecodeResult:
    predictions: np.ndarray
    logical_failures: np.ndarray
    logical_error_rate: float
    elapsed_seconds: float
    microseconds_per_shot: float


def _require_pymatching() -> Any:
    if pymatching is None:
        raise ImportError(
            "PyMatching não está instalado. Execute `pip install pymatching==2.4.0`."
        )
    return pymatching


def build_matching(
    circuit: stim.Circuit,
    enable_correlations: bool = False,
) -> Any:
    """Constrói o grafo de matching a partir do DEM do circuito."""

    module = _require_pymatching()
    dem = circuit.detector_error_model(
        decompose_errors=True,
        approximate_disjoint_errors=True,
    )
    return module.Matching.from_detector_error_model(
        dem,
        enable_correlations=enable_correlations,
    )


def decode_batch(
    matching: Any,
    detections: np.ndarray,
    observables: np.ndarray,
    enable_correlations: bool = False,
) -> DecodeResult:
    """Decodifica um lote e calcula a taxa de falha lógica."""

    detections = np.asarray(detections, dtype=np.bool_)
    observables = np.asarray(observables, dtype=np.bool_)
    if detections.ndim != 2 or observables.ndim != 2:
        raise ValueError("detections e observables devem ser matrizes 2D")
    if detections.shape[0] != observables.shape[0]:
        raise ValueError("detections e observables devem ter o mesmo número de shots")
    if detections.shape[0] == 0:
        raise ValueError("o lote não pode ser vazio")

    start = perf_counter()
    predictions = matching.decode_batch(
        detections,
        enable_correlations=enable_correlations,
    )
    elapsed = perf_counter() - start
    predictions = np.asarray(predictions, dtype=np.bool_)
    if predictions.shape != observables.shape:
        raise RuntimeError(
            f"shape inesperado do decoder: {predictions.shape} != {observables.shape}"
        )
    failures = np.any(predictions != observables, axis=1)
    return DecodeResult(
        predictions=predictions,
        logical_failures=failures,
        logical_error_rate=float(np.mean(failures)),
        elapsed_seconds=float(elapsed),
        microseconds_per_shot=float(1e6 * elapsed / detections.shape[0]),
    )
