import numpy as np
import pytest

stim = pytest.importorskip("stim")
pytest.importorskip("pymatching")

from circuits.surface_code import CircuitConfig, build_memory_circuit, circuit_metadata, sample_syndromes
from decoders.mwpm import build_matching, decode_batch


def test_static_circuit_can_be_sampled_and_decoded() -> None:
    cfg = CircuitConfig(distance=3, rounds=3, basis="X", p=1e-3, seed=7)
    circuit = build_memory_circuit(cfg)
    metadata = circuit_metadata(circuit)
    assert metadata["n_detectors"] > 0
    assert metadata["n_observables"] == 1

    detections, observables = sample_syndromes(circuit, n_shots=64, seed=cfg.seed)
    matching = build_matching(circuit)
    result = decode_batch(matching, detections, observables)
    assert result.predictions.shape == observables.shape
    assert result.logical_failures.shape == (64,)
    assert 0.0 <= result.logical_error_rate <= 1.0
    assert np.isfinite(result.microseconds_per_shot)
