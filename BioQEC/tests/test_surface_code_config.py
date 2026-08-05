import pytest

from circuits.surface_code import CircuitConfig, _build_noise_args


def test_valid_configuration_label() -> None:
    cfg = CircuitConfig(distance=5, rounds=10, basis="X", p=1e-3, seed=12)
    assert cfg.label.startswith("d5_T10_X_depolarizing")


def test_even_distance_is_rejected() -> None:
    with pytest.raises(ValueError):
        CircuitConfig(distance=4, rounds=4, basis="X")


def test_measurement_heavy_scaling() -> None:
    cfg = CircuitConfig(
        distance=3,
        rounds=3,
        basis="Z",
        noise_model="measurement_heavy",
        p=0.1,
    )
    args = _build_noise_args(cfg)
    assert args["before_measure_flip_probability"] == pytest.approx(0.2)


def test_scaled_probability_above_one_is_rejected() -> None:
    with pytest.raises(ValueError):
        CircuitConfig(
            distance=3,
            rounds=3,
            basis="Z",
            noise_model="measurement_heavy",
            p=0.6,
        )
