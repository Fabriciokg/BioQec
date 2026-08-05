import numpy as np
import pytest

from noise.nonstationary import (
    RegimeType,
    NoiseRegime,
    default_regimes,
    make_abrupt_change,
    make_drift,
    make_ood,
    make_recurrence,
    make_stationary,
    split_trajectory,
)


def test_stationary_is_constant() -> None:
    regime = NoiseRegime("base", 1e-3)
    trajectory = make_stationary(regime, T=20, seed=7)
    assert trajectory.regime_type is RegimeType.STATIONARY
    assert trajectory.change_points == ()
    assert np.allclose(trajectory.p_sequence, regime.p)
    assert not trajectory.p_sequence.flags.writeable


def test_abrupt_change_has_expected_segments() -> None:
    regimes = default_regimes()
    trajectory = make_abrupt_change(regimes["low"], regimes["high"], T=40, tc_frac=0.5)
    assert trajectory.change_points == (20,)
    assert np.allclose(trajectory.p_sequence[:20], regimes["low"].p)
    assert np.allclose(trajectory.p_sequence[20:], regimes["high"].p)


def test_drift_is_bounded_and_monotone() -> None:
    regimes = default_regimes()
    trajectory = make_drift(regimes["low"], regimes["high"], T=50)
    assert np.all(np.diff(trajectory.p_sequence) >= -1e-15)
    assert trajectory.p_sequence.min() >= regimes["low"].p
    assert trajectory.p_sequence.max() <= regimes["high"].p


def test_recurrence_returns_to_a() -> None:
    regimes = default_regimes()
    trajectory = make_recurrence(regimes["low"], regimes["high"], T=60)
    assert trajectory.regime_type is RegimeType.RECURRENCE
    assert trajectory.regime_sequence[0] == 0
    assert trajectory.regime_sequence[30] == 1
    assert trajectory.regime_sequence[-1] == 0


def test_ood_changes_model_and_probability() -> None:
    low = default_regimes()["low"]
    trajectory = make_ood(low, T=30, ood_multiplier=4)
    assert trajectory.regime_type is RegimeType.OOD
    assert trajectory.regimes[1].noise_model == "measurement_heavy"
    assert trajectory.regimes[1].p == pytest.approx(4 * low.p)


def test_split_trajectory_covers_all_cycles() -> None:
    regimes = default_regimes()
    trajectory = make_abrupt_change(regimes["low"], regimes["high"], T=23, tc_frac=0.5)
    windows = split_trajectory(trajectory, window_size=5)
    assert windows[0].start == 0
    assert windows[-1].stop == 23
    assert sum(window.rounds for window in windows) == 23
    assert any(window.contains_change for window in windows)


def test_invalid_regime_probability_is_rejected() -> None:
    with pytest.raises(ValueError):
        NoiseRegime("invalid", 0.0)
