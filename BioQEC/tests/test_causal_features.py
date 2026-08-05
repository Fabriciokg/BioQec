import numpy as np
import pytest

from features.causal_features import extract_causal_features


def test_features_are_causal() -> None:
    detections = np.array(
        [[0, 0, 0, 0], [1, 0, 1, 0], [1, 1, 0, 0], [0, 1, 1, 1], [1, 1, 1, 1]],
        dtype=int,
    )
    kwargs = dict(
        check_types=["X", "X", "Z", "Z"],
        adjacency=[(0, 2), (1, 3)],
        t=2,
        window=3,
        leakage=np.zeros_like(detections, dtype=float),
        analog_prob_one=np.full_like(detections, 0.9, dtype=float),
        predictive_probs=np.tile([0.7, 0.1, 0.1, 0.1], (5, 1)),
    )
    before = extract_causal_features(detections, **kwargs)
    changed_future = detections.copy()
    changed_future[3:] = 1 - changed_future[3:]
    after = extract_causal_features(changed_future, **kwargs)
    np.testing.assert_allclose(before.values, after.values, equal_nan=True)


def test_feature_definitions_have_expected_rates() -> None:
    detections = np.array([[0, 0], [1, 0], [1, 1]], dtype=int)
    result = extract_causal_features(
        detections,
        check_types=["X", "Z"],
        adjacency=[(0, 1)],
        window=3,
        leakage=np.zeros_like(detections, dtype=float),
        analog_prob_one=np.full_like(detections, 0.5, dtype=float),
        predictive_probs=np.array([0.25, 0.25, 0.25, 0.25]),
    )
    assert result.values[0] == pytest.approx(2 / 3)
    assert result.values[1] == pytest.approx(1 / 3)
    assert result.values[2] == pytest.approx(1 / 4)
    assert result.values[6] == pytest.approx(1.0)
    assert result.values[7] == pytest.approx(1.0)
