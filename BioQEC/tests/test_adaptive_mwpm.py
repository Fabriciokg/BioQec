import numpy as np

from decoders.adaptive_mwpm import (
    PeriodicRecalibrationConfig,
    probabilities_to_weights,
    should_recalibrate,
    shrink_probabilities,
)


def test_periodic_schedule_is_explicit() -> None:
    cfg = PeriodicRecalibrationConfig(period=4, calibration_window=8)
    assert not should_recalibrate(3, cfg)
    assert should_recalibrate(4, cfg)


def test_shrinkage_and_weights_are_finite() -> None:
    cfg = PeriodicRecalibrationConfig(4, 8, shrinkage_to_initial=0.25)
    p = shrink_probabilities(np.array([0.2, 0.8]), np.array([0.1, 0.1]), cfg)
    np.testing.assert_allclose(p, [0.175, 0.625])
    assert np.isfinite(probabilities_to_weights(p)).all()
