import numpy as np

from monitoring.change_detection import CUSUMConfig, fit_reference, upper_cusum


def test_cusum_responds_to_upward_shift() -> None:
    values = np.r_[np.zeros(20), np.ones(20) * 2.0]
    mean, std = fit_reference(np.r_[np.zeros(10), np.ones(10) * 0.1])
    result = upper_cusum(
        values,
        CUSUMConfig(reference_mean=mean, reference_std=std, allowance=0.5, threshold=5.0),
    )
    assert np.any(result.alarms[20:])
