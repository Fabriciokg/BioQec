import numpy as np
import pytest

from metrics.evaluation import (
    brier_score,
    effective_error_per_round,
    summarize_failures,
    wilson_interval,
)


def test_wilson_interval_contains_rate() -> None:
    estimate = wilson_interval(10, 100)
    assert estimate.ci_low <= estimate.rate <= estimate.ci_high
    assert estimate.rate == pytest.approx(0.1)


def test_zero_events_has_nonzero_upper_bound() -> None:
    estimate = wilson_interval(0, 1000)
    assert estimate.ci_low == 0.0
    assert estimate.ci_high > 0.0


def test_summarize_failures() -> None:
    estimate = summarize_failures(np.array([False, True, False, True]))
    assert estimate.events == 2
    assert estimate.trials == 4


def test_effective_error_per_round() -> None:
    value = effective_error_per_round(0.1, rounds=10)
    assert 0.0 < value < 0.1
    assert (1.0 - value) ** 10 == pytest.approx(0.9)


def test_brier_score_perfect_prediction() -> None:
    assert brier_score(np.array([0.0, 1.0]), np.array([0.0, 1.0])) == 0.0
