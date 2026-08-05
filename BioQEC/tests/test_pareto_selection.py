import pytest

from selection.pareto import CandidateMetrics, select_lexicographic


def candidate(name: str, logical: float, cvar: float) -> CandidateMetrics:
    return CandidateMetrics(name, logical, cvar, 0.03, 10.0, 50.0, 0.5, 0.05, 0.02)


def test_lexicographic_selection_uses_preregistered_order() -> None:
    a = candidate("a", 0.002, 0.04)
    b = candidate("b", 0.001, 0.08)
    assert select_lexicographic([a, b]).name == "b"
    assert select_lexicographic([a, b], order=("cvar", "logical_risk")).name == "a"


def test_lexicographic_selection_rejects_invalid_order() -> None:
    with pytest.raises(ValueError):
        select_lexicographic([candidate("a", 0.1, 0.2)], order=("latency_p99",))
