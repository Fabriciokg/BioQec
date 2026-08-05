from selection.pareto import CandidateMetrics, OperationalLimits, dominates, pareto_front


def candidate(name: str, risk: float, latency: float = 10.0) -> CandidateMetrics:
    return CandidateMetrics(name, risk, 0.2, 0.1, 4.0, latency, 1.0, 0.1, 0.1)


def test_pareto_excludes_infeasible_candidate() -> None:
    limits = OperationalLimits(20.0, 2.0, 0.2, 0.2)
    front = pareto_front([candidate("a", 0.02), candidate("b", 0.01, latency=30.0)], limits)
    assert [item.name for item in front] == ["a"]


def test_dominance_uses_objectives_not_units_of_constraints() -> None:
    assert dominates(candidate("better", 0.01), candidate("active", 0.02))
