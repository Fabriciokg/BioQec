from noise.nonstationary import NoiseRegime
from noise.ood import (
    OODCategory,
    TrainingSupport,
    TrajectoryDescriptor,
    assess_trajectory,
    is_ood,
)


def support() -> TrainingSupport:
    return TrainingSupport(
        noise_models=frozenset({"depolarizing", "measurement_heavy"}),
        p_min=1e-4,
        p_max=5e-3,
        seen_compositions=frozenset({frozenset({"depolarizing", "measurement_heavy"})}),
        max_drift_rate=1e-4,
        max_burst_duration=8,
        max_persistence=20,
    )


def test_unseen_composition_is_independent_of_model_threshold() -> None:
    descriptor = TrajectoryDescriptor(
        regimes=(NoiseRegime("a", 1e-3), NoiseRegime("b", 2e-3)),
        active_mechanisms=frozenset({"depolarizing", "leakage"}),
    )
    categories = assess_trajectory(descriptor, support())
    assert OODCategory.UNSEEN_MECHANISM in categories
    assert OODCategory.UNSEEN_COMPOSITION in categories
    assert is_ood(categories)


def test_temporal_envelope_is_detected() -> None:
    descriptor = TrajectoryDescriptor(
        regimes=(NoiseRegime("a", 1e-3),),
        active_mechanisms=frozenset({"depolarizing"}),
        drift_rate=2e-4,
        burst_duration=12,
    )
    assert assess_trajectory(descriptor, support()) == (
        OODCategory.OUTSIDE_TEMPORAL_ENVELOPE,
    )


def test_in_distribution_trajectory() -> None:
    descriptor = TrajectoryDescriptor(
        regimes=(NoiseRegime("a", 1e-3),),
        active_mechanisms=frozenset({"depolarizing"}),
        drift_rate=5e-5,
    )
    assert assess_trajectory(descriptor, support()) == (OODCategory.IN_DISTRIBUTION,)
