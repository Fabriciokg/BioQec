from noise.nonstationary import NoiseRegime
from noise.ood import OODCategory, TrainingSupport, classify_regime


def test_ood_label_does_not_use_model_novelty_threshold() -> None:
    support = TrainingSupport(frozenset({"depolarizing"}), 1e-3, 5e-3)
    assert classify_regime(NoiseRegime("known", 3e-3), support) is OODCategory.IN_DISTRIBUTION
    assert classify_regime(NoiseRegime("new", 3e-3, noise_model="measurement_heavy"), support) is OODCategory.UNSEEN_MECHANISM
    assert classify_regime(NoiseRegime("strong", 8e-3), support) is OODCategory.OUTSIDE_PARAMETER_SUPPORT
