from monitoring.novelty import ChangeNoveltyCoordinator, ChangeState, NoveltyConfig


def test_cusum_opens_only_one_event_and_distance_classifies_it() -> None:
    coordinator = ChangeNoveltyCoordinator(
        NoveltyConfig(
            cusum_threshold=5.0,
            reset_threshold=1.0,
            grace_cycles=2,
            novelty_window=3,
            novelty_threshold=9.0,
            novelty_fraction=2 / 3,
            reset_cycles=2,
        )
    )
    first = coordinator.step(6.0, 12.0)
    assert first.change_started
    event_id = first.event_id
    coordinator.step(7.0, 11.0)
    classified = coordinator.step(8.0, 10.0)
    assert classified.state is ChangeState.NOVEL
    assert classified.event_id == event_id
    duplicate = coordinator.step(9.0, 15.0)
    assert not duplicate.change_started
    assert duplicate.event_id == event_id


def test_event_resets_after_stability() -> None:
    coordinator = ChangeNoveltyCoordinator(
        NoveltyConfig(5.0, 1.0, grace_cycles=1, novelty_window=1, reset_cycles=2)
    )
    coordinator.step(6.0, 1.0)
    coordinator.step(0.0, 1.0)
    final = coordinator.step(0.0, 1.0)
    assert final.state is ChangeState.NORMAL
    assert final.event_id is None
