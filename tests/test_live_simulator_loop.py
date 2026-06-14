from __future__ import annotations

from random import Random

from scripts.runtime.live_simulator_loop import _jittered_batch_size


def test_live_simulator_jitter_keeps_average_but_breaks_metronome() -> None:
    rng = Random(123)

    sizes = [_jittered_batch_size(1, rng, 1.0) for _ in range(50)]

    assert {0, 1, 2}.issubset(set(sizes))
    assert 0.8 <= sum(sizes) / len(sizes) <= 1.2


def test_live_simulator_jitter_can_be_disabled() -> None:
    rng = Random(123)

    sizes = [_jittered_batch_size(2, rng, 0.0) for _ in range(10)]

    assert sizes == [2] * 10
