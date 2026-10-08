"""Tests for the KV-block selector (teleportcache.prediction.block_selector)."""

from __future__ import annotations

import pytest

from teleportcache.prediction.block_selector import BlockSelector
from teleportcache.types import HandoffEvent


def event(step: int, frm: str, to: str, blocks: tuple[str, ...]) -> HandoffEvent:
    return HandoffEvent(step=step, from_agent=frm, to_agent=to, required_blocks=blocks)


class TestBlockSelector:
    def test_cold_start_returns_empty(self) -> None:
        selector = BlockSelector()
        assert selector.rank("B") == []

    def test_decay_prefers_recent_blocks(self) -> None:
        selector = BlockSelector(half_life=8.0)
        selector.update(event(0, "A", "B", ("b_old",)))
        selector.update(event(20, "B", "B", ("b_new",)))  # B receives again, newer page
        ranked = selector.rank("B")
        assert ranked[0][0] == "b_new"
        assert ranked[1][0] == "b_old"
        # same usage count (1), so recency decay alone orders them
        new_score = ranked[0][1]
        old_score = ranked[1][1]
        assert old_score == pytest.approx(new_score * 0.5 ** (20 / 8.0))

    def test_exclude_set_skips_resident_blocks(self) -> None:
        selector = BlockSelector(half_life=1e9)
        selector.update(event(0, "A", "B", ("b1", "b2", "b3")))
        ranked = selector.rank("B", exclude={"b1", "b3"})
        assert [block for block, _ in ranked] == ["b2"]

    def test_cold_start_for_agent_with_other_agents_known(self) -> None:
        selector = BlockSelector()
        selector.update(event(0, "A", "B", ("b1",)))
        assert selector.rank("C") == []  # C never observed
        # b1 is the only known block of B and was required in its only
        # handoff -> share 1.0 triggers the always-required boost (x2).
        assert selector.rank("B") == [("b1", pytest.approx(2.0))]

    def test_always_required_boost(self) -> None:
        selector = BlockSelector(half_life=1e9, boost_threshold=0.8, boost_weight=2.0)
        # Agent B: 2 handoffs, "b1" in both -> share 1.0 >= 0.8 -> boosted.
        selector.update(event(0, "A", "B", ("b1", "b2")))
        selector.update(event(1, "B", "B", ("b1", "b3")))
        # Agent C: 4 handoffs, "c1" in two -> share 0.5 -> not boosted.
        for step in range(4):
            blocks = ("c1", "c2") if step < 2 else ("c2", "c3")
            selector.update(event(step, "A", "C", blocks))
        b1_score = dict(selector.rank("B"))["b1"]  # count 2, fresh
        c1_score = dict(selector.rank("C"))["c1"]  # count 2, fresh
        assert b1_score == pytest.approx(2 * 2.0)  # boosted
        assert c1_score == pytest.approx(2.0)  # unboosted

    def test_usage_count_breaks_recency_ties(self) -> None:
        selector = BlockSelector(half_life=1e9)
        selector.update(event(0, "A", "B", ("b1",)))
        selector.update(event(1, "B", "B", ("b1", "b2")))
        ranked = selector.rank("B")
        assert ranked[0][0] == "b1"  # count 2 beats count 1
        assert ranked[0][1] == pytest.approx(4.0)  # 2 uses x boost (share 1.0)
        assert ranked[1] == ("b2", pytest.approx(1.0))  # share 0.5, unboosted

    def test_reset_clears_everything(self) -> None:
        selector = BlockSelector()
        selector.update(event(0, "A", "B", ("b1",)))
        selector.reset()
        assert selector.rank("B") == []
        # and re-learns from scratch after reset
        selector.update(event(1, "B", "C", ("c9",)))
        # c9: share 1.0 in C's only handoff -> boosted score 2.0
        assert selector.rank("C") == [("c9", pytest.approx(2.0))]
        assert selector.rank("B") == []

    def test_invalid_half_life_raises(self) -> None:
        with pytest.raises(ValueError):
            BlockSelector(half_life=0.0)
        with pytest.raises(ValueError):
            BlockSelector(half_life=-1.0)
