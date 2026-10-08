"""Tests for the clairvoyant oracle strategy (teleportcache.prediction.oracle)."""

from __future__ import annotations

import pytest

from teleportcache.prediction.oracle import OraclePrefetchStrategy
from teleportcache.types import SimConfig
from tests.prediction.conftest import make_block, make_view


def _workload_dicts():
    """Blocks of two agents with distinct sizes for packing assertions."""
    blocks = {
        "b_small": make_block("b_small", size_bytes=250_000),   # 0.0012 s
        "b_mid": make_block("b_mid", size_bytes=500_000),       # 0.0014 s
        "b_large": make_block("b_large", size_bytes=1_000_000), # 0.0018 s
        "b_huge": make_block("b_huge", size_bytes=1_000_000_000),  # 0.801 s
    }
    return blocks


class TestOraclePlanPrefetch:
    def test_requests_exactly_needed_not_resident_blocks(self) -> None:
        blocks = _workload_dicts()
        from teleportcache.types import HandoffEvent, Workload

        workload = Workload(
            name="w",
            agents=["A", "B"],
            blocks=blocks,
            events=[HandoffEvent(0, "A", "B", ("b_small", "b_mid", "b_large"))],
        )
        oracle = OraclePrefetchStrategy(workload)
        view = make_view(step=0, current_agent="A", blocks=blocks)
        requests = oracle.plan_prefetch(view)
        assert {req.block_id for req in requests} == {"b_small", "b_mid", "b_large"}
        assert all(req.target_agent == "B" for req in requests)

    def test_skips_resident_blocks(self) -> None:
        blocks = _workload_dicts()
        from teleportcache.types import HandoffEvent, Workload

        workload = Workload(
            name="w",
            agents=["A", "B"],
            blocks=blocks,
            events=[HandoffEvent(0, "A", "B", ("b_small", "b_mid", "b_large"))],
        )
        oracle = OraclePrefetchStrategy(workload)
        view = make_view(
            step=0,
            current_agent="A",
            cached={"A": (), "B": ("b_mid",)},
            blocks=blocks,
        )
        assert {req.block_id for req in oracle.plan_prefetch(view)} == {"b_small", "b_large"}

    def test_packs_smallest_first_within_window(self) -> None:
        blocks = _workload_dicts()
        from teleportcache.types import HandoffEvent, Workload

        workload = Workload(
            name="w",
            agents=["A", "B"],
            blocks=blocks,
            events=[HandoffEvent(0, "A", "B", ("b_small", "b_mid", "b_large", "b_huge"))],
        )
        oracle = OraclePrefetchStrategy(workload)
        # Window 0.0030s: small (0.0012) + mid (0.0014) = 0.0026 fits; large would exceed.
        config = SimConfig(capacity_blocks=8, prefetch_window_s=0.0030)
        view = make_view(step=0, current_agent="A", config=config, blocks=blocks)
        requests = oracle.plan_prefetch(view)
        assert [req.block_id for req in requests] == ["b_small", "b_mid"]
        assert sum(0.001 + blocks[r.block_id].size_bytes / 1.25e9 for r in requests) <= 0.0030

    def test_capacity_cap(self) -> None:
        blocks = _workload_dicts()
        from teleportcache.types import HandoffEvent, Workload

        workload = Workload(
            name="w",
            agents=["A", "B"],
            blocks=blocks,
            events=[HandoffEvent(0, "A", "B", ("b_small", "b_mid", "b_large"))],
        )
        oracle = OraclePrefetchStrategy(workload)
        config = SimConfig(capacity_blocks=2)
        view = make_view(step=0, current_agent="A", config=config, blocks=blocks)
        assert len(oracle.plan_prefetch(view)) <= 2

    def test_step_out_of_range_returns_empty(self) -> None:
        blocks = _workload_dicts()
        from teleportcache.types import HandoffEvent, Workload

        workload = Workload(
            name="w",
            agents=["A", "B"],
            blocks=blocks,
            events=[HandoffEvent(0, "A", "B", ("b_small",))],
        )
        oracle = OraclePrefetchStrategy(workload)
        view = make_view(step=5, current_agent="A", blocks=blocks)
        assert oracle.plan_prefetch(view) == []

    def test_protocol_attributes_and_noop_lifecycle(self) -> None:
        blocks = _workload_dicts()
        from teleportcache.types import HandoffEvent, Workload

        workload = Workload(
            name="w",
            agents=["A", "B"],
            blocks=blocks,
            events=[HandoffEvent(0, "A", "B", ("b_small",))],
        )
        oracle = OraclePrefetchStrategy(workload)
        assert oracle.name == "oracle"
        assert oracle.use_cache is True
        assert oracle.miss_handling == "transfer"
        oracle.observe(workload.events[0])
        oracle.reset()  # no state; must be safe no-ops

    def test_is_documented_upper_bound(self) -> None:
        import teleportcache.prediction.oracle as oracle_module

        assert "upper bound" in oracle_module.__doc__.lower()
