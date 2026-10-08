"""Unit and integration tests for Simulator and end-to-end handoff execution."""

from __future__ import annotations

from dataclasses import dataclass, field
import pytest

from teleportcache.core.baselines import RecomputeStrategy, TransferOnDemandStrategy
from teleportcache.core.simulator import Simulator
from teleportcache.types import (
    AgentId,
    BlockId,
    HandoffEvent,
    HandoffResult,
    KVBlock,
    PolicyView,
    PrefetchRequest,
    SimConfig,
    Workload,
)


def make_sample_workload() -> Workload:
    """Creates a deterministic 3-event workload with 2 agents and 3 blocks.
    
    Block properties:
    - size: 1,250,000 bytes (1.25 MB)
    - num_tokens: 1000
    - Bandwidth: 1.25e9 B/s => transfer time = 0.001 + (1.25e6 / 1.25e9) = 0.002s
    - Recompute time: 1000 tokens * 0.001 s/token = 1.0s
    """
    blocks = {
        "b1": KVBlock(block_id="b1", num_tokens=1000, size_bytes=1_250_000),
        "b2": KVBlock(block_id="b2", num_tokens=1000, size_bytes=1_250_000),
        "b3": KVBlock(block_id="b3", num_tokens=1000, size_bytes=1_250_000),
    }
    events = [
        HandoffEvent(step=0, from_agent="A", to_agent="B", required_blocks=("b1", "b2")),
        HandoffEvent(step=1, from_agent="B", to_agent="B", required_blocks=("b2", "b3")),
        HandoffEvent(step=2, from_agent="B", to_agent="B", required_blocks=("b1", "b3")),
    ]
    return Workload(
        name="test_3_event",
        agents=["A", "B"],
        blocks=blocks,
        events=events,
    )


def test_simulator_hand_computed_3_event_workload() -> None:
    workload = make_sample_workload()
    # Cache capacity = 2, LRU policy
    config = SimConfig(
        capacity_blocks=2,
        bandwidth_bytes_per_s=1.25e9,
        transfer_latency_s=0.001,
        recompute_time_per_token_s=0.001,
        prefetch_window_s=0.05,
        eviction_policy="lru",
    )
    simulator = Simulator(config)
    strategy = TransferOnDemandStrategy()
    
    result = simulator.run(workload, strategy)
    
    assert result.workload_name == "test_3_event"
    assert result.strategy_name == "transfer_on_demand"
    assert len(result.handoffs) == 3
    
    # Event 0: B requires (b1, b2). B cache is empty.
    # Miss b1: 0.002s, miss b2: 0.002s -> Latency: 0.004s. Hits: 0, Misses: 2.
    # Cache B after step 0: (b1, b2)
    h0 = result.handoffs[0]
    assert h0.step == 0
    assert h0.hits == 0
    assert h0.misses == 2
    assert pytest.approx(h0.latency_s, rel=1e-6) == 0.004
    assert pytest.approx(h0.bytes_transferred, rel=1e-6) == 2_500_000.0
    assert h0.cache_occupancy == 1.0

    # Event 1: B requires (b2, b3).
    # Hit b2 (0s), Miss b3 (0.002s) -> Latency: 0.002s. Hits: 1, Misses: 1.
    # Insertion of b3 evicts b1 (b2 was touched!).
    # Cache B after step 1: (b2, b3)
    h1 = result.handoffs[1]
    assert h1.step == 1
    assert h1.hits == 1
    assert h1.misses == 1
    assert pytest.approx(h1.latency_s, rel=1e-6) == 0.002
    assert pytest.approx(h1.bytes_transferred, rel=1e-6) == 1_250_000.0
    assert h1.cache_occupancy == 1.0

    # Event 2: B requires (b1, b3).
    # Miss b1 (0.002s, was evicted!), Hit b3 (0s). Latency: 0.002s. Hits: 1, Misses: 1.
    # Insertion of b1 evicts b2.
    # Cache B after step 2: (b3, b1)
    h2 = result.handoffs[2]
    assert h2.step == 2
    assert h2.hits == 1
    assert h2.misses == 1
    assert pytest.approx(h2.latency_s, rel=1e-6) == 0.002
    assert pytest.approx(h2.bytes_transferred, rel=1e-6) == 1_250_000.0


def test_recompute_baseline_never_hits() -> None:
    workload = make_sample_workload()
    config = SimConfig(
        capacity_blocks=2,
        recompute_time_per_token_s=0.001,
    )
    simulator = Simulator(config)
    strategy = RecomputeStrategy()
    
    result = simulator.run(workload, strategy)
    
    # In recompute baseline:
    # Event 0: 2 blocks * 1000 tokens * 0.001s = 2.0s
    # Event 1: 2 blocks * 1000 tokens * 0.001s = 2.0s
    # Event 2: 2 blocks * 1000 tokens * 0.001s = 2.0s
    for h in result.handoffs:
        assert h.hits == 0
        assert h.misses == 2
        assert pytest.approx(h.latency_s, rel=1e-6) == 2.0
        assert pytest.approx(h.recompute_time_s, rel=1e-6) == 2.0
        assert h.bytes_transferred == 0.0
        assert h.cache_occupancy == 0.0


def test_prefetch_window_truncation() -> None:
    """Test that prefetch requests exceeding prefetch_window_s are truncated/dropped."""
    blocks = {
        f"b{i}": KVBlock(block_id=f"b{i}", num_tokens=100, size_bytes=10_000_000)
        for i in range(10)
    }
    # transfer_time per block: 0.001 + (10MB / 1GB/s) = 0.001 + 0.010 = 0.011s
    # window = 0.025s -> exactly 2 blocks fit (2 * 0.011 = 0.022s; 3 * 0.011 = 0.033s > 0.025s)
    config = SimConfig(
        capacity_blocks=10,
        bandwidth_bytes_per_s=1.0e9,
        transfer_latency_s=0.001,
        prefetch_window_s=0.025,
    )

    class OvereagerPrefetchStrategy:
        name = "overeager"
        use_cache = True
        miss_handling = "transfer"

        def plan_prefetch(self, view: PolicyView) -> list[PrefetchRequest]:
            if view.step == 0:
                # Return 5 prefetch requests for agent B
                return [PrefetchRequest(target_agent="B", block_id=f"b{i}") for i in range(5)]
            return []

        def observe(self, event: HandoffEvent) -> None:
            pass

        def reset(self) -> None:
            pass

    workload = Workload(
        name="prefetch_test",
        agents=["A", "B"],
        blocks=blocks,
        events=[
            HandoffEvent(step=0, from_agent="A", to_agent="B", required_blocks=("b0", "b1", "b2")),
        ],
    )

    simulator = Simulator(config)
    result = simulator.run(workload, OvereagerPrefetchStrategy())
    h0 = result.handoffs[0]

    # Only b0 and b1 should have been prefetched
    assert h0.prefetch_issued_blocks == 2
    assert pytest.approx(h0.prefetch_bytes, rel=1e-6) == 20_000_000.0
    # b0 and b1 hit from prefetch (useful); b2 misses on demand
    assert h0.hits == 2
    assert h0.misses == 1
    assert h0.prefetch_useful_blocks == 2


def test_useful_vs_wasted_prefetch_accounting() -> None:
    """Test accounting of useful vs wasted prefetches."""
    blocks = {
        "b1": KVBlock(block_id="b1", num_tokens=100, size_bytes=1000),
        "b2": KVBlock(block_id="b2", num_tokens=100, size_bytes=1000),
        "b3": KVBlock(block_id="b3", num_tokens=100, size_bytes=1000),
    }

    class MockPrefetchStrategy:
        name = "mock_prefetch"
        use_cache = True
        miss_handling = "transfer"

        def plan_prefetch(self, view: PolicyView) -> list[PrefetchRequest]:
            if view.step == 0:
                # Prefetch b1 (which will be hit) and b2 (which will never be hit)
                return [
                    PrefetchRequest(target_agent="B", block_id="b1"),
                    PrefetchRequest(target_agent="B", block_id="b2"),
                ]
            return []

        def observe(self, event: HandoffEvent) -> None:
            pass

        def reset(self) -> None:
            pass

    workload = Workload(
        name="accounting_test",
        agents=["A", "B"],
        blocks=blocks,
        events=[
            # Step 0: accesses b1 (useful prefetch). b2 stays unused in cache.
            HandoffEvent(step=0, from_agent="A", to_agent="B", required_blocks=("b1",)),
        ],
    )

    config = SimConfig(capacity_blocks=5, prefetch_window_s=1.0)
    simulator = Simulator(config)
    result = simulator.run(workload, MockPrefetchStrategy())

    h0 = result.handoffs[0]
    assert h0.prefetch_issued_blocks == 2
    assert h0.prefetch_useful_blocks == 1
    # b2 was never hit and remained resident at end of simulation -> attributed as wasted to final step
    assert h0.prefetch_wasted_blocks == 1


def test_simulator_determinism() -> None:
    workload = make_sample_workload()
    config = SimConfig(capacity_blocks=2)
    sim = Simulator(config)
    
    res1 = sim.run(workload, TransferOnDemandStrategy())
    res2 = sim.run(workload, TransferOnDemandStrategy())
    
    assert res1.workload_name == res2.workload_name
    assert res1.strategy_name == res2.strategy_name
    assert len(res1.handoffs) == len(res2.handoffs)
    for h1, h2 in zip(res1.handoffs, res2.handoffs):
        assert h1 == h2


def test_strategy_never_sees_upcoming_event() -> None:
    workload = make_sample_workload()

    class InspectionStrategy:
        name = "inspector"
        use_cache = True
        miss_handling = "transfer"

        def __init__(self) -> None:
            self.history_lens: list[int] = []

        def plan_prefetch(self, view: PolicyView) -> list[PrefetchRequest]:
            # At step i, view.history must contain strictly i completed events
            assert len(view.history) == view.step
            self.history_lens.append(len(view.history))
            return []

        def observe(self, event: HandoffEvent) -> None:
            pass

        def reset(self) -> None:
            self.history_lens.clear()

    inspector = InspectionStrategy()
    sim = Simulator(SimConfig())
    sim.run(workload, inspector)
    assert inspector.history_lens == [0, 1, 2]


def test_workload_validation_errors() -> None:
    blocks = {"b1": KVBlock(block_id="b1", num_tokens=100, size_bytes=1000)}
    sim = Simulator()

    # Step index mismatch
    bad_step_workload = Workload(
        name="bad_step",
        agents=["A", "B"],
        blocks=blocks,
        events=[HandoffEvent(step=1, from_agent="A", to_agent="B", required_blocks=("b1",))],
    )
    with pytest.raises(ValueError, match="Event step mismatch"):
        sim.run(bad_step_workload, TransferOnDemandStrategy())

    # Unknown agent
    bad_agent_workload = Workload(
        name="bad_agent",
        agents=["A"],
        blocks=blocks,
        events=[HandoffEvent(step=0, from_agent="A", to_agent="UnknownAgent", required_blocks=("b1",))],
    )
    with pytest.raises(ValueError, match="Unknown to_agent"):
        sim.run(bad_agent_workload, TransferOnDemandStrategy())

    # Unknown block
    bad_block_workload = Workload(
        name="bad_block",
        agents=["A", "B"],
        blocks=blocks,
        events=[HandoffEvent(step=0, from_agent="A", to_agent="B", required_blocks=("missing_b",))],
    )
    with pytest.raises(ValueError, match="Unknown required block"):
        sim.run(bad_block_workload, TransferOnDemandStrategy())


def test_strategy_extras_collection() -> None:
    workload = make_sample_workload()

    class StrategyWithStats:
        name = "with_stats"
        use_cache = True
        miss_handling = "transfer"

        def plan_prefetch(self, view: PolicyView) -> list[PrefetchRequest]:
            return []

        def observe(self, event: HandoffEvent) -> None:
            pass

        def reset(self) -> None:
            pass

        def stats(self) -> dict[str, float]:
            return {"prediction_accuracy": 0.88}

    sim = Simulator()
    res = sim.run(workload, StrategyWithStats())
    assert res.extras == {"prediction_accuracy": 0.88}
