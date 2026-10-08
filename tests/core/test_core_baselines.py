"""Unit tests for baseline strategies."""

from teleportcache.core.baselines import RecomputeStrategy, TransferOnDemandStrategy
from teleportcache.types import HandoffEvent, PolicyView, SimConfig, Strategy


def test_baseline_protocols_and_defaults() -> None:
    recompute = RecomputeStrategy()
    assert isinstance(recompute, Strategy)
    assert recompute.name == "recompute"
    assert recompute.use_cache is False
    assert recompute.miss_handling == "recompute"

    demand = TransferOnDemandStrategy()
    assert isinstance(demand, Strategy)
    assert demand.name == "transfer_on_demand"
    assert demand.use_cache is True
    assert demand.miss_handling == "transfer"


def test_baseline_plan_prefetch_and_observe() -> None:
    recompute = RecomputeStrategy()
    demand = TransferOnDemandStrategy()

    dummy_view = PolicyView(
        step=0,
        current_agent="A",
        agents=("A", "B"),
        blocks={},
        config=SimConfig(),
        cached={"A": (), "B": ()},
        history=(),
    )

    assert recompute.plan_prefetch(dummy_view) == []
    assert demand.plan_prefetch(dummy_view) == []

    event = HandoffEvent(step=0, from_agent="A", to_agent="B", required_blocks=())
    # Should execute without errors
    recompute.observe(event)
    demand.observe(event)
    recompute.reset()
    demand.reset()
