"""Tests for PredictivePrefetchStrategy with hand-built PolicyView snapshots."""

from __future__ import annotations

import pytest

from teleportcache.prediction.block_selector import BlockSelector
from teleportcache.prediction.prefetch_strategy import PredictivePrefetchStrategy
from teleportcache.prediction.predictors import MarkovPredictor
from teleportcache.types import HandoffEvent, PolicyView, PrefetchRequest, SimConfig
from tests.prediction.conftest import make_block, make_view

BLOCKS = {
    "b1": make_block("b1"),
    "b2": make_block("b2"),
    "b3": make_block("b3"),
    "c1": make_block("c1"),
    "c2": make_block("c2"),
    "huge": make_block("huge", num_tokens=512, size_bytes=1_000_000_000),
}


class FixedPredictor:
    """Stub predictor always returning a fixed ranking (for deterministic planning)."""

    def __init__(self, ranking: list[tuple[str, float]]) -> None:
        self.ranking = ranking
        self.name = f"fixed({','.join(a for a, _ in ranking)})"
        self.updates: list[HandoffEvent] = []

    def predict(self, current_agent, history, step):
        return list(self.ranking)

    def update(self, event) -> None:
        self.updates.append(event)

    def reset(self) -> None:
        self.updates.clear()


class TestPlanPrefetch:
    def _warmed_strategy(self, ranking, **kwargs) -> tuple[PredictivePrefetchStrategy, FixedPredictor]:
        strategy = PredictivePrefetchStrategy(FixedPredictor(ranking), **kwargs)
        # Teach the selector which blocks B and C tend to require.
        strategy.observe(HandoffEvent(0, "A", "B", ("b1", "b2", "b3")))
        strategy.observe(HandoffEvent(1, "B", "C", ("c1", "c2")))
        return strategy, strategy.predictor

    def test_excludes_already_cached_blocks(self) -> None:
        strategy, _ = self._warmed_strategy([("B", 1.0)])
        view = make_view(
            step=2,
            current_agent="A",
            cached={"A": (), "B": ("b1", "b2", "b3"), "C": ()},
            blocks=BLOCKS,
        )
        assert strategy.plan_prefetch(view) == []

    def test_budget_cap_limits_requests(self) -> None:
        # Window 0.002s fits only one 0.0018s block transfer.
        config = SimConfig(
            capacity_blocks=8,
            transfer_latency_s=0.001,
            bandwidth_bytes_per_s=1.25e9,
            prefetch_window_s=0.002,
        )
        strategy, _ = self._warmed_strategy([("B", 1.0)])
        view = make_view(step=2, current_agent="A", config=config, blocks=BLOCKS)
        requests = strategy.plan_prefetch(view)
        assert len(requests) == 1
        assert requests[0].target_agent == "B"

    def test_individually_oversized_block_skipped(self) -> None:
        strategy, _ = self._warmed_strategy([("B", 1.0)])
        blocks = dict(BLOCKS, huge2=make_block("huge2", size_bytes=1_000_000_000))
        selector_block = HandoffEvent(1, "B", "B", ("huge2",))
        strategy.observe(selector_block)  # teach selector the huge block too
        view = make_view(step=2, current_agent="A", blocks=blocks)
        targets = {req.block_id for req in strategy.plan_prefetch(view)}
        assert "huge2" not in targets
        assert "huge" not in targets

    def test_capacity_cap_limits_requests(self) -> None:
        strategy, _ = self._warmed_strategy([("B", 1.0)])
        config = SimConfig(capacity_blocks=2)  # defaults: window 0.05 fits ~27 blocks
        view = make_view(step=2, current_agent="A", config=config, blocks=BLOCKS)
        requests = strategy.plan_prefetch(view)
        assert len(requests) <= 2
        assert all(req.target_agent == "B" for req in requests)

    def test_max_prefetch_blocks_cap(self) -> None:
        strategy, _ = self._warmed_strategy([("B", 1.0)], max_prefetch_blocks=1)
        view = make_view(step=2, current_agent="A", blocks=BLOCKS)
        requests = strategy.plan_prefetch(view)
        assert len(requests) == 1

    def test_confidence_scales_prefetch_budget(self) -> None:
        # Doc section 6: window budget x confidence.
        # Budget = floor(0.05 / 0.0018) = 27 blocks; selector knows b1..b3.
        high, _ = self._warmed_strategy([("B", 1.0)])
        low, _ = self._warmed_strategy([("B", 0.1)])
        view = make_view(step=2, current_agent="A", blocks=BLOCKS)
        assert len(high.plan_prefetch(view)) == 3   # 27 x 1.0 -> all known blocks
        assert len(low.plan_prefetch(view)) == 2    # 27 x 0.1 -> floor(2.7) = 2 blocks

    def test_top_k_split_proportional_to_probability(self) -> None:
        # min(window budget 27, max_prefetch_blocks 5) = 5 blocks.
        # p(B)=0.6, p(C)=0.4 -> quotas 3 and 2 (largest remainder).
        strategy, _ = self._warmed_strategy(
            [("B", 0.6), ("C", 0.4)], top_k_agents=2, max_prefetch_blocks=5
        )
        view = make_view(step=2, current_agent="A", blocks=BLOCKS)
        requests = strategy.plan_prefetch(view)
        targets = [req.target_agent for req in requests]
        assert len(requests) == 5
        assert set(targets) == {"B", "C"}
        assert targets.count("B") == 3
        assert targets.count("C") == 2

    def test_merged_ordering_prefers_high_probability_high_score(self) -> None:
        strategy, _ = self._warmed_strategy([("B", 0.9), ("C", 0.1)], top_k_agents=2)
        view = make_view(step=2, current_agent="A", blocks=BLOCKS)
        requests = strategy.plan_prefetch(view)
        assert requests[0].target_agent == "B"  # 0.9 * score_B > 0.1 * score_C

    def test_no_requests_without_prediction(self) -> None:
        strategy = PredictivePrefetchStrategy(MarkovPredictor(order=1))
        view = make_view(step=0, current_agent="A", blocks=BLOCKS)
        assert strategy.plan_prefetch(view) == []

    def test_never_peeks_into_the_future(self) -> None:
        strategy = PredictivePrefetchStrategy(MarkovPredictor(order=1))
        events = [
            HandoffEvent(0, "A", "B", ("b1",)),
            HandoffEvent(1, "B", "C", ("c1",)),
            HandoffEvent(2, "C", "A", ("b2",)),
        ]
        for event in events:
            strategy.observe(event)
        view = make_view(step=3, current_agent="A", history=tuple(events), blocks=BLOCKS)
        # Planning must not score or learn: stats untouched, no double update.
        requests = strategy.plan_prefetch(view)
        stats = strategy.stats()
        assert stats["prediction_accuracy_top1"] == 0.0
        assert stats["prefetch_requests"] == len(requests)
        assert strategy.predictor._seq == ["B", "C", "A"]  # unchanged by plan_prefetch

    def test_deterministic_given_same_view(self) -> None:
        strategy, _ = self._warmed_strategy([("B", 1.0)])
        view = make_view(step=2, current_agent="A", blocks=BLOCKS)
        assert strategy.plan_prefetch(view) == strategy.plan_prefetch(view)


class TestObserveAndStats:
    def test_observe_scores_then_learns(self) -> None:
        predictor = MarkovPredictor(order=1)
        strategy = PredictivePrefetchStrategy(predictor)
        events = [
            HandoffEvent(0, "A", "B", ("b1",)),
            HandoffEvent(1, "B", "C", ("c1",)),
            HandoffEvent(2, "C", "A", ("b2",)),
            HandoffEvent(3, "A", "B", ("b1",)),
            HandoffEvent(4, "B", "C", ("c1",)),
            HandoffEvent(5, "C", "A", ("b2",)),
            HandoffEvent(6, "A", "B", ("b1",)),
        ]
        for event in events:
            strategy.plan_prefetch(make_view(step=event.step, current_agent=event.from_agent, blocks=BLOCKS))
            strategy.observe(event)
        stats = strategy.stats()
        # Steps 1-3 are scored but wrong (cold start); steps 4-6 all hit.
        assert stats["prediction_accuracy_top1"] == pytest.approx(0.5)
        assert stats["prediction_accuracy_topk"] == pytest.approx(0.5)
        # predictor saw every event exactly once, after scoring
        assert predictor._seq == ["B", "C", "A", "B", "C", "A", "B"]

    def test_observe_without_plan_does_not_score(self) -> None:
        strategy = PredictivePrefetchStrategy(MarkovPredictor(order=1))
        strategy.observe(HandoffEvent(0, "A", "B", ("b1",)))
        stats = strategy.stats()
        assert stats["prediction_accuracy_top1"] == 0.0
        assert stats["prefetch_requests"] == 0

    def test_reset_clears_everything(self) -> None:
        predictor = MarkovPredictor(order=1)
        selector = BlockSelector()
        strategy = PredictivePrefetchStrategy(predictor, selector=selector)
        strategy.plan_prefetch(make_view(step=0, current_agent="A", blocks=BLOCKS))
        strategy.observe(HandoffEvent(0, "A", "B", ("b1",)))
        strategy.reset()
        assert strategy.stats() == {
            "prediction_accuracy_top1": 0.0,
            "prediction_accuracy_topk": 0.0,
            "prefetch_requests": 0,
        }
        assert predictor._seq == []
        assert selector.rank("B") == []
        # post-reset planning behaves like a fresh strategy
        assert strategy.plan_prefetch(make_view(step=0, current_agent="A", blocks=BLOCKS)) == []

    def test_auto_generated_name(self) -> None:
        assert PredictivePrefetchStrategy(MarkovPredictor(order=1)).name == "predictive(markov1,k=1)"
        assert PredictivePrefetchStrategy(MarkovPredictor(order=2), top_k_agents=3).name == (
            "predictive(markov2,k=3)"
        )
        assert PredictivePrefetchStrategy(MarkovPredictor(order=1), name="custom").name == "custom"

    def test_protocol_attributes(self) -> None:
        strategy = PredictivePrefetchStrategy(MarkovPredictor(order=1))
        assert strategy.use_cache is True
        assert strategy.miss_handling == "transfer"
        assert isinstance(strategy.plan_prefetch(make_view(step=0, current_agent="A")), list)

    def test_invalid_args_raise(self) -> None:
        with pytest.raises(ValueError):
            PredictivePrefetchStrategy(MarkovPredictor(order=1), top_k_agents=0)
        with pytest.raises(ValueError):
            PredictivePrefetchStrategy(MarkovPredictor(order=1), max_prefetch_blocks=-1)
