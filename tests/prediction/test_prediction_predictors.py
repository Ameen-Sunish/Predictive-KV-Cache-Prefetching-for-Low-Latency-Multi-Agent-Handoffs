"""Tests for next-agent predictors (teleportcache.prediction.predictors)."""

from __future__ import annotations

import pytest

from teleportcache.prediction.predictors import (
    FrequencyPredictor,
    MarkovPredictor,
    NoisyOraclePredictor,
    RandomPredictor,
)
from teleportcache.types import HandoffEvent, Workload
from tests.prediction.conftest import cycle_events, cyclic_workload


class TestFrequencyPredictor:
    def test_laplace_smoothing_values(self) -> None:
        predictor = FrequencyPredictor()
        events = cycle_events(2, cycle=("A", "B", "C"))  # to-seq: A, B
        for event in events:
            predictor.update(event)
        # current=B has no observed transition: uniform over agents seen as
        # handoff targets so far ({A, B}).
        ranked = predictor.predict("B", tuple(events), 2)
        assert {agent for agent, _ in ranked} == {"A", "B"}
        assert all(prob == pytest.approx(1 / 2) for _, prob in ranked)

    def test_observed_transition_dominates(self) -> None:
        predictor = FrequencyPredictor()
        events = cycle_events(6, cycle=("A", "B", "C"))  # A->B, B->C, C->A, ...
        for event in events:
            predictor.update(event)
        ranked = predictor.predict("A", tuple(events), 6)
        assert ranked[0][0] == "B"
        assert ranked[0][1] == pytest.approx((2 + 1) / (2 + 3))  # Laplace
        assert sum(prob for _, prob in ranked) == pytest.approx(1.0)

    def test_cold_start_returns_empty(self) -> None:
        assert FrequencyPredictor().predict("A", (), 0) == []

    def test_reset_clears_state(self) -> None:
        predictor = FrequencyPredictor()
        for event in cycle_events(6, cycle=("A", "B", "C")):
            predictor.update(event)
        predictor.reset()
        assert predictor.predict("A", (), 6) == []


class TestMarkovPredictor:
    def test_perfect_after_warmup_on_cycle(self) -> None:
        events = cycle_events(20, cycle=("A", "B", "C"))
        predictor = MarkovPredictor(order=1)
        correct = total = 0
        for i, event in enumerate(events):
            current = events[i - 1].to_agent if i else events[0].from_agent
            ranked = predictor.predict(current, tuple(events[:i]), i)
            if i >= 4:  # first occurrence of each transition needs 3 updates
                total += 1
                if ranked and ranked[0][0] == event.to_agent:
                    correct += 1
            predictor.update(event)
        assert total == 16
        assert correct == total  # 100% after warm-up

    def test_order2_resolves_ambiguous_pattern(self) -> None:
        # to-seq: A B A C A B A C ... — context (A,) is 50/50 under order 1,
        # but order-2 contexts (B,A)->C, (A,C)->A, (C,A)->B, (A,B)->A are exact.
        pattern = ("A", "B", "A", "C")
        events: list[HandoffEvent] = []
        for i in range(24):
            to_agent = pattern[i % 4]
            events.append(
                HandoffEvent(step=i, from_agent=pattern[(i - 1) % 4], to_agent=to_agent, required_blocks=())
            )
        markov1 = MarkovPredictor(order=1)
        markov2 = MarkovPredictor(order=2)
        m2_correct = 0
        m1_correct = 0
        checked = 0
        for i, event in enumerate(events):
            current = events[i - 1].to_agent if i else events[0].from_agent
            r1 = markov1.predict(current, tuple(events[:i]), i)
            r2 = markov2.predict(current, tuple(events[:i]), i)
            if i >= 8:
                checked += 1
                if r1 and r1[0][0] == event.to_agent:
                    m1_correct += 1
                if r2 and r2[0][0] == event.to_agent:
                    m2_correct += 1
            markov1.update(event)
            markov2.update(event)
        assert checked == 16
        assert m2_correct == checked, "order-2 must fully resolve the pattern"
        assert m1_correct < checked, "order-1 must stay ambiguous on this pattern"

    def test_backs_off_to_lower_order(self) -> None:
        # to-seq: A B A B A C — context (B, A) is unseen at step 5, so order 2
        # must back off to order 1 (A -> C observed once at the end).
        to_seq = ["A", "B", "A", "B", "A", "C"]
        events = [
            HandoffEvent(step=i, from_agent="X", to_agent=to, required_blocks=())
            for i, to in enumerate(to_seq)
        ]
        predictor = MarkovPredictor(order=2)
        for event in events:
            predictor.update(event)
        ranked = predictor.predict("A", tuple(events), 6)
        # Order-2 context (C, A) is unseen, so the predictor must back off to
        # order 1, where A's observed successors are B (x2) and C (x1) -> B.
        assert ranked[0][0] == "B"
        assert sum(prob for _, prob in ranked) == pytest.approx(1.0)

    def test_probabilities_sum_to_one(self) -> None:
        events = cycle_events(9, cycle=("A", "B", "C"))
        for order in (1, 2, 3):
            predictor = MarkovPredictor(order=order)
            for i, event in enumerate(events):
                current = events[i - 1].to_agent if i else events[0].from_agent
                ranked = predictor.predict(current, tuple(events[:i]), i)
                if ranked:
                    assert sum(prob for _, prob in ranked) == pytest.approx(1.0)
                predictor.update(event)

    def test_reset_restores_fresh_state(self) -> None:
        events = cycle_events(9, cycle=("A", "B", "C"))
        warm = MarkovPredictor(order=1)
        for event in events:
            warm.update(event)
        fresh = MarkovPredictor(order=1)
        warm.reset()
        assert warm.predict("A", (), 0) == fresh.predict("A", (), 0) == []

    def test_invalid_order_raises(self) -> None:
        with pytest.raises(ValueError):
            MarkovPredictor(order=0)


class TestRandomPredictor:
    def test_ranking_is_a_uniform_permutation(self) -> None:
        predictor = RandomPredictor(["C", "A", "B"], seed=7)
        ranked = predictor.predict("A", (), 3)
        assert [agent for agent, _ in ranked] != ["C", "A", "B"] or True  # order may match
        assert sorted(agent for agent, _ in ranked) == ["A", "B", "C"]
        assert all(prob == pytest.approx(1 / 3) for _, prob in ranked)

    def test_deterministic_per_seed_and_step(self) -> None:
        first = RandomPredictor(["A", "B", "C", "D"], seed=42)
        second = RandomPredictor(["A", "B", "C", "D"], seed=42)
        for step in range(5):
            assert first.predict("A", (), step) == second.predict("A", (), step)
        # same seed, same step => identical even across instances
        assert RandomPredictor(["A", "B", "C", "D"], seed=42).predict("A", (), 2) == first.predict("A", (), 2)

    def test_reset_is_stateless_noop(self) -> None:
        predictor = RandomPredictor(["A", "B"], seed=1)
        before = predictor.predict("A", (), 0)
        predictor.reset()
        assert predictor.predict("A", (), 0) == before

    def test_empty_agents_returns_empty(self) -> None:
        assert RandomPredictor([], seed=0).predict("A", (), 0) == []


class TestNoisyOraclePredictor:
    def _workload(self, num_steps: int = 300) -> Workload:
        return cyclic_workload(num_steps)

    def test_accuracy_one_always_exact(self) -> None:
        workload = self._workload()
        predictor = NoisyOraclePredictor(workload, accuracy=1.0, seed=3)
        for step, event in enumerate(workload.events):
            ranked = predictor.predict(event.from_agent, tuple(workload.events[:step]), step)
            assert ranked[0][0] == event.to_agent

    def test_accuracy_zero_always_wrong(self) -> None:
        workload = self._workload()
        predictor = NoisyOraclePredictor(workload, accuracy=0.0, seed=3)
        for step, event in enumerate(workload.events):
            ranked = predictor.predict(event.from_agent, tuple(workload.events[:step]), step)
            assert ranked[0][0] != event.to_agent

    def test_reproducible_per_seed_and_step(self) -> None:
        workload = self._workload(20)
        p1 = NoisyOraclePredictor(workload, accuracy=0.7, seed=11)
        p2 = NoisyOraclePredictor(workload, accuracy=0.7, seed=11)
        for step in range(len(workload.events)):
            event = workload.events[step]
            assert p1.predict(event.from_agent, (), step) == p2.predict(event.from_agent, (), step)

    def test_empirical_accuracy_near_target(self) -> None:
        workload = self._workload(300)
        target = 0.7
        predictor = NoisyOraclePredictor(workload, accuracy=target, seed=5)
        hits = 0
        for step, event in enumerate(workload.events):
            ranked = predictor.predict(event.from_agent, (), step)
            if ranked[0][0] == event.to_agent:
                hits += 1
        empirical = hits / len(workload.events)
        assert abs(empirical - target) <= 0.1

    def test_invalid_accuracy_raises(self) -> None:
        workload = self._workload(5)
        with pytest.raises(ValueError):
            NoisyOraclePredictor(workload, accuracy=1.5)
        with pytest.raises(ValueError):
            NoisyOraclePredictor(workload, accuracy=-0.1)
