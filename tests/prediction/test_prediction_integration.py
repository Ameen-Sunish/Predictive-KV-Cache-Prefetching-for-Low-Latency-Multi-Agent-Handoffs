"""Integration tests: predictors + strategy + Eldo's Simulator.

Skipped with a clear reason until ``teleportcache.core.simulator`` is
available on main. Uses an inline cyclic workload (no dependency on Ameen's
generators).
"""

from __future__ import annotations

import pytest

simulator_module = pytest.importorskip(
    "teleportcache.core.simulator",
    reason="Eldo's simulator (teleportcache.core.simulator) is not available yet",
)
baselines_module = pytest.importorskip(
    "teleportcache.core.baselines",
    reason="Eldo's baselines (teleportcache.core.baselines) are not available yet",
)

from teleportcache.core.baselines import RecomputeStrategy, TransferOnDemandStrategy
from teleportcache.core.simulator import Simulator
from teleportcache.prediction.oracle import OraclePrefetchStrategy
from teleportcache.prediction.prefetch_strategy import PredictivePrefetchStrategy
from teleportcache.prediction.predictors import MarkovPredictor
from tests.prediction.conftest import SIM_CONFIG, cyclic_workload

NUM_STEPS = 30


def _total_latency(handoffs) -> float:
    return sum(h.latency_s for h in handoffs)


def _total_misses(handoffs) -> int:
    return sum(h.misses for h in handoffs)


@pytest.fixture(scope="module")
def workload():
    return cyclic_workload(num_steps=NUM_STEPS)


@pytest.fixture(scope="module")
def results(workload):
    simulator = Simulator(SIM_CONFIG)
    strategies = {
        "predictive": PredictivePrefetchStrategy(MarkovPredictor(order=1)),
        "on_demand": TransferOnDemandStrategy(),
        "recompute": RecomputeStrategy(),
        "oracle": OraclePrefetchStrategy(workload),
    }
    return {name: simulator.run(workload, strategy) for name, strategy in strategies.items()}


class TestPredictiveBeatsBaselines:
    def test_predictive_lower_latency_than_on_demand(self, results) -> None:
        predictive = _total_latency(results["predictive"].handoffs)
        on_demand = _total_latency(results["on_demand"].handoffs)
        assert predictive < on_demand, (
            f"predictive ({predictive:.4f}s) must beat on-demand transfer ({on_demand:.4f}s)"
        )

    def test_predictive_lower_latency_than_recompute(self, results) -> None:
        predictive = _total_latency(results["predictive"].handoffs)
        recompute = _total_latency(results["recompute"].handoffs)
        assert predictive < recompute

    def test_on_demand_lower_latency_than_recompute(self, results) -> None:
        assert _total_latency(results["on_demand"].handoffs) < _total_latency(
            results["recompute"].handoffs
        )

    def test_oracle_is_upper_bound(self, results) -> None:
        oracle = _total_latency(results["oracle"].handoffs)
        predictive = _total_latency(results["predictive"].handoffs)
        assert oracle <= predictive

    def test_predictive_misses_fewer_blocks_than_on_demand(self, results) -> None:
        assert _total_misses(results["predictive"].handoffs) < _total_misses(
            results["on_demand"].handoffs
        )

    def test_warmup_then_all_hits(self, results) -> None:
        """Deterministic cycle: after a short warm-up every handoff hits."""
        handoffs = results["predictive"].handoffs
        assert _total_misses(handoffs) == 24  # 6 warm-up steps x 4 blocks
        assert all(h.misses == 0 for h in handoffs[6:])

    def test_oracle_never_misses(self, results) -> None:
        assert _total_misses(results["oracle"].handoffs) == 0
        assert _total_latency(results["oracle"].handoffs) == pytest.approx(0.0)


class TestPredictionStats:
    def test_accuracy_stats_and_extras(self, results) -> None:
        run = results["predictive"]
        stats_expected_keys = {"prediction_accuracy_top1", "prediction_accuracy_topk", "prefetch_requests"}
        assert stats_expected_keys <= set(run.extras)
        # Markov(1) on a deterministic 3-cycle: perfect once warm (steps >= 4);
        # steps 1-3 are scored and wrong, step 0 has an empty (unscored) ranking.
        assert run.extras["prediction_accuracy_top1"] == pytest.approx((NUM_STEPS - 4) / (NUM_STEPS - 1))
        assert run.extras["prediction_accuracy_top1"] > 0.8
        assert run.extras["prefetch_requests"] > 0

    def test_prefetches_actually_used(self, results) -> None:
        run = results["predictive"]
        useful = sum(h.prefetch_useful_blocks for h in run.handoffs)
        assert useful > 0, "prefetched blocks must be hit in a deterministic cycle"


class TestDeterminism:
    def test_same_run_twice_identical(self, workload) -> None:
        simulator = Simulator(SIM_CONFIG)
        run1 = simulator.run(workload, PredictivePrefetchStrategy(MarkovPredictor(order=1)))
        run2 = simulator.run(workload, PredictivePrefetchStrategy(MarkovPredictor(order=1)))
        assert _total_latency(run1.handoffs) == _total_latency(run2.handoffs)
        assert [h.misses for h in run1.handoffs] == [h.misses for h in run2.handoffs]
        assert run1.extras == run2.extras
