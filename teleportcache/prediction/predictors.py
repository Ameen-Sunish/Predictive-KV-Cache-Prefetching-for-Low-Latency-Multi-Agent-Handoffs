"""Next-agent predictors for TeleportCache predictive prefetching.

These are the "working-set oracles" of the OS analogy: they watch the stream of
completed agent handoffs (analogous to observed page-reference strings) and
predict which agent will run next, so the runtime can speculatively page in
(prefetch) the next agent's KV-cache blocks during the handoff window.

All predictors implement the ``Predictor`` protocol from ``teleportcache.types``
and learn ONLY from past events via ``update(event)`` — never from the future.
"""

from __future__ import annotations

import random
from collections import Counter
from typing import Sequence

from teleportcache.types import AgentId, HandoffEvent, Workload

_SEED_STRIDE = 1_000_003  # coprime stride for stable per-(seed, step) RNG streams


class BasePredictor:
    """Shared boilerplate for predictors implementing the Predictor protocol.

    Subclasses must set ``name`` and override :meth:`predict`.
    """

    name: str = "base"

    def predict(
        self,
        current_agent: AgentId,
        history: Sequence[HandoffEvent],
        step: int,
    ) -> list[tuple[AgentId, float]]:
        """Return a ranked best-first list of ``(candidate, probability)``.

        Probabilities (when non-empty) sum to 1.
        """
        raise NotImplementedError

    def update(self, event: HandoffEvent) -> None:
        """Learn from a completed handoff event. Overridden by subclasses."""

    def reset(self) -> None:
        """Clear all learned state. Overridden by subclasses."""


class FrequencyPredictor(BasePredictor):
    """Predicts the globally most-frequent next agent per current agent.

    First-order transition counts with Laplace (add-one) smoothing, so the
    returned distribution always sums to 1 over all agents seen so far.
    Cold start (no observations) yields an empty ranking.
    """

    def __init__(self) -> None:
        self._transitions: dict[AgentId, Counter[AgentId]] = {}
        self._known: set[AgentId] = set()
        self._last_agent: AgentId | None = None

    @property
    def name(self) -> str:  # pragma: no cover - trivial
        return "frequency"

    def update(self, event: HandoffEvent) -> None:
        """Record a transition from the previous observed agent to ``event.to_agent``."""
        if self._last_agent is not None:
            self._transitions.setdefault(self._last_agent, Counter())[event.to_agent] += 1
        self._known.add(event.to_agent)
        self._last_agent = event.to_agent

    def predict(
        self,
        current_agent: AgentId,
        history: Sequence[HandoffEvent],
        step: int,
    ) -> list[tuple[AgentId, float]]:
        """Laplace-smoothed distribution P(next | current_agent) over seen agents."""
        if not self._known:
            return []
        k = len(self._known)
        counts = self._transitions.get(current_agent)
        total = sum(counts.values()) if counts else 0
        probs: list[tuple[AgentId, float]] = []
        for agent in sorted(self._known):
            observed = counts[agent] if counts else 0
            probs.append((agent, (observed + 1) / (total + k)))
        probs.sort(key=lambda kv: (-kv[1], kv[0]))
        return probs

    def reset(self) -> None:
        self._transitions = {}
        self._known = set()
        self._last_agent = None


class MarkovPredictor(BasePredictor):
    """Order-k n-gram predictor over the sequence of handoff target agents.

    Context = the last ``order`` ``to_agent`` values; backs off to progressively
    shorter contexts and finally to a :class:`FrequencyPredictor` when the
    context is unseen. At the highest observed context level it uses the raw
    (renormalized) empirical distribution. Ties are broken deterministically by
    agent name.
    """

    def __init__(self, order: int = 1) -> None:
        if order < 1:
            raise ValueError(f"order must be >= 1, got {order}")
        self.order: int = order
        self._fallback: FrequencyPredictor = FrequencyPredictor()
        # counts[j][context][next] for context lengths j = 1..order
        self._counts: dict[int, dict[tuple[AgentId, ...], Counter[AgentId]]] = {
            j: {} for j in range(1, order + 1)
        }
        self._known: set[AgentId] = set()
        self._seq: list[AgentId] = []

    @property
    def name(self) -> str:
        return f"markov{self.order}"

    def update(self, event: HandoffEvent) -> None:
        """Record n-gram transitions of every order up to ``self.order``."""
        self._fallback.update(event)
        if self._seq:
            for j in range(1, self.order + 1):
                if len(self._seq) >= j:
                    ctx = tuple(self._seq[-j:])
                    self._counts[j].setdefault(ctx, Counter())[event.to_agent] += 1
        self._seq.append(event.to_agent)
        self._known.add(event.to_agent)

    def predict(
        self,
        current_agent: AgentId,
        history: Sequence[HandoffEvent],
        step: int,
    ) -> list[tuple[AgentId, float]]:
        """Highest-order observed-context distribution, backing off as needed."""
        if not self._known:
            return []
        seq = [event.to_agent for event in history]
        if not seq or seq[-1] != current_agent:
            seq.append(current_agent)
        for j in range(min(self.order, len(seq)), 0, -1):
            ctx = tuple(seq[-j:])
            counter = self._counts[j].get(ctx)
            if counter:
                total = sum(counter.values())
                probs = [(agent, counter[agent] / total) for agent in counter]
                probs.sort(key=lambda kv: (-kv[1], kv[0]))
                return probs
        return self._fallback.predict(current_agent, history, step)

    def reset(self) -> None:
        self._fallback.reset()
        self._counts = {j: {} for j in range(1, self.order + 1)}
        self._known = set()
        self._seq = []


class RandomPredictor(BasePredictor):
    """Uniform random ranking — the no-information lower-bound baseline.

    Deterministic per ``(seed, step)``: the RNG stream is derived from the pair,
    so repeated runs with the same seed produce identical rankings.
    """

    def __init__(self, agents: list[str], seed: int = 0) -> None:
        self._agents: list[AgentId] = sorted(agents)
        self._seed: int = seed

    @property
    def name(self) -> str:  # pragma: no cover - trivial
        return "random"

    def predict(
        self,
        current_agent: AgentId,
        history: Sequence[HandoffEvent],
        step: int,
    ) -> list[tuple[AgentId, float]]:
        if not self._agents:
            return []
        rng = random.Random(self._seed * _SEED_STRIDE + step)
        ranked = list(self._agents)
        rng.shuffle(ranked)
        n = float(len(ranked))
        return [(agent, 1.0 / n) for agent in ranked]

    def update(self, event: HandoffEvent) -> None:
        """Random predictor is memoryless; nothing to learn."""

    def reset(self) -> None:
        """Stateless; nothing to clear."""


class NoisyOraclePredictor(BasePredictor):
    """Ground-truth predictor corrupted at a controlled rate.

    An evaluation instrument (NOT a real strategy component): it reads the true
    next agent from ``workload.events[step].to_agent``. With probability
    ``accuracy`` it ranks the truth first; otherwise a random wrong agent is
    ranked first. Deterministic per ``(seed, step)``.

    ``accuracy=1.0`` is always exactly right; ``accuracy=0.0`` is always wrong
    (whenever a wrong agent exists). Used for prediction-accuracy sweeps.
    """

    def __init__(self, workload: Workload, accuracy: float, seed: int = 0) -> None:
        if not 0.0 <= accuracy <= 1.0:
            raise ValueError(f"accuracy must be in [0, 1], got {accuracy}")
        self._agents: list[AgentId] = sorted(workload.agents)
        self._events: list[HandoffEvent] = list(workload.events)
        self._accuracy: float = accuracy
        self._seed: int = seed

    @property
    def name(self) -> str:  # pragma: no cover - trivial
        return "noisy_oracle"

    def predict(
        self,
        current_agent: AgentId,
        history: Sequence[HandoffEvent],
        step: int,
    ) -> list[tuple[AgentId, float]]:
        if not (0 <= step < len(self._events)):
            raise ValueError(f"step {step} out of range for workload with {len(self._events)} events")
        truth = self._events[step].to_agent
        others = [agent for agent in self._agents if agent != truth]
        rng = random.Random(self._seed * _SEED_STRIDE + step)
        if self._accuracy < 1.0 and others and rng.random() >= self._accuracy:
            # Failure mode: a random wrong agent leads; truth trails behind.
            wrong = others[rng.randrange(len(others))]
            ranked = [wrong] + [agent for agent in others if agent != wrong] + [truth]
        else:
            ranked = [truth] + others
        return [(agent, 1.0 if i == 0 else 0.0) for i, agent in enumerate(ranked)]

    def update(self, event: HandoffEvent) -> None:
        """Oracle is static; nothing to learn."""

    def reset(self) -> None:
        """Stateless; nothing to clear."""
