"""KV-block (page) selection for predictive prefetching.

Given a predicted next agent B, the :class:`BlockSelector` ranks which of B's
KV-cache blocks (memory pages) B will probably need next. It is the
"working-set estimator" in the OS analogy: like a segment/page fault predictor
that watches which pages a process touches and ages them exponentially so
recently-touched pages are favored for speculative page-in.

Learns only from completed handoff events via ``update(event)`` — no peeking.
"""

from __future__ import annotations

from collections import Counter

from teleportcache.types import AgentId, BlockId, HandoffEvent


class BlockSelector:
    """Ranks the KV blocks a predicted agent will probably require next.

    Scoring model per block ``b`` of agent ``a``:

    ``score(b) = usage_count(b) * 0.5 ** (age / half_life) [* boost]``

    where ``age`` is the number of steps since ``b`` was last required by ``a``
    (exponential recency decay with a configurable half-life), and the optional
    "always-required" boost multiplies the score of blocks present in at least
    ``boost_threshold`` of all past handoffs to ``a`` by ``boost_weight``.

    Cold start (no observed handoffs for the agent) returns an empty ranking.
    """

    def __init__(
        self,
        half_life: float = 8.0,
        boost_threshold: float = 0.8,
        boost_weight: float = 2.0,
    ) -> None:
        if half_life <= 0:
            raise ValueError(f"half_life must be > 0, got {half_life}")
        if boost_weight < 0:
            raise ValueError(f"boost_weight must be >= 0, got {boost_weight}")
        self._half_life: float = half_life
        self._boost_threshold: float = boost_threshold
        self._boost_weight: float = boost_weight
        self._usage: dict[AgentId, Counter[BlockId]] = {}
        self._last_seen: dict[AgentId, dict[BlockId, int]] = {}
        self._required_in: dict[AgentId, Counter[BlockId]] = {}
        self._handoffs: dict[AgentId, int] = {}
        self._now: int = 0

    def update(self, event: HandoffEvent) -> None:
        """Learn which blocks the receiving agent required in this handoff."""
        agent = event.to_agent
        self._now = max(self._now, event.step)
        self._handoffs[agent] = self._handoffs.get(agent, 0) + 1
        for block_id in set(event.required_blocks):
            self._usage.setdefault(agent, Counter())[block_id] += 1
            self._last_seen.setdefault(agent, {})[block_id] = event.step
            self._required_in.setdefault(agent, Counter())[block_id] += 1

    def rank(
        self,
        agent: AgentId,
        exclude: set[BlockId] | frozenset[BlockId] = frozenset(),
    ) -> list[tuple[BlockId, float]]:
        """Return ``[(block_id, score), ...]`` sorted best-first.

        Args:
            agent: The agent whose future block needs are estimated.
            exclude: Blocks already resident in the agent's cache; skipped.

        Returns an empty list when the agent has never been observed
        (cold start) or when every known block is excluded.
        """
        if agent not in self._handoffs:
            return []
        scores: list[tuple[BlockId, float]] = []
        usage = self._usage[agent]
        seen = self._last_seen[agent]
        for block_id, count in usage.items():
            if block_id in exclude:
                continue
            age = max(0, self._now - seen[block_id])
            decayed = count * 0.5 ** (age / self._half_life)
            share = self._required_in[agent][block_id] / self._handoffs[agent]
            if share >= self._boost_threshold:
                decayed *= self._boost_weight
            scores.append((block_id, decayed))
        scores.sort(key=lambda kv: (-kv[1], kv[0]))
        return scores

    def reset(self) -> None:
        """Clear all learned usage history."""
        self._usage = {}
        self._last_seen = {}
        self._required_in = {}
        self._handoffs = {}
        self._now = 0
