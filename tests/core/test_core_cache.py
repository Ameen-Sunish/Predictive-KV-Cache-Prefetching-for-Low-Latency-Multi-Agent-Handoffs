"""Unit tests for cache eviction policies and AgentCache."""

import pytest
from teleportcache.core.cache import AgentCache, FIFOPolicy, LRUPolicy, make_policy


def test_lru_eviction_order() -> None:
    cache = AgentCache(capacity_blocks=3, policy=LRUPolicy())
    
    cache.put("b1")
    cache.put("b2")
    cache.put("b3")
    assert cache.resident_blocks() == ("b1", "b2", "b3")
    
    # Access b1, so b2 becomes the least recently used
    assert cache.touch("b1") is False
    assert cache.resident_blocks() == ("b2", "b3", "b1")
    
    # Insert b4 -> b2 should be evicted
    evicted = cache.put("b4")
    assert evicted == [("b2", False)]
    assert cache.resident_blocks() == ("b3", "b1", "b4")
    assert cache.contains("b2") is False
    assert cache.contains("b4") is True


def test_lru_touch_updates_recency() -> None:
    cache = AgentCache(capacity_blocks=2, policy="lru")
    cache.put("b1")
    cache.put("b2")
    
    # Touch b1 via get()
    assert cache.get("b1") is True
    assert cache.resident_blocks() == ("b2", "b1")
    
    # Put b3 -> b2 evicted
    evicted = cache.put("b3")
    assert evicted == [("b2", False)]
    assert cache.contains("b1") is True
    assert cache.contains("b2") is False
    assert cache.contains("b3") is True


def test_fifo_eviction_order() -> None:
    cache = AgentCache(capacity_blocks=3, policy=FIFOPolicy())
    cache.put("b1")
    cache.put("b2")
    cache.put("b3")
    assert cache.resident_blocks() == ("b1", "b2", "b3")
    
    # Touch b1 -> FIFO must NOT change victim order
    cache.touch("b1")
    assert cache.resident_blocks() == ("b1", "b2", "b3")
    
    # Insert b4 -> oldest (b1) evicted
    evicted = cache.put("b4")
    assert evicted == [("b1", False)]
    assert cache.resident_blocks() == ("b2", "b3", "b4")


def test_capacity_0() -> None:
    cache = AgentCache(capacity_blocks=0, policy="lru")
    assert cache.capacity == 0
    assert len(cache) == 0
    assert cache.occupancy() == 0.0
    
    evicted = cache.put("b1")
    assert evicted == []
    assert cache.contains("b1") is False
    assert cache.get("b1") is False
    assert cache.resident_blocks() == ()


def test_capacity_1() -> None:
    cache = AgentCache(capacity_blocks=1, policy="lru")
    assert cache.capacity == 1
    
    cache.put("b1")
    assert cache.contains("b1") is True
    assert cache.resident_blocks() == ("b1",)
    assert cache.occupancy() == 1.0
    
    evicted = cache.put("b2")
    assert evicted == [("b1", False)]
    assert cache.contains("b1") is False
    assert cache.contains("b2") is True
    assert cache.resident_blocks() == ("b2",)


def test_prefetched_unused_flag_tracking() -> None:
    cache = AgentCache(capacity_blocks=2, policy="lru")
    
    # Put b1 as prefetched
    cache.put("b1", prefetched=True)
    assert cache.get_prefetched_unused_blocks() == ["b1"]
    
    # First touch -> useful prefetch (returns True)
    was_useful = cache.touch("b1")
    assert was_useful is True
    assert cache.get_prefetched_unused_blocks() == []
    
    # Second touch -> already marked used (returns False)
    assert cache.touch("b1") is False
    
    # Put b2 as prefetched, then evict it before use
    cache.put("b2", prefetched=True)
    assert cache.get_prefetched_unused_blocks() == ["b2"]

    # Touch b1 so b1 is MRU and b2 becomes LRU victim
    cache.touch("b1")
    assert cache.resident_blocks() == ("b2", "b1")

    # Put b3 (demand) -> b2 is evicted
    evicted = cache.put("b3", prefetched=False)
    assert evicted == [("b2", True)]  # b2 was prefetched and never used!
    assert cache.get_prefetched_unused_blocks() == []


def test_make_policy_factory() -> None:
    lru = make_policy("lru")
    assert isinstance(lru, LRUPolicy)
    fifo = make_policy("FIFO")
    assert isinstance(fifo, FIFOPolicy)
    
    with pytest.raises(ValueError, match="Unknown eviction policy"):
        make_policy("random")


def test_agent_cache_invalid_capacity() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        AgentCache(capacity_blocks=-1)


def test_touch_non_resident_raises_key_error() -> None:
    cache = AgentCache(capacity_blocks=2)
    with pytest.raises(KeyError):
        cache.touch("missing_block")
