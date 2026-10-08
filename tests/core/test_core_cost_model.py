"""Unit tests for CostModel and hardware parameter translation."""

import pytest
from teleportcache.core.cost_model import CostModel
from teleportcache.types import KVBlock, SimConfig


def test_cost_model_transfer_time() -> None:
    # 1.25e9 B/s bandwidth, 0.001s base transfer latency
    config = SimConfig(
        bandwidth_bytes_per_s=1.25e9,
        transfer_latency_s=0.001,
    )
    model = CostModel(config)
    
    # 12.5 MB block = 12,500,000 bytes
    # Transfer time = 0.001 + (12,500,000 / 1.25e9) = 0.001 + 0.010 = 0.011s
    block = KVBlock(block_id="b1", num_tokens=512, size_bytes=12_500_000)
    assert pytest.approx(model.transfer_time(block), rel=1e-6) == 0.011


def test_cost_model_recompute_time() -> None:
    config = SimConfig(recompute_time_per_token_s=0.002)
    model = CostModel(config)
    
    # 512 tokens * 0.002 s/token = 1.024s
    block = KVBlock(block_id="b1", num_tokens=512, size_bytes=1000)
    assert pytest.approx(model.recompute_time(block), rel=1e-6) == 1.024


def test_cost_model_miss_cost() -> None:
    config = SimConfig(
        bandwidth_bytes_per_s=1.0e9,
        transfer_latency_s=0.002,
        recompute_time_per_token_s=0.001,
    )
    model = CostModel(config)
    block = KVBlock(block_id="b1", num_tokens=100, size_bytes=1_000_000)
    
    # Transfer: 0.002 + (1,000,000 / 1e9) = 0.003s
    assert pytest.approx(model.miss_cost(block, "transfer"), rel=1e-6) == 0.003
    # Recompute: 100 * 0.001 = 0.1s
    assert pytest.approx(model.miss_cost(block, "recompute"), rel=1e-6) == 0.100
    
    with pytest.raises(ValueError, match="Unknown miss_handling"):
        model.miss_cost(block, "invalid_mode")


def test_prefetch_budget_blocks() -> None:
    # 0.05s window, transfer_time = 0.011s => floor(0.05 / 0.011) = 4 blocks
    config = SimConfig(
        bandwidth_bytes_per_s=1.25e9,
        transfer_latency_s=0.001,
        prefetch_window_s=0.05,
    )
    model = CostModel(config)
    block = KVBlock(block_id="b1", num_tokens=512, size_bytes=12_500_000)
    assert model.prefetch_budget_blocks(block) == 4
