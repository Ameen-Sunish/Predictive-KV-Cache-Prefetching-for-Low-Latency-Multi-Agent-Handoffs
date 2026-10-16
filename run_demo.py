"""TeleportCache Demonstration and Output Benchmark Script.

Runs deterministic multi-agent handoff simulation across all caching and prefetching
strategies, prints formatted execution metrics, and generates a comparison plot.
"""

from __future__ import annotations

import os
import matplotlib.pyplot as plt
import numpy as np

from teleportcache.core.baselines import RecomputeStrategy, TransferOnDemandStrategy
from teleportcache.core.simulator import Simulator
from teleportcache.prediction.oracle import OraclePrefetchStrategy
from teleportcache.prediction.prefetch_strategy import PredictivePrefetchStrategy
from teleportcache.prediction.predictors import FrequencyPredictor, MarkovPredictor
from tests.prediction.conftest import SIM_CONFIG, cyclic_workload


def run_benchmark():
    num_steps = 50
    workload = cyclic_workload(num_steps=num_steps)
    sim = Simulator(SIM_CONFIG)

    strategies = {
        "Recompute (No Cache)": RecomputeStrategy(),
        "Transfer On Demand": TransferOnDemandStrategy(),
        "Predictive (Frequency)": PredictivePrefetchStrategy(FrequencyPredictor()),
        "Predictive (Markov-1)": PredictivePrefetchStrategy(MarkovPredictor(order=1)),
        "Oracle (Upper Bound)": OraclePrefetchStrategy(workload),
    }

    results = {}
    print("=" * 80)
    print(f"{'STRATEGY':<30} | {'LATENCY (s)':<12} | {'HITS':<6} | {'MISSES':<6} | {'HIT RATE':<9} | {'PREFETCH USEFUL':<15}")
    print("-" * 80)

    for name, strat in strategies.items():
        res = sim.run(workload, strat)
        results[name] = res
        total_lat = sum(h.latency_s for h in res.handoffs)
        hits = sum(h.hits for h in res.handoffs)
        misses = sum(h.misses for h in res.handoffs)
        total_req = hits + misses
        hit_rate = (hits / total_req * 100) if total_req else 0.0
        useful = sum(h.prefetch_useful_blocks for h in res.handoffs)
        wasted = sum(h.prefetch_wasted_blocks for h in res.handoffs)
        print(f"{name:<30} | {total_lat:<12.4f} | {hits:<6} | {misses:<6} | {hit_rate:>7.1f}% | {useful:<15}")

    print("=" * 80)

    # Generate output benchmark figure for report screenshots
    names = list(strategies.keys())
    latencies = [sum(h.latency_s for h in results[n].handoffs) for n in names]
    hit_rates = [
        (sum(h.hits for h in results[n].handoffs) / (sum(h.hits for h in results[n].handoffs) + sum(h.misses for h in results[n].handoffs)) * 100)
        for n in names
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Latency Plot (Log scale because Recompute is > 100s)
    bars1 = ax1.bar(names, latencies, color=["#e74c3c", "#e67e22", "#3498db", "#2ecc71", "#9b59b6"])
    ax1.set_yscale("log")
    ax1.set_ylabel("Total Handoff Latency (seconds, log-scale)")
    ax1.set_title("Total Handoff Latency Across Strategies (Lower is Better)")
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=25, ha="right")
    ax1.grid(axis="y", linestyle="--", alpha=0.7)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, max(yval * 1.25, 0.0001), f"{yval:.4f}s", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # Hit Rate Plot
    bars2 = ax2.bar(names, hit_rates, color=["#e74c3c", "#e67e22", "#3498db", "#2ecc71", "#9b59b6"])
    ax2.set_ylabel("KV-Cache Hit Rate (%)")
    ax2.set_title("KV-Cache Hit Rate (Higher is Better)")
    ax2.set_ylim(0, 115)
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=25, ha="right")
    ax2.grid(axis="y", linestyle="--", alpha=0.7)
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 2, f"{yval:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

    plt.subplots_adjust(bottom=0.25, top=0.90, left=0.08, right=0.95, wspace=0.25)
    output_img = "benchmark_results.png"
    plt.savefig(output_img, dpi=300)
    print(f"\n[+] Benchmark comparison plot saved successfully to '{output_img}'.")


if __name__ == "__main__":
    run_benchmark()
