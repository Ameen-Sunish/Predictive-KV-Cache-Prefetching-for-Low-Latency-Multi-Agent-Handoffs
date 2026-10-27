"""TeleportCache — Interactive Streamlit Dashboard.

Visualizes and benchmarks Predictive KV-Cache Prefetching for Multi-Agent LLMs,
comparing predictive prefetching against demand paging, recompute baselines,
and a clairvoyant oracle.
"""

from __future__ import annotations

import io
from typing import Dict, List, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from teleportcache.core.baselines import RecomputeStrategy, TransferOnDemandStrategy
from teleportcache.core.simulator import Simulator
from teleportcache.prediction.oracle import OraclePrefetchStrategy
from teleportcache.prediction.prefetch_strategy import PredictivePrefetchStrategy
from teleportcache.prediction.predictors import FrequencyPredictor, MarkovPredictor
from teleportcache.types import (
    HandoffResult,
    RunResult,
    SimConfig,
    Strategy,
    Workload,
)
from teleportcache.workloads.generators import (
    generate_adversarial_workload,
    generate_branching_workload,
    generate_cyclic_workload,
    generate_hub_and_spoke_workload,
)

# -----------------------------------------------------------------------------
# Streamlit Page Setup & Custom Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="TeleportCache Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #6c757d;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 12px 16px;
        border-left: 4px solid #1f77b4;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 44px;
        white-space: pre-wrap;
        border-radius: 6px 6px 0px 0px;
        padding: 8px 16px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Strategy Factory
# -----------------------------------------------------------------------------
STRATEGY_COLORS = {
    "Recompute (No Cache)": "#e74c3c",       # Red
    "Transfer On Demand": "#e67e22",         # Orange
    "Predictive (Frequency)": "#3498db",     # Blue
    "Predictive (Markov-1)": "#2ecc71",      # Emerald Green
    "Predictive (Markov-2)": "#1abc9c",      # Turquoise
    "Oracle (Theoretical Ceiling)": "#9b59b6", # Purple
}


def create_strategies(selected: List[str], workload: Workload) -> Dict[str, Strategy]:
    """Instantiate selected strategies."""
    instances: Dict[str, Strategy] = {}
    for name in selected:
        if name == "Recompute (No Cache)":
            instances[name] = RecomputeStrategy()
        elif name == "Transfer On Demand":
            instances[name] = TransferOnDemandStrategy()
        elif name == "Predictive (Frequency)":
            instances[name] = PredictivePrefetchStrategy(FrequencyPredictor())
        elif name == "Predictive (Markov-1)":
            instances[name] = PredictivePrefetchStrategy(MarkovPredictor(order=1))
        elif name == "Predictive (Markov-2)":
            instances[name] = PredictivePrefetchStrategy(MarkovPredictor(order=2))
        elif name == "Oracle (Theoretical Ceiling)":
            instances[name] = OraclePrefetchStrategy(workload)
    return instances


# -----------------------------------------------------------------------------
# Sidebar: Configuration
# -----------------------------------------------------------------------------
st.sidebar.markdown("## Simulation Controls")

with st.sidebar.expander("Hardware & Cost Model (SimConfig)", expanded=True):
    col_cfg1, col_cfg2 = st.columns(2)
    with col_cfg1:
        capacity_blocks = st.slider("Cache Capacity (Blocks)", min_value=2, max_value=32, value=4, step=1)
        prefetch_window_ms = st.slider("Prefetch Window (ms)", min_value=5, max_value=150, value=50, step=5)
        eviction_policy = st.selectbox("Eviction Policy", options=["lru", "fifo"], index=0)
    with col_cfg2:
        bandwidth_gbps = st.slider("Bandwidth (Gbps)", min_value=1.0, max_value=40.0, value=10.0, step=1.0)
        transfer_latency_ms = st.slider("Fixed Transfer Latency (ms)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
        recompute_per_token_ms = st.slider("Prefill Cost (ms/token)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)

# Convert units
sim_config = SimConfig(
    capacity_blocks=int(capacity_blocks),
    bandwidth_bytes_per_s=float(bandwidth_gbps * 1.25e8),  # 10 Gbps = 1.25 GB/s
    transfer_latency_s=float(transfer_latency_ms / 1000.0),
    recompute_time_per_token_s=float(recompute_per_token_ms / 1000.0),
    prefetch_window_s=float(prefetch_window_ms / 1000.0),
    eviction_policy=str(eviction_policy),
)

with st.sidebar.expander("Multi-Agent Workload", expanded=True):
    workload_type = st.selectbox(
        "Workload Topology",
        options=[
            "Cyclic Pipeline (Canonical A → B → C → A)",
            "Hub & Spoke (Orchestrator → Workers)",
            "Branching Team (Planner → Coder → Reviewer → Tester)",
            "Adversarial (Stochastic Transitions)",
        ],
        index=0,
    )
    num_steps = st.slider("Simulation Steps", min_value=10, max_value=120, value=50, step=5)
    blocks_per_agent = st.slider("Blocks per Handoff", min_value=2, max_value=12, value=4, step=1)
    random_seed = st.number_input("RNG Seed", value=42, step=1)

with st.sidebar.expander("Strategies to Benchmark", expanded=True):
    all_strategies = list(STRATEGY_COLORS.keys())
    selected_strategies = st.multiselect(
        "Select strategies:",
        options=all_strategies,
        default=[
            "Recompute (No Cache)",
            "Transfer On Demand",
            "Predictive (Markov-1)",
            "Oracle (Theoretical Ceiling)",
        ],
    )

if not selected_strategies:
    st.sidebar.warning("Please select at least one strategy.")
    st.stop()


# -----------------------------------------------------------------------------
# Generate Workload & Run Simulation
# -----------------------------------------------------------------------------
@st.cache_data
def get_workload(w_type: str, steps: int, blocks_cnt: int, seed: int) -> Workload:
    if "Cyclic" in w_type:
        return generate_cyclic_workload(num_steps=steps, blocks_per_set=blocks_cnt)
    elif "Hub & Spoke" in w_type:
        return generate_hub_and_spoke_workload(num_steps=steps, blocks_per_agent=blocks_cnt, seed=seed)
    elif "Branching" in w_type:
        return generate_branching_workload(num_steps=steps, blocks_per_agent=blocks_cnt, seed=seed)
    else:
        return generate_adversarial_workload(num_steps=steps, blocks_per_agent=blocks_cnt, seed=seed)


workload = get_workload(workload_type, num_steps, blocks_per_agent, int(random_seed))
simulator = Simulator(sim_config)
strategies = create_strategies(selected_strategies, workload)

results: Dict[str, RunResult] = {}
for name, strat in strategies.items():
    results[name] = simulator.run(workload, strat)

# -----------------------------------------------------------------------------
# Main Header
# -----------------------------------------------------------------------------
st.markdown('<div class="main-header">TeleportCache: Multi-Agent KV-Cache Prefetcher</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">'
    'Eliminating LLM multi-agent handoff latency by <b>predicting upcoming agents</b> '
    'and <b>speculatively prefetching</b> attention KV-cache pages during the transition window.'
    '</div>',
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Top-Level KPI Summary
# -----------------------------------------------------------------------------
# Compute stats
summary_rows = []
for name, res in results.items():
    tot_lat = sum(h.latency_s for h in res.handoffs)
    tot_hits = sum(h.hits for h in res.handoffs)
    tot_miss = sum(h.misses for h in res.handoffs)
    tot_req = tot_hits + tot_miss
    hit_rate = (tot_hits / tot_req * 100.0) if tot_req > 0 else 0.0
    useful = sum(h.prefetch_useful_blocks for h in res.handoffs)
    wasted = sum(h.prefetch_wasted_blocks for h in res.handoffs)
    issued = sum(h.prefetch_issued_blocks for h in res.handoffs)
    bytes_mb = (sum(h.bytes_transferred for h in res.handoffs) + sum(h.prefetch_bytes for h in res.handoffs)) / 1e6
    avg_lat_ms = (tot_lat / len(res.handoffs) * 1000.0) if res.handoffs else 0.0

    summary_rows.append(
        {
            "Strategy": name,
            "Total Latency (s)": round(tot_lat, 4),
            "Avg Latency/Handoff (ms)": round(avg_lat_ms, 2),
            "Hits": tot_hits,
            "Misses": tot_miss,
            "Hit Rate (%)": round(hit_rate, 1),
            "Prefetch Issued": issued,
            "Prefetch Useful": useful,
            "Prefetch Wasted": wasted,
            "Data Moved (MB)": round(bytes_mb, 2),
        }
    )

df_summary = pd.DataFrame(summary_rows)

# Highlight Metrics
recompute_lat = next((r["Total Latency (s)"] for r in summary_rows if "Recompute" in r["Strategy"]), None)
ondemand_lat = next((r["Total Latency (s)"] for r in summary_rows if "Transfer On Demand" in r["Strategy"]), None)
markov_row = next((r for r in summary_rows if "Markov" in r["Strategy"]), None)

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
with kpi1:
    best_strat = df_summary.loc[df_summary["Total Latency (s)"].idxmin()]
    st.metric(
        label="Lowest Latency Strategy",
        value=best_strat["Strategy"].split(" (")[0],
        delta=f"{best_strat['Total Latency (s)']:.4f} s",
        delta_color="inverse",
    )
with kpi2:
    if ondemand_lat and markov_row:
        speedup_od = ondemand_lat / max(markov_row["Total Latency (s)"], 1e-6)
        pct_reduction = (1.0 - (markov_row["Total Latency (s)"] / ondemand_lat)) * 100.0
        st.metric(
            label="Speedup vs Demand Paging",
            value=f"{speedup_od:.1f}x Faster",
            delta=f"{pct_reduction:.1f}% less wait",
        )
    else:
        st.metric("Speedup vs Demand Paging", "N/A")
with kpi3:
    if recompute_lat and markov_row:
        speedup_rc = recompute_lat / max(markov_row["Total Latency (s)"], 1e-6)
        st.metric(
            label="Speedup vs Recompute",
            value=f"{speedup_rc:,.0f}x Faster",
            delta="Prefill eliminated",
        )
    else:
        st.metric("Speedup vs Recompute", "N/A")
with kpi4:
    max_hit = df_summary["Hit Rate (%)"].max()
    st.metric(label="Peak Cache Hit Rate", value=f"{max_hit:.1f}%", delta="Steady-state masked")
with kpi5:
    if markov_row:
        wasted_cnt = markov_row["Prefetch Wasted"]
        st.metric(label="Markov Prefetch Waste", value=f"{wasted_cnt} blocks", delta="Zero wasted bus I/O" if wasted_cnt == 0 else "Minor spec. waste")
    else:
        st.metric("Markov Prefetch Waste", "N/A")

st.markdown("---")

# -----------------------------------------------------------------------------
# Tabs Layout
# -----------------------------------------------------------------------------
tab_benchmark, tab_timeline, tab_inspector, tab_sweeps, tab_theory = st.tabs(
    [
        "Strategy Benchmark",
        "Latency Timeline",
        "Step-by-Step Inspector",
        "Sensitivity Sweeps",
        "OS Analogy Guide",
    ]
)

# -----------------------------------------------------------------------------
# Tab 1: Strategy Benchmark
# -----------------------------------------------------------------------------
with tab_benchmark:
    st.subheader("Benchmark Comparison")

    col_chart1, col_chart2 = st.columns([1, 1])

    with col_chart1:
        # Latency Plot in Log-Scale
        fig_lat = px.bar(
            df_summary,
            x="Strategy",
            y="Total Latency (s)",
            color="Strategy",
            color_discrete_map=STRATEGY_COLORS,
            title="Total Handoff Latency (Seconds, Log-Scale — Lower is Better)",
            text="Total Latency (s)",
            log_y=True,
        )
        fig_lat.update_traces(texttemplate="%{text:.4f}s", textposition="outside")
        fig_lat.update_layout(showlegend=False, xaxis_tickangle=-25, height=450)
        st.plotly_chart(fig_lat, width="stretch")

    with col_chart2:
        # Hit Rate Plot
        fig_hits = px.bar(
            df_summary,
            x="Strategy",
            y="Hit Rate (%)",
            color="Strategy",
            color_discrete_map=STRATEGY_COLORS,
            title="KV-Cache Hit Rate (% — Higher is Better)",
            text="Hit Rate (%)",
        )
        fig_hits.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_hits.update_layout(showlegend=False, xaxis_tickangle=-25, yaxis_range=[0, 115], height=450)
        st.plotly_chart(fig_hits, width="stretch")

    # Prefetch Utilization and Waste Breakdown
    st.markdown("#### Prefetch Efficiency & I/O Breakdown")
    col_pref1, col_pref2 = st.columns([1, 1])

    with col_pref1:
        prefetch_df = df_summary[df_summary["Prefetch Issued"] > 0]
        if not prefetch_df.empty:
            fig_pref = go.Figure(
                data=[
                    go.Bar(name="Useful (Hit)", x=prefetch_df["Strategy"], y=prefetch_df["Prefetch Useful"], marker_color="#2ecc71"),
                    go.Bar(name="Wasted (Dead Eviction)", x=prefetch_df["Strategy"], y=prefetch_df["Prefetch Wasted"], marker_color="#e74c3c"),
                ]
            )
            fig_pref.update_layout(barmode="stack", title="Speculative Prefetch Block Quality", height=350, yaxis_title="Blocks")
            st.plotly_chart(fig_pref, width="stretch")
        else:
            st.info("Select predictive or oracle strategies to view prefetch breakdown.")

    with col_pref2:
        fig_data = px.bar(
            df_summary,
            x="Strategy",
            y="Data Moved (MB)",
            color="Strategy",
            color_discrete_map=STRATEGY_COLORS,
            title="Total Data Moved over Interconnect (MB)",
            text="Data Moved (MB)",
        )
        fig_data.update_traces(texttemplate="%{text:.2f} MB", textposition="outside")
        fig_data.update_layout(showlegend=False, xaxis_tickangle=-25, height=350)
        st.plotly_chart(fig_data, width="stretch")

    # Detailed Table and CSV download
    st.markdown("#### Detailed Metrics Table")
    st.dataframe(df_summary, width="stretch")

    csv_buf = io.StringIO()
    df_summary.to_csv(csv_buf, index=False)
    st.download_button(
        label="Download Benchmark Results (CSV)",
        data=csv_buf.getvalue(),
        file_name=f"teleportcache_benchmark_{workload.name}.csv",
        mime="text/csv",
    )

# -----------------------------------------------------------------------------
# Tab 2: Cumulative Latency Timeline
# -----------------------------------------------------------------------------
with tab_timeline:
    st.subheader("Step-by-Step Cumulative Latency Timeline")
    st.markdown(
        "Observe how latency compounds across handoff steps. "
        "Notice how **Recompute** shoots up linearly, **Demand Paging** incurs a steady stair-step penalty, "
        "while **Predictive Prefetching** stays near flat once initial cold-start pages are learned."
    )

    timeline_data = []
    for strat_name, res in results.items():
        cum_lat = 0.0
        for h in res.handoffs:
            cum_lat += h.latency_s
            timeline_data.append(
                {
                    "Step": h.step,
                    "Cumulative Latency (s)": cum_lat,
                    "Step Latency (ms)": h.latency_s * 1000.0,
                    "Strategy": strat_name,
                    "Handoff": f"{h.from_agent} → {h.to_agent}",
                    "Hits": h.hits,
                    "Misses": h.misses,
                }
            )

    df_timeline = pd.DataFrame(timeline_data)

    fig_cum = px.line(
        df_timeline,
        x="Step",
        y="Cumulative Latency (s)",
        color="Strategy",
        color_discrete_map=STRATEGY_COLORS,
        title="Cumulative Handoff Latency over Simulation Steps",
        hover_data=["Handoff", "Hits", "Misses"],
    )
    fig_cum.update_layout(height=480)
    st.plotly_chart(fig_cum, width="stretch")

    # Per-step Latency distribution (log-scale box plot or instant latency)
    st.markdown("#### Instantaneous Step Latency (ms)")
    fig_step = px.line(
        df_timeline,
        x="Step",
        y="Step Latency (ms)",
        color="Strategy",
        color_discrete_map=STRATEGY_COLORS,
        title="Per-Handoff Latency (ms)",
    )
    fig_step.update_layout(height=350)
    st.plotly_chart(fig_step, width="stretch")

# -----------------------------------------------------------------------------
# Tab 3: Step-by-Step Inspector
# -----------------------------------------------------------------------------
with tab_inspector:
    st.subheader("Interactive Step Inspector")
    st.markdown("Scrub through individual handoff steps to observe how the memory manager behaves in real-time.")

    step_idx = st.slider("Select Simulation Step Index:", min_value=0, max_value=num_steps - 1, value=0)

    # Event details
    current_event = workload.events[step_idx]
    st.info(
        f"**Step {step_idx}**: Handoff from **Agent `{current_event.from_agent}`** → "
        f"**Agent `{current_event.to_agent}`** | "
        f"Required KV Blocks ({len(current_event.required_blocks)}): `{', '.join(current_event.required_blocks)}`"
    )

    # Strategy comparison at this specific step
    step_records = []
    for strat_name, res in results.items():
        h: HandoffResult = res.handoffs[step_idx]
        step_records.append(
            {
                "Strategy": strat_name,
                "Latency (ms)": round(h.latency_s * 1000.0, 3),
                "Hits": h.hits,
                "Misses": h.misses,
                "Prefetches Completed": h.prefetch_issued_blocks,
                "Prefetch Useful": h.prefetch_useful_blocks,
                "Prefetch Wasted": h.prefetch_wasted_blocks,
                "Cache Occupancy": f"{h.cache_occupancy * 100:.0f}%",
            }
        )

    df_step = pd.DataFrame(step_records)
    st.dataframe(df_step, width="stretch")

    # Visualizing Handoff State
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown(f"#### Target Agent `{current_event.to_agent}` Demand Resolution")
        fig_hit_miss = go.Figure(
            data=[
                go.Bar(name="Hits", x=df_step["Strategy"], y=df_step["Hits"], marker_color="#2ecc71"),
                go.Bar(name="Misses (Page Faults)", x=df_step["Strategy"], y=df_step["Misses"], marker_color="#e74c3c"),
            ]
        )
        fig_hit_miss.update_layout(barmode="stack", height=320, title=f"Hits vs Misses at Step {step_idx}")
        st.plotly_chart(fig_hit_miss, width="stretch")

    with col_s2:
        st.markdown(f"#### Step Latency (ms)")
        fig_step_lat = px.bar(
            df_step,
            x="Strategy",
            y="Latency (ms)",
            color="Strategy",
            color_discrete_map=STRATEGY_COLORS,
            title=f"Latency Incurred at Step {step_idx} (ms)",
            text="Latency (ms)",
        )
        fig_step_lat.update_traces(texttemplate="%{text:.2f}ms", textposition="outside")
        fig_step_lat.update_layout(showlegend=False, height=320)
        st.plotly_chart(fig_step_lat, width="stretch")

# -----------------------------------------------------------------------------
# Tab 4: Sensitivity Sweeps ("What-If" Analysis)
# -----------------------------------------------------------------------------
with tab_sweeps:
    st.subheader("Parameter Sensitivity Sweeps")
    st.markdown(
        "Conduct live sensitivity sweeps to analyze where predictive prefetching provides "
        "the highest return on investment and how hardware constraints impact latency."
    )

    sweep_type = st.radio(
        "Select Sensitivity Experiment:",
        options=[
            "Prefetch Window Sweep (Varying Idle Transition Window)",
            "Cache Capacity Sweep (Varying Per-Agent KV Blocks)",
        ],
        horizontal=True,
    )

    if sweep_type == "Prefetch Window Sweep (Varying Idle Transition Window)":
        st.markdown(
            "> **Hypothesis:** As the transition window $T_{\\text{window}}$ increases, "
            "the prefetcher has sufficient I/O time to move more speculative pages, driving handoff latency to zero."
        )
        windows_ms = [5, 10, 20, 30, 40, 50, 60, 80, 100]
        sweep_data = []

        with st.spinner("Executing prefetch window sweep..."):
            for w_ms in windows_ms:
                cfg = SimConfig(
                    capacity_blocks=sim_config.capacity_blocks,
                    bandwidth_bytes_per_s=sim_config.bandwidth_bytes_per_s,
                    transfer_latency_s=sim_config.transfer_latency_s,
                    recompute_time_per_token_s=sim_config.recompute_time_per_token_s,
                    prefetch_window_s=w_ms / 1000.0,
                    eviction_policy=sim_config.eviction_policy,
                )
                sim_sweep = Simulator(cfg)
                strat_dict = create_strategies(
                    ["Transfer On Demand", "Predictive (Markov-1)", "Oracle (Theoretical Ceiling)"],
                    workload,
                )
                for s_name, s_inst in strat_dict.items():
                    r = sim_sweep.run(workload, s_inst)
                    sweep_data.append(
                        {
                            "Prefetch Window (ms)": w_ms,
                            "Total Latency (s)": sum(h.latency_s for h in r.handoffs),
                            "Strategy": s_name,
                        }
                    )

        df_sweep = pd.DataFrame(sweep_data)
        fig_sw = px.line(
            df_sweep,
            x="Prefetch Window (ms)",
            y="Total Latency (s)",
            color="Strategy",
            color_discrete_map=STRATEGY_COLORS,
            markers=True,
            title="Total Handoff Latency vs. Prefetch Window Size",
        )
        fig_sw.update_layout(height=450)
        st.plotly_chart(fig_sw, width="stretch")

    else:
        st.markdown(
            "> **Hypothesis:** When capacity $C < |\\text{Working Set}|$, demand paging thrashes constantly. "
            "Predictive prefetching mitigates thrashing by staging needed pages right before activation."
        )
        capacities = [2, 4, 6, 8, 12, 16, 24]
        sweep_data = []

        with st.spinner("Executing cache capacity sweep..."):
            for cap in capacities:
                cfg = SimConfig(
                    capacity_blocks=cap,
                    bandwidth_bytes_per_s=sim_config.bandwidth_bytes_per_s,
                    transfer_latency_s=sim_config.transfer_latency_s,
                    recompute_time_per_token_s=sim_config.recompute_time_per_token_s,
                    prefetch_window_s=sim_config.prefetch_window_s,
                    eviction_policy=sim_config.eviction_policy,
                )
                sim_sweep = Simulator(cfg)
                strat_dict = create_strategies(
                    ["Transfer On Demand", "Predictive (Markov-1)", "Oracle (Theoretical Ceiling)"],
                    workload,
                )
                for s_name, s_inst in strat_dict.items():
                    r = sim_sweep.run(workload, s_inst)
                    sweep_data.append(
                        {
                            "Cache Capacity (Blocks)": cap,
                            "Total Latency (s)": sum(h.latency_s for h in r.handoffs),
                            "Strategy": s_name,
                        }
                    )

        df_sweep = pd.DataFrame(sweep_data)
        fig_sw = px.line(
            df_sweep,
            x="Cache Capacity (Blocks)",
            y="Total Latency (s)",
            color="Strategy",
            color_discrete_map=STRATEGY_COLORS,
            markers=True,
            title="Total Handoff Latency vs. Cache Capacity",
        )
        fig_sw.update_layout(height=450)
        st.plotly_chart(fig_sw, width="stretch")

# -----------------------------------------------------------------------------
# Tab 5: OS Analogy & Review Guide
# -----------------------------------------------------------------------------
with tab_theory:
    st.subheader("OS Virtual Memory Analogy & Core Principles")
    st.markdown(
        """
        TeleportCache maps modern **LLM multi-agent pipelines** directly onto classic **Operating System virtual memory**:
        """
    )

    analogy_table = pd.DataFrame(
        [
            {"Classic OS Virtual Memory": "Process", "TeleportCache Multi-Agent System": "Individual Agent (Planner, Coder, etc.)"},
            {"Classic OS Virtual Memory": "Physical Memory Frames", "TeleportCache Multi-Agent System": "Agent's KV Scratchpad (capacity_blocks)"},
            {"Classic OS Virtual Memory": "Virtual Memory Page", "TeleportCache Multi-Agent System": "KV Block (fixed-size paged tensor)"},
            {"Classic OS Virtual Memory": "Page Reference String", "TeleportCache Multi-Agent System": "Ordered sequence of HandoffEvents"},
            {"Classic OS Virtual Memory": "Page Fault", "TeleportCache Multi-Agent System": "Cache miss at handoff (transfer / recompute)"},
            {"Classic OS Virtual Memory": "Swap-in over Disk/Bus", "TeleportCache Multi-Agent System": "KV block transfer over interconnect"},
            {"Classic OS Virtual Memory": "Re-execute instruction", "TeleportCache Multi-Agent System": "Prefill KV from raw prompt tokens (recompute)"},
            {"Classic OS Virtual Memory": "Page Replacement (LRU / FIFO)", "TeleportCache Multi-Agent System": "Scratchpad eviction policy on overflow"},
            {"Classic OS Virtual Memory": "Hardware Data Prefetcher", "TeleportCache Multi-Agent System": "Predictive prefetch strategy"},
            {"Classic OS Virtual Memory": "Denning's Working Set Model", "TeleportCache Multi-Agent System": "BlockSelector decayed working-set ranking"},
            {"Classic OS Virtual Memory": "Bélády's Clairvoyant Optimal (OPT)", "TeleportCache Multi-Agent System": "OraclePrefetchStrategy (clairvoyant upper bound)"},
        ]
    )
    st.table(analogy_table)

    st.markdown("### Key Architectural Insights")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
            #### 1. Why Latency Disappears
            At every multi-agent handoff, between when Agent A finishes emitting tokens and Agent B initializes its prompt,
            there is an idle transition window ($T_{\\text{window}} \\approx 20\\text{--}80\\text{ ms}$).
            
            By predicting *who runs next* and *what pages they need*, TeleportCache moves KV pages across the interconnect
            **during the idle window**, masking the bus latency completely behind background I/O.
            """
        )
    with c2:
        st.markdown(
            """
            #### 2. Dual-Layer Prediction
            - **Next-Agent Predictor:** Markov chain ($n$-grams with backoff) or frequency counting predicts target agent probability distribution $P(\\text{Next} \\mid \\text{History})$.
            - **BlockSelector Working-Set Estimation:** Scores candidate blocks via exponential recency decay $0.5^{\\text{age} / \\lambda}$ and adds a $2\\times$ boost for persistent blocks (e.g., system prompts / tool definitions) used in $\\ge 80\\%$ of visits.
            """
        )
