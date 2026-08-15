#!/usr/bin/env python3
"""
Step 5: Analyze and visualize unique causal chains per panel.

Reads step4 outputs under unique_causal_chain_outputs/<panel>/:
  - unique_causal_chains.json
  - claim_canonical_chains.csv (or claim_chain_assignments.csv)

Outputs per panel:
  - chain_respondent_frequency_by_group.png
  - chain_respondent_frequency_distribution.png  (rank–frequency curve per group)
  - chain_respondent_frequency.csv  (all unique chains, ranked)

Run: python step5_analyze_causal_chain.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_ROOT = SCRIPT_DIR / "unique_causal_chain_outputs"
TEXTUAL_ANALYSIS_DIR = Path(__file__).resolve().parents[3]
PROJECT_ROOT = Path(__file__).resolve().parents[4]
for path in (TEXTUAL_ANALYSIS_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from viz_style import (  # noqa: E402
    BAR_EDGE_COLOR,
    BAR_EDGE_WIDTH,
    FONT_AXIS_LABEL,
    FONT_LEGEND,
    FONT_TICK,
    FONT_TITLE,
    apply_plot_style,
    style_axes,
)
from viz_style import SAVE_DPI  # noqa: E402
from viz_config import GROUP_COLORS  # noqa: E402

GROUP_ORDER = ["student", "senior", "GenAI"]
GROUP_DISPLAY = {
    "student": "PhD Students",
    "senior": "Senior Scientists",
    "GenAI": "GenAI",
}
GROUP_COLORS_LOCAL = {
    "student": GROUP_COLORS["phd"],
    "senior": GROUP_COLORS["senior"],
    "GenAI": GROUP_COLORS["genai"],
}

FREQUENCY_PNG = "chain_respondent_frequency_by_group.png"
FREQUENCY_DIST_PNG = "chain_respondent_frequency_distribution.png"
FREQUENCY_CSV = "chain_respondent_frequency.csv"

CORE_MIN_RESPONDENTS = 3
COLOR_TAIL = "#B0BEC5"
COLOR_TAIL_FILL = "#D5DADF"


def save_figure_with_caption(fig: plt.Figure, out_path: Path) -> None:
    """Reserve space for suptitle so it is not clipped on save."""
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(
        out_path,
        dpi=SAVE_DPI,
        bbox_inches="tight",
        pad_inches=0.14,
    )
    plt.close(fig)


def discover_panels(root: Path = INPUT_ROOT) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Input root not found: {root}")
    panels = sorted(
        p for p in root.iterdir() if p.is_dir() and (p / "unique_causal_chains.json").exists()
    )
    if not panels:
        raise ValueError(f"No completed step4 panels under {root}")
    return panels


def load_claims_table(panel_dir: Path) -> pd.DataFrame:
    canonical_path = panel_dir / "claim_canonical_chains.csv"
    assignments_path = panel_dir / "claim_chain_assignments.csv"

    if canonical_path.exists():
        df = pd.read_csv(canonical_path)
        if "unique_chain_id" in df.columns and df["unique_chain_id"].notna().any():
            return df

    if assignments_path.exists():
        return pd.read_csv(assignments_path)

    raise FileNotFoundError(
        f"No claim table with unique_chain_id in {panel_dir}. Run step4 first."
    )


def load_unique_chains(panel_dir: Path) -> dict[str, Any]:
    path = panel_dir / "unique_causal_chains.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def panel_title(meta: dict[str, Any], panel_name: str) -> str:
    stage = meta.get("stage", "")
    outcome = meta.get("outcome", "")
    theory = meta.get("theory_type", "")
    if stage and outcome and theory:
        return f"{stage} · {outcome} · {theory}"
    return panel_name.replace("_", " · ")


def compute_group_diversity(claims_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group in GROUP_ORDER:
        gdf = claims_df[claims_df["group_label"] == group]
        if gdf.empty:
            continue
        n_respondents = int(gdf["participant_name"].nunique())
        n_unique_chains = int(gdf["unique_chain_id"].nunique())
        rows.append(
            {
                "group_label": group,
                "group_display": GROUP_DISPLAY[group],
                "n_respondents": n_respondents,
                "n_unique_chains": n_unique_chains,
                "unique_chains_per_respondent": round(n_unique_chains / n_respondents, 3),
            }
        )
    return pd.DataFrame(rows)


def compute_within_group_chain_frequencies(claims_df: pd.DataFrame) -> pd.DataFrame:
    """Per group × chain: how many respondents in that group propose the chain."""
    df = (
        claims_df.dropna(subset=["unique_chain_id"])
        .assign(unique_chain_id=lambda d: d["unique_chain_id"].astype(int))
    )
    rows: list[dict[str, Any]] = []
    for group in GROUP_ORDER:
        gdf = df[df["group_label"] == group]
        if gdf.empty:
            continue
        freq = (
            gdf.groupby("unique_chain_id", as_index=False)
            .agg(n_respondents=("participant_name", "nunique"))
        )
        for _, row in freq.iterrows():
            rows.append(
                {
                    "group_label": group,
                    "group_display": GROUP_DISPLAY[group],
                    "unique_chain_id": int(row["unique_chain_id"]),
                    "n_respondents": int(row["n_respondents"]),
                }
            )
    return pd.DataFrame(rows)


def _gini_coefficient(values: np.ndarray) -> float:
    x = np.sort(values.astype(float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    return float((2 * np.sum((np.arange(1, n + 1)) * x) - (n + 1) * x.sum()) / (n * x.sum()))


def compute_group_uptake_summary(within_group_freq: pd.DataFrame) -> pd.DataFrame:
    """
    Per group: tail share (chain types) and Gini of uptake across chain types.

    Gini is computed on n_respondents per chain within the group (abundance-weighted
    inequality of the rank-frequency distribution). Higher Gini = uptake concentrated
    on fewer shared chains (stronger core structure).
    """
    rows: list[dict[str, Any]] = []
    for group in GROUP_ORDER:
        gdf = within_group_freq[within_group_freq["group_label"] == group]
        if gdf.empty:
            continue
        uptake = gdf["n_respondents"].to_numpy()
        n_tail = int((gdf["n_respondents"] < CORE_MIN_RESPONDENTS).sum())
        n_core = int((gdf["n_respondents"] >= CORE_MIN_RESPONDENTS).sum())
        total = n_tail + n_core
        rows.append(
            {
                "group_label": group,
                "group_display": GROUP_DISPLAY[group],
                "n_tail_chains": n_tail,
                "n_core_chains": n_core,
                "n_chains_total": total,
                "pct_tail": round(100.0 * n_tail / total, 1) if total else 0.0,
                "pct_core": round(100.0 * n_core / total, 1) if total else 0.0,
                "gini_uptake": round(_gini_coefficient(uptake), 3),
            }
        )
    return pd.DataFrame(rows)


def compute_chain_frequency_table(
    claims_df: pd.DataFrame,
    unique_meta: dict[str, Any],
) -> pd.DataFrame:
    """All unique chains ranked by panel-level respondent frequency."""
    chain_meta = {
        int(c["chain_id"]): c for c in unique_meta.get("unique_causal_chains", [])
    }
    within = compute_within_group_chain_frequencies(claims_df)
    n_panel_respondents = int(claims_df["participant_name"].nunique())

    panel_freq = (
        claims_df.dropna(subset=["unique_chain_id"])
        .assign(unique_chain_id=lambda d: d["unique_chain_id"].astype(int))
        .groupby("unique_chain_id", as_index=False)
        .agg(
            n_respondents=("participant_name", "nunique"),
            n_claims=("claim_uid", "count"),
            representative_label=("representative_label", "first"),
        )
    )

    rows: list[dict[str, Any]] = []
    for _, row in panel_freq.iterrows():
        cid = int(row["unique_chain_id"])
        meta = chain_meta.get(cid, {})
        record: dict[str, Any] = {
            "unique_chain_id": cid,
            "representative_label": row.get("representative_label") or meta.get("representative_label", ""),
            "direction": meta.get("direction", ""),
            "antecedent_canonical": meta.get("antecedent_canonical", ""),
            "mechanism_canonical": meta.get("mechanism_canonical", ""),
            "outcome_canonical": meta.get("outcome_canonical", ""),
            "n_respondents": int(row["n_respondents"]),
            "n_claims": int(row["n_claims"]),
            "share_of_respondents": round(int(row["n_respondents"]) / n_panel_respondents, 4),
        }
        for group in GROUP_ORDER:
            sub = within[
                (within["unique_chain_id"] == cid) & (within["group_label"] == group)
            ]
            record[f"n_respondents_{group}"] = (
                int(sub["n_respondents"].iloc[0]) if len(sub) else 0
            )
        rows.append(record)

    out = pd.DataFrame(rows).sort_values(
        ["n_respondents", "n_claims", "unique_chain_id"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out


def summarize_panel_core_tail(frequency_df: pd.DataFrame) -> dict[str, float | int]:
    n = len(frequency_df)
    tail_n = int((frequency_df["n_respondents"] < CORE_MIN_RESPONDENTS).sum())
    core_n = n - tail_n
    return {
        "n_chains": n,
        "core_n": core_n,
        "tail_n": tail_n,
        "core_pct": 100.0 * core_n / n if n else 0.0,
        "tail_pct": 100.0 * tail_n / n if n else 0.0,
    }


def _group_rank_frequency(
    within_group_freq: pd.DataFrame,
    group: str,
    n_group_respondents: int,
) -> pd.DataFrame:
    gdf = within_group_freq[within_group_freq["group_label"] == group].copy()
    gdf = gdf.sort_values("n_respondents", ascending=False).reset_index(drop=True)
    gdf.insert(0, "rank", np.arange(1, len(gdf) + 1))
    if n_group_respondents > 0:
        gdf["share_respondents"] = gdf["n_respondents"] / n_group_respondents
    else:
        gdf["share_respondents"] = 0.0
    return gdf


def plot_chain_frequency_distribution(
    within_group_freq: pd.DataFrame,
    uptake_df: pd.DataFrame,
    diversity_df: pd.DataFrame,
    title: str,
    out_path: Path,
) -> None:
    """Per-group rank–frequency curves; y-axis = share of group respondents."""
    apply_plot_style()
    groups_present = [g for g in GROUP_ORDER if g in set(within_group_freq["group_label"])]
    if not groups_present:
        return

    group_n = diversity_df.set_index("group_label")["n_respondents"].to_dict()

    n_groups = len(groups_present)
    fig, axes = plt.subplots(1, n_groups, figsize=(5.8 * n_groups, 5.4), squeeze=False)
    axes_flat = axes.ravel()

    global_ymax = 0.05
    for group in groups_present:
        gdf = _group_rank_frequency(within_group_freq, group, group_n.get(group, 0))
        if not gdf.empty:
            global_ymax = max(global_ymax, gdf["share_respondents"].max() * 1.12)

    for ax, group in zip(axes_flat, groups_present):
        n_resp = int(group_n.get(group, 0))
        gdf = _group_rank_frequency(within_group_freq, group, n_resp)
        group_color = GROUP_COLORS_LOCAL[group]
        ranks = gdf["rank"].to_numpy()
        uptake = gdf["share_respondents"].to_numpy()
        is_core = gdf["n_respondents"].to_numpy() >= CORE_MIN_RESPONDENTS
        is_tail = ~is_core

        if is_core.any():
            ax.fill_between(
                ranks[is_core], uptake[is_core], 0,
                step="mid", alpha=0.15, color=group_color, zorder=1,
            )
        if is_tail.any():
            ax.fill_between(
                ranks[is_tail], uptake[is_tail], 0,
                step="mid", alpha=0.55, color=COLOR_TAIL_FILL, zorder=1,
            )

        first_tail_idx = int(np.argmax(is_tail)) if is_tail.any() else len(ranks)
        if first_tail_idx > 0:
            ax.plot(
                ranks[:first_tail_idx], uptake[:first_tail_idx],
                color=group_color, linewidth=1.8, zorder=2,
            )
        if is_tail.any():
            bridge = max(first_tail_idx - 1, 0)
            ax.plot(
                ranks[bridge:], uptake[bridge:],
                color=COLOR_TAIL, linewidth=1.8, zorder=2,
            )

        ax.scatter(
            ranks[is_core], uptake[is_core],
            s=28, color=group_color, edgecolors="white", linewidths=0.4, zorder=3,
            label=f"Core ({CORE_MIN_RESPONDENTS}+ respondents)",
        )
        ax.scatter(
            ranks[is_tail], uptake[is_tail],
            s=22, color=COLOR_TAIL, edgecolors="white", linewidths=0.4, zorder=3,
            label=f"Tail (<{CORE_MIN_RESPONDENTS} respondents)",
        )

        uptake_row = uptake_df[uptake_df["group_label"] == group]
        tail_pct = float(uptake_row["pct_tail"].iloc[0]) if len(uptake_row) else 0.0
        n_chains = int(uptake_row["n_chains_total"].iloc[0]) if len(uptake_row) else len(gdf)

        ax.set_xlabel("Chain rank (by respondent frequency)", fontsize=FONT_AXIS_LABEL - 1, fontweight="bold")
        ax.set_ylabel(
            "Share of group respondents",
            fontsize=FONT_AXIS_LABEL - 1, fontweight="bold",
        )
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        ax.set_title(
            f"{GROUP_DISPLAY[group]} (n={n_resp})\n"
            f"Rank–frequency ({n_chains} chains, tail {tail_pct:.0f}%)",
            fontsize=FONT_TITLE - 2, fontweight="bold", pad=10,
        )
        ax.set_xlim(0.5, max(len(gdf) + 0.5, 1.5))
        ax.set_ylim(0, global_ymax)
        ax.legend(loc="upper right", fontsize=FONT_LEGEND - 2, frameon=False)
        style_axes(ax)
        ax.grid(axis="y", alpha=0.2, zorder=0)

    fig.suptitle(
        f"{title}\nCore–tail structure of unique causal chains by group",
        fontsize=FONT_TITLE, fontweight="bold", y=0.98,
    )
    save_figure_with_caption(fig, out_path)


def plot_chain_frequency_by_group(
    diversity_df: pd.DataFrame,
    uptake_df: pd.DataFrame,
    title: str,
    out_path: Path,
) -> None:
    """Two panels: chain richness (metric A) + uptake inequality (Gini) per group."""
    apply_plot_style()
    groups_present = [g for g in GROUP_ORDER if g in set(diversity_df["group_label"])]
    if not groups_present:
        return

    fig, (ax_ratio, ax_uptake) = plt.subplots(1, 2, figsize=(11.5, 5.4))

    # --- Left: unique chains / respondent (metric A) ---
    div = diversity_df.set_index("group_label").loc[groups_present].reset_index()
    x = np.arange(len(div))
    colors = [GROUP_COLORS_LOCAL[g] for g in div["group_label"]]
    bars = ax_ratio.bar(
        x, div["unique_chains_per_respondent"], color=colors,
        edgecolor=BAR_EDGE_COLOR, linewidth=BAR_EDGE_WIDTH, alpha=0.95, zorder=3,
    )
    ax_ratio.set_xticks(x)
    ax_ratio.set_xticklabels(div["group_display"], fontsize=FONT_TICK - 1, fontweight="bold")
    ax_ratio.set_ylabel(
        "Unique causal chains / respondent",
        fontsize=FONT_AXIS_LABEL - 1, fontweight="bold",
    )
    ax_ratio.set_title(
        "Distinct chain types in group\n÷ number of respondents",
        fontsize=FONT_TITLE - 2, fontweight="bold", pad=10,
    )
    style_axes(ax_ratio)
    ymax = max(div["unique_chains_per_respondent"].max(), 0.1)
    ax_ratio.set_ylim(0, ymax * 1.22)
    for bar, val, n_resp, n_uc in zip(
        bars, div["unique_chains_per_respondent"], div["n_respondents"], div["n_unique_chains"]
    ):
        ax_ratio.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + ymax * 0.03,
            f"{val:.2f}\n({int(n_uc)} chains, n={int(n_resp)})",
            ha="center", va="bottom", fontsize=FONT_TICK - 1,
        )

    # --- Right: Gini coefficient of chain uptake (rank-frequency inequality) ---
    uptake = uptake_df.set_index("group_label").loc[groups_present].reset_index()
    x = np.arange(len(uptake))
    colors = [GROUP_COLORS_LOCAL[g] for g in uptake["group_label"]]
    bars = ax_uptake.bar(
        x, uptake["gini_uptake"], color=colors,
        edgecolor=BAR_EDGE_COLOR, linewidth=BAR_EDGE_WIDTH, alpha=0.95, zorder=3,
    )
    ax_uptake.set_xticks(x)
    ax_uptake.set_xticklabels(uptake["group_display"], fontsize=FONT_TICK - 1, fontweight="bold")
    ax_uptake.set_ylabel(
        "Gini coefficient",
        fontsize=FONT_AXIS_LABEL - 1, fontweight="bold",
    )
    ax_uptake.set_title(
        "Uptake inequality across\nchain types within group",
        fontsize=FONT_TITLE - 2, fontweight="bold", pad=10,
    )
    style_axes(ax_uptake)
    ax_uptake.set_ylim(0, 1.0)
    for bar, gini, tail_pct in zip(bars, uptake["gini_uptake"], uptake["pct_tail"]):
        ax_uptake.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.03,
            f"{gini:.2f}\n(tail {tail_pct:.0f}%)",
            ha="center", va="bottom", fontsize=FONT_TICK - 1,
        )

    fig.suptitle(
        f"{title}\nChain diversity by group",
        fontsize=FONT_TITLE, fontweight="bold", y=0.98,
    )
    save_figure_with_caption(fig, out_path)


def analyze_panel(panel_dir: Path) -> None:
    unique_meta = load_unique_chains(panel_dir)
    claims_df = load_claims_table(panel_dir)

    diversity_df = compute_group_diversity(claims_df)
    within_group_freq = compute_within_group_chain_frequencies(claims_df)
    uptake_df = compute_group_uptake_summary(within_group_freq)
    frequency_table = compute_chain_frequency_table(claims_df, unique_meta)

    title = panel_title(unique_meta, panel_dir.name)
    plot_chain_frequency_by_group(
        diversity_df, uptake_df, title, panel_dir / FREQUENCY_PNG
    )
    plot_chain_frequency_distribution(
        within_group_freq, uptake_df, diversity_df, title, panel_dir / FREQUENCY_DIST_PNG
    )
    frequency_table.to_csv(panel_dir / FREQUENCY_CSV, index=False)

    n_panel_chains = int(unique_meta.get("n_unique_chains", len(frequency_table)))
    n_panel_resp = int(claims_df["participant_name"].nunique())
    core_tail = summarize_panel_core_tail(frequency_table)
    print(f"\n=== {panel_dir.name} ===")
    print(f"Panel: {title}")
    print(f"Respondents: {n_panel_resp} | Unique chains (panel): {n_panel_chains}")
    print(
        f"Core–tail (panel): {core_tail['core_pct']:.1f}% core ({core_tail['core_n']} chains), "
        f"{core_tail['tail_pct']:.1f}% tail ({core_tail['tail_n']} chains)"
    )
    print("Group diversity (unique chains / respondent):")
    for _, row in diversity_df.iterrows():
        print(
            f"  {row['group_display']}: {row['unique_chains_per_respondent']:.3f} "
            f"({row['n_unique_chains']} chains / n={row['n_respondents']})"
        )
    print("Uptake inequality (Gini) and tail share:")
    for _, row in uptake_df.iterrows():
        print(
            f"  {row['group_display']}: Gini={row['gini_uptake']:.3f}, "
            f"tail={row['pct_tail']:.1f}% ({row['n_tail_chains']} tail / "
            f"{row['n_core_chains']} core chains)"
        )
    missing_groups = [GROUP_DISPLAY[g] for g in GROUP_ORDER if g not in set(claims_df["group_label"])]
    if missing_groups:
        print(f"Groups absent in this panel: {', '.join(missing_groups)}")
    print(f"Saved: {panel_dir / FREQUENCY_PNG}")
    print(f"Saved: {panel_dir / FREQUENCY_DIST_PNG}")
    print(f"Saved: {panel_dir / FREQUENCY_CSV} ({len(frequency_table)} chains)")


def main() -> None:
    panels = discover_panels()
    print(f"Analyzing {len(panels)} panel(s): {[p.name for p in panels]}")
    for panel_dir in panels:
        analyze_panel(panel_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
