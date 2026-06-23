"""
Analyze and visualize paragraph length for:
- Q Race.4
- Q Gender.4

Outputs:
- textual_analysis/outputs/length_theoretical_explanation_summary.csv
- textual_analysis/outputs/length_qrace4_by_group.png
- textual_analysis/outputs/length_qgender4_by_group.png
- textual_analysis/outputs/length_qrace4_by_group_human_genai.png
- textual_analysis/outputs/length_qgender4_by_group_human_genai.png
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

TEXTUAL_DIR = Path(__file__).resolve().parent
ROOT = TEXTUAL_DIR.parent
for p in (TEXTUAL_DIR, ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from stats_utils import bootstrap_mean_ci, p_value_welch_ttest
from viz_style import (
    BAR_ALPHA,
    BAR_EDGE_COLOR,
    BAR_EDGE_WIDTH,
    HUMAN_COMPOSITION_NOTE,
    add_legend,
    collapsed_legend_labels,
    ERROR_CAPSIZE,
    ERROR_LINEWIDTH,
    GROUP_COLORS_COLLAPSED,
    GROUP_COLORS_TEXT,
    GROUP_ORDER,
    GROUP_ORDER_COLLAPSED,
    apply_bottom_layout,
    apply_plot_style,
    comparison_pair_label,
    display_label,
    draw_sig_footnote,
    draw_snug_footer_comparison_box,
    legend_entry,
    save_figure,
    set_axis_labels,
    set_figure_title,
    style_axes,
)


CSV_PATH = Path(
    "All_Participants_All_Questions.csv"
)
OUT_DIR = Path(
    "textual_analysis/outputs"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_SUMMARY = OUT_DIR / "length_theoretical_explanation_summary.csv"
OUT_RACE = OUT_DIR / "length_qrace4_by_group.png"
OUT_GENDER = OUT_DIR / "length_qgender4_by_group.png"
OUT_RACE_COLLAPSED = OUT_DIR / "length_qrace4_by_group_human_genai.png"
OUT_GENDER_COLLAPSED = OUT_DIR / "length_qgender4_by_group_human_genai.png"

GROUP_MAP = {"0": "PhD Students", "1": "Experts", "2": "GenAI"}

FIGSIZE_THREE_GROUP = (8.2, 9.4)
FIGSIZE_COLLAPSED = (7.6, 8.8)
BAR_WIDTH = 0.55
LENGTH_FOOTNOTE_Y = 0.062

apply_plot_style()


def find_col(headers: list[str], prefix: str) -> int:
    exact = [i for i, h in enumerate(headers) if h.strip() == prefix]
    if len(exact) == 1:
        return exact[0]
    matches = [i for i, h in enumerate(headers) if h.strip().startswith(prefix)]
    if len(matches) != 1:
        raise ValueError(f"Expected one column for {prefix}, got {matches}")
    return matches[0]


Q_RACE_4_COL = "Q Race.4 pre-ML theory (main effects)"
Q_GENDER_4_COL = "Q Gender.4 pre-ML theory (main effects)"


def word_count(text: str) -> int:
    # Count words robustly for English-like text.
    return len(re.findall(r"\b\w+\b", text))


def summarize(arr: np.ndarray) -> dict[str, float]:
    if len(arr) == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "std": np.nan,
            "median": np.nan,
            "min": np.nan,
            "max": np.nan,
        }
    ci_low, ci_high = bootstrap_mean_ci(arr)
    return {
        "n": int(len(arr)),
        "mean": float(arr.mean()),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def collapse_lengths(lengths_by_group: dict[str, list[float]]) -> dict[str, list[float]]:
    return {
        "Human": lengths_by_group["PhD Students"] + lengths_by_group["Experts"],
        "GenAI": lengths_by_group["GenAI"],
    }


def three_group_comparisons(lengths_by_group: dict[str, np.ndarray]) -> list[tuple[str, float]]:
    return [
        (
            comparison_pair_label("PhD Students", "Experts"),
            p_value_welch_ttest(lengths_by_group["PhD Students"], lengths_by_group["Experts"]),
        ),
        (
            comparison_pair_label("PhD Students", "GenAI"),
            p_value_welch_ttest(lengths_by_group["PhD Students"], lengths_by_group["GenAI"]),
        ),
        (
            comparison_pair_label("Experts", "GenAI"),
            p_value_welch_ttest(lengths_by_group["Experts"], lengths_by_group["GenAI"]),
        ),
    ]


def collapsed_comparisons(lengths_by_group: dict[str, np.ndarray]) -> list[tuple[str, float]]:
    return [
        (
            comparison_pair_label("Human", "GenAI"),
            p_value_welch_ttest(lengths_by_group["Human"], lengths_by_group["GenAI"]),
        ),
    ]


def plot_metric(
    task: str,
    lengths_by_group: dict[str, np.ndarray],
    out_path: Path,
    *,
    groups: list[str],
    colors: dict[str, str],
    comparisons: list[tuple[str, float]],
    collapsed_legend: bool = False,
) -> None:
    x = np.arange(len(groups))

    means = [np.mean(lengths_by_group[g]) if len(lengths_by_group[g]) else np.nan for g in groups]
    cis = [bootstrap_mean_ci(lengths_by_group[g]) for g in groups]
    yerr_low = [m - lo if np.isfinite(m) and np.isfinite(lo) else 0 for m, (lo, _) in zip(means, cis)]
    yerr_high = [hi - m if np.isfinite(m) and np.isfinite(hi) else 0 for m, (_, hi) in zip(means, cis)]

    figsize = FIGSIZE_COLLAPSED if len(groups) == 2 else FIGSIZE_THREE_GROUP
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(
        x,
        means,
        width=BAR_WIDTH,
        color=[colors[g] for g in groups],
        edgecolor=BAR_EDGE_COLOR,
        linewidth=BAR_EDGE_WIDTH,
        alpha=BAR_ALPHA,
        zorder=2,
    )
    ax.errorbar(
        x,
        means,
        yerr=[yerr_low, yerr_high],
        fmt="none",
        ecolor="black",
        elinewidth=ERROR_LINEWIDTH,
        capsize=ERROR_CAPSIZE,
        zorder=3,
    )

    ax.set_xticks(x)
    ax.set_xticklabels([display_label(g) for g in groups])
    set_axis_labels(ax, None, "Word Count (Mean ± 95% CI)", bold_xticks=True)
    set_figure_title(ax, f"Theoretical Explanation Length - {task}")
    style_axes(ax)

    tops = [
        m + eh
        for m, eh in zip(means, yerr_high)
        if np.isfinite(m) and np.isfinite(eh)
    ]
    if tops:
        ax.set_ylim(0, max(tops) * 1.12)

    ns = {g: len(lengths_by_group[g]) for g in groups}
    if collapsed_legend:
        legend_labels = collapsed_legend_labels(groups, ns)
    else:
        legend_labels = [legend_entry(g, ns[g]) for g in groups]
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[g], alpha=BAR_ALPHA) for g in groups]
    legend_loc = "upper right" if collapsed_legend else "upper left"
    add_legend(ax, handles, legend_labels, loc=legend_loc)

    n_comp = len(comparisons)
    apply_bottom_layout(fig, n_lines=n_comp)
    draw_snug_footer_comparison_box(fig, comparisons)
    draw_sig_footnote(fig, y=LENGTH_FOOTNOTE_Y)
    save_figure(fig, out_path)
    print(f"Saved figure: {out_path}")


def main() -> None:
    with CSV_PATH.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    headers = rows[0]
    data = rows[1:]

    idx_group = find_col(headers, "student_0, expert_1, genAI_2")
    idx_name = find_col(headers, "What is your full name?")
    idx_r4 = find_col(headers, Q_RACE_4_COL)
    idx_g4 = find_col(headers, Q_GENDER_4_COL)

    lengths = {
        "Q Race.4": {g: [] for g in GROUP_ORDER},
        "Q Gender.4": {g: [] for g in GROUP_ORDER},
    }

    for r in data:
        gid = r[idx_group].strip() if len(r) > idx_group else ""
        gname = GROUP_MAP.get(gid)
        if gname is None:
            continue
        race_text = r[idx_r4].strip() if len(r) > idx_r4 else ""
        gender_text = r[idx_g4].strip() if len(r) > idx_g4 else ""
        if race_text:
            lengths["Q Race.4"][gname].append(word_count(race_text))
        if gender_text:
            lengths["Q Gender.4"][gname].append(word_count(gender_text))

    # Save summary csv
    fieldnames = [
        "question",
        "group",
        "n",
        "mean",
        "ci_low",
        "ci_high",
        "std",
        "median",
        "min",
        "max",
    ]
    collapsed_lengths = {
        q: collapse_lengths(lengths[q]) for q in ("Q Race.4", "Q Gender.4")
    }

    out_rows = []
    for q in ("Q Race.4", "Q Gender.4"):
        for g in GROUP_ORDER:
            arr = np.asarray(lengths[q][g], dtype=float)
            s = summarize(arr)
            out_rows.append({"question": q, "group": g, **s})
        for g in GROUP_ORDER_COLLAPSED:
            arr = np.asarray(collapsed_lengths[q][g], dtype=float)
            s = summarize(arr)
            out_rows.append({"question": f"{q} (collapsed)", "group": g, **s})

    with OUT_SUMMARY.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"Saved summary: {OUT_SUMMARY}")

    race_arr = {g: np.asarray(lengths["Q Race.4"][g], dtype=float) for g in GROUP_ORDER}
    gender_arr = {g: np.asarray(lengths["Q Gender.4"][g], dtype=float) for g in GROUP_ORDER}
    race_collapsed = {g: np.asarray(collapsed_lengths["Q Race.4"][g], dtype=float) for g in GROUP_ORDER_COLLAPSED}
    gender_collapsed = {g: np.asarray(collapsed_lengths["Q Gender.4"][g], dtype=float) for g in GROUP_ORDER_COLLAPSED}

    plot_metric(
        "Race", race_arr, OUT_RACE,
        groups=GROUP_ORDER,
        colors=GROUP_COLORS_TEXT,
        comparisons=three_group_comparisons(race_arr),
    )
    plot_metric(
        "Gender", gender_arr, OUT_GENDER,
        groups=GROUP_ORDER,
        colors=GROUP_COLORS_TEXT,
        comparisons=three_group_comparisons(gender_arr),
    )
    plot_metric(
        "Race", race_collapsed, OUT_RACE_COLLAPSED,
        groups=GROUP_ORDER_COLLAPSED,
        colors=GROUP_COLORS_COLLAPSED,
        comparisons=collapsed_comparisons(race_collapsed),
        collapsed_legend=True,
    )
    plot_metric(
        "Gender", gender_collapsed, OUT_GENDER_COLLAPSED,
        groups=GROUP_ORDER_COLLAPSED,
        colors=GROUP_COLORS_COLLAPSED,
        comparisons=collapsed_comparisons(gender_collapsed),
        collapsed_legend=True,
    )

    print("\nQuick summary (word count mean [95% CI]):")
    for q, group_order in (("Q Race.4", GROUP_ORDER), ("Q Gender.4", GROUP_ORDER)):
        print(f"\n{q}")
        for g in group_order:
            arr = np.asarray(lengths[q][g], dtype=float)
            lo, hi = bootstrap_mean_ci(arr)
            mean = float(np.mean(arr)) if len(arr) else np.nan
            print(
                f"  {display_label(g):<14} n={len(arr):>2}  mean={mean:.2f}  [{lo:.2f}, {hi:.2f}]"
            )

    for q in ("Q Race.4", "Q Gender.4"):
        print(f"\n{q} (Humans: {HUMAN_COMPOSITION_NOTE})")
        for g in GROUP_ORDER_COLLAPSED:
            arr = np.asarray(collapsed_lengths[q][g], dtype=float)
            lo, hi = bootstrap_mean_ci(arr)
            mean = float(np.mean(arr)) if len(arr) else np.nan
            print(
                f"  {display_label(g):<10} n={len(arr):>2}  mean={mean:.2f}  [{lo:.2f}, {hi:.2f}]"
            )

    # Missing checks
    miss_r = [r[idx_name] for r in data if not (len(r) > idx_r4 and r[idx_r4].strip())]
    miss_g = [r[idx_name] for r in data if not (len(r) > idx_g4 and r[idx_g4].strip())]
    if miss_r:
        print("\nMissing Q Race.4:", ", ".join(miss_r))
    if miss_g:
        print("Missing Q Gender.4:", ", ".join(miss_g))


if __name__ == "__main__":
    main()

