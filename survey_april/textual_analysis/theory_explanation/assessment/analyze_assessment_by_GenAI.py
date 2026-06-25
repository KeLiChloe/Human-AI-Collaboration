"""
Assess theory quality scores by group (Experts, PhD Students, GenAI).

This script reads overall quality scores (average of 5 dimensions) and
generates publication-style figures:
- Race / Gender: Pre-ML + Post-ML, three groups
- Race / Gender: Pre-ML + Post-ML, Human (PhD + Experts) vs GenAI
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable
import sys

import matplotlib.pyplot as plt
import numpy as np

ASSESSMENT_DIR = Path(__file__).resolve().parent
TEXTUAL_DIR = ASSESSMENT_DIR.parent.parent
ROOT = TEXTUAL_DIR.parent
for p in (ASSESSMENT_DIR, TEXTUAL_DIR, ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from stats_utils import (
    bootstrap_mean_ci,
    p_value_paired_ttest_pairs,
    p_value_welch_ttest,
)
from viz_style import (
    ASSESSMENT_SIG_FOOTNOTE,
    BAR_ALPHA,
    BAR_EDGE_COLOR,
    BAR_EDGE_WIDTH,
    HUMAN_COMPOSITION_NOTE,
    ERROR_CAPSIZE,
    ERROR_LINEWIDTH,
    GROUP_COLORS_COLLAPSED,
    GROUP_COLORS_TEXT,
    GROUP_ORDER,
    GROUP_ORDER_COLLAPSED,
    PHASE_HATCH_COLOR,
    add_legend,
    apply_bottom_layout,
    apply_plot_style,
    comparison_pair_label,
    display_label,
    draw_paired_pre_post_bracket,
    draw_phase_comparison_box,
    draw_sig_footnote,
    collapsed_legend_labels,
    legend_entry,
    phase_center_x,
    save_figure,
    set_axis_labels,
    set_figure_title,
    snug_comparison_box_width,
    style_axes,
)

ASSESSMENT_FOOTNOTE_Y = 0.062
ASSESSMENT_PHASE_BOX_MAX_WIDTH = 0.28
ASSESSMENT_SCORE_YMAX = 10
ASSESSMENT_PLOT_YMAX = 11.8


CSV_PATH = ROOT / "All_Participants_All_Questions.csv"
OUT_DIR = ROOT / "textual_analysis" / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TASK_SPECS = [
    (
        "Race",
        {
            "Pre-ML": "Q Race.4 Overall Quality Score",
            "Post-ML": "Q Race.12 Overall Quality Score",
        },
        OUT_DIR / "assessment_race_pre_post_ml.png",
        OUT_DIR / "assessment_race_pre_post_ml_human_genai.png",
    ),
    (
        "Gender",
        {
            "Pre-ML": "Q Gender.4 Overall Quality Score",
            "Post-ML": "Q Gender.12 Overall Quality Score",
        },
        OUT_DIR / "assessment_gender_pre_post_ml.png",
        OUT_DIR / "assessment_gender_pre_post_ml_human_genai.png",
    ),
]

GROUP_MAP = {
    "0": "PhD Students",
    "1": "Experts",
    "2": "GenAI",
}

apply_plot_style()


def to_float(x: str) -> float | None:
    s = str(x).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def find_col_idx(headers: list[str], prefix: str) -> int:
    matches = [i for i, h in enumerate(headers) if h.strip().startswith(prefix)]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one column for prefix '{prefix}', got {matches}")
    return matches[0]


def summarize(values: Iterable[float]) -> dict[str, float]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return {"n": 0, "mean": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    ci_low, ci_high = bootstrap_mean_ci(arr)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def collapse_grouped_values(grouped_values: dict[str, list[float]]) -> dict[str, list[float]]:
    return {
        "Human": grouped_values["PhD Students"] + grouped_values["Experts"],
        "GenAI": grouped_values["GenAI"],
    }


def collapse_paired_by_group(
    paired_by_group: dict[str, list[tuple[float, float]]],
) -> dict[str, list[tuple[float, float]]]:
    return {
        "Human": paired_by_group["PhD Students"] + paired_by_group["Experts"],
        "GenAI": paired_by_group["GenAI"],
    }


def load_phase_and_paired_values(
    headers: list[str],
    data: list[list[str]],
    group_col: int,
    phase_map: dict[str, str],
) -> tuple[dict[str, dict[str, list[float]]], dict[str, list[tuple[float, float]]]]:
    pre_col = find_col_idx(headers, phase_map["Pre-ML"])
    post_col = find_col_idx(headers, phase_map["Post-ML"])
    phase_to_grouped_values = {
        "Pre-ML": {g: [] for g in GROUP_ORDER},
        "Post-ML": {g: [] for g in GROUP_ORDER},
    }
    paired_by_group: dict[str, list[tuple[float, float]]] = {g: [] for g in GROUP_ORDER}

    for row in data:
        gid = row[group_col].strip() if len(row) > group_col else ""
        gname = GROUP_MAP.get(gid)
        if gname is None:
            continue
        pre = to_float(row[pre_col]) if len(row) > pre_col else None
        post = to_float(row[post_col]) if len(row) > post_col else None
        if pre is not None:
            phase_to_grouped_values["Pre-ML"][gname].append(pre)
        if post is not None:
            phase_to_grouped_values["Post-ML"][gname].append(post)
        if pre is not None and post is not None:
            paired_by_group[gname].append((pre, post))

    return phase_to_grouped_values, paired_by_group


def _draw_group_pre_post_brackets(
    ax,
    groups: list[str],
    x: np.ndarray,
    width: float,
    offsets: dict[str, float],
    bar_tops: dict[str, float],
    paired_by_group: dict[str, list[tuple[float, float]]],
) -> None:
    for gi, group in enumerate(groups):
        x_pre = float(x[gi] + offsets["Pre-ML"])
        x_post = float(x[gi] + offsets["Post-ML"])
        p_val = p_value_paired_ttest_pairs(paired_by_group.get(group, []))
        draw_paired_pre_post_bracket(ax, x_pre, x_post, bar_tops[group], p_val)


def plot_pre_post_in_one(
    task: str,
    phase_to_grouped_values: dict[str, dict[str, list[float]]],
    paired_by_group: dict[str, list[tuple[float, float]]],
    out_path: Path,
) -> None:
    phase_order = ["Pre-ML", "Post-ML"]
    x = np.arange(len(GROUP_ORDER))
    width = 0.34
    offsets = {"Pre-ML": -width / 2, "Post-ML": width / 2}

    fig, ax = plt.subplots(1, 1, figsize=(8.2, 9.4))

    group_ns: dict[str, int] = {}
    bar_tops: dict[str, float] = {}
    for group in GROUP_ORDER:
        means = []
        yerr_low = []
        yerr_high = []
        for phase in phase_order:
            stats = summarize(phase_to_grouped_values[phase][group])
            mean = float(stats["mean"])
            lo = float(stats["ci_low"])
            hi = float(stats["ci_high"])
            means.append(mean)
            yerr_low.append(max(0.0, mean - lo) if np.isfinite(mean) and np.isfinite(lo) else 0.0)
            yerr_high.append(max(0.0, hi - mean) if np.isfinite(mean) and np.isfinite(hi) else 0.0)
            group_ns[group] = int(stats["n"])
        bar_tops[group] = max(means[i] + yerr_high[i] for i in range(len(phase_order)))

        for i, phase in enumerate(phase_order):
            xpos = x[GROUP_ORDER.index(group)] + offsets[phase]
            bar = ax.bar(
                [xpos],
                [means[i]],
                width=width,
                color=GROUP_COLORS_TEXT[group],
                alpha=BAR_ALPHA,
                edgecolor=BAR_EDGE_COLOR,
                linewidth=BAR_EDGE_WIDTH,
                zorder=2,
            )
            if phase == "Post-ML":
                bar[0].set_hatch("///")
            ax.errorbar(
                [xpos],
                [means[i]],
                yerr=[[yerr_low[i]], [yerr_high[i]]],
                fmt="none",
                ecolor="black",
                elinewidth=ERROR_LINEWIDTH,
                capsize=ERROR_CAPSIZE,
                zorder=3,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([display_label(g) for g in GROUP_ORDER])
    set_axis_labels(ax, None, "Overall Quality Score (Mean ± 95% CI)", bold_xticks=True)
    set_figure_title(ax, f"Theory Quality by Group - {task}")
    style_axes(ax)
    ax.set_ylim(0, ASSESSMENT_PLOT_YMAX)
    ax.set_yticks(list(range(0, ASSESSMENT_SCORE_YMAX + 1, 2)))
    _draw_group_pre_post_brackets(
        ax, GROUP_ORDER, x, width, offsets, bar_tops, paired_by_group
    )

    group_legend_labels = [legend_entry(g, group_ns.get(g, 0)) for g in GROUP_ORDER]
    group_handles = [
        plt.Rectangle((0, 0), 1, 1, color=GROUP_COLORS_TEXT[g], alpha=BAR_ALPHA)
        for g in GROUP_ORDER
    ]
    phase_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=PHASE_HATCH_COLOR, alpha=BAR_ALPHA, edgecolor=BAR_EDGE_COLOR),
        plt.Rectangle((0, 0), 1, 1, facecolor=PHASE_HATCH_COLOR, alpha=BAR_ALPHA, edgecolor=BAR_EDGE_COLOR, hatch="///"),
    ]
    phase_labels = ["Pre-ML", "Post-ML"]
    legend1 = add_legend(ax, group_handles, group_legend_labels, loc="upper left")
    ax.add_artist(legend1)
    add_legend(ax, phase_handles, phase_labels, loc="upper right")

    n_lines = 4  # phase header + 3 pairwise comparisons
    apply_bottom_layout(fig, n_lines=n_lines)
    phase_lines: dict[str, list[tuple[str, float]]] = {}
    for phase in phase_order:
        g_phd = np.asarray(phase_to_grouped_values[phase]["PhD Students"], dtype=float)
        g_exp = np.asarray(phase_to_grouped_values[phase]["Experts"], dtype=float)
        g_gen = np.asarray(phase_to_grouped_values[phase]["GenAI"], dtype=float)
        phase_lines[phase] = [
            (
                comparison_pair_label("Experts", "PhD Students"),
                p_value_welch_ttest(g_exp, g_phd),
            ),
            (
                comparison_pair_label("PhD Students", "GenAI"),
                p_value_welch_ttest(g_phd, g_gen),
            ),
            (
                comparison_pair_label("Experts", "GenAI"),
                p_value_welch_ttest(g_exp, g_gen),
            ),
        ]
    fig.canvas.draw()
    layout_width = max(
        snug_comparison_box_width(fig, lines) for lines in phase_lines.values()
    )
    phase_centers = phase_center_x(layout_width)
    for phase, lines in phase_lines.items():
        draw_phase_comparison_box(fig, phase, phase_centers[phase], lines)
    draw_sig_footnote(fig, y=ASSESSMENT_FOOTNOTE_Y, text=ASSESSMENT_SIG_FOOTNOTE)
    save_figure(fig, out_path)
    print(f"Saved figure: {out_path}")


def plot_pre_post_collapsed(
    task: str,
    phase_to_grouped_values: dict[str, dict[str, list[float]]],
    paired_by_group: dict[str, list[tuple[float, float]]],
    out_path: Path,
) -> None:
    """Pre-ML + Post-ML with PhD Students + Experts collapsed into Human vs GenAI."""
    phase_order = ["Pre-ML", "Post-ML"]
    x = np.arange(len(GROUP_ORDER_COLLAPSED))
    width = 0.34
    offsets = {"Pre-ML": -width / 2, "Post-ML": width / 2}

    fig, ax = plt.subplots(1, 1, figsize=(7.6, 8.8))
    group_ns: dict[str, int] = {}
    bar_tops: dict[str, float] = {}

    for group in GROUP_ORDER_COLLAPSED:
        means = []
        yerr_low = []
        yerr_high = []
        for phase in phase_order:
            stats = summarize(phase_to_grouped_values[phase][group])
            mean = float(stats["mean"])
            lo = float(stats["ci_low"])
            hi = float(stats["ci_high"])
            means.append(mean)
            yerr_low.append(max(0.0, mean - lo) if np.isfinite(mean) and np.isfinite(lo) else 0.0)
            yerr_high.append(max(0.0, hi - mean) if np.isfinite(mean) and np.isfinite(hi) else 0.0)
            group_ns[group] = int(stats["n"])
        bar_tops[group] = max(means[i] + yerr_high[i] for i in range(len(phase_order)))

        for i, phase in enumerate(phase_order):
            xpos = x[GROUP_ORDER_COLLAPSED.index(group)] + offsets[phase]
            bar = ax.bar(
                [xpos],
                [means[i]],
                width=width,
                color=GROUP_COLORS_COLLAPSED[group],
                alpha=BAR_ALPHA,
                edgecolor=BAR_EDGE_COLOR,
                linewidth=BAR_EDGE_WIDTH,
                zorder=2,
            )
            if phase == "Post-ML":
                bar[0].set_hatch("///")
            ax.errorbar(
                [xpos],
                [means[i]],
                yerr=[[yerr_low[i]], [yerr_high[i]]],
                fmt="none",
                ecolor="black",
                elinewidth=ERROR_LINEWIDTH,
                capsize=ERROR_CAPSIZE,
                zorder=3,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([display_label(g) for g in GROUP_ORDER_COLLAPSED])
    set_axis_labels(ax, None, "Overall Quality Score (Mean ± 95% CI)", bold_xticks=True)
    set_figure_title(ax, f"Theory Quality by Group - {task}")
    style_axes(ax)
    ax.set_ylim(0, ASSESSMENT_PLOT_YMAX)
    ax.set_yticks(list(range(0, ASSESSMENT_SCORE_YMAX + 1, 2)))
    _draw_group_pre_post_brackets(
        ax, GROUP_ORDER_COLLAPSED, x, width, offsets, bar_tops, paired_by_group
    )

    group_legend_labels = collapsed_legend_labels(GROUP_ORDER_COLLAPSED, group_ns)
    group_handles = [
        plt.Rectangle((0, 0), 1, 1, color=GROUP_COLORS_COLLAPSED[g], alpha=BAR_ALPHA)
        for g in GROUP_ORDER_COLLAPSED
    ]
    phase_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=PHASE_HATCH_COLOR, alpha=BAR_ALPHA, edgecolor=BAR_EDGE_COLOR),
        plt.Rectangle((0, 0), 1, 1, facecolor=PHASE_HATCH_COLOR, alpha=BAR_ALPHA, edgecolor=BAR_EDGE_COLOR, hatch="///"),
    ]
    phase_labels = ["Pre-ML", "Post-ML"]
    legend1 = add_legend(ax, group_handles, group_legend_labels, loc="upper left")
    ax.add_artist(legend1)
    add_legend(ax, phase_handles, phase_labels, loc="upper right")

    n_lines = 2  # phase header + 1 pairwise comparison
    apply_bottom_layout(fig, n_lines=n_lines)
    phase_lines: dict[str, list[tuple[str, float]]] = {}
    for phase in phase_order:
        g_human = np.asarray(phase_to_grouped_values[phase]["Human"], dtype=float)
        g_genai = np.asarray(phase_to_grouped_values[phase]["GenAI"], dtype=float)
        phase_lines[phase] = [
            (
                comparison_pair_label("Human", "GenAI"),
                p_value_welch_ttest(g_human, g_genai),
            )
        ]

    fig.canvas.draw()
    layout_width = max(
        snug_comparison_box_width(
            fig, lines, max_width=ASSESSMENT_PHASE_BOX_MAX_WIDTH
        )
        for lines in phase_lines.values()
    )
    phase_centers = phase_center_x(layout_width)
    for phase, lines in phase_lines.items():
        draw_phase_comparison_box(fig, phase, phase_centers[phase], lines)
    draw_sig_footnote(fig, y=ASSESSMENT_FOOTNOTE_Y, text=ASSESSMENT_SIG_FOOTNOTE)
    save_figure(fig, out_path)
    print(f"Saved figure: {out_path}")


def main() -> None:
    with CSV_PATH.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    headers = rows[0]
    data = rows[1:]

    group_col = find_col_idx(headers, "student_0, expert_1, genAI_2")

    for task, phase_map, out_path, collapsed_out_path in TASK_SPECS:
        phase_to_grouped_values, paired_by_group = load_phase_and_paired_values(
            headers, data, group_col, phase_map
        )

        print(f"\n{task}")
        for phase in ("Pre-ML", "Post-ML"):
            print(f"  {phase}")
            for g in GROUP_ORDER:
                s = summarize(phase_to_grouped_values[phase][g])
                print(
                    f"    {display_label(g):<14} n={int(s['n']):>2}  mean={s['mean']:.3f}  "
                    f"[{s['ci_low']:.3f}, {s['ci_high']:.3f}]"
                )

        for g in GROUP_ORDER:
            pairs = paired_by_group[g]
            if pairs:
                p_within = p_value_paired_ttest_pairs(pairs)
                print(f"  Pre vs Post ({display_label(g)}): p = {p_within:.4f}")

        plot_pre_post_in_one(task, phase_to_grouped_values, paired_by_group, out_path)

        collapsed_phase_values = {
            phase: collapse_grouped_values(values)
            for phase, values in phase_to_grouped_values.items()
        }
        collapsed_paired = collapse_paired_by_group(paired_by_group)
        print(f"\n{task} (Humans: {HUMAN_COMPOSITION_NOTE})")
        for phase in ("Pre-ML", "Post-ML"):
            print(f"  {phase}")
            for g in GROUP_ORDER_COLLAPSED:
                s = summarize(collapsed_phase_values[phase][g])
                print(
                    f"    {display_label(g):<10} n={int(s['n']):>2}  mean={s['mean']:.3f}  "
                    f"[{s['ci_low']:.3f}, {s['ci_high']:.3f}]"
                )

        plot_pre_post_collapsed(task, collapsed_phase_values, collapsed_paired, collapsed_out_path)


if __name__ == "__main__":
    main()
