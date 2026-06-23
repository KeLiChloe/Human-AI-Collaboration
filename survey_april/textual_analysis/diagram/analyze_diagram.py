"""
Analyze Q5 diagram structure metrics by group.

Metrics:
- Number of paths
- Maximum path length
- Number of latent variables
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

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
    COMPARE_LINE_STEP_METRIC,
    ERROR_CAPSIZE,
    ERROR_LINEWIDTH,
    GROUP_COLORS_COLLAPSED,
    GROUP_COLORS_TEXT,
    GROUP_ORDER,
    GROUP_ORDER_COLLAPSED,
    apply_plot_style,
    comparison_pair_label,
    display_label,
    finalize_metric_figure,
    format_comparison_line,
    legend_entry,
    set_axis_labels,
    set_figure_title,
    style_axes,
)


CSV_PATH = ROOT / "All_Participants_All_Questions.csv"
OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FIG_RACE = OUT_DIR / "diagram_group_mean_ci_race.png"
OUT_FIG_GENDER = OUT_DIR / "diagram_group_mean_ci_gender.png"
OUT_FIG_RACE_HUMAN_GENAI = OUT_DIR / "diagram_group_mean_ci_race_human_genai.png"
OUT_FIG_GENDER_HUMAN_GENAI = OUT_DIR / "diagram_group_mean_ci_gender_human_genai.png"


GROUP_MAP = {
    "0": "PhD Students",
    "1": "Experts",
    "2": "GenAI",
}

GROUP_MAP_COLLAPSED = {
    "0": "Human",
    "1": "Human",
    "2": "GenAI",
}

METRIC_ORDER = [
    "Number of paths",
    "Maximum path length",
    "Number of latent variables",
]

METRIC_DISPLAY = {
    "Number of paths": "Number of\npaths",
    "Maximum path length": "Maximum\npath length",
    "Number of latent variables": "Number of\nlatent variables",
}

FIGSIZE_THREE_GROUP = (10.0, 7.4)
FIGSIZE_COLLAPSED = (8.5, 6.0)

LAYOUT_THREE_GROUP = dict(left=0.08, right=0.98, top=0.87, bottom=0.40)
LAYOUT_COLLAPSED = dict(left=0.10, right=0.98, top=0.87, bottom=0.33)

COMPARE_Y_THREE_GROUP = -0.24
COMPARE_Y_COLLAPSED = -0.22
DIAGRAM_FOOTNOTE_Y = 0.062
DIAGRAM_Y_HEADROOM = 1.2

apply_plot_style()


def apply_diagram_ylim(ax) -> None:
    """Extend y-axis above bars/error bars for legend clearance."""
    _, ymax = ax.get_ylim()
    if ymax > 0:
        ax.set_ylim(0, ymax * DIAGRAM_Y_HEADROOM)


@dataclass(frozen=True)
class MetricDef:
    task: str
    label: str
    prefix: str


METRICS: list[MetricDef] = [
    MetricDef("Race", "Number of paths", "Q Race.5 Number of paths"),
    MetricDef("Race", "Maximum path length", "Q Race.5 Maximum path length"),
    MetricDef("Race", "Number of latent variables", "Q Race.5 Number of latent variables"),
    MetricDef("Gender", "Number of paths", "Q Gender.5 Number of paths"),
    MetricDef("Gender", "Maximum path length", "Q Gender.5 Maximum path length"),
    MetricDef("Gender", "Number of latent variables", "Q Gender.5 Number of latent variables"),
]


def to_float(x: str) -> float | None:
    s = x.strip()
    if not s:
        return None

    try:
        return float(s)
    except ValueError:
        return None


def find_col_idx(headers: list[str], prefix: str) -> int:
    prefix_clean = prefix.strip().lower()

    exact_matches = [
        i for i, h in enumerate(headers)
        if h.strip().lower().startswith(prefix_clean)
    ]

    if len(exact_matches) == 1:
        return exact_matches[0]

    loose_matches = [
        i for i, h in enumerate(headers)
        if prefix_clean in h.strip().lower()
    ]

    if len(loose_matches) == 1:
        return loose_matches[0]

    raise ValueError(
        f"Expected exactly one column for prefix '{prefix}', "
        f"got exact={exact_matches}, loose={loose_matches}"
    )


def summarize(values: Iterable[float]) -> dict[str, float]:
    arr = np.asarray(list(values), dtype=float)

    if arr.size == 0:
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
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def plot_mean_ci_single_task(
    out_rows: list[dict[str, object]],
    task: str,
    out_path: Path,
    raw_values_by_task_metric_group: dict[tuple[str, str, str], np.ndarray],
) -> None:
    x = np.arange(len(METRIC_ORDER))
    width = 0.24

    offsets = {
        "PhD Students": -width,
        "Experts": 0.0,
        "GenAI": width,
    }

    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE_THREE_GROUP)
    group_ns: dict[str, int] = {}

    for group in GROUP_ORDER:
        means = []
        yerr_low = []
        yerr_high = []

        for metric in METRIC_ORDER:
            rec = next(
                r for r in out_rows
                if r["task"] == task
                and r["metric"] == metric
                and r["group"] == group
            )

            mean = float(rec["mean"])
            lo = float(rec["ci_low"])
            hi = float(rec["ci_high"])
            n = int(rec["n"])

            means.append(mean)
            yerr_low.append(max(0.0, mean - lo) if np.isfinite(mean) and np.isfinite(lo) else 0.0)
            yerr_high.append(max(0.0, hi - mean) if np.isfinite(mean) and np.isfinite(hi) else 0.0)
            group_ns[group] = n

        xpos = x + offsets[group]

        ax.bar(
            xpos,
            means,
            width=width,
            color=GROUP_COLORS_TEXT[group],
            alpha=BAR_ALPHA,
            edgecolor=BAR_EDGE_COLOR,
            linewidth=BAR_EDGE_WIDTH,
            label=group,
            zorder=2,
        )

        ax.errorbar(
            xpos,
            means,
            yerr=[yerr_low, yerr_high],
            fmt="none",
            ecolor="black",
            elinewidth=ERROR_LINEWIDTH,
            capsize=ERROR_CAPSIZE,
            zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_DISPLAY[m] for m in METRIC_ORDER])
    set_figure_title(ax, f"Diagram Metrics (Pre-ML) - {task}")
    set_axis_labels(ax, None, "Mean ± 95% CI", bold_xticks=True)
    style_axes(ax)
    apply_diagram_ylim(ax)

    comparisons: list[tuple[int, list[str], list[float], float, float]] = []
    for i, metric in enumerate(METRIC_ORDER):
        g_phd = raw_values_by_task_metric_group[(task, metric, "PhD Students")]
        g_exp = raw_values_by_task_metric_group[(task, metric, "Experts")]
        g_gen = raw_values_by_task_metric_group[(task, metric, "GenAI")]

        p_phd_exp = p_value_welch_ttest(g_phd, g_exp)
        p_phd_gen = p_value_welch_ttest(g_phd, g_gen)
        p_exp_gen = p_value_welch_ttest(g_exp, g_gen)

        comparisons.append(
            (
                i,
                [
                    format_comparison_line(
                        comparison_pair_label("PhD Students", "Experts"), p_phd_exp
                    ),
                    format_comparison_line(
                        comparison_pair_label("PhD Students", "GenAI"), p_phd_gen
                    ),
                    format_comparison_line(
                        comparison_pair_label("Experts", "GenAI"), p_exp_gen
                    ),
                ],
                [p_phd_exp, p_phd_gen, p_exp_gen],
                COMPARE_Y_THREE_GROUP,
                COMPARE_LINE_STEP_METRIC,
            )
        )

    legend_labels = [legend_entry(g, group_ns.get(g, 0)) for g in GROUP_ORDER]
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=GROUP_COLORS_TEXT[g], alpha=BAR_ALPHA)
        for g in GROUP_ORDER
    ]

    add_legend(ax, handles, legend_labels, loc="upper right")

    finalize_metric_figure(
        fig,
        ax,
        out_path,
        layout=LAYOUT_THREE_GROUP,
        comparisons=comparisons,
        footnote_y=DIAGRAM_FOOTNOTE_Y,
    )


def plot_mean_ci_single_task_collapsed(
    out_rows: list[dict[str, object]],
    task: str,
    out_path: Path,
    raw_values_by_task_metric_group: dict[tuple[str, str, str], np.ndarray],
) -> None:
    x = np.arange(len(METRIC_ORDER))
    width = 0.34

    offsets = {
        "Human": -width / 2,
        "GenAI": width / 2,
    }

    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE_COLLAPSED)
    group_ns: dict[str, int] = {}

    for group in GROUP_ORDER_COLLAPSED:
        means = []
        yerr_low = []
        yerr_high = []

        for metric in METRIC_ORDER:
            rec = next(
                r for r in out_rows
                if r["task"] == task
                and r["metric"] == metric
                and r["group"] == group
            )

            mean = float(rec["mean"])
            lo = float(rec["ci_low"])
            hi = float(rec["ci_high"])
            n = int(rec["n"])

            means.append(mean)
            yerr_low.append(max(0.0, mean - lo) if np.isfinite(mean) and np.isfinite(lo) else 0.0)
            yerr_high.append(max(0.0, hi - mean) if np.isfinite(mean) and np.isfinite(hi) else 0.0)
            group_ns[group] = n

        xpos = x + offsets[group]

        ax.bar(
            xpos,
            means,
            width=width,
            color=GROUP_COLORS_COLLAPSED[group],
            alpha=BAR_ALPHA,
            edgecolor=BAR_EDGE_COLOR,
            linewidth=BAR_EDGE_WIDTH,
            label=group,
            zorder=2,
        )

        ax.errorbar(
            xpos,
            means,
            yerr=[yerr_low, yerr_high],
            fmt="none",
            ecolor="black",
            elinewidth=ERROR_LINEWIDTH,
            capsize=ERROR_CAPSIZE,
            zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_DISPLAY[m] for m in METRIC_ORDER])
    set_figure_title(ax, f"Diagram Metrics (Pre-ML) - {task}")
    set_axis_labels(ax, None, "Mean ± 95% CI", bold_xticks=True)
    style_axes(ax)
    apply_diagram_ylim(ax)

    comparisons: list[tuple[int, list[str], list[float], float, float]] = []
    for i, metric in enumerate(METRIC_ORDER):
        g_human = raw_values_by_task_metric_group[(task, metric, "Human")]
        g_genai = raw_values_by_task_metric_group[(task, metric, "GenAI")]

        p_human_genai = p_value_welch_ttest(g_human, g_genai)

        comparisons.append(
            (
                i,
                [
                    format_comparison_line(
                        comparison_pair_label("Human", "GenAI"), p_human_genai
                    )
                ],
                [p_human_genai],
                COMPARE_Y_COLLAPSED,
                COMPARE_LINE_STEP_METRIC,
            )
        )

    legend_labels = collapsed_legend_labels(GROUP_ORDER_COLLAPSED, group_ns)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=GROUP_COLORS_COLLAPSED[g], alpha=BAR_ALPHA)
        for g in GROUP_ORDER_COLLAPSED
    ]

    add_legend(ax, handles, legend_labels, loc="upper right")

    finalize_metric_figure(
        fig,
        ax,
        out_path,
        layout=LAYOUT_COLLAPSED,
        comparisons=comparisons,
        footnote_y=DIAGRAM_FOOTNOTE_Y,
    )


def build_grouped_values_collapsed(
    data: list[list[str]],
    group_col: int,
    metric_cols: dict[MetricDef, int],
) -> tuple[list[dict[str, object]], dict[tuple[str, str, str], np.ndarray]]:
    out_rows: list[dict[str, object]] = []
    raw_values: dict[tuple[str, str, str], np.ndarray] = {}

    for m in METRICS:
        col = metric_cols[m]
        grouped_values: dict[str, list[float]] = {
            g: [] for g in GROUP_ORDER_COLLAPSED
        }

        for r in data:
            gid = r[group_col].strip() if len(r) > group_col else ""
            gname = GROUP_MAP_COLLAPSED.get(gid)

            if gname is None:
                continue

            v = to_float(r[col]) if len(r) > col else None

            if v is not None:
                grouped_values[gname].append(v)

        for g in GROUP_ORDER_COLLAPSED:
            s = summarize(grouped_values[g])
            out_rows.append({
                "task": m.task,
                "metric": m.label,
                "group": g,
                **s,
            })
            raw_values[(m.task, m.label, g)] = np.asarray(grouped_values[g], dtype=float)

    return out_rows, raw_values


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV file not found: {CSV_PATH}")

    with CSV_PATH.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))

    if not rows:
        raise ValueError(f"CSV file is empty: {CSV_PATH}")

    headers = rows[0]
    data = rows[1:]

    group_col = find_col_idx(headers, "student_0, expert_1, genAI_2")

    try:
        name_col = find_col_idx(headers, "What is your full name?")
    except ValueError:
        name_col = None

    metric_cols = {m: find_col_idx(headers, m.prefix) for m in METRICS}

    out_rows: list[dict[str, object]] = []
    raw_values_by_task_metric_group: dict[tuple[str, str, str], np.ndarray] = {}

    for m in METRICS:
        col = metric_cols[m]
        grouped_values: dict[str, list[float]] = {g: [] for g in GROUP_ORDER}

        for r in data:
            gid = r[group_col].strip() if len(r) > group_col else ""
            gname = GROUP_MAP.get(gid)

            if gname is None:
                continue

            v = to_float(r[col]) if len(r) > col else None

            if v is not None:
                grouped_values[gname].append(v)

        for g in GROUP_ORDER:
            s = summarize(grouped_values[g])
            out_rows.append({
                "task": m.task,
                "metric": m.label,
                "group": g,
                **s,
            })
            raw_values_by_task_metric_group[(m.task, m.label, g)] = np.asarray(
                grouped_values[g],
                dtype=float,
            )

    plot_mean_ci_single_task(
        out_rows,
        "Race",
        OUT_FIG_RACE,
        raw_values_by_task_metric_group,
    )

    plot_mean_ci_single_task(
        out_rows,
        "Gender",
        OUT_FIG_GENDER,
        raw_values_by_task_metric_group,
    )

    out_rows_collapsed, raw_collapsed = build_grouped_values_collapsed(
        data,
        group_col,
        metric_cols,
    )

    plot_mean_ci_single_task_collapsed(
        out_rows_collapsed,
        "Race",
        OUT_FIG_RACE_HUMAN_GENAI,
        raw_collapsed,
    )

    plot_mean_ci_single_task_collapsed(
        out_rows_collapsed,
        "Gender",
        OUT_FIG_GENDER_HUMAN_GENAI,
        raw_collapsed,
    )

    print("\nQuick view (mean [95% CI]):")

    for task in ("Race", "Gender"):
        print(f"\n{task}:")
        for metric in METRIC_ORDER:
            print(f"  {metric}")
            rows_here = [
                r for r in out_rows
                if r["task"] == task and r["metric"] == metric
            ]
            rows_here.sort(key=lambda r: GROUP_ORDER.index(str(r["group"])))

            for r in rows_here:
                print(
                    f"    {display_label(str(r['group'])):<14} n={r['n']:>2}  "
                    f"mean={r['mean']:.3f}  "
                    f"[{r['ci_low']:.3f}, {r['ci_high']:.3f}]"
                )

    print(f"\nCollapsed (Humans: {HUMAN_COMPOSITION_NOTE}) (mean [95% CI]):")

    for task in ("Race", "Gender"):
        print(f"\n{task}:")
        for metric in METRIC_ORDER:
            print(f"  {metric}")
            rows_here = [
                r for r in out_rows_collapsed
                if r["task"] == task and r["metric"] == metric
            ]
            rows_here.sort(key=lambda r: GROUP_ORDER_COLLAPSED.index(str(r["group"])))

            for r in rows_here:
                print(
                    f"    {display_label(str(r['group'])):<10} n={r['n']:>2}  "
                    f"mean={r['mean']:.3f}  "
                    f"[{r['ci_low']:.3f}, {r['ci_high']:.3f}]"
                )

    if name_col is not None:
        print("\nRows with missing Q5 numeric fields:")

        for m in METRICS:
            col = metric_cols[m]
            missing_names = []

            for r in data:
                v = to_float(r[col]) if len(r) > col else None

                if v is None:
                    missing_names.append(
                        r[name_col] if len(r) > name_col else "<unknown>"
                    )

            if missing_names:
                print(f"  {m.task} / {m.label}: {', '.join(missing_names)}")


if __name__ == "__main__":
    main()