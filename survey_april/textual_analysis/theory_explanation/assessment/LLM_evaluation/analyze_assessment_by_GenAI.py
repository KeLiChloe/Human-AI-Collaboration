"""
Assess theory quality scores by group (Senior Scientists, PhD Students, GenAI).

Reads overall quality scores (mean of 5 dimensions) and plots Pre/Post bars:

1. 2×2 by task × effect (Race/Gender × Main/Interactions)
   - PhD / Senior / Topic Experts / GenAI
   - Human / Topic Experts / GenAI
2. Same group splits with all four panels pooled
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

ASSESSMENT_DIR = Path(__file__).resolve().parent
TEXTUAL_DIR = ASSESSMENT_DIR.parent.parent.parent  # textual_analysis/
ROOT = TEXTUAL_DIR.parent  # survey_april/
for p in (ASSESSMENT_DIR, TEXTUAL_DIR, ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from stats_utils import (  # noqa: E402
    bootstrap_mean_ci,
    p_value_paired_ttest_pairs,
    p_value_welch_ttest,
)
from viz_style import (  # noqa: E402
    BAR_ALPHA,
    BAR_EDGE_COLOR,
    BAR_EDGE_WIDTH,
    ERROR_CAPSIZE,
    ERROR_LINEWIDTH,
    GROUP_COLORS_COLLAPSED,
    GROUP_COLORS_TEXT,
    HUMAN_COMPOSITION_NOTE,
    PHASE_HATCH_COLOR,
    add_legend,
    apply_plot_style,
    comparison_pair_label,
    display_label,
    draw_pre_post_bracket,
    draw_pairwise_group_brackets,
    draw_pairwise_sig_legend,
    draw_pairwise_sig_color_legend,
    draw_pre_post_sig_columns,
    format_comparison_line,
    format_p_value_label,
    is_significant,
    save_figure,
    set_axis_labels,
    significance_label,
    style_axes,
    SAVE_DPI,
    SAVE_PAD_INCHES,
)

ASSESSMENT_SCORE_YMAX = 10
ASSESSMENT_PLOT_YMAX = 12.2

CSV_PATH = ROOT / "All_Participants_All_Questions.csv"
OUT_DIR = ASSESSMENT_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Four panels: task × effect (main / interactions=SOI).
PANEL_SPECS: tuple[tuple[str, dict[str, str]], ...] = (
    (
        "Racial inequality — Main effects",
        {
            "Pre-ML": "Q Race.4 Overall Quality Score",
            "Post-ML": "Q Race.12 Overall Quality Score",
        },
    ),
    (
        "Racial inequality — Interactions",
        {
            "Pre-ML": "Q Race.10 Overall Quality Score",
            "Post-ML": "Q Race.15 Overall Quality Score",
        },
    ),
    (
        "Gender inequality — Main effects",
        {
            "Pre-ML": "Q Gender.4 Overall Quality Score",
            "Post-ML": "Q Gender.12 Overall Quality Score",
        },
    ),
    (
        "Gender inequality — Interactions",
        {
            "Pre-ML": "Q Gender.10 Overall Quality Score",
            "Post-ML": "Q Gender.15 Overall Quality Score",
        },
    ),
)

POOLED_TASK_LABEL = "Race & Gender × Main & Interactions (pooled)"

GROUP_ORDER = ["PhD Students", "Senior Scientists", "Topic Experts", "GenAI"]
GROUP_ORDER_COLLAPSED = ["Human", "Topic Experts", "GenAI"]
NON_TOPIC_GROUP = "Non-Topic Experts"
VALUE_GROUPS = (*GROUP_ORDER, NON_TOPIC_GROUP)

GROUP_MAP = {
    "0": "PhD Students",
    "1": "Senior Scientists",
    "2": "GenAI",
}

PAIRWISE_THREE = (
    ("PhD Students", "GenAI"),
    ("Senior Scientists", "GenAI"),
    ("Senior Scientists", "PhD Students"),
    ("Topic Experts", NON_TOPIC_GROUP),
)
PAIRWISE_COLLAPSED = (
    ("Human", "GenAI"),
    ("Topic Experts", NON_TOPIC_GROUP),
)

PHASE_ORDER = ("Pre-ML", "Post-ML")

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
        "Human": grouped_values["PhD Students"] + grouped_values["Senior Scientists"],
        "Topic Experts": list(grouped_values["Topic Experts"]),
        NON_TOPIC_GROUP: list(grouped_values[NON_TOPIC_GROUP]),
        "GenAI": grouped_values["GenAI"],
    }


def collapse_paired_by_group(
    paired_by_group: dict[str, list[tuple[float, float]]],
) -> dict[str, list[tuple[float, float]]]:
    return {
        "Human": paired_by_group["PhD Students"] + paired_by_group["Senior Scientists"],
        "Topic Experts": list(paired_by_group["Topic Experts"]),
        NON_TOPIC_GROUP: list(paired_by_group[NON_TOPIC_GROUP]),
        "GenAI": paired_by_group["GenAI"],
    }


def merge_phase_values(
    parts: list[dict[str, dict[str, list[float]]]],
) -> dict[str, dict[str, list[float]]]:
    merged = {phase: {g: [] for g in VALUE_GROUPS} for phase in PHASE_ORDER}
    for part in parts:
        for phase, by_group in part.items():
            for group, vals in by_group.items():
                merged[phase][group].extend(vals)
    return merged


def merge_paired_by_group(
    parts: list[dict[str, list[tuple[float, float]]]],
) -> dict[str, list[tuple[float, float]]]:
    merged = {g: [] for g in VALUE_GROUPS}
    for part in parts:
        for group, pairs in part.items():
            merged[group].extend(pairs)
    return merged


def load_phase_and_paired_values(
    headers: list[str],
    data: list[list[str]],
    group_col: int,
    phase_map: dict[str, str],
    topic_expert_col: int,
) -> tuple[dict[str, dict[str, list[float]]], dict[str, list[tuple[float, float]]]]:
    pre_col = find_col_idx(headers, phase_map["Pre-ML"])
    post_col = find_col_idx(headers, phase_map["Post-ML"])
    phase_to_grouped_values = {
        "Pre-ML": {g: [] for g in VALUE_GROUPS},
        "Post-ML": {g: [] for g in VALUE_GROUPS},
    }
    paired_by_group: dict[str, list[tuple[float, float]]] = {
        g: [] for g in VALUE_GROUPS
    }

    for row in data:
        gid = row[group_col].strip() if len(row) > group_col else ""
        gname = GROUP_MAP.get(gid)
        if gname is None:
            continue
        pre = to_float(row[pre_col]) if len(row) > pre_col else None
        post = to_float(row[post_col]) if len(row) > post_col else None

        targets = [gname]
        if gname in ("PhD Students", "Senior Scientists") and len(row) > topic_expert_col:
            flag = row[topic_expert_col].strip()
            if flag == "1":
                targets.append("Topic Experts")
            elif flag == "0":
                targets.append(NON_TOPIC_GROUP)

        for target in targets:
            if pre is not None:
                phase_to_grouped_values["Pre-ML"][target].append(pre)
            if post is not None:
                phase_to_grouped_values["Post-ML"][target].append(post)
            if pre is not None and post is not None:
                paired_by_group[target].append((pre, post))

    return phase_to_grouped_values, paired_by_group


def _pairwise_for_groups(
    group_order: list[str] | tuple[str, ...],
) -> tuple[tuple[str, str], ...]:
    if list(group_order) == list(GROUP_ORDER):
        return PAIRWISE_THREE
    if list(group_order) == list(GROUP_ORDER_COLLAPSED):
        return PAIRWISE_COLLAPSED
    raise ValueError(f"Unknown group_order: {group_order}")


def _draw_panel(
    ax,
    values: dict[str, dict[str, list[float]]],
    title: str,
    *,
    group_order: list[str] | tuple[str, ...],
    group_colors: dict[str, str],
    pairwise: tuple[tuple[str, str], ...],
    note_fontsize: float,
    notes_layout: str = "stacked",
) -> None:
    x = np.arange(len(group_order))
    width = 0.34
    offsets = {"Pre-ML": -width / 2, "Post-ML": width / 2}
    bar_tops: dict[str, float] = {}
    phase_bar_tops: dict[str, float] = {phase: 0.0 for phase in PHASE_ORDER}

    for gi, group in enumerate(group_order):
        means, yerr_lo, yerr_hi = [], [], []
        for phase in PHASE_ORDER:
            stats = summarize(values[phase][group])
            mean = float(stats["mean"])
            lo = float(stats["ci_low"])
            hi = float(stats["ci_high"])
            means.append(mean)
            yerr_lo.append(
                max(0.0, mean - lo) if np.isfinite(mean) and np.isfinite(lo) else 0.0
            )
            yerr_hi.append(
                max(0.0, hi - mean) if np.isfinite(mean) and np.isfinite(hi) else 0.0
            )
        finite = [
            means[i] + yerr_hi[i]
            for i in range(len(PHASE_ORDER))
            if np.isfinite(means[i])
        ]
        bar_tops[group] = max(finite) if finite else 0.0

        for i, phase in enumerate(PHASE_ORDER):
            xpos = float(x[gi] + offsets[phase])
            bar = ax.bar(
                [xpos],
                [means[i]],
                width=width,
                color=group_colors[group],
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
                yerr=[[yerr_lo[i]], [yerr_hi[i]]],
                fmt="none",
                ecolor="black",
                elinewidth=ERROR_LINEWIDTH,
                capsize=ERROR_CAPSIZE,
                zorder=3,
            )
            phase_bar_tops[phase] = max(
                phase_bar_tops[phase],
                means[i] + yerr_hi[i] if np.isfinite(means[i]) else 0.0,
            )

        p_pre_post = p_value_welch_ttest(
            np.asarray(values["Pre-ML"][group], dtype=float),
            np.asarray(values["Post-ML"][group], dtype=float),
        )
        if notes_layout == "sig_color_pvals":
            draw_pre_post_bracket(
                ax,
                float(x[gi] + offsets["Pre-ML"]),
                float(x[gi] + offsets["Post-ML"]),
                bar_tops[group],
                p_pre_post,
                label=format_p_value_label(p_pre_post)
                if is_significant(p_pre_post)
                else "NS",
                fontsize=note_fontsize,
            )
        else:
            draw_pre_post_bracket(
                ax,
                float(x[gi] + offsets["Pre-ML"]),
                float(x[gi] + offsets["Post-ML"]),
                bar_tops[group],
                p_pre_post,
            )

    phase_comp_sigs: dict[str, list[str]] = {phase: [] for phase in PHASE_ORDER}
    phase_comp_pvals: dict[str, list[float]] = {phase: [] for phase in PHASE_ORDER}
    pair_labels: list[str] = []
    for left, right in pairwise:
        pair_labels.append(comparison_pair_label(left, right))
        for phase in PHASE_ORDER:
            p_val = p_value_welch_ttest(
                np.asarray(values[phase][left], dtype=float),
                np.asarray(values[phase][right], dtype=float),
            )
            phase_comp_sigs[phase].append(significance_label(p_val))
            phase_comp_pvals[phase].append(p_val)

    ax.set_xticks(x)
    ax.set_xticklabels([display_label(g) for g in group_order], fontsize=10)
    if title:
        ax.set_title(title, fontsize=13, pad=8)
    else:
        ax.set_title("")
    set_axis_labels(ax, None, None, bold_xticks=True)
    style_axes(ax)
    ax.tick_params(axis="y", labelsize=14.5)
    ax.tick_params(axis="x", labelsize=10)
    for label in ax.get_xticklabels():
        label.set_fontsize(10)
        label.set_fontweight("bold")
    ax.set_ylim(0, ASSESSMENT_PLOT_YMAX)
    ax.set_yticks(list(range(0, ASSESSMENT_SCORE_YMAX + 1, 2)))

    if notes_layout == "pairwise_brackets":
        draw_pairwise_group_brackets(
            ax,
            x,
            group_order,
            pairwise,
            PHASE_ORDER,
            offsets,
            phase_comp_sigs,
            phase_bar_tops,
            fontsize=note_fontsize,
        )
    elif notes_layout == "sig_legend":
        draw_pairwise_sig_legend(
            ax,
            [
                ("Pre-ML", list(zip(phase_comp_sigs["Pre-ML"], pair_labels))),
                ("Post-ML", list(zip(phase_comp_sigs["Post-ML"], pair_labels))),
            ],
            loc="upper right",
            fontsize=note_fontsize,
        )
    elif notes_layout in ("sig_color_legend", "sig_color_pvals"):
        use_p = notes_layout == "sig_color_pvals"
        draw_pairwise_sig_color_legend(
            ax,
            [
                (
                    "Pre-ML",
                    list(
                        zip(
                            phase_comp_pvals["Pre-ML"]
                            if use_p
                            else phase_comp_sigs["Pre-ML"],
                            pairwise,
                        )
                    ),
                ),
                (
                    "Post-ML",
                    list(
                        zip(
                            phase_comp_pvals["Post-ML"]
                            if use_p
                            else phase_comp_sigs["Post-ML"],
                            pairwise,
                        )
                    ),
                ),
            ],
            group_colors=group_colors,
            loc="upper right",
            fontsize=note_fontsize,
            label_pvalues=use_p,
        )
    elif notes_layout == "pre_post_columns":
        # Bold Pre/Post headers; only significant comparison rows in red.
        draw_pre_post_sig_columns(
            ax,
            [
                ("Pre-ML", list(zip(phase_comp_sigs["Pre-ML"], pair_labels))),
                ("Post-ML", list(zip(phase_comp_sigs["Post-ML"], pair_labels))),
            ],
            y0=-0.14,
            fontsize=note_fontsize,
        )
    else:
        phase_notes = []
        for phase in PHASE_ORDER:
            comps = [
                f"{lab}: {sig}"
                for lab, sig in zip(pair_labels, phase_comp_sigs[phase])
            ]
            phase_notes.append(f"{phase}: " + "; ".join(comps))
        ax.text(
            0.5,
            -0.22,
            "\n".join(phase_notes),
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=note_fontsize,
            color="#333333",
            clip_on=False,
        )


def _pooled_legend_label(group_key: str) -> str:
    label = display_label(group_key)
    if group_key == "Human":
        return f"{label} ({HUMAN_COMPOSITION_NOTE})"
    return label


def _style_legend_frame(legend) -> None:
    """Match phase/hatch gray used elsewhere (#9e9e9e), not near-black."""
    frame = legend.get_frame()
    frame.set_visible(True)
    frame.set_linewidth(0.8)
    frame.set_edgecolor(PHASE_HATCH_COLOR)
    frame.set_facecolor("white")
    frame.set_alpha(1.0)


def plot_by_task_effect_2x2(
    panel_data: list[
        tuple[
            str,
            dict[str, dict[str, list[float]]],
            dict[str, list[tuple[float, float]]],
        ]
    ],
    out_path: Path,
    *,
    group_order: list[str] | tuple[str, ...],
    group_colors: dict[str, str],
    figsize: tuple[float, float],
    title: str,
    note_fontsize: float,
    layout_bottom: float,
) -> None:
    pairwise = _pairwise_for_groups(group_order)
    fig, axes = plt.subplots(2, 2, figsize=figsize, sharey=True)
    axes_flat = axes.ravel()

    for ax, (panel_title, values, _paired) in zip(axes_flat, panel_data):
        _draw_panel(
            ax,
            values,
            panel_title,
            group_order=group_order,
            group_colors=group_colors,
            pairwise=pairwise,
            note_fontsize=note_fontsize,
        )

    # Shared legend labels (no sample sizes).
    group_legend_labels = [_pooled_legend_label(g) for g in group_order]

    axes[0, 0].set_ylabel("Overall Quality Score (Mean ± 95% CI)", fontsize=15)
    axes[1, 0].set_ylabel("Overall Quality Score (Mean ± 95% CI)", fontsize=15)

    group_handles = [
        plt.Rectangle((0, 0), 1, 1, color=group_colors[g], alpha=BAR_ALPHA)
        for g in group_order
    ]
    phase_handles = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor=PHASE_HATCH_COLOR,
            alpha=BAR_ALPHA,
            edgecolor=BAR_EDGE_COLOR,
        ),
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor=PHASE_HATCH_COLOR,
            alpha=BAR_ALPHA,
            edgecolor=BAR_EDGE_COLOR,
            hatch="///",
        ),
    ]
    legend1 = fig.legend(
        group_handles,
        group_legend_labels,
        loc="upper left",
        bbox_to_anchor=(0.06, 0.97),
        frameon=False,
        fontsize=11,
    )
    fig.add_artist(legend1)
    fig.legend(
        phase_handles,
        ["Pre-ML", "Post-ML"],
        loc="upper right",
        bbox_to_anchor=(0.98, 0.97),
        frameon=False,
        fontsize=11,
    )

    fig.suptitle(title, fontsize=16, y=0.995)
    fig.tight_layout(rect=(0.02, layout_bottom, 1.0, 0.90))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path)
    print(f"Saved figure: {out_path}")


def plot_pooled(
    values: dict[str, dict[str, list[float]]],
    out_path: Path,
    *,
    group_order: list[str] | tuple[str, ...],
    group_colors: dict[str, str],
    figsize: tuple[float, float],
    title: str,
    note_fontsize: float,
) -> None:
    pairwise = _pairwise_for_groups(group_order)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    _draw_panel(
        ax,
        values,
        "",
        group_order=group_order,
        group_colors=group_colors,
        pairwise=pairwise,
        note_fontsize=note_fontsize,
        notes_layout="pre_post_columns",
    )

    group_legend_labels = [_pooled_legend_label(g) for g in group_order]

    ax.set_ylabel("Overall Quality Score (Mean ± 95% CI)", fontsize=15)
    group_handles = [
        plt.Rectangle((0, 0), 1, 1, color=group_colors[g], alpha=BAR_ALPHA)
        for g in group_order
    ]
    phase_handles = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor=PHASE_HATCH_COLOR,
            alpha=BAR_ALPHA,
            edgecolor=BAR_EDGE_COLOR,
        ),
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor=PHASE_HATCH_COLOR,
            alpha=BAR_ALPHA,
            edgecolor=BAR_EDGE_COLOR,
            hatch="///",
        ),
    ]
    # One framed legend: contributor groups (row 1) + Pre/Post (row 2).
    all_handles = group_handles + phase_handles
    all_labels = group_legend_labels + ["Pre-ML", "Post-ML"]
    with plt.rc_context({"legend.frameon": True}):
        legend = fig.legend(
            all_handles,
            all_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.90),
            frameon=True,
            fancybox=False,
            edgecolor=PHASE_HATCH_COLOR,
            facecolor="white",
            framealpha=1.0,
            fontsize=10.5,
            ncol=len(group_order),
            handletextpad=0.35,
            columnspacing=1.0,
            borderpad=0.5,
            labelspacing=0.4,
        )
    _style_legend_frame(legend)

    fig.suptitle(title, fontsize=19, fontweight="bold", y=0.97)
    # Inset the bar panel so it does not dominate the canvas.
    # Do not use bbox_inches='tight' here — that would crop the margin padding.
    fig.subplots_adjust(left=0.14, right=0.90, top=0.76, bottom=0.28)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=SAVE_DPI, pad_inches=SAVE_PAD_INCHES)
    plt.close(fig)
    print(f"Saved figure: {out_path}")


def discover_overall_models(headers: list[str]) -> list[str]:
    """Models for which all panel Pre/Post overall columns exist."""
    required_stems = [
        stem for _, phase_map in PANEL_SPECS for stem in phase_map.values()
    ]
    candidates: set[str] | None = None
    for stem in required_stems:
        models_for_stem: set[str] = set()
        prefix = f"{stem} ("
        for h in headers:
            if h.startswith(prefix) and h.endswith(")"):
                models_for_stem.add(h[len(prefix) : -1])
        if candidates is None:
            candidates = models_for_stem
        else:
            candidates &= models_for_stem
    return sorted(candidates or [])


def model_out_dir(model: str) -> Path:
    d = OUT_DIR / model.replace("/", "_")
    d.mkdir(parents=True, exist_ok=True)
    return d


def model_display_name(model: str) -> str:
    """Pretty label for figure titles, e.g. gpt-5.5 → GPT-5.5."""
    special = {
        "gpt-5.5": "GPT-5.5",
        "gpt-5.6-sol": "GPT-5.6 Sol",
        "gpt-5.6-terra": "GPT-5.6 Terra",
        "gpt-5.6-luna": "GPT-5.6 Luna",
        "gpt-5.6": "GPT-5.6",
    }
    if model in special:
        return special[model]
    return model


def figure_title(model: str, *, humans_note: bool = False) -> str:
    base = f"Theory quality evaluated by LLM ({model_display_name(model)})"
    if humans_note:
        return f"{base}; Humans: {HUMAN_COMPOSITION_NOTE}"
    return base


def _print_panel(
    title: str,
    values: dict[str, dict[str, list[float]]],
    paired: dict[str, list[tuple[float, float]]],
    group_order: list[str] | tuple[str, ...],
) -> None:
    print(f"\n{title}")
    for phase in PHASE_ORDER:
        print(f"  {phase}")
        for g in group_order:
            s = summarize(values[phase][g])
            print(
                f"    {display_label(g):<18} n={int(s['n']):>2}  mean={s['mean']:.3f}  "
                f"[{s['ci_low']:.3f}, {s['ci_high']:.3f}]"
            )
    for g in group_order:
        pairs = paired.get(g, [])
        if pairs:
            print(
                f"  Pre vs Post ({display_label(g)}): "
                f"p = {p_value_paired_ttest_pairs(pairs):.4f}"
            )


def main() -> None:
    with CSV_PATH.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    headers = rows[0]
    data = rows[1:]

    group_col = find_col_idx(headers, "student_0, senior_1, genAI_2")
    topic_expert_col = find_col_idx(headers, "topic_expert")
    models = discover_overall_models(headers)
    if not models:
        raise SystemExit(
            "No model with complete task×effect Pre+Post Overall Quality columns. "
            "Run compute_assessment_overall_quality.py first."
        )

    print(f"Models with overall scores: {', '.join(models)}")

    for model in models:
        print(f"\n######## model = {model} ########")
        model_dir = model_out_dir(model)

        panel_data_three: list[
            tuple[
                str,
                dict[str, dict[str, list[float]]],
                dict[str, list[tuple[float, float]]],
            ]
        ] = []
        panel_data_collapsed: list[
            tuple[
                str,
                dict[str, dict[str, list[float]]],
                dict[str, list[tuple[float, float]]],
            ]
        ] = []
        merge_parts: list[dict[str, dict[str, list[float]]]] = []
        merge_paired_parts: list[dict[str, list[tuple[float, float]]]] = []

        for panel_title, stems in PANEL_SPECS:
            phase_map = {phase: f"{stem} ({model})" for phase, stem in stems.items()}
            values, paired = load_phase_and_paired_values(
                headers, data, group_col, phase_map, topic_expert_col
            )
            collapsed_values = {
                phase: collapse_grouped_values(by_g) for phase, by_g in values.items()
            }
            collapsed_paired = collapse_paired_by_group(paired)

            _print_panel(f"{panel_title} [{model}]", values, paired, GROUP_ORDER)
            panel_data_three.append((panel_title, values, paired))
            panel_data_collapsed.append(
                (panel_title, collapsed_values, collapsed_paired)
            )
            merge_parts.append(values)
            merge_paired_parts.append(paired)

        plot_by_task_effect_2x2(
            panel_data_three,
            model_dir / "by_task.svg",
            group_order=GROUP_ORDER,
            group_colors=GROUP_COLORS_TEXT,
            figsize=(14.5, 11.5),
            title=figure_title(model),
            note_fontsize=7.2,
            layout_bottom=0.08,
        )
        plot_by_task_effect_2x2(
            panel_data_collapsed,
            model_dir / "by_task_human_genai.svg",
            group_order=GROUP_ORDER_COLLAPSED,
            group_colors=GROUP_COLORS_COLLAPSED,
            figsize=(13.0, 10.8),
            title=figure_title(model, humans_note=True),
            note_fontsize=8.5,
            layout_bottom=0.07,
        )

        pooled = merge_phase_values(merge_parts)
        pooled_paired = merge_paired_by_group(merge_paired_parts)
        pooled_collapsed = {
            phase: collapse_grouped_values(by_g) for phase, by_g in pooled.items()
        }

        _print_panel(
            f"{POOLED_TASK_LABEL} [{model}]", pooled, pooled_paired, GROUP_ORDER
        )
        plot_pooled(
            pooled,
            model_dir / "pooled.svg",
            group_order=GROUP_ORDER,
            group_colors=GROUP_COLORS_TEXT,
            figsize=(10.0, 9.5),
            title=figure_title(model),
            note_fontsize=8.5,
        )
        plot_pooled(
            pooled_collapsed,
            model_dir / "pooled_human_genai.svg",
            group_order=GROUP_ORDER_COLLAPSED,
            group_colors=GROUP_COLORS_COLLAPSED,
            figsize=(9.0, 8.8),
            title=figure_title(model, humans_note=True),
            note_fontsize=9,
        )


if __name__ == "__main__":
    main()
