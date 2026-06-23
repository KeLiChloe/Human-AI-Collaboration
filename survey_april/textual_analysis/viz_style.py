"""Shared matplotlib style for textual_analysis visualization scripts."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import Bbox

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from viz_config import COLOR_AGG_HUMAN, GROUP_COLORS

SIG_TEXT_COLOR = "#C62828"
FOOTNOTE_COLOR = "#444444"
BOX_EDGE_NEUTRAL = "#555555"
PHASE_HATCH_COLOR = "#9e9e9e"

GROUP_COLORS_TEXT = {
    "PhD Students": GROUP_COLORS["phd"],
    "Experts": GROUP_COLORS["expert"],
    "GenAI": GROUP_COLORS["genai"],
}
GROUP_COLORS_COLLAPSED = {
    "Human": COLOR_AGG_HUMAN,
    "GenAI": GROUP_COLORS["genai"],
}

GROUP_ORDER = ["PhD Students", "Experts", "GenAI"]
GROUP_ORDER_COLLAPSED = ["Human", "GenAI"]

DISPLAY_LABELS = {
    "PhD Students": "PhD Students",
    "Experts": "Experts",
    "GenAI": "GenAI",
    "Human": "Humans",
}
HUMAN_COMPOSITION_NOTE = "PhD Students + Experts"

SAVE_DPI = 600
SAVE_PAD_INCHES = 0.08
FONT_TITLE = 20
TITLE_PAD = 22
FONT_AXIS_LABEL = 13
FONT_TICK = 11.5
FONT_LEGEND = 13
FONT_COMPARISON = 11
FONT_FOOTNOTE = 10
SUBPLOT_LEFT = 0.10
SUBPLOT_RIGHT = 0.98
SUBPLOT_TOP = 0.87

FOOTNOTE_Y = 0.038
SIG_FOOTNOTE = (
    "Welch t-test | NS (p ≥ 0.05), * (p < 0.05), ** (p < 0.01), *** (p < 0.001)"
)
ASSESSMENT_SIG_FOOTNOTE = (
    "Between-group comparisons: Welch t-test",
    "Within-group (Pre vs Post): paired t-test",
    "NS (p ≥ 0.05), * (p < 0.05), ** (p < 0.01), *** (p < 0.001)",
)
FOOTNOTE_LINE_STEP = 0.014
PAIRED_BRACKET_LIFT = 0.035
PAIRED_BRACKET_HEIGHT = 0.045
PAIRED_BRACKET_LABEL_GAP = 0.014

COMPARISON_BOX_BOTTOM = 0.085
COMPARISON_AXIS_GAP = 0.065
COMPARISON_LINE_STEP = 0.028
PHASE_COMPARE_LINE_STEP = COMPARISON_LINE_STEP
COMPARISON_BOX_PAD = 0.010
FIGURE_COMPARE_PAD_PX = 8
PHASE_BOX_SIDE_MARGIN = 0.08
BOX_GAP = 0.08
BOX_WIDTH_ONE_LINE = 0.34
BOX_WIDTH_THREE_LINES = 0.35
BOX_WIDTH_CENTERED = 0.62
BOX_STYLE_PAD = 0.008

BAR_ALPHA = 0.95
BAR_EDGE_COLOR = "white"
BAR_EDGE_WIDTH = 0.7
ERROR_CAPSIZE = 3
ERROR_LINEWIDTH = 1.0

COMPARE_PAD_PX = 12
COMPARE_LINE_STEP_METRIC = 0.045

RC_PARAMS = {
    "figure.dpi": 180,
    "savefig.dpi": SAVE_DPI,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10.5,
    "axes.titlesize": FONT_TITLE,
    "axes.labelsize": 11,
    "axes.linewidth": 0.9,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": FONT_LEGEND,
    "legend.frameon": False,
    "grid.alpha": 0.2,
    "lines.linewidth": 1.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def apply_plot_style() -> None:
    plt.rcParams.update(RC_PARAMS)


def set_figure_title(ax, title: str, *, pad: float | None = None) -> None:
    ax.set_title(
        title,
        fontweight="bold",
        fontsize=FONT_TITLE,
        pad=TITLE_PAD if pad is None else pad,
    )


def display_label(group_key: str) -> str:
    return DISPLAY_LABELS.get(group_key, group_key)


def comparison_pair_label(left_key: str, right_key: str) -> str:
    return f"{display_label(left_key)} vs. {display_label(right_key)}"


def legend_entry(group_key: str, n: int, *, include_composition: bool = False) -> str:
    label = display_label(group_key)
    if include_composition and group_key == "Human":
        return f"{label} (n={n}, {HUMAN_COMPOSITION_NOTE})"
    return f"{label} (n={n})"


def collapsed_legend_labels(groups: list[str], ns: dict[str, int]) -> list[str]:
    return [
        legend_entry(g, ns[g], include_composition=(g == "Human"))
        for g in groups
    ]


def add_legend(ax, handles: list, labels: list[str], *, loc: str = "upper left"):
    return ax.legend(
        handles,
        labels,
        loc=loc,
        frameon=False,
        fontsize=FONT_LEGEND,
    )


def _apply_bold_xticklabels(ax) -> None:
    for label in ax.get_xticklabels():
        label.set_fontweight("bold")


def style_axes(ax) -> None:
    ax.grid(axis="y", alpha=0.2, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=FONT_TICK)
    ax.tick_params(axis="y", labelsize=FONT_TICK)
    if getattr(ax, "_viz_bold_xticks", False):
        _apply_bold_xticklabels(ax)


def set_axis_labels(
    ax,
    xlabel: str | None,
    ylabel: str,
    *,
    xlabel_pad: int = 10,
    bold_xticks: bool = False,
) -> None:
    ax._viz_bold_xticks = bold_xticks
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=FONT_AXIS_LABEL, labelpad=xlabel_pad)
    ax.set_ylabel(ylabel, fontsize=FONT_AXIS_LABEL)
    if bold_xticks:
        _apply_bold_xticklabels(ax)


def save_figure(fig, out_path: Path) -> None:
    fig.savefig(out_path, dpi=SAVE_DPI, pad_inches=SAVE_PAD_INCHES)
    plt.close(fig)


def fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "NA"
    if p < 1e-4:
        return "<1e-4"
    return f"{p:.4f}"


def significance_label(p: float) -> str:
    if not np.isfinite(p):
        return "n/a"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "NS"


def is_significant(p: float) -> bool:
    return np.isfinite(p) and p < 0.05


def format_comparison_line(label: str, p: float) -> str:
    sig = significance_label(p)
    if sig in {"NS", "n/a"}:
        return f"{label}: {sig}"
    return f"{label}: {sig} (p={fmt_p(p)})"


def format_paired_pre_post_line(p: float) -> str:
    sig = significance_label(p)
    if sig in {"NS", "n/a"}:
        return f"Pre vs Post: {sig}"
    return f"Pre vs Post: {sig} (p={fmt_p(p)})"


def comparison_box_height(n_lines: int) -> float:
    """Tight height estimate for bottom margin (matches bbox-hug phase boxes)."""
    return 0.026 + COMPARISON_LINE_STEP * max(n_lines - 1, 0) + 0.016


def comparison_box_width(n_lines: int, *, centered: bool = False) -> float:
    if centered:
        return BOX_WIDTH_CENTERED
    return BOX_WIDTH_ONE_LINE if n_lines == 1 else BOX_WIDTH_THREE_LINES


def phase_center_x(box_width: float) -> dict[str, float]:
    gap = BOX_GAP
    span = 2 * box_width + gap
    max_span = 1.0 - 2 * PHASE_BOX_SIDE_MARGIN
    if span > max_span:
        gap = max(0.06, max_span - 2 * box_width)
        span = 2 * box_width + gap
    left_edge = (1.0 - span) / 2.0
    left_edge = max(left_edge, PHASE_BOX_SIDE_MARGIN)
    if left_edge + span > 1.0 - PHASE_BOX_SIDE_MARGIN:
        left_edge = 1.0 - PHASE_BOX_SIDE_MARGIN - span
    return {
        "Pre-ML": left_edge + box_width / 2,
        "Post-ML": left_edge + box_width + gap + box_width / 2,
    }


def apply_bottom_layout(
    fig,
    n_lines: int,
    *,
    box_bottom: float = COMPARISON_BOX_BOTTOM,
    axis_gap: float = COMPARISON_AXIS_GAP,
) -> None:
    box_height = comparison_box_height(n_lines)
    comparison_top = box_bottom + box_height
    bottom_margin = min(0.90, comparison_top + axis_gap)
    fig.subplots_adjust(
        left=SUBPLOT_LEFT,
        right=SUBPLOT_RIGHT,
        top=SUBPLOT_TOP,
        bottom=bottom_margin,
    )


def draw_sig_footnote(
    fig,
    y: float = FOOTNOTE_Y,
    *,
    text: str | tuple[str, ...] | list[str] | None = None,
) -> None:
    content = SIG_FOOTNOTE if text is None else text
    lines = [content] if isinstance(content, str) else list(content)
    for i, line in enumerate(lines):
        fig.text(
            0.5,
            y - i * FOOTNOTE_LINE_STEP,
            line,
            ha="center",
            va="bottom",
            fontsize=FONT_FOOTNOTE,
            color=FOOTNOTE_COLOR,
            transform=fig.transFigure,
            clip_on=False,
            zorder=1,
        )


def draw_paired_pre_post_bracket(
    ax,
    x_pre: float,
    x_post: float,
    y_base: float,
    p: float,
) -> None:
    """Bracket over Pre/Post bars for within-group paired comparison."""
    ylo, yhi = ax.get_ylim()
    span = yhi - ylo
    y_bar = y_base + span * PAIRED_BRACKET_LIFT
    y_tip = y_bar + span * PAIRED_BRACKET_HEIGHT
    label_y = y_tip + span * PAIRED_BRACKET_LABEL_GAP
    sig = is_significant(p)
    color = SIG_TEXT_COLOR if sig else "black"
    weight = "bold" if sig else "normal"
    ax.plot(
        [x_pre, x_pre, x_post, x_post],
        [y_bar, y_tip, y_tip, y_bar],
        color=color,
        linewidth=1.1,
        clip_on=False,
        zorder=6,
    )
    ax.text(
        (x_pre + x_post) / 2,
        label_y,
        format_paired_pre_post_line(p),
        ha="center",
        va="bottom",
        fontsize=FONT_COMPARISON,
        fontweight=weight,
        color=color,
        clip_on=False,
        zorder=7,
    )


def _add_comparison_box(
    fig,
    box_left: float,
    box_bottom: float,
    box_width: float,
    box_height: float,
    has_sig: bool,
) -> None:
    fig.add_artist(
        mpatches.FancyBboxPatch(
            (box_left, box_bottom),
            box_width,
            box_height,
            boxstyle=f"round,pad={BOX_STYLE_PAD}",
            transform=fig.transFigure,
            facecolor="white",
            edgecolor=SIG_TEXT_COLOR if has_sig else BOX_EDGE_NEUTRAL,
            alpha=0.96,
            clip_on=False,
            linewidth=1.2 if has_sig else 0.9,
            zorder=2,
        )
    )


def _figure_text_width(fig, text: str, *, fontsize: float) -> float:
    """Return text width in figure coordinates."""
    renderer = fig.canvas.get_renderer()
    tmp = fig.text(0.5, 0.5, text, fontsize=fontsize, transform=fig.transFigure)
    bb = tmp.get_window_extent(renderer=renderer)
    tmp.remove()
    inv = fig.transFigure.inverted()
    x0, _ = inv.transform((bb.x0, bb.y0))
    x1, _ = inv.transform((bb.x1, bb.y0))
    return abs(x1 - x0)


def _auto_box_width(
    fig,
    lines: list[tuple[str, float]],
    *,
    min_width: float,
    max_width: float | None = None,
) -> float:
    if not lines:
        return min_width
    texts = [format_comparison_line(label, p) for label, p in lines]
    text_w = max(_figure_text_width(fig, t, fontsize=FONT_COMPARISON) for t in texts)
    pad = 2 * COMPARISON_BOX_PAD + 0.02
    cap = 1.0 - 2 * PHASE_BOX_SIDE_MARGIN if max_width is None else max_width
    return min(max(text_w + pad, min_width), cap)


def draw_centered_comparison_box(
    fig,
    comparisons: list[tuple[str, float]],
    *,
    center_x: float = 0.5,
    box_bottom: float = COMPARISON_BOX_BOTTOM,
    min_box_width: float | None = None,
    max_box_width: float | None = None,
) -> None:
    n = len(comparisons)
    min_width = (
        comparison_box_width(n, centered=True)
        if min_box_width is None
        else min_box_width
    )
    box_width = _auto_box_width(
        fig, comparisons, min_width=min_width, max_width=max_box_width
    )
    box_height = comparison_box_height(n)
    box_top = box_bottom + box_height
    box_left = center_x - box_width / 2
    has_sig = any(is_significant(p) for _, p in comparisons)

    _add_comparison_box(
        fig, box_left, box_bottom, box_width, box_height, has_sig
    )

    line_ys = [
        box_top - COMPARISON_BOX_PAD - i * COMPARISON_LINE_STEP for i in range(n)
    ]
    for (label, pval), y in zip(comparisons, line_ys):
        sig = is_significant(pval)
        fig.text(
            center_x,
            y,
            format_comparison_line(label, pval),
            transform=fig.transFigure,
            ha="center",
            va="top",
            fontsize=FONT_COMPARISON,
            fontweight="bold" if sig else "normal",
            color=SIG_TEXT_COLOR if sig else "black",
            clip_on=False,
            zorder=3,
        )


def snug_comparison_box_width(
    fig,
    lines: list[tuple[str, float]],
    *,
    max_width: float | None = None,
) -> float:
    """Text-fitted comparison box width (no default minimum)."""
    return _auto_box_width(fig, lines, min_width=0.0, max_width=max_width)


def shared_phase_box_width(
    fig,
    phase_lines: list[list[tuple[str, float]]],
    n_lines: int,
) -> float:
    fig.canvas.draw()
    min_width = comparison_box_width(n_lines)
    return max(
        _auto_box_width(fig, lines, min_width=min_width) for lines in phase_lines
    )


def _draw_snug_figure_patch(fig, text_objs: list, has_sig: bool) -> None:
    """Wrap figure-coordinate text objects in a tight rounded box (diagram style)."""
    if not text_objs:
        return
    renderer = fig.canvas.get_renderer()
    bb = Bbox.union([t.get_window_extent(renderer=renderer) for t in text_objs])
    bb = Bbox.from_extents(
        bb.x0 - FIGURE_COMPARE_PAD_PX,
        bb.y0 - FIGURE_COMPARE_PAD_PX,
        bb.x1 + FIGURE_COMPARE_PAD_PX,
        bb.y1 + FIGURE_COMPARE_PAD_PX,
    )
    inv = fig.transFigure.inverted()
    x0, y0 = inv.transform((bb.x0, bb.y0))
    x1, y1 = inv.transform((bb.x1, bb.y1))
    fig.add_artist(
        mpatches.FancyBboxPatch(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            boxstyle=f"round,pad={BOX_STYLE_PAD}",
            transform=fig.transFigure,
            facecolor="white",
            edgecolor=SIG_TEXT_COLOR if has_sig else BOX_EDGE_NEUTRAL,
            alpha=0.96,
            clip_on=False,
            linewidth=1.2 if has_sig else 0.9,
            zorder=3,
        )
    )


def draw_phase_comparison_box(
    fig,
    phase: str,
    center_x: float,
    lines: list[tuple[str, float]],
    *,
    box_width: float | None = None,
) -> None:
    del box_width  # spacing handled by caller; box size hugs text
    n = len(lines)
    has_sig = any(is_significant(p) for _, p in lines)
    n_with_header = n + 1
    y_top = COMPARISON_BOX_BOTTOM + comparison_box_height(n_with_header) - 0.006
    text_objs = []
    text_objs.append(
        fig.text(
            center_x,
            y_top,
            phase,
            transform=fig.transFigure,
            ha="center",
            va="top",
            fontsize=FONT_COMPARISON,
            fontweight="bold",
            color="black",
            clip_on=False,
            zorder=4,
        )
    )
    for i, (label, pval) in enumerate(lines):
        sig = is_significant(pval)
        text_objs.append(
            fig.text(
                center_x,
                y_top - (i + 1) * PHASE_COMPARE_LINE_STEP,
                format_comparison_line(label, pval),
                transform=fig.transFigure,
                ha="center",
                va="top",
                fontsize=FONT_COMPARISON,
                fontweight="bold" if sig else "normal",
                color=SIG_TEXT_COLOR if sig else "black",
                clip_on=False,
                zorder=4,
            )
        )
    fig.canvas.draw()
    _draw_snug_figure_patch(fig, text_objs, has_sig)


def draw_snug_footer_comparison_box(
    fig,
    lines: list[tuple[str, float]],
    *,
    center_x: float = 0.5,
) -> None:
    """Single footer box with snug bbox (diagram / assessment style, no phase header)."""
    n = len(lines)
    has_sig = any(is_significant(p) for _, p in lines)
    y_top = COMPARISON_BOX_BOTTOM + comparison_box_height(n) - 0.006
    text_objs = []
    for i, (label, pval) in enumerate(lines):
        sig = is_significant(pval)
        text_objs.append(
            fig.text(
                center_x,
                y_top - i * PHASE_COMPARE_LINE_STEP,
                format_comparison_line(label, pval),
                transform=fig.transFigure,
                ha="center",
                va="top",
                fontsize=FONT_COMPARISON,
                fontweight="bold" if sig else "normal",
                color=SIG_TEXT_COLOR if sig else "black",
                clip_on=False,
                zorder=4,
            )
        )
    fig.canvas.draw()
    _draw_snug_figure_patch(fig, text_objs, has_sig)


def _place_metric_comparison_texts(
    ax,
    x_idx: int,
    comparison_lines: list[str],
    p_values: list[float],
    y: float,
    line_step: float,
) -> tuple[list, bool]:
    trans = ax.get_xaxis_transform()
    has_sig = any(is_significant(p) for p in p_values)
    text_objs = []

    for i, (line, p) in enumerate(zip(comparison_lines, p_values)):
        sig = is_significant(p)
        text_objs.append(
            ax.text(
                x_idx,
                y - i * line_step,
                line,
                transform=trans,
                ha="center",
                va="top",
                fontsize=FONT_COMPARISON,
                fontweight="bold" if sig else "normal",
                color=SIG_TEXT_COLOR if sig else "black",
                clip_on=False,
                zorder=4,
            )
        )

    return text_objs, has_sig


def _draw_metric_patch(ax, text_objs: list, has_sig: bool) -> None:
    if not text_objs:
        return

    trans = ax.get_xaxis_transform()
    renderer = ax.figure.canvas.get_renderer()
    bb = Bbox.union([t.get_window_extent(renderer=renderer) for t in text_objs])
    bb = Bbox.from_extents(
        bb.x0 - COMPARE_PAD_PX,
        bb.y0 - COMPARE_PAD_PX,
        bb.x1 + COMPARE_PAD_PX,
        bb.y1 + COMPARE_PAD_PX,
    )

    inv = trans.inverted()
    x0, y0 = inv.transform((bb.x0, bb.y0))
    x1, y1 = inv.transform((bb.x1, bb.y1))

    ax.add_patch(
        mpatches.FancyBboxPatch(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            boxstyle=f"round,pad={BOX_STYLE_PAD}",
            transform=trans,
            facecolor="white",
            edgecolor=SIG_TEXT_COLOR if has_sig else BOX_EDGE_NEUTRAL,
            alpha=0.96,
            linewidth=1.2 if has_sig else 0.9,
            clip_on=False,
            zorder=3,
        )
    )


def draw_metric_comparison_boxes(
    ax,
    comparisons: list[tuple[int, list[str], list[float], float, float]],
) -> None:
    placed: list[tuple[list, bool]] = []
    for x_idx, lines, p_values, y, line_step in comparisons:
        placed.append(
            _place_metric_comparison_texts(ax, x_idx, lines, p_values, y, line_step)
        )

    ax.figure.canvas.draw()

    for text_objs, has_sig in placed:
        _draw_metric_patch(ax, text_objs, has_sig)


def finalize_metric_figure(
    fig,
    ax,
    out_path: Path,
    *,
    layout: dict,
    comparisons: list[tuple[int, list[str], list[float], float, float]],
    footnote_y: float = FOOTNOTE_Y,
) -> None:
    fig.subplots_adjust(**layout)
    draw_metric_comparison_boxes(ax, comparisons)
    draw_sig_footnote(fig, y=footnote_y)
    save_figure(fig, out_path)
    print(f"Saved figure: {out_path}")
