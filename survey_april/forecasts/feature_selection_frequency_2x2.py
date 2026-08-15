"""
2×2 composite of feature / SOI selection-frequency figures.

Layout:
  (a) Main Effects - Race       (b) Main Effects - Gender
  (c) Interactions - Race       (d) Interactions - Gender

Bar labels: n selected (sign-alignment % vs LR among those selectors).
Sign % is only defined for ML top features (have an LR ground-truth sign).

Output: forecasts/outputs/feature_selection_frequency_2x2.{pdf,svg}
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

FORECASTS = Path(__file__).resolve().parent
ROOT = FORECASTS.parent
TEXTUAL_DIR = ROOT / "textual_analysis"
for p in (ROOT, TEXTUAL_DIR, FORECASTS, FORECASTS / "main_effects", FORECASTS / "second_order_interactions"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from viz_style import apply_plot_style  # noqa: E402

import main_effects_importance as me  # noqa: E402
import soi_importance as soi  # noqa: E402

OUT_DIR = FORECASTS / "outputs"

PANEL_SPECS = [
    ("a", "Main Effects - Race", "me", "race"),
    ("b", "Main Effects - Gender", "me", "gender"),
    ("c", "Interactions - Race", "soi", "race"),
    ("d", "Interactions - Gender", "soi", "gender"),
]

# Tilt y-tick labels to shrink left margin.
Y_LABEL_ROTATION = 30


def _set_rotated_yticklabels(ax, labels: list[str], *, fontsize: float) -> None:
    ax.set_yticklabels(
        labels,
        fontsize=fontsize,
        rotation=Y_LABEL_ROTATION,
        ha="right",
        va="center",
        rotation_mode="anchor",
    )


def _me_ml_signs(task: str) -> dict[str, str]:
    with open(me.ML_PATH, encoding="utf-8") as f:
        raw = json.load(f)
    return {e["feature"]: e["sign"] for e in raw[task]}


def _me_q3_cols(task: str) -> dict[str, int]:
    prefix = "Q Race.3" if task == "race" else "Q Gender.3"
    return {
        re.sub(rf"^{re.escape(prefix)} \(sign\) - ", "", h): i
        for i, h in enumerate(me.headers)
        if re.match(rf"^{re.escape(prefix)} \(sign\) - ", h)
    }


def _me_sign_align(task: str) -> dict[str, dict[str, int]]:
    """Among selectors of each ML feature, count correct LR direction."""
    ml_signs = _me_ml_signs(task)
    q1 = me.r1_col if task == "race" else me.g1_col
    q3 = _me_q3_cols(task)
    out = {f: {"n_selected": 0, "n_aligned": 0} for f in ml_signs}
    for row in me.data:
        cell = row[q1].strip()
        if not cell:
            continue
        selected = {x.strip() for x in cell.split(",") if x.strip()}
        for feat, ml_sign in ml_signs.items():
            if feat not in selected:
                continue
            out[feat]["n_selected"] += 1
            human = row[q3[feat]].strip() if feat in q3 else ""
            if human == ml_sign:
                out[feat]["n_aligned"] += 1
    return out


def _soi_ml_signs(task: str) -> dict[tuple[str, str], str]:
    with open(soi.ML_PATH, encoding="utf-8") as f:
        raw = json.load(f)
    return {
        soi.canon_pair(e["feature_1"], e["feature_2"]): e["sign"]
        for e in raw[task]
    }


def _soi_sign_align(task: str) -> dict[tuple[str, str], dict[str, int]]:
    """Among selectors of each ML interaction, count correct LR direction."""
    ml_signs = _soi_ml_signs(task)
    pair_cols = soi.r_cols if task == "race" else soi.g_cols
    sign_prefix = "Q Race.9" if task == "race" else "Q Gender.9"
    sign_cols = [
        next(i for i, h in enumerate(soi.headers) if h.strip() == f"{sign_prefix} (SOI, sign, 1st)"),
        next(i for i, h in enumerate(soi.headers) if h.strip() == f"{sign_prefix} (SOI, sign, 2nd)"),
        next(i for i, h in enumerate(soi.headers) if h.strip() == f"{sign_prefix} (SOI, sign, 3rd)"),
    ]
    out = {p: {"n_selected": 0, "n_aligned": 0} for p in ml_signs}
    for row in soi.data:
        chosen: dict[tuple[str, str], str] = {}
        for pc, sc in zip(pair_cols, sign_cols):
            p = soi.parse_pair(row[pc], soi.feature_set)
            if p is None:
                continue
            chosen[p] = row[sc].strip()
        for p, ml_sign in ml_signs.items():
            if p not in chosen:
                continue
            out[p]["n_selected"] += 1
            if chosen[p] == ml_sign:
                out[p]["n_aligned"] += 1
    return out


def _format_bar_label(n_sel: int, align: dict[str, int] | None) -> str:
    if align is None or align["n_selected"] <= 0:
        return f"{n_sel}"
    pct = align["n_aligned"] / align["n_selected"] * 100
    # Explicit tag so the trailing % is not read as a selection rate.
    return f"{n_sel}  ({pct:.0f}% sign-aligned)"


def _draw_me_panel(ax, *, task: str, title: str, letter: str) -> None:
    counts = me.race_counts if task == "race" else me.gender_counts
    n = me.n_race if task == "race" else me.n_gender
    ml_set = me.ml_top5[task]
    align = _me_sign_align(task)
    ranked = sorted(me.FEATURES, key=lambda f: counts[f], reverse=True)
    labels = list(ranked)  # raw feature keys, e.g. social_science
    vals = [counts[f] for f in ranked]
    colors = [me.COLOR_ML if f in ml_set else me.COLOR_DEFAULT for f in ranked]

    y = np.arange(len(ranked))
    bars = ax.barh(y, vals, color=colors, height=0.65, edgecolor="white", linewidth=0.6)
    for b, v, feat in zip(bars, vals, ranked):
        ax.text(
            b.get_width() + 0.5,
            b.get_y() + b.get_height() / 2,
            _format_bar_label(v, align.get(feat)),
            va="center",
            ha="left",
            fontsize=8.5,
        )

    ax.set_yticks(y)
    _set_rotated_yticklabels(ax, labels, fontsize=11.5)
    ax.invert_yaxis()
    ax.set_xlabel(
        "Number of contributors selecting feature\n"
        "(% = percent of selectors sign-aligned with LR)",
        fontsize=11.0,
    )
    ax.set_xlim(0, n + n * 0.38)
    ax.axvline(n, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=10.5)

    ax.legend(
        handles=[
            mpatches.Patch(color=me.COLOR_ML, label="In ML top-5"),
            mpatches.Patch(color=me.COLOR_DEFAULT, label="Not in ML top-5"),
        ],
        loc="lower right",
        fontsize=10,
        frameon=False,
    )
    ax.set_title(title, fontsize=13, fontweight="bold", pad=8)
    ax.text(
        -0.02, 1.06, letter, transform=ax.transAxes,
        fontsize=14, fontweight="bold", va="bottom", ha="right", clip_on=False,
    )


def _draw_soi_panel(ax, *, task: str, title: str, letter: str, top_n: int = 8) -> None:
    counts = soi.race_counts if task == "race" else soi.gender_counts
    n = soi.n_race if task == "race" else soi.n_gender
    ml_set = soi.ml_pairs[task]
    align = _soi_sign_align(task)
    ranked = sorted(soi.pairs, key=lambda p: counts[p], reverse=True)[:top_n]
    labels = [f"{a} * {b}" for a, b in ranked]  # raw feature keys
    vals = [counts[p] for p in ranked]
    colors = [soi.COLOR_ML if p in ml_set else soi.COLOR_DEFAULT for p in ranked]

    y = np.arange(len(ranked))
    bars = ax.barh(y, vals, color=colors, height=0.65, edgecolor="white", linewidth=0.6)
    for b, v, pair in zip(bars, vals, ranked):
        ax.text(
            b.get_width() + 0.5,
            b.get_y() + b.get_height() / 2,
            _format_bar_label(v, align.get(pair)),
            va="center",
            ha="left",
            fontsize=8.5,
        )

    ax.set_yticks(y)
    _set_rotated_yticklabels(ax, labels, fontsize=11.0)
    ax.invert_yaxis()
    ax.set_xlabel(
        "Number of contributors selecting interaction\n"
        "(% = percent of selectors sign-aligned with LR)",
        fontsize=11.0,
    )
    ax.set_xlim(0, n + n * 0.38)
    ax.axvline(n, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=10.5)

    ax.legend(
        handles=[
            mpatches.Patch(color=soi.COLOR_ML, label="In ML top-3"),
            mpatches.Patch(color=soi.COLOR_DEFAULT, label="Not in ML top-3"),
        ],
        loc="lower right",
        fontsize=10,
        frameon=False,
    )
    ax.set_title(title, fontsize=13, fontweight="bold", pad=8)
    ax.text(
        -0.02, 1.06, letter, transform=ax.transAxes,
        fontsize=14, fontweight="bold", va="bottom", ha="right", clip_on=False,
    )


def plot_feature_selection_frequency_2x2(
    out_stem: Path | None = None,
) -> list[Path]:
    out_stem = out_stem or (OUT_DIR / "feature_selection_frequency_2x2")
    out_stem.parent.mkdir(parents=True, exist_ok=True)

    # me/soi modules set Times; re-apply Nature-style Helvetica/Arial.
    apply_plot_style()

    fig, axes = plt.subplots(2, 2, figsize=(18.5, 11.5))
    for ax, (letter, title, kind, task) in zip(axes.ravel(), PANEL_SPECS):
        if kind == "me":
            _draw_me_panel(ax, task=task, title=title, letter=letter)
        else:
            _draw_soi_panel(ax, task=task, title=title, letter=letter)

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.40, wspace=0.58)

    paths: list[Path] = []
    for fmt in ("pdf", "svg"):
        p = out_stem.with_suffix(f".{fmt}")
        fig.savefig(p, format=fmt, dpi=400, bbox_inches="tight", pad_inches=0.08)
        paths.append(p)
        print(f"Figure saved → {p}")
    plt.close(fig)
    return paths


if __name__ == "__main__":
    plot_feature_selection_frequency_2x2()
