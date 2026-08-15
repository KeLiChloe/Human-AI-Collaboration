"""
Equal-sized crowd aggregation analysis.

Matches Humans and GenAI at the same crowd size k by repeatedly sampling
k forecasters without replacement from each pool, aggregating forecast
vectors with the same sum→cosine rule as aggregation_analysis / 06_*,
and summarizing aggregated accuracy, raw gain, and ceiling-normalized gain.

Normalized gain = (agg − mean) / (1 − mean), NaN when remaining room ≤ ε.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

FORECASTS = Path(__file__).resolve().parent
ROOT = FORECASTS.parent
for p in (ROOT, FORECASTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from aggregation_analysis import (
    HUMAN_GROUP_IDS,
    GENAI_GROUP_IDS,
    compute_from_plot_pts,
    cosine_sim,
    load_main_effects_records,
    load_soi_records,
    plot_pts_main_effects,
    plot_pts_soi,
)
from viz_config import COLOR_AGG_HUMAN, GROUP_COLORS

OUT_DIR = FORECASTS / "outputs"
OUT_DIR.mkdir(exist_ok=True)

ME_FIG_DIR = FORECASTS / "main_effects" / "figures"
SOI_FIG_DIR = FORECASTS / "second_order_interactions" / "figures"
ME_FIG_DIR.mkdir(parents=True, exist_ok=True)
SOI_FIG_DIR.mkdir(parents=True, exist_ok=True)

B = 1000
SEED = 20260714
NORM_EPS = 1e-6
K_MIN = 2

# Human-subgroup equal-size figure (Senior / PhD / Topic / Non-Topic).
# Each entry: (label, group_id|None, is_topic_expert|None, color)
HUMAN_SUBGROUP_DEFS = (
    ("Senior Scientists", "1", None, GROUP_COLORS["senior"]),
    ("PhD Students", "0", None, GROUP_COLORS["phd"]),
    ("Topic Experts", None, True, GROUP_COLORS["topic"]),
    ("Non-Topic Experts", None, False, GROUP_COLORS["non_topic"]),
)

TASKS = ("Race", "Gender")
TASK_LABELS = {
    "Race": "Racial Inequality",
    "Gender": "Gender Inequality",
}

plt.rcParams.update({
    "figure.dpi": 180,
    "savefig.dpi": 300,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "DejaVu Sans", "Arial"],
    "mathtext.fontset": "dejavusans",
    "mathtext.default": "regular",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "axes.linewidth": 1.0,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11.5,
    "legend.frameon": False,
    "grid.alpha": 0.22,
    "grid.linestyle": ":",
    "lines.linewidth": 1.85,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.titleweight": "normal",
    "axes.labelweight": "normal",
})


def _split_pools(pts: list[dict]) -> tuple[list[dict], list[dict]]:
    humans = [p for p in pts if p["group"] in HUMAN_GROUP_IDS]
    genai = [p for p in pts if p["group"] in GENAI_GROUP_IDS]
    return humans, genai


def _sample_metrics(
    pool: list[dict],
    ml_vec: np.ndarray,
    k: int,
    rng: np.random.Generator,
) -> tuple[float, float, float, float]:
    """Return mean_ind, agg, gain, norm_gain for one subsample of size k."""
    idx = rng.choice(len(pool), size=k, replace=False)
    selected = [pool[i] for i in idx]
    scores = np.array([p["score"] for p in selected], dtype=float)
    vecs = np.array([p["vec"] for p in selected], dtype=float)
    mean_ind = float(np.mean(scores))
    agg = cosine_sim(np.sum(vecs, axis=0), ml_vec)
    gain = agg - mean_ind
    room = 1.0 - mean_ind
    norm_gain = float(gain / room) if room > NORM_EPS else np.nan
    return mean_ind, agg, gain, norm_gain


def resample_curve(
    pool: list[dict],
    ml_vec: np.ndarray,
    k_values: list[int],
    n_draws: int,
    rng: np.random.Generator,
) -> dict[int, dict[str, np.ndarray]]:
    """For each k, arrays of length n_draws for mean_ind / agg / gain / norm_gain."""
    out: dict[int, dict[str, np.ndarray]] = {}
    n = len(pool)
    for k in k_values:
        if k > n:
            continue
        mean_inds = np.empty(n_draws)
        aggs = np.empty(n_draws)
        gains = np.empty(n_draws)
        norms = np.empty(n_draws)
        for b in range(n_draws):
            mean_inds[b], aggs[b], gains[b], norms[b] = _sample_metrics(
                pool, ml_vec, k, rng
            )
        out[k] = {
            "mean_ind": mean_inds,
            "agg": aggs,
            "gain": gains,
            "norm_gain": norms,
        }
    return out


def _array_summary(arr: np.ndarray) -> dict[str, float]:
    """Mean / median / subsample percentiles, plus 95% CI of the Monte Carlo mean."""
    clean = arr[~np.isnan(arr)]
    if clean.size == 0:
        return {
            "mean": np.nan,
            "median": np.nan,
            "p025": np.nan,
            "p975": np.nan,
            "ci_lo": np.nan,
            "ci_hi": np.nan,
        }
    mean = float(np.mean(clean))
    n = int(clean.size)
    if n >= 2:
        se = float(np.std(clean, ddof=1) / np.sqrt(n))
        half = 1.96 * se
    else:
        half = 0.0
    return {
        "mean": mean,
        "median": float(np.median(clean)),
        "p025": float(np.percentile(clean, 2.5)),
        "p975": float(np.percentile(clean, 97.5)),
        "ci_lo": mean - half,
        "ci_hi": mean + half,
    }


def summarize_curves(
    analysis: str,
    task: str,
    group: str,
    n_pool: int,
    curves: dict[int, dict[str, np.ndarray]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for k, arrays in sorted(curves.items()):
        mean_s = _array_summary(arrays["mean_ind"])
        agg_s = _array_summary(arrays["agg"])
        gain_s = _array_summary(arrays["gain"])
        norm_s = _array_summary(arrays["norm_gain"])
        rows.append({
            "analysis": analysis,
            "task": task,
            "group": group,
            "k": k,
            "n_pool": n_pool,
            "n_draws": len(arrays["agg"]),
            "mean_individual_mean": mean_s["mean"],
            "mean_individual_median": mean_s["median"],
            "mean_individual_p025": mean_s["p025"],
            "mean_individual_p975": mean_s["p975"],
            "mean_individual_ci_lo": mean_s["ci_lo"],
            "mean_individual_ci_hi": mean_s["ci_hi"],
            "agg_mean": agg_s["mean"],
            "agg_median": agg_s["median"],
            "agg_p025": agg_s["p025"],
            "agg_p975": agg_s["p975"],
            "agg_ci_lo": agg_s["ci_lo"],
            "agg_ci_hi": agg_s["ci_hi"],
            "gain_mean": gain_s["mean"],
            "gain_median": gain_s["median"],
            "gain_p025": gain_s["p025"],
            "gain_p975": gain_s["p975"],
            "gain_ci_lo": gain_s["ci_lo"],
            "gain_ci_hi": gain_s["ci_hi"],
            "norm_gain_mean": norm_s["mean"],
            "norm_gain_median": norm_s["median"],
            "norm_gain_p025": norm_s["p025"],
            "norm_gain_p975": norm_s["p975"],
            "norm_gain_ci_lo": norm_s["ci_lo"],
            "norm_gain_ci_hi": norm_s["ci_hi"],
        })
    return rows


def _plot_series(
    ax,
    k_values: list[int],
    rows_by_k: dict[int, dict[str, object]],
    *,
    mean_key: str,
    lo_key: str,
    hi_key: str,
    color: str,
    label: str,
):
    ks = [k for k in k_values if k in rows_by_k]
    if not ks:
        return
    means = [float(rows_by_k[k][mean_key]) for k in ks]
    los = [float(rows_by_k[k][lo_key]) for k in ks]
    his = [float(rows_by_k[k][hi_key]) for k in ks]
    ax.plot(ks, means, color=color, label=label)
    ax.fill_between(ks, los, his, color=color, alpha=0.18, linewidth=0)


def plot_analysis_panel(
    summary_rows: list[dict[str, object]],
    analysis: str,
    full_refs: dict[str, dict[str, dict[str, float]]],
    out_path: Path,
    title: str,
):
    """2 rows (Race, Gender) × 3 cols (agg, raw gain, norm gain)."""
    fig, axes = plt.subplots(2, 3, figsize=(14.2, 8.6), sharex="col")
    metric_specs = [
        (
            "Aggregated Cosine Accuracy",
            "agg_mean", "agg_p025", "agg_p975", "agg",
            "Cosine similarity",
        ),
        (
            "Aggregation Gain",
            "gain_mean", "gain_p025", "gain_p975", "gain",
            r"$\Delta$ (agg. $-$ mean ind.)",
        ),
        (
            "Normalized Aggregation Gain",
            "norm_gain_mean", "norm_gain_p025", "norm_gain_p975", "norm",
            r"$\Delta$ / $(1 -$ mean ind.$)$",
        ),
    ]

    for row_i, task in enumerate(TASKS):
        task_rows = [
            r for r in summary_rows
            if r["analysis"] == analysis and r["task"] == task
        ]
        human_by_k = {int(r["k"]): r for r in task_rows if r["group"] == "Humans"}
        genai_by_k = {int(r["k"]): r for r in task_rows if r["group"] == "GenAI"}
        k_values = sorted(set(human_by_k) | set(genai_by_k))
        ref = full_refs[analysis][task]

        for col_i, (col_title, mean_key, lo_key, hi_key, kind, y_metric) in enumerate(metric_specs):
            ax = axes[row_i, col_i]
            _plot_series(
                ax, k_values, human_by_k,
                mean_key=mean_key, lo_key=lo_key, hi_key=hi_key,
                color=COLOR_AGG_HUMAN, label="Humans",
            )
            _plot_series(
                ax, k_values, genai_by_k,
                mean_key=mean_key, lo_key=lo_key, hi_key=hi_key,
                color=GROUP_COLORS["genai"], label="GenAI",
            )

            # Full-crowd reference: dashed horizontals (unmatched N)
            if kind == "agg":
                ax.axhline(ref["agg_human"], color=COLOR_AGG_HUMAN, ls="--", lw=1.15, alpha=0.75)
                ax.axhline(ref["agg_genai"], color=GROUP_COLORS["genai"], ls="--", lw=1.15, alpha=0.75)
            elif kind == "gain":
                ax.axhline(ref["gain_human"], color=COLOR_AGG_HUMAN, ls="--", lw=1.15, alpha=0.75)
                ax.axhline(ref["gain_genai"], color=GROUP_COLORS["genai"], ls="--", lw=1.15, alpha=0.75)
            else:
                room_h = 1.0 - ref["avg_human"]
                room_g = 1.0 - ref["avg_genai"]
                if room_h > NORM_EPS:
                    ax.axhline(ref["gain_human"] / room_h, color=COLOR_AGG_HUMAN, ls="--", lw=1.15, alpha=0.75)
                if room_g > NORM_EPS:
                    ax.axhline(ref["gain_genai"] / room_g, color=GROUP_COLORS["genai"], ls="--", lw=1.15, alpha=0.75)

            ax.grid(True, axis="both")
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(axis="both", labelsize=12, length=4, width=0.9)
            if k_values:
                ax.set_xlim(k_values[0] - 0.35, k_values[-1] + 0.35)
                ax.set_xticks(_crowd_size_ticks(k_values))

            if row_i == 0:
                ax.set_title(col_title, fontsize=14, fontweight="bold", pad=10)
            if col_i == 0:
                task_math = TASK_LABELS[task].replace(" ", r"\ ")
                ax.set_ylabel(
                    rf"$\mathbf{{{task_math}}}$" + f"\n{y_metric}",
                    fontsize=13.5,
                    linespacing=1.35,
                )
            else:
                ax.set_ylabel(y_metric, fontsize=13)
            if row_i == 1:
                ax.set_xlabel("Crowd size", fontsize=14)

    legend_handles = [
        Line2D([0], [0], color=COLOR_AGG_HUMAN, lw=2.2, label="Humans"),
        Line2D([0], [0], color=GROUP_COLORS["genai"], lw=2.2, label="GenAI"),
        Line2D(
            [0], [0], color="0.35", lw=1.2, ls="--",
            label="Full crowd (same color)",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=3,
        fontsize=12.5,
        frameon=False,
        handlelength=2.4,
        columnspacing=1.6,
    )

    fig.suptitle(title, fontsize=17, fontweight="bold", y=1.01)
    fig.text(
        0.5,
        0.005,
        (
            f"Equal-sized resampling without replacement ($B={B}$). "
            "Shaded bands: 2.5–97.5th percentiles across resamples. "
            "Dashed lines: full crowds "
            fr"(Humans $n\approx{int(full_refs[analysis]['Race']['n_human'])}$, "
            fr"GenAI $n={int(full_refs[analysis]['Race']['n_genai'])}$)."
        ),
        ha="center",
        va="bottom",
        fontsize=10.5,
        style="italic",
        color="0.25",
    )
    fig.tight_layout(rect=(0.01, 0.045, 0.99, 0.92))
    out_path = Path(out_path)
    stem = out_path.with_suffix("")
    for fmt in ("pdf", "svg"):
        fig.savefig(Path(f"{stem}.{fmt}"), format=fmt, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def _fill_metric_2x2_axes(
    axes,
    summary_rows: list[dict[str, object]],
    full_refs: dict[str, dict[str, dict[str, float]]],
    *,
    mean_key: str,
    lo_key: str,
    hi_key: str,
    ref_human_key: str | None,
    ref_genai_key: str | None,
    ylabel: str,
    panel_letters: tuple[str, str, str, str] = ("a", "b", "c", "d"),
):
    """Draw the 2×2 equal-size metric panels onto an existing axes grid."""
    panels = [
        (panel_letters[0], "Main Effects", "Race", "Racial Inequality — Main Effects"),
        (panel_letters[1], "Main Effects", "Gender", "Gender Inequality — Main Effects"),
        (panel_letters[2], "Interactions", "Race", "Racial Inequality — Interactions"),
        (panel_letters[3], "Interactions", "Gender", "Gender Inequality — Interactions"),
    ]
    draw_refs = ref_human_key is not None and ref_genai_key is not None
    flat_axes = np.asarray(axes).ravel()

    for ax, (letter, analysis, task, panel_title) in zip(flat_axes, panels):
        task_rows = [
            r for r in summary_rows
            if r["analysis"] == analysis and r["task"] == task
        ]
        human_by_k = {int(r["k"]): r for r in task_rows if r["group"] == "Humans"}
        genai_by_k = {int(r["k"]): r for r in task_rows if r["group"] == "GenAI"}
        k_values = sorted(set(human_by_k) | set(genai_by_k))

        _plot_series(
            ax, k_values, human_by_k,
            mean_key=mean_key, lo_key=lo_key, hi_key=hi_key,
            color=COLOR_AGG_HUMAN, label="Humans",
        )
        _plot_series(
            ax, k_values, genai_by_k,
            mean_key=mean_key, lo_key=lo_key, hi_key=hi_key,
            color=GROUP_COLORS["genai"], label="GenAI",
        )
        if draw_refs:
            ref = full_refs[analysis][task]
            ax.axhline(ref[ref_human_key], color=COLOR_AGG_HUMAN, ls="--", lw=1.15, alpha=0.75)
            ax.axhline(ref[ref_genai_key], color=GROUP_COLORS["genai"], ls="--", lw=1.15, alpha=0.75)

        ax.grid(True, axis="both")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="both", labelsize=11, length=4, width=0.9)
        if k_values:
            ax.set_xlim(k_values[0] - 0.35, k_values[-1] + 0.35)
            ax.set_xticks(_crowd_size_ticks(k_values))

        ax.set_title(
            f"{letter}.  {panel_title}",
            fontsize=12.5,
            fontweight="bold",
            pad=8,
            loc="left",
        )
        ax.set_ylabel(ylabel, fontsize=12)
        # sharex hides top-row labels; keep Crowd size on every panel for the paper figure.
        ax.set_xlabel("Crowd size", fontsize=12.5)
        ax.tick_params(axis="x", labelbottom=True)


def plot_main_text_metric_2x2(
    summary_rows: list[dict[str, object]],
    full_refs: dict[str, dict[str, dict[str, float]]],
    out_path: Path,
    *,
    mean_key: str,
    lo_key: str,
    hi_key: str,
    ref_human_key: str | None,
    ref_genai_key: str | None,
    ylabel: str,
    ref_legend_label: str = "Full crowd (same color)",
    panel_letters: tuple[str, str, str, str] = ("a", "b", "c", "d"),
):
    """Main-text 2×2 figure (ME / Interactions × Race / Gender) for one metric."""
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 8.2), sharex=True, sharey=False)
    _fill_metric_2x2_axes(
        axes,
        summary_rows,
        full_refs,
        mean_key=mean_key,
        lo_key=lo_key,
        hi_key=hi_key,
        ref_human_key=ref_human_key,
        ref_genai_key=ref_genai_key,
        ylabel=ylabel,
        panel_letters=panel_letters,
    )
    draw_refs = ref_human_key is not None and ref_genai_key is not None
    legend_handles = [
        Line2D([0], [0], color=COLOR_AGG_HUMAN, lw=2.2, label="Humans"),
        Line2D([0], [0], color=GROUP_COLORS["genai"], lw=2.2, label="GenAI"),
    ]
    if draw_refs:
        legend_handles.append(
            Line2D(
                [0], [0], color="0.35", lw=1.2, ls="--",
                label=ref_legend_label,
            )
        )
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=len(legend_handles),
        fontsize=12.5,
        frameon=False,
        handlelength=2.4,
        columnspacing=1.6,
    )
    fig.tight_layout(rect=(0.01, 0.02, 0.99, 0.94))
    out_path = Path(out_path)
    stem = out_path.with_suffix("")
    for fmt in ("pdf", "svg"):
        fig.savefig(Path(f"{stem}.{fmt}"), format=fmt, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def plot_main_text_gain_2x2(
    summary_rows: list[dict[str, object]],
    full_refs: dict[str, dict[str, dict[str, float]]],
    out_path: Path,
):
    """Main-text 2×2 figure: Aggregation Gain only (no reference dashed lines)."""
    plot_main_text_metric_2x2(
        summary_rows,
        full_refs,
        out_path,
        mean_key="gain_mean",
        lo_key="gain_p025",
        hi_key="gain_p975",
        ref_human_key=None,
        ref_genai_key=None,
        ylabel="Aggregation gain",
    )


def plot_main_text_accuracy_2x2(
    summary_rows: list[dict[str, object]],
    full_refs: dict[str, dict[str, dict[str, float]]],
    out_path: Path,
    *,
    panel_letters: tuple[str, str, str, str] = ("a", "b", "c", "d"),
):
    """Main-text 2×2 figure: Aggregated Accuracy (equal-sized Humans vs GenAI)."""
    plot_main_text_metric_2x2(
        summary_rows,
        full_refs,
        out_path,
        mean_key="agg_mean",
        lo_key="agg_p025",
        hi_key="agg_p975",
        ref_human_key="avg_human",
        ref_genai_key="avg_genai",
        ylabel="Aggregated accuracy",
        ref_legend_label="Mean accuracy (same color)",
        panel_letters=panel_letters,
    )


def _filter_subgroup_pool(
    pts: list[dict],
    *,
    group_id: str | None,
    is_topic_expert: bool | None,
) -> list[dict]:
    out: list[dict] = []
    for p in pts:
        if group_id is not None:
            if p["group"] == group_id:
                out.append(p)
            continue
        if p["group"] not in HUMAN_GROUP_IDS:
            continue
        if bool(p.get("is_topic_expert")) == bool(is_topic_expert):
            out.append(p)
    return out


def _split_human_subgroup_pools(pts: list[dict]) -> dict[str, list[dict]]:
    return {
        label: _filter_subgroup_pool(
            pts, group_id=group_id, is_topic_expert=is_topic
        )
        for label, group_id, is_topic, _ in HUMAN_SUBGROUP_DEFS
    }


def _pool_mean_accuracy(pool: list[dict]) -> float:
    if not pool:
        return float("nan")
    return float(np.mean([p["score"] for p in pool]))


def run_one_human_subgroups(
    analysis: str,
    pts: list[dict],
    ml_vec: np.ndarray,
    task: str,
    rng: np.random.Generator,
) -> tuple[list[dict[str, object]], dict[str, float]]:
    pools = _split_human_subgroup_pools(pts)
    rows: list[dict[str, object]] = []
    refs: dict[str, float] = {}
    for label, pool in pools.items():
        refs[f"avg_{label}"] = _pool_mean_accuracy(pool)
        refs[f"n_{label}"] = float(len(pool))
        if len(pool) < K_MIN:
            continue
        ks = list(range(K_MIN, len(pool) + 1))
        curves = resample_curve(pool, ml_vec, ks, B, rng)
        rows.extend(summarize_curves(analysis, task, label, len(pool), curves))
    return rows, refs


def plot_human_subgroup_accuracy_2x2(
    summary_rows: list[dict[str, object]],
    full_refs: dict[str, dict[str, dict[str, float]]],
    out_path: Path,
    *,
    panel_letters: tuple[str, str, str, str] = ("a", "b", "c", "d"),
):
    """2×2 aggregated-accuracy figure for Senior / PhD / Topic / Non-Topic."""
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 8.2), sharex=True, sharey=False)
    panels = [
        (panel_letters[0], "Main Effects", "Race", "Racial Inequality — Main Effects"),
        (panel_letters[1], "Main Effects", "Gender", "Gender Inequality — Main Effects"),
        (panel_letters[2], "Interactions", "Race", "Racial Inequality — Interactions"),
        (panel_letters[3], "Interactions", "Gender", "Gender Inequality — Interactions"),
    ]
    group_labels = [label for label, _, _, _ in HUMAN_SUBGROUP_DEFS]
    group_colors = {label: color for label, _, _, color in HUMAN_SUBGROUP_DEFS}

    for ax, (letter, analysis, task, panel_title) in zip(np.asarray(axes).ravel(), panels):
        task_rows = [
            r for r in summary_rows
            if r["analysis"] == analysis and r["task"] == task
        ]
        by_group = {
            label: {int(r["k"]): r for r in task_rows if r["group"] == label}
            for label in group_labels
        }
        k_values = sorted({k for d in by_group.values() for k in d})
        ref = full_refs[analysis][task]

        for label in group_labels:
            _plot_series(
                ax,
                k_values,
                by_group[label],
                mean_key="agg_mean",
                lo_key="agg_p025",
                hi_key="agg_p975",
                color=group_colors[label],
                label=label,
            )
            mean_acc = ref.get(f"avg_{label}", np.nan)
            if np.isfinite(mean_acc):
                ax.axhline(
                    mean_acc,
                    color=group_colors[label],
                    ls="--",
                    lw=1.15,
                    alpha=0.75,
                )

        ax.grid(True, axis="both")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="both", labelsize=11, length=4, width=0.9)
        if k_values:
            ax.set_xlim(k_values[0] - 0.35, k_values[-1] + 0.35)
            ax.set_xticks(_crowd_size_ticks(k_values))
        ax.set_title(
            f"{letter}.  {panel_title}",
            fontsize=12.5,
            fontweight="bold",
            pad=8,
            loc="left",
        )
        ax.set_ylabel("Aggregated accuracy", fontsize=12)
        ax.set_xlabel("Crowd size", fontsize=12.5)
        ax.tick_params(axis="x", labelbottom=True)

    legend_handles = [
        Line2D([0], [0], color=group_colors[label], lw=2.2, label=label)
        for label in group_labels
    ]
    legend_handles.append(
        Line2D(
            [0], [0], color="0.35", lw=1.2, ls="--",
            label="Mean accuracy (same color)",
        )
    )
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=3,
        fontsize=11.5,
        frameon=False,
        handlelength=2.4,
        columnspacing=1.4,
    )
    fig.tight_layout(rect=(0.01, 0.02, 0.99, 0.92))
    out_path = Path(out_path)
    stem = out_path.with_suffix("")
    for fmt in ("pdf", "svg"):
        fig.savefig(Path(f"{stem}.{fmt}"), format=fmt, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def run_one(
    analysis: str,
    pts: list[dict],
    ml_vec: np.ndarray,
    task: str,
    rng: np.random.Generator,
) -> tuple[list[dict[str, object]], dict[str, float]]:
    humans, genai = _split_pools(pts)
    full = compute_from_plot_pts(pts, ml_vec)
    # Sample each group up to its own pool size (Humans to n_H, GenAI to n_AI).
    human_ks = list(range(K_MIN, len(humans) + 1))
    genai_ks = list(range(K_MIN, len(genai) + 1))

    human_curves = resample_curve(humans, ml_vec, human_ks, B, rng)
    genai_curves = resample_curve(genai, ml_vec, genai_ks, B, rng)

    rows = []
    rows.extend(summarize_curves(analysis, task, "Humans", len(humans), human_curves))
    rows.extend(summarize_curves(analysis, task, "GenAI", len(genai), genai_curves))
    return rows, full


def _crowd_size_ticks(k_values: list[int]) -> list[int]:
    if not k_values:
        return []
    lo, hi = k_values[0], k_values[-1]
    span = hi - lo
    if span > 40:
        step = 10
    elif span > 20:
        step = 5
    else:
        step = 2
    # Evenly spaced labels; stop at the last grid point ≤ hi (e.g. 72 when hi=73).
    last_label = hi - ((hi - lo) % step)
    return list(range(lo, last_label + 1, step))


def main():
    csv_path = ROOT / "All_Participants_All_Questions.csv"
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        rows_csv = list(csv.reader(f))
    headers, data = rows_csv[0], rows_csv[1:]

    me_records, me_ml = load_main_effects_records(headers, data)
    soi_records, soi_ml = load_soi_records(headers, data)

    rng = np.random.default_rng(SEED)
    all_summary: list[dict[str, object]] = []
    full_refs: dict[str, dict[str, dict[str, float]]] = {
        "Main Effects": {},
        "Interactions": {},
    }
    subgroup_summary: list[dict[str, object]] = []
    subgroup_refs: dict[str, dict[str, dict[str, float]]] = {
        "Main Effects": {},
        "Interactions": {},
    }

    for task in TASKS:
        task_key = "cos_race" if task == "Race" else "cos_gender"
        vec_key = "vec_race_bin" if task == "Race" else "vec_gender_bin"

        me_pts = plot_pts_main_effects(me_records, task_key, vec_key)
        me_rows, me_full = run_one("Main Effects", me_pts, me_ml[task], task, rng)
        all_summary.extend(me_rows)
        full_refs["Main Effects"][task] = me_full
        me_sg_rows, me_sg_refs = run_one_human_subgroups(
            "Main Effects", me_pts, me_ml[task], task, rng
        )
        subgroup_summary.extend(me_sg_rows)
        subgroup_refs["Main Effects"][task] = me_sg_refs

        soi_pts = plot_pts_soi(soi_records, task_key, vec_key)
        soi_rows, soi_full = run_one("Interactions", soi_pts, soi_ml[task], task, rng)
        all_summary.extend(soi_rows)
        full_refs["Interactions"][task] = soi_full
        soi_sg_rows, soi_sg_refs = run_one_human_subgroups(
            "Interactions", soi_pts, soi_ml[task], task, rng
        )
        subgroup_summary.extend(soi_sg_rows)
        subgroup_refs["Interactions"][task] = soi_sg_refs

    summary_csv = OUT_DIR / "equal_size_aggregation_summary.csv"
    fieldnames = [
        "analysis", "task", "group", "k", "n_pool", "n_draws",
        "mean_individual_mean", "mean_individual_median",
        "mean_individual_p025", "mean_individual_p975",
        "mean_individual_ci_lo", "mean_individual_ci_hi",
        "agg_mean", "agg_median", "agg_p025", "agg_p975",
        "agg_ci_lo", "agg_ci_hi",
        "gain_mean", "gain_median", "gain_p025", "gain_p975",
        "gain_ci_lo", "gain_ci_hi",
        "norm_gain_mean", "norm_gain_median", "norm_gain_p025", "norm_gain_p975",
        "norm_gain_ci_lo", "norm_gain_ci_hi",
    ]
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_summary:
            writer.writerow({
                k: (
                    f"{row[k]:.8f}" if isinstance(row[k], float) else row[k]
                )
                for k in fieldnames
            })

    me_pdf = ME_FIG_DIR / "equal_size_aggregation_curves.pdf"
    plot_analysis_panel(
        all_summary, "Main Effects", full_refs, me_pdf,
        "Equal-sized Aggregation — Main Effects",
    )

    main_text_gain_pdf = OUT_DIR / "equal_size_aggregation_gain_2x2.pdf"
    plot_main_text_gain_2x2(all_summary, full_refs, main_text_gain_pdf)

    main_text_accuracy_pdf = OUT_DIR / "equal_size_aggregation_accuracy_2x2.pdf"
    plot_main_text_accuracy_2x2(all_summary, full_refs, main_text_accuracy_pdf)

    subgroup_accuracy_pdf = OUT_DIR / "equal_size_aggregation_accuracy_2x2_human_subgroups.pdf"
    plot_human_subgroup_accuracy_2x2(
        subgroup_summary, subgroup_refs, subgroup_accuracy_pdf
    )

    subgroup_csv = OUT_DIR / "equal_size_aggregation_summary_human_subgroups.csv"
    with subgroup_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in subgroup_summary:
            writer.writerow({
                k: (
                    f"{row[k]:.8f}" if isinstance(row[k], float) else row[k]
                )
                for k in fieldnames
            })

    for stale in (
        OUT_DIR / "aggregation_table_and_accuracy_combined.pdf",
        OUT_DIR / "aggregation_accuracy_and_table_combined.pdf",
        OUT_DIR / "aggregation_accuracy_and_table_combined.tex",
        OUT_DIR / "equal_size_aggregation_accuracy_panelA.pdf",
        OUT_DIR / "aggregation_gain_table_embed.tex",
        OUT_DIR / "aggregation_gain_table_embed_standalone.tex",
        OUT_DIR / "aggregation_gain_table_embed_standalone.pdf",
        SOI_FIG_DIR / "equal_size_aggregation_curves.pdf",
        SOI_FIG_DIR / "equal_size_aggregation_curves.svg",
    ):
        if stale.is_file():
            stale.unlink()

    # Console checks
    print(f"Equal-sized aggregation (B={B}, seed={SEED})")
    print(f"Saved: {summary_csv}")
    print(f"Saved: {me_pdf}")
    print(f"Saved: {main_text_gain_pdf}")
    print(f"Saved: {main_text_accuracy_pdf}")
    print(f"Saved: {subgroup_accuracy_pdf}")
    print(f"Saved: {subgroup_csv}\n")

    for analysis in ("Main Effects", "Interactions"):
        print(f"=== {analysis} ===")
        for task in TASKS:
            ref = full_refs[analysis][task]
            print(
                f"  Full crowds {task}: "
                f"n_H={int(ref['n_human'])} n_AI={int(ref['n_genai'])} | "
                f"agg H={ref['agg_human']:.4f} AI={ref['agg_genai']:.4f} | "
                f"gain H={ref['gain_human']:+.4f} AI={ref['gain_genai']:+.4f}"
            )
            k_star = int(ref["n_genai"])
            for group in ("Humans", "GenAI"):
                match = [
                    r for r in all_summary
                    if r["analysis"] == analysis
                    and r["task"] == task
                    and r["group"] == group
                    and int(r["k"]) == k_star
                ]
                if not match:
                    continue
                r = match[0]
                tag = "full-pool check" if group == "GenAI" else "matched-k"
                print(
                    f"    k={k_star} {group} ({tag}): "
                    f"agg={float(r['agg_mean']):.4f} "
                    f"gain={float(r['gain_mean']):+.4f} "
                    f"norm={float(r['norm_gain_mean']):+.4f}"
                )
                if group == "GenAI":
                    delta = abs(float(r["agg_mean"]) - ref["agg_genai"])
                    print(f"      |agg_mean − full GenAI agg| = {delta:.6f}")
            # Matched-k gain gap at k_star
            h = next(
                r for r in all_summary
                if r["analysis"] == analysis and r["task"] == task
                and r["group"] == "Humans" and int(r["k"]) == k_star
            )
            g = next(
                r for r in all_summary
                if r["analysis"] == analysis and r["task"] == task
                and r["group"] == "GenAI" and int(r["k"]) == k_star
            )
            print(
                f"    Matched-k gain gap (H−AI) at k={k_star}: "
                f"{float(h['gain_mean']) - float(g['gain_mean']):+.4f} "
                f"(norm: {float(h['norm_gain_mean']) - float(g['norm_gain_mean']):+.4f})"
            )
        print()


if __name__ == "__main__":
    main()
