"""
Second-Order Interactions (SOI) Analysis
=======================================
Quantitative Metrics (78-dimensional interaction space)

Dimension = C(13, 2) = 78 unordered interactions.

Metrics:
1) Cosine similarity (binary signed: +1 / -1 / 0)
"""

import csv
import json
import re
import sys
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
TEXTUAL_DIR = ROOT / "textual_analysis"
for p in (ROOT, TEXTUAL_DIR):
    if str(p) not in sys.path:
        sys.path.append(str(p))
from stats_utils import bootstrap_ci_half_width, welch_test
from viz_config import COLOR_AGG_HUMAN, GROUP_COLORS, SIGN_COLORS
from viz_style import (
    apply_bottom_layout,
    comparison_pair_label,
    draw_centered_comparison_box,
    draw_sig_footnote,
    SIG_LEVEL_LEGEND,
)

SOI_WELCH_THREE_GROUP_FOOTNOTE = (
    "Brackets: two-sided Welch t-test on pairwise group mean ML feature-selection accuracy "
    "(PhD Students vs Experts, PhD Students vs GenAI, Experts vs GenAI).",
    SIG_LEVEL_LEGEND,
)
SOI_WELCH_HUMAN_GENAI_FOOTNOTE = (
    "Brackets: two-sided Welch t-test on mean ML feature-selection accuracy (Humans vs GenAI).",
    SIG_LEVEL_LEGEND,
)

QUANT_COMPARISON_BOX_BOTTOM = 0.062
QUANT_COLLAPSED_COMPARISON_BOX_BOTTOM = 0.034
QUANT_COLLAPSED_AXIS_GAP = 0.11
QUANT_FOOTNOTE_Y = 0.001
QUANT_COLLAPSED_FOOTNOTE_Y = -0.010
QUANT_BOX_MAX_WIDTH_FRAC = 0.90

plt.rcParams.update({
    "figure.dpi": 180,
    "savefig.dpi": 900,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10.5,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.linewidth": 0.9,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 12,
    "legend.frameon": False,
    "grid.alpha": 0.2,
    "lines.linewidth": 1.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def save_figure(out_path):
    """Export high-res PNG only."""
    out_path = Path(out_path)
    plt.savefig(out_path, dpi=900, bbox_inches="tight")

BASE = Path(__file__).parent
CSV_PATH = BASE.parent / "All_Participants_All_Questions.csv"
ML_PATH = BASE / "ML_results.json"
OUT_DIR = BASE / "figures"
OUT_DIR.mkdir(exist_ok=True)
COLOR_RANDOM_BENCH = "#4A4A4A"
SOI_COSINE_SUMMARY_TITLE = (
    "Second-Order Interactions Accuracy (combining importance and sign) - Mean ± 95% CI"
)

def canon_pair(a, b):
    return tuple(sorted((a.strip(), b.strip())))


def parse_pair(cell, valid_features):
    cell = cell.strip()
    if not cell or "," not in cell:
        return None
    parts = [x.strip() for x in cell.split(",")]
    if len(parts) != 2 or parts[0] == parts[1]:
        return None
    if parts[0] not in valid_features or parts[1] not in valid_features:
        return None
    return canon_pair(parts[0], parts[1])


with open(CSV_PATH, encoding="utf-8-sig", newline="") as f:
    rows = list(csv.reader(f))
headers = rows[0]
data = rows[1:]

with open(ML_PATH) as f:
    ml_raw = json.load(f)

features = [
    re.sub(r"^Q Race\.2 \(rank\) - ", "", h)
    for h in headers
    if re.match(r"^Q Race\.2 \(rank\) - ", h)
]
feature_set = set(features)
pairs = list(combinations(sorted(features), 2))  # 78
PAIR_IDX = {p: i for i, p in enumerate(pairs)}
SIGN_MAP = {"+": 1, "-": -1}

# Pooled respondent subsets for aggregate metrics (not per-bar group means).
# Human = PhD Students (0) + Expert (1) combined; GenAI (2) is computed separately.
HUMAN_GROUP_IDS = frozenset({"0", "1"})
EXPERT_GROUP_IDS = frozenset({"1"})
PHD_GROUP_IDS = frozenset({"0"})
GENAI_GROUP_IDS = frozenset({"2"})

expert_col = next(i for i, h in enumerate(headers) if "expert_1" in h)
r_pair_cols = [
    next(i for i, h in enumerate(headers) if h.strip() == "Q Race.6 (SOI, 1st)"),
    next(i for i, h in enumerate(headers) if h.strip() == "Q Race.7 (SOI, 2nd)"),
    next(i for i, h in enumerate(headers) if h.strip() == "Q Race.8 (SOI, 3rd)"),
]
r_sign_cols = [
    next(i for i, h in enumerate(headers) if h.strip() == "Q Race.9 (SOI, sign, 1st)"),
    next(i for i, h in enumerate(headers) if h.strip() == "Q Race.9 (SOI, sign, 2nd)"),
    next(i for i, h in enumerate(headers) if h.strip() == "Q Race.9 (SOI, sign, 3rd)"),
]
g_pair_cols = [
    next(i for i, h in enumerate(headers) if h.strip() == "Q Gender.6 (SOI, 1st)"),
    next(i for i, h in enumerate(headers) if h.strip() == "Q Gender.7 (SOI, 2nd)"),
    next(i for i, h in enumerate(headers) if h.strip() == "Q Gender.8 (SOI, 3rd)"),
]
g_sign_cols = [
    next(i for i, h in enumerate(headers) if h.strip() == "Q Gender.9 (SOI, sign, 1st)"),
    next(i for i, h in enumerate(headers) if h.strip() == "Q Gender.9 (SOI, sign, 2nd)"),
    next(i for i, h in enumerate(headers) if h.strip() == "Q Gender.9 (SOI, sign, 3rd)"),
]


def build_binary_vector(pair_cols, sign_cols, row):
    v = np.zeros(len(pairs))
    for pc, sc in zip(pair_cols, sign_cols):
        p = parse_pair(row[pc], feature_set)
        if p is None:
            continue
        v[PAIR_IDX[p]] = SIGN_MAP.get(row[sc].strip(), 0)
    return v


def build_ml_binary(entries):
    v = np.zeros(len(pairs))
    for e in entries:
        p = canon_pair(e["feature_1"], e["feature_2"])
        if p in PAIR_IDX:
            v[PAIR_IDX[p]] = SIGN_MAP.get(e["sign"], 0)
    return v


def cosine_sim(a, b):
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d else np.nan


RANDOM_BENCHMARK_N = 1000


def random_benchmark_cosine(ml_vec, vec_dim, nonzero_count, n_random=RANDOM_BENCHMARK_N):
    """Monte Carlo estimate of E[cosine(random signed sparse vector, ML vector)]."""
    rng = np.random.default_rng()
    ml_norm = np.linalg.norm(ml_vec)
    if ml_norm == 0:
        return np.nan
    scores = np.empty(n_random, dtype=float)
    for i in range(n_random):
        vec = np.zeros(vec_dim, dtype=float)
        idx = rng.choice(vec_dim, size=nonzero_count, replace=False)
        vec[idx] = rng.choice([-1, 1], size=nonzero_count)
        vnorm = np.linalg.norm(vec)
        scores[i] = float(np.dot(vec, ml_vec) / (vnorm * ml_norm)) if vnorm else np.nan
    return float(np.nanmean(scores))


def compute_hit_rate(pair_cols, ml_pairs_signs, row):
    """Fraction of ML-important pairs that the respondent selected (sign-agnostic)."""
    selected = set()
    for pc in pair_cols:
        p = parse_pair(row[pc], feature_set)
        if p is not None:
            selected.add(p)
    ml_pairs = set(ml_pairs_signs.keys())
    return len(selected & ml_pairs) / len(ml_pairs) if ml_pairs else np.nan


def compute_sign_align_rate(pair_cols, sign_cols, ml_pairs_signs, row):
    """Among selected ML-important pairs, fraction with correct sign.
    Returns nan if the respondent selected no ML-important pairs at all."""
    selected_with_sign = {}
    for pc, sc in zip(pair_cols, sign_cols):
        p = parse_pair(row[pc], feature_set)
        if p is not None:
            selected_with_sign[p] = SIGN_MAP.get(row[sc].strip(), 0)
    overlapping = {p: s for p, s in selected_with_sign.items() if p in ml_pairs_signs}
    if not overlapping:
        return np.nan
    aligned = sum(
        1 for p, s in overlapping.items()
        if s == SIGN_MAP.get(ml_pairs_signs[p], 0)
    )
    return aligned / len(overlapping)


def aggregated_sign_align_majority_excluding_ties(pair_cols, sign_cols, ml_pairs_signs, pooled_subset_ids):
    """Majority-sign vote per ML pair over rows in pooled_subset_ids; ties excluded.

    Typical use: HUMAN_GROUP_IDS (Expert+PhD pooled) or GENAI_GROUP_IDS alone.
    """
    vote_sums = {p: 0 for p in ml_pairs_signs}
    for row in data:
        if row[expert_col].strip() not in pooled_subset_ids:
            continue
        for pc, sc in zip(pair_cols, sign_cols):
            p = parse_pair(row[pc], feature_set)
            if p is None or p not in ml_pairs_signs:
                continue
            vote_sums[p] += SIGN_MAP.get(row[sc].strip(), 0)

    aligned = 0
    considered = 0
    tie_count = 0
    for p, sum_vote in vote_sums.items():
        if sum_vote == 0:
            tie_count += 1
            continue
        considered += 1
        majority_sign = 1 if sum_vote > 0 else -1
        if majority_sign == SIGN_MAP.get(ml_pairs_signs[p], 0):
            aligned += 1
    rate = (aligned / considered) if considered else np.nan
    return rate, tie_count, considered, len(vote_sums)


ml_bin_r = build_ml_binary(ml_raw["race"])
ml_bin_g = build_ml_binary(ml_raw["gender"])

RANDOM_BENCHMARK_BY_TASK = {
    "cos_race": random_benchmark_cosine(ml_bin_r, len(pairs), 3),
    "cos_gender": random_benchmark_cosine(ml_bin_g, len(pairs), 3),
}

# {canon_pair: sign_str} — used for hit rate and sign alignment rate
ml_pairs_signs_r = {canon_pair(e["feature_1"], e["feature_2"]): e["sign"] for e in ml_raw["race"]}
ml_pairs_signs_g = {canon_pair(e["feature_1"], e["feature_2"]): e["sign"] for e in ml_raw["gender"]}

records = []
for row in data:
    gid = row[expert_col].strip()
    hr_bin = build_binary_vector(r_pair_cols, r_sign_cols, row)
    hg_bin = build_binary_vector(g_pair_cols, g_sign_cols, row)

    records.append(
        {
            "group": gid,
            "vec_race_bin": hr_bin,
            "vec_gender_bin": hg_bin,
            "cos_race":         cosine_sim(hr_bin, ml_bin_r),
            "cos_gender":       cosine_sim(hg_bin, ml_bin_g),
            "hit_race":         compute_hit_rate(r_pair_cols, ml_pairs_signs_r, row),
            "hit_gender":       compute_hit_rate(g_pair_cols, ml_pairs_signs_g, row),
            "sign_align_race":  compute_sign_align_rate(r_pair_cols, r_sign_cols, ml_pairs_signs_r, row),
            "sign_align_gender": compute_sign_align_rate(g_pair_cols, g_sign_cols, ml_pairs_signs_g, row),
        }
    )


def stats_by_group(key, group_val):
    vals = [r[key] for r in records if r["group"] == group_val and not np.isnan(r[key])]
    return {
        "n": len(vals),
        "mean": np.mean(vals) if vals else np.nan,
        "median": np.median(vals) if vals else np.nan,
        "std": np.std(vals) if vals else np.nan,
        "vals": vals,
    }


def stats_all(key):
    vals = [r[key] for r in records if not np.isnan(r[key])]
    return {"n": len(vals), "mean": np.mean(vals), "median": np.median(vals), "std": np.std(vals), "vals": vals}

def stats_human(key):
    vals = [r[key] for r in records
            if r["group"] in HUMAN_GROUP_IDS and not np.isnan(r[key])]
    return {"n": len(vals), "mean": np.mean(vals), "median": np.median(vals), "std": np.std(vals), "vals": vals}

def stats_genai(key):
    vals = [r[key] for r in records
            if r["group"] in GENAI_GROUP_IDS and not np.isnan(r[key])]
    return {"n": len(vals), "mean": np.mean(vals), "median": np.median(vals), "std": np.std(vals), "vals": vals}

def format_legend_value(value):
    return "n/a" if np.isnan(value) else f"{value:.3f}"


def _collapsed_panel_ylim(means, errs, default_ylim):
    err_vals = [0 if not np.isfinite(e) else e for e in errs]
    data_lo = min(m - e for m, e in zip(means, err_vals))
    data_hi = max(m + e for m, e in zip(means, err_vals))
    span = max(data_hi - data_lo, 0.01)
    pad = max(span * 0.18, 0.04)
    ymin = data_lo - pad
    ymax = data_hi + pad
    ymax += (ymax - ymin) * 0.14
    return max(ymin, default_ylim[0]), min(ymax, default_ylim[1])


def _cosine_aggregation_scores(pts, ml_vec):
    def _agg(group_ids):
        vecs = [p["vec"] for p in pts if p["group"] in group_ids]
        return cosine_sim(np.sum(vecs, axis=0), ml_vec) if vecs else np.nan

    return {
        "expert": _agg(EXPERT_GROUP_IDS),
        "phd": _agg(PHD_GROUP_IDS),
        "human": _agg(HUMAN_GROUP_IDS),
        "genai": _agg(GENAI_GROUP_IDS),
    }


def _draw_horizontal_reference(ax, y, color, linestyle, linewidth=1.5, alpha=0.95, zorder=1):
    if not np.isnan(y):
        ax.axhline(y, color=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha, zorder=zorder)


def _legend_line(color, linestyle, label, linewidth=1.5, marker=None, markevery=None, markersize=6.5):
    kwargs = {"color": color, "linestyle": linestyle, "linewidth": linewidth, "label": label}
    if marker:
        kwargs.update(
            marker=marker,
            markevery=markevery or [1],
            markersize=markersize,
            markerfacecolor=color,
            markeredgecolor="#2a2a2a",
            markeredgewidth=0.55,
        )
        return plt.Line2D([0, 0.5, 1], [0, 0, 0], **kwargs)
    return plt.Line2D([0], [0], **kwargs)


def _build_sorted_figure_legend(ax, aggregations, extra_lines, scatter_legend=True):
    handles = []
    if scatter_legend:
        handles.extend([
            plt.Line2D(
                [0], [0], marker="o", linestyle="None", color=GROUP_COLORS["expert"],
                markersize=8, label="Experts",
            ),
            plt.Line2D(
                [0], [0], marker="o", linestyle="None", color=GROUP_COLORS["phd"],
                markersize=8, label="PhD Students",
            ),
            plt.Line2D(
                [0], [0], marker="o", linestyle="None", color=GROUP_COLORS["genai"],
                markersize=8, label="GenAI",
            ),
        ])
    handles.extend([
        _legend_line(
            COLOR_AGG_HUMAN, "-",
            f"Aggregated Humans = {format_legend_value(aggregations['human'])}",
        ),
        _legend_line(
            GROUP_COLORS["genai"], "-",
            f"Aggregated GenAI = {format_legend_value(aggregations['genai'])}",
        ),
    ])
    handles.extend(extra_lines)
    lg = ax.legend(handles=handles, loc="lower right", frameon=True, fontsize=13.5)
    lg.get_frame().set_edgecolor("#666666")
    lg.get_frame().set_linewidth(0.8)
    return lg


def plot_metric(r_key, g_key, ylabel, ylim, out_path, show_distribution=True):
    return _plot_metric_panels(r_key, g_key, ylabel, ylim, out_path, show_distribution, panel="both")


def plot_metric_collapsed(r_key, g_key, ylabel, ylim, out_path, show_distribution=False):
    return _plot_metric_panels(
        r_key, g_key, ylabel, ylim, out_path, show_distribution, panel="both", collapsed=True
    )


def _plot_metric_panels(
    r_key,
    g_key,
    ylabel,
    ylim,
    out_path,
    show_distribution=True,
    panel="both",
    *,
    collapsed=False,
):
    """
    panel:
      - "both": 1x2 panels (race + gender) — current default output
      - "race": race panel only
      - "gender": gender panel only
    """
    is_summary_panel = not show_distribution
    if panel not in {"both", "race", "gender"}:
        raise ValueError(f"Invalid panel={panel!r}")
    if panel != "both" and show_distribution:
        raise ValueError("Single-panel export does not support show_distribution=True")

    if show_distribution:
        fig, axes = plt.subplots(2, 2, figsize=(12, 9), gridspec_kw={"width_ratios": [1, 1.6]})
    else:
        if panel == "both":
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
        else:
            fig, axes = plt.subplots(1, 1, figsize=(6.2, 4.8))
    out_name = Path(out_path).name
    if collapsed and panel != "both":
        out_name = out_name.replace("_human_genai", f"_{panel}_human_genai")
    is_cosine_fig = out_name.startswith("03_soi_cosine_similarity")
    use_cosine_summary_title = is_cosine_fig and is_summary_panel
    if not use_cosine_summary_title:
        suptitle = f"Quantitative Accuracy — {ylabel}"
        fig.suptitle(
            suptitle,
            fontsize=15 if is_summary_panel else 13,
            fontweight="bold",
            y=1.01,
            x=0.5,
            ha="center",
        )

    if panel == "both":
        panel_defs = [(r_key, "Racial Inequality Task"), (g_key, "Gender Inequality Task")]
    elif panel == "race":
        panel_defs = [(r_key, "Racial Inequality Task")]
    else:
        panel_defs = [(g_key, "Gender Inequality Task")]

    panel_comparison_boxes: list[tuple[object, list[tuple[str, float]]]] = []

    for ridx, (key, title) in enumerate(panel_defs):
        e = stats_by_group(key, "1")
        n = stats_by_group(key, "0")
        g = stats_by_group(key, "2")
        h = stats_human(key)
        if collapsed:
            p_human_genai = welch_test(h, g, values_key="vals")
            comparisons = [
                (comparison_pair_label("Human", "GenAI"), p_human_genai),
            ]
        else:
            p_exp_phd = welch_test(e, n, values_key="vals")
            p_gen_exp = welch_test(g, e, values_key="vals")
            p_gen_phd = welch_test(g, n, values_key="vals")
            comparisons = [
                (comparison_pair_label("Experts", "PhD Students"), p_exp_phd),
                (comparison_pair_label("PhD Students", "GenAI"), p_gen_phd),
                (comparison_pair_label("Experts", "GenAI"), p_gen_exp),
            ]

        if show_distribution:
            ax = axes[ridx][0]
        else:
            ax = axes[ridx] if panel == "both" else axes
        use_ci = is_summary_panel
        if collapsed:
            groups = [f"Humans\n(n={h['n']})", f"GenAI\n(n={g['n']})"]
            means = [h["mean"], g["mean"]]
            if use_ci:
                errs = [bootstrap_ci_half_width(h["vals"]), bootstrap_ci_half_width(g["vals"])]
            else:
                errs = [h["std"], g["std"]]
            colors = [COLOR_AGG_HUMAN, GROUP_COLORS["genai"]]
        else:
            groups = [f"Experts\n(n={e['n']})", f"PhD Students\n(n={n['n']})", f"GenAI\n(n={g['n']})"]
            means = [e["mean"], n["mean"], g["mean"]]
            if use_ci:
                errs = [
                    bootstrap_ci_half_width(e["vals"]),
                    bootstrap_ci_half_width(n["vals"]),
                    bootstrap_ci_half_width(g["vals"]),
                ]
            else:
                errs = [e["std"], n["std"], g["std"]]
            colors = [GROUP_COLORS["expert"], GROUP_COLORS["phd"], GROUP_COLORS["genai"]]
        panel_ylim = (
            _collapsed_panel_ylim(means, errs, ylim)
            if collapsed and is_summary_panel
            else ylim
        )
        bars = ax.bar(groups, means, yerr=errs, capsize=5, color=colors, edgecolor="white", linewidth=0.8)
        for b, m, err in zip(bars, means, errs):
            y_text = m + (0 if np.isnan(err) else err) + (panel_ylim[1] - panel_ylim[0]) * 0.03
            ax.text(b.get_x() + b.get_width() / 2, y_text, f"{m:.3f}",
                    ha="center", va="bottom", fontsize=11 if is_summary_panel else 9)
        ax.set_ylim(panel_ylim)
        ax.set_ylabel(ylabel, fontsize=11.5 if is_summary_panel else 9.5)
        if is_summary_panel:
            ax.tick_params(axis="x", labelsize=11.5)
            ax.tick_params(axis="y", labelsize=11.5)
        spread_label = "95% CI" if use_ci else "SD"
        title_pad = 16 if (is_summary_panel and is_cosine_fig) else 6
        if use_cosine_summary_title:
            task_short = "Race Task" if key == r_key else "Gender Task"
            ax.set_title(
                f"{SOI_COSINE_SUMMARY_TITLE}\n{task_short}",
                fontsize=12,
                fontweight="bold",
                loc="center",
                pad=title_pad,
                linespacing=1.85,
            )
        else:
            ax.set_title(
                f"{title}\nMean ± {spread_label}",
                fontsize=12 if is_summary_panel else 10,
                fontweight="bold",
                loc="center",
                pad=title_pad,
            )
        ax.spines[["top", "right"]].set_visible(False)
        errbar_legend = (
            "Bars: group mean\nWhiskers: 95% CI"
            if use_ci else
            "Bars: group mean\nWhiskers: ±1 SD from mean"
        )
        if is_summary_panel:
            panel_comparison_boxes.append((ax, comparisons))
            lg = ax.legend(
                handles=[plt.Line2D([0], [0], color="none", label=errbar_legend)],
                loc="upper left",
                frameon=True,
                fontsize=11,
            )
            lg.get_frame().set_edgecolor("#666666")
            lg.get_frame().set_linewidth(0.8)
            lg.get_frame().set_alpha(1.0)
        else:
            lg_e = ax.legend(
                handles=[plt.Line2D([0], [0], color="none", label=errbar_legend)],
                loc="lower left",
                frameon=True,
                fontsize=11,
            )
            lg_e.get_frame().set_edgecolor("#666666")
            lg_e.get_frame().set_linewidth(0.8)
            lg_e.get_frame().set_alpha(1.0)

        if not show_distribution:
            continue
        ax2 = axes[ridx][1]
        np.random.seed(42)
        for y, vals, color in [(1, e["vals"], GROUP_COLORS["expert"]), (2, n["vals"], GROUP_COLORS["phd"]), (3, g["vals"], GROUP_COLORS["genai"])]:
            jit = np.random.uniform(-0.08, 0.08, len(vals))
            ax2.scatter(np.array(vals), np.full(len(vals), y) + jit, color=color, alpha=0.55, s=36)
            med = np.median(vals)
            mn = np.mean(vals)
            ax2.plot([med, med], [y - 0.22, y + 0.22], color=color, linewidth=2.4, solid_capstyle="round")
            ax2.scatter([mn], [y], color="white", edgecolors=color, s=52, linewidths=1.7, zorder=5)
        ax2.set_yticks([1, 2, 3])
        ax2.set_yticklabels(["Experts", "PhD Students", "GenAI"])
        ax2.set_xlim(ylim)
        ax2.set_xlabel(ylabel, fontsize=9.5)
        ax2.set_title("Distribution by Group\n(line = median, ○ = mean)", fontsize=10, fontweight="bold")
        ax2.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    if is_summary_panel:
        n_comp_lines = 1 if collapsed else 3
        if collapsed:
            comp_box_bottom = QUANT_COLLAPSED_COMPARISON_BOX_BOTTOM
            comp_axis_gap = QUANT_COLLAPSED_AXIS_GAP
        else:
            comp_box_bottom = QUANT_COMPARISON_BOX_BOTTOM
            comp_axis_gap = None
        apply_bottom_layout(
            fig,
            n_lines=n_comp_lines,
            box_bottom=comp_box_bottom,
            **({} if comp_axis_gap is None else {"axis_gap": comp_axis_gap}),
        )
        fig.canvas.draw()
        for ax, comparisons in panel_comparison_boxes:
            bbox = ax.get_position()
            panel_w = bbox.x1 - bbox.x0
            center_x = (bbox.x0 + bbox.x1) / 2
            draw_centered_comparison_box(
                fig,
                comparisons,
                center_x=center_x,
                box_bottom=comp_box_bottom,
                min_box_width=0.0,
                max_box_width=panel_w * QUANT_BOX_MAX_WIDTH_FRAC,
            )
        footnote_y = QUANT_COLLAPSED_FOOTNOTE_Y if collapsed else QUANT_FOOTNOTE_Y
        footnote_text = (
            SOI_WELCH_HUMAN_GENAI_FOOTNOTE
            if collapsed
            else SOI_WELCH_THREE_GROUP_FOOTNOTE
        )
        draw_sig_footnote(fig, y=footnote_y, text=footnote_text)
    out_path = Path(out_path)
    if panel == "both":
        final_path = out_path
    elif collapsed:
        stem = out_path.stem.replace("_human_genai", "")
        final_path = out_path.with_name(f"{stem}_{panel}_human_genai{out_path.suffix}")
    else:
        final_path = out_path.with_name(f"{out_path.stem}_{panel}{out_path.suffix}")
    save_figure(final_path)
    print(f"Figure saved → {final_path}")


def plot_sorted_cosine_individual_separate(ylim, panel="both"):
    if panel not in {"both", "race", "gender"}:
        raise ValueError(f"Invalid panel={panel!r}")
    all_panels = [
        ("cos_race", "vec_race_bin", ml_bin_r, "Race Task"),
        ("cos_gender", "vec_gender_bin", ml_bin_g, "Gender Task"),
    ]
    sorted_title = (
        "Second-Order Interactions Accuracy - "
        "(Cosine similarity, measuring both importance and sign)"
    )
    if panel == "both":
        panels = all_panels
        fig, axes = plt.subplots(1, 2, figsize=(20, 7.2))
    elif panel == "race":
        panels = [all_panels[0]]
        fig, axes = plt.subplots(1, 1, figsize=(10, 7.2))
    else:
        panels = [all_panels[1]]
        fig, axes = plt.subplots(1, 1, figsize=(10, 7.2))
    for idx, (task_key, vec_key, ml_vec, title) in enumerate(panels):
        ax = axes[idx] if panel == "both" else axes
        pts = [
            {"score": r[task_key], "group": r["group"], "vec": r[vec_key]}
            for r in records
            if not np.isnan(r[task_key])
        ]
        pts.sort(key=lambda x: x["score"], reverse=False)

        n_pt = len(pts)
        total_n = len(records)
        x = np.linspace(1.0, float(total_n), n_pt) if n_pt else np.array([])
        y = np.array([p["score"] for p in pts])
        is_exp = np.array([p["group"] == "1" for p in pts], dtype=bool)
        is_non = np.array([p["group"] == "0" for p in pts], dtype=bool)
        is_gen = np.array([p["group"] == "2" for p in pts], dtype=bool)

        _edg, _lw = "#2a2a2a", 0.55
        ax.scatter(x[is_exp], y[is_exp], c=GROUP_COLORS["expert"], marker="o", s=52,
                   alpha=1.0, edgecolors=_edg, linewidths=_lw, zorder=5)
        ax.scatter(x[is_non], y[is_non], c=GROUP_COLORS["phd"], marker="o", s=52,
                   alpha=1.0, edgecolors=_edg, linewidths=_lw, zorder=5)
        ax.scatter(x[is_gen], y[is_gen], c=GROUP_COLORS["genai"], marker="o", s=52,
                   alpha=1.0, edgecolors=_edg, linewidths=_lw, zorder=5)

        aggregations = _cosine_aggregation_scores(pts, ml_vec)
        random_bench_score = RANDOM_BENCHMARK_BY_TASK[task_key]

        _draw_horizontal_reference(ax, aggregations["human"], COLOR_AGG_HUMAN, "-", 1.5)
        _draw_horizontal_reference(ax, aggregations["genai"], GROUP_COLORS["genai"], "-", 1.5)
        _draw_horizontal_reference(ax, random_bench_score, COLOR_RANDOM_BENCH, "--", 1.5)
        _draw_horizontal_reference(ax, 1.0, SIGN_COLORS["aligned"], ":", 1.5, alpha=0.98, zorder=2)

        extra_legend = [
            _legend_line(
                COLOR_RANDOM_BENCH, "--",
                f"Random benchmark = {format_legend_value(random_bench_score)}",
            ),
            _legend_line(
                SIGN_COLORS["aligned"], ":",
                "ML benchmark = 1.000",
                linewidth=1.5,
            ),
        ]
        _build_sorted_figure_legend(ax, aggregations, extra_legend)

        ax.set_title(
            f"{sorted_title}\n{title}",
            fontsize=18,
            fontweight="bold",
            pad=16,
            linespacing=1.85,
            loc="center",
        )
        ax.set_xlabel("Respondent rank (sorted low → high by cosine similarity score)", fontsize=15.5)
        ax.set_ylabel("Cosine Similarity", fontsize=16)
        ax.set_xlim(0, total_n + 1)
        dynamic_ymin = (float(np.min(y)) - 0.1) if n_pt else ylim[0]
        if not np.isnan(random_bench_score):
            dynamic_ymin = min(dynamic_ymin, float(random_bench_score) - 0.1)
        ax.set_ylim(dynamic_ymin, ylim[1])
        step = max(1, (total_n + 7) // 8)
        rank_ticks = list(range(0, total_n + 1, step))
        if rank_ticks[-1] != total_n:
            rank_ticks.append(total_n)
        ax.set_xticks(rank_ticks)
        ax.set_xticklabels([str(r) for r in rank_ticks])
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", labelsize=10)
        ax.tick_params(axis="y", labelsize=15)
    plt.tight_layout()
    out_path = (
        OUT_DIR / "06_soi_cosine_sorted_individuals.png"
        if panel == "both"
        else OUT_DIR / f"06_soi_cosine_sorted_individuals_{panel}.png"
    )
    save_figure(out_path)
    print(f"Figure saved → {out_path}")


metric_defs = [
    ("cos_race", "cos_gender", "Cosine Similarity", (-0.4, 1.2), OUT_DIR / "03_soi_cosine_similarity.png", False),
]

for rk, gk, yl, lim, path, show_dist in metric_defs:
    _plot_metric_panels(rk, gk, yl, lim, path, show_dist, panel="race")
    _plot_metric_panels(rk, gk, yl, lim, path, show_dist, panel="gender")

_cos_collapsed_path = OUT_DIR / "03_soi_cosine_similarity_human_genai.png"
_plot_metric_panels(
    "cos_race", "cos_gender", "Cosine Similarity", (-0.4, 1.2),
    _cos_collapsed_path, False, panel="race", collapsed=True,
)
_plot_metric_panels(
    "cos_race", "cos_gender", "Cosine Similarity", (-0.4, 1.2),
    _cos_collapsed_path, False, panel="gender", collapsed=True,
)

race_sign_agg, race_tie, race_considered, race_total = aggregated_sign_align_majority_excluding_ties(
    r_pair_cols, r_sign_cols, ml_pairs_signs_r, HUMAN_GROUP_IDS
)
gender_sign_agg, gender_tie, gender_considered, gender_total = aggregated_sign_align_majority_excluding_ties(
    g_pair_cols, g_sign_cols, ml_pairs_signs_g, HUMAN_GROUP_IDS
)
print(
    f"\n[SOI] Aggregated sign alignment (excluding ties) — Race: "
    f"{race_sign_agg:.4f} | ties excluded = {race_tie}/{race_total} | considered = {race_considered}"
)
print(
    f"[SOI] Aggregated sign alignment (excluding ties) — Gender: "
    f"{gender_sign_agg:.4f} | ties excluded = {gender_tie}/{gender_total} | considered = {gender_considered}"
)

plot_sorted_cosine_individual_separate((-1.0, 1.05), panel="both")
plot_sorted_cosine_individual_separate((-1.0, 1.05), panel="race")
plot_sorted_cosine_individual_separate((-1.0, 1.05), panel="gender")

for rk, gk, name in [
    ("cos_race",        "cos_gender",        "Cosine Similarity (combining importance and sign)"),
    ("hit_race",        "hit_gender",        "Hit Rate \n percentage of Top-3 Important (ML) Interactions that were selected by each respondent"),
    ("sign_align_race", "sign_align_gender", "Sign Alignment Rate \n among selected Top-3 Important (ML) Interactions"),
]:
    for key, task in [(rk, "RACE"), (gk, "GENDER")]:
        e = stats_by_group(key, "1")
        n = stats_by_group(key, "0")
        g = stats_by_group(key, "2")
        a = stats_all(key)
        h = stats_human(key)
        print(f"\n── {task} — {name} ─────────────────────────────────────")
        print(f"{'Group':<12} {'N':>4} {'Mean':>7} {'Median':>8} {'SD':>7}")
        print("-" * 44)
        print(f"{'Expert':<12} {e['n']:>4} {e['mean']:>7.4f} {e['median']:>8.4f} {e['std']:>7.4f}")
        print(f"{'PhD Students':<12} {n['n']:>4} {n['mean']:>7.4f} {n['median']:>8.4f} {n['std']:>7.4f}")
        print(f"{'GenAI':<12} {g['n']:>4} {g['mean']:>7.4f} {g['median']:>8.4f} {g['std']:>7.4f}")
        print(f"{'All':<12} {a['n']:>4} {a['mean']:>7.4f} {a['median']:>8.4f} {a['std']:>7.4f}")
        for a_lbl, a_grp, b_lbl, b_grp in [
            ("Expert", e, "PhD Students", n),
            ("Expert", e, "GenAI", g),
            ("PhD Students", n, "GenAI", g),
            ("GenAI", g, "All Humans", h),
        ]:
            p = welch_test(a_grp, b_grp, values_key="vals")
            if not np.isnan(p):
                print(f"  Welch t-test ({a_lbl} vs {b_lbl}) = p = {p:.4f}")
