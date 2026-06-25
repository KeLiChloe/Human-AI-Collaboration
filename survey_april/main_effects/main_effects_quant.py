"""
Main Effects Analysis
=====================
Quantitative Metrics (Q1 + Q2 + Q3 vs ML vector)

Two metrics per respondent per task:

1. Cosine Similarity (binary signed)
   Vector: +1 / −1 / 0  (selected with sign / not selected)
   Range: [−1, 1]

All metrics compared across Expert vs Non-Expert groups.
"""

import csv
import json
import re
import sys
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

MAIN_EFFECTS_WELCH_THREE_GROUP_FOOTNOTE = (
    "Brackets: two-sided Welch t-test on pairwise group mean ML feature-selection accuracy "
    "(PhD Students vs Experts, PhD Students vs GenAI, Experts vs GenAI).",
    SIG_LEVEL_LEGEND,
)
MAIN_EFFECTS_WELCH_HUMAN_GENAI_FOOTNOTE = (
    "Brackets: two-sided Welch t-test on mean ML feature-selection accuracy (Humans vs GenAI).",
    SIG_LEVEL_LEGEND,
)

# Layout tuned for 1×2 quant panels: snug comparison boxes, extra footnote clearance.
QUANT_COMPARISON_BOX_BOTTOM = 0.040
QUANT_THREE_GROUP_AXIS_GAP = 0.10
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

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE     = Path(__file__).parent
CSV_PATH = BASE.parent / "All_Participants_All_Questions.csv"
ML_PATH  = BASE / "ML_results.json"
OUT_DIR  = BASE / "figures"
OUT_DIR.mkdir(exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
with open(CSV_PATH, encoding="utf-8-sig", newline="") as f:
    rows = list(csv.reader(f))

headers = rows[0]
data    = rows[1:]

with open(ML_PATH) as f:
    _ml_raw = json.load(f)   # {"race": [{rank, feature, sign}, ...], "gender": [...]}

# Parse into lookup structures used across the file
# ml_signs[task]  : {feature: sign} — for binary vector & exact recovery
ml_signs  = {}
for task, entries in _ml_raw.items():
    sorted_entries = sorted(entries, key=lambda e: e["rank"])
    ml_signs[task]  = {e["feature"]: e["sign"] for e in sorted_entries}

# ── Feature list ──────────────────────────────────────────────────────────────
FEATURES = [
    re.sub(r"^Q Race\.2 \(rank\) - ", "", h)
    for h in headers
    if re.match(r"^Q Race\.2 \(rank\) - ", h)
]
FEAT_IDX = {f: i for i, f in enumerate(FEATURES)}
SIGN_MAP = {"+": 1, "-": -1}

# Pooled respondent subsets for aggregate metrics (not per-bar group means).
# Human = PhD Students (0) + Expert (1) combined; GenAI (2) is computed separately.
HUMAN_GROUP_IDS = frozenset({"0", "1"})
EXPERT_GROUP_IDS = frozenset({"1"})
PHD_GROUP_IDS = frozenset({"0"})
GENAI_GROUP_IDS = frozenset({"2"})

# ── Column index maps ─────────────────────────────────────────────────────────
expert_col = next(i for i, h in enumerate(headers) if "expert_1" in h)
r1_col     = next(i for i, h in enumerate(headers) if h.strip() == "Q Race.1")
g1_col     = next(i for i, h in enumerate(headers) if h.strip() == "Q Gender.1")

r3_cols = {re.sub(r"^Q Race\.3 \(sign\) - ",   "", h): i
           for i, h in enumerate(headers) if re.match(r"^Q Race\.3 \(sign\) - ",   h)}
g3_cols = {re.sub(r"^Q Gender\.3 \(sign\) - ", "", h): i
           for i, h in enumerate(headers) if re.match(r"^Q Gender\.3 \(sign\) - ", h)}

# ── Vector builders ───────────────────────────────────────────────────────────
def build_binary_vector(q1_col, q3_col_map, row):
    """±1 / 0 signed vector."""
    vec = np.zeros(len(FEATURES))
    cell = row[q1_col].strip()
    if not cell:
        return None
    for feat in cell.split(","):
        feat = feat.strip()
        if feat not in FEAT_IDX:
            continue
        sign_str = row[q3_col_map[feat]].strip() if feat in q3_col_map else ""
        vec[FEAT_IDX[feat]] = SIGN_MAP.get(sign_str, 0)
    return vec

def build_ml_binary_vector(signs_dict):
    """±1 / 0 ML vector from {feature: sign} dict."""
    vec = np.zeros(len(FEATURES))
    for feat, sign_str in signs_dict.items():
        if feat in FEAT_IDX:
            vec[FEAT_IDX[feat]] = SIGN_MAP.get(sign_str, 0)
    return vec

# ── Metric functions ──────────────────────────────────────────────────────────
def cosine_sim(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else np.nan


RANDOM_BENCHMARK_N = 1000


def random_benchmark_cosine(ml_vec, vec_dim, nonzero_count, n_random=RANDOM_BENCHMARK_N):
    """Monte Carlo estimate of E[cosine(random signed sparse vector, ML vector)].

    Uses the mean of per-draw cosines (not cosine of the averaged vector). The
    latter has near-zero norm after averaging and produces unstable angles.
    """
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


def compute_hit_rate(q1_col, ml_signs_dict, row):
    """Fraction of ML-important features that the respondent selected (sign-agnostic)."""
    cell = row[q1_col].strip()
    if not cell:
        return np.nan
    selected = {f.strip() for f in cell.split(",") if f.strip() in FEAT_IDX}
    ml_feats = set(ml_signs_dict.keys())
    return len(selected & ml_feats) / len(ml_feats) if ml_feats else np.nan

def compute_sign_align_rate(q1_col, q3_col_map, ml_signs_dict, row):
    """Among selected ML-important features, fraction with correct sign.
    Returns nan if the respondent selected no ML features at all."""
    cell = row[q1_col].strip()
    if not cell:
        return np.nan
    selected = [f.strip() for f in cell.split(",") if f.strip() in FEAT_IDX]
    overlapping = [f for f in selected if f in ml_signs_dict]
    if not overlapping:
        return np.nan
    aligned = sum(
        1 for f in overlapping
        if (row[q3_col_map[f]].strip() if f in q3_col_map else "") == ml_signs_dict[f]
    )
    return aligned / len(overlapping)


def aggregated_sign_align_majority_excluding_ties(q1_col, q3_col_map, ml_signs_dict, pooled_subset_ids):
    """Majority-sign vote per ML feature over rows in pooled_subset_ids; ties excluded.

    Typical use: HUMAN_GROUP_IDS (Expert+PhD pooled) or GENAI_GROUP_IDS alone.
    """
    vote_sums = {feat: 0 for feat in ml_signs_dict}
    for row in data:
        if row[expert_col].strip() not in pooled_subset_ids:
            continue
        cell = row[q1_col].strip()
        if not cell:
            continue
        for feat in cell.split(","):
            feat = feat.strip()
            if feat not in ml_signs_dict or feat not in FEAT_IDX:
                continue
            sign_str = row[q3_col_map[feat]].strip() if feat in q3_col_map else ""
            vote_sums[feat] += SIGN_MAP.get(sign_str, 0)

    aligned = 0
    considered = 0
    tie_count = 0
    for feat, sum_vote in vote_sums.items():
        if sum_vote == 0:
            tie_count += 1
            continue
        considered += 1
        majority_sign = 1 if sum_vote > 0 else -1
        if majority_sign == SIGN_MAP.get(ml_signs_dict[feat], 0):
            aligned += 1
    rate = (aligned / considered) if considered else np.nan
    return rate, tie_count, considered, len(vote_sums)

# ── Pre-build ML vectors ───────────────────────────────────────────────────────
ml_bin_race    = build_ml_binary_vector(ml_signs["race"])
ml_bin_gender  = build_ml_binary_vector(ml_signs["gender"])

RANDOM_BENCHMARK_BY_TASK = {
    "cos_race": random_benchmark_cosine(ml_bin_race, len(FEATURES), 5),
    "cos_gender": random_benchmark_cosine(ml_bin_gender, len(FEATURES), 5),
}

# ── Compute all metrics per respondent ────────────────────────────────────────
records = []

for row in data:
    gid = row[expert_col].strip()   # 0=non-expert, 1=expert, 2=GenAI

    # Binary signed vectors
    vr_bin = build_binary_vector(r1_col, r3_cols, row)
    vg_bin = build_binary_vector(g1_col, g3_cols, row)

    records.append({
        "group": gid,
        "vec_race_bin": vr_bin,
        "vec_gender_bin": vg_bin,
        "cos_race":         cosine_sim(vr_bin, ml_bin_race)   if vr_bin is not None else np.nan,
        "cos_gender":       cosine_sim(vg_bin, ml_bin_gender) if vg_bin is not None else np.nan,
        "hit_race":         compute_hit_rate(r1_col, ml_signs["race"],   row),
        "hit_gender":       compute_hit_rate(g1_col, ml_signs["gender"], row),
        "sign_align_race":  compute_sign_align_rate(r1_col, r3_cols, ml_signs["race"],   row),
        "sign_align_gender": compute_sign_align_rate(g1_col, g3_cols, ml_signs["gender"], row),
    })

# ── Summary stats helper ──────────────────────────────────────────────────────
def group_stats(key, group_val):
    vals = [r[key] for r in records
            if r["group"] == group_val and not np.isnan(r[key])]
    if not vals:
        return dict(n=0, mean=np.nan, median=np.nan, std=np.nan, sims=[])
    return dict(n=len(vals), mean=np.mean(vals),
                median=np.median(vals), std=np.std(vals), sims=vals)

def all_stats(key):
    vals = [r[key] for r in records if not np.isnan(r[key])]
    return dict(n=len(vals), mean=np.mean(vals),
                median=np.median(vals), std=np.std(vals), sims=vals)

def human_stats(key):
    vals = [r[key] for r in records
            if r["group"] in HUMAN_GROUP_IDS and not np.isnan(r[key])]
    return dict(n=len(vals), mean=np.mean(vals),
                median=np.median(vals), std=np.std(vals), sims=vals)

def genai_stats(key):
    vals = [r[key] for r in records
            if r["group"] in GENAI_GROUP_IDS and not np.isnan(r[key])]
    return dict(n=len(vals), mean=np.mean(vals),
                median=np.median(vals), std=np.std(vals), sims=vals)

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
        kwargs.update(marker=marker, markevery=markevery or [1], markersize=markersize,
                      markerfacecolor=color, markeredgecolor="#2a2a2a", markeredgewidth=0.55)
        return plt.Line2D([0, 0.5, 1], [0, 0, 0], **kwargs)
    return plt.Line2D([0], [0], **kwargs)


def _build_sorted_figure_legend(ax, aggregations, extra_lines, scatter_legend=True):
    """Legend entries include numeric values for aggregation reference lines."""
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

# ── Print summary tables ──────────────────────────────────────────────────────
METRICS = [
    ("cos_race",        "cos_gender",        "Cosine Similarity (binary)"),
    ("hit_race",        "hit_gender",        "Hit Rate"),
    ("sign_align_race", "sign_align_gender", "Sign Alignment Rate"),
]

for r_key, g_key, metric_label in METRICS:
    for task_key, task_label in [(r_key, "Racial Inequality"),
                                 (g_key, "Gender Inequality")]:
        e  = group_stats(task_key, "1")
        ne = group_stats(task_key, "0")
        ga = group_stats(task_key, "2")
        al = all_stats(task_key)
        hu = human_stats(task_key)
        print(f"\n── {task_label} — {metric_label} ──────────────────────────")
        print(f"{'Group':<15} {'N':>4} {'Mean':>7} {'Median':>8} {'SD':>7}")
        print("─" * 46)
        print(f"{'Expert':<15} {e['n']:>4} {e['mean']:>7.4f} {e['median']:>8.4f} {e['std']:>7.4f}")
        print(f"{'PhD Students':<15} {ne['n']:>4} {ne['mean']:>7.4f} {ne['median']:>8.4f} {ne['std']:>7.4f}")
        print(f"{'GenAI':<15} {ga['n']:>4} {ga['mean']:>7.4f} {ga['median']:>8.4f} {ga['std']:>7.4f}")
        print(f"{'All':<15} {al['n']:>4} {al['mean']:>7.4f} {al['median']:>8.4f} {al['std']:>7.4f}")
        for a_lbl, a_grp, b_lbl, b_grp in [
            ("Expert", e, "PhD Students", ne),
            ("Expert", e, "GenAI", ga),
            ("PhD Students", ne, "GenAI", ga),
            ("GenAI", ga, "All Humans", hu),
        ]:
            p = welch_test(a_grp, b_grp)
            if not np.isnan(p):
                print(f"  Welch t-test ({a_lbl} vs {b_lbl}) = p = {p:.4f}")

# ── Visualization ─────────────────────────────────────────────────────────────
COLOR_EXPERT    = GROUP_COLORS["expert"]
COLOR_NONEXPERT = GROUP_COLORS["phd"]
COLOR_RANDOM_BENCH = "#4A4A4A"

METRIC_CONFIGS = [
    # (race_key, gender_key, ylabel, ylim, file_suffix, show_distribution)
    ("cos_race", "cos_gender", "Cosine Similarity", (-0.05, 1.2), "03_cosine_similarity", False),
]

_SUPTITLE_MAP = {
    "03_cosine_similarity": "Cosine Similarity (combining importance and sign)",
    "03_cosine_similarity_human_genai": "Cosine Similarity (combining importance and sign)",
}
COSINE_SUMMARY_TITLE = (
    "Main Effects Accuracy (combining importance and sign) - Mean ± 95% CI"
)

def plot_metric(r_key, g_key, ylabel, ylim, file_suffix, show_distribution):
    return _plot_metric_panels(r_key, g_key, ylabel, ylim, file_suffix, show_distribution, panel="both")


def plot_metric_collapsed(r_key, g_key, ylabel, ylim, file_suffix, show_distribution=False):
    return _plot_metric_panels(
        r_key, g_key, ylabel, ylim, file_suffix, show_distribution, panel="both", collapsed=True
    )


def _plot_metric_panels(
    r_key,
    g_key,
    ylabel,
    ylim,
    file_suffix,
    show_distribution,
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
        fig, axes = plt.subplots(2, 2, figsize=(12, 9), gridspec_kw={"width_ratios": [1, 1.5]})
    else:
        if panel == "both":
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
        else:
            fig, axes = plt.subplots(1, 1, figsize=(6.2, 4.8))
    title_key = file_suffix
    if collapsed:
        title_key = (
            f"{file_suffix}_human_genai"
            if panel == "both"
            else f"{file_suffix}_{panel}_human_genai"
        )
    suptitle = _SUPTITLE_MAP.get(title_key, _SUPTITLE_MAP.get(file_suffix, f"Quantitative Accuracy — {ylabel}"))
    is_cosine_fig = file_suffix == "03_cosine_similarity"
    use_cosine_summary_title = is_cosine_fig and is_summary_panel
    if not use_cosine_summary_title:
        fig.suptitle(
            suptitle,
            fontsize=15 if is_summary_panel else 13,
            fontweight="bold",
            y=1.045 if is_cosine_fig else 1.01,
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

    for idx, (task_key, task_label) in enumerate(panel_defs):
        e  = group_stats(task_key, "1")
        ne = group_stats(task_key, "0")
        ga = group_stats(task_key, "2")
        hu = human_stats(task_key)
        if collapsed:
            p_human_genai = welch_test(hu, ga)
            comparisons = [
                (comparison_pair_label("Human", "GenAI"), p_human_genai),
            ]
        else:
            p_exp_phd = welch_test(e, ne)
            p_gen_exp = welch_test(ga, e)
            p_gen_phd = welch_test(ga, ne)
            comparisons = [
                (comparison_pair_label("Experts", "PhD Students"), p_exp_phd),
                (comparison_pair_label("PhD Students", "GenAI"), p_gen_phd),
                (comparison_pair_label("Experts", "GenAI"), p_gen_exp),
            ]

        # ── Left: bar chart mean ± SD (or mean ± 95% CI for panel 03) ───────
        if show_distribution:
            ax = axes[idx][0]
        else:
            ax = axes[idx] if panel == "both" else axes
        use_ci = is_summary_panel
        if collapsed:
            groups = [f"Humans\n(n={hu['n']})", f"GenAI\n(n={ga['n']})"]
            means = [hu["mean"], ga["mean"]]
            if use_ci:
                errs = [bootstrap_ci_half_width(hu["sims"]), bootstrap_ci_half_width(ga["sims"])]
            else:
                errs = [hu["std"], ga["std"]]
            colors = [COLOR_AGG_HUMAN, GROUP_COLORS["genai"]]
        else:
            groups = [f"Experts\n(n={e['n']})", f"PhD Students\n(n={ne['n']})", f"GenAI\n(n={ga['n']})"]
            means  = [e["mean"], ne["mean"], ga["mean"]]
            if use_ci:
                errs = [
                    bootstrap_ci_half_width(e["sims"]),
                    bootstrap_ci_half_width(ne["sims"]),
                    bootstrap_ci_half_width(ga["sims"]),
                ]
            else:
                errs = [e["std"], ne["std"], ga["std"]]
            colors = [COLOR_EXPERT, COLOR_NONEXPERT, GROUP_COLORS["genai"]]

        panel_ylim = (
            _collapsed_panel_ylim(means, errs, ylim)
            if collapsed and is_summary_panel
            else ylim
        )
        bars = ax.bar(groups, means, yerr=errs, color=colors, width=0.5,
                      capsize=5, error_kw={"linewidth": 1.2},
                      edgecolor="white", linewidth=0.8)
        for bar, m, err in zip(bars, means, errs):
            y_text = m + (0 if np.isnan(err) else err) + (panel_ylim[1] - panel_ylim[0]) * 0.03
            ax.text(bar.get_x() + bar.get_width() / 2,
                    y_text,
                    f"{m:.3f}", ha="center", va="bottom", fontsize=11 if is_summary_panel else 9)

        spread_label = "95% CI" if use_ci else "SD"
        title_pad = 16 if (is_summary_panel and is_cosine_fig) else 6
        if use_cosine_summary_title:
            task_short = "Race Task" if task_key == r_key else "Gender Task"
            ax.set_title(
                f"{COSINE_SUMMARY_TITLE}\n{task_short}",
                fontsize=12,
                fontweight="bold",
                loc="center",
                pad=title_pad,
                linespacing=1.85,
            )
        else:
            ax.set_title(
                f"{task_label}\nMean ± {spread_label}",
                fontsize=12 if is_summary_panel else 10,
                fontweight="bold",
                loc="center",
                pad=title_pad,
            )
        ax.set_ylabel(ylabel, fontsize=11.5 if is_summary_panel else 9.5)
        if is_summary_panel:
            ax.tick_params(axis="x", labelsize=11.5)
            ax.tick_params(axis="y", labelsize=11.5)
        ax.set_ylim(panel_ylim)
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
        # ── Right: strip plot + median line ──────────────────────────────────
        ax2 = axes[idx][1]
        np.random.seed(42)
        jitter = 0.08

        for y_pos, grp, color in [(1, e, COLOR_EXPERT), (2, ne, COLOR_NONEXPERT), (3, ga, GROUP_COLORS["genai"])]:
            sims = grp["sims"]
            jit  = np.random.uniform(-jitter, jitter, len(sims))
            ax2.scatter(np.array(sims), np.full(len(sims), y_pos) + jit,
                        color=color, alpha=0.5, s=38, zorder=3)
            med = np.median(sims)
            ax2.plot([med, med], [y_pos - 0.22, y_pos + 0.22],
                     color=color, linewidth=2.5, zorder=4, solid_capstyle="round")
            mn = np.mean(sims)
            ax2.scatter([mn], [y_pos], color="white", edgecolors=color,
                        s=55, zorder=5, linewidths=1.8)

        ax2.set_yticks([1, 2, 3])
        ax2.set_yticklabels(["Experts", "PhD Students", "GenAI"], fontsize=10.5)
        ax2.set_xlabel(ylabel, fontsize=9.5)
        ax2.set_title("Distribution by Group\n(line = median,  ○ = mean)",
                      fontsize=10, fontweight="bold")
        ax2.set_xlim(ylim[0], ylim[1])
        ax2.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    if is_summary_panel:
        n_comp_lines = 1 if collapsed else 3
        if collapsed:
            comp_box_bottom = QUANT_COLLAPSED_COMPARISON_BOX_BOTTOM
            comp_axis_gap = QUANT_COLLAPSED_AXIS_GAP
        else:
            comp_box_bottom = QUANT_COMPARISON_BOX_BOTTOM
            comp_axis_gap = QUANT_THREE_GROUP_AXIS_GAP
        apply_bottom_layout(
            fig,
            n_lines=n_comp_lines,
            box_bottom=comp_box_bottom,
            axis_gap=comp_axis_gap,
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
            MAIN_EFFECTS_WELCH_HUMAN_GENAI_FOOTNOTE
            if collapsed
            else MAIN_EFFECTS_WELCH_THREE_GROUP_FOOTNOTE
        )
        draw_sig_footnote(fig, y=footnote_y, text=footnote_text)
    if collapsed:
        if panel == "both":
            out_path = OUT_DIR / f"{file_suffix}_human_genai.png"
        else:
            out_path = OUT_DIR / f"{file_suffix}_{panel}_human_genai.png"
    else:
        out_path = OUT_DIR / (f"{file_suffix}.png" if panel == "both" else f"{file_suffix}_{panel}.png")
    save_figure(out_path)
    print(f"Figure saved → {out_path}")


def plot_sorted_cosine_individual_separate(panel="both"):
    if panel not in {"both", "race", "gender"}:
        raise ValueError(f"Invalid panel={panel!r}")
    all_panels = [
        ("cos_race", "vec_race_bin", ml_bin_race, "Race Task"),
        ("cos_gender", "vec_gender_bin", ml_bin_gender, "Gender Task"),
    ]
    sorted_title = (
        "Main Effects Accuracy - (Cosine similarity, measuring both importance and sign)"
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
            if not np.isnan(r[task_key]) and r[vec_key] is not None
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
        ax.scatter(x[is_exp], y[is_exp], c=COLOR_EXPERT, marker="o", s=52,
                   alpha=1.0, edgecolors=_edg, linewidths=_lw, zorder=5)
        ax.scatter(x[is_non], y[is_non], c=COLOR_NONEXPERT, marker="o", s=52,
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
        lower_candidates = [float(np.min(y))] if n_pt else [-0.1]
        if not np.isnan(random_bench_score):
            lower_candidates.append(float(random_bench_score))
        dynamic_ymin = min(lower_candidates) - 0.1
        ax.set_ylim(dynamic_ymin, 1.1)
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
        OUT_DIR / "06_cosine_sorted_individuals.png"
        if panel == "both"
        else OUT_DIR / f"06_cosine_sorted_individuals_{panel}.png"
    )
    save_figure(out_path)
    print(f"Figure saved → {out_path}")

for r_key, g_key, ylabel, ylim, suffix, show_dist in METRIC_CONFIGS:
    _plot_metric_panels(r_key, g_key, ylabel, ylim, suffix, show_dist, panel="race")
    _plot_metric_panels(r_key, g_key, ylabel, ylim, suffix, show_dist, panel="gender")

_plot_metric_panels(
    "cos_race", "cos_gender", "Cosine Similarity", (-0.05, 1.2),
    "03_cosine_similarity", False, panel="race", collapsed=True,
)
_plot_metric_panels(
    "cos_race", "cos_gender", "Cosine Similarity", (-0.05, 1.2),
    "03_cosine_similarity", False, panel="gender", collapsed=True,
)

race_sign_agg, race_tie, race_considered, race_total = aggregated_sign_align_majority_excluding_ties(
    r1_col, r3_cols, ml_signs["race"], HUMAN_GROUP_IDS
)
gender_sign_agg, gender_tie, gender_considered, gender_total = aggregated_sign_align_majority_excluding_ties(
    g1_col, g3_cols, ml_signs["gender"], HUMAN_GROUP_IDS
)
print(
    f"\n[Main Effects] Aggregated sign alignment (excluding ties) — Race: "
    f"{race_sign_agg:.4f} | ties excluded = {race_tie}/{race_total} | considered = {race_considered}"
)
print(
    f"[Main Effects] Aggregated sign alignment (excluding ties) — Gender: "
    f"{gender_sign_agg:.4f} | ties excluded = {gender_tie}/{gender_total} | considered = {gender_considered}"
)

plot_sorted_cosine_individual_separate(panel="both")
plot_sorted_cosine_individual_separate(panel="race")
plot_sorted_cosine_individual_separate(panel="gender")
