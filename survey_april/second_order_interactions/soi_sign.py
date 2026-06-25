"""
Second-Order Interactions (SOI) Analysis
=======================================
Sign Alignment Analysis (Q9)

Evaluate sign correctness for ML top-3 interactions.
Interaction is treated as an unordered pair.
"""

import csv
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
from viz_config import GROUP_COLORS, SIGN_COLORS

ML_PLUS_COLOR = "#1f77b4"
ML_MINUS_COLOR = "#d62728"

FEATURE_LABELS = {
    "social_science":                    "Social Science",
    "natural_science":                   "Natural Science",
    "engineering_and_technology":        "Engineering & Tech",
    "num_authors":                       "Num. Authors",
    "female":                            "Female",
    "asian":                             "Asian",
    "black":                             "Black",
    "hispanic_and_other":                "Hispanic & Other",
    "white":                             "White",
    "authors_race_diversity_score":      "Author Race Diversity",
    "country_race_diversity_score":      "Country Race Diversity",
    "news_inequality_mentions_3_years":  "\"Inequality\" Mentions in News (3yr)",
    "paper_inequality_mentions_3_years": "\"Inequality\" Mentions in Papers (3yr)",
}

plt.rcParams.update({
    "figure.dpi": 180,
    "savefig.dpi": 600,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 13.5,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "axes.linewidth": 0.9,
    "xtick.labelsize": 12.5,
    "ytick.labelsize": 12.5,
    "legend.fontsize": 14.5,
    "legend.frameon": False,
    "grid.alpha": 0.2,
    "lines.linewidth": 1.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

BASE = Path(__file__).parent
CSV_PATH = BASE.parent / "All_Participants_All_Questions.csv"
ML_PATH = BASE / "ML_results.json"
OUT_DIR = BASE / "figures"
OUT_DIR.mkdir(exist_ok=True)


def canon_pair(a, b):
    return tuple(sorted((a.strip(), b.strip())))


def parse_pair(cell):
    cell = cell.strip()
    if not cell or "," not in cell:
        return None
    parts = [x.strip() for x in cell.split(",")]
    if len(parts) != 2 or parts[0] == parts[1]:
        return None
    return canon_pair(parts[0], parts[1])


def format_interaction_label(pair, ml_sign):
    left = FEATURE_LABELS.get(pair[0], pair[0])
    right = FEATURE_LABELS.get(pair[1], pair[1])
    return f"{left}\n{right}"


with open(CSV_PATH, encoding="utf-8-sig", newline="") as f:
    rows = list(csv.reader(f))
headers = rows[0]
data = rows[1:]

with open(ML_PATH) as f:
    ml_raw = json.load(f)

ml_signs = {
    task: {canon_pair(e["feature_1"], e["feature_2"]): e["sign"] for e in entries}
    for task, entries in ml_raw.items()
}

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
group_col = next(i for i, h in enumerate(headers) if "expert_1" in h)


def sign_alignment(pair_cols, sign_cols, ml_map, group_filter=None):
    out = {p: {"n_selected": 0, "n_aligned": 0, "n_not_aligned": 0} for p in ml_map}
    for row in data:
        if group_filter is not None and row[group_col].strip() != group_filter:
            continue
        chosen = {}
        for pc, sc in zip(pair_cols, sign_cols):
            p = parse_pair(row[pc])
            if p is None:
                continue
            chosen[p] = row[sc].strip()

        for p, ml_sign in ml_map.items():
            if p not in chosen:
                continue
            out[p]["n_selected"] += 1
            if chosen[p] == ml_sign:
                out[p]["n_aligned"] += 1
            elif chosen[p] in ("+", "-"):
                out[p]["n_not_aligned"] += 1
    return out


def plot(acc_dict, ml_map, title, out_path):
    pairs = sorted(
        acc_dict.keys(),
        key=lambda p: acc_dict[p]["n_aligned"] / max(acc_dict[p]["n_selected"], 1),
        reverse=True,
    )
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    _draw_sign_alignment_panel(ax, pairs, acc_dict, ml_map)
    ax.set_title(title, fontsize=16, fontweight="bold", pad=10)
    plt.tight_layout()
    legend_kwargs = {"frameon": False, "fontsize": 15.5, "loc": "upper right"}
    if any(tag in str(out_path).lower() for tag in ("gender", "race")):
        fig.subplots_adjust(right=0.84)
        legend_kwargs.update({"loc": "center left", "bbox_to_anchor": (1.02, 0.55)})
    ax.legend(**legend_kwargs)
    plt.savefig(out_path, dpi=600, bbox_inches="tight")
    print(f"Figure saved → {out_path}")


def _draw_sign_alignment_panel(ax, pairs, acc_dict, ml_map):
    pairs = sorted(
        pairs,
        key=lambda p: acc_dict[p]["n_aligned"] / max(acc_dict[p]["n_selected"], 1),
        reverse=True,
    )
    labels = [format_interaction_label(p, ml_map[p]) for p in pairs]
    n_sel = [acc_dict[p]["n_selected"] for p in pairs]
    n_aligned = [acc_dict[p]["n_aligned"] for p in pairs]
    n_not_aligned = [acc_dict[p]["n_not_aligned"] for p in pairs]
    pct_aligned = [c / max(n, 1) * 100 for c, n in zip(n_aligned, n_sel)]

    y = np.arange(len(pairs))
    ax.barh(y, n_aligned, color=SIGN_COLORS["aligned"], alpha=0.98, edgecolor="white",
            linewidth=0.8, height=0.62, label="Aligned with LR")
    ax.barh(y, n_not_aligned, left=n_aligned, color=SIGN_COLORS["not_aligned"], alpha=0.98, edgecolor="white",
            linewidth=0.8, height=0.62, label="Not aligned with LR")

    pct_offset = max(n_sel) * 0.02 if n_sel else 0.4
    for i, (a, w, p) in enumerate(zip(n_aligned, n_not_aligned, pct_aligned)):
        ax.text(a + w + pct_offset, i, f"{p:.0f}%", va="center", ha="left", fontsize=11.8)
    # Keep LR sign labels clearly separated from bar ends.
    ml_offset = max(n_sel) * 0.32 if n_sel else 1.0
    for i, p in enumerate(pairs):
        ml_color = ML_PLUS_COLOR if ml_map[p] == "+" else ML_MINUS_COLOR
        ax.text(
            n_sel[i] + ml_offset,
            i,
            f"LR: {ml_map[p]}",
            va="center",
            ha="left",
            fontsize=13.0,
            color=ml_color,
            fontweight="bold",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=18.5, rotation=18, ha="right", rotation_mode="anchor")
    ax.invert_yaxis()
    ax.set_xlabel(
        "Number of respondents selecting this interaction. \nPercent indicate sign-alignment rate with LR among those respondents.",
        fontsize=15.5
    )
    xmax = max(75.0, float(max(n_sel)) * 1.5) if n_sel else 75.0
    ax.set_xlim(0, max(xmax, float(max(n_sel)) * 1.9 if n_sel else xmax))
    ax.spines[["top", "right"]].set_visible(False)


race_acc = sign_alignment(r_pair_cols, r_sign_cols, ml_signs["race"])
gend_acc = sign_alignment(g_pair_cols, g_sign_cols, ml_signs["gender"])

plot(
    race_acc,
    ml_signs["race"],
    "Sign Alignment between Respondents and LR for Top-3 Important (ML) Interactions \n Race",
    OUT_DIR / "02a_soi_sign_alignment_race.png",
)
plot(
    gend_acc,
    ml_signs["gender"],
    "Sign Alignment between Respondents and LR for Top-3 Important (ML) Interactions \n Gender",
    OUT_DIR / "02b_soi_sign_alignment_gender.png",
)

# Group-wise figures
def plot_by_group(pair_cols, sign_cols, ml_map, title, out_path):
    group_defs = [("1", "Experts", GROUP_COLORS["expert"]), ("0", "PhD Students", GROUP_COLORS["phd"]), ("2", "GenAI", GROUP_COLORS["genai"])]
    group_sizes = {
        gid: sum(1 for row in data if row[group_col].strip() == gid)
        for gid, _, _ in group_defs
    }
    acc_by_group = {
        gid: sign_alignment(pair_cols, sign_cols, ml_map, group_filter=gid)
        for gid, _, _ in group_defs
    }
    pairs = list(ml_map.keys())
    pairs = sorted(
        pairs,
        key=lambda p: np.nanmean([
            (acc_by_group[gid][p]["n_aligned"] / acc_by_group[gid][p]["n_selected"] * 100)
            if acc_by_group[gid][p]["n_selected"] else np.nan
            for gid, _, _ in group_defs
        ]),
        reverse=True,
    )

    y = np.arange(len(pairs))
    h = 0.23
    fig, ax = plt.subplots(figsize=(10.2, 5.1))
    for off, (gid, glabel, color) in zip([h, 0, -h], group_defs):
        n_aligned_pct = []
        n_not_aligned_pct = []
        pct_aligned = []
        denom = max(group_sizes.get(gid, 0), 1)
        for p in pairs:
            d = acc_by_group[gid][p]
            n_aligned_pct.append(d["n_aligned"] / denom * 100)
            n_not_aligned_pct.append(d["n_not_aligned"] / denom * 100)
            pct_aligned.append((d["n_aligned"] / d["n_selected"] * 100) if d["n_selected"] else np.nan)
        ax.barh(y + off, n_aligned_pct, height=h, color=color, alpha=0.98, edgecolor="white",
                linewidth=0.8, label=f"{glabel} aligned with LR")
        ax.barh(y + off, n_not_aligned_pct, left=n_aligned_pct, height=h, color=SIGN_COLORS["not_aligned"], alpha=0.98,
                edgecolor="white", linewidth=0.8, label="_nolegend_")
        pct_offset = max(
            [
                (acc_by_group[g][p]["n_selected"] / max(group_sizes.get(g, 0), 1) * 100)
                for g, _, _ in group_defs for p in pairs
            ],
            default=1.0,
        ) * 0.02
        for i, (r, w, p) in enumerate(zip(n_aligned_pct, n_not_aligned_pct, pct_aligned)):
            if np.isnan(p):
                continue
            # Special case: selected but 0 aligned -> full grey bar.
            # Add group tag so readers can identify which group this grey bar belongs to.
            if np.isclose(r, 0.0) and (r + w) > 0:
                pct_label = f"{p:.0f}% ({glabel})"
            else:
                pct_label = f"{p:.0f}%"
            ax.text(r + w + pct_offset, y[i] + off, pct_label,
                    va="center", ha="left", fontsize=11.0)

    ax.set_yticks(y)
    ax.set_yticklabels(
        [format_interaction_label(p, ml_map[p]) for p in pairs],
        fontsize=15.6,
        rotation=18,
        ha="right",
        rotation_mode="anchor",
    )
    ax.invert_yaxis()
    max_total_pct = max(
        acc_by_group[gid][p]["n_selected"] / max(group_sizes.get(gid, 0), 1) * 100
        for gid, _, _ in group_defs
        for p in pairs
    )
    # Put LR labels just to the right of per-row bars and vertically above the cluster.
    row_max_totals = [
        max(
            acc_by_group[gid][p]["n_selected"] / max(group_sizes.get(gid, 0), 1) * 100
            for gid, _, _ in group_defs
        )
        for p in pairs
    ]
    # Nudge LR sign labels slightly further right to avoid visual crowding.
    label_gap = max_total_pct * 0.18 if max_total_pct else 1.5
    lr_xs = []
    for i, p in enumerate(pairs):
        ml_color = ML_PLUS_COLOR if ml_map[p] == "+" else ML_MINUS_COLOR
        lr_x = row_max_totals[i] + pct_offset + label_gap
        if i == 0 and "race" in str(out_path).lower():
            lr_x = max(0.0, lr_x - 3.0)
        lr_xs.append(lr_x)
        lr_y = y[i] - h * 0.75
        if i == len(pairs) - 1:
            lr_y = y[i] + h * 0.15
        ax.text(
            lr_x,
            lr_y,
            f"LR: {ml_map[p]}",
            va="center",
            ha="left",
            fontsize=12.6,
            color=ml_color,
            fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.88, pad=0.6),
            clip_on=False,
        )
    xmax = max(
        120.0,
        max(lr_xs) + 8.0,
        max(row_max_totals) + pct_offset + 18.0,
    )
    ax.set_xlim(0, xmax)
    ax.set_xlabel(
        "Percentage of respondents selecting this interaction. \nPercent labels indicate sign-alignment rate with LR among those selecting it.",
        fontsize=13.5
    )
    ax.set_title(title, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    handles = [
        mpatches.Patch(color=group_defs[0][2], alpha=0.98, label="Aligned with LR by Experts"),
        mpatches.Patch(color=group_defs[1][2], alpha=0.98, label="Aligned with LR by PhD Students"),
        mpatches.Patch(color=group_defs[2][2], alpha=0.98, label="Aligned with LR by GenAI"),
        mpatches.Patch(color=SIGN_COLORS["not_aligned"], alpha=0.98, label="Not aligned with LR"),
    ]
    plt.tight_layout()
    legend_kwargs = {"handles": handles, "loc": "upper right", "frameon": False, "fontsize": 14}
    if "gender" in str(out_path).lower():
        fig.subplots_adjust(right=0.70)
        legend_kwargs.update({"loc": "center left", "bbox_to_anchor": (1.14, 0.55)})
    elif "race" in str(out_path).lower():
        fig.subplots_adjust(right=0.74)
        legend_kwargs.update({"loc": "center left", "bbox_to_anchor": (1.10, 0.55)})
    ax.legend(**legend_kwargs)
    plt.savefig(out_path, dpi=600, bbox_inches="tight")
    print(f"Figure saved → {out_path}")

plot_by_group(
    r_pair_cols,
    r_sign_cols,
    ml_signs["race"],
    "Sign Alignment (By Group) of ML-Identified Interactions — Race",
    OUT_DIR / "02c_soi_sign_alignment_by_group_race.png",
)
plot_by_group(
    g_pair_cols,
    g_sign_cols,
    ml_signs["gender"],
    "Sign Alignment (By Group) of ML-Identified Interactions — Gender",
    OUT_DIR / "02d_soi_sign_alignment_by_group_gender.png",
)

for task, acc, ml in [
    ("RACE", race_acc, ml_signs["race"]),
    ("GENDER", gend_acc, ml_signs["gender"]),
]:
    print(f"\n── {task} SOI Sign Alignment (ML top-3) ───────────────────────────")
    print(f"{'Interaction':<75} {'ML':>3} {'n_sel':>6} {'aligned':>8} {'not_aligned':>7} {'alignment%':>7}")
    print("-" * 115)
    for p in sorted(acc.keys(), key=lambda x: acc[x]["n_aligned"] / max(acc[x]["n_selected"], 1), reverse=True):
        d = acc[p]
        a = d["n_aligned"] / max(d["n_selected"], 1) * 100
        print(f"{p[0] + ' * ' + p[1]:<75} {ml[p]:>3} {d['n_selected']:>6} {d['n_aligned']:>8} {d['n_not_aligned']:>7} {a:>6.1f}%")
