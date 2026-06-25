"""
Main Effects Analysis
=====================
Sign alignment analysis (Q3)

For each feature that a respondent selected (Q1), they also indicated
a sign (Q3: + or −). We compare their sign to the ML-estimated sign
and compute sign alignment.

Only features that are in the ML top-5 are evaluated, because only
those have a ground-truth ML sign to compare against.

Metrics per feature (restricted to respondents who selected it):
  - n_selected   : number of respondents who selected this feature
  - n_aligned    : number who gave the sign aligned with LR
  - alignment (%) : n_aligned / n_selected
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

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE     = Path(__file__).parent
CSV_PATH = "All_Participants_All_Questions.csv"
ML_PATH  = "main_effects/ML_results.json"
OUT_DIR  = BASE / "figures"
OUT_DIR.mkdir(exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
with open(CSV_PATH, encoding="utf-8-sig", newline="") as f:
    rows = list(csv.reader(f))

headers = rows[0]
data    = rows[1:]

with open(ML_PATH) as f:
    _ml_raw = json.load(f)   # {"race": [{rank, feature, sign}, ...], "gender": [...]}

# Parse into {task: {feature: sign}} for sign alignment lookups
ml_results = {
    task: {e["feature"]: e["sign"] for e in entries}
    for task, entries in _ml_raw.items()
}

# ── Feature list & labels ─────────────────────────────────────────────────────
FEATURES = [
    re.sub(r"^Q Race\.2 \(rank\) - ", "", h)
    for h in headers
    if re.match(r"^Q Race\.2 \(rank\) - ", h)
]

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

# ── Column index helpers ──────────────────────────────────────────────────────
def find_col(pattern):
    return next(i for i, h in enumerate(headers) if re.match(pattern, h))

def find_col_exact(name):
    return next(i for i, h in enumerate(headers) if h.strip() == name)

# Q1 (selected features), Q3 (sign per feature) — Race & Gender
r1_col  = find_col_exact("Q Race.1")
g1_col  = find_col_exact("Q Gender.1")

r3_cols = {
    re.sub(r"^Q Race\.3 \(sign\) - ", "", h): i
    for i, h in enumerate(headers)
    if re.match(r"^Q Race\.3 \(sign\) - ", h)
}
g3_cols = {
    re.sub(r"^Q Gender\.3 \(sign\) - ", "", h): i
    for i, h in enumerate(headers)
    if re.match(r"^Q Gender\.3 \(sign\) - ", h)
}
group_col = next(i for i, h in enumerate(headers) if "expert_1" in h)

# ── Compute sign alignment ─────────────────────────────────────────────────────
def sign_alignment(q1_col, q3_col_map, ml_signs, group_filter=None):
    """
    For each ML top-5 feature, among respondents who selected it,
    count how many gave the correct sign.

    Returns dict: feat -> {"n_selected": int, "n_aligned": int,
                           "n_not_aligned": int, "n_missing": int}
    """
    results = {f: {"n_selected": 0, "n_aligned": 0,
                   "n_not_aligned": 0, "n_missing": 0}
               for f in ml_signs}

    for row in data:
        if group_filter is not None and row[group_col].strip() != group_filter:
            continue
        selected_cell = row[q1_col].strip()
        if not selected_cell:
            continue
        selected = {f.strip() for f in selected_cell.split(",")}

        for feat, ml_sign in ml_signs.items():
            if feat not in selected:
                continue
            results[feat]["n_selected"] += 1
            human_sign = row[q3_col_map[feat]].strip() if feat in q3_col_map else ""
            if human_sign == ml_sign:
                results[feat]["n_aligned"] += 1
            elif human_sign in ("+", "-"):
                results[feat]["n_not_aligned"] += 1
            else:
                results[feat]["n_missing"] += 1

    return results

race_sign_acc   = sign_alignment(r1_col, r3_cols, ml_results["race"])
gender_sign_acc = sign_alignment(g1_col, g3_cols, ml_results["gender"])

# ── Plot helper ───────────────────────────────────────────────────────────────
COLOR_ALIGNED = SIGN_COLORS["aligned"]
COLOR_NOT_ALIGNED = SIGN_COLORS["not_aligned"]
ML_PLUS_COLOR = "#1f77b4"
ML_MINUS_COLOR = "#d62728"


def _draw_sign_alignment_panel(ax, acc_dict, ml_signs, task_label):
    feats = list(acc_dict.keys())
    feats = sorted(
        feats,
        key=lambda f: acc_dict[f]["n_aligned"] / max(acc_dict[f]["n_selected"], 1),
        reverse=True,
    )

    labels = [FEATURE_LABELS[f] for f in feats]
    n_sel = [acc_dict[f]["n_selected"] for f in feats]
    n_aligned = [acc_dict[f]["n_aligned"] for f in feats]
    n_not_aligned = [acc_dict[f]["n_not_aligned"] for f in feats]
    alignment = [c / max(s, 1) * 100 for c, s in zip(n_aligned, n_sel)]

    y_pos = np.arange(len(feats))
    ax.barh(
        y_pos,
        n_aligned,
        color=COLOR_ALIGNED,
        alpha=0.98,
        edgecolor="white",
        linewidth=0.8,
        height=0.62,
        label="Aligned with LR",
    )
    ax.barh(
        y_pos,
        n_not_aligned,
        left=n_aligned,
        color=COLOR_NOT_ALIGNED,
        alpha=0.98,
        edgecolor="white",
        linewidth=0.8,
        height=0.62,
        label="Not aligned with LR",
    )

    pct_offset = max(n_sel) * 0.02 if n_sel else 0.4
    for i, (a, w, p) in enumerate(zip(n_aligned, n_not_aligned, alignment)):
        ax.text(a + w + pct_offset, y_pos[i], f"{p:.0f}%", va="center", ha="left", fontsize=11.8)

    ml_offset = max(n_sel) * 0.20 if n_sel else 1.0
    for i, f in enumerate(feats):
        ml_color = ML_PLUS_COLOR if ml_signs[f] == "+" else ML_MINUS_COLOR
        ax.text(
            n_sel[i] + ml_offset,
            y_pos[i],
            f"LR: {ml_signs[f]}",
            va="center",
            ha="left",
            fontsize=12.8,
            color=ml_color,
            fontweight="bold",
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=13.5, rotation=18, ha="right", rotation_mode="anchor")
    ax.invert_yaxis()
    ax.set_xlabel(
        "Number of respondents who selected this feature. \nPercent labels indicate sign-alignment rate with LR among those respondents.",
        fontsize=13.5,
    )
    ax.set_title(task_label, fontsize=14, fontweight="bold", pad=10)
    ax.set_xlim(0, max(n_sel) * 1.9 if n_sel else 1.0)
    ax.spines[["top", "right"]].set_visible(False)

def plot_sign_alignment(acc_dict, ml_signs, task_label, out_path):
    fig, ax = plt.subplots(figsize=(8.5, max(3.5, len(acc_dict) * 0.75 + 1.5)))
    _draw_sign_alignment_panel(ax, acc_dict, ml_signs, task_label)
    ax.set_title(
        f"Sign Alignment between Respondents and LR for Top 5 Important (ML) Features \n {task_label}",
        fontsize=14, fontweight="bold", pad=10
    )
    ax.legend(loc="lower right", fontsize=14.5, frameon=False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=600, bbox_inches="tight")
    print(f"Figure saved → {out_path}")


# ── Run plots ─────────────────────────────────────────────────────────────────
plot_sign_alignment(
    race_sign_acc, ml_results["race"],
    "Race",
    OUT_DIR / "02a_sign_alignment_race.png",
)

plot_sign_alignment(
    gender_sign_acc, ml_results["gender"],
    "Gender",
    OUT_DIR / "02b_sign_alignment_gender.png",
)

# ── Group-wise sign alignment plots ────────────────────────────────────────────
def plot_sign_alignment_by_group(acc_by_group, ml_signs, task_label, out_path):
    group_order = [
        ("1", "Expert", GROUP_COLORS["expert"]),
        ("0", "PhD Students", GROUP_COLORS["phd"]),
        ("2", "GenAI", GROUP_COLORS["genai"]),
    ]
    group_sizes = {
        gid: sum(1 for row in data if row[group_col].strip() == gid)
        for gid, *_ in group_order
    }
    feats = list(ml_signs.keys())
    feats = sorted(
        feats,
        key=lambda f: np.nanmean([
            (acc_by_group[gid][f]["n_aligned"] / acc_by_group[gid][f]["n_selected"] * 100.0)
            if acc_by_group[gid][f]["n_selected"] else np.nan
            for gid, *_ in group_order
        ]),
        reverse=True
    )

    y = np.arange(len(feats))
    h = 0.23
    fig, ax = plt.subplots(figsize=(10, max(3.8, len(feats) * 0.8 + 1.3)))

    for offset, (gid, glabel, color) in zip([h, 0, -h], group_order):
        denom = max(group_sizes.get(gid, 0), 1)
        n_aligned = [acc_by_group[gid][f]["n_aligned"] / denom * 100.0 for f in feats]
        n_not_aligned = [acc_by_group[gid][f]["n_not_aligned"] / denom * 100.0 for f in feats]
        pct_aligned = [
            (acc_by_group[gid][f]["n_aligned"] / acc_by_group[gid][f]["n_selected"] * 100.0)
            if acc_by_group[gid][f]["n_selected"] else np.nan
            for f in feats
        ]
        ax.barh(y + offset, n_aligned, height=h, color=color, alpha=0.98, edgecolor="white",
                linewidth=0.8, label=f"{glabel} aligned with LR")
        ax.barh(y + offset, n_not_aligned, left=n_aligned, height=h, color=COLOR_NOT_ALIGNED,
                alpha=0.98, edgecolor="white", linewidth=0.8, label="_nolegend_")
        pct_offset = max(
            [
                (acc_by_group[g][f]["n_selected"] / max(group_sizes.get(g, 0), 1) * 100.0)
                for g, *_ in group_order for f in feats
            ],
            default=1.0,
        ) * 0.02
        for i, (r, w, p) in enumerate(zip(n_aligned, n_not_aligned, pct_aligned)):
            if np.isnan(p):
                continue
            ax.text(r + w + pct_offset, y[i] + offset, f"{p:.0f}%",
                    va="center", ha="left", fontsize=11.2)

    labels = [FEATURE_LABELS[f] for f in feats]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=13, rotation=18, ha="right", rotation_mode="anchor")
    ax.invert_yaxis()
    max_total = max(
        acc_by_group[gid][f]["n_selected"] / max(group_sizes.get(gid, 0), 1) * 100.0
        for gid, *_ in group_order
        for f in feats
    )
    ml_offset = max_total * 0.12 if max_total else 0.8
    for i, f in enumerate(feats):
        feat_max = max(
            acc_by_group[gid][f]["n_selected"] / max(group_sizes.get(gid, 0), 1) * 100.0
            for gid, *_ in group_order
        )
        ml_color = ML_PLUS_COLOR if ml_signs[f] == "+" else ML_MINUS_COLOR
        lr_x = min(feat_max + ml_offset, 118.0)
        lr_y = y[i]
        if ml_signs[f] == "-":
            lr_y = y[i] - h * 0.45
        ax.text(
            lr_x,
            lr_y,
            f"LR: {ml_signs[f]}",
            va="center",
            ha="left",
            fontsize=12.2,
            color=ml_color,
            fontweight="bold",
        )
    ax.set_xlim(0, 120)
    ax.set_xlabel(
        "Percentage of respondents selecting this feature. \nPercent labels indicate sign-alignment rate with LR among those selecting it.",
        fontsize=13.5
    )
    ax.set_title(
        f"Sign Alignment between Respondents and LR for Top 5 Important (ML) Features \n {task_label}",
        fontsize=14, fontweight="bold"
    )
    ax.spines[["top", "right"]].set_visible(False)
    handles = [
        mpatches.Patch(color=group_order[0][2], alpha=0.98, label="Aligned with LR by Experts"),
        mpatches.Patch(color=group_order[1][2], alpha=0.98, label="Aligned with LR by PhD Students"),
        mpatches.Patch(color=group_order[2][2], alpha=0.98, label="Aligned with LR by GenAI"),
        mpatches.Patch(color=COLOR_NOT_ALIGNED, alpha=0.98, label="Not Aligned"),
    ]
    ax.legend(
        handles=handles,
        loc="best",
        fontsize=14,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=600, bbox_inches="tight")
    print(f"Figure saved → {out_path}")


race_sign_acc_by_group = {
    "1": sign_alignment(r1_col, r3_cols, ml_results["race"], group_filter="1"),
    "0": sign_alignment(r1_col, r3_cols, ml_results["race"], group_filter="0"),
    "2": sign_alignment(r1_col, r3_cols, ml_results["race"], group_filter="2"),
}
gender_sign_acc_by_group = {
    "1": sign_alignment(g1_col, g3_cols, ml_results["gender"], group_filter="1"),
    "0": sign_alignment(g1_col, g3_cols, ml_results["gender"], group_filter="0"),
    "2": sign_alignment(g1_col, g3_cols, ml_results["gender"], group_filter="2"),
}

plot_sign_alignment_by_group(
    race_sign_acc_by_group, ml_results["race"],
    "Race",
    OUT_DIR / "02c_sign_alignment_by_group_race.png",
)
plot_sign_alignment_by_group(
    gender_sign_acc_by_group, ml_results["gender"],
    "Gender",
    OUT_DIR / "02d_sign_alignment_by_group_gender.png",
)

# ── Print summary tables ──────────────────────────────────────────────────────
for task, acc_dict, ml_signs in [
    ("RACE",   race_sign_acc,   ml_results["race"]),
    ("GENDER", gender_sign_acc, ml_results["gender"]),
]:
    print(f"\n── {task} — Sign Alignment (ML top-5 features) ────────────────────")
    print(f"{'Feature':<35} {'ML':>4} {'n_sel':>6} {'aligned':>8} {'not_aligned':>7} {'alignment%':>7}")
    print("─" * 70)
    for f in sorted(acc_dict, key=lambda x: acc_dict[x]["n_aligned"] / max(acc_dict[x]["n_selected"], 1), reverse=True):
        d = acc_dict[f]
        acc = d["n_aligned"] / max(d["n_selected"], 1) * 100
        print(f"{FEATURE_LABELS[f]:<35} {ml_signs[f]:>4} "
              f"{d['n_selected']:>6} {d['n_aligned']:>8} "
              f"{d['n_not_aligned']:>7}  {acc:>5.1f}%")
    print("─" * 70)
    overall_sel = sum(d["n_selected"] for d in acc_dict.values())
    overall_aligned = sum(d["n_aligned"]  for d in acc_dict.values())
    print(f"{'Overall (across features)':<35} {'':>4} "
            f"{overall_sel:>6} {overall_aligned:>8} {'':>7} {'':>8}  "
          f"{overall_aligned/max(overall_sel,1)*100:>5.1f}%")
