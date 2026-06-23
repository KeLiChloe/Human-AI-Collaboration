"""
Second-Order Interactions (SOI) Analysis
=======================================
Descriptive Analysis — Interaction Selection Frequency (Q6-Q8)

Each respondent selects top-3 second-order interactions.
Interaction is treated as an unordered pair.
"""

import csv
import json
import re
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
from viz_config import GROUP_COLORS

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

BASE = Path(__file__).parent
CSV_PATH = BASE.parent / "All_Participants_All_Questions.csv"
ML_PATH = BASE / "ML_results.json"
OUT_DIR = BASE / "figures"
OUT_DIR.mkdir(exist_ok=True)


def canon_pair(a, b):
    return tuple(sorted((a.strip(), b.strip())))


def parse_pair(cell, valid_features):
    cell = cell.strip()
    if not cell or "," not in cell:
        return None
    parts = [x.strip() for x in cell.split(",")]
    if len(parts) != 2:
        return None
    if parts[0] not in valid_features or parts[1] not in valid_features:
        return None
    if parts[0] == parts[1]:
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

pairs = list(combinations(sorted(features), 2))  # 78 unordered pairs
pair_label = {
    p: f"{FEATURE_LABELS.get(p[0], p[0])} * {FEATURE_LABELS.get(p[1], p[1])}"
    for p in pairs
}

ml_pairs = {
    task: {canon_pair(e["feature_1"], e["feature_2"]) for e in entries}
    for task, entries in ml_raw.items()
}

expert_col = next(i for i, h in enumerate(headers) if "expert_1" in h)
r_cols = [
    next(i for i, h in enumerate(headers) if h.strip() == "Q Race.6 (SOI, 1st)"),
    next(i for i, h in enumerate(headers) if h.strip() == "Q Race.7 (SOI, 2nd)"),
    next(i for i, h in enumerate(headers) if h.strip() == "Q Race.8 (SOI, 3rd)"),
]
g_cols = [
    next(i for i, h in enumerate(headers) if h.strip() == "Q Gender.6 (SOI, 1st)"),
    next(i for i, h in enumerate(headers) if h.strip() == "Q Gender.7 (SOI, 2nd)"),
    next(i for i, h in enumerate(headers) if h.strip() == "Q Gender.8 (SOI, 3rd)"),
]


def count_soi(cols, group_filter=None):
    c = Counter({p: 0 for p in pairs})
    n = 0
    for row in data:
        gid = row[expert_col].strip()
        if group_filter == "expert" and gid != "1":
            continue
        if group_filter == "non_expert" and gid != "0":
            continue
        if group_filter == "genai" and gid != "2":
            continue
        if group_filter == "human" and gid not in {"0", "1"}:
            continue
        n += 1
        for col in cols:
            p = parse_pair(row[col], feature_set)
            if p is not None:
                c[p] += 1
    return c, n


def plot_overall(
    counts,
    n,
    ml_set,
    title,
    out_path,
    top_n=10,
    subgroup_label="All Respondents",
    mark_unselected_ml_ticks=False,
):
    ranked = sorted(pairs, key=lambda p: counts[p], reverse=True)[:top_n]
    if mark_unselected_ml_ticks:
        missing_ml = [p for p in ml_set if p not in ranked]
        if missing_ml:
            non_ml_ranked = [p for p in ranked if p not in ml_set]
            keep_non_ml = max(0, top_n - len(ml_set))
            ranked = non_ml_ranked[:keep_non_ml] + [p for p in ranked if p in ml_set] + missing_ml

    labels = []
    for p in ranked:
        lbl = pair_label[p]
        add_ml_mark = mark_unselected_ml_ticks and (p in ml_set) and (counts[p] == 0)
        labels.append(f"{lbl} (ML)" if add_ml_mark else lbl)
    vals = [counts[p] for p in ranked]
    colors = ["#2E7D32" if p in ml_set else "#9E9E9E" for p in ranked]

    y = np.arange(len(ranked))
    fig, ax = plt.subplots(figsize=(9.2, 7.2))
    bar_h = 0.48
    bars = ax.barh(y, vals, color=colors, height=bar_h, edgecolor="white", linewidth=0.6)
    for b, v in zip(bars, vals):
        ax.text(v + 0.2, b.get_y() + b.get_height() / 2, f"{v} ({v / n * 100:.0f}%)",
                va="center", ha="left", fontsize=12)

    ax.set_yticks(y)    
    ax.set_yticklabels(labels, fontsize=15.5, rotation=22, ha="right", rotation_mode="anchor")
    for t in ax.get_yticklabels():
        if "(ML)" in t.get_text():
            t.set_color("#C62828")
    ax.invert_yaxis()
    ax.set_xlabel("Number of respondents selecting interaction", fontsize=14)
    ax.set_title(
        f"{title}\n(Top-3 interactions selected by respondents, {subgroup_label}, N = {n})",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlim(0, max(vals) * 1.3 if vals else 1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=12.5)
    ax.legend(
        handles=[
            mpatches.Patch(color="#2E7D32", label="In ML top-3"),
            mpatches.Patch(color="#9E9E9E", label="Not in ML top-3"),
        ],
        loc="lower right",
        frameon=False,
        fontsize=14.5,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=600, bbox_inches="tight")
    print(f"Figure saved → {out_path}")


def plot_group_comparison(exp_counts, n_exp, non_counts, n_non, gen_counts, n_gen, ml_set, title, out_path, top_n=10):
    ranked = sorted(
        pairs,
        key=lambda p: (exp_counts[p] / max(n_exp, 1) + non_counts[p] / max(n_non, 1) + gen_counts[p] / max(n_gen, 1)) / 3,
        reverse=True,
    )[:top_n]
    labels = [pair_label[p] + (" (ML)" if p in ml_set else "") for p in ranked]
    exp_rate = [exp_counts[p] / max(n_exp, 1) * 100 for p in ranked]
    non_rate = [non_counts[p] / max(n_non, 1) * 100 for p in ranked]
    gen_rate = [gen_counts[p] / max(n_gen, 1) * 100 for p in ranked]

    y = np.arange(len(ranked))
    h = 0.25
    fig, ax = plt.subplots(figsize=(10.2, 7.2))
    ax.barh(y + h, exp_rate, height=h, color=GROUP_COLORS["expert"], label=f"Experts (n={n_exp})")
    ax.barh(y, non_rate, height=h, color=GROUP_COLORS["phd"], label=f"PhD Students (n={n_non})")
    ax.barh(y - h, gen_rate, height=h, color=GROUP_COLORS["genai"], label=f"GenAI (n={n_gen})")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11.3, rotation=22, ha="right", rotation_mode="anchor")
    for t in ax.get_yticklabels():
        if "(ML)" in t.get_text():
            t.set_color("#C62828")
    ax.invert_yaxis()
    ax.set_xlabel("Percentage of participants selecting interaction (%)")
    ax.set_xlim(0, max(exp_rate + non_rate + gen_rate) * 1.2 if (exp_rate + non_rate + gen_rate) else 1)
    ax.set_title(f"{title}\nTop {top_n} by average selection rate across groups", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=11.5, loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=600, bbox_inches="tight")
    print(f"Figure saved → {out_path}")


race_counts, n_race = count_soi(r_cols)
gender_counts, n_gender = count_soi(g_cols)
race_exp, n_race_exp = count_soi(r_cols, group_filter="expert")
race_non, n_race_non = count_soi(r_cols, group_filter="non_expert")
race_gen, n_race_gen = count_soi(r_cols, group_filter="genai")
race_human, n_race_human = count_soi(r_cols, group_filter="human")
gend_exp, n_gend_exp = count_soi(g_cols, group_filter="expert")
gend_non, n_gend_non = count_soi(g_cols, group_filter="non_expert")
gend_gen, n_gend_gen = count_soi(g_cols, group_filter="genai")
gend_human, n_gend_human = count_soi(g_cols, group_filter="human")

print(f"Respondents — Race: all={n_race}, expert={n_race_exp}, non-expert={n_race_non}, genai={n_race_gen}")
print(f"Respondents — Gender: all={n_gender}, expert={n_gend_exp}, non-expert={n_gend_non}, genai={n_gend_gen}")

plot_overall(
    race_counts, n_race, ml_pairs["race"], "SOI Selection Frequency — Race",
    OUT_DIR / "01a_soi_selection_frequency_race.png",
    top_n=8,
    subgroup_label="All Respondents",
)
plot_overall(
    race_human, n_race_human, ml_pairs["race"], "SOI Selection Frequency — Race",
    OUT_DIR / "01a_soi_selection_frequency_race_humans.png",
    top_n=8,
    subgroup_label="Humans",
)
plot_overall(
    race_gen, n_race_gen, ml_pairs["race"], "SOI Selection Frequency — Race",
    OUT_DIR / "01a_soi_selection_frequency_race_genai.png",
    top_n=8,
    subgroup_label="GenAI",
    mark_unselected_ml_ticks=True,
)
plot_overall(
    gender_counts, n_gender, ml_pairs["gender"], "SOI Selection Frequency — Gender",
    OUT_DIR / "01b_soi_selection_frequency_gender.png",
    top_n=8,
    subgroup_label="All Respondents",
)
plot_overall(
    gend_human, n_gend_human, ml_pairs["gender"], "SOI Selection Frequency — Gender",
    OUT_DIR / "01b_soi_selection_frequency_gender_humans.png",
    top_n=8,
    subgroup_label="Humans",
)
plot_overall(
    gend_gen, n_gend_gen, ml_pairs["gender"], "SOI Selection Frequency — Gender",
    OUT_DIR / "01b_soi_selection_frequency_gender_genai.png",
    top_n=8,
    subgroup_label="GenAI",
    mark_unselected_ml_ticks=True,
)
plot_group_comparison(race_exp, n_race_exp, race_non, n_race_non, race_gen, n_race_gen, ml_pairs["race"],
                      "SOI: By Group — Race",
                      OUT_DIR / "01c_soi_expert_vs_nonexpert_race.png",
                      top_n=6)
plot_group_comparison(gend_exp, n_gend_exp, gend_non, n_gend_non, gend_gen, n_gend_gen, ml_pairs["gender"],
                      "SOI: By Group — Gender",
                      OUT_DIR / "01d_soi_expert_vs_nonexpert_gender.png",
                      top_n=6)

for task_name, counts, n, ml_set in [
    ("RACE", race_counts, n_race, ml_pairs["race"]),
    ("GENDER", gender_counts, n_gender, ml_pairs["gender"]),
]:
    print(f"\n── {task_name} SOI Top-15 ─────────────────────────────────────────")
    ranked = sorted(pairs, key=lambda p: counts[p], reverse=True)[:15]
    for i, p in enumerate(ranked, start=1):
        mark = "(ML)" if p in ml_set else " "
        print(f"{i:>2}. {pair_label[p]:<75} {counts[p]:>3} ({counts[p] / n * 100:>5.1f}%) {mark}")
