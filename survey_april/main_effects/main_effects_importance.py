"""
Main Effects Analysis
=====================
Descriptive Analysis — Feature Selection Frequency (Q1)

Count how many times each feature is selected into the top 5
by respondents. Race and Gender tasks are analysed separately.

Figures produced:
  01a / 01b  — overall frequency (all respondents)
  01a_* / 01b_* — overall frequency (Humans only, GenAI only)
  01c / 01d  — by-group selection rate comparison
"""

import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
from viz_config import GROUP_COLORS

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

ml_top5 = {task: {e["feature"] for e in entries}
           for task, entries in _ml_raw.items()}

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

# ── Column indices ────────────────────────────────────────────────────────────
expert_col = next(i for i, h in enumerate(headers) if "expert_1" in h)
r1_col     = next(i for i, h in enumerate(headers) if h.strip() == "Q Race.1")
g1_col     = next(i for i, h in enumerate(headers) if h.strip() == "Q Gender.1")

# ── Count selection frequency ─────────────────────────────────────────────────
def count_selections(col_idx, group_filter=None):
    """
    group_filter: None | 'expert' | 'non_expert' | 'genai' | 'human'
    """
    counter = Counter({f: 0 for f in FEATURES})
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
        cell = row[col_idx].strip()
        if not cell:
            continue
        n += 1
        for feat in cell.split(","):
            feat = feat.strip()
            if feat in counter:
                counter[feat] += 1
    return counter, n

# All / Expert / PhD Students / GenAI counts
race_counts,     n_race     = count_selections(r1_col)
gender_counts,   n_gender   = count_selections(g1_col)
race_exp,        n_race_exp = count_selections(r1_col, group_filter="expert")
race_non,        n_race_non = count_selections(r1_col, group_filter="non_expert")
race_gen,        n_race_gen = count_selections(r1_col, group_filter="genai")
race_human,      n_race_human = count_selections(r1_col, group_filter="human")
gender_exp,      n_gend_exp = count_selections(g1_col, group_filter="expert")
gender_non,      n_gend_non = count_selections(g1_col, group_filter="non_expert")
gender_gen,      n_gend_gen = count_selections(g1_col, group_filter="genai")
gender_human,    n_gend_human = count_selections(g1_col, group_filter="human")

print(f"Respondents — Race:   all={n_race}, expert={n_race_exp}, non-expert={n_race_non}, genai={n_race_gen}")
print(f"Respondents — Gender: all={n_gender}, expert={n_gend_exp}, non-expert={n_gend_non}, genai={n_gend_gen}")

# ── Colors ────────────────────────────────────────────────────────────────────
COLOR_DEFAULT    = "#9E9E9E"
COLOR_ML         = "#2E7D32"
COLOR_EXPERT     = GROUP_COLORS["expert"]
COLOR_NONEXPERT  = GROUP_COLORS["phd"]
COLOR_GENAI      = GROUP_COLORS["genai"]

# ── Figure 1: Overall frequency ───────────────────────────────────────────────
def plot_overall(
    counts,
    ml_top5_set,
    task_label,
    n_resp,
    out_path,
    subgroup_label="All Respondents",
    mark_unselected_ml_ticks=False,
):
    sorted_feats = sorted(FEATURES, key=lambda f: counts[f], reverse=True)
    labels  = [FEATURE_LABELS[f] for f in sorted_feats]
    values  = [counts[f] for f in sorted_feats]
    colors  = [COLOR_ML if f in ml_top5_set else COLOR_DEFAULT
               for f in sorted_feats]

    fig, ax = plt.subplots(figsize=(8, 6))
    y_pos = np.arange(len(sorted_feats))
    bars  = ax.barh(y_pos, values, color=colors, edgecolor="white",
                    linewidth=0.6, height=0.65)

    for bar, v in zip(bars, values):
        pct = v / n_resp * 100
        ax.text(bar.get_width() + 0.5,
                bar.get_y() + bar.get_height() / 2,
                f"{v}  ({pct:.0f}%)", va="center", ha="left", fontsize=9)

    tick_labels = []
    for f in sorted_feats:
        lbl = FEATURE_LABELS[f]
        add_ml_mark = mark_unselected_ml_ticks and (f in ml_top5_set) and (counts[f] == 0)
        tick_labels.append(f"{lbl} (ML)" if add_ml_mark else lbl)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(tick_labels, fontsize=10.5)
    for t in ax.get_yticklabels():
        if "(ML)" in t.get_text():
            t.set_color("#C62828")
    ax.invert_yaxis()
    ax.set_xlabel("Number of respondents selecting feature (percentage)", fontsize=10)
    ax.set_title(
        f"Feature Selection Frequency — {task_label}\n"
        f"(Top-5 features selected by respondents, {subgroup_label}, N = {n_resp})",
        fontsize=11, fontweight="bold", pad=12
    )
    ax.set_xlim(0, n_resp + n_resp * 0.25)
    ax.axvline(n_resp, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=9)

    ax2 = ax.twiny()
    ax2.set_xlim(0, (n_resp + n_resp * 0.25) / n_resp * 100)
    ax2.set_xlabel("Selection rate (%)", fontsize=9, color="grey")
    ax2.tick_params(axis="x", labelsize=8.5, colors="grey")
    ax2.spines[["right", "left", "bottom"]].set_visible(False)

    legend_handles = [
        mpatches.Patch(color=COLOR_ML,      label="In ML top-5"),
        mpatches.Patch(color=COLOR_DEFAULT, label="Not in ML top-5"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=12, frameon=False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=600, bbox_inches="tight")
    print(f"Figure saved → {out_path}")

# ── Figure 2: Expert vs PhD Students distribution ─────────────────────────────
def plot_group_comparison(exp_counts, n_exp, non_counts, n_non, gen_counts, n_gen,
                          ml_top5_set, task_label, out_path):
    # Sort features by average selection rate across 3 groups.
    sorted_feats = sorted(
        FEATURES,
        key=lambda f: (exp_counts[f] / max(n_exp, 1) +
                       non_counts[f] / max(n_non, 1) +
                       gen_counts[f] / max(n_gen, 1)) / 3,
        reverse=True
    )

    labels   = [FEATURE_LABELS[f] for f in sorted_feats]
    exp_pct  = [exp_counts[f]  / n_exp  * 100 for f in sorted_feats]
    non_pct  = [non_counts[f]  / n_non  * 100 for f in sorted_feats]
    gen_pct  = [gen_counts[f]  / n_gen  * 100 for f in sorted_feats]

    n_feats = len(sorted_feats)
    y_pos   = np.arange(n_feats)
    height  = 0.25

    fig, ax = plt.subplots(figsize=(9, 6.5))

    bars_e = ax.barh(y_pos + height, exp_pct, height=height,
                     color=COLOR_EXPERT, label=f"Experts (n={n_exp})",
                     edgecolor="white", linewidth=0.5)
    bars_n = ax.barh(y_pos, non_pct, height=height,
                     color=COLOR_NONEXPERT, label=f"PhD Students (n={n_non})",
                     edgecolor="white", linewidth=0.5)
    bars_g = ax.barh(y_pos - height, gen_pct, height=height,
                     color=COLOR_GENAI, label=f"GenAI (n={n_gen})",
                     edgecolor="white", linewidth=0.5)

    # Mark ML top-5 directly on y-ticks.
    tick_labels = []
    for f in sorted_feats:
        lbl = FEATURE_LABELS[f]
        tick_labels.append(f"{lbl} (ML)" if f in ml_top5_set else lbl)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(tick_labels, fontsize=12)
    for t in ax.get_yticklabels():
        if "(ML)" in t.get_text():
            t.set_color("#C62828")
    ax.invert_yaxis()
    ax.set_xlabel("Percentage of participants selecting feature (%)", fontsize=10)
    ax.set_title(
        f"Feature Selection Rate (By Group) — {task_label}",
        fontsize=11, fontweight="bold", pad=12
    )
    ax.set_xlim(0, max(exp_pct + non_pct + gen_pct + [1]) * 1.2)
    ax.spines[["top", "right"]].set_visible(False)

    legend_handles = [
        mpatches.Patch(color=COLOR_EXPERT,    label=f"Experts (n={n_exp})"),
        mpatches.Patch(color=COLOR_NONEXPERT, label=f"PhD Students (n={n_non})"),
        mpatches.Patch(color=COLOR_GENAI,     label=f"GenAI (n={n_gen})"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=12, frameon=False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=600, bbox_inches="tight")
    print(f"Figure saved → {out_path}")

# ── Run all plots ─────────────────────────────────────────────────────────────
plot_overall(race_counts,   ml_top5["race"],
             "Race",  n_race,
             OUT_DIR / "01a_feature_selection_frequency_race.png",
             subgroup_label="All Respondents")
plot_overall(race_human,    ml_top5["race"],
             "Race",  n_race_human,
             OUT_DIR / "01a_feature_selection_frequency_race_humans.png",
             subgroup_label="Humans")
plot_overall(race_gen,      ml_top5["race"],
             "Race",  n_race_gen,
             OUT_DIR / "01a_feature_selection_frequency_race_genai.png",
             subgroup_label="GenAI",
             mark_unselected_ml_ticks=True)

plot_overall(gender_counts, ml_top5["gender"],
             "Gender", n_gender,
             OUT_DIR / "01b_feature_selection_frequency_gender.png",
             subgroup_label="All Respondents")
plot_overall(gender_human,  ml_top5["gender"],
             "Gender", n_gend_human,
             OUT_DIR / "01b_feature_selection_frequency_gender_humans.png",
             subgroup_label="Humans")
plot_overall(gender_gen,    ml_top5["gender"],
             "Gender", n_gend_gen,
             OUT_DIR / "01b_feature_selection_frequency_gender_genai.png",
             subgroup_label="GenAI",
             mark_unselected_ml_ticks=True)

plot_group_comparison(race_exp, n_race_exp, race_non, n_race_non, race_gen, n_race_gen,
                      ml_top5["race"], "Race",
                      OUT_DIR / "01c_expert_vs_nonexpert_race.png")

plot_group_comparison(gender_exp, n_gend_exp, gender_non, n_gend_non, gender_gen, n_gend_gen,
                      ml_top5["gender"], "Gender",
                      OUT_DIR / "01d_expert_vs_nonexpert_gender.png")

# ── Print summary tables ──────────────────────────────────────────────────────
for task, counts, n, exp_c, n_exp, non_c, n_non, gen_c, n_gen, ml_key in [
    ("RACE",   race_counts,   n_race,   race_exp, n_race_exp, race_non, n_race_non, race_gen, n_race_gen, "race"),
    ("GENDER", gender_counts, n_gender, gender_exp, n_gend_exp, gender_non, n_gend_non, gender_gen, n_gend_gen, "gender"),
]:
    print(f"\n── {task} — Feature Selection Frequency ───────────────────────────────────────")
    print(f"{'Rank':<5} {'Feature':<35} {'All':>5} {'All%':>6}  {'Exp':>4} {'Exp%':>6}  {'PhD':>4} {'PhD%':>6}  {'Gen':>4} {'Gen%':>6}  ML")
    print("─" * 80)
    for rank, f in enumerate(
        sorted(FEATURES, key=lambda x: counts[x], reverse=True), 1
    ):
        ml_mark = "(ML)" if f in ml_top5[ml_key] else " "
        print(f"{rank:<5} {FEATURE_LABELS[f]:<35} "
              f"{counts[f]:>5}  {counts[f]/n*100:>4.0f}%  "
              f"{exp_c[f]:>4}  {exp_c[f]/n_exp*100:>4.0f}%  "
              f"{non_c[f]:>4}  {non_c[f]/n_non*100:>4.0f}%  "
              f"{gen_c[f]:>4}  {gen_c[f]/n_gen*100:>4.0f}%  {ml_mark}")
    print("─" * 80)
    print("  (ML) = also in ML top-5")
