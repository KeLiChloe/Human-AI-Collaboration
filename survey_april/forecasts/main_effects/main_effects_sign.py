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

ROOT = Path(__file__).resolve().parent.parent.parent  # survey_april/
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent
CSV_PATH = ROOT / "All_Participants_All_Questions.csv"
ML_PATH = BASE / "ML_results.json"

# ── Load data ─────────────────────────────────────────────────────────────────
with open(CSV_PATH, encoding="utf-8-sig", newline="") as f:
    rows = list(csv.reader(f))

headers = rows[0]
data = rows[1:]

with open(ML_PATH) as f:
    _ml_raw = json.load(f)  # {"race": [{rank, feature, sign}, ...], "gender": [...]}

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
    "social_science": "Social Science",
    "natural_science": "Natural Science",
    "engineering_and_technology": "Engineering & Tech",
    "num_authors": "Num. Authors",
    "female": "Female",
    "asian": "Asian",
    "black": "Black",
    "hispanic_and_other": "Hispanic & Other",
    "white": "White",
    "authors_race_diversity_score": "Author Race Diversity",
    "country_race_diversity_score": "Country Race Diversity",
    "news_inequality_mentions_3_years": '"Inequality" Mentions in News (3yr)',
    "paper_inequality_mentions_3_years": '"Inequality" Mentions in Papers (3yr)',
}

# ── Column index helpers ──────────────────────────────────────────────────────
def find_col_exact(name):
    return next(i for i, h in enumerate(headers) if h.strip() == name)

# Q1 (selected features), Q3 (sign per feature) — Race & Gender
r1_col = find_col_exact("Q Race.1")
g1_col = find_col_exact("Q Gender.1")

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
group_col = next(i for i, h in enumerate(headers) if "senior_1" in h)


# ── Compute sign alignment ─────────────────────────────────────────────────────
def sign_alignment(q1_col, q3_col_map, ml_signs, group_filter=None):
    """
    For each ML top-5 feature, among respondents who selected it,
    count how many gave the correct sign.

    Returns dict: feat -> {"n_selected": int, "n_aligned": int,
                           "n_not_aligned": int, "n_missing": int}
    """
    results = {
        f: {"n_selected": 0, "n_aligned": 0, "n_not_aligned": 0, "n_missing": 0}
        for f in ml_signs
    }

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


race_sign_acc = sign_alignment(r1_col, r3_cols, ml_results["race"])
gender_sign_acc = sign_alignment(g1_col, g3_cols, ml_results["gender"])

# ── Print summary tables ──────────────────────────────────────────────────────
for task, acc_dict, ml_signs in [
    ("RACE", race_sign_acc, ml_results["race"]),
    ("GENDER", gender_sign_acc, ml_results["gender"]),
]:
    print(f"\n── {task} — Sign Alignment (ML top-5 features) ────────────────────")
    print(f"{'Feature':<35} {'ML':>4} {'n_sel':>6} {'aligned':>8} {'not_aligned':>7} {'alignment%':>7}")
    print("─" * 70)
    for f in sorted(
        acc_dict,
        key=lambda x: acc_dict[x]["n_aligned"] / max(acc_dict[x]["n_selected"], 1),
        reverse=True,
    ):
        d = acc_dict[f]
        acc = d["n_aligned"] / max(d["n_selected"], 1) * 100
        print(
            f"{FEATURE_LABELS[f]:<35} {ml_signs[f]:>4} "
            f"{d['n_selected']:>6} {d['n_aligned']:>8} "
            f"{d['n_not_aligned']:>7}  {acc:>5.1f}%"
        )
    print("─" * 70)
    overall_sel = sum(d["n_selected"] for d in acc_dict.values())
    overall_aligned = sum(d["n_aligned"] for d in acc_dict.values())
    print(
        f"{'Overall (across features)':<35} {'':>4} "
        f"{overall_sel:>6} {overall_aligned:>8} {'':>7} {'':>8}  "
        f"{overall_aligned / max(overall_sel, 1) * 100:>5.1f}%"
    )
