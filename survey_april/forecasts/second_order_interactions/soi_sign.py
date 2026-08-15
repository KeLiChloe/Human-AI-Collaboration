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

ROOT = Path(__file__).resolve().parent.parent.parent  # survey_april/
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

BASE = Path(__file__).parent
CSV_PATH = ROOT / "All_Participants_All_Questions.csv"
ML_PATH = BASE / "ML_results.json"


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
group_col = next(i for i, h in enumerate(headers) if "senior_1" in h)


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


race_acc = sign_alignment(r_pair_cols, r_sign_cols, ml_signs["race"])
gend_acc = sign_alignment(g_pair_cols, g_sign_cols, ml_signs["gender"])

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
        print(
            f"{p[0] + ' * ' + p[1]:<75} {ml[p]:>3} {d['n_selected']:>6} "
            f"{d['n_aligned']:>8} {d['n_not_aligned']:>7} {a:>6.1f}%"
        )
