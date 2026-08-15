#!/usr/bin/env python3
"""Rebuild blinded theory pool from All_Participants_All_Questions.csv.

4 cells (task × effect): race/main, race/soi, gender/main, gender/soi.
Each cell pools pre-ML and post-ML (LLM-refined) theories for blind sampling.
Each item includes the respondent's feature selections + signs for that cell.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "All_Participants_All_Questions.csv"
OUT_PATH = Path(__file__).resolve().parent / "data" / "theories.json"

GROUP_COL = "student_0, senior_1, genAI_2"
NAME_COL = "What is your full name?"
CELLS = [
    # task_key, effect_key, topic, effect_label, phase, csv_column
    ("race", "main", "Racial Inequality", "Main Effects", "pre", "Q Race.4 pre-ML theory (main effects)"),
    ("race", "main", "Racial Inequality", "Main Effects", "post", "Q Race.12 LLM_refined post-ML theory (main effects)"),
    ("race", "soi", "Racial Inequality", "Interactions", "pre", "Q Race.10 pre-ML theory (SOI)"),
    ("race", "soi", "Racial Inequality", "Interactions", "post", "Q Race.15 LLM_refined post-ML theory (SOI)"),
    ("gender", "main", "Gender Inequality", "Main Effects", "pre", "Q Gender.4 pre-ML theory (main effects)"),
    ("gender", "main", "Gender Inequality", "Main Effects", "post", "Q Gender.12 LLM_refined post-ML theory (main effects)"),
    ("gender", "soi", "Gender Inequality", "Interactions", "pre", "Q Gender.10 pre-ML theory (SOI)"),
    ("gender", "soi", "Gender Inequality", "Interactions", "post", "Q Gender.15 LLM_refined post-ML theory (SOI)"),
]


def _parse_rank(raw: str | None) -> int | None:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s or s.lower() == "nan":
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def _parse_sign(raw: str | None) -> str:
    s = (raw or "").strip()
    if s in {"+", "-", "±", "+/-"}:
        return s
    if s.lower() in {"plus", "positive"}:
        return "+"
    if s.lower() in {"minus", "negative"}:
        return "-"
    return s


def extract_main_selections(row: dict[str, str], task: str) -> list[dict]:
    """Ranked main-effect features + signs from Q*.1 / Q*.2 / Q*.3."""
    prefix = "Q Race" if task == "race" else "Q Gender"
    selected = (row.get(f"{prefix}.1") or "").strip()
    if not selected:
        return []
    feats = [f.strip() for f in selected.split(",") if f.strip()]
    out: list[dict] = []
    for feat in feats:
        rank = _parse_rank(row.get(f"{prefix}.2 (rank) - {feat}"))
        sign = _parse_sign(row.get(f"{prefix}.3 (sign) - {feat}"))
        out.append(
            {
                "feature": feat,
                "rank": rank,
                "sign": sign,
            }
        )
    out.sort(key=lambda x: (x["rank"] is None, x["rank"] if x["rank"] is not None else 999, x["feature"]))
    return out


def extract_soi_selections(row: dict[str, str], task: str) -> list[dict]:
    """Top-3 second-order interactions + signs from Q*.6–9."""
    prefix = "Q Race" if task == "race" else "Q Gender"
    slots = (
        (f"{prefix}.6 (SOI, 1st)", f"{prefix}.9 (SOI, sign, 1st)", 1),
        (f"{prefix}.7 (SOI, 2nd)", f"{prefix}.9 (SOI, sign, 2nd)", 2),
        (f"{prefix}.8 (SOI, 3rd)", f"{prefix}.9 (SOI, sign, 3rd)", 3),
    )
    out: list[dict] = []
    for feat_col, sign_col, rank in slots:
        raw = (row.get(feat_col) or "").strip()
        if not raw:
            continue
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        label = " × ".join(parts) if len(parts) >= 2 else raw
        out.append(
            {
                "feature": label,
                "features": parts if parts else [raw],
                "rank": rank,
                "sign": _parse_sign(row.get(sign_col)),
            }
        )
    return out


def extract_selections(row: dict[str, str], task: str, effect: str) -> list[dict]:
    if effect == "soi":
        return extract_soi_selections(row, task)
    return extract_main_selections(row, task)


def main() -> None:
    items = []
    missing = Counter()
    with CSV_PATH.open(encoding="utf-8-sig", newline="") as f:
        for i, row in enumerate(csv.DictReader(f)):
            group = (row.get(GROUP_COL) or "").strip()
            name = (row.get(NAME_COL) or "").strip()
            source = "genai" if group == "2" else "human"
            for task_key, effect_key, topic, effect_label, phase, col in CELLS:
                text = (row.get(col) or "").strip()
                if not text:
                    missing[(task_key, effect_key, phase)] += 1
                    continue
                raw = f"{task_key}|{effect_key}|{phase}|{i}|{group}|{name}|{text[:80]}"
                blind_id = hashlib.sha256(raw.encode()).hexdigest()[:12]
                items.append(
                    {
                        "id": blind_id,
                        "text": text,
                        "topic": topic,
                        "task": task_key,
                        "effect": effect_key,
                        "effect_label": effect_label,
                        "phase": phase,
                        "source": source,
                        "group": group,
                        "participant_name": name,
                        "row_index": i,
                        "selections": extract_selections(row, task_key, effect_key),
                    }
                )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(items)} theories → {OUT_PATH}")
    print("by source:", dict(Counter(x["source"] for x in items)))
    print(
        "by cell:",
        dict(Counter((x["task"], x["effect"]) for x in items)),
    )
    print(
        "by phase:",
        dict(Counter(x["phase"] for x in items)),
    )
    n_with_sel = sum(1 for x in items if x["selections"])
    print(f"with selections: {n_with_sel}/{len(items)}")
    if missing:
        print("empty cells (row counts):", dict(missing))


if __name__ == "__main__":
    main()
