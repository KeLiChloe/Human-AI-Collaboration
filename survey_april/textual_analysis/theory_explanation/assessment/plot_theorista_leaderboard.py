"""Theorista leaderboard: quality + cosine similarity with ML Evidence (all participants)."""

from __future__ import annotations

import csv
import json
import re
import sys
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ASSESSMENT_DIR = Path(__file__).resolve().parent
TEXTUAL_DIR = ASSESSMENT_DIR.parent.parent
ROOT = TEXTUAL_DIR.parent
for p in (ASSESSMENT_DIR, TEXTUAL_DIR, ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from viz_style import apply_plot_style

apply_plot_style()

CSV_PATH = ROOT / "All_Participants_All_Questions.csv"
OUT_DIR = ASSESSMENT_DIR / "theorista_leaderboard"

PANEL_TITLE_SIZE = 26
FIG_SUPTITLE_SIZE = 30
FIG_XLABEL_SIZE = 14

THEORISTA = [
    "Theorista_GPT5.5_instant",
    "Theorista_GPT5.5_thinking_standard",
]
# thinking = red; instant = blue; all others grey
THEORISTA_COLORS = {
    "Theorista_GPT5.5_thinking_standard": "#E45756",
    "Theorista_GPT5.5_instant": "#4C78A8",
}
DEFAULT_BAR_COLOR = "#D8D8D8"
GROUP_LABEL = {"0": "PhD", "1": "Expert", "2": "GenAI"}

QUALITY_DIMS = {
    "race_pre": [
        "Q Race.4 Clarity and Coherence",
        "Q Race.4 Causal Reasoning",
        "Q Race.4 Theoretical Depth",
        "Q Race.4 Creativity",
        "Q Race.4 Persuasiveness",
    ],
    "race_post": [
        "Q Race.12 Updated Theory Clarity and Coherence",
        "Q Race.12 Updated Theory Causal Reasoning",
        "Q Race.12 Updated Theory Theoretical Depth",
        "Q Race.12 Updated Theory Creativity",
        "Q Race.12 Updated Theory Persuasiveness",
    ],
    "gender_pre": [
        "Q Gender.4 Clarity and Coherence",
        "Q Gender.4 Causal Reasoning",
        "Q Gender.4 Theoretical Depth",
        "Q Gender.4 Creativity",
        "Q Gender.4 Persuasiveness",
    ],
    "gender_post": [
        "Q Gender.12 Clarity and Coherence",
        "Q Gender.12 Causal Reasoning",
        "Q Gender.12 Theoretical Depth",
        "Q Gender.12 Creativity",
        "Q Gender.12 Persuasiveness",
    ],
}


def col_idx(headers: list[str], exact: str) -> int:
    for i, h in enumerate(headers):
        if h.strip() == exact.strip():
            return i
    raise KeyError(exact)


def mean_dims(row: list[str], headers: list[str], cols: list[str]) -> float:
    vals = []
    for c in cols:
        v = row[col_idx(headers, c)].strip()
        if v:
            vals.append(float(v))
    return float(np.mean(vals)) if len(vals) == 5 else np.nan


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else np.nan


def build_all_metrics() -> pd.DataFrame:
    with CSV_PATH.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    headers, data = rows[0], rows[1:]

    with (ROOT / "main_effects" / "ML_results.json").open() as f:
        ml_raw = json.load(f)
    ml_signs = {
        task: {e["feature"]: e["sign"] for e in sorted(entries, key=lambda e: e["rank"])}
        for task, entries in ml_raw.items()
    }

    features = [
        re.sub(r"^Q Race\.2 \(rank\) - ", "", h)
        for h in headers
        if re.match(r"^Q Race\.2 \(rank\) - ", h)
    ]
    feat_idx = {f: i for i, f in enumerate(features)}
    sign_map = {"+": 1, "-": -1}

    r1_col = col_idx(headers, "Q Race.1")
    g1_col = col_idx(headers, "Q Gender.1")
    r3_cols = {
        re.sub(r"^Q Race\.3 \(sign\) - ", "", h): col_idx(headers, h)
        for h in headers
        if h.startswith("Q Race.3 (sign)")
    }
    g3_cols = {
        re.sub(r"^Q Gender\.3 \(sign\) - ", "", h): col_idx(headers, h)
        for h in headers
        if h.startswith("Q Gender.3 (sign)")
    }

    def build_binary_vector(q1_col: int, q3_col_map: dict, row: list[str]) -> np.ndarray | None:
        vec = np.zeros(len(features))
        cell = row[q1_col].strip()
        if not cell:
            return None
        for feat in cell.split(","):
            feat = feat.strip()
            if feat not in feat_idx:
                continue
            sign_str = row[q3_col_map[feat]].strip() if feat in q3_col_map else ""
            vec[feat_idx[feat]] = sign_map.get(sign_str, 0)
        return vec

    def build_ml_binary(signs_dict: dict) -> np.ndarray:
        vec = np.zeros(len(features))
        for feat, sign_str in signs_dict.items():
            if feat in feat_idx:
                vec[feat_idx[feat]] = sign_map.get(sign_str, 0)
        return vec

    ml_bin_race = build_ml_binary(ml_signs["race"])
    ml_bin_gender = build_ml_binary(ml_signs["gender"])

    with (ROOT / "second_order_interactions" / "ML_results.json").open() as f:
        ml_soi = json.load(f)
    feature_set = set(features)
    pairs = list(combinations(sorted(features), 2))
    pair_idx = {p: i for i, p in enumerate(pairs)}

    def canon_pair(a: str, b: str) -> tuple[str, str]:
        return tuple(sorted((a.strip(), b.strip())))

    def parse_pair(cell: str) -> tuple[str, str] | None:
        cell = cell.strip()
        if not cell or "," not in cell:
            return None
        parts = [x.strip() for x in cell.split(",")]
        if len(parts) != 2 or parts[0] == parts[1]:
            return None
        if parts[0] not in feature_set or parts[1] not in feature_set:
            return None
        return canon_pair(parts[0], parts[1])

    r_pair_cols = [
        col_idx(headers, "Q Race.6 (SOI, 1st)"),
        col_idx(headers, "Q Race.7 (SOI, 2nd)"),
        col_idx(headers, "Q Race.8 (SOI, 3rd)"),
    ]
    r_sign_cols = [col_idx(headers, f"Q Race.9 (SOI, sign, {s})") for s in ["1st", "2nd", "3rd"]]
    g_pair_cols = [
        col_idx(headers, "Q Gender.6 (SOI, 1st)"),
        col_idx(headers, "Q Gender.7 (SOI, 2nd)"),
        col_idx(headers, "Q Gender.8 (SOI, 3rd)"),
    ]
    g_sign_cols = [col_idx(headers, f"Q Gender.9 (SOI, sign, {s})") for s in ["1st", "2nd", "3rd"]]

    def build_soi_vector(pair_cols: list[int], sign_cols: list[int], row: list[str]) -> np.ndarray:
        vec = np.zeros(len(pairs))
        for pc, sc in zip(pair_cols, sign_cols):
            p = parse_pair(row[pc])
            if p is None:
                continue
            vec[pair_idx[p]] = sign_map.get(row[sc].strip(), 0)
        return vec

    def build_ml_soi(entries: list[dict]) -> np.ndarray:
        vec = np.zeros(len(pairs))
        for e in entries:
            p = canon_pair(e["feature_1"], e["feature_2"])
            if p in pair_idx:
                vec[pair_idx[p]] = sign_map.get(e["sign"], 0)
        return vec

    ml_soi_race = build_ml_soi(ml_soi["race"])
    ml_soi_gender = build_ml_soi(ml_soi["gender"])

    name_ix = col_idx(headers, "What is your full name?")
    type_ix = col_idx(headers, "student_0, expert_1, genAI_2")

    records = []
    for row in data:
        gid = str(row[type_ix]).strip()
        name = row[name_ix]
        rec = {"name": name, "group": gid, "group_label": GROUP_LABEL.get(gid, gid)}
        vr = build_binary_vector(r1_col, r3_cols, row)
        vg = build_binary_vector(g1_col, g3_cols, row)
        rec["me_cos_race"] = cosine_sim(vr, ml_bin_race) if vr is not None else np.nan
        rec["me_cos_gender"] = cosine_sim(vg, ml_bin_gender) if vg is not None else np.nan
        sr = build_soi_vector(r_pair_cols, r_sign_cols, row)
        sg = build_soi_vector(g_pair_cols, g_sign_cols, row)
        rec["soi_cos_race"] = cosine_sim(sr, ml_soi_race)
        rec["soi_cos_gender"] = cosine_sim(sg, ml_soi_gender)
        rec["me_cos_avg"] = float(np.nanmean([rec["me_cos_race"], rec["me_cos_gender"]]))
        rec["soi_cos_avg"] = float(np.nanmean([rec["soi_cos_race"], rec["soi_cos_gender"]]))
        for key, cols in QUALITY_DIMS.items():
            rec[key] = mean_dims(row, headers, cols)
        rec["quality_avg"] = float(np.nanmean([rec[k] for k in QUALITY_DIMS]))
        records.append(rec)
    return pd.DataFrame(records)


COSINE_PANELS = [
    ("me_cos_race", "Main Effects · Race"),
    ("me_cos_gender", "Main Effects · Gender"),
    ("me_cos_avg", "Main Effects · Avg"),
    ("soi_cos_race", "SOI · Race"),
    ("soi_cos_gender", "SOI · Gender"),
    ("soi_cos_avg", "SOI · Avg"),
]
COSINE_AVG_PANELS = [
    ("me_cos_avg", "Main Effects"),
    ("soi_cos_avg", "SOI"),
]


def display_name(name: str) -> str:
    return name.replace("(1)", "")


def canon_participant_name(name: str) -> str:
    return name.replace("(1)", "")


def bar_color(name: str) -> str:
    return THEORISTA_COLORS.get(canon_participant_name(name), DEFAULT_BAR_COLOR)


def theorista_names_in(df: pd.DataFrame) -> list[str]:
    wanted = {canon_participant_name(n) for n in THEORISTA}
    return [n for n in df["name"].unique() if canon_participant_name(n) in wanted]


def rank_info(df: pd.DataFrame, metric: str, name: str) -> tuple[int | None, int, float | None, bool]:
    """Return (rank, n_pool, value, in_top10) using dense rank among all scored rows."""
    pool = df.dropna(subset=[metric]).copy()
    pool = pool.sort_values(metric, ascending=False).reset_index(drop=True)
    pool["rank"] = pool[metric].rank(ascending=False, method="min").astype(int)
    top10_names = set(pool.head(10)["name"])
    row = pool[pool["name"] == name]
    if row.empty:
        return None, len(pool), None, False
    return int(row["rank"].iloc[0]), len(pool), float(row[metric].iloc[0]), name in top10_names


def plot_genai_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    metric: str,
    title: str,
    *,
    label_size: float = 8,
    title_size: float = 11,
    show_yticklabels: bool = True,
    title_position: str = "top",
    y_label_offset: float = 3,
) -> None:
    pool = df.dropna(subset=[metric]).copy()
    ranked = pool.sort_values(metric, ascending=True)

    ylabels = [display_name(n) for n in ranked["name"]]
    colors = [bar_color(n) for n in ranked["name"]]
    bars = ax.barh(ylabels, ranked[metric], color=colors, edgecolor="#333333", linewidth=0.5)

    if title:
        if title_position == "bottom":
            ax.set_xlabel(title, fontsize=title_size, fontweight="bold", labelpad=14)
        else:
            ax.set_title(title, fontsize=title_size, fontweight="bold", pad=8)
    ax.grid(axis="x", alpha=0.25)
    ax.tick_params(axis="x", labelsize=label_size)
    ax.tick_params(axis="y", labelsize=label_size + y_label_offset + 3, labelrotation=15)
    ax.tick_params(axis="y", labelleft=show_yticklabels)
    ax.set_ylabel("")

    xmax = max(ranked[metric].max(), 1e-9)
    for bar, val in zip(bars, ranked[metric]):
        ax.text(
            val + xmax * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center",
            fontsize=label_size - 0.5,
        )


def plot_overall_quality(df: pd.DataFrame, metric: str, title: str, filename: str, xlabel: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 14))
    plot_genai_panel(
        ax,
        df,
        metric,
        title,
        label_size=8,
        title_size=PANEL_TITLE_SIZE,
        y_label_offset=7,
    )
    ax.set_xlabel(xlabel, fontsize=FIG_XLABEL_SIZE, fontweight="bold")
    fig.tight_layout()
    fig.subplots_adjust(left=0.28)
    fig.savefig(OUT_DIR / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_cosine_grid(df: pd.DataFrame, filename: str = "top10_cosine_similarity_6panel.png") -> None:
    fig, axes = plt.subplots(1, 2, figsize=(22, 14))

    for ax, (metric, title) in zip(axes, COSINE_AVG_PANELS):
        plot_genai_panel(
            ax,
            df,
            metric,
            title,
            label_size=11,
            title_size=PANEL_TITLE_SIZE,
            show_yticklabels=True,
            title_position="bottom",
        )

    fig.suptitle(
        "Cosine Similarity with ML Evidence",
        fontsize=FIG_SUPTITLE_SIZE,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    fig.subplots_adjust(left=0.22, top=0.95, bottom=0.08, wspace=0.5)
    fig.savefig(OUT_DIR / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    genai = build_all_metrics()
    genai = genai[genai["group"] == "2"].copy()
    genai.to_csv(OUT_DIR / "genai_metrics.csv", index=False)

    plot_overall_quality(
        genai,
        "quality_avg",
        "Overall Quality Assessed by GPT 5.5",
        "top10_overall_quality.png",
        "Overall Quality Score",
    )
    plot_cosine_grid(genai)

    metrics = [
        ("quality_avg", "Overall Quality Assessed by GPT 5.5"),
        *[(m, t) for m, t in COSINE_PANELS],
    ]
    rank_rows = []
    for metric, title in metrics:
        for t in theorista_names_in(genai):
            rk, n, val, in_top = rank_info(genai, metric, t)
            rank_rows.append(
                {
                    "metric": title,
                    "metric_key": metric,
                    "theorista": canon_participant_name(t).replace("Theorista_", ""),
                    "rank": rk,
                    "n_genai": n,
                    "value": round(val, 4) if val is not None else None,
                    "in_top10": in_top,
                }
            )
    pd.DataFrame(rank_rows).to_csv(OUT_DIR / "theorista_genai_rankings.csv", index=False)
    print(f"Saved figures and rankings to {OUT_DIR}")


if __name__ == "__main__":
    main()
