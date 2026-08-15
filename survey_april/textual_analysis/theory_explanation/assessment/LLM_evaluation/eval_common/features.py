"""Extract selected predictors + signs for main-effects and SOI theories."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .constants import FEATURE_NAMES


@dataclass(frozen=True)
class SelectedPredictor:
    """One selected main-effects feature with rank and sign."""

    name: str
    rank: int
    sign: str  # '+' or '-'


@dataclass(frozen=True)
class SelectedInteraction:
    """One SOI term (comma-separated feature pair) with sign."""

    features: str  # e.g. "social_science,authors_race_diversity_score"
    order: int  # 1, 2, 3
    sign: str


def _cell(row: pd.Series, col: str) -> str:
    if col not in row.index:
        return ""
    val = row[col]
    if pd.isna(val):
        return ""
    return str(val).strip()


def extract_main_predictors(row: pd.Series, task: str) -> list[SelectedPredictor]:
    """
    Main-effects: features with nonempty Q{task}.2 rank, sorted by rank.
    Sign from Q{task}.3.
    """
    out: list[SelectedPredictor] = []
    for name in FEATURE_NAMES:
        rank_col = f"Q {task}.2 (rank) - {name}"
        sign_col = f"Q {task}.3 (sign) - {name}"
        rank_raw = _cell(row, rank_col)
        if not rank_raw:
            continue
        try:
            rank = int(float(rank_raw))
        except ValueError:
            continue
        sign = _cell(row, sign_col) or "?"
        out.append(SelectedPredictor(name=name, rank=rank, sign=sign))
    out.sort(key=lambda p: p.rank)
    return out


def extract_soi_interactions(row: pd.Series, task: str) -> list[SelectedInteraction]:
    """SOI: Q{task}.6/7/8 feature pairs + Q{task}.9 signs."""
    specs = (
        (1, f"Q {task}.6 (SOI, 1st)", f"Q {task}.9 (SOI, sign, 1st)"),
        (2, f"Q {task}.7 (SOI, 2nd)", f"Q {task}.9 (SOI, sign, 2nd)"),
        (3, f"Q {task}.8 (SOI, 3rd)", f"Q {task}.9 (SOI, sign, 3rd)"),
    )
    out: list[SelectedInteraction] = []
    for order, feat_col, sign_col in specs:
        feats = _cell(row, feat_col)
        if not feats:
            continue
        sign = _cell(row, sign_col) or "?"
        out.append(SelectedInteraction(features=feats, order=order, sign=sign))
    return out


def format_main_predictors(predictors: list[SelectedPredictor]) -> str:
    if not predictors:
        return "(no predictors recorded)"
    lines = ["Selected predictors (ranked):"]
    for p in predictors:
        lines.append(f"{p.rank}. {p.name} ({p.sign})")
    return "\n".join(lines)


def format_soi_interactions(interactions: list[SelectedInteraction]) -> str:
    if not interactions:
        return "(no interactions recorded)"
    lines = ["Selected interactions (ordered):"]
    for it in interactions:
        lines.append(f"{it.order}. {it.features} ({it.sign})")
    return "\n".join(lines)
