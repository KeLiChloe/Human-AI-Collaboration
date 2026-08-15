"""Score readiness checks and CSV writeback helpers."""

from __future__ import annotations

import pandas as pd

from .catalog import ConditionConfig, TheoryItem
from .constants import (
    BRIEF_REASONING_SUFFIX,
    SCORE_FIELD_TO_SUFFIX,
    SCORE_MODEL_TAG,
    SCORE_SUFFIXES,
    metric_col,
)
from .schemas import TheoryScore

# Import ensure_columns_after from sibling package module
import sys
from pathlib import Path

_EVAL_DIR = Path(__file__).resolve().parents[1]
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

from csv_score_io import ensure_columns_after  # noqa: E402


def metric_columns_for_prefix(
    prefix: str,
    *,
    model_tag: str = SCORE_MODEL_TAG,
) -> list[str]:
    cols = [metric_col(prefix, s, model_tag=model_tag) for s in SCORE_SUFFIXES]
    cols.append(metric_col(prefix, BRIEF_REASONING_SUFFIX, model_tag=model_tag))
    return cols


def ensure_condition_columns(
    df: pd.DataFrame,
    cfg: ConditionConfig,
    *,
    model_tag: str = SCORE_MODEL_TAG,
) -> None:
    """Create missing score + brief_reasoning columns for each stage."""
    for stage in cfg.stages:
        cols = metric_columns_for_prefix(stage.score_prefix, model_tag=model_tag)
        ensure_columns_after(df, stage.insert_after, cols)


def is_item_scored(df: pd.DataFrame, item: TheoryItem) -> bool:
    """True if all five numeric dimension scores are present and in 1–10."""
    for col in item.score_columns:
        if col not in df.columns:
            return False
        val = df.at[item.row_index, col]
        if pd.isna(val) or str(val).strip() == "":
            return False
        try:
            num = float(val)
        except (TypeError, ValueError):
            return False
        if num < 1 or num > 10:
            return False
    return True


def write_score(df: pd.DataFrame, item: TheoryItem, score: TheoryScore) -> None:
    # CSV is loaded as strings (Arrow string dtype); write scores as str.
    for field, suffix in SCORE_FIELD_TO_SUFFIX.items():
        col = metric_col(item.score_prefix, suffix, model_tag=item.model_tag)
        df.at[item.row_index, col] = str(int(getattr(score, field)))
    df.at[item.row_index, item.brief_reasoning_column] = str(score.brief_reasoning)
