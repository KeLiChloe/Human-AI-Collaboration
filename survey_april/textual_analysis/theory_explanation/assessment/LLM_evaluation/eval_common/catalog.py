"""Build shuffled anonymous catalogs of theories for one evaluation condition."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import pandas as pd

from .constants import (
    BRIEF_REASONING_SUFFIX,
    NAME_COLUMN,
    SCORE_MODEL_TAG,
    SCORE_SUFFIXES,
    SHUFFLE_SEED,
    metric_col,
)


@dataclass(frozen=True)
class StageSpec:
    """One pre or post stage within a condition."""

    stage: str  # 'pre' | 'post'
    theory_columns: tuple[str, ...]  # preference order (first nonempty wins)
    score_prefix: str  # e.g. 'Q Race.4' or 'Q Race.12 Updated Theory'
    insert_after: str  # column after which to ensure score columns


@dataclass(frozen=True)
class ConditionConfig:
    """One of the four evaluation files."""

    key: str  # e.g. 'race_main'
    task: str  # 'Race' | 'Gender'
    effect: str  # 'main' | 'soi'
    stages: tuple[StageSpec, ...]


@dataclass
class TheoryItem:
    theory_id: str
    row_index: int
    participant_name: str
    stage: str
    theory_column: str
    score_prefix: str
    theory_text: str
    predictors_text: str = ""  # unused; kept for backward-compatible TheoryItem shape
    model_tag: str = SCORE_MODEL_TAG
    score_columns: tuple[str, ...] = field(default_factory=tuple)
    brief_reasoning_column: str = ""

    def __post_init__(self) -> None:
        if not self.score_columns:
            object.__setattr__(
                self,
                "score_columns",
                tuple(
                    metric_col(self.score_prefix, s, model_tag=self.model_tag)
                    for s in SCORE_SUFFIXES
                ),
            )
        if not self.brief_reasoning_column:
            object.__setattr__(
                self,
                "brief_reasoning_column",
                metric_col(
                    self.score_prefix,
                    BRIEF_REASONING_SUFFIX,
                    model_tag=self.model_tag,
                ),
            )

    @property
    def write_columns(self) -> tuple[str, ...]:
        return (*self.score_columns, self.brief_reasoning_column)


def _nonempty_text(val: object) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    text = str(val).strip()
    return text


def resolve_theory_text(row: pd.Series, columns: tuple[str, ...]) -> tuple[str, str]:
    """Return (column_used, text) for the first nonempty candidate column."""
    for col in columns:
        if col not in row.index:
            continue
        text = _nonempty_text(row[col])
        if text:
            return col, text
    return "", ""


def collect_items(
    df: pd.DataFrame,
    cfg: ConditionConfig,
    *,
    model_tag: str = SCORE_MODEL_TAG,
) -> list[TheoryItem]:
    """Collect all nonempty pre/post theories (no ids yet).

    Post theories must come from the configured LLM_refined column only;
    a missing column or empty cell raises ValueError (no raw-post fallback).
    """
    for stage in cfg.stages:
        if stage.stage != "post":
            continue
        for col in stage.theory_columns:
            if col not in df.columns:
                raise ValueError(
                    f"[{cfg.key}] required post theory column missing from CSV: {col!r}"
                )

    items: list[TheoryItem] = []
    for i in range(len(df)):
        row = df.iloc[i]
        name = _nonempty_text(row.get(NAME_COLUMN, "")) or f"row_{i}"
        for stage in cfg.stages:
            col, text = resolve_theory_text(row, stage.theory_columns)
            if not text:
                if stage.stage == "post":
                    expected = ", ".join(repr(c) for c in stage.theory_columns)
                    raise ValueError(
                        f"[{cfg.key}] missing LLM_refined post theory for "
                        f"row {i} ({name}); expected nonempty in {expected}"
                    )
                continue
            items.append(
                TheoryItem(
                    theory_id="",  # assigned after shuffle
                    row_index=int(i),
                    participant_name=name,
                    stage=stage.stage,
                    theory_column=col,
                    score_prefix=stage.score_prefix,
                    theory_text=text,
                    predictors_text="",
                    model_tag=model_tag,
                )
            )
    return items


def assign_ids_and_shuffle(
    items: list[TheoryItem],
    *,
    seed: int = SHUFFLE_SEED,
) -> list[TheoryItem]:
    """Shuffle with fixed seed, then assign T001… globally within the condition."""
    shuffled = list(items)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    out: list[TheoryItem] = []
    for n, item in enumerate(shuffled, start=1):
        out.append(
            TheoryItem(
                theory_id=f"T{n:03d}",
                row_index=item.row_index,
                participant_name=item.participant_name,
                stage=item.stage,
                theory_column=item.theory_column,
                score_prefix=item.score_prefix,
                theory_text=item.theory_text,
                predictors_text=item.predictors_text,
                model_tag=item.model_tag,
            )
        )
    return out


def chunked(items: list[TheoryItem], batch_size: int) -> list[list[TheoryItem]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]
