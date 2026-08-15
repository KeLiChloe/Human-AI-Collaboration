"""
Shared batch evaluation engine for theory quality scoring.

Design
------
- Four thin runners (race/gender × main/soi) each define a ConditionConfig.
- Within a condition, pre-ML and post-ML theories are pooled, shuffled with a
  fixed seed (not paired), assigned anonymous theory_ids (T001…), then scored
  in batches of BATCH_SIZE via a single API call per batch.
- Each item presents theory_id + theory text only (study background
  describes the 5-feature / 3-SOI task; per-person selections are not shown).
- Scores are written to the survey CSV immediately after each successful batch.
- Already-scored rows are skipped so runs are resumable.
"""

from .constants import BATCH_SIZE, MODEL, REASONING_EFFORT, SHUFFLE_SEED
from .runner import run_condition

__all__ = [
    "BATCH_SIZE",
    "MODEL",
    "REASONING_EFFORT",
    "SHUFFLE_SEED",
    "run_condition",
]
