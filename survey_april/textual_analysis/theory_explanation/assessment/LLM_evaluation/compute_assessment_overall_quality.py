"""
Compute Overall Quality Score = mean of 5 dimension scores.

Discovers every model tag present in AI-rating column names
(e.g. gpt-5.5, gpt-5.6) and writes a separate overall column per
(stage × model), e.g.:

  Q Race.4 Overall Quality Score (gpt-5.5)
  Q Race.4 Overall Quality Score (gpt-5.6)
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

INPUT_CSV = Path(__file__).resolve().parents[4] / "All_Participants_All_Questions.csv"

DIM_SUFFIXES: tuple[str, ...] = (
    "Clarity and Coherence",
    "Causal Reasoning",
    "Theoretical Depth",
    "Creativity",
    "Persuasiveness",
)

# (block_name, dimension_column_stem, overall_column_stem)
# dimension columns look like: f"{dim_stem} {suffix} ({model})"
# overall column looks like:   f"{overall_stem} Overall Quality Score ({model})"
STAGE_SPECS: tuple[tuple[str, str, str], ...] = (
    ("Q Race.4", "Q Race.4", "Q Race.4"),
    ("Q Race.12", "Q Race.12 Updated Theory", "Q Race.12"),
    ("Q Race.10", "Q Race.10", "Q Race.10"),
    ("Q Race.15", "Q Race.15", "Q Race.15"),
    ("Q Gender.4", "Q Gender.4", "Q Gender.4"),
    ("Q Gender.12", "Q Gender.12", "Q Gender.12"),
    ("Q Gender.10", "Q Gender.10", "Q Gender.10"),
    ("Q Gender.15", "Q Gender.15", "Q Gender.15"),
)

_MODEL_TAG_RE = re.compile(r"^.+ \(([^()]+)\)\s*$")


def discover_models(columns: list[str]) -> list[str]:
    """Models that have at least one Clarity-and-Coherence rating column."""
    found: set[str] = set()
    for col in columns:
        for suffix in DIM_SUFFIXES:
            needle = f" {suffix} ("
            if needle not in col:
                continue
            m = _MODEL_TAG_RE.match(col)
            if m:
                found.add(m.group(1))
            break
    return sorted(found)


def dim_col(dim_stem: str, suffix: str, model: str) -> str:
    return f"{dim_stem} {suffix} ({model})"


def overall_col(overall_stem: str, model: str) -> str:
    return f"{overall_stem} Overall Quality Score ({model})"


def upsert_overall_column(
    df: pd.DataFrame,
    dimension_cols: list[str],
    output_col: str,
    insert_after_col: str,
) -> None:
    score_frame = df[dimension_cols].apply(pd.to_numeric, errors="coerce")
    # Only average rows with at least one non-missing dimension score.
    overall = score_frame.mean(axis=1, skipna=True)
    all_missing = score_frame.isna().all(axis=1)
    overall = overall.where(~all_missing, other="")

    if output_col not in df.columns:
        insert_at = int(df.columns.get_loc(insert_after_col)) + 1
        df.insert(insert_at, output_col, overall)
    else:
        df[output_col] = overall


def main() -> None:
    df = pd.read_csv(INPUT_CSV, dtype=str, keep_default_na=False)
    columns = list(df.columns)
    models = discover_models(columns)
    if not models:
        print("No model-tagged dimension columns found; nothing to do.")
        return

    print(f"Models found: {', '.join(models)}")
    n_wrote = 0
    n_skip = 0

    for model in models:
        print(f"\n=== {model} ===")
        for block_name, dim_stem, overall_stem in STAGE_SPECS:
            dims = [dim_col(dim_stem, s, model) for s in DIM_SUFFIXES]
            missing = [c for c in dims if c not in df.columns]
            if missing:
                print(f"  - {block_name}: skipped (missing {len(missing)}/5 dims)")
                n_skip += 1
                continue

            out = overall_col(overall_stem, model)
            upsert_overall_column(
                df,
                dimension_cols=dims,
                output_col=out,
                insert_after_col=dims[-1],  # after Persuasiveness
            )
            print(f"  - {block_name}: wrote '{out}'")
            n_wrote += 1

    df.to_csv(INPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\nDone. wrote={n_wrote} skipped={n_skip}")
    print(f"Updated: {INPUT_CSV}")


if __name__ == "__main__":
    main()
