"""Main evaluation loop: catalog → filter → batch API → immediate CSV write."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tqdm import tqdm

from .api_batch import score_batch
from .catalog import (
    ConditionConfig,
    TheoryItem,
    assign_ids_and_shuffle,
    chunked,
    collect_items,
)
from .constants import BATCH_SIZE, INPUT_CSV, MODEL, REASONING_EFFORT, SHUFFLE_SEED
from .prompts import build_system_prompt, build_user_prompt
from .scoring import ensure_condition_columns, is_item_scored, write_score

_EVAL_DIR = Path(__file__).resolve().parents[1]
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

from csv_score_io import locked_csv, locked_csv_read  # noqa: E402


def _estimate_tokens(text: str) -> int:
    return max(len(text) // 4, int(len(text.split()) * 1.3))


def build_pending_catalog(
    cfg: ConditionConfig,
    *,
    seed: int = SHUFFLE_SEED,
    model_tag: str = MODEL,
    limit: int | None = None,
) -> tuple[list[TheoryItem], list[TheoryItem], int]:
    """
    Returns (all_shuffled, pending_unscored, n_already_scored).
    Catalog ids are assigned on the full shuffled set so ids stay stable
    across resumes; pending is a filtered view preserving that order.
    Column names are tagged with model_tag (must match the API model).
    """
    with locked_csv(INPUT_CSV) as df:
        ensure_condition_columns(df, cfg, model_tag=model_tag)

    with locked_csv_read(INPUT_CSV) as df:
        raw = collect_items(df, cfg, model_tag=model_tag)
        catalog = assign_ids_and_shuffle(raw, seed=seed)
        pending = [item for item in catalog if not is_item_scored(df, item)]
        n_done = len(catalog) - len(pending)
        if limit is not None:
            pending = pending[:limit]
        return catalog, pending, n_done


def run_condition(
    cfg: ConditionConfig,
    *,
    batch_size: int = BATCH_SIZE,
    seed: int = SHUFFLE_SEED,
    model: str = MODEL,
    reasoning_effort: str = REASONING_EFFORT,
    dry_run: bool = False,
    limit: int | None = None,
) -> None:
    catalog, pending, n_done = build_pending_catalog(
        cfg, seed=seed, model_tag=model, limit=limit
    )
    batches = chunked(pending, batch_size)

    print(f"Condition: {cfg.key}  (task={cfg.task}, effect={cfg.effect})")
    print(f"CSV: {INPUT_CSV}")
    print(f"Catalog size: {len(catalog)}  already scored: {n_done}  pending: {len(pending)}")
    print(
        f"Batches: {len(batches)} × up to {batch_size}  "
        f"model={model}  reasoning.effort={reasoning_effort}  seed={seed}"
    )
    if catalog:
        print(f"Score columns tagged: ({model})  e.g. {catalog[0].score_columns[0]}")

    if dry_run:
        system = build_system_prompt(task=cfg.task, effect=cfg.effect)
        sys_tok = _estimate_tokens(system)
        print(f"System prompt ≈ {sys_tok} tokens")
        for bi, batch in enumerate(batches, start=1):
            user = build_user_prompt(batch)
            print(
                f"  batch {bi:02d}: n={len(batch):2d}  "
                f"ids={batch[0].theory_id}…{batch[-1].theory_id}  "
                f"user≈{_estimate_tokens(user):,} tok  "
                f"total≈{sys_tok + _estimate_tokens(user):,} tok"
            )
            # show one sample block header
            if bi == 1 and batch:
                sample = batch[0]
                print(
                    f"    sample {sample.theory_id}: stage={sample.stage} "
                    f"row={sample.row_index} col={sample.theory_column!r}"
                )
                preview = sample.theory_text.replace("\n", " ").strip()
                print(f"    theory preview: {preview[:220]}{'…' if len(preview) > 220 else ''}")
        print("Dry-run only; no API calls.")
        return

    if not pending:
        print("Nothing to score.")
        return

    for bi, batch in enumerate(
        tqdm(batches, desc=f"{cfg.key} batches", unit="batch"),
        start=1,
    ):
        scores = score_batch(
            batch,
            task=cfg.task,
            effect=cfg.effect,
            model=model,
            reasoning_effort=reasoning_effort,
        )
        with locked_csv(INPUT_CSV) as df:
            ensure_condition_columns(df, cfg, model_tag=model)
            for item, score in zip(batch, scores):
                write_score(df, item, score)
        print(
            f"  wrote batch {bi}/{len(batches)} "
            f"({batch[0].theory_id}…{batch[-1].theory_id}, n={len(batch)})"
        )

    print(f"Done: {cfg.key}")


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--dry-run", action="store_true", help="Catalog + token estimate only")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=SHUFFLE_SEED)
    parser.add_argument("--model", type=str, default=MODEL)
    parser.add_argument(
        "--effort",
        type=str,
        default=REASONING_EFFORT,
        choices=["none", "low", "medium", "high", "xhigh", "max"],
        help="Responses API reasoning.effort (default: high)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Score at most N pending items (after shuffle/filter)",
    )
    return parser


def main_for(cfg: ConditionConfig, argv: list[str] | None = None) -> None:
    parser = add_common_args(
        argparse.ArgumentParser(description=f"LLM theory evaluation: {cfg.key}")
    )
    args = parser.parse_args(argv)
    run_condition(
        cfg,
        batch_size=args.batch_size,
        seed=args.seed,
        model=args.model,
        reasoning_effort=args.effort,
        dry_run=args.dry_run,
        limit=args.limit,
    )
