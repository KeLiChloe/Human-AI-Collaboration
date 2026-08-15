#!/usr/bin/env python3
"""
Build interactive HTML demos for semantic threshold network figures (plot 05).

Each static PNG under visualizations/network/ gets a sibling .html file with
hover tooltips showing participant name + theory explanation.

Example:
    python build_network_interactive_demos.py
    python build_network_interactive_demos.py --embeddings-root textual_analysis/.../embeddings_openai
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
NETWORK_PKG = SCRIPT_DIR / "embeddings_openai/visualizations/network"
for p in (SCRIPT_DIR, NETWORK_PKG):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from step2_embedding_analysis import (  # noqa: E402
    ANALYSIS_SEED,
    COLLAPSED_PARTICIPANT_TYPE_COL,
    DEFAULT_EMBEDDING_COLUMNS,
    NETWORK_INTERACTIVE_COLLAPSED_HTML,
    NETWORK_INTERACTIVE_HTML,
    NETWORK_INTERACTIVE_INDEX,
    NETWORK_SUBDIR,
    PARTICIPANT_TYPE_COL,
    THRESHOLD_QUANTILE,
    VISUALIZATIONS_DIRNAME,
    available_embedding_columns,
    batch_visualizations_root,
    build_group_threshold_network_payload,
    discover_embedding_set_dirs,
    embedding_set_label,
    load_theory_texts_for_embedding_set,
    normalize,
    resolve_network_dir,
    respondent_name_map_for_embeddings_root,
    semantic_neighbor_degree,
    stack_embeddings,
)
from network_interactive.render import (  # noqa: E402
    render_network_index_html,
    render_network_interactive_html,
)


def collect_network_demo_jobs(
    embedding_set_dir: Path,
    embeddings_root: Path,
    embedding_col: str,
    *,
    seed: int = ANALYSIS_SEED,
    threshold_quantile: float = THRESHOLD_QUANTILE,
    name_map: dict[str, str] | None = None,
) -> list[tuple[dict, Path, dict]]:
    """Build payloads and index metadata without writing HTML yet."""
    df = pd.read_parquet(embedding_set_dir / "embeddings_wide.parquet")
    X = normalize(stack_embeddings(df, embedding_col))
    _, similarity_threshold = semantic_neighbor_degree(
        X,
        labels=df[PARTICIPANT_TYPE_COL].values,
        threshold_quantile=threshold_quantile,
    )
    label_text = embedding_set_label(embedding_set_dir)
    theory_by_name = load_theory_texts_for_embedding_set(embedding_set_dir)
    network_dir = resolve_network_dir(embeddings_root, embedding_set_dir, embedding_col)
    jobs: list[tuple[dict, Path, dict]] = []

    variants = (
        (PARTICIPANT_TYPE_COL, NETWORK_INTERACTIVE_HTML, "Three groups (PhDs, Senior Scientists, GenAI)"),
        (COLLAPSED_PARTICIPANT_TYPE_COL, NETWORK_INTERACTIVE_COLLAPSED_HTML, "Two groups (Humans and GenAI)"),
    )
    for group_col, html_name, variant_label in variants:
        payload = build_group_threshold_network_payload(
            df,
            X,
            similarity_threshold,
            threshold_quantile,
            label_text,
            seed=seed,
            group_col=group_col,
            theory_by_name=theory_by_name,
            name_map=name_map,
        )
        outpath = network_dir / html_name
        rel_href = outpath.relative_to(
            batch_visualizations_root(embeddings_root) / NETWORK_SUBDIR
        )
        parts = embedding_set_dir.relative_to(embeddings_root).parts
        task, theory_type, phase = parts[0], parts[1], parts[2]
        variant_key = "collapsed" if group_col == COLLAPSED_PARTICIPANT_TYPE_COL else "three_groups"
        jobs.append(
            (
                payload,
                outpath,
                {
                    "href": str(rel_href).replace("\\", "/"),
                    "task": task,
                    "theory_type": theory_type,
                    "phase": phase,
                    "variant_key": variant_key,
                    "title": label_text,
                    "subtitle": payload["title"],
                    "variant": variant_label,
                },
            )
        )
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build interactive network HTML demos.")
    parser.add_argument(
        "--embeddings-root",
        type=Path,
        default=SCRIPT_DIR / "embeddings_openai",
        help="Root folder containing task/phase embedding sets.",
    )
    parser.add_argument(
        "--embedding-col",
        default=DEFAULT_EMBEDDING_COLUMNS[0],
        help="Embedding column in embeddings_wide.parquet.",
    )
    args = parser.parse_args()
    embeddings_root = args.embeddings_root.expanduser().resolve()
    set_dirs = discover_embedding_set_dirs(str(embeddings_root))
    name_map = respondent_name_map_for_embeddings_root(embeddings_root, seed=ANALYSIS_SEED)

    all_jobs: list[tuple[dict, Path, dict]] = []
    for set_dir in set_dirs:
        cols = available_embedding_columns(
            pd.read_parquet(set_dir / "embeddings_wide.parquet"),
            [args.embedding_col],
        )
        for col in cols:
            print(f"\n=== {embedding_set_label(set_dir)} · {col} ===")
            all_jobs.extend(
                collect_network_demo_jobs(
                    set_dir,
                    embeddings_root,
                    col,
                    name_map=name_map,
                )
            )

    all_index = [entry for _, _, entry in all_jobs]
    for payload, outpath, _ in all_jobs:
        render_network_interactive_html(payload, outpath, nav_entries=all_index)
        print(f"  Saved {outpath}")

    index_path = (
        batch_visualizations_root(embeddings_root)
        / NETWORK_SUBDIR
        / NETWORK_INTERACTIVE_INDEX
    )
    render_network_index_html(all_index, index_path)
    print(f"\nGallery index: {index_path}")
    print(f"Open: file://{index_path}")


if __name__ == "__main__":
    main()
