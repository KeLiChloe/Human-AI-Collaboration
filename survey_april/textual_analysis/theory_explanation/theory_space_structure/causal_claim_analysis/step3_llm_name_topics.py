#!/usr/bin/env python3
"""
LLM naming for BERTopic candidate topics.

Reads candidate_topics_for_llm_labeling.json from step2, writes topic_labels.json,
and generates claims_umap_by_topic.png with LLM topic labels.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from openai import OpenAI
from pydantic import BaseModel, Field

from step2_bertopic_claim_topic_modeling import (
    DEFAULT_N_NEIGHBORS,
    DEFAULT_OPENAI_DIMENSIONS,
    DEFAULT_OPENAI_EMBEDDING_MODEL,
    DEFAULT_SEED,
    EMBEDDING_CACHE_PATH,
    embed_openai,
)

TEXTUAL_ANALYSIS_DIR = Path(__file__).resolve().parents[3]
if str(TEXTUAL_ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(TEXTUAL_ANALYSIS_DIR))

from viz_style import (  # noqa: E402
    BAR_EDGE_COLOR,
    BAR_EDGE_WIDTH,
    FONT_AXIS_LABEL,
    FONT_LEGEND,
    FONT_TICK,
    FONT_TITLE,
    apply_plot_style,
    style_axes,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "bertopic_outputs" / "pre-ML_race_main-effects"
DEFAULT_INPUT_JSON = DEFAULT_OUTPUT_DIR / "candidate_topics_for_llm_labeling.json"
DEFAULT_ASSIGNMENTS_CSV = DEFAULT_OUTPUT_DIR / "claim_topic_assignments.csv"
DEFAULT_OUTPUT_JSON = DEFAULT_OUTPUT_DIR / "topic_labels.json"
DEFAULT_MODEL = "gpt-5.5"

CLAIMS_UMAP_PNG = "claims_umap_by_topic.png"

MAP_FIGSIZE = (12.4, 7.6)
MAP_FONT_BUMP = 3
MAP_TITLE_Y = 0.995
MAP_SUBTITLE_Y = 0.902
SCATTER_SIZE = 52
SCATTER_ALPHA = 0.82
OUTLIER_COLOR = "#B8B8B8"
TOPIC_PALETTE = [
    "#4E79A7",
    "#F28E2B",
    "#E15759",
    "#76B7B2",
    "#59A14F",
    "#B07AA1",
    "#EDC948",
    "#FF9DA7",
    "#9C755F",
    "#86BCB6",
]

SYSTEM_PROMPT = """\
You help qualitative researchers name BERTopic candidate clusters from theory-explanation claims.

Propose a short thematic label and a one-sentence definition for the topic cluster below.
Use only the provided examples and top terms. Do not invent themes not supported by the examples.

Guidelines:
- Name the substantive theoretical theme (antecedent, mechanism, or discourse context), not generic words like "papers" or "inequality".
- Keep the label concise (2-6 words).
- The definition should describe what participants are claiming, not whether it is correct.
- In cohesion_notes, note if the examples mix multiple distinct sub-themes worth splitting on review; use "none" if cohesive.
"""


class TopicLabelResponse(BaseModel):
    label: str = Field(description="Short thematic label for the topic cluster.")
    definition: str = Field(description="One-sentence definition of the shared theoretical theme.")
    cohesion_notes: str = Field(
        description="Note if this cluster mixes distinct sub-themes worth splitting; use 'none' if cohesive."
    )


def build_user_prompt(topic: dict[str, Any]) -> str:
    lines = [
        f"Topic ID: {topic['topic_id']}",
        f"Claims: {topic['n_claims']} | Respondents: {topic['n_respondents']}",
        f"Top c-TF-IDF terms: {topic.get('top_terms', '')}",
        "",
        "Representative claim examples:",
    ]
    for i, ex in enumerate(topic.get("examples", [])[:20], start=1):
        lines.append(f"{i}. [{ex.get('group', '')}] {ex.get('claim_text', '')}")
        ant = ex.get("antecedent_text", "")
        mech = ex.get("mechanism_text", "")
        if ant:
            lines.append(f"   antecedent: {ant}")
        if mech and str(mech).lower() != "unspecified":
            lines.append(f"   mechanism: {mech}")
    lines.append("")
    lines.append("Return a short label, one-sentence definition, and cohesion_notes.")
    return "\n".join(lines)


def label_topic(client: OpenAI, model: str, topic: dict[str, Any]) -> TopicLabelResponse:
    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(topic)},
        ],
        response_format=TopicLabelResponse,
    )
    parsed = completion.choices[0].message.parsed
    if parsed is None:
        raise RuntimeError(f"Model returned no parsed label for topic {topic['topic_id']}.")
    return parsed


def topic_label_map(labeled: list[dict[str, Any]]) -> dict[int, str]:
    out: dict[int, str] = {}
    for topic in labeled:
        topic_id = int(topic["topic_id"])
        if topic.get("is_outlier_topic"):
            out[topic_id] = "outlier"
        else:
            out[topic_id] = str(topic["label"])
    out.setdefault(-1, "outlier")
    return out


def topic_color_map(label_map: dict[int, str]) -> dict[str, str]:
    labels = sorted({label for label in label_map.values() if label != "outlier"})
    colors = {label: TOPIC_PALETTE[i % len(TOPIC_PALETTE)] for i, label in enumerate(labels)}
    colors["outlier"] = OUTLIER_COLOR
    return colors


def _style_map_axes(ax: plt.Axes, *, tick_fontsize: float | None = None) -> None:
    style_axes(ax)
    ax.grid(True, alpha=0.22, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", labelsize=tick_fontsize or (FONT_TICK + MAP_FONT_BUMP))


def _save_map_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(
        path,
        dpi=600,
        pad_inches=0.12,
        bbox_inches="tight",
    )
    plt.close(fig)
    print(f"Saved: {path}")


def save_claims_umap_by_topic(
    assignments: pd.DataFrame,
    embeddings: np.ndarray,
    label_map: dict[int, str],
    output_dir: Path,
    seed: int,
    n_neighbors: int,
) -> None:
    path = output_dir / CLAIMS_UMAP_PNG
    try:
        from umap import UMAP
    except ImportError as e:
        print(f"[warning] Could not save {path.name}: {e}")
        return

    apply_plot_style()
    n_docs = len(assignments)
    neighbors = max(2, min(n_neighbors, n_docs - 1))
    coords = UMAP(
        n_components=2,
        n_neighbors=neighbors,
        min_dist=0.12,
        metric="cosine",
        random_state=seed,
    ).fit_transform(embeddings)

    plot_df = assignments.copy()
    plot_df["umap_x"] = coords[:, 0]
    plot_df["umap_y"] = coords[:, 1]
    plot_df["topic_label"] = plot_df["topic_id"].apply(
        lambda t: label_map.get(int(t), f"topic {int(t)}")
    )
    colors = topic_color_map(label_map)
    label_counts = plot_df.groupby("topic_label", sort=False).size().to_dict()
    legend_order = sorted(
        label_counts.keys(),
        key=lambda x: (x == "outlier", -label_counts.get(x, 0), x),
    )

    fig, ax = plt.subplots(figsize=MAP_FIGSIZE)
    for label in legend_order:
        subset = plot_df[plot_df["topic_label"] == label]
        ax.scatter(
            subset["umap_x"],
            subset["umap_y"],
            c=colors.get(label, "#888888"),
            s=SCATTER_SIZE,
            alpha=SCATTER_ALPHA if label != "outlier" else 0.55,
            edgecolors=BAR_EDGE_COLOR,
            linewidths=BAR_EDGE_WIDTH,
            zorder=1 if label == "outlier" else 2,
        )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=colors.get(label, "#888888"),
            markeredgecolor=BAR_EDGE_COLOR,
            markeredgewidth=BAR_EDGE_WIDTH,
            markersize=8,
            alpha=SCATTER_ALPHA if label != "outlier" else 0.55,
        )
        for label in legend_order
    ]

    axis_label_fs = FONT_AXIS_LABEL + MAP_FONT_BUMP
    ax.set_xlabel(
        "UMAP dimension 1",
        fontsize=axis_label_fs,
        fontweight="bold",
        labelpad=10,
    )
    ax.set_ylabel(
        "UMAP dimension 2",
        fontsize=axis_label_fs,
        fontweight="bold",
        labelpad=10,
    )
    _style_map_axes(ax)
    fig.suptitle(
        "Claim-Level Semantic Map by Topic Theme",
        fontsize=FONT_TITLE + MAP_FONT_BUMP,
        fontweight="bold",
        y=MAP_TITLE_Y,
    )
    fig.text(
        0.5,
        MAP_SUBTITLE_Y,
        "Each point is one causal claim; colors show LLM-labeled BERTopic themes.",
        ha="center",
        va="top",
        fontsize=FONT_TICK + MAP_FONT_BUMP,
        color="#444444",
    )

    legend = ax.legend(
        legend_handles,
        legend_order,
        title="Topic theme",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=False,
        fontsize=FONT_LEGEND - 1.5 + MAP_FONT_BUMP,
        title_fontsize=FONT_LEGEND - 0.5 + MAP_FONT_BUMP,
        handlelength=1.2,
        handletextpad=0.6,
        borderaxespad=0.0,
    )
    legend.get_title().set_fontweight("bold")

    fig.subplots_adjust(left=0.08, right=0.58, bottom=0.12, top=0.84)
    _save_map_figure(fig, path)


def load_run_viz_params(output_dir: Path) -> tuple[int, int, str, int | None]:
    run_info_path = output_dir / "run_info.json"
    if run_info_path.exists():
        with run_info_path.open(encoding="utf-8") as f:
            info = json.load(f)
        return (
            int(info.get("seed", DEFAULT_SEED)),
            int(info.get("n_neighbors", DEFAULT_N_NEIGHBORS)),
            str(info.get("openai_embedding_model", DEFAULT_OPENAI_EMBEDDING_MODEL)),
            info.get("openai_dimensions", DEFAULT_OPENAI_DIMENSIONS),
        )
    return DEFAULT_SEED, DEFAULT_N_NEIGHBORS, DEFAULT_OPENAI_EMBEDDING_MODEL, DEFAULT_OPENAI_DIMENSIONS


def load_claim_embeddings(
    assignments: pd.DataFrame,
    model: str,
    dimensions: int | None,
) -> np.ndarray:
    texts = assignments["topic_model_text"].astype(str).tolist()
    return embed_openai(
        texts,
        model=model,
        dimensions=dimensions,
        cache_path=EMBEDDING_CACHE_PATH,
    )


def save_labeled_visualizations(
    assignments_path: Path,
    labeled: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    assignments = pd.read_csv(assignments_path)
    seed, n_neighbors, model, dimensions = load_run_viz_params(output_dir)
    embeddings = load_claim_embeddings(assignments, model=model, dimensions=dimensions)
    label_map = topic_label_map(labeled)
    save_claims_umap_by_topic(
        assignments, embeddings, label_map, output_dir, seed=seed, n_neighbors=n_neighbors
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM topic labeling and publication PNG maps.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_JSON)
    parser.add_argument("--assignments", type=Path, default=DEFAULT_ASSIGNMENTS_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--viz-only",
        action="store_true",
        help="Skip LLM labeling; regenerate claims_umap_by_topic.png from existing topic_labels.json.",
    )
    args = parser.parse_args()

    input_json = args.input.expanduser().resolve()
    assignments_path = args.assignments.expanduser().resolve()
    output_json = args.output.expanduser().resolve()
    output_dir = output_json.parent

    if args.viz_only:
        if not output_json.exists():
            raise FileNotFoundError(f"--viz-only requires existing labels file: {output_json}")
        with output_json.open(encoding="utf-8") as f:
            labeled = json.load(f).get("topics", [])
        if not labeled:
            raise ValueError(f"No topics found in {output_json}")
        save_labeled_visualizations(assignments_path, labeled, output_dir)
        return

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")

    model = args.model

    with input_json.open(encoding="utf-8") as f:
        payload = json.load(f)

    topics = payload.get("topics", [])
    if not topics:
        raise ValueError(f"No topics found in {input_json}")

    client = OpenAI()
    labeled: list[dict[str, Any]] = []

    for topic in topics:
        topic_id = int(topic["topic_id"])
        if topic.get("is_outlier_topic"):
            labeled.append(
                {
                    "topic_id": topic_id,
                    "is_outlier_topic": True,
                    "n_claims": topic.get("n_claims", 0),
                    "n_respondents": topic.get("n_respondents", 0),
                    "top_terms": topic.get("top_terms", ""),
                    "label": "outlier",
                    "definition": "BERTopic outlier bucket; no stable shared theme.",
                    "cohesion_notes": "none",
                    "examples": topic.get("examples", []),
                }
            )
            print(f"topic {topic_id}: skipped (outlier)")
            continue

        result = label_topic(client, model, topic)
        labeled.append(
            {
                "topic_id": topic_id,
                "is_outlier_topic": False,
                "n_claims": topic.get("n_claims", 0),
                "n_respondents": topic.get("n_respondents", 0),
                "top_terms": topic.get("top_terms", ""),
                "label": result.label,
                "definition": result.definition,
                "cohesion_notes": result.cohesion_notes,
                "examples": topic.get("examples", []),
            }
        )
        print(f"topic {topic_id}: {result.label}")

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump({"topics": labeled, "model": model, "source": str(input_json)}, f, ensure_ascii=False, indent=2)

    print(f"\nSaved: {output_json}")

    if assignments_path.exists():
        save_labeled_visualizations(assignments_path, labeled, output_dir)
    else:
        print(f"[warning] Skipping visualizations; assignments not found: {assignments_path}")


if __name__ == "__main__":
    main()
