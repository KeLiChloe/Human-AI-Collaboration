"""
Q5 pre vs post comparisons (apple-to-apple by task).

Each task pairs pre-ML and post-ML under the same topic and theory block, e.g.:
    race/main-effects/pre-ML  vs  race/main-effects/post-ML
    gender/soi/pre-ML         vs  gender/soi/post-ML

Prediction 1 reads plot-06 CSV outputs from analysis.py:
    visualizations/data/<phase>__<task>__<embedding>/semantic_clustering_*_collapsed.csv

Run analysis.py on each pre-ML / post-ML embedding set before compare_pre_post.py.

Outputs (batch root):
    visualizations/comparisons_pre_and_post/core_tail/
    visualizations/comparisons_pre_and_post/diversity/

Example:
    python compare_pre_post.py
    python compare_pre_post.py --embeddings-root textual_analysis/.../embeddings_openai
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, ttest_rel

TEXTUAL_DIR = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = TEXTUAL_DIR.parent
for p in (SCRIPT_DIR, TEXTUAL_DIR, ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from stats_utils import bootstrap_mean_ci
from viz_style import (
    BAR_EDGE_COLOR,
    BAR_EDGE_WIDTH,
    FONT_AXIS_LABEL,
    FONT_COMPARISON,
    FONT_FOOTNOTE,
    FONT_LEGEND,
    FONT_TICK,
    FOOTNOTE_COLOR,
    FOOTNOTE_LINE_STEP,
    SAVE_DPI,
    SAVE_PAD_INCHES,
    apply_plot_style,
    draw_paired_pre_post_bracket,
    SIG_LEVEL_LEGEND,
    significance_label,
)

from analysis import (
    ANALYSIS_SEED,
    COLLAPSED_PARTICIPANT_TYPE_COL,
    COMPARISONS_CORE_TAIL_SUBDIR,
    COMPARISONS_DIVERSITY_SUBDIR,
    DEFAULT_EMBEDDING_COLUMNS,
    DEFAULT_EMBEDDINGS_ROOT,
    EMBEDDING_SET_PART_LABELS,
    PARTICIPANT_NAME_COL,
    PARTICIPANT_TYPE_COL,
    SEMANTIC_CLUSTERING_PARTICIPANT_CSV,
    SEMANTIC_CLUSTERING_SUMMARY_CSV,
    available_embedding_columns,
    comparisons_pre_post_dir,
    infer_embeddings_root,
    resolve_task_data_dir,
    run_diversity_prediction_comparison,
    safe_name,
)

HUMAN_TYPES = ("student", "expert")
GENAI_TYPE = "GenAI"

TASK_PANEL_ORDER = [
    "race/main-effects",
    "race/soi",
    "gender/main-effects",
    "gender/soi",
]

PREDICTION1_FOOTNOTE = (
    "Metric: % in semantic tail (HDBSCAN outliers within collapsed Humans group).",
    "One-sided paired t-test on tail % (directional: post < pre).",
    SIG_LEVEL_LEGEND,
)

SUPTITLE_FONTSIZE = 26
SUPTITLE_LINE_SPACING = 1.45
PREDICTION1_FOOTNOTE_FONTSIZE = FONT_FOOTNOTE + 7
PREDICTION1_FOOTNOTE_LINE_STEP = FOOTNOTE_LINE_STEP * 1.2
PANEL_TITLE_FONTSIZE = 18.0
PANEL_XTICK_FONTSIZE = 18.0
PANEL_YTICK_FONTSIZE = FONT_TICK + 5
PANEL_YLABEL_FONTSIZE = FONT_AXIS_LABEL + 8
PREDICTION1_LEGEND_FONTSIZE = FONT_LEGEND + 5
PREDICTION1_BRACKET_FONTSIZE = FONT_COMPARISON + 5
LEGEND_BBOX_Y = 0.895
PANEL_ROW_GAP = 0.30
PANEL_COL_GAP = 0.30
PANEL_BOX_ASPECT = 0.96
FIGSIZE = (10.8, 12.8)
PANEL_YLIM_TOP_DEFAULT = 100.0
PANEL_YTICK_MAX_DEFAULT = 100.0
PANEL_YTICK_STEP = 20.0

PRE_BAR_COLOR = "#4C72B0"
POST_BAR_COLOR = "#DD8452"

AUDIENCE_CONFIGS = {
    "human": {
        "csv": "human_core_tail_pre_post_by_task.csv",
        "fig": "human_tail_pct_pre_post_by_task.png",
        "audience_group": "Human",
        "suptitle": (
            "Core-tail structure of Humans' theoretical explanations\n"
            "(pre-data v.s. post-data)"
        ),
        "ylabel": "Tail share among Humans (%)",
        "count_label": "human",
        "ylim_top": 120.0,
        "ytick_max": 100.0,
    },
    "genai": {
        "csv": "genai_core_tail_pre_post_by_task.csv",
        "fig": "genai_tail_pct_pre_post_by_task.png",
        "audience_group": GENAI_TYPE,
        "suptitle": (
            "Core-tail structure of GenAI-generated theoretical explanations\n"
            "(pre-data v.s. post-data)"
        ),
        "ylabel": "Tail share among GenAI respondents (%)",
        "count_label": "genai",
        "ylim_top": PANEL_YLIM_TOP_DEFAULT,
        "ytick_max": PANEL_YTICK_MAX_DEFAULT,
    },
}

apply_plot_style()


def format_task_part(part: str) -> str:
    key = part.lower()
    if key in EMBEDDING_SET_PART_LABELS:
        return EMBEDDING_SET_PART_LABELS[key]
    return part.replace("-", " ").title().replace("Ml", "ML")


def task_label_from_key(task_key: str) -> str:
    return " · ".join(format_task_part(part) for part in task_key.split("/"))


def discover_task_pairs(embeddings_root: Path) -> List[Tuple[Path, Path, str]]:
    """Return (pre_dir, post_dir, task_key) for each topic/task with both phases."""
    pairs: List[Tuple[Path, Path, str]] = []
    for pre_dir in sorted(embeddings_root.rglob("pre-ML")):
        if not (pre_dir / "embeddings_wide.parquet").exists():
            continue
        post_dir = pre_dir.parent / "post-ML"
        if not (post_dir / "embeddings_wide.parquet").exists():
            continue
        task_key = str(pre_dir.relative_to(embeddings_root).parent)
        pairs.append((pre_dir, post_dir, task_key))
    if not pairs:
        raise FileNotFoundError(
            f"No pre-ML/post-ML pairs found under {embeddings_root}."
        )
    return pairs


def clustering_analysis_dir(
    embedding_set_dir: Path,
    embedding_col: str,
    embeddings_root: Path,
) -> Path:
    return resolve_task_data_dir(
        embeddings_root, embedding_set_dir, embedding_col
    )


def clustering_csv_paths(
    embedding_set_dir: Path,
    embedding_col: str,
    embeddings_root: Path,
) -> dict[str, Path]:
    space_dir = clustering_analysis_dir(
        embedding_set_dir, embedding_col, embeddings_root
    )
    collapsed = "_collapsed"
    return {
        "summary": space_dir / SEMANTIC_CLUSTERING_SUMMARY_CSV.format(suffix=collapsed),
        "participant": space_dir / SEMANTIC_CLUSTERING_PARTICIPANT_CSV.format(
            suffix=collapsed
        ),
    }


def require_analysis_csvs(paths: dict[str, Path], embedding_set_dir: Path) -> None:
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing analysis clustering CSV(s). Run analysis.py on this embedding set first:\n"
            f"  {embedding_set_dir}\n"
            + "\n".join(f"  - {p}" for p in missing)
        )


def load_group_summary_row(csv_path: Path, audience_group: str) -> dict:
    summary_df = pd.read_csv(csv_path)
    match = summary_df.loc[summary_df["participant_type"] == audience_group]
    if match.empty:
        raise ValueError(
            f"No row for {audience_group!r} in {csv_path.name} "
            f"(available: {summary_df['participant_type'].tolist()})"
        )
    return match.iloc[0].to_dict()


def load_participant_clustering(csv_path: Path, audience_group: str) -> pd.DataFrame:
    participant_df = pd.read_csv(csv_path)
    group_col = COLLAPSED_PARTICIPANT_TYPE_COL
    if group_col not in participant_df.columns:
        raise ValueError(f"Expected column {group_col!r} in {csv_path}")
    subset = participant_df.loc[
        participant_df[group_col] == audience_group
    ].copy()
    if subset.empty:
        raise ValueError(f"No {audience_group} rows in {csv_path}")
    return subset


def participant_type_lookup(embedding_set_dir: Path) -> pd.DataFrame:
    df = pd.read_parquet(embedding_set_dir / "embeddings_wide.parquet")
    return df[[PARTICIPANT_NAME_COL, PARTICIPANT_TYPE_COL]].drop_duplicates()


def paired_cohens_d(pre: np.ndarray, post: np.ndarray) -> float:
    mask = np.isfinite(pre) & np.isfinite(post)
    pre = pre[mask]
    post = post[mask]
    if len(pre) < 2:
        return np.nan
    diff = post - pre
    sd = float(np.std(diff, ddof=1))
    if sd < 1e-12:
        return np.nan
    return float(np.mean(diff) / sd)


def p_value_paired_one_sided_less(pre: np.ndarray, post: np.ndarray) -> float:
    """H1: post < pre (e.g. tail indicator lower post-data)."""
    mask = np.isfinite(pre) & np.isfinite(post)
    pre = pre[mask]
    post = post[mask]
    if len(pre) < 2:
        return np.nan
    try:
        return float(
            ttest_rel(pre, post, alternative="greater").pvalue
        )
    except Exception:
        return np.nan


def analyze_prediction1_task(
    pre_dir: Path,
    post_dir: Path,
    task_key: str,
    embedding_col: str,
    embeddings_root: Path,
    *,
    audience_group: str,
    count_label: str,
) -> dict:
    pre_paths = clustering_csv_paths(pre_dir, embedding_col, embeddings_root)
    post_paths = clustering_csv_paths(post_dir, embedding_col, embeddings_root)
    require_analysis_csvs(pre_paths, pre_dir)
    require_analysis_csvs(post_paths, post_dir)

    pre_sum = load_group_summary_row(pre_paths["summary"], audience_group)
    post_sum = load_group_summary_row(post_paths["summary"], audience_group)

    pre_tbl = load_participant_clustering(pre_paths["participant"], audience_group)
    post_tbl = load_participant_clustering(post_paths["participant"], audience_group)

    merged = pre_tbl.merge(
        post_tbl,
        on=[PARTICIPANT_NAME_COL],
        how="inner",
        suffixes=("_pre", "_post"),
    )
    if merged.empty:
        raise ValueError(f"No matched {count_label} participants for task {task_key}")

    if audience_group == "Human":
        types = participant_type_lookup(pre_dir)
        types = types[types[PARTICIPANT_TYPE_COL].isin(HUMAN_TYPES)]
        merged = merged.merge(types, on=PARTICIPANT_NAME_COL, how="left")
    else:
        merged[PARTICIPANT_TYPE_COL] = GENAI_TYPE

    pre_tail = merged["is_tail_pre"].astype(float).values
    post_tail = merged["is_tail_post"].astype(float).values
    tail_diff = post_tail - pre_tail

    n_became_core = int(((pre_tail == 1) & (post_tail == 0)).sum())
    n_became_tail = int(((pre_tail == 0) & (post_tail == 1)).sum())
    n_unchanged_tail = int(((pre_tail == 1) & (post_tail == 1)).sum())
    n_unchanged_core = int(((pre_tail == 0) & (post_tail == 0)).sum())

    paired_p_two_sided = float(
        ttest_rel(pre_tail, post_tail, alternative="two-sided").pvalue
    )
    paired_p_one_sided = p_value_paired_one_sided_less(pre_tail, post_tail)
    diff_lo, diff_hi = bootstrap_mean_ci(tail_diff, seed=ANALYSIS_SEED)

    post_still_core_tail = bool(post_sum["core_pct"] > 0 and post_sum["tail_pct"] > 0)
    tail_shorter_post = bool(post_sum["tail_pct"] < pre_sum["tail_pct"])

    post_tail_binom = binomtest(
        int(post_sum["tail_n"]),
        n=int(post_sum["n"]),
        p=0.0,
        alternative="greater",
    )

    return {
        "task_key": task_key,
        "task_label": task_label_from_key(task_key),
        "embedding_col": embedding_col,
        "audience_group": audience_group,
        f"n_{count_label}_paired": len(merged),
        "n_student": int((merged[PARTICIPANT_TYPE_COL] == "student").sum()),
        "n_expert": int((merged[PARTICIPANT_TYPE_COL] == "expert").sum()),
        "n_genai": int((merged[PARTICIPANT_TYPE_COL] == GENAI_TYPE).sum()),
        "pre_core_pct": float(pre_sum["core_pct"]),
        "pre_tail_pct": float(pre_sum["tail_pct"]),
        "post_core_pct": float(post_sum["core_pct"]),
        "post_tail_pct": float(post_sum["tail_pct"]),
        "tail_pct_delta_post_minus_pre": float(post_sum["tail_pct"] - pre_sum["tail_pct"]),
        "post_still_has_core_and_tail": post_still_core_tail,
        "post_tail_shorter_than_pre": tail_shorter_post,
        "post_tail_binom_p": float(post_tail_binom.pvalue),
        "paired_tail_p_two_sided": paired_p_two_sided,
        "paired_tail_p_one_sided_post_lt_pre": paired_p_one_sided,
        "paired_tail_significance_two_sided": significance_label(paired_p_two_sided),
        "paired_tail_significance_directional": significance_label(paired_p_one_sided),
        "paired_cohens_d_post_minus_pre": paired_cohens_d(pre_tail, post_tail),
        "paired_tail_diff_ci_low": diff_lo,
        "paired_tail_diff_ci_high": diff_hi,
        "n_became_core_pre_tail_to_post_core": n_became_core,
        "n_became_tail_pre_core_to_post_tail": n_became_tail,
        "n_stayed_tail": n_unchanged_tail,
        "n_stayed_core": n_unchanged_core,
        "pre_n_clusters": int(pre_sum["n_clusters"]),
        "post_n_clusters": int(post_sum["n_clusters"]),
    }


def draw_prediction1_footnote(fig, y: float = 0.048) -> None:
    for i, line in enumerate(PREDICTION1_FOOTNOTE):
        fig.text(
            0.5,
            y - i * PREDICTION1_FOOTNOTE_LINE_STEP,
            line,
            ha="center",
            va="bottom",
            fontsize=PREDICTION1_FOOTNOTE_FONTSIZE,
            color=FOOTNOTE_COLOR,
            transform=fig.transFigure,
            clip_on=False,
        )


def apply_panel_ylim(ax, *, ylim_top: float, ytick_max: float) -> None:
    """Extend ymax for bracket headroom; show tick labels only up to ytick_max."""
    ax.set_ylim(0.0, ylim_top)
    ticks = np.arange(0.0, ytick_max + 0.1, PANEL_YTICK_STEP)
    ax.set_yticks(ticks)


def plot_prediction1_bars(
    summary_df: pd.DataFrame,
    outpath: Path,
    *,
    suptitle: str,
    ylabel: str,
    ylim_top: float = PANEL_YLIM_TOP_DEFAULT,
    ytick_max: float = PANEL_YTICK_MAX_DEFAULT,
) -> None:
    order = {key: i for i, key in enumerate(TASK_PANEL_ORDER)}
    plot_df = summary_df.sort_values(
        "task_key",
        key=lambda s: s.map(order),
    ).reset_index(drop=True)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=FIGSIZE,
        gridspec_kw={"hspace": PANEL_ROW_GAP, "wspace": PANEL_COL_GAP},
    )
    axes_flat = axes.ravel()
    bar_x = np.array([0.0, 1.0])
    bar_width = 0.55

    for ax, (_, row) in zip(axes_flat, plot_df.iterrows()):
        pre_val = float(row["pre_tail_pct"])
        post_val = float(row["post_tail_pct"])

        ax.bar(
            bar_x[0],
            pre_val,
            bar_width,
            label="Pre-data",
            color=PRE_BAR_COLOR,
            edgecolor=BAR_EDGE_COLOR,
            linewidth=BAR_EDGE_WIDTH,
        )
        ax.bar(
            bar_x[1],
            post_val,
            bar_width,
            label="Post-data",
            color=POST_BAR_COLOR,
            edgecolor=BAR_EDGE_COLOR,
            linewidth=BAR_EDGE_WIDTH,
        )
        ax.set_xticks(bar_x)
        ax.set_xticklabels(["Pre-data", "Post-data"], fontsize=PANEL_XTICK_FONTSIZE)
        ax.set_title(
            row["task_label"],
            fontweight="bold",
            fontsize=PANEL_TITLE_FONTSIZE,
            pad=8,
        )
        ax.tick_params(axis="x", labelsize=PANEL_XTICK_FONTSIZE)
        ax.tick_params(axis="y", labelsize=PANEL_YTICK_FONTSIZE)
        apply_panel_ylim(ax, ylim_top=ylim_top, ytick_max=ytick_max)
        ax.set_box_aspect(PANEL_BOX_ASPECT)
        ax.grid(axis="y", alpha=0.25)

    for ax, (_, row) in zip(axes_flat, plot_df.iterrows()):
        pre_val = float(row["pre_tail_pct"])
        post_val = float(row["post_tail_pct"])
        draw_paired_pre_post_bracket(
            ax,
            bar_x[0],
            bar_x[1],
            max(pre_val, post_val),
            float(row["paired_tail_p_one_sided_post_lt_pre"]),
            fontsize=PREDICTION1_BRACKET_FONTSIZE,
        )

    fig.supylabel(
        ylabel,
        fontweight="bold",
        x=0.04,
        fontsize=PANEL_YLABEL_FONTSIZE,
    )

    footnote_lines = len(PREDICTION1_FOOTNOTE)
    fig.subplots_adjust(
        left=0.11,
        right=0.98,
        top=0.80,
        bottom=0.08 + footnote_lines * PREDICTION1_FOOTNOTE_LINE_STEP,
        hspace=PANEL_ROW_GAP,
    )
    title = fig.suptitle(
        suptitle,
        fontweight="bold",
        fontsize=SUPTITLE_FONTSIZE,
        y=0.975,
    )
    title.set_linespacing(SUPTITLE_LINE_SPACING)
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=True,
        fontsize=PREDICTION1_LEGEND_FONTSIZE,
        bbox_to_anchor=(0.5, LEGEND_BBOX_Y),
        borderaxespad=0.0,
    )
    draw_prediction1_footnote(fig)
    fig.savefig(outpath, dpi=SAVE_DPI, pad_inches=SAVE_PAD_INCHES)
    plt.close(fig)


def run_prediction1(
    embeddings_root: Path,
    outdir: Path,
    embedding_col: str,
    audience_cfg: dict,
) -> pd.DataFrame:
    count_label = audience_cfg["count_label"]
    rows = []
    for pre_dir, post_dir, task_key in discover_task_pairs(embeddings_root):
        print(f"\n=== {task_label_from_key(task_key)} ({count_label}) ===")
        row = analyze_prediction1_task(
            pre_dir,
            post_dir,
            task_key,
            embedding_col,
            embeddings_root,
            audience_group=audience_cfg["audience_group"],
            count_label=count_label,
        )
        rows.append(row)
        print(
            f"  Tail%: pre {row['pre_tail_pct']:.1f}% "
            f"→ post {row['post_tail_pct']:.1f}% "
            f"(Δ {row['tail_pct_delta_post_minus_pre']:+.1f} pp)"
        )
        print(
            f"  Post still core+tail: {row['post_still_has_core_and_tail']} | "
            f"Post tail shorter: {row['post_tail_shorter_than_pre']} | "
            f"Paired directional p: {row['paired_tail_p_one_sided_post_lt_pre']:.4f} "
            f"{row['paired_tail_significance_directional']}"
        )

    summary_df = pd.DataFrame(rows)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / audience_cfg["csv"]
    summary_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    plot_prediction1_bars(
        summary_df,
        outdir / audience_cfg["fig"],
        suptitle=audience_cfg["suptitle"],
        ylabel=audience_cfg["ylabel"],
        ylim_top=float(audience_cfg.get("ylim_top", PANEL_YLIM_TOP_DEFAULT)),
        ytick_max=float(audience_cfg.get("ytick_max", PANEL_YTICK_MAX_DEFAULT)),
    )
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {outdir / audience_cfg['fig']}")
    return summary_df


def run_all_prediction1(
    embeddings_root: Path,
    embedding_col: str,
) -> None:
    outdir = comparisons_pre_post_dir(embeddings_root, COMPARISONS_CORE_TAIL_SUBDIR)
    for audience_cfg in AUDIENCE_CONFIGS.values():
        print(f"\n{'=' * 72}")
        print(f"Audience: {audience_cfg['count_label']}")
        print(f"Output: {outdir}")
        run_prediction1(embeddings_root, outdir, embedding_col, audience_cfg)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Q5 apple-to-apple pre/post comparisons by task."
    )
    parser.add_argument(
        "--embeddings-root",
        type=Path,
        default=DEFAULT_EMBEDDINGS_ROOT,
        help="Root folder containing topic/task/pre-ML and post-ML sets.",
    )
    parser.add_argument(
        "--embedding-col",
        default=DEFAULT_EMBEDDING_COLUMNS[0],
        help="Embedding column to use (default: raw 3072d).",
    )
    args = parser.parse_args()

    embeddings_root = args.embeddings_root.expanduser().resolve()
    if not embeddings_root.is_absolute():
        embeddings_root = (Path.cwd() / embeddings_root).resolve()

    sample_pair = discover_task_pairs(embeddings_root)[0]
    sample_df = pd.read_parquet(sample_pair[0] / "embeddings_wide.parquet")
    embedding_col = available_embedding_columns(sample_df, [args.embedding_col])[0]

    print(f"Embeddings root: {embeddings_root}")
    run_all_prediction1(embeddings_root, embedding_col)

    print(f"\n{'=' * 72}")
    print("Diversity pre/post comparisons (Humans vs GenAI)")
    try:
        run_diversity_prediction_comparison(embeddings_root, embedding_col)
        diversity_out = comparisons_pre_post_dir(
            embeddings_root, COMPARISONS_DIVERSITY_SUBDIR
        )
        print(f"Diversity outputs: {diversity_out}")
    except FileNotFoundError as exc:
        print(f"Skipping diversity figures: {exc}")
        print("Run analysis.py on all embedding sets first to generate CSV tables.")


if __name__ == "__main__":
    main()
