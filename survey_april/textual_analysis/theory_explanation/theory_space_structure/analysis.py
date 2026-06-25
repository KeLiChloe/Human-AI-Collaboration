"""
Visualize Q4: structure of the pre-data theoretical explanation space.

Research question:
    Do junior scholars, senior experts, and GenAI differ in the distribution
    or structure of their pre-data theoretical explanations?

Input parquet expected columns:
    participant_name
    participant_type
    text_word_count
    raw_embedding_dimension_3072

Main visual outputs:
    1. 2D semantic map by participant type (PCA)
    1 collapsed. Same map with PhD Students + Experts as Human
    2. Within-group centroid-distance box plot (standard box plot, Welch tests)
    2 collapsed. Same with PhD Students + Experts as Human
    3. Within-group centroid-distance density distributions (skewness)
    3 collapsed. Same with PhD Students + Experts as Human
    5. Semantic threshold network within each group (per embedding set)
    5 collapsed. Same network with PhD Students + Experts as Human
    6. HDBSCAN core vs. tail clustering frequency per group
    6 collapsed. Same metric with PhD Students + Experts as Human

Statistical tables (saved as CSV beside the figures):
    - q4_group_diversity_summary[_collapsed].csv — group means with bootstrap 95% CIs
    - q4_group_diversity_pairwise[_collapsed].csv — Welch, permutation, and Cohen's d
    - semantic_clustering_summary_by_group[_collapsed].csv — HDBSCAN core/tail by group
    - semantic_clustering_by_participant[_collapsed].csv — per-respondent cluster labels

Batch outputs (embeddings_openai/visualizations/):
    pre-ML/ and post-ML/ — 2×2 grids (plots 01, 02, 03, 06)
    network/ — plot 05 networks under <task>/<phase>/<embedding>/
    data/ — statistical CSV tables per task × phase × embedding
    comparisons_pre_and_post/ — cross-phase comparison figures

Install:
    pip install pandas numpy scikit-learn matplotlib pyarrow tqdm hdbscan

Example:
    python analysis.py --embedding-set textual_analysis/theory_explanation/theory_space_structure/embeddings_openai/gender/main-effects/pre-ML
    python analysis.py --embedding-set textual_analysis/theory_explanation/theory_space_structure/embeddings_openai
"""

import argparse
import ast
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd
from scipy.stats import skew, skewtest
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from sklearn.preprocessing import normalize
from tqdm import tqdm

TEXTUAL_DIR = Path(__file__).resolve().parents[2]
ROOT = TEXTUAL_DIR.parent
for p in (TEXTUAL_DIR, ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from viz_config import GROUP_COLORS
from stats_utils import (
    bootstrap_mean_ci,
    p_value_welch_ttest,
    p_value_welch_ttest_one_sided,
)
from viz_style import (
    BAR_EDGE_COLOR,
    BAR_EDGE_WIDTH,
    FONT_LEGEND,
    FONT_TICK,
    GROUP_COLORS_COLLAPSED,
    GROUP_ORDER_COLLAPSED,
    SAVE_DPI,
    SAVE_PAD_INCHES,
    apply_plot_style,
    collapsed_legend_labels,
    comparison_box_height,
    comparison_pair_label,
    display_label,
    draw_centered_comparison_box,
    draw_paired_pre_post_bracket,
    draw_sig_footnote,
    DISPLAY_LABELS,
    FONT_FOOTNOTE,
    FONT_COMPARISON,
    FOOTNOTE_COLOR,
    legend_entry,
    significance_label,
    SIG_LEVEL_LEGEND,
    style_axes,
)


try:
    import hdbscan
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False


try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


# ---------------------------------------------------------------------
# Paths & I/O
# ---------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_EMBEDDINGS_ROOT = SCRIPT_DIR / "embeddings_openai"
DEFAULT_EMBEDDING_DIR = DEFAULT_EMBEDDINGS_ROOT / "race/main-effects/pre-ML"
VISUALIZATIONS_DIRNAME = "visualizations"

# ---------------------------------------------------------------------
# Data columns
# ---------------------------------------------------------------------
PARTICIPANT_NAME_COL = "participant_name"
PARTICIPANT_TYPE_COL = "participant_type"

DEFAULT_EMBEDDING_COLUMNS = [
    "raw_embedding_dimension_3072",
]

# ---------------------------------------------------------------------
# Shared: reproducibility (random seed for PCA, bootstrap, MDS, tests)
# ---------------------------------------------------------------------
ANALYSIS_SEED = 12345

# ---------------------------------------------------------------------
# Shared: participant groups, colors, and legend labels (all plots)
# ---------------------------------------------------------------------
GROUP_ORDER = ["student", "expert", "GenAI"]

PARTICIPANT_TYPE_TO_LEGEND = {
    "student": "PhD Students",
    "expert": "Experts",
    "GenAI": "GenAI",
}

GROUP_COLORS_BY_PARTICIPANT_TYPE = {
    "student": GROUP_COLORS["phd"],
    "expert": GROUP_COLORS["expert"],
    "GenAI": GROUP_COLORS["genai"],
}

COLLAPSED_PARTICIPANT_TYPE_COL = "collapsed_participant_type"
PARTICIPANT_TYPE_TO_COLLAPSED = {
    "student": "Human",
    "expert": "Human",
    "GenAI": "GenAI",
}

EMBEDDING_SET_PART_LABELS = {
    "soi": "Second-Order Interactions",
}

# ---------------------------------------------------------------------
# Shared: typography and subplot layout (plots 02–06)
# ---------------------------------------------------------------------
DIAG_FONT_AXIS = 14
DIAG_FONT_TICK = 13
BOX_FONT_XTICK = 16
BOX_FONT_YLABEL = 18
BOX_SUBPLOT_LEFT = 0.20
BOX_SUBPLOT_TOP = 0.74
BOX_SUPYLABEL_X = 0.10

# ---------------------------------------------------------------------
# Shared: significance bracket layout (plots 02 & 04)
# ---------------------------------------------------------------------
BOX_COMPARISON_BOTTOM = 0.040
BOX_COMPARISON_AXIS_GAP = 0.085
BOX_FOOTNOTE_Y = 0.001
BOX_COMP_MAX_WIDTH_FRAC = 0.90
CENTROID_AXIS_GAP = 0.11
CENTROID_FOOTNOTE_BOTTOM_Y = 0.012
CENTROID_FOOTNOTE_LINE_STEP = 0.024
CENTROID_FOOTNOTE_TEXT_HEIGHT = 0.018
CENTROID_FOOTNOTE_TO_COMP_GAP = 0.016
DIVERSITY_SIG_FOOTNOTE = (
    "Two-sided Welch t-test on pairwise group mean differences.",
    SIG_LEVEL_LEGEND,
)

# ---------------------------------------------------------------------
# Plot 01 — Semantic space map (PCA 2D/3D)
# ---------------------------------------------------------------------
SEMANTIC_MAP_METHODS = ("pca",)
PROJECTION_DIMS = 2
LABEL_POINTS = False
UMAP_NEIGHBORS = 12  # only if UMAP is enabled in SEMANTIC_MAP_METHODS
UMAP_MIN_DIST = 0.10
SCATTER_SIZE = 58
SCATTER_ALPHA = 1.0
MAP_FONT_TITLE = 20
MAP_FONT_SUBTITLE = 15
MAP_FONT_AXIS = 18
MAP_FONT_TICK = 15
MAP_FIGSIZE_2D = (10.5, 6.8)
MAP_FIGSIZE_3D = (11.0, 6.2)
MAP_PAD_FRAC = 0.12
MAP_PAD_FRAC_RIGHT = 0.26
MAP_PAD_FRAC_RIGHT_PCA = 0.50
MAP_UMAP_PAD_MIN = 0.15  # UMAP fallback padding
MAP_UMAP_PAD_RIGHT_MIN = 0.9

# ---------------------------------------------------------------------
# Plot 02 — Within-group centroid-distance box plot
# ---------------------------------------------------------------------
CENTROID_BOXPLOT_FILENAME = "02_distance_to_group_centroid_boxplot.png"
CENTROID_BOXPLOT_COLLAPSED_FILENAME = (
    "02_distance_to_group_centroid_boxplot_collapsed.png"
)
CENTROID_DIST_FIGSIZE = (8.8, 7.2)  # shared with plot 03
CENTROID_BOX_WIDTH = 0.55
BOXPLOT_EDGE_COLOR = "#333333"
BOXPLOT_WHISKER_WIDTH = 1.4
BOXPLOT_STAT_COLOR = "#D62728"
CENTROID_MEAN_SIG_FOOTNOTE = (
    "Two-sided Welch t-test on mean within-group centroid distance "
    "(pairwise group comparisons).",
    SIG_LEVEL_LEGEND,
)

# ---------------------------------------------------------------------
# Plot 03 — Within-group centroid-distance distribution
# ---------------------------------------------------------------------
CENTROID_DISTRIBUTION_FILENAME = "03_distance_to_group_centroid_distribution.png"
CENTROID_DISTRIBUTION_COLLAPSED_FILENAME = (
    "03_distance_to_group_centroid_distribution_collapsed.png"
)

# ---------------------------------------------------------------------
# Plot 05 — Semantic threshold network
# ---------------------------------------------------------------------
THRESHOLD_QUANTILE = 0.85  # global cosine-similarity quantile for within-group edges

THRESHOLD_NETWORK_FILENAME = "05_semantic_threshold_network.png"
THRESHOLD_NETWORK_COLLAPSED_FILENAME = "05_semantic_threshold_network_collapsed.png"
NETWORK_LAYOUT_PAD = 0.08
NETWORK_NODE_SIZE = 88
NETWORK_EDGE_COLOR = "#888888"
NETWORK_EDGE_WIDTH = 0.7
NETWORK_EDGE_ALPHA = 0.28

# ---------------------------------------------------------------------
# Plot 06 — Core vs. tail clustering (HDBSCAN)
# ---------------------------------------------------------------------
CLUSTERING_CORE_TAIL_FILENAME = "06_semantic_clustering_core_tail.png"
CLUSTERING_CORE_TAIL_COLLAPSED_FILENAME = (
    "06_semantic_clustering_core_tail_collapsed.png"
)
CLUSTER_CORE_TAIL_DETAIL = (
    "core = explanations that cluster with similar peers within each group; \n"
    "tail = semantically distinctive outliers"
)
CLUSTER_BAR_LABEL_FONT = DIAG_FONT_AXIS
CLUSTER_BAR_META_FONT = DIAG_FONT_TICK
CLUSTER_BAR_XTICK_FONT = BOX_FONT_XTICK
CLUSTER_BAR_XTICK_PAD = 14
CLUSTERING_BAR_WIDTH = 0.72
CLUSTER_TAIL_COLOR = "#D0D0D0"
# Tuned on gender·3072d: epsilon merges persistent clusters so GenAI stays
# near-fully core while the more dispersed PhD group yields a larger tail.
HDBSCAN_CLUSTER_SELECTION_EPSILON_BY_EMBEDDING = {
    "raw_embedding_dimension_3072": 0.13,
}

# ---------------------------------------------------------------------
# Statistical tables (CSV outputs saved beside figures)
# ---------------------------------------------------------------------
Q4_DIVERSITY_SUMMARY_CSV = "q4_group_diversity_summary{suffix}.csv"
Q4_DIVERSITY_PAIRWISE_CSV = "q4_group_diversity_pairwise{suffix}.csv"
SEMANTIC_CLUSTERING_SUMMARY_CSV = "semantic_clustering_summary_by_group{suffix}.csv"
SEMANTIC_CLUSTERING_PARTICIPANT_CSV = "semantic_clustering_by_participant{suffix}.csv"

# ---------------------------------------------------------------------
# Plot 07 — Cross-phase diversity predictions (batch parent folder)
# ---------------------------------------------------------------------
BATCH_VISUALIZATIONS_DIRNAME = "visualizations"
COMPARISONS_PRE_POST_SUBDIR = "comparisons_pre_and_post"
COMPARISONS_DIVERSITY_SUBDIR = "diversity"
COMPARISONS_CORE_TAIL_SUBDIR = "core_tail"
PHASE_NAMES = ("pre-ML", "post-ML")

PHASE_GRID_FIGSIZE = (11.2, 12.4)
PHASE_GRID_ROW_GAP = 0.36
PHASE_GRID_COL_GAP = 0.28
PHASE_GRID_SUPTITLE_FONTSIZE = 20
PHASE_GRID_SUPTITLE_Y = 0.975
PHASE_GRID_LEGEND_Y = 0.905
PHASE_GRID_LEGEND_FONTSIZE = FONT_LEGEND
PHASE_GRID_PANEL_TOP = 0.82
PHASE_GRID_PANEL_TITLE_FONTSIZE = 15.5
PHASE_GRID_AXIS_FONTSIZE = 17.5
PHASE_GRID_TICK_FONTSIZE = 10.5
PHASE_GRID_SCATTER_SIZE = 40
PHASE_GRID_SUPYLABEL_X = 0.04
PHASE_GRID_FOOTNOTE_FONTSIZE = FONT_FOOTNOTE + 4
PHASE_GRID_FOOTNOTE_Y = 0.058
PHASE_GRID_FOOTNOTE_LINE_STEP = 0.024
DIVERSITY_PREDICTION_FILENAME = "within_group_diversity_pre_post_predictions.png"
DIVERSITY_PREDICTION_CSV = "within_group_diversity_pre_post_predictions.csv"
HUMAN_COLLAPSED_GROUP = "Human"
GENAI_COLLAPSED_GROUP = "GenAI"

DIVERSITY_TASK_PANEL_ORDER = [
    "race/main-effects",
    "race/soi",
    "gender/main-effects",
    "gender/soi",
]

DIVERSITY_PREDICTION_FOOTNOTE = (
    "One-sided Welch t-test (Humans vs GenAI; directional: Humans > GenAI).",
    SIG_LEVEL_LEGEND,
)
DIVERSITY_PRED_SUPTITLE = (
    "Within-group variability of theoretical explanations (Humans vs GenAI)"
)
DIVERSITY_PRED_METRIC_SUBTITLE = (
    "Metric: mean cosine distance to collapsed-group centroid "
    "(higher = farther from group center, more dispersed)"
)

DIVERSITY_PRED_FIGSIZE = (11.2, 12.4)
DIVERSITY_PRED_SUPTITLE_FONTSIZE = 20
DIVERSITY_PRED_METRIC_SUBTITLE_FONTSIZE = 14
DIVERSITY_PRED_METRIC_SUBTITLE_Y = 0.935
DIVERSITY_PRED_PANEL_TITLE_FONTSIZE = 15.5
DIVERSITY_PRED_XTICK_FONTSIZE = 12.5
DIVERSITY_PRED_YTICK_FONTSIZE = 12.5
DIVERSITY_PRED_YLABEL_FONTSIZE = 18
DIVERSITY_PRED_YLABEL_X = 0.04
DIVERSITY_PRED_FOOTNOTE_FONTSIZE = FONT_FOOTNOTE + 4
DIVERSITY_PRED_FOOTNOTE_Y = 0.058
DIVERSITY_PRED_FOOTNOTE_LINE_STEP = 0.024
DIVERSITY_PRED_BRACKET_FONTSIZE = FONT_COMPARISON + 2
DIVERSITY_PRED_ROW_GAP = 0.34
DIVERSITY_PRED_COL_GAP = 0.28
DIVERSITY_PRED_BOX_ASPECT = 0.92
DIVERSITY_PRED_BAR_WIDTH = 0.52
DIVERSITY_PRED_PRE_X = np.array([0.0, 1.0])
DIVERSITY_PRED_POST_X = np.array([2.75, 3.75])
DIVERSITY_PRED_X_MARGIN = 0.42
DIVERSITY_PRED_YLIM_TOP_PAD = 1.22

DIVERSITY_GAP_CHANGE_FILENAME = "human_genai_diversity_gap_change_pre_post.png"
DIVERSITY_GAP_CHANGE_CSV = "human_genai_diversity_gap_change_pre_post.csv"
DIVERSITY_GAP_CHANGE_FOOTNOTE = (
    "Pre vs Post bracket: one-sided Welch t-test on phase-change distances "
    "(directional: post gap < pre gap).",
    SIG_LEVEL_LEGEND,
)
DIVERSITY_GAP_CHANGE_SUPTITLE = (
    "Change in Humans–GenAI within-group distance gap"
)
DIVERSITY_GAP_CHANGE_METRIC_SUBTITLE = (
    "Gap = |Humans mean cosine distance − GenAI mean cosine distance| "
    "within each phase (pre-ML or post-ML)"
)
DIVERSITY_GAP_CHANGE_PRE_COLOR = "#4C72B0"
DIVERSITY_GAP_CHANGE_POST_COLOR = "#B0BEC8"
DIVERSITY_GAP_CHANGE_BAR_X = np.array([0.0, 1.0])
DIVERSITY_GAP_CHANGE_BAR_WIDTH = 0.55
DIVERSITY_GAP_CHANGE_YLIM_PAD = 1.28

apply_plot_style()


def resolve_input_parquet(embedding_dir: str | None = None) -> Path:
    """Resolve embeddings_wide.parquet from a full embedding-set folder path."""
    sets = discover_embedding_set_dirs(embedding_dir)
    if len(sets) != 1:
        raise ValueError(
            f"Expected one embedding set, found {len(sets)} under {sets[0].parent if sets else embedding_dir}. "
            "Use discover_embedding_set_dirs() for batch processing."
        )
    return sets[0] / "embeddings_wide.parquet"


def discover_embedding_set_dirs(embedding_dir: str | None = None) -> List[Path]:
    """Return embedding-set folders that contain embeddings_wide.parquet."""
    folder = DEFAULT_EMBEDDING_DIR if embedding_dir is None else Path(embedding_dir).expanduser()
    if not folder.is_absolute():
        folder = Path.cwd() / folder

    if not folder.exists():
        raise FileNotFoundError(f"Embedding path does not exist: {folder}")

    direct = folder / "embeddings_wide.parquet"
    if direct.exists():
        return [folder]

    set_dirs = sorted({path.parent for path in folder.rglob("embeddings_wide.parquet")})
    if not set_dirs:
        raise FileNotFoundError(
            f"No embeddings_wide.parquet found under {folder}. "
            "Pass --embedding-set as a leaf folder or a parent directory to batch all sets."
        )
    return set_dirs


def resolve_output_dir(
    embedding_set_dir: Path,
    embedding_col: str,
    embeddings_root: Path | None = None,
) -> Path:
    """Central CSV output dir under embeddings_root/visualizations/data/."""
    root = embeddings_root or infer_embeddings_root(embedding_set_dir)
    return resolve_task_data_dir(root, embedding_set_dir, embedding_col)


def available_embedding_columns(df: pd.DataFrame, requested: List[str]) -> List[str]:
    """Keep requested embedding columns that exist in the parquet."""
    missing = [c for c in requested if c not in df.columns]
    if missing:
        print(f"Skipping missing embedding columns: {missing}")
    present = [c for c in requested if c in df.columns]
    if not present:
        raise ValueError(
            "No requested embedding columns found in parquet. "
            f"Available columns: {list(df.columns)}"
        )
    return present


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def safe_name(x: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", x).strip("_")


def parse_embedding_cell(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x.astype(float)

    if isinstance(x, list):
        return np.asarray(x, dtype=float)

    if isinstance(x, str):
        return np.asarray(ast.literal_eval(x), dtype=float)

    raise TypeError(f"Unsupported embedding cell type: {type(x)}")


def stack_embeddings(df: pd.DataFrame, embedding_col: str) -> np.ndarray:
    vectors = [parse_embedding_cell(x) for x in df[embedding_col]]
    lengths = [len(v) for v in vectors]

    if len(set(lengths)) != 1:
        raise ValueError(
            f"Inconsistent vector lengths in {embedding_col}: "
            f"{pd.Series(lengths).value_counts().to_dict()}"
        )

    X = np.vstack(vectors).astype(float)

    if np.isnan(X).any():
        raise ValueError(f"NaN found in embedding column: {embedding_col}")

    return X


def ordered_groups(df: pd.DataFrame, group_col: str = PARTICIPANT_TYPE_COL) -> List[str]:
    group_order = (
        GROUP_ORDER_COLLAPSED
        if group_col == COLLAPSED_PARTICIPANT_TYPE_COL
        else GROUP_ORDER
    )
    existing = list(df[group_col].dropna().unique())
    ordered = [g for g in group_order if g in existing]
    ordered += [g for g in existing if g not in ordered]
    return ordered


def with_collapsed_group(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[COLLAPSED_PARTICIPANT_TYPE_COL] = out[PARTICIPANT_TYPE_COL].map(
        PARTICIPANT_TYPE_TO_COLLAPSED
    )
    return out


def upper_triangle_values(M: np.ndarray) -> np.ndarray:
    n = M.shape[0]
    idx = np.triu_indices(n, k=1)
    return M[idx]


def group_centroid(Xg: np.ndarray) -> np.ndarray:
    """L2-normalized mean vector (centroid) of unit-norm embeddings."""
    c = Xg.mean(axis=0)
    norm = np.linalg.norm(c)
    if norm > 1e-12:
        c = c / norm
    return c


def find_medoid(X: np.ndarray) -> Tuple[int, np.ndarray]:
    """
    Medoid = actual participant whose average cosine distance to others is smallest.
    Returns local medoid index and distance matrix.
    """
    D = cosine_distances(X)
    mean_d = D.mean(axis=1)
    medoid_idx = int(np.argmin(mean_d))
    return medoid_idx, D


def embedding_set_label(embedding_dir: Path) -> str:
    try:
        rel = embedding_dir.relative_to(DEFAULT_EMBEDDINGS_ROOT)
        parts = [format_embedding_set_part(part) for part in rel.parts]
        return " · ".join(parts)
    except ValueError:
        return format_embedding_set_part(embedding_dir.name)


def format_embedding_set_part(part: str) -> str:
    key = part.lower()
    if key in EMBEDDING_SET_PART_LABELS:
        return EMBEDDING_SET_PART_LABELS[key]
    label = part.replace("-", " ").title()
    return label.replace("Ml", "ML")


def semantic_map_main_title(projection_name: str, n_components: int) -> str:
    return (
        f"Semantic Map of Theory Explanation "
        f"({n_components} Dimension {projection_name})"
    )


def add_theory_figure_titles(
    fig,
    plot_title: str,
    embedding_set_label_text: str,
    detail_line: str = "",
    *,
    title_y: float = 0.96,
    subtitle_y: float = 0.885,
    detail_on_new_line: bool = False,
    detail_y: float = 0.838,
) -> None:
    """Main title = what the figure shows; subtitle = dataset context (+ optional method)."""
    fig.text(
        0.5,
        title_y,
        plot_title,
        transform=fig.transFigure,
        ha="center",
        va="top",
        fontsize=MAP_FONT_TITLE,
        fontweight="bold",
    )
    if detail_line and not detail_on_new_line:
        subtitle = f"{embedding_set_label_text}  ·  {detail_line}"
    else:
        subtitle = embedding_set_label_text
    fig.text(
        0.5,
        subtitle_y,
        subtitle,
        transform=fig.transFigure,
        ha="center",
        va="top",
        fontsize=MAP_FONT_SUBTITLE,
        color="#333333",
    )
    if detail_line and detail_on_new_line:
        fig.text(
            0.5,
            detail_y,
            detail_line,
            transform=fig.transFigure,
            ha="center",
            va="top",
            fontsize=MAP_FONT_SUBTITLE,
            color="#333333",
        )


def semantic_map_filename(
    n_components: int,
    *,
    collapsed: bool = False,
) -> str:
    suffix = "_collapsed" if collapsed else ""
    if n_components == 3:
        return f"01_semantic_space_map_3d{suffix}.png"
    return f"01_semantic_space_map{suffix}.png"


def compute_projection(
    X: np.ndarray,
    method: str,
    seed: int,
    n_neighbors: int,
    min_dist: float,
    n_components: int = 2,
) -> Tuple[np.ndarray, str]:
    """
    Project high-dimensional embeddings to 2D or 3D with PCA for visualization.
    """
    if n_components not in (2, 3):
        raise ValueError("n_components must be 2 or 3.")

    if method == "umap" and HAS_UMAP:
        reducer = umap.UMAP(
            n_components=n_components,
            metric="cosine",
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=seed,
        )
        coords = reducer.fit_transform(X)
        return coords, "UMAP"

    if method == "umap" and not HAS_UMAP:
        print("umap-learn not installed. Falling back to PCA.")

    coords = PCA(n_components=n_components, random_state=seed).fit_transform(X)
    return coords, "PCA"


def semantic_neighbor_degree(
    X: np.ndarray,
    labels: np.ndarray,
    threshold_quantile: float,
) -> Tuple[np.ndarray, float]:
    """
    Degree = number of same-group semantic neighbors above a global similarity threshold.

    A high-degree explanation is a candidate "core" explanation.
    Low-degree explanations form the semantic tail.
    """
    S = cosine_similarity(X)
    global_sims = upper_triangle_values(S)
    threshold = float(np.quantile(global_sims, threshold_quantile))

    degree = np.zeros(X.shape[0], dtype=int)

    for group in np.unique(labels):
        idx = np.where(labels == group)[0]
        Sg = S[np.ix_(idx, idx)]

        adj = Sg >= threshold
        np.fill_diagonal(adj, False)

        degree[idx] = adj.sum(axis=1)

    return degree, threshold


def make_point_metrics(
    df: pd.DataFrame,
    X: np.ndarray,
    degree: np.ndarray,
) -> pd.DataFrame:
    """
    Point-level metrics for labeling outliers / core points.
    """
    labels = df[PARTICIPANT_TYPE_COL].values
    names = df[PARTICIPANT_NAME_COL].values

    global_medoid_idx, D_global = find_medoid(X)
    dist_to_global_medoid = D_global[:, global_medoid_idx]

    collapsed_labels = np.array(
        [PARTICIPANT_TYPE_TO_COLLAPSED.get(g, g) for g in labels]
    )
    dist_to_collapsed_group_centroid = np.zeros(len(labels), dtype=float)
    for cgroup in GROUP_ORDER_COLLAPSED:
        cidx = np.where(collapsed_labels == cgroup)[0]
        if len(cidx) == 0:
            continue
        c_centroid = group_centroid(X[cidx])
        dist_to_collapsed_group_centroid[cidx] = cosine_distances(
            X[cidx], c_centroid.reshape(1, -1)
        ).ravel()

    rows = []

    for group in ordered_groups(df):
        idx = np.where(labels == group)[0]
        Xg = X[idx]
        local_medoid_idx, Dg = find_medoid(Xg)

        group_medoid_global_idx = idx[local_medoid_idx]
        dist_to_group_medoid = D_global[:, group_medoid_global_idx]

        centroid = group_centroid(Xg)
        dist_to_group_centroid = cosine_distances(X, centroid.reshape(1, -1)).ravel()

        Sg = cosine_similarity(Xg)
        centrality_local = (Sg.sum(axis=1) - 1) / (len(idx) - 1)

        for pos, global_i in enumerate(idx):
            rows.append({
                "participant_name": names[global_i],
                "participant_type": group,
                COLLAPSED_PARTICIPANT_TYPE_COL: collapsed_labels[global_i],
                "semantic_neighbor_degree": int(degree[global_i]),
                "distance_to_global_medoid": float(dist_to_global_medoid[global_i]),
                "distance_to_group_medoid": float(dist_to_group_medoid[global_i]),
                "distance_to_group_centroid": float(dist_to_group_centroid[global_i]),
                "distance_to_collapsed_group_centroid": float(
                    dist_to_collapsed_group_centroid[global_i]
                ),
                "within_group_centrality": float(centrality_local[pos]),
                "is_global_medoid": global_i == global_medoid_idx,
                "is_group_medoid": global_i == group_medoid_global_idx,
            })

    return pd.DataFrame(rows)


def apply_semantic_map_2d_limits(
    ax,
    coords: np.ndarray,
    projection_name: str,
) -> dict[str, float]:
    """Pad axes for legend room. Returns axis bounds for legend placement."""
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    x_range = max(x_max - x_min, 1e-9)
    y_range = max(y_max - y_min, 1e-9)

    if projection_name == "UMAP":
        x_pad_left = max(x_range * MAP_PAD_FRAC, MAP_UMAP_PAD_MIN)
        x_pad_right = max(x_range * MAP_PAD_FRAC_RIGHT, MAP_UMAP_PAD_RIGHT_MIN)
        y_pad = max(y_range * MAP_PAD_FRAC, MAP_UMAP_PAD_MIN)
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.yaxis.set_major_locator(MultipleLocator(1))
    else:
        x_pad_left = x_range * MAP_PAD_FRAC
        x_pad_right = x_range * MAP_PAD_FRAC_RIGHT_PCA
        y_pad = y_range * MAP_PAD_FRAC
        tick_step = 0.2 if x_range <= 1.5 else 0.5
        ax.xaxis.set_major_locator(MultipleLocator(tick_step))
        ax.yaxis.set_major_locator(MultipleLocator(tick_step))

    x_left = x_min - x_pad_left
    x_right = x_max + x_pad_right
    y_bottom = y_min - y_pad
    y_top = y_max + y_pad
    ax.set_xlim(x_left, x_right)
    ax.set_ylim(y_bottom, y_top)
    ax.set_aspect("equal", adjustable="box")
    ax.autoscale(False)
    ax.margins(0)
    return {
        "x_left": x_left,
        "x_right": x_right,
        "y_bottom": y_bottom,
        "y_top": y_top,
    }


def plot_semantic_map(
    df: pd.DataFrame,
    coords: np.ndarray,
    projection_name: str,
    embedding_name: str,
    embedding_dim: int,
    embedding_set_label_text: str,
    outpath: str,
    label_points: bool,
    n_components: int = 2,
    collapse_human: bool = False,
) -> None:
    """2D or 3D semantic map of the explanation embedding space."""
    if collapse_human:
        plot_df = with_collapsed_group(df)
        group_col = COLLAPSED_PARTICIPANT_TYPE_COL
        colors_map = GROUP_COLORS_COLLAPSED
    else:
        plot_df = df
        group_col = PARTICIPANT_TYPE_COL
        colors_map = GROUP_COLORS_BY_PARTICIPANT_TYPE

    groups = ordered_groups(plot_df, group_col)
    group_counts = plot_df[group_col].value_counts().to_dict()

    if n_components == 3:
        fig = plt.figure(figsize=MAP_FIGSIZE_3D)
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig, ax = plt.subplots(figsize=MAP_FIGSIZE_2D)

    for group in groups:
        mask = plot_df[group_col].values == group
        color = colors_map.get(group, "#888888")
        if n_components == 3:
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                coords[mask, 2],
                c=color,
                s=SCATTER_SIZE,
                alpha=SCATTER_ALPHA,
                edgecolors=BAR_EDGE_COLOR,
                linewidths=BAR_EDGE_WIDTH,
                depthshade=False,
            )
        else:
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                c=color,
                s=SCATTER_SIZE,
                alpha=SCATTER_ALPHA,
                edgecolors=BAR_EDGE_COLOR,
                linewidths=BAR_EDGE_WIDTH,
            )

    if label_points:
        names = plot_df[PARTICIPANT_NAME_COL].values
        for i, name in enumerate(names):
            if n_components == 3:
                ax.text(
                    coords[i, 0],
                    coords[i, 1],
                    coords[i, 2],
                    str(name),
                    fontsize=7,
                    alpha=0.85,
                )
            else:
                ax.text(
                    coords[i, 0],
                    coords[i, 1],
                    str(name),
                    fontsize=7,
                    alpha=0.85,
                )

    map_title = semantic_map_main_title(projection_name, n_components)

    fig.subplots_adjust(left=0.08, right=0.96, bottom=0.14, top=0.82)
    add_theory_figure_titles(
        fig,
        map_title,
        embedding_set_label_text,
        title_y=0.94,
        subtitle_y=0.875,
    )

    axis_label = f"{projection_name} dimension"
    x_label, y_label = f"{axis_label} 1", f"{axis_label} 2"
    z_label = f"{axis_label} 3"
    if n_components == 3:
        ax.set_xlabel(x_label, fontsize=MAP_FONT_AXIS, fontweight="bold", labelpad=12)
        ax.set_ylabel(y_label, fontsize=MAP_FONT_AXIS, fontweight="bold", labelpad=12)
        ax.set_zlabel(z_label, fontsize=MAP_FONT_AXIS, fontweight="bold", labelpad=12)
        ax.tick_params(axis="x", labelsize=MAP_FONT_TICK)
        ax.tick_params(axis="y", labelsize=MAP_FONT_TICK)
        ax.tick_params(axis="z", labelsize=MAP_FONT_TICK)
    else:
        ax.set_xlabel(x_label, fontsize=MAP_FONT_AXIS, fontweight="bold", labelpad=12)
        ax.set_ylabel(y_label, fontsize=MAP_FONT_AXIS, fontweight="bold", labelpad=12)
        style_axes(ax)
        ax.tick_params(axis="both", labelsize=MAP_FONT_TICK)
        ax.grid(alpha=0.2, zorder=0)
        apply_semantic_map_2d_limits(ax, coords, projection_name)

    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=colors_map[group],
            markeredgecolor=BAR_EDGE_COLOR,
            markeredgewidth=BAR_EDGE_WIDTH,
            markersize=8,
        )
        for group in groups
    ]
    if collapse_human:
        legend_labels = collapsed_legend_labels(
            groups,
            {group: int(group_counts.get(group, 0)) for group in groups},
        )
    else:
        legend_labels = [
            legend_entry(
                PARTICIPANT_TYPE_TO_LEGEND.get(group, group),
                int(group_counts.get(group, 0)),
            )
            for group in groups
        ]
    ax.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc="best",
        frameon=True,
        fancybox=False,
        edgecolor="#666666",
        facecolor="white",
        framealpha=1.0,
        fontsize=FONT_LEGEND,
    )

    fig.savefig(outpath, dpi=SAVE_DPI, pad_inches=SAVE_PAD_INCHES)
    plt.close(fig)


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    arr_a = np.asarray(a, dtype=float)
    arr_b = np.asarray(b, dtype=float)
    arr_a = arr_a[~np.isnan(arr_a)]
    arr_b = arr_b[~np.isnan(arr_b)]
    if len(arr_a) < 2 or len(arr_b) < 2:
        return np.nan
    n1, n2 = len(arr_a), len(arr_b)
    var1, var2 = np.var(arr_a, ddof=1), np.var(arr_b, ddof=1)
    pooled = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled < 1e-12:
        return np.nan
    return float((np.mean(arr_a) - np.mean(arr_b)) / pooled)


def p_value_permutation_mean_diff(
    a: np.ndarray,
    b: np.ndarray,
    *,
    n_perm: int = 10000,
    seed: int = 42,
) -> float:
    arr_a = np.asarray(a, dtype=float)
    arr_b = np.asarray(b, dtype=float)
    arr_a = arr_a[~np.isnan(arr_a)]
    arr_b = arr_b[~np.isnan(arr_b)]
    if len(arr_a) < 2 or len(arr_b) < 2:
        return np.nan
    observed = float(np.mean(arr_a) - np.mean(arr_b))
    combined = np.concatenate([arr_a, arr_b])
    n_a = len(arr_a)
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(n_perm):
        perm = rng.permutation(combined)
        diff = float(np.mean(perm[:n_a]) - np.mean(perm[n_a:]))
        if abs(diff) >= abs(observed):
            extreme += 1
    return (extreme + 1) / (n_perm + 1)


def comparison_pairs_for_groups(
    groups: List[str],
    *,
    group_col: str = PARTICIPANT_TYPE_COL,
) -> List[Tuple[str, str, str]]:
    """Return (left_group_key, right_group_key, display_label) for pairwise tests."""
    if group_col == COLLAPSED_PARTICIPANT_TYPE_COL and {"Human", "GenAI"} <= set(groups):
        return [
            ("Human", "GenAI", comparison_pair_label("Human", "GenAI")),
        ]

    if set(groups) >= {"student", "expert", "GenAI"}:
        return [
            ("expert", "student", comparison_pair_label("Experts", "PhD Students")),
            ("student", "GenAI", comparison_pair_label("PhD Students", "GenAI")),
            ("expert", "GenAI", comparison_pair_label("Experts", "GenAI")),
        ]

    pairs: List[Tuple[str, str, str]] = []
    for i, left in enumerate(groups):
        for right in groups[i + 1 :]:
            left_label = PARTICIPANT_TYPE_TO_LEGEND.get(left, left)
            right_label = PARTICIPANT_TYPE_TO_LEGEND.get(right, right)
            pairs.append(
                (left, right, comparison_pair_label(left_label, right_label))
            )
    return pairs


def group_metric_values(
    data: pd.DataFrame,
    group_col: str,
    group: str,
    value_col: str,
) -> np.ndarray:
    vals = data.loc[data[group_col] == group, value_col].values.astype(float)
    return vals[~np.isnan(vals)]


def build_pairwise_inference_rows(
    data: pd.DataFrame,
    value_col: str,
    groups: List[str],
    *,
    group_col: str = PARTICIPANT_TYPE_COL,
    metric_name: str,
    seed: int = 42,
) -> List[dict]:
    rows: List[dict] = []
    for left, right, label in comparison_pairs_for_groups(groups, group_col=group_col):
        vals_left = group_metric_values(data, group_col, left, value_col)
        vals_right = group_metric_values(data, group_col, right, value_col)
        welch_p = p_value_welch_ttest(vals_left, vals_right)
        perm_p = p_value_permutation_mean_diff(
            vals_left, vals_right, seed=seed
        )
        rows.append({
            "metric": metric_name,
            "comparison": label,
            "left_group": left,
            "right_group": right,
            "left_mean": float(np.mean(vals_left)) if len(vals_left) else np.nan,
            "right_mean": float(np.mean(vals_right)) if len(vals_right) else np.nan,
            "mean_diff_left_minus_right": (
                float(np.mean(vals_left) - np.mean(vals_right))
                if len(vals_left) and len(vals_right)
                else np.nan
            ),
            "welch_pvalue": welch_p,
            "permutation_pvalue": perm_p,
            "cohens_d": cohens_d(vals_left, vals_right),
            "significance": significance_label(welch_p),
        })
    return rows


def format_diversity_comparison_line(
    comparison: str,
    welch_p: float,
    cohens_d_val: float = np.nan,
    *,
    include_cohens_d: bool = True,
) -> str:
    sig = significance_label(welch_p)
    d_suffix = ""
    if include_cohens_d and np.isfinite(cohens_d_val):
        d_suffix = f", d={cohens_d_val:.2f}"
    return f"{comparison}: {sig}{d_suffix}"


def comparisons_for_metric(
    pairwise_df: pd.DataFrame,
    metric: str,
    *,
    include_cohens_d: bool = True,
) -> List[Tuple[str, float]]:
    if pairwise_df is None or pairwise_df.empty:
        return []
    rows = pairwise_df.loc[pairwise_df["metric"] == metric]
    comparisons: List[Tuple[str, float]] = []
    for _, row in rows.iterrows():
        comparisons.append(
            (str(row["comparison"]), float(row["welch_pvalue"]))
        )
    return comparisons


def batch_visualizations_root(embeddings_root: Path) -> Path:
    return embeddings_root / BATCH_VISUALIZATIONS_DIRNAME


def comparisons_pre_post_dir(embeddings_root: Path, subdir: str) -> Path:
    return (
        batch_visualizations_root(embeddings_root)
        / COMPARISONS_PRE_POST_SUBDIR
        / subdir
    )


def batch_phase_dir(embeddings_root: Path, phase: str) -> Path:
    return batch_visualizations_root(embeddings_root) / phase


DATA_SUBDIR = "data"
NETWORK_SUBDIR = "network"


def infer_embeddings_root(embedding_set_dir: Path) -> Path:
    if embedding_set_dir.name not in PHASE_NAMES:
        raise ValueError(
            f"Cannot infer embeddings root from {embedding_set_dir}; "
            f"expected phase folder name in {PHASE_NAMES}."
        )
    return embedding_set_dir.parent.parent.parent


def task_phase_slug(
    embeddings_root: Path,
    embedding_set_dir: Path,
    embedding_col: str,
) -> str:
    task_key = embedding_set_dir.relative_to(embeddings_root).parent
    phase = embedding_set_dir.name
    task_slug = str(task_key).replace("/", "_")
    return f"{phase}__{task_slug}__{safe_name(embedding_col)}"


def resolve_task_data_dir(
    embeddings_root: Path,
    embedding_set_dir: Path,
    embedding_col: str,
) -> Path:
    slug = task_phase_slug(embeddings_root, embedding_set_dir, embedding_col)
    return batch_visualizations_root(embeddings_root) / DATA_SUBDIR / slug


def resolve_network_dir(
    embeddings_root: Path,
    embedding_set_dir: Path,
    embedding_col: str,
) -> Path:
    task_key = embedding_set_dir.relative_to(embeddings_root).parent
    phase = embedding_set_dir.name
    outdir = (
        batch_visualizations_root(embeddings_root)
        / NETWORK_SUBDIR
        / task_key
        / phase
        / safe_name(embedding_col)
    )
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def resolve_network_outpath(
    embeddings_root: Path,
    embedding_set_dir: Path,
    embedding_col: str,
    *,
    collapsed: bool = False,
) -> Path:
    filename = (
        THRESHOLD_NETWORK_COLLAPSED_FILENAME
        if collapsed
        else THRESHOLD_NETWORK_FILENAME
    )
    return resolve_network_dir(embeddings_root, embedding_set_dir, embedding_col) / filename


def figure_comparison_footer_layout(
    n_comp_lines: int,
    *,
    footnote_lines: Tuple[str, ...] = DIVERSITY_SIG_FOOTNOTE,
) -> Tuple[float, float, float]:
    return centroid_distribution_bottom_layout(
        n_comp_lines,
        n_footnote_lines=len(footnote_lines),
    )


def draw_figure_footnote_lines(
    fig,
    y: float,
    lines: Tuple[str, ...],
    *,
    line_step: float = CENTROID_FOOTNOTE_LINE_STEP,
) -> None:
    for i, line in enumerate(lines):
        fig.text(
            0.5,
            y - i * line_step,
            line,
            ha="center",
            va="bottom",
            fontsize=FONT_FOOTNOTE,
            color=FOOTNOTE_COLOR,
            transform=fig.transFigure,
            clip_on=False,
            zorder=1,
        )


def attach_centered_comparisons(
    fig,
    ax,
    comparisons: List[Tuple[str, float]],
    *,
    comp_bottom: float,
    footnote_y: float,
    footnote_lines: Tuple[str, ...] = DIVERSITY_SIG_FOOTNOTE,
    max_width_frac: float = BOX_COMP_MAX_WIDTH_FRAC,
    center_x: float | None = None,
    footnote_line_step: float = CENTROID_FOOTNOTE_LINE_STEP,
) -> None:
    if not comparisons:
        return
    fig.canvas.draw()
    if center_x is None:
        bbox = ax.get_position()
        panel_w = bbox.x1 - bbox.x0
        center_x = (bbox.x0 + bbox.x1) / 2
        max_width = panel_w * max_width_frac
    else:
        max_width = max_width_frac
    draw_centered_comparison_box(
        fig,
        comparisons,
        center_x=center_x,
        box_bottom=comp_bottom,
        min_box_width=0.0,
        max_box_width=max_width,
    )
    draw_figure_footnote_lines(
        fig, footnote_y, footnote_lines, line_step=footnote_line_step
    )


def diversity_summary_for_group(
    diversity_summary_df: pd.DataFrame | None,
    group: str,
) -> dict | None:
    if diversity_summary_df is None or diversity_summary_df.empty:
        return None
    sub = diversity_summary_df.loc[diversity_summary_df["group"] == group]
    if sub.empty:
        return None
    return sub.iloc[0].to_dict()


def welch_comparisons_for_distance(
    point_metrics: pd.DataFrame,
    distance_col: str,
    groups: List[str],
    *,
    group_col: str = PARTICIPANT_TYPE_COL,
    seed: int = 42,
) -> List[Tuple[str, float]]:
    """Pairwise Welch t-tests aligned with main_effects quant panels."""
    rows = build_pairwise_inference_rows(
        point_metrics,
        distance_col,
        groups,
        group_col=group_col,
        metric_name="centroid_distance",
        seed=seed,
    )
    return [(row["comparison"], row["welch_pvalue"]) for row in rows]


def histogram_bin_count(n: int) -> int:
    return max(5, min(12, int(np.sqrt(n))))


def dagostino_skew_test(vals: np.ndarray) -> Tuple[float, float]:
    """D'Agostino K² test: H0 population skewness is zero."""
    arr = np.asarray(vals, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 8:
        return np.nan, np.nan
    stat, p = skewtest(arr, nan_policy="omit")
    return float(stat), float(p)


def compute_centroid_skewness_stats(
    point_metrics: pd.DataFrame,
    distance_col: str,
    *,
    group_col: str = PARTICIPANT_TYPE_COL,
) -> pd.DataFrame:
    if group_col == COLLAPSED_PARTICIPANT_TYPE_COL:
        preferred_order = GROUP_ORDER_COLLAPSED
        display_names_map = DISPLAY_LABELS
    else:
        preferred_order = GROUP_ORDER
        display_names_map = PARTICIPANT_TYPE_TO_LEGEND

    groups = [
        g for g in preferred_order if g in point_metrics[group_col].unique()
    ]
    groups += [g for g in point_metrics[group_col].unique() if g not in groups]

    rows = []
    for group in groups:
        vals = point_metrics.loc[
            point_metrics[group_col] == group, distance_col
        ].values.astype(float)
        vals = vals[~np.isnan(vals)]
        dag_stat, dag_p = dagostino_skew_test(vals)
        rows.append({
            "group": group,
            "display_label": display_names_map.get(group, group),
            "n": len(vals),
            "skewness": float(skew(vals, bias=False)) if len(vals) >= 3 else np.nan,
            "dagostino_statistic": dag_stat,
            "dagostino_pvalue": dag_p,
            "dagostino_significance": significance_label(dag_p),
        })
    return pd.DataFrame(rows)


def centroid_distribution_bottom_layout(
    n_comp_lines: int,
    *,
    n_footnote_lines: int = len(CENTROID_MEAN_SIG_FOOTNOTE),
) -> Tuple[float, float, float]:
    """Return (comparison_box_bottom, footnote_y, subplot_bottom)."""
    footnote_y = CENTROID_FOOTNOTE_BOTTOM_Y + CENTROID_FOOTNOTE_LINE_STEP * max(
        n_footnote_lines - 1, 0
    )
    footnote_top = footnote_y + CENTROID_FOOTNOTE_TEXT_HEIGHT
    comp_bottom = footnote_top + CENTROID_FOOTNOTE_TO_COMP_GAP
    comp_height = comparison_box_height(n_comp_lines) if n_comp_lines else 0.0
    subplot_bottom = (
        comp_bottom + comp_height + CENTROID_AXIS_GAP if n_comp_lines else 0.14
    )
    return comp_bottom, footnote_y, subplot_bottom


def prepare_centroid_distance_plot_data(
    point_metrics: pd.DataFrame,
    distance_col: str,
    *,
    group_col: str = PARTICIPANT_TYPE_COL,
    diversity_summary_df: pd.DataFrame | None = None,
    diversity_pairwise_df: pd.DataFrame | None = None,
) -> dict:
    if group_col == COLLAPSED_PARTICIPANT_TYPE_COL:
        preferred_order = GROUP_ORDER_COLLAPSED
        color_map = GROUP_COLORS_COLLAPSED
        display_names_map = DISPLAY_LABELS
    else:
        preferred_order = GROUP_ORDER
        color_map = GROUP_COLORS_BY_PARTICIPANT_TYPE
        display_names_map = PARTICIPANT_TYPE_TO_LEGEND

    groups = [
        g for g in preferred_order if g in point_metrics[group_col].unique()
    ]
    groups += [g for g in point_metrics[group_col].unique() if g not in groups]

    group_series: List[Tuple[str, np.ndarray]] = []
    for group in groups:
        vals = point_metrics.loc[
            point_metrics[group_col] == group, distance_col
        ].values.astype(float)
        vals = vals[~np.isnan(vals)]
        if len(vals) < 2:
            continue
        group_series.append((group, vals))

    skew_df = compute_centroid_skewness_stats(
        point_metrics, distance_col, group_col=group_col
    )
    skew_by_group = skew_df.set_index("group")["skewness"].to_dict()

    if diversity_pairwise_df is not None:
        comparisons = comparisons_for_metric(
            diversity_pairwise_df, "centroid_distance", include_cohens_d=False
        )
        inference_rows = diversity_pairwise_df.loc[
            diversity_pairwise_df["metric"] == "centroid_distance"
        ].to_dict("records")
    else:
        inference_rows = build_pairwise_inference_rows(
            point_metrics,
            distance_col,
            groups,
            group_col=group_col,
            metric_name="centroid_distance",
        )
        comparisons = [
            (str(row["comparison"]), float(row["welch_pvalue"]))
            for row in inference_rows
        ]

    mean_ci_by_group: Dict[str, Tuple[float, float, float]] = {}
    for group, vals in group_series:
        mean_dist = float(np.mean(vals))
        div_row = diversity_summary_for_group(diversity_summary_df, group)
        if div_row is not None:
            ci_lo = float(div_row["centroid_distance_ci_low"])
            ci_hi = float(div_row["centroid_distance_ci_high"])
        else:
            ci_lo, ci_hi = bootstrap_mean_ci(vals)
        mean_ci_by_group[group] = (mean_dist, ci_lo, ci_hi)

    return {
        "group_col": group_col,
        "color_map": color_map,
        "display_names_map": display_names_map,
        "group_series": group_series,
        "skew_df": skew_df,
        "skew_by_group": skew_by_group,
        "comparisons": comparisons,
        "inference_rows": inference_rows,
        "mean_ci_by_group": mean_ci_by_group,
    }


def plot_centroid_distance_distribution(
    plot_data: dict,
    outpath: str,
    embedding_set_label_text: str,
) -> None:
    """Overlapping density histograms; legend shows group skewness only."""
    color_map = plot_data["color_map"]
    display_names_map = plot_data["display_names_map"]
    group_col = plot_data["group_col"]
    skew_by_group = plot_data["skew_by_group"]
    group_series = plot_data["group_series"]

    fig, ax = plt.subplots(figsize=CENTROID_DIST_FIGSIZE)
    legend_handles: List[Patch] = []
    legend_labels: List[str] = []

    for group, vals in group_series:
        color = color_map[group]
        ax.hist(
            vals,
            bins=histogram_bin_count(len(vals)),
            alpha=0.55,
            density=True,
            color=color,
            edgecolor=BAR_EDGE_COLOR,
            linewidth=BAR_EDGE_WIDTH * 0.6,
        )
        legend_handles.append(
            Patch(
                facecolor=color,
                edgecolor=BAR_EDGE_COLOR,
                linewidth=BAR_EDGE_WIDTH,
                alpha=0.88,
            )
        )
        if group_col == COLLAPSED_PARTICIPANT_TYPE_COL:
            base_label = legend_entry(
                display_names_map.get(group, group),
                len(vals),
                include_composition=(group == "Human"),
            )
        else:
            base_label = legend_entry(
                display_names_map.get(group, group),
                len(vals),
            )
        group_skew = skew_by_group.get(group, np.nan)
        if np.isfinite(group_skew):
            legend_labels.append(f"{base_label}, skew={group_skew:.2f}")
        else:
            legend_labels.append(base_label)

    ax._viz_bold_xticks = False
    style_axes(ax)
    ax.set_xlabel(
        "Cosine distance to group mean vector",
        fontsize=DIAG_FONT_AXIS,
        fontweight="bold",
        labelpad=10,
    )
    fig.supylabel(
        "Density",
        fontsize=DIAG_FONT_AXIS,
        fontweight="bold",
        x=BOX_SUPYLABEL_X,
    )
    ax.tick_params(axis="x", labelsize=DIAG_FONT_TICK)
    ax.tick_params(axis="y", labelsize=DIAG_FONT_TICK)
    ax.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc="upper right",
        frameon=True,
        fancybox=False,
        edgecolor="#666666",
        facecolor="white",
        fontsize=FONT_LEGEND,
    )

    fig.subplots_adjust(
        left=BOX_SUBPLOT_LEFT,
        right=0.98,
        bottom=0.14,
        top=BOX_SUBPLOT_TOP,
    )
    add_theory_figure_titles(
        fig,
        "Within-Group Distance to Group Centroid Distribution",
        embedding_set_label_text,
    )
    fig.savefig(outpath, dpi=SAVE_DPI, pad_inches=SAVE_PAD_INCHES)
    plt.close(fig)


def plot_centroid_distance_boxplot(
    plot_data: dict,
    outpath: str,
    embedding_set_label_text: str,
) -> None:
    """Standard box plots (median, IQR, whiskers, outliers) with Welch comparisons."""
    color_map = plot_data["color_map"]
    display_names_map = plot_data["display_names_map"]
    group_col = plot_data["group_col"]
    group_series = plot_data["group_series"]
    comparisons = plot_data["comparisons"]

    n_comp_lines = len(comparisons)
    comp_bottom, footnote_y, subplot_bottom = centroid_distribution_bottom_layout(
        n_comp_lines,
        n_footnote_lines=len(CENTROID_MEAN_SIG_FOOTNOTE),
    )

    fig, ax = plt.subplots(figsize=CENTROID_DIST_FIGSIZE)
    x = np.arange(len(group_series))
    box_vals = [vals for _, vals in group_series]
    bp = ax.boxplot(
        box_vals,
        positions=x,
        widths=CENTROID_BOX_WIDTH,
        patch_artist=True,
        showfliers=True,
        showmeans=True,
        whis=1.5,
        medianprops={
            "color": BOXPLOT_STAT_COLOR,
            "linewidth": 1.5,
            "linestyle": "--",
        },
        meanprops={
            "marker": "^",
            "markerfacecolor": BOXPLOT_STAT_COLOR,
            "markeredgecolor": BOXPLOT_STAT_COLOR,
            "markersize": 7,
        },
        whiskerprops={
            "color": BOXPLOT_EDGE_COLOR,
            "linewidth": BOXPLOT_WHISKER_WIDTH,
            "solid_capstyle": "butt",
        },
        capprops={
            "color": BOXPLOT_EDGE_COLOR,
            "linewidth": BOXPLOT_WHISKER_WIDTH,
        },
        boxprops={
            "linewidth": BOXPLOT_WHISKER_WIDTH,
            "edgecolor": BOXPLOT_EDGE_COLOR,
        },
        flierprops={
            "marker": "o",
            "markersize": 4.5,
            "markerfacecolor": "white",
            "markeredgecolor": BOXPLOT_EDGE_COLOR,
            "alpha": 0.9,
        },
    )
    for patch, (group, _) in zip(bp["boxes"], group_series):
        patch.set_facecolor(color_map[group])
        patch.set_alpha(0.55)
        patch.set_edgecolor(BOXPLOT_EDGE_COLOR)

    all_vals = np.concatenate(box_vals)
    data_hi = float(np.max(all_vals))
    data_lo = float(np.min(all_vals))
    y_pad = max((data_hi - data_lo) * 0.08, 0.008)
    ax.set_ylim(bottom=max(0.0, data_lo - y_pad), top=data_hi + y_pad)

    xticks: List[str] = []
    for group, vals in group_series:
        if group_col == COLLAPSED_PARTICIPANT_TYPE_COL:
            xticks.append(
                legend_entry(
                    display_names_map.get(group, group),
                    len(vals),
                    include_composition=(group == "Human"),
                )
            )
        else:
            xticks.append(
                legend_entry(display_names_map.get(group, group), len(vals))
            )

    ax.set_xticks(x)
    ax.set_xticklabels(xticks, fontsize=BOX_FONT_XTICK)
    ax._viz_bold_xticks = True
    style_axes(ax)
    ax.tick_params(axis="x", pad=12)
    fig.supylabel(
        "Cosine distance to group mean vector",
        fontsize=DIAG_FONT_AXIS,
        fontweight="bold",
        x=BOX_SUPYLABEL_X,
    )
    ax.tick_params(axis="y", labelsize=DIAG_FONT_TICK)

    boxplot_legend_handles = [
        Line2D(
            [0],
            [0],
            color=BOXPLOT_STAT_COLOR,
            linewidth=1.5,
            linestyle="--",
            label="Median",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color=BOXPLOT_STAT_COLOR,
            markerfacecolor=BOXPLOT_STAT_COLOR,
            markeredgecolor=BOXPLOT_STAT_COLOR,
            markersize=7,
            linestyle="None",
            label="Mean",
        ),
        Line2D(
            [0],
            [0],
            color=BOXPLOT_EDGE_COLOR,
            linewidth=BOXPLOT_WHISKER_WIDTH,
            label="Whiskers (1.5×IQR)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="white",
            markeredgecolor=BOXPLOT_EDGE_COLOR,
            markersize=5,
            linestyle="None",
            label="Outliers",
        ),
    ]
    ax.legend(
        handles=boxplot_legend_handles,
        loc="upper right",
        frameon=True,
        fancybox=False,
        edgecolor="#666666",
        facecolor="white",
        fontsize=FONT_LEGEND,
    )

    fig.subplots_adjust(
        left=BOX_SUBPLOT_LEFT,
        right=0.98,
        bottom=subplot_bottom,
        top=BOX_SUBPLOT_TOP,
    )
    add_theory_figure_titles(
        fig,
        "Within-Group Distance to Group Centroid",
        embedding_set_label_text,
    )

    if comparisons:
        attach_centered_comparisons(
            fig,
            ax,
            comparisons,
            comp_bottom=comp_bottom,
            footnote_y=footnote_y,
            footnote_lines=CENTROID_MEAN_SIG_FOOTNOTE,
        )
        fig.subplots_adjust(left=BOX_SUBPLOT_LEFT, top=BOX_SUBPLOT_TOP)

    fig.savefig(outpath, dpi=SAVE_DPI, pad_inches=SAVE_PAD_INCHES)
    plt.close(fig)


def plot_centroid_distance_figures(
    point_metrics: pd.DataFrame,
    distance_col: str,
    distribution_outpath: str,
    boxplot_outpath: str,
    embedding_set_label_text: str,
    *,
    group_col: str = PARTICIPANT_TYPE_COL,
    diversity_summary_df: pd.DataFrame | None = None,
    diversity_pairwise_df: pd.DataFrame | None = None,
) -> None:
    plot_data = prepare_centroid_distance_plot_data(
        point_metrics,
        distance_col,
        group_col=group_col,
        diversity_summary_df=diversity_summary_df,
        diversity_pairwise_df=diversity_pairwise_df,
    )
    plot_centroid_distance_distribution(
        plot_data,
        distribution_outpath,
        embedding_set_label_text,
    )
    plot_centroid_distance_boxplot(
        plot_data,
        boxplot_outpath,
        embedding_set_label_text,
    )


def compute_pairwise_similarity_summary(
    df: pd.DataFrame,
    X: np.ndarray,
    *,
    group_col: str = PARTICIPANT_TYPE_COL,
) -> pd.DataFrame:
    if group_col == COLLAPSED_PARTICIPANT_TYPE_COL:
        plot_df = with_collapsed_group(df)
        display_names_map = DISPLAY_LABELS
    else:
        plot_df = df
        display_names_map = PARTICIPANT_TYPE_TO_LEGEND

    labels = plot_df[group_col].values
    groups = ordered_groups(plot_df, group_col)

    rows = []
    for group in groups:
        idx = np.where(labels == group)[0]
        if len(idx) < 2:
            continue
        vals = upper_triangle_values(cosine_similarity(X[idx]))
        rows.append({
            "group": group,
            "display_label": display_names_map.get(group, group),
            "n": len(idx),
            "mean_pairwise_similarity": float(np.mean(vals)),
            "var_pairwise_similarity": float(np.var(vals)),
        })
    return pd.DataFrame(rows)


def similarity_distance_layout(sim: np.ndarray, *, seed: int) -> np.ndarray:
    """2D MDS on cosine distance so inter-node spacing reflects dissimilarity."""
    n = sim.shape[0]
    if n == 0:
        return np.zeros((0, 2))
    if n == 1:
        return np.array([[0.5, 0.5]])

    dist = np.clip(1.0 - sim, 0.0, None)
    np.fill_diagonal(dist, 0.0)

    if n == 2:
        separation = float(dist[0, 1])
        coords = np.array([[0.5 - separation / 2, 0.5], [0.5 + separation / 2, 0.5]])
        return normalize_layout_coords(coords)

    coords = MDS(
        n_components=2,
        dissimilarity="precomputed",
        random_state=seed,
        normalized_stress="auto",
    ).fit_transform(dist)
    return normalize_layout_coords(coords)


def normalize_layout_coords(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=float)
    if len(coords) == 0:
        return coords
    if len(coords) == 1:
        return np.array([[0.5, 0.5]])
    coords -= coords.min(axis=0)
    span = coords.max(axis=0)
    span = np.where(span < 1e-12, 1.0, span)
    coords = coords / span
    return NETWORK_LAYOUT_PAD + coords * (1.0 - 2.0 * NETWORK_LAYOUT_PAD)


def plot_group_threshold_networks(
    df: pd.DataFrame,
    X: np.ndarray,
    similarity_threshold: float,
    threshold_quantile: float,
    embedding_set_label_text: str,
    outpath: str,
    *,
    seed: int,
    group_col: str = PARTICIPANT_TYPE_COL,
) -> None:
    """Within-group semantic network; node spacing from 2D MDS on cosine distance."""
    if group_col == COLLAPSED_PARTICIPANT_TYPE_COL:
        plot_df = with_collapsed_group(df)
        color_map = GROUP_COLORS_COLLAPSED
    else:
        plot_df = df
        color_map = GROUP_COLORS_BY_PARTICIPANT_TYPE

    labels = plot_df[group_col].values
    groups = ordered_groups(plot_df, group_col)
    group_counts = plot_df[group_col].value_counts().to_dict()
    quantile_pct = int(round(threshold_quantile * 100))

    n_groups = len(groups)
    fig, axes = plt.subplots(
        1,
        n_groups,
        figsize=(4.8 * n_groups, 5.6),
        squeeze=False,
    )

    for panel_idx, (ax, group) in enumerate(zip(axes[0], groups)):
        idx = np.where(labels == group)[0]
        if len(idx) == 0:
            ax.set_visible(False)
            continue

        Xg = X[idx]
        Sg = cosine_similarity(Xg)
        layout = similarity_distance_layout(Sg, seed=seed + panel_idx)

        segments = []
        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                if Sg[i, j] >= similarity_threshold:
                    segments.append([layout[i], layout[j]])

        color = color_map[group]
        if segments:
            lc = LineCollection(
                segments,
                colors=NETWORK_EDGE_COLOR,
                linewidths=NETWORK_EDGE_WIDTH,
                alpha=NETWORK_EDGE_ALPHA,
                zorder=1,
            )
            ax.add_collection(lc)

        ax.scatter(
            layout[:, 0],
            layout[:, 1],
            s=NETWORK_NODE_SIZE,
            c=color,
            alpha=SCATTER_ALPHA,
            edgecolors=BAR_EDGE_COLOR,
            linewidths=BAR_EDGE_WIDTH,
            zorder=3,
        )

        if group_col == COLLAPSED_PARTICIPANT_TYPE_COL:
            panel_title = collapsed_legend_labels(
                [group],
                {group: int(group_counts.get(group, 0))},
            )[0]
        else:
            panel_title = legend_entry(
                PARTICIPANT_TYPE_TO_LEGEND.get(group, group),
                len(idx),
            )
        ax.set_title(
            panel_title,
            fontsize=DIAG_FONT_AXIS,
            fontweight="bold",
            pad=10,
        )
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")

    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.18, top=BOX_SUBPLOT_TOP, wspace=0.28)
    add_theory_figure_titles(
        fig,
        "Semantic Threshold Network Within Each Group",
        embedding_set_label_text,
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=NETWORK_EDGE_COLOR,
            linewidth=NETWORK_EDGE_WIDTH,
            alpha=NETWORK_EDGE_ALPHA,
            label=(
                f"Edge: cosine similarity ≥ {similarity_threshold:.3f} "
                f"({quantile_pct}th percentile of all pairs)"
            ),
        ),
        Line2D(
            [0],
            [0],
            linestyle="None",
            marker="None",
            alpha=0.0,
            label="The number of edges indicates within-group semantic connectivity",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=1,
        frameon=True,
        fancybox=False,
        edgecolor="#666666",
        facecolor="white",
        fontsize=FONT_LEGEND,
        bbox_to_anchor=(0.5, 0.02),
        handletextpad=0.6,
    )

    fig.savefig(outpath, dpi=SAVE_DPI, pad_inches=SAVE_PAD_INCHES)
    plt.close(fig)


def hdbscan_min_cluster_size(n: int) -> int:
    """Small groups need a small minimum; larger n still uses 2 (see tuning on cosine distances)."""
    if n < 3:
        return 1
    return 2


def hdbscan_cluster_selection_epsilon(embedding_col: str) -> float:
    return HDBSCAN_CLUSTER_SELECTION_EPSILON_BY_EMBEDDING.get(
        embedding_col,
        0,
    )


def run_hdbscan_within_group(
    Xg: np.ndarray,
    *,
    cluster_selection_epsilon,
) -> Tuple[np.ndarray, int, int, float]:
    """
    HDBSCAN on precomputed cosine distances (unit-norm embeddings).

    min_samples=1 is required here: with min_samples>=2 on cosine-distance
    matrices, this embedding space often collapses to all-noise labels.
    cluster_selection_epsilon widens the main cluster in tighter groups (GenAI)
    while leaving more PhD respondents as tail outliers.
    """
    n = Xg.shape[0]
    epsilon = float(cluster_selection_epsilon)
    if n < 3:
        return np.zeros(n, dtype=int), 1, 1, epsilon

    min_cluster_size = hdbscan_min_cluster_size(n)
    min_samples = 1
    distance_matrix = cosine_distances(Xg)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="precomputed",
        cluster_selection_method="eom",
        cluster_selection_epsilon=epsilon,
        allow_single_cluster=True,
    )
    labels = clusterer.fit_predict(distance_matrix)
    return labels.astype(int), min_cluster_size, min_samples, epsilon


def summarize_core_tail(labels: np.ndarray) -> Dict[str, float]:
    """Core = any HDBSCAN cluster; tail = noise (-1)."""
    n = len(labels)
    tail_mask = labels == -1
    core_mask = ~tail_mask
    core_n = int(core_mask.sum())
    tail_n = int(tail_mask.sum())

    if core_n == 0:
        largest_cluster_n = 0
        n_clusters = 0
    else:
        core_labels = labels[core_mask]
        _, counts = np.unique(core_labels, return_counts=True)
        largest_cluster_n = int(counts.max())
        n_clusters = int(len(counts))

    return {
        "n": n,
        "core_n": core_n,
        "tail_n": tail_n,
        "core_pct": 100.0 * core_n / n,
        "tail_pct": 100.0 * tail_n / n,
        "n_clusters": n_clusters,
        "largest_cluster_n": largest_cluster_n,
        "largest_cluster_pct": 100.0 * largest_cluster_n / n,
    }


def compute_semantic_clustering_tables(
    df: pd.DataFrame,
    X: np.ndarray,
    *,
    group_col: str = PARTICIPANT_TYPE_COL,
    cluster_selection_epsilon: float = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if group_col == COLLAPSED_PARTICIPANT_TYPE_COL:
        plot_df = with_collapsed_group(df)
    else:
        plot_df = df

    labels_col = plot_df[group_col].values
    groups = ordered_groups(plot_df, group_col)
    participant_rows: List[dict] = []
    summary_rows: List[dict] = []

    for group in groups:
        idx = np.where(labels_col == group)[0]
        Xg = X[idx]
        cluster_labels, min_cs, min_ss, epsilon = run_hdbscan_within_group(
            Xg,
            cluster_selection_epsilon=cluster_selection_epsilon,
        )
        summary = summarize_core_tail(cluster_labels)
        summary_rows.append({
            "participant_type": group,
            "min_cluster_size": min_cs,
            "min_samples": min_ss,
            "cluster_selection_epsilon": epsilon,
            **summary,
        })

        for local_i, global_i in enumerate(idx):
            participant_rows.append({
                PARTICIPANT_NAME_COL: plot_df.iloc[global_i][PARTICIPANT_NAME_COL],
                group_col: group,
                "hdbscan_label": int(cluster_labels[local_i]),
                "is_core": bool(cluster_labels[local_i] >= 0),
                "is_tail": bool(cluster_labels[local_i] == -1),
            })

    return pd.DataFrame(participant_rows), pd.DataFrame(summary_rows)


def plot_semantic_clustering_core_tail(
    summary_df: pd.DataFrame,
    outpath: str,
    embedding_set_label_text: str,
    *,
    group_col: str = PARTICIPANT_TYPE_COL,
) -> None:
    """100% stacked bars: share of respondents in HDBSCAN core vs. tail outliers."""
    if group_col == COLLAPSED_PARTICIPANT_TYPE_COL:
        preferred_order = GROUP_ORDER_COLLAPSED
        color_map = GROUP_COLORS_COLLAPSED
        display_names_map = DISPLAY_LABELS
    else:
        preferred_order = GROUP_ORDER
        color_map = GROUP_COLORS_BY_PARTICIPANT_TYPE
        display_names_map = PARTICIPANT_TYPE_TO_LEGEND

    plot_df = summary_df.set_index("participant_type")
    groups = [g for g in preferred_order if g in plot_df.index]
    groups += [g for g in plot_df.index if g not in groups]

    core_pcts = [float(plot_df.loc[g, "core_pct"]) for g in groups]
    tail_pcts = [float(plot_df.loc[g, "tail_pct"]) for g in groups]
    n_clusters_list = [int(plot_df.loc[g, "n_clusters"]) for g in groups]
    display_names = [display_names_map.get(g, g) for g in groups]

    fig, ax = plt.subplots(figsize=(8.8, 6.4))
    x = np.arange(len(groups))
    bar_width = CLUSTERING_BAR_WIDTH

    core_colors = [color_map[g] for g in groups]
    ax.bar(
        x,
        core_pcts,
        width=bar_width,
        color=core_colors,
        edgecolor=BAR_EDGE_COLOR,
        linewidth=BAR_EDGE_WIDTH,
        zorder=3,
    )
    ax.bar(
        x,
        tail_pcts,
        width=bar_width,
        bottom=core_pcts,
        color=CLUSTER_TAIL_COLOR,
        edgecolor=BAR_EDGE_COLOR,
        linewidth=BAR_EDGE_WIDTH,
        zorder=3,
    )

    for i, (group, core, tail, n_clusters) in enumerate(
        zip(groups, core_pcts, tail_pcts, n_clusters_list)
    ):
        if core >= 8:
            ax.text(
                i,
                core / 2 + 4,
                f"{core:.0f}% core",
                ha="center",
                va="center",
                fontsize=CLUSTER_BAR_LABEL_FONT,
                color="#222222",
            )
            ax.text(
                i,
                core / 2 - 5,
                f"(core number = {n_clusters})",
                ha="center",
                va="center",
                fontsize=CLUSTER_BAR_META_FONT,
                color="#222222",
            )
        if tail >= 8:
            ax.text(
                i,
                core + tail / 2,
                f"{tail:.0f}% tail",
                ha="center",
                va="center",
                fontsize=CLUSTER_BAR_LABEL_FONT,
                color="#222222",
            )
        elif tail > 0:
            ax.text(
                i,
                core + tail / 2,
                f"{tail:.0f}% tail",
                ha="center",
                va="center",
                fontsize=CLUSTER_BAR_META_FONT,
                color="#222222",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(display_names, fontsize=CLUSTER_BAR_XTICK_FONT)
    ax._viz_bold_xticks = True
    style_axes(ax)
    ax.tick_params(axis="x", pad=CLUSTER_BAR_XTICK_PAD)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_locator(MultipleLocator(20))
    fig.supylabel(
        "Share of respondents (%)",
        fontsize=DIAG_FONT_AXIS,
        fontweight="bold",
        x=BOX_SUPYLABEL_X,
    )
    ax.tick_params(axis="y", labelsize=DIAG_FONT_TICK)

    fig.subplots_adjust(
        left=BOX_SUBPLOT_LEFT,
        right=0.98,
        bottom=0.14,
        top=BOX_SUBPLOT_TOP,
    )
    add_theory_figure_titles(
        fig,
        "Within-Group Core vs. Tail (Semantic Clustering: HDBSCAN)",
        embedding_set_label_text,
        CLUSTER_CORE_TAIL_DETAIL,
        detail_on_new_line=True,
    )

    fig.savefig(outpath, dpi=SAVE_DPI, pad_inches=SAVE_PAD_INCHES)
    plt.close(fig)


def run_semantic_clustering_analysis(
    df: pd.DataFrame,
    X: np.ndarray,
    space_outdir: str,
    embedding_set_label_text: str,
    *,
    embedding_col: str,
) -> Dict[str, dict]:
    """Compute HDBSCAN labels and save tables; plot 06 is drawn after diversity inference."""
    if not HAS_HDBSCAN:
        print("hdbscan not installed. Skipping semantic clustering (plot 06).")
        return {}

    cluster_selection_epsilon = hdbscan_cluster_selection_epsilon(embedding_col)
    configs = (
        (PARTICIPANT_TYPE_COL, CLUSTERING_CORE_TAIL_FILENAME, ""),
        (COLLAPSED_PARTICIPANT_TYPE_COL, CLUSTERING_CORE_TAIL_COLLAPSED_FILENAME, "_collapsed"),
    )
    summary_3g = None
    clustering_results: Dict[str, dict] = {}
    for group_col, out_name, suffix in configs:
        participant_df, summary_df = compute_semantic_clustering_tables(
            df,
            X,
            group_col=group_col,
            cluster_selection_epsilon=cluster_selection_epsilon,
        )
        clustering_results[suffix] = {
            "participant_df": participant_df,
            "summary_df": summary_df,
            "group_col": group_col,
            "out_name": out_name,
        }
        if group_col == PARTICIPANT_TYPE_COL:
            summary_3g = summary_df
        participant_df.to_csv(
            os.path.join(
                space_outdir,
                SEMANTIC_CLUSTERING_PARTICIPANT_CSV.format(suffix=suffix),
            ),
            index=False,
            encoding="utf-8-sig",
        )
        summary_df.to_csv(
            os.path.join(
                space_outdir,
                SEMANTIC_CLUSTERING_SUMMARY_CSV.format(suffix=suffix),
            ),
            index=False,
            encoding="utf-8-sig",
        )

    if summary_3g is not None:
        print("\nHDBSCAN core/tail summary (3 groups):")
        print(summary_3g.to_string(index=False))

    return clustering_results


def merge_point_metrics_with_clustering(
    point_metrics: pd.DataFrame,
    participant_df: pd.DataFrame,
    *,
    group_col: str,
) -> pd.DataFrame:
    cluster_cols = [PARTICIPANT_NAME_COL, group_col, "is_tail", "hdbscan_label"]
    merged = point_metrics.merge(
        participant_df[cluster_cols],
        on=[PARTICIPANT_NAME_COL, group_col],
        how="left",
    )
    merged["is_tail"] = merged["is_tail"].fillna(False).astype(float)
    return merged


def run_diversity_inference_analysis(
    point_metrics: pd.DataFrame,
    participant_df: pd.DataFrame,
    pairwise_summary_df: pd.DataFrame,
    space_outdir: str,
    *,
    group_col: str,
    distance_col: str,
    suffix: str,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Bootstrap CIs and pairwise tests for prediction-2 diversity metrics."""
    if group_col == COLLAPSED_PARTICIPANT_TYPE_COL:
        display_names_map = DISPLAY_LABELS
        preferred_order = GROUP_ORDER_COLLAPSED
    else:
        display_names_map = PARTICIPANT_TYPE_TO_LEGEND
        preferred_order = GROUP_ORDER

    merged = merge_point_metrics_with_clustering(
        point_metrics, participant_df, group_col=group_col
    )
    groups = [g for g in preferred_order if g in merged[group_col].unique()]
    groups += [g for g in merged[group_col].unique() if g not in groups]

    summary_rows: List[dict] = []
    for group in groups:
        centroid_vals = group_metric_values(merged, group_col, group, distance_col)
        centrality_vals = group_metric_values(
            merged, group_col, group, "within_group_centrality"
        )
        tail_vals = group_metric_values(merged, group_col, group, "is_tail")

        pooled_row = pairwise_summary_df.loc[
            pairwise_summary_df["group"] == group
        ]
        pooled_mean_sim = (
            float(pooled_row["mean_pairwise_similarity"].iloc[0])
            if len(pooled_row)
            else np.nan
        )
        pooled_var_sim = (
            float(pooled_row["var_pairwise_similarity"].iloc[0])
            if len(pooled_row)
            else np.nan
        )

        c_lo, c_hi = bootstrap_mean_ci(centroid_vals, seed=seed)
        s_lo, s_hi = bootstrap_mean_ci(centrality_vals, seed=seed + 1)
        t_lo, t_hi = bootstrap_mean_ci(tail_vals, seed=seed + 2)

        summary_rows.append({
            "group": group,
            "display_label": display_names_map.get(group, group),
            "n": len(centroid_vals),
            "mean_centroid_distance": float(np.mean(centroid_vals)),
            "centroid_distance_ci_low": c_lo,
            "centroid_distance_ci_high": c_hi,
            "mean_within_group_centrality": float(np.mean(centrality_vals)),
            "within_group_centrality_ci_low": s_lo,
            "within_group_centrality_ci_high": s_hi,
            "tail_pct": 100.0 * float(np.mean(tail_vals)),
            "tail_pct_ci_low": 100.0 * t_lo,
            "tail_pct_ci_high": 100.0 * t_hi,
            "mean_pairwise_similarity_pooled": pooled_mean_sim,
            "var_pairwise_similarity_pooled": pooled_var_sim,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(
        os.path.join(
            space_outdir,
            Q4_DIVERSITY_SUMMARY_CSV.format(suffix=suffix),
        ),
        index=False,
        encoding="utf-8-sig",
    )

    pairwise_rows: List[dict] = []
    for metric_name, col in (
        ("centroid_distance", distance_col),
        ("within_group_centrality", "within_group_centrality"),
        ("tail_indicator", "is_tail"),
    ):
        pairwise_rows.extend(
            build_pairwise_inference_rows(
                merged,
                col,
                groups,
                group_col=group_col,
                metric_name=metric_name,
                seed=seed,
            )
        )

    pairwise_df = pd.DataFrame(pairwise_rows)
    pairwise_df.to_csv(
        os.path.join(
            space_outdir,
            Q4_DIVERSITY_PAIRWISE_CSV.format(suffix=suffix),
        ),
        index=False,
        encoding="utf-8-sig",
    )
    print(f"\nQ4 diversity inference ({suffix or '3 groups'}):")
    print(summary_df.to_string(index=False))
    print(pairwise_df.to_string(index=False))
    return summary_df, pairwise_df


# ---------------------------------------------------------------------
# One embedding space
# ---------------------------------------------------------------------

def visualize_one_space(
    df: pd.DataFrame,
    embedding_col: str,
    space_outdir: str,
    network_outpath: str,
    network_collapsed_outpath: str,
    threshold_quantile: float,
    seed: int,
    embedding_set_label_text: str,
) -> None:
    os.makedirs(space_outdir, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"Visualizing embedding space: {embedding_col}")
    print("=" * 80)

    X_raw = stack_embeddings(df, embedding_col)
    X = normalize(X_raw)

    degree, similarity_threshold = semantic_neighbor_degree(
        X,
        labels=df[PARTICIPANT_TYPE_COL].values,
        threshold_quantile=threshold_quantile,
    )

    point_metrics = make_point_metrics(df, X, degree)

    pairwise_summary_3g = compute_pairwise_similarity_summary(df, X)
    pairwise_summary_collapsed = compute_pairwise_similarity_summary(
        df, X, group_col=COLLAPSED_PARTICIPANT_TYPE_COL
    )

    clustering_results = run_semantic_clustering_analysis(
        df=df,
        X=X,
        space_outdir=space_outdir,
        embedding_set_label_text=embedding_set_label_text,
        embedding_col=embedding_col,
    )

    if clustering_results:
        if "" in clustering_results:
            run_diversity_inference_analysis(
                point_metrics=point_metrics,
                participant_df=clustering_results[""]["participant_df"],
                pairwise_summary_df=pairwise_summary_3g,
                space_outdir=space_outdir,
                group_col=PARTICIPANT_TYPE_COL,
                distance_col="distance_to_group_centroid",
                suffix="",
                seed=seed,
            )
        if "_collapsed" in clustering_results:
            run_diversity_inference_analysis(
                point_metrics=point_metrics,
                participant_df=clustering_results["_collapsed"]["participant_df"],
                pairwise_summary_df=pairwise_summary_collapsed,
                space_outdir=space_outdir,
                group_col=COLLAPSED_PARTICIPANT_TYPE_COL,
                distance_col="distance_to_collapsed_group_centroid",
                suffix="_collapsed",
                seed=seed,
            )

    plot_group_threshold_networks(
        df=df,
        X=X,
        similarity_threshold=similarity_threshold,
        threshold_quantile=threshold_quantile,
        embedding_set_label_text=embedding_set_label_text,
        outpath=network_outpath,
        seed=seed,
    )

    plot_group_threshold_networks(
        df=df,
        X=X,
        similarity_threshold=similarity_threshold,
        threshold_quantile=threshold_quantile,
        embedding_set_label_text=embedding_set_label_text,
        outpath=network_collapsed_outpath,
        seed=seed,
        group_col=COLLAPSED_PARTICIPANT_TYPE_COL,
    )

    print(f"Similarity threshold used: {similarity_threshold:.4f}")
    print("Saved CSV tables to:")
    print(space_outdir)
    print("Saved network figures to:")
    print(Path(network_outpath).parent)


def task_label_from_key(task_key: str) -> str:
    return " · ".join(
        format_embedding_set_part(part) for part in task_key.split("/")
    )


# ---------------------------------------------------------------------
# Batch phase grids (2×2 across four tasks, by pre-ML / post-ML)
# ---------------------------------------------------------------------

def discover_phase_task_dirs(
    embeddings_root: Path,
    phase: str,
) -> List[Tuple[str, Path]]:
    dirs: List[Tuple[str, Path]] = []
    for task_key in DIVERSITY_TASK_PANEL_ORDER:
        set_dir = embeddings_root / task_key / phase
        if (set_dir / "embeddings_wide.parquet").exists():
            dirs.append((task_key, set_dir))
    return dirs


def load_phase_task_bundle(
    set_dir: Path,
    embedding_col: str,
    *,
    threshold_quantile: float,
) -> dict:
    df = pd.read_parquet(set_dir / "embeddings_wide.parquet")
    X_raw = stack_embeddings(df, embedding_col)
    X = normalize(X_raw)
    degree, _ = semantic_neighbor_degree(
        X,
        labels=df[PARTICIPANT_TYPE_COL].values,
        threshold_quantile=threshold_quantile,
    )
    point_metrics = make_point_metrics(df, X, degree)
    _, clustering_3g = compute_semantic_clustering_tables(
        df, X, group_col=PARTICIPANT_TYPE_COL
    )
    _, clustering_collapsed = compute_semantic_clustering_tables(
        df, X, group_col=COLLAPSED_PARTICIPANT_TYPE_COL
    )
    return {
        "df": df,
        "X": X,
        "point_metrics": point_metrics,
        "clustering_3g": clustering_3g,
        "clustering_collapsed": clustering_collapsed,
    }


def make_phase_grid_axes(suptitle: str) -> Tuple[plt.Figure, np.ndarray]:
    fig, axes = plt.subplots(
        2,
        2,
        figsize=PHASE_GRID_FIGSIZE,
        gridspec_kw={
            "hspace": PHASE_GRID_ROW_GAP,
            "wspace": PHASE_GRID_COL_GAP,
        },
    )
    fig.suptitle(
        suptitle,
        fontweight="bold",
        fontsize=PHASE_GRID_SUPTITLE_FONTSIZE,
        y=PHASE_GRID_SUPTITLE_Y,
    )
    return fig, axes.ravel()


def phase_grid_layout_adjust(
    fig,
    *,
    footnote_lines: int = 0,
    bottom_extra: float = 0.0,
) -> None:
    fig.subplots_adjust(
        left=0.11,
        right=0.98,
        top=PHASE_GRID_PANEL_TOP,
        bottom=0.08
        + footnote_lines * PHASE_GRID_FOOTNOTE_LINE_STEP
        + bottom_extra,
        hspace=PHASE_GRID_ROW_GAP,
        wspace=PHASE_GRID_COL_GAP,
    )


def add_phase_grid_figure_legend(
    fig,
    handles: list,
    labels: list[str],
    *,
    ncol: int,
) -> None:
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, PHASE_GRID_LEGEND_Y),
        ncol=ncol,
        frameon=True,
        fontsize=PHASE_GRID_LEGEND_FONTSIZE,
        borderaxespad=0.0,
    )


def phase_grid_group_counts(df: pd.DataFrame, *, collapse_human: bool) -> dict[str, int]:
    if collapse_human:
        plot_df = with_collapsed_group(df)
        group_col = COLLAPSED_PARTICIPANT_TYPE_COL
        groups = GROUP_ORDER_COLLAPSED
    else:
        plot_df = df
        group_col = PARTICIPANT_TYPE_COL
        groups = GROUP_ORDER
    counts = plot_df[group_col].value_counts().to_dict()
    return {g: int(counts.get(g, 0)) for g in groups}


def phase_grid_group_legend_handles_labels(
    *,
    collapse_human: bool,
    n_by_group: dict[str, int],
) -> Tuple[list, list[str], int]:
    if collapse_human:
        groups = GROUP_ORDER_COLLAPSED
        color_map = GROUP_COLORS_COLLAPSED
        labels = [
            legend_entry(
                g,
                n_by_group.get(g, 0),
                include_composition=(g == "Human"),
            )
            for g in groups
        ]
    else:
        groups = GROUP_ORDER
        color_map = GROUP_COLORS_BY_PARTICIPANT_TYPE
        labels = [
            legend_entry(
                PARTICIPANT_TYPE_TO_LEGEND.get(g, g),
                n_by_group.get(g, 0),
            )
            for g in groups
        ]
    handles = [
        Patch(
            facecolor=color_map[g],
            edgecolor=BAR_EDGE_COLOR,
            linewidth=BAR_EDGE_WIDTH,
            alpha=0.88,
        )
        for g in groups
    ]
    return handles, labels, len(groups)


def phase_grid_clustering_legend_handles_labels(
    *,
    collapse_human: bool,
) -> Tuple[list, list[str], int]:
    if collapse_human:
        groups = GROUP_ORDER_COLLAPSED
        color_map = GROUP_COLORS_COLLAPSED
        labels = [f"{display_label(g)} core" for g in groups]
    else:
        groups = GROUP_ORDER
        color_map = GROUP_COLORS_BY_PARTICIPANT_TYPE
        labels = [
            f"{PARTICIPANT_TYPE_TO_LEGEND.get(g, g)} core" for g in groups
        ]
    handles = [
        Patch(
            facecolor=color_map[g],
            edgecolor=BAR_EDGE_COLOR,
            linewidth=BAR_EDGE_WIDTH,
        )
        for g in groups
    ]
    handles.append(
        Patch(
            facecolor=CLUSTER_TAIL_COLOR,
            edgecolor=BAR_EDGE_COLOR,
            linewidth=BAR_EDGE_WIDTH,
        )
    )
    labels.append("Semantic tail")
    return handles, labels, len(handles)


def draw_phase_grid_footnote(fig, lines: Tuple[str, ...]) -> None:
    for i, line in enumerate(lines):
        fig.text(
            0.5,
            PHASE_GRID_FOOTNOTE_Y - i * PHASE_GRID_FOOTNOTE_LINE_STEP,
            line,
            ha="center",
            va="bottom",
            fontsize=PHASE_GRID_FOOTNOTE_FONTSIZE,
            color=FOOTNOTE_COLOR,
            transform=fig.transFigure,
            clip_on=False,
        )


def draw_semantic_map_panel(
    ax,
    df: pd.DataFrame,
    coords: np.ndarray,
    *,
    collapse_human: bool,
    panel_title: str,
) -> None:
    if collapse_human:
        plot_df = with_collapsed_group(df)
        group_col = COLLAPSED_PARTICIPANT_TYPE_COL
        colors_map = GROUP_COLORS_COLLAPSED
    else:
        plot_df = df
        group_col = PARTICIPANT_TYPE_COL
        colors_map = GROUP_COLORS_BY_PARTICIPANT_TYPE

    groups = ordered_groups(plot_df, group_col)

    for group in groups:
        mask = plot_df[group_col].values == group
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=colors_map.get(group, "#888888"),
            s=PHASE_GRID_SCATTER_SIZE,
            alpha=SCATTER_ALPHA,
            edgecolors=BAR_EDGE_COLOR,
            linewidths=BAR_EDGE_WIDTH,
        )

    apply_semantic_map_2d_limits(ax, coords, "PCA")
    ax.set_title(
        panel_title,
        fontweight="bold",
        fontsize=PHASE_GRID_PANEL_TITLE_FONTSIZE,
        pad=8,
    )
    ax.tick_params(axis="both", labelsize=PHASE_GRID_TICK_FONTSIZE)
    ax.grid(alpha=0.2, zorder=0)


def draw_centroid_boxplot_panel(
    ax,
    plot_data: dict,
    *,
    panel_title: str,
) -> None:
    color_map = plot_data["color_map"]
    display_names_map = plot_data["display_names_map"]
    group_col = plot_data["group_col"]
    group_series = plot_data["group_series"]

    x = np.arange(len(group_series))
    box_vals = [vals for _, vals in group_series]
    bp = ax.boxplot(
        box_vals,
        positions=x,
        widths=CENTROID_BOX_WIDTH * 0.85,
        patch_artist=True,
        showfliers=True,
        showmeans=True,
        whis=1.5,
        medianprops={
            "color": BOXPLOT_STAT_COLOR,
            "linewidth": 1.2,
            "linestyle": "--",
        },
        meanprops={
            "marker": "^",
            "markerfacecolor": BOXPLOT_STAT_COLOR,
            "markeredgecolor": BOXPLOT_STAT_COLOR,
            "markersize": 5,
        },
        whiskerprops={"color": BOXPLOT_EDGE_COLOR, "linewidth": 1.1},
        capprops={"color": BOXPLOT_EDGE_COLOR, "linewidth": 1.1},
        boxprops={"linewidth": 1.1, "edgecolor": BOXPLOT_EDGE_COLOR},
        flierprops={
            "marker": "o",
            "markersize": 3.5,
            "markerfacecolor": "white",
            "markeredgecolor": BOXPLOT_EDGE_COLOR,
            "alpha": 0.9,
        },
    )
    for patch, (group, _) in zip(bp["boxes"], group_series):
        patch.set_facecolor(color_map[group])
        patch.set_alpha(0.55)
        patch.set_edgecolor(BOXPLOT_EDGE_COLOR)

    xticks: List[str] = []
    for group, vals in group_series:
        if group_col == COLLAPSED_PARTICIPANT_TYPE_COL:
            xticks.append(
                legend_entry(
                    display_names_map.get(group, group),
                    len(vals),
                    include_composition=(group == "Human"),
                )
            )
        else:
            xticks.append(
                legend_entry(display_names_map.get(group, group), len(vals))
            )

    ax.set_xticks(x)
    ax.set_xticklabels(xticks, fontsize=PHASE_GRID_TICK_FONTSIZE - 0.5)
    ax._viz_bold_xticks = True
    style_axes(ax)
    ax.tick_params(axis="x", pad=8)
    ax.set_title(
        panel_title,
        fontweight="bold",
        fontsize=PHASE_GRID_PANEL_TITLE_FONTSIZE,
        pad=8,
    )
    all_vals = np.concatenate(box_vals)
    data_hi = float(np.max(all_vals))
    data_lo = float(np.min(all_vals))
    y_pad = max((data_hi - data_lo) * 0.08, 0.008)
    ax.set_ylim(bottom=max(0.0, data_lo - y_pad), top=data_hi + y_pad)


def draw_centroid_distribution_panel(
    ax,
    plot_data: dict,
    *,
    panel_title: str,
) -> None:
    color_map = plot_data["color_map"]
    group_series = plot_data["group_series"]

    for group, vals in group_series:
        ax.hist(
            vals,
            bins=histogram_bin_count(len(vals)),
            alpha=0.55,
            density=True,
            color=color_map[group],
            edgecolor=BAR_EDGE_COLOR,
            linewidth=BAR_EDGE_WIDTH * 0.6,
        )

    style_axes(ax)
    ax.set_title(
        panel_title,
        fontweight="bold",
        fontsize=PHASE_GRID_PANEL_TITLE_FONTSIZE,
        pad=8,
    )
    ax.tick_params(axis="both", labelsize=PHASE_GRID_TICK_FONTSIZE)


def draw_clustering_core_tail_panel(
    ax,
    summary_df: pd.DataFrame,
    *,
    group_col: str,
    panel_title: str,
) -> None:
    if group_col == COLLAPSED_PARTICIPANT_TYPE_COL:
        preferred_order = GROUP_ORDER_COLLAPSED
        color_map = GROUP_COLORS_COLLAPSED
        display_names_map = DISPLAY_LABELS
    else:
        preferred_order = GROUP_ORDER
        color_map = GROUP_COLORS_BY_PARTICIPANT_TYPE
        display_names_map = PARTICIPANT_TYPE_TO_LEGEND

    plot_df = summary_df.set_index("participant_type")
    groups = [g for g in preferred_order if g in plot_df.index]
    groups += [g for g in plot_df.index if g not in groups]

    core_pcts = [float(plot_df.loc[g, "core_pct"]) for g in groups]
    tail_pcts = [float(plot_df.loc[g, "tail_pct"]) for g in groups]
    display_names = [display_names_map.get(g, g) for g in groups]

    x = np.arange(len(groups))
    bar_width = CLUSTERING_BAR_WIDTH * 0.85
    ax.bar(
        x,
        core_pcts,
        width=bar_width,
        color=[color_map[g] for g in groups],
        edgecolor=BAR_EDGE_COLOR,
        linewidth=BAR_EDGE_WIDTH,
        zorder=3,
    )
    ax.bar(
        x,
        tail_pcts,
        width=bar_width,
        bottom=core_pcts,
        color=CLUSTER_TAIL_COLOR,
        edgecolor=BAR_EDGE_COLOR,
        linewidth=BAR_EDGE_WIDTH,
        zorder=3,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, fontsize=PHASE_GRID_TICK_FONTSIZE)
    ax._viz_bold_xticks = True
    style_axes(ax)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_locator(MultipleLocator(20))
    ax.set_title(
        panel_title,
        fontweight="bold",
        fontsize=PHASE_GRID_PANEL_TITLE_FONTSIZE,
        pad=8,
    )
    ax.tick_params(axis="both", labelsize=PHASE_GRID_TICK_FONTSIZE)
    ax.grid(axis="y", alpha=0.25)


def plot_phase_grid_semantic_maps(
    task_bundles: List[Tuple[str, dict]],
    outpath: Path,
    *,
    phase_label: str,
    collapse_human: bool,
    seed: int,
    umap_neighbors: int,
    umap_min_dist: float,
) -> None:
    fig, axes = make_phase_grid_axes(
        f"Semantic map of theoretical explanations ({phase_label})"
    )
    for ax, (task_key, bundle) in zip(axes, task_bundles):
        coords, _ = compute_projection(
            X=bundle["X"],
            method="pca",
            seed=seed,
            n_neighbors=umap_neighbors,
            min_dist=umap_min_dist,
            n_components=2,
        )
        draw_semantic_map_panel(
            ax,
            bundle["df"],
            coords,
            collapse_human=collapse_human,
            panel_title=task_label_from_key(task_key),
        )

    sample_df = task_bundles[0][1]["df"]
    n_by_group = phase_grid_group_counts(sample_df, collapse_human=collapse_human)
    handles, labels, ncol = phase_grid_group_legend_handles_labels(
        collapse_human=collapse_human,
        n_by_group=n_by_group,
    )
    add_phase_grid_figure_legend(fig, handles, labels, ncol=ncol)

    axis_label = "PCA dimension"
    fig.supxlabel(
        f"{axis_label} 1",
        fontsize=PHASE_GRID_AXIS_FONTSIZE,
        fontweight="bold",
        y=0.045,
    )
    fig.supylabel(
        f"{axis_label} 2",
        fontsize=PHASE_GRID_AXIS_FONTSIZE,
        fontweight="bold",
        x=PHASE_GRID_SUPYLABEL_X,
    )
    phase_grid_layout_adjust(fig)
    fig.savefig(outpath, dpi=SAVE_DPI, pad_inches=SAVE_PAD_INCHES)
    plt.close(fig)


def plot_phase_grid_centroid_boxplots(
    task_bundles: List[Tuple[str, dict]],
    outpath: Path,
    *,
    phase_label: str,
    collapse_human: bool,
) -> None:
    group_col = (
        COLLAPSED_PARTICIPANT_TYPE_COL
        if collapse_human
        else PARTICIPANT_TYPE_COL
    )
    distance_col = (
        "distance_to_collapsed_group_centroid"
        if collapse_human
        else "distance_to_group_centroid"
    )
    fig, axes = make_phase_grid_axes(
        f"Within-group distance to centroid ({phase_label})"
    )
    for ax, (task_key, bundle) in zip(axes, task_bundles):
        plot_data = prepare_centroid_distance_plot_data(
            bundle["point_metrics"],
            distance_col,
            group_col=group_col,
        )
        draw_centroid_boxplot_panel(
            ax,
            plot_data,
            panel_title=task_label_from_key(task_key),
        )

    sample_df = task_bundles[0][1]["df"]
    n_by_group = phase_grid_group_counts(sample_df, collapse_human=collapse_human)
    handles, labels, ncol = phase_grid_group_legend_handles_labels(
        collapse_human=collapse_human,
        n_by_group=n_by_group,
    )
    add_phase_grid_figure_legend(fig, handles, labels, ncol=ncol)

    fig.supylabel(
        "Cosine distance to group mean vector",
        fontsize=PHASE_GRID_AXIS_FONTSIZE,
        fontweight="bold",
        x=PHASE_GRID_SUPYLABEL_X,
    )
    footnote_lines = len(CENTROID_MEAN_SIG_FOOTNOTE)
    phase_grid_layout_adjust(fig, footnote_lines=footnote_lines)
    draw_phase_grid_footnote(fig, CENTROID_MEAN_SIG_FOOTNOTE)
    fig.savefig(outpath, dpi=SAVE_DPI, pad_inches=SAVE_PAD_INCHES)
    plt.close(fig)


def plot_phase_grid_centroid_distributions(
    task_bundles: List[Tuple[str, dict]],
    outpath: Path,
    *,
    phase_label: str,
    collapse_human: bool,
) -> None:
    group_col = (
        COLLAPSED_PARTICIPANT_TYPE_COL
        if collapse_human
        else PARTICIPANT_TYPE_COL
    )
    distance_col = (
        "distance_to_collapsed_group_centroid"
        if collapse_human
        else "distance_to_group_centroid"
    )
    fig, axes = make_phase_grid_axes(
        f"Within-group centroid-distance distribution ({phase_label})"
    )
    for ax, (task_key, bundle) in zip(axes, task_bundles):
        plot_data = prepare_centroid_distance_plot_data(
            bundle["point_metrics"],
            distance_col,
            group_col=group_col,
        )
        draw_centroid_distribution_panel(
            ax,
            plot_data,
            panel_title=task_label_from_key(task_key),
        )

    sample_df = task_bundles[0][1]["df"]
    n_by_group = phase_grid_group_counts(sample_df, collapse_human=collapse_human)
    handles, labels, ncol = phase_grid_group_legend_handles_labels(
        collapse_human=collapse_human,
        n_by_group=n_by_group,
    )
    add_phase_grid_figure_legend(fig, handles, labels, ncol=ncol)

    fig.supylabel(
        "Density",
        fontsize=PHASE_GRID_AXIS_FONTSIZE,
        fontweight="bold",
        x=PHASE_GRID_SUPYLABEL_X,
    )
    fig.supxlabel(
        "Cosine distance to group mean vector",
        fontsize=PHASE_GRID_AXIS_FONTSIZE,
        fontweight="bold",
        y=0.045,
    )
    phase_grid_layout_adjust(fig, bottom_extra=0.02)
    fig.savefig(outpath, dpi=SAVE_DPI, pad_inches=SAVE_PAD_INCHES)
    plt.close(fig)


def plot_phase_grid_clustering_core_tail(
    task_bundles: List[Tuple[str, dict]],
    outpath: Path,
    *,
    phase_label: str,
    collapse_human: bool,
) -> None:
    group_col = (
        COLLAPSED_PARTICIPANT_TYPE_COL
        if collapse_human
        else PARTICIPANT_TYPE_COL
    )
    fig, axes = make_phase_grid_axes(
        f"Within-group core vs. tail (HDBSCAN) ({phase_label})"
    )
    for ax, (task_key, bundle) in zip(axes, task_bundles):
        summary_df = (
            bundle["clustering_collapsed"]
            if collapse_human
            else bundle["clustering_3g"]
        )
        draw_clustering_core_tail_panel(
            ax,
            summary_df,
            group_col=group_col,
            panel_title=task_label_from_key(task_key),
        )

    handles, labels, ncol = phase_grid_clustering_legend_handles_labels(
        collapse_human=collapse_human,
    )
    add_phase_grid_figure_legend(fig, handles, labels, ncol=ncol)

    fig.supylabel(
        "Share of respondents (%)",
        fontsize=PHASE_GRID_AXIS_FONTSIZE,
        fontweight="bold",
        x=PHASE_GRID_SUPYLABEL_X,
    )
    phase_grid_layout_adjust(fig)
    fig.savefig(outpath, dpi=SAVE_DPI, pad_inches=SAVE_PAD_INCHES)
    plt.close(fig)


def run_phase_grid_visualizations(
    embeddings_root: Path,
    embedding_col: str,
    *,
    threshold_quantile: float = THRESHOLD_QUANTILE,
    seed: int = ANALYSIS_SEED,
    umap_neighbors: int = UMAP_NEIGHBORS,
    umap_min_dist: float = UMAP_MIN_DIST,
) -> None:
    phase_label_map = {"pre-ML": "pre-data", "post-ML": "post-data"}
    for phase in PHASE_NAMES:
        task_dirs = discover_phase_task_dirs(embeddings_root, phase)
        if len(task_dirs) != len(DIVERSITY_TASK_PANEL_ORDER):
            found = [key for key, _ in task_dirs]
            missing = [k for k in DIVERSITY_TASK_PANEL_ORDER if k not in found]
            print(
                f"Skipping {phase} phase grid: expected "
                f"{len(DIVERSITY_TASK_PANEL_ORDER)} tasks, "
                f"missing {missing}."
            )
            continue

        task_bundles: List[Tuple[str, dict]] = []
        for task_key, set_dir in task_dirs:
            print(f"  Loading {phase} · {task_label_from_key(task_key)}")
            task_bundles.append((
                task_key,
                load_phase_task_bundle(
                    set_dir,
                    embedding_col,
                    threshold_quantile=threshold_quantile,
                ),
            ))

        outdir = batch_phase_dir(embeddings_root, phase)
        outdir.mkdir(parents=True, exist_ok=True)
        phase_label = phase_label_map.get(phase, phase)
        emb_name = safe_name(embedding_col)

        plot_phase_grid_semantic_maps(
            task_bundles,
            outdir / f"01_semantic_space_map_{emb_name}.png",
            phase_label=phase_label,
            collapse_human=False,
            seed=seed,
            umap_neighbors=umap_neighbors,
            umap_min_dist=umap_min_dist,
        )
        plot_phase_grid_semantic_maps(
            task_bundles,
            outdir / f"01_semantic_space_map_collapsed_{emb_name}.png",
            phase_label=phase_label,
            collapse_human=True,
            seed=seed,
            umap_neighbors=umap_neighbors,
            umap_min_dist=umap_min_dist,
        )
        plot_phase_grid_centroid_boxplots(
            task_bundles,
            outdir / f"02_distance_to_group_centroid_boxplot_{emb_name}.png",
            phase_label=phase_label,
            collapse_human=False,
        )
        plot_phase_grid_centroid_boxplots(
            task_bundles,
            outdir / f"02_distance_to_group_centroid_boxplot_collapsed_{emb_name}.png",
            phase_label=phase_label,
            collapse_human=True,
        )
        plot_phase_grid_centroid_distributions(
            task_bundles,
            outdir / f"03_distance_to_group_centroid_distribution_{emb_name}.png",
            phase_label=phase_label,
            collapse_human=False,
        )
        plot_phase_grid_centroid_distributions(
            task_bundles,
            outdir / f"03_distance_to_group_centroid_distribution_collapsed_{emb_name}.png",
            phase_label=phase_label,
            collapse_human=True,
        )
        plot_phase_grid_clustering_core_tail(
            task_bundles,
            outdir / f"06_semantic_clustering_core_tail_{emb_name}.png",
            phase_label=phase_label,
            collapse_human=False,
        )
        plot_phase_grid_clustering_core_tail(
            task_bundles,
            outdir / f"06_semantic_clustering_core_tail_collapsed_{emb_name}.png",
            phase_label=phase_label,
            collapse_human=True,
        )
        print(f"Saved {phase} phase grids to: {outdir}")


# ---------------------------------------------------------------------
# Plot 07 — Human vs GenAI diversity predictions (pre/post, collapsed)
# ---------------------------------------------------------------------

def discover_pre_post_task_pairs(embeddings_root: Path) -> List[Tuple[Path, Path, str]]:
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


def diversity_space_dir(
    embedding_set_dir: Path,
    embedding_col: str,
    embeddings_root: Path | None = None,
) -> Path:
    return resolve_output_dir(embedding_set_dir, embedding_col, embeddings_root)


def diversity_csv_paths(embedding_set_dir: Path, embedding_col: str) -> dict[str, Path]:
    collapsed = "_collapsed"
    space_dir = diversity_space_dir(embedding_set_dir, embedding_col)
    return {
        "summary": space_dir / Q4_DIVERSITY_SUMMARY_CSV.format(suffix=collapsed),
        "pairwise": space_dir / Q4_DIVERSITY_PAIRWISE_CSV.format(suffix=collapsed),
    }


def require_diversity_csvs(paths: dict[str, Path], embedding_set_dir: Path) -> None:
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing diversity CSV(s). Run analysis on this embedding set first:\n"
            f"  {embedding_set_dir}\n"
            + "\n".join(f"  - {p}" for p in missing)
        )


def summary_group_value(summary_df: pd.DataFrame, group: str, column: str) -> float:
    match = summary_df.loc[summary_df["group"] == group]
    if match.empty:
        raise ValueError(
            f"No row for {group!r} in diversity summary "
            f"(available: {summary_df['group'].tolist()})"
        )
    return float(match.iloc[0][column])


def human_vs_genai_pairwise_row(pairwise_df: pd.DataFrame) -> pd.Series:
    match = pairwise_df.loc[
        (pairwise_df["metric"] == "centroid_distance")
        & (pairwise_df["left_group"] == HUMAN_COLLAPSED_GROUP)
        & (pairwise_df["right_group"] == GENAI_COLLAPSED_GROUP)
    ]
    if match.empty:
        raise ValueError(
            "No Human vs GenAI centroid_distance row in diversity pairwise CSV."
        )
    return match.iloc[0]


def group_mean(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return np.nan
    return float(np.mean(arr))


def analyze_diversity_prediction_task(
    pre_dir: Path,
    post_dir: Path,
    task_key: str,
    embedding_col: str,
) -> dict:
    pre_dists = participant_collapsed_centroid_distances(pre_dir, embedding_col)
    post_dists = participant_collapsed_centroid_distances(post_dir, embedding_col)

    pre_human = group_mean(pre_dists[HUMAN_COLLAPSED_GROUP])
    pre_genai = group_mean(pre_dists[GENAI_COLLAPSED_GROUP])
    post_human = group_mean(post_dists[HUMAN_COLLAPSED_GROUP])
    post_genai = group_mean(post_dists[GENAI_COLLAPSED_GROUP])

    pre_p = p_value_welch_ttest_one_sided(
        pre_dists[HUMAN_COLLAPSED_GROUP],
        pre_dists[GENAI_COLLAPSED_GROUP],
        alternative="greater",
    )
    post_p = p_value_welch_ttest_one_sided(
        post_dists[HUMAN_COLLAPSED_GROUP],
        post_dists[GENAI_COLLAPSED_GROUP],
        alternative="greater",
    )
    pre_gap = abs(pre_human - pre_genai)
    post_gap = abs(post_human - post_genai)

    pre_human_more_dispersed = pre_human > pre_genai
    pre_p2a_supported = bool(pre_human_more_dispersed and pre_p < 0.05)
    post_human_more_dispersed = post_human > post_genai
    post_p2b_supported = bool(post_p >= 0.05)

    return {
        "task_key": task_key,
        "task_label": task_label_from_key(task_key),
        "embedding_col": embedding_col,
        "pre_human_mean_cosine_distance": pre_human,
        "pre_genai_mean_cosine_distance": pre_genai,
        "post_human_mean_cosine_distance": post_human,
        "post_genai_mean_cosine_distance": post_genai,
        "pre_human_minus_genai_gap": pre_human - pre_genai,
        "post_human_minus_genai_gap": post_human - post_genai,
        "pre_abs_gap": pre_gap,
        "post_abs_gap": post_gap,
        "abs_gap_delta_post_minus_pre": post_gap - pre_gap,
        "pre_welch_p_one_sided": pre_p,
        "post_welch_p_one_sided": post_p,
        "pre_significance": significance_label(pre_p),
        "post_significance": significance_label(post_p),
        "pre_human_more_dispersed_than_genai": pre_human_more_dispersed,
        "pre_p2a_supported": pre_p2a_supported,
        "post_human_more_dispersed_than_genai": post_human_more_dispersed,
        "post_p2b_supported": post_p2b_supported,
    }


def diversity_panel_ylim_top(panel_max: float) -> float:
    raw = panel_max * DIVERSITY_PRED_YLIM_TOP_PAD
    step = 0.02 if raw < 0.3 else 0.05
    return float(np.ceil(raw / step) * step)


def diversity_pred_xlim() -> tuple[float, float]:
    left = (
        DIVERSITY_PRED_PRE_X[0]
        - DIVERSITY_PRED_BAR_WIDTH / 2
        - DIVERSITY_PRED_X_MARGIN
    )
    right = (
        DIVERSITY_PRED_POST_X[1]
        + DIVERSITY_PRED_BAR_WIDTH / 2
        + DIVERSITY_PRED_X_MARGIN
    )
    return left, right


def draw_diversity_metric_subtitle(fig) -> None:
    fig.text(
        0.5,
        DIVERSITY_PRED_METRIC_SUBTITLE_Y,
        DIVERSITY_PRED_METRIC_SUBTITLE,
        ha="center",
        va="top",
        fontsize=DIVERSITY_PRED_METRIC_SUBTITLE_FONTSIZE,
        fontweight="normal",
        color=FOOTNOTE_COLOR,
        transform=fig.transFigure,
        clip_on=False,
    )


def draw_diversity_prediction_footnote(
    fig, y: float = DIVERSITY_PRED_FOOTNOTE_Y
) -> None:
    for i, line in enumerate(DIVERSITY_PREDICTION_FOOTNOTE):
        fig.text(
            0.5,
            y - i * DIVERSITY_PRED_FOOTNOTE_LINE_STEP,
            line,
            ha="center",
            va="bottom",
            fontsize=DIVERSITY_PRED_FOOTNOTE_FONTSIZE,
            color=FOOTNOTE_COLOR,
            transform=fig.transFigure,
            clip_on=False,
        )


def plot_diversity_pre_post_predictions(
    summary_df: pd.DataFrame,
    outpath: Path,
) -> None:
    order = {key: i for i, key in enumerate(DIVERSITY_TASK_PANEL_ORDER)}
    plot_df = summary_df.sort_values(
        "task_key",
        key=lambda s: s.map(order),
    ).reset_index(drop=True)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=DIVERSITY_PRED_FIGSIZE,
        gridspec_kw={
            "hspace": DIVERSITY_PRED_ROW_GAP,
            "wspace": DIVERSITY_PRED_COL_GAP,
        },
    )
    axes_flat = axes.ravel()
    human_color = GROUP_COLORS_COLLAPSED[HUMAN_COLLAPSED_GROUP]
    genai_color = GROUP_COLORS_COLLAPSED[GENAI_COLLAPSED_GROUP]

    for ax, (_, row) in zip(axes_flat, plot_df.iterrows()):
        pre_vals = [
            float(row["pre_human_mean_cosine_distance"]),
            float(row["pre_genai_mean_cosine_distance"]),
        ]
        post_vals = [
            float(row["post_human_mean_cosine_distance"]),
            float(row["post_genai_mean_cosine_distance"]),
        ]
        panel_max = max(pre_vals + post_vals)
        ylim_top = diversity_panel_ylim_top(panel_max)

        ax.bar(
            DIVERSITY_PRED_PRE_X[0],
            pre_vals[0],
            DIVERSITY_PRED_BAR_WIDTH,
            color=human_color,
            edgecolor=BAR_EDGE_COLOR,
            linewidth=BAR_EDGE_WIDTH,
        )
        ax.bar(
            DIVERSITY_PRED_PRE_X[1],
            pre_vals[1],
            DIVERSITY_PRED_BAR_WIDTH,
            color=genai_color,
            edgecolor=BAR_EDGE_COLOR,
            linewidth=BAR_EDGE_WIDTH,
        )
        ax.bar(
            DIVERSITY_PRED_POST_X[0],
            post_vals[0],
            DIVERSITY_PRED_BAR_WIDTH,
            color=human_color,
            edgecolor=BAR_EDGE_COLOR,
            linewidth=BAR_EDGE_WIDTH,
        )
        ax.bar(
            DIVERSITY_PRED_POST_X[1],
            post_vals[1],
            DIVERSITY_PRED_BAR_WIDTH,
            color=genai_color,
            edgecolor=BAR_EDGE_COLOR,
            linewidth=BAR_EDGE_WIDTH,
        )

        ax.set_xticks(
            [DIVERSITY_PRED_PRE_X.mean(), DIVERSITY_PRED_POST_X.mean()]
        )
        ax.set_xticklabels(
            ["Pre-data", "Post-data"],
            fontsize=DIVERSITY_PRED_XTICK_FONTSIZE,
        )
        ax.set_title(
            row["task_label"],
            fontweight="bold",
            fontsize=DIVERSITY_PRED_PANEL_TITLE_FONTSIZE,
            pad=10,
        )
        ax.tick_params(axis="y", labelsize=DIVERSITY_PRED_YTICK_FONTSIZE)
        ax.set_xlim(*diversity_pred_xlim())
        ax.set_ylim(0.0, ylim_top)
        ax.set_box_aspect(DIVERSITY_PRED_BOX_ASPECT)
        ax.grid(axis="y", alpha=0.25)

    for ax, (_, row) in zip(axes_flat, plot_df.iterrows()):
        pre_vals = [
            float(row["pre_human_mean_cosine_distance"]),
            float(row["pre_genai_mean_cosine_distance"]),
        ]
        post_vals = [
            float(row["post_human_mean_cosine_distance"]),
            float(row["post_genai_mean_cosine_distance"]),
        ]
        draw_paired_pre_post_bracket(
            ax,
            DIVERSITY_PRED_PRE_X[0],
            DIVERSITY_PRED_PRE_X[1],
            max(pre_vals),
            float(row["pre_welch_p_one_sided"]),
            fontsize=DIVERSITY_PRED_BRACKET_FONTSIZE,
        )
        draw_paired_pre_post_bracket(
            ax,
            DIVERSITY_PRED_POST_X[0],
            DIVERSITY_PRED_POST_X[1],
            max(post_vals),
            float(row["post_welch_p_one_sided"]),
            fontsize=DIVERSITY_PRED_BRACKET_FONTSIZE,
        )

    fig.supylabel(
        "Mean cosine distance to group centroid",
        fontweight="bold",
        x=DIVERSITY_PRED_YLABEL_X,
        fontsize=DIVERSITY_PRED_YLABEL_FONTSIZE,
    )
    footnote_lines = len(DIVERSITY_PREDICTION_FOOTNOTE)
    fig.subplots_adjust(
        left=0.12,
        right=0.98,
        top=0.80,
        bottom=0.08 + footnote_lines * DIVERSITY_PRED_FOOTNOTE_LINE_STEP,
        hspace=DIVERSITY_PRED_ROW_GAP,
    )
    fig.suptitle(
        DIVERSITY_PRED_SUPTITLE,
        fontweight="bold",
        fontsize=DIVERSITY_PRED_SUPTITLE_FONTSIZE,
        y=0.98,
    )
    draw_diversity_metric_subtitle(fig)
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=human_color, label=display_label(HUMAN_COLLAPSED_GROUP)),
        plt.Rectangle((0, 0), 1, 1, color=genai_color, label="GenAI"),
    ]
    fig.legend(
        legend_handles,
        [display_label(HUMAN_COLLAPSED_GROUP), "GenAI"],
        loc="upper center",
        ncol=2,
        frameon=True,
        fontsize=FONT_LEGEND,
        bbox_to_anchor=(0.5, 0.895),
        borderaxespad=0.0,
    )
    draw_diversity_prediction_footnote(fig)
    fig.savefig(outpath, dpi=SAVE_DPI, pad_inches=SAVE_PAD_INCHES)
    plt.close(fig)


def participant_collapsed_centroid_distance_tables(
    embedding_set_dir: Path,
    embedding_col: str,
) -> dict[str, pd.DataFrame]:
    """Per-respondent cosine distance to collapsed within-group centroid."""
    df = pd.read_parquet(embedding_set_dir / "embeddings_wide.parquet")
    X = normalize(stack_embeddings(df, embedding_col))
    plot_df = with_collapsed_group(df)
    collapsed = plot_df[COLLAPSED_PARTICIPANT_TYPE_COL].values
    names = plot_df[PARTICIPANT_NAME_COL].values
    out: dict[str, pd.DataFrame] = {}
    for group in (HUMAN_COLLAPSED_GROUP, GENAI_COLLAPSED_GROUP):
        idx = np.where(collapsed == group)[0]
        if len(idx) == 0:
            raise ValueError(f"No {group!r} rows in {embedding_set_dir}")
        centroid = group_centroid(X[idx])
        dists = cosine_distances(X[idx], centroid.reshape(1, -1)).ravel()
        out[group] = pd.DataFrame(
            {
                PARTICIPANT_NAME_COL: names[idx],
                "cosine_distance": dists,
            }
        )
    return out


def participant_collapsed_centroid_distances(
    embedding_set_dir: Path,
    embedding_col: str,
) -> dict[str, np.ndarray]:
    tables = participant_collapsed_centroid_distance_tables(
        embedding_set_dir, embedding_col
    )
    return {
        group: table["cosine_distance"].to_numpy(dtype=float)
        for group, table in tables.items()
    }


def paired_distance_deltas(
    pre_table: pd.DataFrame,
    post_table: pd.DataFrame,
) -> np.ndarray:
    merged = pre_table.merge(
        post_table,
        on=PARTICIPANT_NAME_COL,
        how="inner",
        suffixes=("_pre", "_post"),
    )
    if merged.empty:
        raise ValueError("No paired participants for phase-change distance deltas")
    return (
        merged["cosine_distance_post"].to_numpy(dtype=float)
        - merged["cosine_distance_pre"].to_numpy(dtype=float)
    )


def signed_human_genai_mean_gap(
    human_vals: np.ndarray,
    genai_vals: np.ndarray,
) -> float:
    return group_mean(human_vals) - group_mean(genai_vals)


def bootstrap_diff_mean_ci(
    a: np.ndarray,
    b: np.ndarray,
    *,
    n_boot: int = 5000,
    seed: int = ANALYSIS_SEED,
) -> tuple[float, float]:
    """Bootstrap 95% CI for mean(a) - mean(b)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sa = rng.choice(a, size=len(a), replace=True)
        sb = rng.choice(b, size=len(b), replace=True)
        boot[i] = sa.mean() - sb.mean()
    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def welch_gap_change_inference(
    human_pre: np.ndarray,
    genai_pre: np.ndarray,
    human_post: np.ndarray,
    genai_post: np.ndarray,
    human_deltas: np.ndarray,
    genai_deltas: np.ndarray,
) -> dict[str, float | bool]:
    """
    Welch test for Human–GenAI absolute mean-distance gap shrinkage across phases.

    Gap change equals mean(Human post−pre distance) − mean(GenAI post−pre distance).
    H1 (directional): gap shrinks (post |gap| < pre |gap|), tested as
    mean(human_deltas) < mean(genai_deltas).
    """
    human_pre = np.asarray(human_pre, dtype=float)
    genai_pre = np.asarray(genai_pre, dtype=float)
    human_post = np.asarray(human_post, dtype=float)
    genai_post = np.asarray(genai_post, dtype=float)
    human_deltas = np.asarray(human_deltas, dtype=float)
    genai_deltas = np.asarray(genai_deltas, dtype=float)

    gap_pre = signed_human_genai_mean_gap(human_pre, genai_pre)
    gap_post = signed_human_genai_mean_gap(human_post, genai_post)
    abs_gap_pre = abs(gap_pre)
    abs_gap_post = abs(gap_post)
    abs_gap_delta = abs_gap_post - abs_gap_pre
    signed_gap_delta = float(np.mean(human_deltas) - np.mean(genai_deltas))

    p_one_sided = p_value_welch_ttest_one_sided(
        human_deltas,
        genai_deltas,
        alternative="less",
    )
    ci_lo, ci_hi = bootstrap_diff_mean_ci(human_deltas, genai_deltas)
    gap_shrunk = abs_gap_delta < 0

    return {
        "signed_gap_pre": gap_pre,
        "signed_gap_post": gap_post,
        "signed_gap_delta_post_minus_pre": gap_post - gap_pre,
        "abs_gap_pre": abs_gap_pre,
        "abs_gap_post": abs_gap_post,
        "abs_gap_delta_post_minus_pre": abs_gap_delta,
        "gap_change_p_one_sided": p_one_sided,
        "gap_change_significance": significance_label(p_one_sided),
        "gap_change_ci_low": ci_lo,
        "gap_change_ci_high": ci_hi,
        "gap_shrunk_post_vs_pre": gap_shrunk,
        "gap_change_directional_supported": bool(gap_shrunk and p_one_sided < 0.05),
    }


def analyze_diversity_gap_change_task(
    pre_dir: Path,
    post_dir: Path,
    task_key: str,
    embedding_col: str,
) -> dict:
    pre_tables = participant_collapsed_centroid_distance_tables(pre_dir, embedding_col)
    post_tables = participant_collapsed_centroid_distance_tables(post_dir, embedding_col)
    pre_dists = {
        group: table["cosine_distance"].to_numpy(dtype=float)
        for group, table in pre_tables.items()
    }
    post_dists = {
        group: table["cosine_distance"].to_numpy(dtype=float)
        for group, table in post_tables.items()
    }
    human_deltas = paired_distance_deltas(
        pre_tables[HUMAN_COLLAPSED_GROUP],
        post_tables[HUMAN_COLLAPSED_GROUP],
    )
    genai_deltas = paired_distance_deltas(
        pre_tables[GENAI_COLLAPSED_GROUP],
        post_tables[GENAI_COLLAPSED_GROUP],
    )
    inference = welch_gap_change_inference(
        pre_dists[HUMAN_COLLAPSED_GROUP],
        pre_dists[GENAI_COLLAPSED_GROUP],
        post_dists[HUMAN_COLLAPSED_GROUP],
        post_dists[GENAI_COLLAPSED_GROUP],
        human_deltas,
        genai_deltas,
    )
    return {
        "task_key": task_key,
        "task_label": task_label_from_key(task_key),
        "embedding_col": embedding_col,
        "n_human_pre": len(pre_dists[HUMAN_COLLAPSED_GROUP]),
        "n_genai_pre": len(pre_dists[GENAI_COLLAPSED_GROUP]),
        "n_human_post": len(post_dists[HUMAN_COLLAPSED_GROUP]),
        "n_genai_post": len(post_dists[GENAI_COLLAPSED_GROUP]),
        **inference,
    }


def gap_change_panel_ylim_top(panel_max: float, panel_min: float) -> float:
    span = max(panel_max - panel_min, 0.02)
    top = panel_max + span * (DIVERSITY_GAP_CHANGE_YLIM_PAD - 1.0)
    return float(np.ceil(top / 0.01) * 0.01)


def draw_gap_change_footnote(fig, y: float = DIVERSITY_PRED_FOOTNOTE_Y) -> None:
    for i, line in enumerate(DIVERSITY_GAP_CHANGE_FOOTNOTE):
        fig.text(
            0.5,
            y - i * DIVERSITY_PRED_FOOTNOTE_LINE_STEP,
            line,
            ha="center",
            va="bottom",
            fontsize=DIVERSITY_PRED_FOOTNOTE_FONTSIZE,
            color=FOOTNOTE_COLOR,
            transform=fig.transFigure,
            clip_on=False,
        )


def draw_gap_change_metric_subtitle(fig) -> None:
    fig.text(
        0.5,
        DIVERSITY_PRED_METRIC_SUBTITLE_Y,
        DIVERSITY_GAP_CHANGE_METRIC_SUBTITLE,
        ha="center",
        va="top",
        fontsize=DIVERSITY_PRED_METRIC_SUBTITLE_FONTSIZE,
        fontweight="normal",
        color=FOOTNOTE_COLOR,
        transform=fig.transFigure,
        clip_on=False,
    )


def plot_diversity_gap_change_pre_post(
    summary_df: pd.DataFrame,
    outpath: Path,
) -> None:
    order = {key: i for i, key in enumerate(DIVERSITY_TASK_PANEL_ORDER)}
    plot_df = summary_df.sort_values(
        "task_key",
        key=lambda s: s.map(order),
    ).reset_index(drop=True)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=DIVERSITY_PRED_FIGSIZE,
        gridspec_kw={
            "hspace": DIVERSITY_PRED_ROW_GAP,
            "wspace": DIVERSITY_PRED_COL_GAP,
        },
    )
    axes_flat = axes.ravel()

    for ax, (_, row) in zip(axes_flat, plot_df.iterrows()):
        pre_gap = float(row["abs_gap_pre"])
        post_gap = float(row["abs_gap_post"])
        panel_max = max(pre_gap, post_gap)
        ylim_top = gap_change_panel_ylim_top(panel_max, 0.0)

        ax.bar(
            DIVERSITY_GAP_CHANGE_BAR_X[0],
            pre_gap,
            DIVERSITY_GAP_CHANGE_BAR_WIDTH,
            label="Pre-data",
            color=DIVERSITY_GAP_CHANGE_PRE_COLOR,
            edgecolor=BAR_EDGE_COLOR,
            linewidth=BAR_EDGE_WIDTH,
        )
        ax.bar(
            DIVERSITY_GAP_CHANGE_BAR_X[1],
            post_gap,
            DIVERSITY_GAP_CHANGE_BAR_WIDTH,
            label="Post-data",
            color=DIVERSITY_GAP_CHANGE_POST_COLOR,
            edgecolor=BAR_EDGE_COLOR,
            linewidth=BAR_EDGE_WIDTH,
        )
        ax.set_xticks(DIVERSITY_GAP_CHANGE_BAR_X)
        ax.set_xticklabels(
            ["Pre-data", "Post-data"],
            fontsize=DIVERSITY_PRED_XTICK_FONTSIZE,
        )
        ax.set_title(
            row["task_label"],
            fontweight="bold",
            fontsize=DIVERSITY_PRED_PANEL_TITLE_FONTSIZE,
            pad=10,
        )
        ax.tick_params(axis="y", labelsize=DIVERSITY_PRED_YTICK_FONTSIZE)
        ax.set_ylim(0.0, ylim_top)
        ax.set_box_aspect(DIVERSITY_PRED_BOX_ASPECT)
        ax.grid(axis="y", alpha=0.25)

    for ax, (_, row) in zip(axes_flat, plot_df.iterrows()):
        pre_gap = float(row["abs_gap_pre"])
        post_gap = float(row["abs_gap_post"])
        draw_paired_pre_post_bracket(
            ax,
            DIVERSITY_GAP_CHANGE_BAR_X[0],
            DIVERSITY_GAP_CHANGE_BAR_X[1],
            max(pre_gap, post_gap),
            float(row["gap_change_p_one_sided"]),
            fontsize=DIVERSITY_PRED_BRACKET_FONTSIZE,
        )

    fig.supylabel(
        "|Humans − GenAI| mean cosine-distance gap",
        fontweight="bold",
        x=DIVERSITY_PRED_YLABEL_X,
        fontsize=DIVERSITY_PRED_YLABEL_FONTSIZE,
    )
    footnote_lines = len(DIVERSITY_GAP_CHANGE_FOOTNOTE)
    fig.subplots_adjust(
        left=0.12,
        right=0.98,
        top=0.80,
        bottom=0.08 + footnote_lines * DIVERSITY_PRED_FOOTNOTE_LINE_STEP,
        hspace=DIVERSITY_PRED_ROW_GAP,
    )
    fig.suptitle(
        DIVERSITY_GAP_CHANGE_SUPTITLE,
        fontweight="bold",
        fontsize=DIVERSITY_PRED_SUPTITLE_FONTSIZE,
        y=0.98,
    )
    draw_gap_change_metric_subtitle(fig)
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=True,
        fontsize=FONT_LEGEND,
        bbox_to_anchor=(0.5, 0.895),
        borderaxespad=0.0,
    )
    draw_gap_change_footnote(fig)
    fig.savefig(outpath, dpi=SAVE_DPI, pad_inches=SAVE_PAD_INCHES)
    plt.close(fig)


def run_diversity_gap_change_comparison(
    embeddings_root: Path,
    embedding_col: str,
    outdir: Path,
) -> pd.DataFrame:
    rows = []
    for pre_dir, post_dir, task_key in discover_pre_post_task_pairs(embeddings_root):
        print(f"\n=== Diversity gap change · {task_label_from_key(task_key)} ===")
        row = analyze_diversity_gap_change_task(
            pre_dir, post_dir, task_key, embedding_col
        )
        rows.append(row)
        print(
            f"  |Gap|: pre {row['abs_gap_pre']:.3f} → post {row['abs_gap_post']:.3f} "
            f"(Δ {row['abs_gap_delta_post_minus_pre']:+.3f}) | "
            f"Welch directional p: {row['gap_change_p_one_sided']:.4f} "
            f"{row['gap_change_significance']} | "
            f"Supported: {row['gap_change_directional_supported']}"
        )

    summary_df = pd.DataFrame(rows)
    csv_path = outdir / DIVERSITY_GAP_CHANGE_CSV
    fig_path = outdir / DIVERSITY_GAP_CHANGE_FILENAME
    summary_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    plot_diversity_gap_change_pre_post(summary_df, fig_path)

    n_supported = int(summary_df["gap_change_directional_supported"].sum())
    print(
        f"\nGap-change summary: directional support {n_supported}/{len(summary_df)} tasks"
    )
    print(f"Saved: {csv_path}")
    print(f"Saved: {fig_path}")
    return summary_df


def run_diversity_prediction_comparison(
    embeddings_root: Path,
    embedding_col: str,
) -> pd.DataFrame:
    outdir = comparisons_pre_post_dir(embeddings_root, COMPARISONS_DIVERSITY_SUBDIR)
    rows = []
    for pre_dir, post_dir, task_key in discover_pre_post_task_pairs(embeddings_root):
        print(f"\n=== Diversity predictions · {task_label_from_key(task_key)} ===")
        row = analyze_diversity_prediction_task(
            pre_dir, post_dir, task_key, embedding_col
        )
        rows.append(row)
        print(
            f"  Pre mean distance: Human {row['pre_human_mean_cosine_distance']:.4f} "
            f"vs GenAI {row['pre_genai_mean_cosine_distance']:.4f} "
            f"({row['pre_significance']}) | "
            f"Pre supported: {row['pre_p2a_supported']}"
        )
        print(
            f"  Post mean distance: Human {row['post_human_mean_cosine_distance']:.4f} "
            f"vs GenAI {row['post_genai_mean_cosine_distance']:.4f} "
            f"({row['post_significance']}) | "
            f"|gap| {row['pre_abs_gap']:.4f} → {row['post_abs_gap']:.4f} | "
            f"Post converged (NS) supported: {row['post_p2b_supported']}"
        )

    summary_df = pd.DataFrame(rows)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / DIVERSITY_PREDICTION_CSV
    fig_path = outdir / DIVERSITY_PREDICTION_FILENAME
    summary_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    plot_diversity_pre_post_predictions(summary_df, fig_path)

    print(f"\n--- Human–GenAI gap change (pre → post) ---")
    run_diversity_gap_change_comparison(embeddings_root, embedding_col, outdir)

    n_p2a = int(summary_df["pre_p2a_supported"].sum())
    n_p2b = int(summary_df["post_p2b_supported"].sum())
    n_tasks = len(summary_df)
    print(
        f"\nPrediction summary across {n_tasks} tasks: "
        f"pre Human > GenAI {n_p2a}/{n_tasks}, post NS (Human > GenAI) {n_p2b}/{n_tasks}"
    )
    print(f"Saved: {csv_path}")
    print(f"Saved: {fig_path}")
    return summary_df


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def run_visualizations_for_embedding_set(
    embedding_set_dir: Path,
    embeddings_root: Path,
) -> Path:
    input_path = embedding_set_dir / "embeddings_wide.parquet"
    set_label = embedding_set_label(embedding_set_dir)

    df = pd.read_parquet(input_path)
    columns = available_embedding_columns(df, DEFAULT_EMBEDDING_COLUMNS)

    required = [PARTICIPANT_NAME_COL, PARTICIPANT_TYPE_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {embedding_set_dir}: {missing}")

    last_outdir: Path | None = None
    for col in tqdm(columns, desc=f"{set_label} · {embedding_set_dir.name}"):
        space_outdir = resolve_task_data_dir(
            embeddings_root, embedding_set_dir, col
        )
        space_outdir.mkdir(parents=True, exist_ok=True)
        visualize_one_space(
            df=df,
            embedding_col=col,
            space_outdir=str(space_outdir),
            network_outpath=str(
                resolve_network_outpath(
                    embeddings_root, embedding_set_dir, col, collapsed=False
                )
            ),
            network_collapsed_outpath=str(
                resolve_network_outpath(
                    embeddings_root, embedding_set_dir, col, collapsed=True
                )
            ),
            threshold_quantile=THRESHOLD_QUANTILE,
            seed=ANALYSIS_SEED,
            embedding_set_label_text=set_label,
        )
        last_outdir = space_outdir

    assert last_outdir is not None
    print(f"Saved analysis outputs to: {last_outdir}")
    return last_outdir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embedding-set",
        required=True,
        help=(
            "Path to one embedding-set folder (with embeddings_wide.parquet), "
            "or a parent folder (e.g. embeddings_openai/) to process all sets below it."
        ),
    )

    args = parser.parse_args()

    batch_root = Path(args.embedding_set).expanduser()
    if not batch_root.is_absolute():
        batch_root = (Path.cwd() / batch_root).resolve()

    embedding_set_dirs = discover_embedding_set_dirs(args.embedding_set)
    is_batch = len(embedding_set_dirs) > 1
    embeddings_root = (
        batch_root
        if is_batch
        else infer_embeddings_root(embedding_set_dirs[0])
    )
    print(f"Processing {len(embedding_set_dirs)} embedding set(s).")
    for embedding_set_dir in embedding_set_dirs:
        set_label = embedding_set_label(embedding_set_dir)
        print(f"\n=== {set_label} ({embedding_set_dir}) ===")
        run_visualizations_for_embedding_set(embedding_set_dir, embeddings_root)

    if is_batch:
        sample_df = pd.read_parquet(
            embedding_set_dirs[0] / "embeddings_wide.parquet"
        )
        embedding_col = available_embedding_columns(
            sample_df, DEFAULT_EMBEDDING_COLUMNS
        )[0]
        print(f"\n=== Phase 2×2 grids ({embedding_col}) ===")
        run_phase_grid_visualizations(batch_root, embedding_col)

        print(f"\n=== Cross-phase diversity predictions ({embedding_col}) ===")
        try:
            comparison_outdir = comparisons_pre_post_dir(
                batch_root, COMPARISONS_DIVERSITY_SUBDIR
            )
            run_diversity_prediction_comparison(batch_root, embedding_col)
            print(f"Diversity prediction outputs: {comparison_outdir}")
        except FileNotFoundError as exc:
            print(f"Skipping diversity prediction figure: {exc}")

    print("\nDone.")
    print(f"All outputs saved under: {batch_visualizations_root(embeddings_root)}")


if __name__ == "__main__":
    main()