"""
Human evaluation of theory quality (EU / PP ratings).

For each cohort under human_rating_data/ (theory_ratings_<COHORT>_seed*.csv):
merge seeds, write a combined CSV, and plot Pre/Post scores:

1. PhD Students / Senior Scientists / Topic Experts / GenAI — by task × effect
   (Topic Experts bar overlaps PhD/Senior; bottom tests include
   Topic Experts vs Non-Topic Experts — Non-Topic is not drawn as a bar)
2. Humans / GenAI — by task × effect
   (bottom tests: Humans vs GenAI; Topic Experts vs Non-Topic Experts)
3. Same two group splits with all task × effect pooled

Figures go to outputs/<COHORT>/*.svg.

Duplicate handling
------------------
Independent seeds can resample the same theory
(participant_name, task, effect, phase). Overlapping ratings are kept as
independent samples (no averaging / deduplication).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HUMAN_EVAL_DIR = Path(__file__).resolve().parent
DATA_DIR = HUMAN_EVAL_DIR / "human_rating_data"
OUT_DIR = HUMAN_EVAL_DIR / "outputs"
TEXTUAL_DIR = HUMAN_EVAL_DIR.parents[2]  # textual_analysis/
ROOT = TEXTUAL_DIR.parent  # survey_april/
for p in (TEXTUAL_DIR, ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from stats_utils import bootstrap_mean_ci, p_value_welch_ttest  # noqa: E402
from viz_style import (  # noqa: E402
    BAR_ALPHA,
    BAR_EDGE_COLOR,
    BAR_EDGE_WIDTH,
    ERROR_CAPSIZE,
    ERROR_LINEWIDTH,
    GROUP_COLORS_COLLAPSED,
    GROUP_COLORS_TEXT,
    HUMAN_COMPOSITION_NOTE,
    PHASE_HATCH_COLOR,
    SAVE_DPI,
    SAVE_PAD_INCHES,
    apply_plot_style,
    comparison_pair_label,
    display_label,
    draw_pre_post_bracket,
    draw_pairwise_group_brackets,
    draw_pairwise_sig_legend,
    draw_pairwise_sig_color_legend,
    draw_pre_post_sig_columns,
    format_p_value_label,
    is_significant,
    set_axis_labels,
    significance_label,
    style_axes,
)


@dataclass(frozen=True)
class CohortSpec:
    key: str  # 'EU' | 'PP'
    seed_glob: str
    combined_csv_name: str

    @property
    def out_dir(self) -> Path:
        return OUT_DIR / self.key

    @property
    def combined_csv(self) -> Path:
        return DATA_DIR / self.combined_csv_name


COHORTS: tuple[CohortSpec, ...] = (
    CohortSpec(
        key="EU",
        seed_glob="theory_ratings_EU_seed*.csv",
        combined_csv_name="theory_ratings_EU_combined.csv",
    ),
    CohortSpec(
        key="PP",
        seed_glob="theory_ratings_PP_seed*.csv",
        combined_csv_name="theory_ratings_PP_combined.csv",
    ),
)

SCORE_DIMS = [
    "clarity_coherence",
    "causal_reasoning",
    "theoretical_depth",
    "creativity",
    "persuasiveness",
]
OVERALL_COL = "overall_quality"
THEORY_KEYS = ("participant_name", "task", "effect", "phase")
SURVEY_CSV = ROOT / "All_Participants_All_Questions.csv"
NAME_COLUMN = "What is your full name?"
TOPIC_EXPERT_COLUMN = "topic_expert"

NON_TOPIC_GROUP = "Non-Topic Experts"
GROUP_ORDER = ["PhD Students", "Senior Scientists", "Topic Experts", "GenAI"]
GROUP_ORDER_COLLAPSED = ["Human", "GenAI"]
# Bars + Non-Topic (tests only) + Human (collapsed bars).
VALUE_GROUPS = (
    "PhD Students",
    "Senior Scientists",
    "Topic Experts",
    NON_TOPIC_GROUP,
    "Human",
    "GenAI",
)

GROUP_MAP = {
    0: "PhD Students",
    1: "Senior Scientists",
    2: "GenAI",
}

PAIRWISE_THREE = (
    ("PhD Students", "GenAI"),
    ("Senior Scientists", "GenAI"),
    ("Senior Scientists", "PhD Students"),
    ("Topic Experts", NON_TOPIC_GROUP),
)
PAIRWISE_COLLAPSED = (
    ("Human", "GenAI"),
    ("Topic Experts", NON_TOPIC_GROUP),
)

PHASE_ORDER = ("pre", "post")
PHASE_LABELS = {"pre": "Pre-ML", "post": "Post-ML"}

PANEL_SPECS = (
    ("race", "main", "Racial inequality — Main effects"),
    ("race", "interactions", "Racial inequality — Interactions"),
    ("gender", "main", "Gender inequality — Main effects"),
    ("gender", "interactions", "Gender inequality — Interactions"),
)

SCORE_YMAX = 10
PLOT_YMAX = 12.2

apply_plot_style()


def compute_overall_quality(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    scores = out[SCORE_DIMS].apply(pd.to_numeric, errors="coerce")
    out[OVERALL_COL] = scores.mean(axis=1)
    return out


def attach_topic_expert(df: pd.DataFrame) -> pd.DataFrame:
    """Merge survey ``topic_expert`` onto ratings via participant name."""
    survey = pd.read_csv(SURVEY_CSV, dtype=str, keep_default_na=False)
    if NAME_COLUMN not in survey.columns or TOPIC_EXPERT_COLUMN not in survey.columns:
        raise KeyError(
            f"Survey CSV missing {NAME_COLUMN!r} or {TOPIC_EXPERT_COLUMN!r}"
        )
    lookup = (
        survey[[NAME_COLUMN, TOPIC_EXPERT_COLUMN]]
        .assign(_name_key=lambda d: d[NAME_COLUMN].astype(str).str.strip().str.lower())
        .drop_duplicates(subset=["_name_key"], keep="first")
        .set_index("_name_key")[TOPIC_EXPERT_COLUMN]
    )
    out = df.copy()
    keys = out["participant_name"].astype(str).str.strip().str.lower()
    out[TOPIC_EXPERT_COLUMN] = keys.map(lookup)
    missing = out[TOPIC_EXPERT_COLUMN].isna() | (
        out[TOPIC_EXPERT_COLUMN].astype(str).str.strip() == ""
    )
    if missing.any():
        names = sorted(out.loc[missing, "participant_name"].astype(str).unique())
        raise ValueError(
            f"Could not match topic_expert for {missing.sum()} rating rows; "
            f"unmatched names: {names[:10]}"
        )
    out[TOPIC_EXPERT_COLUMN] = out[TOPIC_EXPERT_COLUMN].astype(str).str.strip()
    return out


def audience_targets(group: object, topic_flag: object) -> list[str]:
    """Overlapping labels for bars + Topic/Non-Topic test buckets."""
    try:
        gid = int(group)
    except (TypeError, ValueError):
        return []
    gname = GROUP_MAP.get(gid)
    if gname is None:
        return []
    targets = [gname]
    if gname in ("PhD Students", "Senior Scientists"):
        targets.append("Human")
        flag = str(topic_flag).strip()
        if flag == "1":
            targets.append("Topic Experts")
        elif flag == "0":
            targets.append(NON_TOPIC_GROUP)
    return targets


def _normalize_frame(df: pd.DataFrame, *, source: str) -> pd.DataFrame:
    out = df.copy()
    for col in THEORY_KEYS:
        out[col] = out[col].astype(str).str.strip()
    out["task"] = out["task"].str.lower()
    out["effect"] = out["effect"].str.lower()
    out["phase"] = out["phase"].str.lower()
    out["group"] = pd.to_numeric(out["group"], errors="coerce").astype("Int64")
    for col in SCORE_DIMS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["source_seed"] = source
    if OVERALL_COL in out.columns:
        out = out.drop(columns=[OVERALL_COL])
    return out


def load_seed_frames(cohort: CohortSpec, data_dir: Path = DATA_DIR) -> list[pd.DataFrame]:
    paths = sorted(data_dir.glob(cohort.seed_glob))
    if not paths:
        raise FileNotFoundError(f"No {cohort.seed_glob} in {data_dir}")
    prefix = f"theory_ratings_{cohort.key}_seed"
    frames = []
    for path in paths:
        seed = path.stem.replace(prefix, "")
        frames.append(_normalize_frame(pd.read_csv(path), source=seed))
        print(f"Loaded {path.name}: n={len(frames[-1])}")
    return frames


def combine_seed_ratings(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate seeds; keep overlapping theories as independent samples."""
    combined = compute_overall_quality(pd.concat(frames, ignore_index=True))
    n_unique = combined.groupby(list(THEORY_KEYS), sort=False).ngroups
    n_overlap = len(combined) - n_unique
    print(
        f"Combined ratings: n={len(combined)} "
        f"(unique theories={n_unique}, overlap rows kept as independent={n_overlap})"
    )
    return combined


def merge_eu_pp_averaged(eu_df: pd.DataFrame, pp_df: pd.DataFrame) -> pd.DataFrame:
    """Merge EU and PP human ratings: overlap theories averaged, EU-only kept."""
    keys = list(THEORY_KEYS)
    parts = []
    for cohort, df in [("EU", eu_df), ("PP", pp_df)]:
        d = df.copy()
        d["cohort"] = cohort
        parts.append(d)
    both = pd.concat(parts, ignore_index=True)
    by_cohort = both.groupby(
        keys + ["cohort", "group", TOPIC_EXPERT_COLUMN],
        as_index=False,
    ).agg({OVERALL_COL: "mean"})
    return by_cohort.groupby(
        keys + ["group", TOPIC_EXPERT_COLUMN],
        as_index=False,
    ).agg({OVERALL_COL: "mean"})


def summarize(values: list[float] | np.ndarray) -> dict[str, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0, "mean": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    lo, hi = bootstrap_mean_ci(arr)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
    }


def panel_values(
    df: pd.DataFrame,
    task: str,
    effect: str,
) -> dict[str, dict[str, list[float]]]:
    sub = df.loc[(df["task"] == task) & (df["effect"] == effect)].copy()
    return _values_from_subset(sub)


def pooled_values(df: pd.DataFrame) -> dict[str, dict[str, list[float]]]:
    return _values_from_subset(df.copy())


def _values_from_subset(sub: pd.DataFrame) -> dict[str, dict[str, list[float]]]:
    out: dict[str, dict[str, list[float]]] = {
        phase: {g: [] for g in VALUE_GROUPS} for phase in PHASE_ORDER
    }
    for _, row in sub.iterrows():
        phase = str(row["phase"]).strip().lower()
        if phase not in out:
            continue
        score = row[OVERALL_COL]
        if pd.isna(score):
            continue
        val = float(score)
        for audience in audience_targets(row["group"], row[TOPIC_EXPERT_COLUMN]):
            out[phase][audience].append(val)
    return out


def _draw_panel(
    ax,
    values: dict[str, dict[str, list[float]]],
    title: str,
    *,
    group_order: list[str] | tuple[str, ...],
    group_colors: dict[str, str],
    pairwise: tuple[tuple[str, str], ...],
    note_fontsize: float = 10,
    notes_layout: str = "stacked",
) -> None:
    x = np.arange(len(group_order))
    width = 0.34
    offsets = {"pre": -width / 2, "post": width / 2}
    bar_tops: dict[str, float] = {}
    phase_bar_tops: dict[str, float] = {phase: 0.0 for phase in PHASE_ORDER}

    for gi, group in enumerate(group_order):
        means, yerr_lo, yerr_hi = [], [], []
        for phase in PHASE_ORDER:
            stats = summarize(values[phase][group])
            mean = float(stats["mean"])
            lo = float(stats["ci_low"])
            hi = float(stats["ci_high"])
            means.append(mean)
            yerr_lo.append(
                max(0.0, mean - lo) if np.isfinite(mean) and np.isfinite(lo) else 0.0
            )
            yerr_hi.append(
                max(0.0, hi - mean) if np.isfinite(mean) and np.isfinite(hi) else 0.0
            )
        finite = [
            means[i] + yerr_hi[i]
            for i in range(len(PHASE_ORDER))
            if np.isfinite(means[i])
        ]
        bar_tops[group] = max(finite) if finite else 0.0

        for i, phase in enumerate(PHASE_ORDER):
            xpos = float(x[gi] + offsets[phase])
            bar = ax.bar(
                [xpos],
                [means[i]],
                width=width,
                color=group_colors[group],
                alpha=BAR_ALPHA,
                edgecolor=BAR_EDGE_COLOR,
                linewidth=BAR_EDGE_WIDTH,
                zorder=2,
            )
            if phase == "post":
                bar[0].set_hatch("///")
            ax.errorbar(
                [xpos],
                [means[i]],
                yerr=[[yerr_lo[i]], [yerr_hi[i]]],
                fmt="none",
                ecolor="black",
                elinewidth=ERROR_LINEWIDTH,
                capsize=ERROR_CAPSIZE,
                zorder=3,
            )
            phase_bar_tops[phase] = max(
                phase_bar_tops[phase],
                means[i] + yerr_hi[i] if np.isfinite(means[i]) else 0.0,
            )

        p_pre_post = p_value_welch_ttest(
            np.asarray(values["pre"][group], dtype=float),
            np.asarray(values["post"][group], dtype=float),
        )
        if notes_layout == "sig_color_pvals":
            draw_pre_post_bracket(
                ax,
                float(x[gi] + offsets["pre"]),
                float(x[gi] + offsets["post"]),
                bar_tops[group],
                p_pre_post,
                label=format_p_value_label(p_pre_post)
                if is_significant(p_pre_post)
                else "NS",
                fontsize=note_fontsize,
            )
        else:
            draw_pre_post_bracket(
                ax,
                float(x[gi] + offsets["pre"]),
                float(x[gi] + offsets["post"]),
                bar_tops[group],
                p_pre_post,
            )

    phase_comp_sigs: dict[str, list[str]] = {phase: [] for phase in PHASE_ORDER}
    phase_comp_pvals: dict[str, list[float]] = {phase: [] for phase in PHASE_ORDER}
    pair_labels: list[str] = []
    for left, right in pairwise:
        pair_labels.append(comparison_pair_label(left, right))
        for phase in PHASE_ORDER:
            p_val = p_value_welch_ttest(
                np.asarray(values[phase][left], dtype=float),
                np.asarray(values[phase][right], dtype=float),
            )
            phase_comp_sigs[phase].append(significance_label(p_val))
            phase_comp_pvals[phase].append(p_val)

    ax.set_xticks(x)
    ax.set_xticklabels([display_label(g) for g in group_order], fontsize=10)
    if title:
        ax.set_title(title, fontsize=13, pad=8)
    else:
        ax.set_title("")
    set_axis_labels(ax, None, None, bold_xticks=True)
    style_axes(ax)
    ax.tick_params(axis="y", labelsize=14.5)
    ax.tick_params(axis="x", labelsize=10)
    for label in ax.get_xticklabels():
        label.set_fontsize(10)
        label.set_fontweight("bold")
    ax.set_ylim(0, PLOT_YMAX)
    ax.set_yticks(list(range(0, SCORE_YMAX + 1, 2)))

    if notes_layout == "pairwise_brackets":
        draw_pairwise_group_brackets(
            ax,
            x,
            group_order,
            pairwise,
            PHASE_ORDER,
            offsets,
            phase_comp_sigs,
            phase_bar_tops,
            fontsize=note_fontsize,
        )
    elif notes_layout == "sig_legend":
        draw_pairwise_sig_legend(
            ax,
            [
                (PHASE_LABELS["pre"], list(zip(phase_comp_sigs["pre"], pair_labels))),
                (PHASE_LABELS["post"], list(zip(phase_comp_sigs["post"], pair_labels))),
            ],
            loc="upper right",
            fontsize=note_fontsize,
        )
    elif notes_layout in ("sig_color_legend", "sig_color_pvals"):
        use_p = notes_layout == "sig_color_pvals"
        draw_pairwise_sig_color_legend(
            ax,
            [
                (
                    PHASE_LABELS["pre"],
                    list(
                        zip(
                            phase_comp_pvals["pre"] if use_p else phase_comp_sigs["pre"],
                            pairwise,
                        )
                    ),
                ),
                (
                    PHASE_LABELS["post"],
                    list(
                        zip(
                            phase_comp_pvals["post"] if use_p else phase_comp_sigs["post"],
                            pairwise,
                        )
                    ),
                ),
            ],
            group_colors=group_colors,
            loc="upper right",
            fontsize=note_fontsize,
            label_pvalues=use_p,
        )
    elif notes_layout == "pre_post_columns":
        draw_pre_post_sig_columns(
            ax,
            [
                (PHASE_LABELS["pre"], list(zip(phase_comp_sigs["pre"], pair_labels))),
                (PHASE_LABELS["post"], list(zip(phase_comp_sigs["post"], pair_labels))),
            ],
            y0=-0.14,
            fontsize=note_fontsize,
        )
    else:
        phase_notes = []
        for phase in PHASE_ORDER:
            comps = [
                f"{lab}: {sig}"
                for lab, sig in zip(pair_labels, phase_comp_sigs[phase])
            ]
            phase_notes.append(f"{PHASE_LABELS[phase]}: " + "; ".join(comps))
        ax.text(
            0.5,
            -0.22,
            "\n".join(phase_notes),
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=note_fontsize,
            color="#333333",
            clip_on=False,
        )


def _pooled_legend_label(group_key: str) -> str:
    """Legend label without sample size (human ratings are a subset)."""
    label = display_label(group_key)
    if group_key == "Human":
        return f"{label} ({HUMAN_COMPOSITION_NOTE})"
    return label


def _style_legend_frame(legend) -> None:
    frame = legend.get_frame()
    frame.set_visible(True)
    frame.set_linewidth(0.8)
    frame.set_edgecolor(PHASE_HATCH_COLOR)
    frame.set_facecolor("white")
    frame.set_alpha(1.0)


def plot_human_ratings(
    df: pd.DataFrame,
    out_path: Path,
    *,
    group_order: list[str] | tuple[str, ...],
    group_colors: dict[str, str],
    pairwise: tuple[tuple[str, str], ...],
    figsize: tuple[float, float],
    title: str,
    note_fontsize: float = 10,
    layout_bottom: float = 0.06,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=figsize, sharey=True)
    axes_flat = axes.ravel()

    for ax, (task, effect, panel_title) in zip(axes_flat, PANEL_SPECS):
        values = panel_values(df, task, effect)
        _draw_panel(
            ax,
            values,
            panel_title,
            group_order=group_order,
            group_colors=group_colors,
            pairwise=pairwise,
            note_fontsize=note_fontsize,
        )

    axes[0, 0].set_ylabel("Overall Quality Score (Mean ± 95% CI)", fontsize=12)
    axes[1, 0].set_ylabel("Overall Quality Score (Mean ± 95% CI)", fontsize=12)

    group_handles = [
        plt.Rectangle((0, 0), 1, 1, color=group_colors[g], alpha=BAR_ALPHA)
        for g in group_order
    ]
    group_labels = [_pooled_legend_label(g) for g in group_order]
    phase_handles = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor=PHASE_HATCH_COLOR,
            alpha=BAR_ALPHA,
            edgecolor=BAR_EDGE_COLOR,
        ),
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor=PHASE_HATCH_COLOR,
            alpha=BAR_ALPHA,
            edgecolor=BAR_EDGE_COLOR,
            hatch="///",
        ),
    ]
    legend1 = fig.legend(
        group_handles,
        group_labels,
        loc="upper left",
        bbox_to_anchor=(0.06, 0.97),
        frameon=False,
        fontsize=11,
    )
    fig.add_artist(legend1)
    fig.legend(
        phase_handles,
        ["Pre-ML", "Post-ML"],
        loc="upper right",
        bbox_to_anchor=(0.98, 0.97),
        frameon=False,
        fontsize=11,
    )

    fig.suptitle(title, fontsize=16, y=0.995)
    fig.tight_layout(rect=(0.02, layout_bottom, 1.0, 0.90))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="svg", bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"Saved figure: {out_path}")


def figure_title(cohort_key: str, *, humans_note: bool = False) -> str:
    base = f"Theory quality evaluated by human evaluators ({cohort_key}, subset)"
    if humans_note:
        return f"{base}; Humans: {HUMAN_COMPOSITION_NOTE}"
    return base


def plot_pooled_human_ratings(
    df: pd.DataFrame,
    out_path: Path,
    *,
    group_order: list[str] | tuple[str, ...],
    group_colors: dict[str, str],
    pairwise: tuple[tuple[str, str], ...],
    figsize: tuple[float, float],
    title: str,
    note_fontsize: float = 10,
) -> None:
    values = pooled_values(df)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    _draw_panel(
        ax,
        values,
        "",
        group_order=group_order,
        group_colors=group_colors,
        pairwise=pairwise,
        note_fontsize=note_fontsize,
        notes_layout="pre_post_columns",
    )

    group_legend_labels = [_pooled_legend_label(g) for g in group_order]

    ax.set_ylabel("Overall Quality Score (Mean ± 95% CI)", fontsize=15)
    group_handles = [
        plt.Rectangle((0, 0), 1, 1, color=group_colors[g], alpha=BAR_ALPHA)
        for g in group_order
    ]
    phase_handles = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor=PHASE_HATCH_COLOR,
            alpha=BAR_ALPHA,
            edgecolor=BAR_EDGE_COLOR,
        ),
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor=PHASE_HATCH_COLOR,
            alpha=BAR_ALPHA,
            edgecolor=BAR_EDGE_COLOR,
            hatch="///",
        ),
    ]
    all_handles = group_handles + phase_handles
    all_labels = group_legend_labels + ["Pre-ML", "Post-ML"]
    with plt.rc_context({"legend.frameon": True}):
        legend = fig.legend(
            all_handles,
            all_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.90),
            frameon=True,
            fancybox=False,
            edgecolor=PHASE_HATCH_COLOR,
            facecolor="white",
            framealpha=1.0,
            fontsize=10.5,
            ncol=len(group_order),
            handletextpad=0.35,
            columnspacing=1.0,
            borderpad=0.5,
            labelspacing=0.4,
        )
    _style_legend_frame(legend)

    fig.suptitle(title, fontsize=19, fontweight="bold", y=0.97)
    fig.subplots_adjust(left=0.14, right=0.90, top=0.76, bottom=0.28)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=SAVE_DPI, pad_inches=SAVE_PAD_INCHES)
    plt.close(fig)
    print(f"Saved figure: {out_path}")


def _print_values_summary(
    values: dict[str, dict[str, list[float]]],
    *,
    group_order: list[str] | tuple[str, ...],
    header: str,
) -> None:
    print(f"\n=== {header} ===")
    for phase in PHASE_ORDER:
        for g in group_order:
            s = summarize(values[phase][g])
            print(
                f"  {PHASE_LABELS[phase]:<7} {display_label(g):<18} "
                f"n={s['n']:>2}  mean={s['mean']:.3f}  "
                f"[{s['ci_low']:.3f}, {s['ci_high']:.3f}]"
            )


def _print_panel_summaries(
    df: pd.DataFrame,
    *,
    group_order: list[str] | tuple[str, ...],
    header: str,
) -> None:
    print(f"\n=== {header} ===")
    for task, effect, title in PANEL_SPECS:
        values = panel_values(df, task, effect)
        print(f"\n{title}")
        for phase in PHASE_ORDER:
            for g in group_order:
                s = summarize(values[phase][g])
                print(
                    f"  {PHASE_LABELS[phase]:<7} {display_label(g):<18} "
                    f"n={s['n']:>2}  mean={s['mean']:.3f}  "
                    f"[{s['ci_low']:.3f}, {s['ci_high']:.3f}]"
                )


def run_cohort(cohort: CohortSpec) -> None:
    print(f"\n######## cohort = {cohort.key} ########")
    frames = load_seed_frames(cohort)
    df = attach_topic_expert(combine_seed_ratings(frames))
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cohort.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(cohort.combined_csv, index=False)
    print(f"Wrote combined theories: {cohort.combined_csv}")
    print(
        f"Overall quality: mean={df[OVERALL_COL].mean():.3f}, "
        f"sd={df[OVERALL_COL].std():.3f}, n={len(df)}"
    )
    for fr in frames:
        seed = str(fr["source_seed"].iloc[0])
        one = compute_overall_quality(fr)
        print(
            f"  seed {seed} alone: n={len(one)}, "
            f"mean overall={one[OVERALL_COL].mean():.3f}"
        )

    _print_panel_summaries(
        df,
        group_order=GROUP_ORDER,
        header=f"{cohort.key}: PhD / Senior / Topic Experts / GenAI",
    )
    plot_human_ratings(
        df,
        cohort.out_dir / "by_task.svg",
        group_order=GROUP_ORDER,
        group_colors=GROUP_COLORS_TEXT,
        pairwise=PAIRWISE_THREE,
        figsize=(14.5, 11.5),
        title=f"Human evaluation of theory quality ({cohort.key})",
        note_fontsize=7.2,
        layout_bottom=0.08,
    )

    _print_panel_summaries(
        df,
        group_order=GROUP_ORDER_COLLAPSED,
        header=f"{cohort.key}: Humans vs GenAI",
    )
    plot_human_ratings(
        df,
        cohort.out_dir / "by_task_human_genai.svg",
        group_order=GROUP_ORDER_COLLAPSED,
        group_colors=GROUP_COLORS_COLLAPSED,
        pairwise=PAIRWISE_COLLAPSED,
        figsize=(11.5, 10.5),
        title=f"Human evaluation of theory quality ({cohort.key}; Humans vs GenAI)",
        note_fontsize=9,
        layout_bottom=0.06,
    )

    pooled = pooled_values(df)
    _print_values_summary(
        pooled,
        group_order=GROUP_ORDER,
        header=f"{cohort.key}: PhD / Senior / Topic Experts / GenAI (pooled)",
    )
    plot_pooled_human_ratings(
        df,
        cohort.out_dir / "pooled.svg",
        group_order=GROUP_ORDER,
        group_colors=GROUP_COLORS_TEXT,
        pairwise=PAIRWISE_THREE,
        figsize=(10.0, 9.5),
        title=figure_title(cohort.key),
        note_fontsize=8.5,
    )

    _print_values_summary(
        pooled,
        group_order=GROUP_ORDER_COLLAPSED,
        header=f"{cohort.key}: Humans vs GenAI (pooled)",
    )
    plot_pooled_human_ratings(
        df,
        cohort.out_dir / "pooled_human_genai.svg",
        group_order=GROUP_ORDER_COLLAPSED,
        group_colors=GROUP_COLORS_COLLAPSED,
        pairwise=PAIRWISE_COLLAPSED,
        figsize=(9.0, 8.8),
        title=figure_title(cohort.key, humans_note=True),
        note_fontsize=9,
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for cohort in COHORTS:
        run_cohort(cohort)


if __name__ == "__main__":
    main()
