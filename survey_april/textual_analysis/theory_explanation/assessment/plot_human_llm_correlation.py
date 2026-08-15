#!/usr/bin/env python3
"""Human vs LLM overall-quality correlation utilities.

Scatter panel is drawn in plot_quality_gap_combined.py (Panel A).
This module also writes the correlation table CSV when run as __main__.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredText
from scipy import stats

ASSESSMENT_DIR = Path(__file__).resolve().parent
TEXTUAL_DIR = ASSESSMENT_DIR.parent.parent  # textual_analysis/
ROOT = TEXTUAL_DIR.parent  # survey_april/
for p in (ASSESSMENT_DIR, TEXTUAL_DIR, ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from viz_style import (  # noqa: E402
    BAR_ALPHA,
    GROUP_COLORS_COLLAPSED,
    GROUP_COLORS_TEXT,
    GROUP_ORDER_COLLAPSED,
    PHASE_HATCH_COLOR,
    apply_plot_style,
    display_label,
    format_p_value_label,
    set_axis_labels,
    style_axes,
)

apply_plot_style()

OUT_DIR = ASSESSMENT_DIR / "outputs_human_llm_corr"
HE_DIR = ASSESSMENT_DIR / "human_evaluation" / "human_rating_data"
NAME_COL = "What is your full name?"

DIMS = [
    "clarity_coherence",
    "causal_reasoning",
    "theoretical_depth",
    "creativity",
    "persuasiveness",
]

# CSV column tag → axis / filename display
LLM_MODELS: dict[str, dict[str, str]] = {
    "gpt-5.5": {
        "display": "GPT-5.5",
        "csv_stem": "human_llm_correlation_by_cohort",
    },
}


def _llm_map(model_tag: str) -> dict[tuple[str, str, str], str]:
    return {
        ("race", "main", "pre"): f"Q Race.4 Overall Quality Score ({model_tag})",
        ("race", "main", "post"): f"Q Race.12 Overall Quality Score ({model_tag})",
        ("race", "interactions", "pre"): f"Q Race.10 Overall Quality Score ({model_tag})",
        ("race", "interactions", "post"): f"Q Race.15 Overall Quality Score ({model_tag})",
        ("gender", "main", "pre"): f"Q Gender.4 Overall Quality Score ({model_tag})",
        ("gender", "main", "post"): f"Q Gender.12 Overall Quality Score ({model_tag})",
        ("gender", "interactions", "pre"): f"Q Gender.10 Overall Quality Score ({model_tag})",
        ("gender", "interactions", "post"): f"Q Gender.15 Overall Quality Score ({model_tag})",
    }


def _fnum(x) -> float:
    try:
        v = float(str(x).strip())
        return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan


def _load_matched(model_tag: str) -> pd.DataFrame:
    survey = pd.read_csv(
        ROOT / "All_Participants_All_Questions.csv", dtype=str, keep_default_na=False
    )
    llm_map = _llm_map(model_tag)
    llm_rows = []
    for _, r in survey.iterrows():
        name = str(r[NAME_COL]).strip()
        for (task, effect, phase), col in llm_map.items():
            if col not in survey.columns:
                continue
            v = _fnum(r[col])
            if np.isnan(v):
                continue
            llm_rows.append(
                dict(
                    participant_name=name,
                    task=task,
                    effect=effect,
                    phase=phase,
                    llm=v,
                )
            )
    if not llm_rows:
        raise RuntimeError(f"No LLM overall scores found for model_tag={model_tag!r}")
    llm = pd.DataFrame(llm_rows)
    llm["name_key"] = llm["participant_name"].str.lower()

    parts = []
    for cohort, fn in [
        ("EU", "theory_ratings_EU_combined.csv"),
        ("PP", "theory_ratings_PP_combined.csv"),
    ]:
        d = pd.read_csv(HE_DIR / fn)
        d["cohort"] = cohort
        parts.append(d)
    hum = pd.concat(parts, ignore_index=True)
    for c in DIMS:
        hum[c] = pd.to_numeric(hum[c], errors="coerce")
    hum["human"] = hum[DIMS].mean(axis=1)
    hum["task"] = hum["task"].astype(str).str.lower().str.strip()
    hum["effect"] = hum["effect"].astype(str).str.lower().str.strip()
    hum["phase"] = hum["phase"].astype(str).str.lower().str.strip()
    hum["name_key"] = hum["participant_name"].astype(str).str.strip().str.lower()
    hum["group"] = pd.to_numeric(hum["group"], errors="coerce").map(
        {0: "PhD Students", 1: "Senior Scientists", 2: "GenAI"}
    )

    m = hum.merge(llm, on=["name_key", "task", "effect", "phase"], how="inner")
    keys = ["name_key", "task", "effect", "phase", "cohort", "group"]
    return m.groupby(keys, as_index=False).agg(
        human=("human", "mean"),
        llm=("llm", "mean"),
        n_ratings=("human", "size"),
    )


def _corr_stats(df: pd.DataFrame) -> dict:
    x = df["human"].to_numpy(float)
    y = df["llm"].to_numpy(float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 3:
        return dict(
            n=n,
            pearson=np.nan,
            pearson_p=np.nan,
            spearman=np.nan,
            spearman_p=np.nan,
            slope=np.nan,
            intercept=np.nan,
        )
    pr, pp = stats.pearsonr(x, y)
    sr, sp = stats.spearmanr(x, y)
    slope, intercept, *_ = stats.linregress(x, y)
    return dict(
        n=n,
        pearson=pr,
        pearson_p=pp,
        spearman=sr,
        spearman_p=sp,
        slope=slope,
        intercept=intercept,
    )


def _fmt_r(v: float) -> str:
    return f"{v:.2f}" if np.isfinite(v) else "NA"


def _fmt_corr(r: float, p: float) -> str:
    return f"{_fmt_r(r)} ({format_p_value_label(p)})"


LEGEND_EDGE = PHASE_HATCH_COLOR


def _style_legend_frame(legend) -> None:
    frame = legend.get_frame()
    frame.set_visible(True)
    frame.set_linewidth(0.7)
    frame.set_edgecolor(LEGEND_EDGE)
    frame.set_facecolor("white")
    frame.set_alpha(1.0)


def _merge_cohort_means(m: pd.DataFrame) -> pd.DataFrame:
    """One point per theory: average human (and llm) across EU/PP cohorts."""
    return m.groupby(
        ["name_key", "task", "effect", "phase", "group"],
        as_index=False,
    ).agg(
        human=("human", "mean"),
        llm=("llm", "mean"),
        n_cohorts=("cohort", "nunique"),
        n_ratings=("n_ratings", "sum"),
    )


SCATTER_COLORS = {
    **GROUP_COLORS_COLLAPSED,
    "GenAI": GROUP_COLORS_TEXT["GenAI"],
}


def _scatter_mask(df: pd.DataFrame, group: str) -> pd.Series:
    if group == "Human":
        return df["group"].isin(["PhD Students", "Senior Scientists"])
    return df["group"] == group


def _draw_scatter_panel(
    ax,
    df: pd.DataFrame,
    *,
    title: str,
    llm_display: str | None = None,
    show_ylabel: bool = True,
    marker_size: float = 32,
    corr_fontsize: float = 10,
    axis_label_fontsize: float = 12,
    square_aspect: bool = True,
) -> None:
    for g in GROUP_ORDER_COLLAPSED:
        sub = df.loc[_scatter_mask(df, g)]
        ax.scatter(
            sub["human"],
            sub["llm"],
            s=marker_size,
            alpha=0.85,
            color=SCATTER_COLORS[g],
            edgecolors="white",
            linewidths=0.35,
            zorder=3,
        )

    s = _corr_stats(df)
    xs = np.linspace(1.0, 10.0, 50)
    ax.plot([1, 10], [1, 10], ls="--", color="#BDBDBD", lw=1.0, zorder=1)
    if np.isfinite(s["slope"]):
        ax.plot(
            xs,
            s["intercept"] + s["slope"] * xs,
            color="#333333",
            lw=1.25,
            zorder=2,
        )

    ax.set_xlim(0.5, 10.5)
    ax.set_ylim(0.5, 10.5)
    ax.set_xticks(list(range(2, 11, 2)))
    ax.set_yticks(list(range(2, 11, 2)))
    if square_aspect:
        ax.set_box_aspect(1)
    if title:
        ax.set_title(title, fontsize=14, pad=8, fontweight="bold")

    pearson_line = f"{'Pearson':<9}  r = {_fmt_corr(s['pearson'], s['pearson_p'])}"
    spearman_line = f"{'Spearman':<9}  ρ = {_fmt_corr(s['spearman'], s['spearman_p'])}"
    corr_box = AnchoredText(
        f"{pearson_line}\n{spearman_line}",
        loc="lower right",
        prop={
            "size": corr_fontsize,
            "color": "#333333",
            "family": "Helvetica",
        },
        frameon=True,
        borderpad=1.15,
        pad=0.4,
    )
    corr_box.patch.set_boxstyle("square,pad=0.35")
    corr_box.patch.set_facecolor("white")
    corr_box.patch.set_edgecolor(PHASE_HATCH_COLOR)
    corr_box.patch.set_linewidth(0.8)
    ax.add_artist(corr_box)
    set_axis_labels(ax, "Score by human evaluators", None, xlabel_pad=8)
    if show_ylabel:
        ax.set_ylabel(
            f"Score by LLM ({llm_display})" if llm_display else "Score by LLM",
            fontsize=axis_label_fontsize,
        )
    style_axes(ax)
    ax.tick_params(axis="y", labelleft=True)


def _legend_handles(contributor_label=display_label) -> list:
    group_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=SCATTER_COLORS[g],
            markeredgecolor="white",
            markeredgewidth=0.35,
            markersize=8.5,
            alpha=BAR_ALPHA,
            label=contributor_label(g),
        )
        for g in GROUP_ORDER_COLLAPSED
    ]
    line_handles = [
        Line2D([0], [0], color="#333333", lw=1.25, label="OLS fit"),
        Line2D(
            [0],
            [0],
            color="#BDBDBD",
            lw=1.0,
            ls="--",
            label="Identity (y = x)",
        ),
    ]
    return group_handles + line_handles


def _subset_rows(df: pd.DataFrame):
    return [
        ("All", df),
        ("Humans only", df[df["group"].isin(["PhD Students", "Senior Scientists"])]),
        ("PhD Students", df[df["group"] == "PhD Students"]),
        ("Senior Scientists", df[df["group"] == "Senior Scientists"]),
        ("GenAI", df[df["group"] == "GenAI"]),
    ]


def _write_corr_table(m: pd.DataFrame, csv_path: Path, *, model_tag: str) -> None:
    rows = []
    m_avg = _merge_cohort_means(m)
    key = ["name_key", "task", "effect", "phase", "group"]

    def _n_ann_for_avg(sub: pd.DataFrame) -> int:
        keys = set(map(tuple, sub[key].to_numpy()))
        return int(m[key].apply(tuple, axis=1).isin(keys).sum())

    for cohort, df in [
        ("EU", m[m["cohort"] == "EU"]),
        ("PP", m[m["cohort"] == "PP"]),
        ("EU+PP averaged", m_avg),
    ]:
        for label, sub in _subset_rows(df):
            s = _corr_stats(sub)
            pr, pp = (np.nan, np.nan)
            sr, sp = (np.nan, np.nan)
            if s["n"] >= 3:
                x = sub["human"].to_numpy(float)
                y = sub["llm"].to_numpy(float)
                mask = np.isfinite(x) & np.isfinite(y)
                pr, pp = stats.pearsonr(x[mask], y[mask])
                sr, sp = stats.spearmanr(x[mask], y[mask])
            n_ann = _n_ann_for_avg(sub) if cohort == "EU+PP averaged" else s["n"]
            rows.append(
                dict(
                    model=model_tag,
                    cohort=cohort,
                    subset=label,
                    n=n_ann,
                    n_points=s["n"],
                    pearson_r=pr,
                    pearson_p=pp,
                    spearman_rho=sr,
                    spearman_p=sp,
                )
            )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Saved table: {csv_path}")


def run_for_model(model_tag: str) -> None:
    meta = LLM_MODELS[model_tag]
    m = _load_matched(model_tag)
    _write_corr_table(m, OUT_DIR / f"{meta['csv_stem']}.csv", model_tag=model_tag)

    m_avg = _merge_cohort_means(m)
    s_all = _corr_stats(m_avg)
    print(
        f"  [{model_tag}] EU+PP averaged: "
        f"r={s_all['pearson']:.3f}  ρ={s_all['spearman']:.3f}  "
        f"n_ann={len(m)}  n_points={s_all['n']}"
    )
    for cohort in ["EU", "PP"]:
        s = _corr_stats(m[m["cohort"] == cohort])
        print(
            f"  [{model_tag}] {cohort}: "
            f"r={s['pearson']:.3f}  ρ={s['spearman']:.3f}  n={s['n']}"
        )


def main() -> None:
    for model_tag in LLM_MODELS:
        print(f"\n=== {model_tag} ===")
        run_for_model(model_tag)


if __name__ == "__main__":
    main()
