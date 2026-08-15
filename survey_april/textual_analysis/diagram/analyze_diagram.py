"""
Analyze diagram structure metrics (Q5 / Q13).

Nature/Science-style LaTeX tables (booktabs) + CSV + bar figures:
1) Between-group comparisons within Pre-ML and within Post-ML
   (group means: Senior / PhD / Experts / Non-Experts / GenAI;
    Welch p for Humans = PhD+Senior vs GenAI)
2) Bar figure: Humans vs GenAI means by phase × task

Metrics: number of paths, maximum path length, number of latent variables.

Exclusion (listwise per timing × task): empty or < 0 → skip.
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
TEXTUAL_DIR = SCRIPT_DIR.parent
SURVEY_ROOT = SCRIPT_DIR.parents[1]
for p in (TEXTUAL_DIR, SURVEY_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from latex_table_pdf import compile_standalone_table  # noqa: E402
from stats_utils import (  # noqa: E402
    bootstrap_mean_ci,
    p_value_welch_ttest,
)
from viz_style import (  # noqa: E402
    BAR_ALPHA,
    BAR_EDGE_COLOR,
    BAR_EDGE_WIDTH,
    ERROR_CAPSIZE,
    ERROR_LINEWIDTH,
    GROUP_COLORS_COLLAPSED,
    GROUP_ORDER,
    GROUP_ORDER_COLLAPSED,
    apply_plot_style,
    draw_paired_pre_post_bracket,
    legend_entry,
    set_axis_labels,
    significance_label,
    style_axes,
    FONT_LEGEND,
    FONT_TICK,
)

CSV_PATH = SURVEY_ROOT / "All_Participants_All_Questions.csv"
OUT_DIR = SCRIPT_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STEM_BETWEEN = "theory_complexity_between_groups_three_group"
STEM_BETWEEN_HUMAN_PAIRS = "theory_complexity_between_groups_human_pairs"
STEM_BAR_HUMAN_GENAI = "theory_complexity_between_groups_human_genai"

GROUP_MAP = {
    "0": "PhD Students",
    "1": "Senior Scientists",
    "2": "GenAI",
}

# Group-mean columns in the between-group table (header short labels).
MEAN_GROUP_DEFS = (
    ("Humans", "Humans"),
    ("Senior Scientists", "Senior"),
    ("PhD Students", "PhD"),
    ("Topic Experts", "Experts"),
    ("Non-Topic Experts", "Non-Experts"),
    ("GenAI", "GenAI"),
)
MEAN_GROUP_ORDER = tuple(g for g, _ in MEAN_GROUP_DEFS)

TIMINGS = ("Pre-ML", "Post-ML")
TASKS = ("Race", "Gender")
TASK_DISPLAY = {
    "Race": "Racial inequality",
    "Gender": "Gender inequality",
}
TASK_MAKECELL = {
    "Race": r"\makecell[l]{Racial\\inequality}",
    "Gender": r"\makecell[l]{Gender\\inequality}",
}

METRIC_ORDER = [
    "Number of paths",
    "Maximum path length",
    "Number of latent variables",
]
METRIC_TEX = {
    "Number of paths": r"No.\ of paths",
    "Maximum path length": r"Maximum path length",
    "Number of latent variables": r"No.\ of latent variables",
}
METRIC_DISPLAY = {
    "Number of paths": "Number of\npaths",
    "Maximum path length": "Maximum\npath length",
    "Number of latent variables": "Number of\nlatent variables",
}

FIGSIZE_HUMAN_GENAI_PANEL = (12.5, 9.2)
DIAGRAM_Y_HEADROOM = 1.38

apply_plot_style()

_LEGACY_OUTPUTS = (
    OUT_DIR / "theory_complexity_between_groups_human_genai.csv",
    OUT_DIR / "theory_complexity_between_groups_human_genai.tex",
    OUT_DIR / "theory_complexity_between_groups_human_genai_standalone.tex",
    OUT_DIR / "theory_complexity_between_groups_human_genai_standalone.pdf",
    OUT_DIR / "theory_complexity_pre_post_within_group_human_genai.csv",
    OUT_DIR / "theory_complexity_pre_post_within_group_human_genai.tex",
    OUT_DIR / "theory_complexity_pre_post_within_group_human_genai_standalone.tex",
    OUT_DIR / "theory_complexity_pre_post_within_group_human_genai_standalone.pdf",
    OUT_DIR / "theory_complexity_pre_post_within_group_three_group.csv",
    OUT_DIR / "theory_complexity_pre_post_within_group_three_group.tex",
    OUT_DIR / "theory_complexity_pre_post_within_group_three_group_standalone.tex",
    OUT_DIR / "theory_complexity_pre_post_within_group_three_group_standalone.pdf",
    OUT_DIR / f"{STEM_BETWEEN}_race_pre.png",
    OUT_DIR / f"{STEM_BETWEEN}_race_post.png",
    OUT_DIR / f"{STEM_BETWEEN}_gender_pre.png",
    OUT_DIR / f"{STEM_BETWEEN}_gender_post.png",
)


@dataclass(frozen=True)
class MetricDef:
    timing: str
    task: str
    label: str
    prefix: str


METRICS: list[MetricDef] = [
    MetricDef("Pre-ML", "Race", "Number of paths", "Q Race.5 Number of paths"),
    MetricDef("Pre-ML", "Race", "Maximum path length", "Q Race.5 Maximum path length"),
    MetricDef("Pre-ML", "Race", "Number of latent variables", "Q Race.5 Number of latent variables"),
    MetricDef("Pre-ML", "Gender", "Number of paths", "Q Gender.5 Number of paths"),
    MetricDef("Pre-ML", "Gender", "Maximum path length", "Q Gender.5 Maximum path length"),
    MetricDef("Pre-ML", "Gender", "Number of latent variables", "Q Gender.5 Number of latent variables"),
    MetricDef("Post-ML", "Race", "Number of paths", "Q Race.13 Number of paths"),
    MetricDef("Post-ML", "Race", "Maximum path length", "Q Race.13 Maximum path length"),
    MetricDef("Post-ML", "Race", "Number of latent variables", "Q Race.13 Number of latent variables"),
    MetricDef("Post-ML", "Gender", "Number of paths", "Q Gender.13 Number of paths"),
    MetricDef("Post-ML", "Gender", "Maximum path length", "Q Gender.13 Maximum path length"),
    MetricDef("Post-ML", "Gender", "Number of latent variables", "Q Gender.13 Number of latent variables"),
]


def to_float(x: str) -> float | None:
    s = x.strip()
    if not s:
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    if v < 0:
        return None
    return v


def find_col_idx(headers: list[str], prefix: str) -> int | None:
    prefix_clean = prefix.strip().lower()
    exact_matches = [
        i for i, h in enumerate(headers)
        if h.strip().lower().startswith(prefix_clean)
    ]
    if len(exact_matches) == 1:
        return exact_matches[0]
    if len(exact_matches) == 0:
        loose = [i for i, h in enumerate(headers) if prefix_clean in h.strip().lower()]
        if len(loose) == 1:
            return loose[0]
        return None
    raise ValueError(
        f"Expected at most one column for prefix '{prefix}', got {exact_matches}"
    )


def _panel_metric_defs(timing: str, task: str) -> list[MetricDef]:
    return [m for m in METRICS if m.timing == timing and m.task == task]


def _row_panel_values(
    row: list[str],
    metric_defs: list[MetricDef],
    metric_cols: dict[MetricDef, int | None],
) -> dict[str, float] | None:
    values: dict[str, float] = {}
    for m in metric_defs:
        col = metric_cols[m]
        if col is None:
            return None
        v = to_float(row[col]) if len(row) > col else None
        if v is None:
            return None
        values[m.label] = v
    return values


def build_summary(
    data: list[list[str]],
    group_col: int,
    metric_cols: dict[MetricDef, int | None],
    topic_expert_col: int | None = None,
) -> dict[tuple[str, str, str, str], np.ndarray]:
    raw: dict[tuple[str, str, str, str], np.ndarray] = {}
    all_groups = list(GROUP_ORDER) + ["Topic Experts", "Non-Topic Experts"]
    for timing in TIMINGS:
        for task in TASKS:
            metrics = _panel_metric_defs(timing, task)
            grouped: dict[str, dict[str, list[float]]] = {
                g: {label: [] for label in METRIC_ORDER} for g in all_groups
            }
            for r in data:
                gid = r[group_col].strip() if len(r) > group_col else ""
                gname = GROUP_MAP.get(gid)
                if gname is None:
                    continue
                vals = _row_panel_values(r, metrics, metric_cols)
                if vals is None:
                    continue
                for label, v in vals.items():
                    grouped[gname][label].append(v)
                if (
                    topic_expert_col is not None
                    and gname in ("PhD Students", "Senior Scientists")
                    and len(r) > topic_expert_col
                ):
                    is_topic = r[topic_expert_col].strip() == "1"
                    expert_key = (
                        "Topic Experts" if is_topic else "Non-Topic Experts"
                    )
                    for label, v in vals.items():
                        grouped[expert_key][label].append(v)
            for m in metrics:
                for g in all_groups:
                    raw[(timing, task, m.label, g)] = np.asarray(
                        grouped[g][m.label], dtype=float
                    )
    return raw


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved CSV: {path}")


def _stars_tex(p: float) -> str:
    if not np.isfinite(p) or p >= 0.05:
        return ""
    if p < 0.001:
        return r"^{***}"
    if p < 0.01:
        return r"^{**}"
    return r"^{*}"


def _fmt_mean_tex(arr: np.ndarray) -> str:
    if arr.size == 0 or not np.isfinite(arr.mean()):
        return "---"
    return f"{float(arr.mean()):.2f}"


def _fmt_mean_with_pre_post_arrow_tex(
    post_arr: np.ndarray,
    pre_arr: np.ndarray,
) -> str:
    """Post-ML mean with ↑/↓ relative to the matching Pre-ML group mean."""
    base = _fmt_mean_tex(post_arr)
    if base == "---":
        return base
    if pre_arr.size == 0 or not np.isfinite(pre_arr.mean()):
        return base
    post_m = float(post_arr.mean())
    pre_m = float(pre_arr.mean())
    if post_m > pre_m:
        return rf"{base}{{\(\uparrow\)}}"
    if post_m < pre_m:
        return rf"{base}{{\(\downarrow\)}}"
    return base


def _fmt_p_tex(p: float) -> str:
    r"""Significant: p with stars; otherwise NS only. Math mode: \(...\)."""
    if not np.isfinite(p):
        return "---"
    if p >= 0.05:
        return "NS"
    if p < 0.001:
        return rf"\(<0.001{_stars_tex(p)}\)"
    return rf"\({p:.3f}{_stars_tex(p)}\)"


def _human_genai_arrays(
    raw: dict[tuple[str, str, str, str], np.ndarray],
    timing: str,
    task: str,
    metric: str,
) -> tuple[np.ndarray, np.ndarray]:
    phd = raw.get(
        (timing, task, metric, "PhD Students"), np.asarray([], dtype=float)
    )
    exp = raw.get((timing, task, metric, "Senior Scientists"), np.asarray([], dtype=float))
    gen = raw.get((timing, task, metric, "GenAI"), np.asarray([], dtype=float))
    parts = [a for a in (phd, exp) if a.size]
    human = np.concatenate(parts) if parts else np.asarray([], dtype=float)
    return human, gen


def plot_between_human_genai_panel(
    raw: dict[tuple[str, str, str, str], np.ndarray],
    out_path: Path,
) -> None:
    """2×2 panels (Pre/Post × Race/Gender): Humans vs GenAI, sig on bars."""
    x = np.arange(len(METRIC_ORDER))
    width = 0.34
    offsets = {"Human": -width / 2, "GenAI": width / 2}
    panels = (
        (0, 0, "Pre-ML", "Race"),
        (0, 1, "Pre-ML", "Gender"),
        (1, 0, "Post-ML", "Race"),
        (1, 1, "Post-ML", "Gender"),
    )

    fig, axes = plt.subplots(
        2, 2, figsize=FIGSIZE_HUMAN_GENAI_PANEL, sharey=True
    )
    bar_meta: list[tuple[object, list[float], list[float]]] = []

    for row, col, timing, task in panels:
        ax = axes[row, col]
        tops: list[float] = []
        pvals: list[float] = []
        for i, metric in enumerate(METRIC_ORDER):
            human, gen = _human_genai_arrays(raw, timing, task, metric)
            arrays = {"Human": human, "GenAI": gen}
            tops_i: list[float] = []
            for group in GROUP_ORDER_COLLAPSED:
                arr = arrays[group]
                if arr.size == 0:
                    mean = np.nan
                    yerr_lo = yerr_hi = 0.0
                else:
                    mean = float(arr.mean())
                    lo, hi = bootstrap_mean_ci(arr)
                    yerr_lo = max(0.0, mean - lo) if np.isfinite(lo) else 0.0
                    yerr_hi = max(0.0, hi - mean) if np.isfinite(hi) else 0.0
                xpos = float(x[i] + offsets[group])
                ax.bar(
                    [xpos],
                    [mean if np.isfinite(mean) else 0.0],
                    width=width,
                    color=GROUP_COLORS_COLLAPSED[group],
                    alpha=BAR_ALPHA,
                    edgecolor=BAR_EDGE_COLOR,
                    linewidth=BAR_EDGE_WIDTH,
                    zorder=2,
                )
                ax.errorbar(
                    [xpos],
                    [mean if np.isfinite(mean) else 0.0],
                    yerr=[[yerr_lo], [yerr_hi]],
                    fmt="none",
                    ecolor="black",
                    elinewidth=ERROR_LINEWIDTH,
                    capsize=ERROR_CAPSIZE,
                    zorder=3,
                )
                tops_i.append(
                    (mean if np.isfinite(mean) else 0.0) + yerr_hi
                )
            tops.append(max(tops_i) if tops_i else 0.0)
            pvals.append(p_value_welch_ttest(human, gen))

        ax.set_xticks(x)
        ax.set_xticklabels([METRIC_DISPLAY[m] for m in METRIC_ORDER])
        ax.set_title(
            f"{timing} · {TASK_DISPLAY[task]}",
            fontsize=14,
            fontweight="bold",
            pad=10,
        )
        if col == 0:
            set_axis_labels(ax, None, "Mean ± 95% CI", bold_xticks=False)
        else:
            set_axis_labels(ax, None, None, bold_xticks=False)
        style_axes(ax)
        ax.tick_params(axis="x", labelsize=FONT_TICK + 2)
        bar_meta.append((ax, tops, pvals))

    ymax = 0.0
    for _, tops, _ in bar_meta:
        if tops:
            ymax = max(ymax, max(tops))
    ylim_top = ymax * DIAGRAM_Y_HEADROOM if ymax > 0 else 1.0
    for ax, tops, pvals in bar_meta:
        ax.set_ylim(0, ylim_top)
        for i, (y_base, p) in enumerate(zip(tops, pvals)):
            draw_paired_pre_post_bracket(
                ax,
                float(x[i] + offsets["Human"]),
                float(x[i] + offsets["GenAI"]),
                y_base,
                p,
                label=significance_label(p),
            )

    handles = [
        plt.Rectangle(
            (0, 0), 1, 1, color=GROUP_COLORS_COLLAPSED[g], alpha=BAR_ALPHA
        )
        for g in GROUP_ORDER_COLLAPSED
    ]
    # Legend uses full-sample N (Humans = PhD + Senior Scientists in CSV), not listwise n.
    labels = [
        legend_entry("Human", 73),
        legend_entry("GenAI", 22),
    ]
    fig.subplots_adjust(
        left=0.08, right=0.98, top=0.80, bottom=0.08, wspace=0.18, hspace=0.40
    )
    fig.suptitle(
        "Between-group comparisons of theory complexity",
        fontsize=22,
        fontweight="bold",
        y=0.96,
    )
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        fontsize=FONT_LEGEND + 2,
        bbox_to_anchor=(0.5, 0.915),
    )
    fig.savefig(out_path, dpi=600, bbox_inches="tight", pad_inches=0.30)
    plt.close(fig)
    print(f"Saved figure: {out_path}")


def _group_array(
    raw: dict[tuple[str, str, str, str], np.ndarray],
    timing: str,
    task: str,
    metric: str,
    group: str,
) -> np.ndarray:
    return raw.get((timing, task, metric, group), np.asarray([], dtype=float))


def build_between_rows(
    diagram_raw: dict[tuple[str, str, str, str], np.ndarray],
) -> list[dict[str, object]]:
    """Between-group means (Humans + subgroups + GenAI) and Humans vs GenAI p."""
    rows: list[dict[str, object]] = []
    for timing in TIMINGS:
        for task in TASKS:
            for metric in METRIC_ORDER:
                human, gen = _human_genai_arrays(diagram_raw, timing, task, metric)
                arrays = {
                    g: _group_array(diagram_raw, timing, task, metric, g)
                    for g in MEAN_GROUP_ORDER
                    if g not in ("Humans", "GenAI")
                }
                arrays["Humans"] = human
                arrays["GenAI"] = gen
                p = p_value_welch_ttest(human, gen)
                row: dict[str, object] = {
                    "timing": timing,
                    "task": task,
                    "metric": metric,
                    "p_human_vs_genai": p,
                    "sig_human_vs_genai": significance_label(p),
                    "_human": human,
                    "_gen": gen,
                }
                for g in MEAN_GROUP_ORDER:
                    arr = arrays[g]
                    row[f"mean_{g}"] = float(arr.mean()) if arr.size else np.nan
                    row[f"_{g}"] = arr
                rows.append(row)
    return rows


def build_between_tex(rows: list[dict[str, object]]) -> str:
    body_lines: list[str] = []
    n_per_phase = len(TASKS) * len(METRIC_ORDER)
    n_metrics = len(METRIC_ORDER)
    n_mean = len(MEAN_GROUP_ORDER)
    pre_by = {
        (str(r["task"]), str(r["metric"])): r
        for r in rows
        if r["timing"] == "Pre-ML"
    }
    for ti, timing in enumerate(TIMINGS):
        timing_rows = [r for r in rows if r["timing"] == timing]
        prev_task: str | None = None
        for j, r in enumerate(timing_rows):
            task = str(r["task"])
            metric = str(r["metric"])
            phase_cell = (
                rf"\multirow{{{n_per_phase}}}{{*}}{{{timing}}}" if j == 0 else ""
            )
            if task != prev_task:
                if prev_task is not None:
                    body_lines.append(r"\addlinespace[0.25em]")
                task_cell = (
                    rf"\multirow{{{n_metrics}}}{{*}}{{{TASK_MAKECELL[task]}}}"
                )
                prev_task = task
            else:
                task_cell = ""
            mean_cells: list[str] = []
            if timing == "Post-ML":
                pre = pre_by[(task, metric)]
                for g in MEAN_GROUP_ORDER:
                    post_arr = r[f"_{g}"]
                    pre_arr = pre[f"_{g}"]
                    assert isinstance(post_arr, np.ndarray)
                    assert isinstance(pre_arr, np.ndarray)
                    mean_cells.append(
                        _fmt_mean_with_pre_post_arrow_tex(post_arr, pre_arr)
                    )
            else:
                for g in MEAN_GROUP_ORDER:
                    arr = r[f"_{g}"]
                    assert isinstance(arr, np.ndarray)
                    mean_cells.append(_fmt_mean_tex(arr))
            body_lines.append(
                " & ".join([
                    phase_cell,
                    task_cell,
                    METRIC_TEX[metric],
                    *mean_cells,
                    _fmt_p_tex(float(r["p_human_vs_genai"])),
                ])
                + r" \\"
            )
            body_lines.append("")
        if ti < len(TIMINGS) - 1:
            body_lines.append(r"\midrule")
            body_lines.append("")

    mean_col = (
        r"  S[table-format=1.2,"
        r"table-space-text-post={\(\uparrow\)}]"
    )
    mean_header = " & ".join(
        rf"{{{short}}}" for _, short in MEAN_GROUP_DEFS
    )
    cmid_end = 3 + n_mean  # Phase, Task, Metric, then means
    return "\n".join([
        "% Auto-generated by analyze_diagram.py",
        "",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.18}",
        "",
        r"\sisetup{",
        r"  table-number-alignment = center,",
        r"  table-format = 1.2,",
        r"  detect-weight = true,",
        r"  detect-family = true",
        r"}",
        "",
        r"\begin{tabular}{",
        r"  @{}",
        r"  ll",
        r"  l",
        *([mean_col] * n_mean),
        r"  c",
        r"  @{}",
        r"}",
        r"\toprule",
        r"\multirow{2}{*}{\textbf{Phase}}",
        r"& \multirow{2}{*}{\textbf{Task}}",
        r"& \multirow{2}{*}{\textbf{Metric}}",
        rf"& \multicolumn{{{n_mean}}}{{c}}{{\textbf{{Group means}}}}",
        r"& {\textbf{Pairwise comparison ($p$\ value)}} \\",
        rf"\cmidrule(lr){{4-{cmid_end}}}",
        r"&",
        r"&",
        f"& {mean_header}",
        r"& {\makecell{Humans\\vs GenAI}} \\",
        r"\midrule",
        "",
        *body_lines,
        r"\bottomrule",
        r"\end{tabular}",
        "",
    ])


def _csv_public_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for r in rows:
        out.append({k: v for k, v in r.items() if not str(k).startswith("_")})
    return out


def build_human_pair_p_rows(
    diagram_raw: dict[tuple[str, str, str, str], np.ndarray],
) -> list[dict[str, object]]:
    """Welch p-values: Senior vs PhD; Topic Experts vs Non-Topic Experts."""
    rows: list[dict[str, object]] = []
    for timing in TIMINGS:
        for task in TASKS:
            for metric in METRIC_ORDER:
                senior = _group_array(
                    diagram_raw, timing, task, metric, "Senior Scientists"
                )
                phd = _group_array(
                    diagram_raw, timing, task, metric, "PhD Students"
                )
                experts = _group_array(
                    diagram_raw, timing, task, metric, "Topic Experts"
                )
                non_experts = _group_array(
                    diagram_raw, timing, task, metric, "Non-Topic Experts"
                )
                p_sen_phd = p_value_welch_ttest(senior, phd)
                p_exp = p_value_welch_ttest(experts, non_experts)
                rows.append({
                    "timing": timing,
                    "task": task,
                    "metric": metric,
                    "p_senior_vs_phd": p_sen_phd,
                    "sig_senior_vs_phd": significance_label(p_sen_phd),
                    "p_experts_vs_non_experts": p_exp,
                    "sig_experts_vs_non_experts": significance_label(p_exp),
                    "n_senior": int(senior.size),
                    "n_phd": int(phd.size),
                    "n_experts": int(experts.size),
                    "n_non_experts": int(non_experts.size),
                })
    return rows


def build_human_pair_p_tex(rows: list[dict[str, object]]) -> str:
    """Standalone p-value table (no caption / footnotes)."""
    body_lines: list[str] = []
    n_per_phase = len(TASKS) * len(METRIC_ORDER)
    n_metrics = len(METRIC_ORDER)
    for ti, timing in enumerate(TIMINGS):
        timing_rows = [r for r in rows if r["timing"] == timing]
        prev_task: str | None = None
        for j, r in enumerate(timing_rows):
            task = str(r["task"])
            metric = str(r["metric"])
            phase_cell = (
                rf"\multirow{{{n_per_phase}}}{{*}}{{{timing}}}" if j == 0 else ""
            )
            if task != prev_task:
                if prev_task is not None:
                    body_lines.append(r"\addlinespace[0.25em]")
                task_cell = (
                    rf"\multirow{{{n_metrics}}}{{*}}{{{TASK_MAKECELL[task]}}}"
                )
                prev_task = task
            else:
                task_cell = ""
            body_lines.append(
                " & ".join([
                    phase_cell,
                    task_cell,
                    METRIC_TEX[metric],
                    _fmt_p_tex(float(r["p_senior_vs_phd"])),
                    _fmt_p_tex(float(r["p_experts_vs_non_experts"])),
                ])
                + r" \\"
            )
            body_lines.append("")
        if ti < len(TIMINGS) - 1:
            body_lines.append(r"\midrule")
            body_lines.append("")

    return "\n".join([
        "% Auto-generated by analyze_diagram.py",
        "",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{6pt}",
        r"\renewcommand{\arraystretch}{1.18}",
        "",
        r"\begin{tabular}{@{}lllcc@{}}",
        r"\toprule",
        r"\textbf{Phase} & \textbf{Task} & \textbf{Metric}",
        r"& \textbf{Senior vs.\ PhD}",
        r"& \textbf{Experts vs.\ Non-Experts} \\",
        r"\midrule",
        "",
        *body_lines,
        r"\bottomrule",
        r"\end{tabular}",
        "",
    ])


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV file not found: {CSV_PATH}")

    with CSV_PATH.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        raise ValueError(f"CSV file is empty: {CSV_PATH}")

    headers, data = rows[0], rows[1:]
    group_col = find_col_idx(headers, "student_0, senior_1, genAI_2")
    if group_col is None:
        raise ValueError("Group column not found.")
    topic_expert_col = find_col_idx(headers, "topic_expert")

    metric_cols: dict[MetricDef, int | None] = {
        m: find_col_idx(headers, m.prefix) for m in METRICS
    }
    missing_cols = [m.prefix for m, idx in metric_cols.items() if idx is None]
    if missing_cols:
        print("Warning: missing metric columns (treated as empty):")
        for c in missing_cols:
            print(f"  - {c}")

    raw = build_summary(data, group_col, metric_cols, topic_expert_col)

    between = build_between_rows(raw)
    _write_csv(OUT_DIR / f"{STEM_BETWEEN}.csv", _csv_public_rows(between))
    pdf_b = compile_standalone_table(
        OUT_DIR,
        STEM_BETWEEN,
        build_between_tex(between),
        output_format="pdf+svg",
        crop="standalone",
        extra_packages=[
            r"\usepackage{makecell}",
            r"\usepackage{siunitx}",
        ],
    )
    print(f"Saved LaTeX/PDF/SVG: {pdf_b}")
    print(f"  SVG: {OUT_DIR / f'{STEM_BETWEEN}_standalone.svg'}")

    human_pairs = build_human_pair_p_rows(raw)
    _write_csv(
        OUT_DIR / f"{STEM_BETWEEN_HUMAN_PAIRS}.csv",
        _csv_public_rows(human_pairs),
    )
    pdf_pairs = compile_standalone_table(
        OUT_DIR,
        STEM_BETWEEN_HUMAN_PAIRS,
        build_human_pair_p_tex(human_pairs),
        output_format="pdf+svg",
        crop="standalone",
        extra_packages=[
            r"\usepackage{makecell}",
        ],
    )
    print(f"Saved LaTeX/PDF/SVG: {pdf_pairs}")
    print(f"  SVG: {OUT_DIR / f'{STEM_BETWEEN_HUMAN_PAIRS}_standalone.svg'}")

    for legacy in _LEGACY_OUTPUTS:
        if legacy.exists():
            legacy.unlink()
            print(f"Removed: {legacy.name}")
    # Do not keep the Humans vs GenAI bar panel (removed by request).
    bar_path = OUT_DIR / f"{STEM_BAR_HUMAN_GENAI}.png"
    if bar_path.exists():
        bar_path.unlink()

    print("\nTables:")
    print(
        f"  1) {STEM_BETWEEN}.tex  →  {pdf_b.name}  "
        "(group means + Humans vs GenAI $p$)"
    )
    print(
        f"  2) {STEM_BETWEEN_HUMAN_PAIRS}.tex  →  {pdf_pairs.name}  "
        "(Senior vs PhD; Experts vs Non-Experts $p$)"
    )


if __name__ == "__main__":
    main()
