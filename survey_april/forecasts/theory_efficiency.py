"""
Theory-efficiency analysis: does diagram complexity predict forecasting accuracy?

Primary design (Pre-ML Q5 complexity × forecasting accuracy):

  Accuracy = β0 + β1 C_z + β2 GenAI + β3 (C_z × GenAI) + ε

where C_z is complexity standardized to mean 0, SD 1 (same SD for all
models, from person×task rows). Coefficients are thus per 1 SD of C.

Stratified by Racial/Gender × Main-effects/Interactions (HC1), plus a
pooled specification stacking all task×effect cells with task×effect cell FE
(reference = Racial × Main effects) and person-clustered (CR1) SEs.

A significant positive β1 means complexity buys accuracy among humans.
A near-zero or negative β3 means GenAI's extra complexity does not buy
proportionally more accuracy (ornamental complexity).

Also fits a multivariate model with all three complexity metrics + interactions,
separately by task (HC1 SEs; Main-effects accuracy).

Outputs under forecasts/outputs/:
  theory_efficiency_coef_table_standalone.{pdf,svg}
  theory_efficiency_summary.csv
  theory_efficiency_multivariate.csv
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from scipy.stats import t as t_dist

FORECASTS = Path(__file__).resolve().parent
ROOT = FORECASTS.parent
for p in (ROOT, FORECASTS, ROOT / "moderator_analysis"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from aggregation_analysis import (  # noqa: E402
    GENAI_GROUP_IDS,
    HUMAN_GROUP_IDS,
    load_main_effects_records,
    load_soi_records,
)
from moderator_test import ols_hc1_inference  # noqa: E402

OUT_DIR = FORECASTS / "outputs"
OUT_DIR.mkdir(exist_ok=True)
CSV_PATH = ROOT / "All_Participants_All_Questions.csv"

COMPLEXITY_DEFS = (
    ("n_paths", "Number of paths", "Q {task}.5 Number of paths"),
    ("max_path_len", "Maximum path length", "Q {task}.5 Maximum path length"),
    ("n_latents", "Number of latent variables", "Q {task}.5 Number of latent variables"),
)
TASKS = (
    ("Race", "Racial inequality", "cos_race"),
    ("Gender", "Gender inequality", "cos_gender"),
)


def _to_float(x: str) -> float | None:
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


def _find_col(headers: list[str], name: str) -> int:
    key = name.strip().lower()
    for i, h in enumerate(headers):
        if h.strip().lower() == key:
            return i
    for i, h in enumerate(headers):
        if h.strip().lower().startswith(key):
            return i
    raise KeyError(name)


def build_person_rows() -> list[dict]:
    """One row per person × task with ME accuracy + Pre-ML complexity."""
    with CSV_PATH.open(encoding="utf-8-sig", newline="") as f:
        rows_csv = list(csv.reader(f))
    headers, data = rows_csv[0], rows_csv[1:]
    me_records, _ = load_main_effects_records(headers, data)
    soi_records, _ = load_soi_records(headers, data)
    if len(me_records) != len(data) or len(soi_records) != len(data):
        raise ValueError("Record length mismatch with CSV rows")

    group_col = _find_col(headers, "student_0, senior_1, genAI_2")
    out: list[dict] = []
    for i, row in enumerate(data):
        gid = row[group_col].strip()
        if gid in HUMAN_GROUP_IDS:
            group = "Human"
        elif gid in GENAI_GROUP_IDS:
            group = "GenAI"
        else:
            continue
        for task_key, task_label, acc_key in TASKS:
            ok = True
            vals: dict[str, float] = {}
            for ckey, _, prefix_tmpl in COMPLEXITY_DEFS:
                col = _find_col(headers, prefix_tmpl.format(task=task_key))
                v = _to_float(row[col]) if len(row) > col else None
                if v is None:
                    ok = False
                    break
                vals[ckey] = v
            if not ok:
                continue
            acc_me = float(me_records[i][acc_key])
            acc_soi = float(soi_records[i][acc_key])
            if not np.isfinite(acc_me):
                continue
            out.append({
                "person_id": i,
                "group": group,
                "group_id": gid,
                "task": task_key,
                "task_label": task_label,
                "acc_me": acc_me,
                "acc_soi": acc_soi if np.isfinite(acc_soi) else np.nan,
                **vals,
            })
    return out


def complexity_z_stats(person_rows: list[dict]) -> dict[str, tuple[float, float]]:
    """Mean and SD of each complexity metric (person×task rows; population SD)."""
    out: dict[str, tuple[float, float]] = {}
    for ckey, _, _ in COMPLEXITY_DEFS:
        xs = np.asarray([float(r[ckey]) for r in person_rows], dtype=float)
        mu = float(xs.mean())
        sd = float(xs.std(ddof=0))
        if sd <= 0:
            sd = 1.0
        out[ckey] = (mu, sd)
    return out


def _z(c: float, mu: float, sd: float) -> float:
    return (c - mu) / sd


def _ols_simple(
    y: np.ndarray,
    C: np.ndarray,
    genai: np.ndarray,
) -> dict[str, float]:
    """Accuracy ~ 1 + C_z + GenAI + C_z×GenAI (HC1)."""
    inter = C * genai
    X = np.column_stack([np.ones(len(y)), C, genai, inter])
    beta, se, p = ols_hc1_inference(X, y)
    # Human slope = β1; GenAI slope = β1 + β3
    return {
        "n": float(len(y)),
        "beta_c": float(beta[1]),
        "se_c": float(se[1]),
        "p_c": float(p[1]),
        "beta_genai": float(beta[2]),
        "se_genai": float(se[2]),
        "p_genai": float(p[2]),
        "beta_inter": float(beta[3]),
        "se_inter": float(se[3]),
        "p_inter": float(p[3]),
        "slope_human": float(beta[1]),
        "slope_genai": float(beta[1] + beta[3]),
        "intercept_human": float(beta[0]),
        "intercept_genai": float(beta[0] + beta[2]),
    }


ACCURACY_DEFS = (
    ("Main effects", "acc_me", "Main effects"),
    ("Interactions", "acc_soi", "Interactions"),
)


def fit_per_metric_models(
    person_rows: list[dict],
    z_stats: dict[str, tuple[float, float]],
) -> list[dict]:
    results: list[dict] = []
    for task_key, task_label, _ in TASKS:
        for acc_label, acc_key, acc_short in ACCURACY_DEFS:
            for ckey, clabel, _ in COMPLEXITY_DEFS:
                rows = [
                    r for r in person_rows
                    if r["task"] == task_key and np.isfinite(r[acc_key])
                ]
                if len(rows) < 8:
                    continue
                mu, sd = z_stats[ckey]
                y = np.asarray([r[acc_key] for r in rows], dtype=float)
                C = np.asarray(
                    [_z(float(r[ckey]), mu, sd) for r in rows], dtype=float
                )
                genai = np.asarray(
                    [1.0 if r["group"] == "GenAI" else 0.0 for r in rows],
                    dtype=float,
                )
                fit = _ols_simple(y, C, genai)
                results.append({
                    "task": task_key,
                    "task_label": task_label,
                    "complexity_key": ckey,
                    "complexity_label": clabel,
                    "accuracy": acc_label,
                    "accuracy_short": acc_short,
                    "pooled": False,
                    "c_mean": mu,
                    "c_sd": sd,
                    **fit,
                })
    return results


def _cluster_cr1_cov(
    X: np.ndarray,
    y: np.ndarray,
    clusters: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """OLS CR1: beta, se, p, cov, G (same finite-sample correction as moderator_test)."""
    n, k = X.shape
    beta, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    uniq, inv = np.unique(clusters, return_inverse=True)
    g = len(uniq)
    if g <= k or n <= k:
        nan = np.full(k, np.nan)
        return beta, nan, nan, np.full((k, k), np.nan), g
    meat = np.zeros((k, k))
    for j in range(g):
        idx = inv == j
        score = X[idx].T @ resid[idx]
        meat += np.outer(score, score)
    xtx_inv = np.linalg.inv(X.T @ X)
    scale = (g / (g - 1.0)) * ((n - 1.0) / (n - rank))
    cov = scale * (xtx_inv @ meat @ xtx_inv)
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    df = max(g - 1, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stats = beta / se
    p = 2 * t_dist.sf(np.abs(t_stats), df)
    return beta, se, p, cov, g


def _ci95(coef: float, se: float, df: float) -> tuple[float, float]:
    if not (np.isfinite(coef) and np.isfinite(se) and df > 0):
        return (float("nan"), float("nan"))
    tcrit = float(t_dist.ppf(0.975, df))
    return (coef - tcrit * se, coef + tcrit * se)


def _cell_fe_dummies(task: str, acc_label: str) -> list[float]:
    """
    task × effect four-cell FE; reference = Racial × Main effects.
    Returns [Race×Interactions, Gender×Main effects, Gender×Interactions].
    """
    return [
        1.0 if task == "Race" and acc_label == "Interactions" else 0.0,
        1.0 if task == "Gender" and acc_label == "Main effects" else 0.0,
        1.0 if task == "Gender" and acc_label == "Interactions" else 0.0,
    ]


def _pooled_design_matrix(
    person_rows: list[dict],
    z_stats: dict[str, tuple[float, float]],
    ckey: str | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Stack all task × forecasting-accuracy cells.
    Always include GenAI + task×effect cell FE (ref = Racial × Main effects).
    If ckey is set, also include standardized C.

    Columns without C: 1, GenAI, cell FE (3)
    Columns with C:    1, GenAI, C, cell FE (3)
    """
    y_list: list[float] = []
    x_rows: list[list[float]] = []
    clusters: list[int] = []
    mu = sd = 0.0
    if ckey is not None:
        mu, sd = z_stats[ckey]
    for r in person_rows:
        for acc_label, acc_key, _ in ACCURACY_DEFS:
            yv = r[acc_key]
            if not np.isfinite(yv):
                continue
            genai = 1.0 if r["group"] == "GenAI" else 0.0
            cell = _cell_fe_dummies(r["task"], acc_label)
            if ckey is None:
                x_rows.append([genai, *cell])
            else:
                x_rows.append([genai, _z(float(r[ckey]), mu, sd), *cell])
            y_list.append(float(yv))
            clusters.append(int(r["person_id"]))
    y = np.asarray(y_list, dtype=float)
    X = np.column_stack([np.ones(len(y)), np.asarray(x_rows, dtype=float)])
    return y, X, np.asarray(clusters, dtype=int)


def fit_pooled_per_metric_models(
    person_rows: list[dict],
    z_stats: dict[str, tuple[float, float]],
) -> list[dict]:
    """
    Pool all task × forecasting-accuracy cells (C standardized).

      Accuracy = β0 + β1 C + β2 GenAI + β3 (C×GenAI)
                 + δ_{task×effect} + ε

    CR1 SEs clustered by person. Cell FE reference = Racial × Main effects.
    """
    results: list[dict] = []
    for ckey, clabel, _ in COMPLEXITY_DEFS:
        mu, sd = z_stats[ckey]
        y_list: list[float] = []
        C_list: list[float] = []
        genai_list: list[float] = []
        cell_rows: list[list[float]] = []
        clusters: list[int] = []
        for r in person_rows:
            for acc_label, acc_key, _ in ACCURACY_DEFS:
                yv = r[acc_key]
                if not np.isfinite(yv):
                    continue
                y_list.append(float(yv))
                C_list.append(_z(float(r[ckey]), mu, sd))
                genai_list.append(1.0 if r["group"] == "GenAI" else 0.0)
                cell_rows.append(_cell_fe_dummies(r["task"], acc_label))
                clusters.append(int(r["person_id"]))

        y = np.asarray(y_list, dtype=float)
        C = np.asarray(C_list, dtype=float)
        genai = np.asarray(genai_list, dtype=float)
        inter = C * genai
        cells = np.asarray(cell_rows, dtype=float)
        X = np.column_stack([
            np.ones(len(y)),
            C,
            genai,
            inter,
            cells,
        ])
        cl = np.asarray(clusters, dtype=int)
        beta, se, p, cov, g = _cluster_cr1_cov(X, y, cl)
        df = float(max(g - 1, 1))

        # Indices: 0 intercept, 1 C, 2 GenAI, 3 C×GenAI, 4–6 cell FE
        slope_h = float(beta[1])
        slope_g = float(beta[1] + beta[3])
        se_slope_g = float(np.sqrt(max(
            cov[1, 1] + cov[3, 3] + 2.0 * cov[1, 3], 0.0
        )))
        ci_c = _ci95(float(beta[1]), float(se[1]), df)
        ci_genai = _ci95(float(beta[2]), float(se[2]), df)
        ci_inter = _ci95(float(beta[3]), float(se[3]), df)
        ci_slope_h = ci_c
        ci_slope_g = _ci95(slope_g, se_slope_g, df)

        results.append({
            "task": "Pooled",
            "task_label": "Pooled",
            "complexity_key": ckey,
            "complexity_label": clabel,
            "accuracy": "All (task×effect FE)",
            "accuracy_short": "Pooled",
            "pooled": True,
            "c_mean": mu,
            "c_sd": sd,
            "n": float(len(y)),
            "n_clusters": float(g),
            "df": df,
            "beta_c": float(beta[1]),
            "se_c": float(se[1]),
            "p_c": float(p[1]),
            "ci_c_lo": ci_c[0],
            "ci_c_hi": ci_c[1],
            "beta_genai": float(beta[2]),
            "se_genai": float(se[2]),
            "p_genai": float(p[2]),
            "ci_genai_lo": ci_genai[0],
            "ci_genai_hi": ci_genai[1],
            "beta_inter": float(beta[3]),
            "se_inter": float(se[3]),
            "p_inter": float(p[3]),
            "ci_inter_lo": ci_inter[0],
            "ci_inter_hi": ci_inter[1],
            "slope_human": slope_h,
            "slope_genai": slope_g,
            "se_slope_genai": se_slope_g,
            "ci_slope_human_lo": ci_slope_h[0],
            "ci_slope_human_hi": ci_slope_h[1],
            "ci_slope_genai_lo": ci_slope_g[0],
            "ci_slope_genai_hi": ci_slope_g[1],
            "intercept_human": float(beta[0]),
            "intercept_genai": float(beta[0] + beta[2]),
        })
    return results


def _bootstrap_delta_genai(
    person_rows: list[dict],
    z_stats: dict[str, tuple[float, float]],
    *,
    n_boot: int = 2000,
    seed: int = 20260807,
) -> dict[str, dict[str, float]]:
    """
    Person-cluster bootstrap of Δ = β'_GenAI − β_GenAI for each complexity metric.
    Returns se / percentile CI / t-based p (df = G−1) per ckey.
    """
    by_pid: dict[int, list[dict]] = {}
    for r in person_rows:
        by_pid.setdefault(int(r["person_id"]), []).append(r)
    pids = np.asarray(sorted(by_pid), dtype=int)
    g = int(len(pids))
    df = float(max(g - 1, 1))
    rng = np.random.default_rng(seed)

    point: dict[str, float] = {}
    y1, X1, cl1 = _pooled_design_matrix(person_rows, z_stats, ckey=None)
    b1, *_ = _cluster_cr1_cov(X1, y1, cl1)
    beta_m1 = float(b1[1])
    for ckey, _, _ in COMPLEXITY_DEFS:
        y2, X2, cl2 = _pooled_design_matrix(person_rows, z_stats, ckey=ckey)
        b2, *_ = _cluster_cr1_cov(X2, y2, cl2)
        point[ckey] = float(b2[1]) - beta_m1

    boots: dict[str, list[float]] = {ckey: [] for ckey, _, _ in COMPLEXITY_DEFS}
    for _ in range(n_boot):
        draw = rng.choice(pids, size=g, replace=True)
        rows_b: list[dict] = []
        for i, pid in enumerate(draw):
            for r in by_pid[int(pid)]:
                rr = dict(r)
                rr["person_id"] = i
                rows_b.append(rr)
        try:
            y1b, X1b, cl1b = _pooled_design_matrix(rows_b, z_stats, ckey=None)
            b1b, *_ = _cluster_cr1_cov(X1b, y1b, cl1b)
            beta1b = float(b1b[1])
            for ckey, _, _ in COMPLEXITY_DEFS:
                y2b, X2b, cl2b = _pooled_design_matrix(rows_b, z_stats, ckey=ckey)
                b2b, *_ = _cluster_cr1_cov(X2b, y2b, cl2b)
                boots[ckey].append(float(b2b[1]) - beta1b)
        except Exception:
            continue

    out: dict[str, dict[str, float]] = {}
    for ckey, _, _ in COMPLEXITY_DEFS:
        arr = np.asarray(boots[ckey], dtype=float)
        se = float(arr.std(ddof=1)) if len(arr) > 1 else float("nan")
        d0 = point[ckey]
        if np.isfinite(se) and se > 0:
            tstat = d0 / se
            p = float(2 * t_dist.sf(abs(tstat), df))
            ci = _ci95(d0, se, df)
        else:
            p = float("nan")
            ci = (float("nan"), float("nan"))
        # Percentile CI kept for reference / CSV
        if len(arr) > 0:
            plo, phi = (float(x) for x in np.quantile(arr, [0.025, 0.975]))
        else:
            plo = phi = float("nan")
        out[ckey] = {
            "delta": d0,
            "se": se,
            "p": p,
            "ci_lo": ci[0],
            "ci_hi": ci[1],
            "ci_pct_lo": plo,
            "ci_pct_hi": phi,
            "n_boot": float(len(arr)),
            "df": df,
        }
    return out


def fit_genai_gap_robustness(
    person_rows: list[dict],
    z_stats: dict[str, tuple[float, float]],
) -> list[dict]:
    """
    Panel b robustness: does controlling for C attenuate the GenAI gap?

      Model 1: Accuracy = α + β GenAI + δ_{task×effect} + ε
      Model 2: Accuracy = α + β' GenAI + β_C C + δ_{task×effect} + ε

    Inference for Δ = β' − β uses person-cluster bootstrap SEs.
    """
    y1, X1, cl1 = _pooled_design_matrix(person_rows, z_stats, ckey=None)
    beta1, se1, p1, _, g1 = _cluster_cr1_cov(X1, y1, cl1)
    df1 = float(max(g1 - 1, 1))
    # X1 cols: 0 intercept, 1 GenAI, 2–4 cell FE
    b_m1 = float(beta1[1])
    se_m1 = float(se1[1])
    p_m1 = float(p1[1])
    ci_m1 = _ci95(b_m1, se_m1, df1)

    delta_boot = _bootstrap_delta_genai(person_rows, z_stats)

    out: list[dict] = []
    for ckey, clabel, _ in COMPLEXITY_DEFS:
        y2, X2, cl2 = _pooled_design_matrix(person_rows, z_stats, ckey=ckey)
        beta2, se2, p2, _, g2 = _cluster_cr1_cov(X2, y2, cl2)
        df2 = float(max(g2 - 1, 1))
        # X2 cols: 0 intercept, 1 GenAI, 2 C, 3–5 cell FE
        b_m2 = float(beta2[1])
        se_m2 = float(se2[1])
        p_m2 = float(p2[1])
        ci_m2 = _ci95(b_m2, se_m2, df2)
        b_c = float(beta2[2])
        se_c = float(se2[2])
        p_c = float(p2[2])
        ci_c = _ci95(b_c, se_c, df2)
        db = delta_boot[ckey]
        out.append({
            "complexity_key": ckey,
            "complexity_label": clabel,
            "n": float(len(y2)),
            "n_clusters": float(g2),
            "df": df2,
            "beta_genai_m1": b_m1,
            "se_genai_m1": se_m1,
            "p_genai_m1": p_m1,
            "ci_genai_m1_lo": ci_m1[0],
            "ci_genai_m1_hi": ci_m1[1],
            "beta_genai_m2": b_m2,
            "se_genai_m2": se_m2,
            "p_genai_m2": p_m2,
            "ci_genai_m2_lo": ci_m2[0],
            "ci_genai_m2_hi": ci_m2[1],
            "beta_c_m2": b_c,
            "se_c_m2": se_c,
            "p_c_m2": p_c,
            "ci_c_m2_lo": ci_c[0],
            "ci_c_m2_hi": ci_c[1],
            "delta_genai": db["delta"],
            "se_delta_genai": db["se"],
            "p_delta_genai": db["p"],
            "ci_delta_genai_lo": db["ci_lo"],
            "ci_delta_genai_hi": db["ci_hi"],
        })
    return out


MULTI_TERM_NAMES = (
    "intercept",
    "n_paths",
    "max_path_len",
    "n_latents",
    "genai",
    "n_paths_x_genai",
    "max_path_len_x_genai",
    "n_latents_x_genai",
)


def fit_multivariate_by_task(
    person_rows: list[dict],
    z_stats: dict[str, tuple[float, float]],
) -> dict[str, dict]:
    """
    Per task (Main-effects accuracy, HC1; standardized C_k):
      Accuracy = β0 + Σ β_k C_k,z + β_g GenAI + Σ γ_k (C_k,z × GenAI) + ε
    """
    out: dict[str, dict] = {}
    for task_key, task_label, _ in TASKS:
        rows = [r for r in person_rows if r["task"] == task_key]
        y = np.asarray([float(r["acc_me"]) for r in rows], dtype=float)
        x_rows: list[list[float]] = []
        for r in rows:
            c_vals = [
                _z(float(r[ckey]), *z_stats[ckey])
                for ckey, _, _ in COMPLEXITY_DEFS
            ]
            genai = 1.0 if r["group"] == "GenAI" else 0.0
            inter = [c * genai for c in c_vals]
            x_rows.append([*c_vals, genai, *inter])
        X = np.column_stack([np.ones(len(y)), np.asarray(x_rows, dtype=float)])
        beta, se, p = ols_hc1_inference(X, y)
        task_out: dict = {
            "task": task_key,
            "task_label": task_label,
            "n_obs": len(y),
        }
        for j, name in enumerate(MULTI_TERM_NAMES):
            task_out[name] = {
                "coef": float(beta[j]),
                "se": float(se[j]),
                "p": float(p[j]),
            }
        out[task_key] = task_out
    return out


def _sig_star(p: float) -> str:
    if not np.isfinite(p) or p >= 0.05:
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    return "*"


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {path}")


def _fmt_coef_tex(coef: float, p: float) -> str:
    if not np.isfinite(coef):
        return "---"
    star = _sig_star(p)
    star_tex = f"$^{{{star}}}$" if star else ""
    if coef < 0:
        return rf"$-{abs(coef):.3f}${star_tex}"
    return f"{coef:.3f}{star_tex}"


def _fmt_ci_tex(lo: float, hi: float) -> str:
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "---"

    def _one(v: float) -> str:
        if v < 0:
            return f"-{abs(v):.3f}"
        return f"{v:.3f}"

    return rf"$[{_one(lo)},\ {_one(hi)}]$"


def _fmt_coef_ci_cell(coef: float, lo: float, hi: float, p: float) -> str:
    return (
        r"\begin{tabular}{@{}c@{}}"
        + _fmt_coef_tex(coef, p)
        + r"\\"
        + _fmt_ci_tex(lo, hi)
        + r"\end{tabular}"
    )


def build_coef_tables_tex(
    pooled_fits: list[dict],
    gap_fits: list[dict],
) -> str:
    """Panel a: interaction model; Panel b: GenAI gap with/without C."""
    a_lines: list[str] = []
    for f in pooled_fits:
        se_sg = float(f["se_slope_genai"])
        if np.isfinite(se_sg) and se_sg > 0:
            p_slope_g = float(
                2 * t_dist.sf(abs(f["slope_genai"] / se_sg), f["df"])
            )
        else:
            p_slope_g = float("nan")
        a_lines.append(
            " & ".join([
                f["complexity_label"],
                _fmt_coef_ci_cell(
                    f["beta_c"], f["ci_c_lo"], f["ci_c_hi"], f["p_c"]
                ),
                _fmt_coef_ci_cell(
                    f["beta_genai"], f["ci_genai_lo"], f["ci_genai_hi"], f["p_genai"]
                ),
                _fmt_coef_ci_cell(
                    f["beta_inter"], f["ci_inter_lo"], f["ci_inter_hi"], f["p_inter"]
                ),
                _fmt_coef_ci_cell(
                    f["slope_human"],
                    f["ci_slope_human_lo"],
                    f["ci_slope_human_hi"],
                    f["p_c"],
                ),
                _fmt_coef_ci_cell(
                    f["slope_genai"],
                    f["ci_slope_genai_lo"],
                    f["ci_slope_genai_hi"],
                    p_slope_g,
                ),
            ])
            + r" \\"
        )
        a_lines.append("")

    b_lines: list[str] = []
    for f in gap_fits:
        b_lines.append(
            " & ".join([
                f["complexity_label"],
                _fmt_coef_ci_cell(
                    f["beta_genai_m1"],
                    f["ci_genai_m1_lo"],
                    f["ci_genai_m1_hi"],
                    f["p_genai_m1"],
                ),
                _fmt_coef_ci_cell(
                    f["beta_genai_m2"],
                    f["ci_genai_m2_lo"],
                    f["ci_genai_m2_hi"],
                    f["p_genai_m2"],
                ),
                _fmt_coef_ci_cell(
                    f["delta_genai"],
                    f["ci_delta_genai_lo"],
                    f["ci_delta_genai_hi"],
                    f["p_delta_genai"],
                ),
            ])
            + r" \\"
        )
        b_lines.append("")

    return "\n".join([
        "% Auto-generated by forecasts/theory_efficiency.py",
        "",
        r"\begin{tabular}{@{}l@{}}",
        r"{\small\textbf{a}\enspace Association between diagram complexity and forecasting accuracy}\\[0.25em]",
        r"{\footnotesize",
        r"\setlength{\tabcolsep}{5pt}",
        r"\renewcommand{\arraystretch}{1.12}",
        r"\begin{tabular}{@{}lccccc@{}}",
        r"\toprule",
        r"\textbf{Complexity metric ($C$)}",
        r"& $\boldsymbol{\beta}_{1}$ ($C$)",
        r"& $\boldsymbol{\beta}_{2}$ (GenAI)",
        r"& $\boldsymbol{\beta}_{3}$ ($C{\times}$GenAI)",
        r"& \textbf{Human slope}",
        r"& \textbf{GenAI slope} \\",
        r"\midrule",
        *a_lines,
        r"\bottomrule",
        r"\end{tabular}",
        r"} \\",
        # Zero-width rule row: \\[dim] changes of ~0.8em are hard to see; use a
        # clearly larger strut so panel a/b separation is visible.
        r"\rule{0pt}{1.3em} \\",
        r"{\small\textbf{b}\enspace GenAI--human accuracy differences adjusting for complexity}\\[0.25em]",
        r"{\footnotesize",
        r"\setlength{\tabcolsep}{5pt}",
        r"\renewcommand{\arraystretch}{1.12}",
        r"\begin{tabular}{@{}lccc@{}}",
        r"\toprule",
        r"\textbf{Complexity metric ($C$)}",
        r"& $\boldsymbol{\gamma}$ (GenAI, no $C$)",
        r"& $\boldsymbol{\gamma}'$ (GenAI, with $C$)",
        r"& $\boldsymbol{\gamma}'-\boldsymbol{\gamma}$ \\",
        r"\midrule",
        *b_lines,
        r"\bottomrule",
        r"\end{tabular}",
        r"} \\",
        r"\end{tabular}",
        "",
    ])


def write_coef_tables(
    pooled_fits: list[dict],
    gap_fits: list[dict],
) -> Path:
    from latex_table_pdf import compile_standalone_table

    stem = "theory_efficiency_coef_table"
    body = build_coef_tables_tex(pooled_fits, gap_fits)
    path = compile_standalone_table(
        OUT_DIR,
        stem,
        body,
        output_format="pdf+svg",
        crop="standalone",
        extra_packages=[r"\usepackage{multirow}"],
    )
    (OUT_DIR / f"{stem}_standalone.png").unlink(missing_ok=True)
    print(f"Saved {OUT_DIR / f'{stem}_standalone.pdf'}")
    print(f"Saved {OUT_DIR / f'{stem}_standalone.svg'}")
    return path


def main() -> None:
    person_rows = build_person_rows()
    z_stats = complexity_z_stats(person_rows)
    print("Complexity standardization (person×task):")
    for ckey, clabel, _ in COMPLEXITY_DEFS:
        mu, sd = z_stats[ckey]
        print(f"  {clabel}: mean={mu:.3f}, SD={sd:.3f}")
    n_human = sum(1 for r in person_rows if r["group"] == "Human" and r["task"] == "Race")
    n_genai = sum(1 for r in person_rows if r["group"] == "GenAI" and r["task"] == "Race")
    print(f"Person×task rows: {len(person_rows)}  (Race: Humans={n_human}, GenAI={n_genai})")

    fits = fit_per_metric_models(person_rows, z_stats)
    pooled_fits = fit_pooled_per_metric_models(person_rows, z_stats)
    gap_fits = fit_genai_gap_robustness(person_rows, z_stats)
    multi_by_task = fit_multivariate_by_task(person_rows, z_stats)

    write_coef_tables(pooled_fits, gap_fits)

    summary_rows = []
    for f in fits + pooled_fits:
        summary_rows.append({
            "task": f["task"],
            "accuracy": f["accuracy"],
            "pooled": bool(f.get("pooled", False)),
            "complexity": f["complexity_key"],
            "complexity_label": f["complexity_label"],
            "c_mean": f.get("c_mean", ""),
            "c_sd": f.get("c_sd", ""),
            "n": int(f["n"]),
            "n_clusters": int(f["n_clusters"]) if "n_clusters" in f else "",
            "slope_human": f["slope_human"],
            "slope_genai": f["slope_genai"],
            "beta_c": f["beta_c"],
            "se_c": f["se_c"],
            "p_c": f["p_c"],
            "ci_c_lo": f.get("ci_c_lo", ""),
            "ci_c_hi": f.get("ci_c_hi", ""),
            "beta_genai": f["beta_genai"],
            "se_genai": f["se_genai"],
            "p_genai": f["p_genai"],
            "ci_genai_lo": f.get("ci_genai_lo", ""),
            "ci_genai_hi": f.get("ci_genai_hi", ""),
            "beta_interaction": f["beta_inter"],
            "se_interaction": f["se_inter"],
            "p_interaction": f["p_inter"],
            "ci_interaction_lo": f.get("ci_inter_lo", ""),
            "ci_interaction_hi": f.get("ci_inter_hi", ""),
            "ci_slope_human_lo": f.get("ci_slope_human_lo", ""),
            "ci_slope_human_hi": f.get("ci_slope_human_hi", ""),
            "ci_slope_genai_lo": f.get("ci_slope_genai_lo", ""),
            "ci_slope_genai_hi": f.get("ci_slope_genai_hi", ""),
            "sig_interaction": _sig_star(f["p_inter"]) or "NS",
        })
    _write_csv(OUT_DIR / "theory_efficiency_summary.csv", summary_rows)

    multi_rows = []
    slope_of = {
        "n_paths": "n_paths_x_genai",
        "max_path_len": "max_path_len_x_genai",
        "n_latents": "n_latents_x_genai",
    }
    for task_key, _, _ in TASKS:
        m = multi_by_task[task_key]
        for name in MULTI_TERM_NAMES:
            stats = m[name]
            row = {
                "task": task_key,
                "term": name,
                "coef": stats["coef"],
                "se": stats["se"],
                "p": stats["p"],
                "sig": _sig_star(stats["p"]) or "NS",
                "n_obs": m["n_obs"],
                "slope_human": "",
                "slope_genai": "",
            }
            if name in slope_of:
                beta = float(stats["coef"])
                gamma = float(m[slope_of[name]]["coef"])
                row["slope_human"] = beta
                row["slope_genai"] = beta + gamma
            multi_rows.append(row)
    _write_csv(OUT_DIR / "theory_efficiency_multivariate.csv", multi_rows)

    print("\nPer-metric C×GenAI interactions (stratified):")
    for f in fits:
        print(
            f"  {f['accuracy']:14} {f['task_label']:22} {f['complexity_label']:28} "
            f"γ={f['beta_inter']:+.4f}  p={f['p_inter']:.3f} "
            f"({_sig_star(f['p_inter']) or 'NS'})  "
            f"slopes H={f['slope_human']:+.4f} AI={f['slope_genai']:+.4f}"
        )
    print("\nPooled interaction models (Panel a):")
    for f in pooled_fits:
        print(
            f"  {f['complexity_label']:28} "
            f"β3={f['beta_inter']:+.4f}  p={f['p_inter']:.3f} "
            f"({_sig_star(f['p_inter']) or 'NS'})  "
            f"slopes H={f['slope_human']:+.4f} AI={f['slope_genai']:+.4f}  "
            f"N={int(f['n'])}  clusters={int(f['n_clusters'])}"
        )
    print("\nGenAI gap with/without C (Panel b):")
    for f in gap_fits:
        print(
            f"  {f['complexity_label']:28} "
            f"β={f['beta_genai_m1']:+.4f}  β'={f['beta_genai_m2']:+.4f}  "
            f"Δ={f['delta_genai']:+.4f}  p_Δ={f['p_delta_genai']:.3f} "
            f"({_sig_star(f['p_delta_genai']) or 'NS'})  "
            f"CI_Δ=[{f['ci_delta_genai_lo']:+.4f}, {f['ci_delta_genai_hi']:+.4f}]  "
            f"β_C={f['beta_c_m2']:+.4f}"
        )
    print("\nMultivariate by task (HC1):")
    for row in multi_rows:
        if "x_genai" in row["term"] or row["term"] in (
            "n_paths", "max_path_len", "n_latents", "genai"
        ):
            print(
                f"  {row['task']:8} {row['term']:24} β={row['coef']:+.4f}  "
                f"p={row['p']:.3f} ({row['sig']})"
            )


if __name__ == "__main__":
    main()
