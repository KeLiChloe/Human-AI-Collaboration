#!/usr/bin/env python3
"""
BERTopic candidate-topic discovery for claim-level theory explanations.

Run: python step2_bertopic_claim_topic_modeling.py
Writes all 8 panels under bertopic_outputs/
(pre/post-ML × race/gender × main-effects/soi).

Discovery only: inspect candidate topic families, then run step3 for LLM naming.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_JSON = SCRIPT_DIR / "claim_level_dataset.json"
BERTOPIC_OUTPUTS_ROOT = SCRIPT_DIR / "bertopic_outputs"
EMBEDDING_CACHE_PATH = SCRIPT_DIR / "cache" / "openai_embedding_cache.jsonl"

OUTCOME_SLUG_TO_LABEL = {
    "race": "racial inequality",
    "gender": "gender inequality",
}
THEORY_SLUG_TO_LABEL = {
    "main-effects": "main effects",
    "soi": "second-order interactions",
}

# (stage, outcome_slug, theory_slug) — same 8 panels as step1 extraction jobs
PANEL_SPECS: list[tuple[str, str, str]] = [
    ("pre-ML", "race", "main-effects"),
    ("pre-ML", "race", "soi"),
    ("post-ML", "race", "main-effects"),
    ("post-ML", "race", "soi"),
    ("pre-ML", "gender", "main-effects"),
    ("pre-ML", "gender", "soi"),
    ("post-ML", "gender", "main-effects"),
    ("post-ML", "gender", "soi"),
]

CLAIM_ASSIGNMENTS_CSV = "claim_topic_assignments.csv"
CANDIDATE_TOPICS_JSON = "candidate_topics_for_llm_labeling.json"

# Defaults tuned for claim_text on current dataset (~73–128 causal claims per 8-way cell;
# median claim length ~150 chars, min ~51; review flags ~5–13% of causal claims).
DEFAULT_MIN_CHARS = 50
DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-large"
DEFAULT_OPENAI_DIMENSIONS = 1024
DEFAULT_EMBEDDING_BATCH_SIZE = 64
DEFAULT_MIN_TOPIC_SIZE = 5
DEFAULT_MIN_SAMPLES = 2
DEFAULT_N_NEIGHBORS = 10
DEFAULT_N_COMPONENTS = 10
DEFAULT_TOP_N_WORDS = 10
DEFAULT_TOP_K_EXAMPLES = 20
DEFAULT_SEED = 12345

# Boilerplate that appears in almost every claim (outcome framing, generic verbs).
# Do NOT add observable-feature vocabulary here (author, female, social science, news, etc.).
DOMAIN_STOP_WORDS = [
    "paper",
    "papers",
    "discuss",
    "discusses",
    "discussing",
    "discussed",
    "discussion",
    "inequality",
    "inequalities",
    "probability",
    "probabilities",
    "likely",
    "likelihood",
    "effect",
    "effects",
    "increase",
    "increases",
    "increased",
    "decrease",
    "decreases",
    "decreased",
    "positive",
    "negative",
    "relationship",
    "mention",
    "mentions",
    "mentioned",
]


def stable_text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def safe_get_nested(d: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    cur: Any = d
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def normalize_group_label(x: Any) -> str:
    if x is None:
        return "unknown"
    s = str(x).strip()
    mapping = {
        "0": "student",
        "1": "senior",
        "2": "GenAI",
        "genai": "GenAI",
        "ai": "GenAI",
        "student": "student",
        "senior": "senior",
    }
    return mapping.get(s.lower(), s)


def flatten_claim_json(
    input_json: Path,
    stage: str,
    outcome: str,
    theory_type: str,
    text_field: str,
    causal_only: bool,
    exclude_review: bool,
    min_chars: int,
) -> pd.DataFrame:
    with input_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    rows: list[dict[str, Any]] = []
    for participant_name, pdata in data.items():
        group = pdata.get("group")
        group_label = normalize_group_label(pdata.get("group_label", group))
        claims = safe_get_nested(pdata, [stage, outcome, theory_type], default=[])
        if not isinstance(claims, list):
            continue

        for claim in claims:
            if not isinstance(claim, dict):
                continue
            if causal_only and claim.get("is_causal_claim") is not True:
                continue
            if exclude_review and claim.get("needs_human_review") is True:
                continue
            text = str(claim.get(text_field, "") or "").strip()
            if text.lower() == "unspecified":
                continue
            if len(text) < min_chars:
                continue

            rows.append(
                {
                    "participant_name": participant_name,
                    "group": group,
                    "group_label": group_label,
                    "stage": stage,
                    "outcome": outcome,
                    "theory_type": theory_type,
                    "claim_id": claim.get("claim_id"),
                    "claim_text": claim.get("claim_text", ""),
                    "supporting_text": claim.get("supporting_text", ""),
                    "topic_model_text": text,
                    "is_causal_claim": claim.get("is_causal_claim"),
                    "antecedent_text": claim.get("antecedent_text", ""),
                    "mechanism_text": claim.get("mechanism_text", ""),
                    "direction": claim.get("direction", ""),
                    "needs_human_review": claim.get("needs_human_review", False),
                    "review_reason": claim.get("review_reason", ""),
                    "claim_uid": stable_text_hash(
                        f"{participant_name}||{stage}||{outcome}||{theory_type}"
                        f"||{claim.get('claim_id')}||{text}"
                    )[:16],
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(
            "No claims matched the requested filters. Check stage/outcome/theory_type "
            "or turn off causal/review filters."
        )
    return df


def load_embedding_cache(cache_path: Path) -> dict[str, list[float]]:
    cache: dict[str, list[float]] = {}
    if not cache_path.exists():
        return cache
    with cache_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                cache[obj["key"]] = obj["embedding"]
            except Exception:
                continue
    return cache


def append_embedding_cache(cache_path: Path, items: Iterable[tuple[str, list[float]]]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a", encoding="utf-8") as f:
        for key, emb in items:
            f.write(json.dumps({"key": key, "embedding": emb}) + "\n")


def batch_iter(xs: list[str], batch_size: int) -> Iterable[list[str]]:
    for i in range(0, len(xs), batch_size):
        yield xs[i : i + batch_size]


def embed_openai(
    texts: list[str],
    model: str,
    dimensions: int | None,
    cache_path: Path,
    batch_size: int = 64,
) -> np.ndarray:
    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImportError("Please install the OpenAI SDK: pip install openai") from e

    cache = load_embedding_cache(cache_path)
    keys = [stable_text_hash(f"openai||{model}||{dimensions}||{t}") for t in texts]
    missing_indices = [i for i, key in enumerate(keys) if key not in cache]

    if missing_indices and not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")

    client = OpenAI() if missing_indices else None

    for batch_indices in batch_iter(missing_indices, batch_size):
        assert client is not None
        batch_texts = [texts[i] for i in batch_indices]
        kwargs: dict[str, Any] = {"model": model, "input": batch_texts}
        if dimensions is not None:
            kwargs["dimensions"] = dimensions
        resp = client.embeddings.create(**kwargs)
        new_items: list[tuple[str, list[float]]] = []
        for idx, item in zip(batch_indices, resp.data):
            key = keys[idx]
            emb = list(item.embedding)
            cache[key] = emb
            new_items.append((key, emb))
        append_embedding_cache(cache_path, new_items)

    return np.array([cache[key] for key in keys], dtype=np.float32)


def vectorizer_stop_words() -> list[str]:
    return sorted(set(ENGLISH_STOP_WORDS) | set(DOMAIN_STOP_WORDS))


def fit_bertopic(
    docs: list[str],
    embeddings: np.ndarray,
    min_topic_size: int,
    min_samples: int | None,
    n_neighbors: int,
    n_components: int,
    seed: int,
    top_n_words: int,
) -> tuple[Any, list[int], np.ndarray | None]:
    try:
        from bertopic import BERTopic
        from hdbscan import HDBSCAN
        from sklearn.feature_extraction.text import CountVectorizer
        from umap import UMAP
    except ImportError as e:
        raise ImportError(
            "Please install BERTopic dependencies: pip install bertopic hdbscan umap-learn scikit-learn"
        ) from e

    n_docs = len(docs)
    if n_docs < 5:
        raise ValueError("Too few documents for BERTopic. Need at least about 5 claims.")

    effective_neighbors = max(2, min(n_neighbors, n_docs - 1))
    effective_min_topic_size = max(2, min(min_topic_size, max(2, n_docs // 3)))
    effective_min_samples = min_samples
    if effective_min_samples is not None:
        effective_min_samples = max(1, min(effective_min_samples, effective_min_topic_size))

    umap_model = UMAP(
        n_neighbors=effective_neighbors,
        n_components=n_components,
        min_dist=0.0,
        metric="cosine",
        random_state=seed,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=effective_min_topic_size,
        min_samples=effective_min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    vectorizer_model = CountVectorizer(
        stop_words=vectorizer_stop_words(),
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
    )

    topic_model = BERTopic(
        embedding_model=None,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        top_n_words=top_n_words,
        calculate_probabilities=True,
        verbose=True,
    )
    topics, probs = topic_model.fit_transform(docs, embeddings)
    return topic_model, topics, probs


def assignment_probabilities(probs: np.ndarray | None, topics: list[int]) -> list[float | None]:
    if probs is None:
        return [None] * len(topics)
    arr = np.asarray(probs, dtype=float)
    if arr.ndim == 1:
        return [float(v) for v in arr]
    out: list[float | None] = []
    for i, topic_id in enumerate(topics):
        if int(topic_id) == -1:
            out.append(None)
        else:
            out.append(float(np.max(arr[i])))
    return out


def build_topic_summary(df: pd.DataFrame, topic_model: Any, top_k_examples: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    topic_ids = sorted(df["topic_id"].dropna().unique().tolist())
    for topic_id in topic_ids:
        tdf = df[df["topic_id"] == topic_id].copy()
        topic_words = topic_model.get_topic(int(topic_id)) if int(topic_id) != -1 else []
        top_terms = "; ".join([w for w, _ in (topic_words or [])[:10]])

        row: dict[str, Any] = {
            "topic_id": int(topic_id),
            "is_outlier_topic": int(topic_id) == -1,
            "n_claims": int(len(tdf)),
            "n_respondents": int(tdf["participant_name"].nunique()),
            "top_terms": top_terms,
        }

        for group_label in ["student", "senior", "GenAI"]:
            gdf = tdf[tdf["group_label"] == group_label]
            row[f"{group_label}_claims"] = int(len(gdf))
            row[f"{group_label}_respondents"] = int(gdf["participant_name"].nunique())

        if "topic_probability" in tdf.columns and tdf["topic_probability"].notna().any():
            tdf = tdf.sort_values("topic_probability", ascending=False)

        for i, (_, ex) in enumerate(tdf.head(top_k_examples).iterrows(), start=1):
            row[f"example_{i}_participant"] = ex.get("participant_name", "")
            row[f"example_{i}_group"] = ex.get("group_label", "")
            row[f"example_{i}_claim_text"] = ex.get("claim_text", "")
            row[f"example_{i}_supporting_text"] = ex.get("supporting_text", "")
            row[f"example_{i}_antecedent"] = ex.get("antecedent_text", "")
            row[f"example_{i}_mechanism"] = ex.get("mechanism_text", "")

        rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["is_outlier_topic", "n_respondents", "n_claims"], ascending=[True, False, False])
    return out


def build_llm_review_json(topic_summary: pd.DataFrame, output_path: Path) -> None:
    topics: list[dict[str, Any]] = []
    for _, row in topic_summary.iterrows():
        examples = []
        for i in range(1, 21):
            claim_col = f"example_{i}_claim_text"
            if claim_col not in row or pd.isna(row[claim_col]) or not str(row[claim_col]).strip():
                continue
            examples.append(
                {
                    "participant": row.get(f"example_{i}_participant", ""),
                    "group": row.get(f"example_{i}_group", ""),
                    "claim_text": row.get(f"example_{i}_claim_text", ""),
                    "supporting_text": row.get(f"example_{i}_supporting_text", ""),
                    "antecedent_text": row.get(f"example_{i}_antecedent", ""),
                    "mechanism_text": row.get(f"example_{i}_mechanism", ""),
                }
            )
        topics.append(
            {
                "topic_id": int(row["topic_id"]),
                "is_outlier_topic": bool(row["is_outlier_topic"]),
                "n_claims": int(row["n_claims"]),
                "n_respondents": int(row["n_respondents"]),
                "top_terms": row.get("top_terms", ""),
                "examples": examples,
            }
        )
    with output_path.open("w", encoding="utf-8") as f:
        json.dump({"topics": topics}, f, ensure_ascii=False, indent=2)


def build_diagnostics(df: pd.DataFrame, topics: list[int]) -> dict[str, Any]:
    topic_series = pd.Series(topics, name="topic_id")
    n_outliers = int((topic_series == -1).sum())
    n_topics = int(topic_series[topic_series != -1].nunique())
    review_flagged = int(df["needs_human_review"].fillna(False).astype(bool).sum())

    group_respondents = {
        str(k): int(v)
        for k, v in df.groupby("group_label")["participant_name"].nunique().items()
    }
    topic_counts = (
        df[df["topic_id"] != -1]
        .groupby("topic_id")
        .agg(n_claims=("claim_uid", "count"), n_respondents=("participant_name", "nunique"))
        .reset_index()
        .sort_values("n_respondents", ascending=False)
        .to_dict(orient="records")
    )

    return {
        "n_claims": int(len(df)),
        "n_respondents": int(df["participant_name"].nunique()),
        "n_review_flagged_claims": review_flagged,
        "review_flagged_pct": round(100.0 * review_flagged / len(df), 1),
        "n_topics_excluding_outliers": n_topics,
        "n_outlier_claims": n_outliers,
        "outlier_pct": round(100.0 * n_outliers / len(df), 1),
        "respondents_by_group": group_respondents,
        "claims_per_respondent_median": float(df.groupby("participant_name").size().median()),
        "claims_per_respondent_mean": round(float(df.groupby("participant_name").size().mean()), 2),
        "topic_sizes": topic_counts,
    }


def print_diagnostics(diagnostics: dict[str, Any]) -> None:
    print("\n=== Diagnostics ===")
    print(f"Claims: {diagnostics['n_claims']} | Respondents: {diagnostics['n_respondents']}")
    print(
        f"Review-flagged claims: {diagnostics['n_review_flagged_claims']} "
        f"({diagnostics['review_flagged_pct']}%)"
    )
    print(
        f"Topics (excl. outliers): {diagnostics['n_topics_excluding_outliers']} | "
        f"Outlier claims: {diagnostics['n_outlier_claims']} ({diagnostics['outlier_pct']}%)"
    )
    print(f"Respondents by group: {diagnostics['respondents_by_group']}")
    print(
        "Claims per respondent: "
        f"median={diagnostics['claims_per_respondent_median']}, "
        f"mean={diagnostics['claims_per_respondent_mean']}"
    )
    if diagnostics["topic_sizes"]:
        print("Top topics by respondents:")
        for row in diagnostics["topic_sizes"][:8]:
            print(
                f"  topic {row['topic_id']}: "
                f"{row['n_respondents']} respondents, {row['n_claims']} claims"
            )


def all_panels(root: Path = BERTOPIC_OUTPUTS_ROOT) -> list[tuple[Path, str, str, str]]:
    panels: list[tuple[Path, str, str, str]] = []
    for stage, outcome_slug, theory_slug in PANEL_SPECS:
        outcome = OUTCOME_SLUG_TO_LABEL[outcome_slug]
        theory_type = THEORY_SLUG_TO_LABEL[theory_slug]
        output_dir = root / f"{stage}_{outcome_slug}_{theory_slug}"
        panels.append((output_dir, stage, outcome, theory_type))
    return panels


def run_panel(
    input_path: Path,
    output_dir: Path,
    stage: str,
    outcome: str,
    theory_type: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        df = flatten_claim_json(
            input_json=input_path,
            stage=stage,
            outcome=outcome,
            theory_type=theory_type,
            text_field="claim_text",
            causal_only=True,
            exclude_review=False,
            min_chars=DEFAULT_MIN_CHARS,
        )
    except ValueError as exc:
        print(f"\n=== {output_dir.name} ===\nSkipped: {exc}")
        return

    print(
        f"\n=== {output_dir.name} ===\n"
        f"Claims for topic modeling: {len(df)} claims, "
        f"{df['participant_name'].nunique()} respondents"
    )

    docs = df["topic_model_text"].astype(str).tolist()
    embeddings = embed_openai(
        docs,
        model=DEFAULT_OPENAI_EMBEDDING_MODEL,
        dimensions=DEFAULT_OPENAI_DIMENSIONS,
        cache_path=EMBEDDING_CACHE_PATH,
        batch_size=DEFAULT_EMBEDDING_BATCH_SIZE,
    )

    try:
        topic_model, topics, probs = fit_bertopic(
            docs=docs,
            embeddings=embeddings,
            min_topic_size=DEFAULT_MIN_TOPIC_SIZE,
            min_samples=DEFAULT_MIN_SAMPLES,
            n_neighbors=DEFAULT_N_NEIGHBORS,
            n_components=DEFAULT_N_COMPONENTS,
            seed=DEFAULT_SEED,
            top_n_words=DEFAULT_TOP_N_WORDS,
        )
    except ValueError as exc:
        print(f"Skipped: {exc}")
        return

    df["topic_id"] = topics
    df["topic_probability"] = assignment_probabilities(probs, topics)

    diagnostics = build_diagnostics(df, topics)
    print_diagnostics(diagnostics)

    assignments_path = output_dir / CLAIM_ASSIGNMENTS_CSV
    df.to_csv(assignments_path, index=False)
    print(f"Saved: {assignments_path}")

    summary = build_topic_summary(df, topic_model, top_k_examples=DEFAULT_TOP_K_EXAMPLES)
    llm_review_path = output_dir / CANDIDATE_TOPICS_JSON
    build_llm_review_json(summary, llm_review_path)
    print(f"Saved: {llm_review_path}")


def main() -> None:
    input_path = DEFAULT_INPUT_JSON.expanduser().resolve()
    BERTOPIC_OUTPUTS_ROOT.mkdir(parents=True, exist_ok=True)
    panels = all_panels()

    print(f"Input: {input_path}")
    print(f"Panels ({len(panels)}): {[p[0].name for p in panels]}")

    for output_dir, stage, outcome, theory_type in panels:
        run_panel(
            input_path=input_path,
            output_dir=output_dir,
            stage=stage,
            outcome=outcome,
            theory_type=theory_type,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
