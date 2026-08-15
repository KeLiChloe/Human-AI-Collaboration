#!/usr/bin/env python3
"""
Step 4: Canonical causal chains and LLM deduplication.

1. canonicalize_claims()  — deterministic per-claim chains (no LLM)
2. dedupe_unique_chains() — LLM merges semantically equivalent chains

Run: python step4_llm_unique_causal_chains.py
Writes all 8 panels under unique_causal_chain_outputs/
  - claim_canonical_chains.csv (per-claim chains + unique_chain_id assignment)
  - unique_causal_chains.json (panel-level unique chain types)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from openai import OpenAI
from pydantic import BaseModel, Field

from step2_bertopic_claim_topic_modeling import (
    DEFAULT_INPUT_JSON,
    flatten_claim_json,
)

SCRIPT_DIR = Path(__file__).resolve().parent
CANONICAL_OUTPUTS_ROOT = SCRIPT_DIR / "unique_causal_chain_outputs"
DEFAULT_MODEL = "gpt-5.5"
DEDUPE_BATCH_SIZE = 75

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

OUTCOME_CANONICAL = {
    "racial inequality": "racial inequality discussion",
    "gender inequality": "gender inequality discussion",
}

DIRECTION_PREFIX = {
    "positive": "(+)",
    "negative": "(-)",
    "mixed": "(±)",
    "unclear": "(?)",
    "not_applicable": "(n/a)",
}

OBSERVABLE_FEATURES_BLOCK = """\
## Journal field

social_science: The paper is published in a social science journal.

natural_science: The paper is published in a natural science journal.

engineering_and_technology: The paper is published in an engineering or technology journal.

## Author characteristics

num_authors: The total number of authors.

female: The proportion of female authors.

asian: The proportion of Asian authors.

black: The proportion of Black authors.

white: The proportion of White authors.

hispanic_and_other: The proportion of Hispanic authors and authors from racial groups \
other than White, Black, or Asian.

authors_race_diversity_score: The racial diversity of the author team. Phrases such as \
"racially diverse teams," "mixed-race author teams," or "diverse coauthors" may correspond \
to this feature.

country_race_diversity_score: The racial diversity of the authors' countries of birth. \
References to diversity across authors' countries of origin or international diversity \
may correspond to this feature.

## Discourse context

news_inequality_mentions_3_years: The level of media attention to inequality during the \
three years before publication. Phrases such as "media attention," "news coverage," \
"public discourse," or "news discussion of inequality" may correspond to this feature.

paper_inequality_mentions_3_years: The level of academic attention to inequality during \
the three years before publication. Phrases such as "prior research," "recent papers," \
"scholarly attention," or "academic discussion of inequality" may correspond to this feature.
"""

DEDUPE_SYSTEM_PROMPT = f"""\
You help researchers count distinct causal-chain types in a theory-explanation survey.

# Survey context

Participants explain why certain features increase or decrease the probability that a paper discusses an inequality topic. Features are \
listed below. 

Each input row is one causal claim already canonicalized as:
  (direction) antecedent → mechanism → outcome   (when mechanism is stated)
  (direction) antecedent → outcome               (when mechanism is omitted)

Your task: group these per-claim chains into unique_causal_chains — distinct causal-chain \
types in this panel.

# Observable features (13)

Participants may use their own wording; interpret antecedents using these definitions when \
deciding whether two chains refer to the same feature or mechanism.

{OBSERVABLE_FEATURES_BLOCK}


# Merge rules (strict)

Use reasoning and semantic understanding. Read each chain's causal logic: which feature(s), which direction, and (if present) which pathway \
to the outcome. Use the 13 feature definitions above to interpret free-text antecedents and mechanisms.

Two or more chains belong to the SAME unique chain if ALL of the following match:
1. direction (positive / negative / mixed / unclear / not_applicable) — NEVER merge different \
directions, even for the same antecedent.
2. substantive antecedent (same feature or same composite antecedent; e.g. social_science \
and "social science" ARE the same; social science and natural science are NOT).
3. substantive mechanism: both null/unspecified OR the same mechanism idea in meaning (not exact wording).

Merge when two chains express the same substantive causal story, even if wording differs \
(e.g. paraphrases, synonyms, or different granularity of the same idea).

When merging, list every claim_uid assigned to that unique chain.

CRITICAL: Every input claim_uid must appear exactly once across all unique_causal_chains. \
The total number of claim_uids in your output must equal the input n_claims.

Return JSON only matching the schema.
"""

META_MERGE_SYSTEM_PROMPT = f"""\
You merge partial unique-causal-chain lists produced by batched deduplication into one final list.

Apply the SAME semantic merge rules as the main dedupe task (direction, antecedent, mechanism).

Each input partial chain already groups claim_uids from one batch. Merge partial chains when \
they represent the same unique causal-chain type; union their claim_uids.

CRITICAL: Every input claim_uid must appear exactly once in the final unique_causal_chains.

{OBSERVABLE_FEATURES_BLOCK}

Return JSON only matching the schema.
"""


class CanonicalChainRecord(BaseModel):
    claim_uid: str
    participant_name: str
    direction: str
    antecedent_canonical: str
    mechanism_canonical: str | None
    outcome_canonical: str
    chain_display: str


class UniqueCausalChain(BaseModel):
    chain_id: int = Field(description="Sequential id starting at 1.")
    direction: Literal["positive", "negative", "mixed", "unclear", "not_applicable"]
    antecedent_canonical: str
    mechanism_canonical: str | None = None
    outcome_canonical: str
    representative_label: str = Field(
        description="Clearest chain_display for this unique type, including direction prefix."
    )
    claim_uids: list[str] = Field(
        description="All claim_uid values merged into this unique chain."
    )


class DedupeResponse(BaseModel):
    unique_causal_chains: list[UniqueCausalChain]
    merge_notes: str = Field(
        description="Brief notes on ambiguous merge/split decisions; use 'none' if straightforward."
    )


class PartialUniqueChain(BaseModel):
    temp_id: str
    direction: Literal["positive", "negative", "mixed", "unclear", "not_applicable"]
    antecedent_canonical: str
    mechanism_canonical: str | None = None
    outcome_canonical: str
    representative_label: str
    claim_uids: list[str]


def normalize_whitespace(text: str) -> str:
    return " ".join(str(text or "").strip().split())


def normalize_antecedent(text: str) -> str:
    raw = normalize_whitespace(text.replace("_", " "))
    if not raw or raw.lower() == "unspecified":
        return "unspecified"
    return raw


def normalize_mechanism(text: str) -> str | None:
    raw = normalize_whitespace(text.replace("_", " "))
    if not raw or raw.lower() == "unspecified":
        return None
    return raw


def direction_prefix(direction: str) -> str:
    key = normalize_whitespace(direction).lower()
    return DIRECTION_PREFIX.get(key, "(?)")


def normalize_direction(
    direction: str,
) -> Literal["positive", "negative", "mixed", "unclear", "not_applicable"]:
    key = normalize_whitespace(direction).lower()
    if key in ("positive", "negative", "mixed", "unclear", "not_applicable"):
        return key  # type: ignore[return-value]
    return "unclear"


def build_chain_display(
    direction: str,
    antecedent: str,
    mechanism: str | None,
    outcome: str,
) -> str:
    prefix = direction_prefix(direction)
    if mechanism:
        return f"{prefix} {antecedent} → {mechanism} → {outcome}"
    return f"{prefix} {antecedent} → {outcome}"


def canonicalize_claims(df: pd.DataFrame, outcome: str) -> pd.DataFrame:
    outcome_canonical = OUTCOME_CANONICAL.get(outcome, f"{outcome} discussion")
    rows: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        direction = normalize_whitespace(str(row.get("direction", ""))).lower() or "unclear"
        antecedent = normalize_antecedent(str(row.get("antecedent_text", "")))
        mechanism = normalize_mechanism(str(row.get("mechanism_text", "")))
        chain_display = build_chain_display(direction, antecedent, mechanism, outcome_canonical)

        rows.append(
            {
                "claim_uid": row["claim_uid"],
                "participant_name": row["participant_name"],
                "group_label": row.get("group_label", ""),
                "claim_id": row.get("claim_id"),
                "direction": direction,
                "antecedent_canonical": antecedent,
                "mechanism_canonical": mechanism,
                "outcome_canonical": outcome_canonical,
                "chain_display": chain_display,
                "claim_text": row.get("claim_text", ""),
                "needs_human_review": row.get("needs_human_review", False),
            }
        )

    return pd.DataFrame(rows)


def panel_context_block(stage: str, outcome: str, theory_type: str) -> str:
    outcome_canonical = OUTCOME_CANONICAL.get(outcome, outcome)
    return f"""\
Panel:
  stage: {stage}
  outcome: {outcome}
  theory_type: {theory_type}
  fixed outcome phrase in every chain: {outcome_canonical}

Task for this panel: participants explained how observable paper features relate to \
{"discussing racial inequality" if "racial" in outcome else "discussing gender inequality"} \
(main-effects theory: one feature at a time, not interactions).
"""


def build_dedupe_user_prompt(
    canonical_df: pd.DataFrame,
    stage: str,
    outcome: str,
    theory_type: str,
) -> str:
    chains = canonical_df[
        [
            "claim_uid",
            "participant_name",
            "direction",
            "antecedent_canonical",
            "mechanism_canonical",
            "outcome_canonical",
            "chain_display",
        ]
    ].to_dict(orient="records")

    payload = {
        "n_claims": len(chains),
        "chains": chains,
    }
    return (
        panel_context_block(stage, outcome, theory_type)
        + "\n"
        + f"Input: {len(chains)} per-claim canonical chains.\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
        + "\n\nGroup into unique_causal_chains. Every claim_uid must appear exactly once."
    )


def dedupe_unique_chains(
    client: OpenAI,
    model: str,
    canonical_df: pd.DataFrame,
    stage: str,
    outcome: str,
    theory_type: str,
) -> DedupeResponse:
    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": DEDUPE_SYSTEM_PROMPT},
            {"role": "user", "content": build_dedupe_user_prompt(
                canonical_df, stage, outcome, theory_type
            )},
        ],
        response_format=DedupeResponse,
    )
    parsed = completion.choices[0].message.parsed
    if parsed is None:
        raise RuntimeError("LLM returned no parsed dedupe response.")
    return parsed


def build_meta_merge_user_prompt(
    partial_chains: list[PartialUniqueChain],
    n_claims: int,
    stage: str,
    outcome: str,
    theory_type: str,
) -> str:
    payload = {
        "n_claims_total": n_claims,
        "n_partial_chains": len(partial_chains),
        "partial_unique_chains": [c.model_dump() for c in partial_chains],
    }
    return (
        panel_context_block(stage, outcome, theory_type)
        + "\n"
        + f"Merge {len(partial_chains)} partial unique chains (from batched dedupe) into "
        + f"final unique_causal_chains covering all {n_claims} claim_uids.\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
        + "\n\nEvery claim_uid must appear exactly once in the output."
    )


def meta_merge_unique_chains(
    client: OpenAI,
    model: str,
    partial_chains: list[PartialUniqueChain],
    n_claims: int,
    stage: str,
    outcome: str,
    theory_type: str,
) -> DedupeResponse:
    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": META_MERGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_meta_merge_user_prompt(
                    partial_chains, n_claims, stage, outcome, theory_type
                ),
            },
        ],
        response_format=DedupeResponse,
    )
    parsed = completion.choices[0].message.parsed
    if parsed is None:
        raise RuntimeError("LLM returned no parsed meta-merge response.")
    return parsed


def coverage_gaps(
    dedupe: DedupeResponse,
    canonical_df: pd.DataFrame,
) -> tuple[set[str], set[str]]:
    input_uids = set(canonical_df["claim_uid"].astype(str))
    output_uids: set[str] = set()
    for chain in dedupe.unique_causal_chains:
        output_uids.update(str(uid) for uid in chain.claim_uids)
    return input_uids - output_uids, output_uids - input_uids


def append_singleton_chains(
    dedupe: DedupeResponse,
    canonical_df: pd.DataFrame,
    missing: set[str],
) -> DedupeResponse:
    if not missing:
        return dedupe

    uid_rows = canonical_df.set_index("claim_uid")
    next_id = max((c.chain_id for c in dedupe.unique_causal_chains), default=0) + 1
    new_chains = list(dedupe.unique_causal_chains)

    for uid in sorted(missing):
        row = uid_rows.loc[uid]
        new_chains.append(
            UniqueCausalChain(
                chain_id=next_id,
                direction=normalize_direction(str(row["direction"])),
                antecedent_canonical=str(row["antecedent_canonical"]),
                mechanism_canonical=row["mechanism_canonical"]
                if pd.notna(row["mechanism_canonical"]) and row["mechanism_canonical"]
                else None,
                outcome_canonical=str(row["outcome_canonical"]),
                representative_label=str(row["chain_display"]),
                claim_uids=[str(uid)],
            )
        )
        next_id += 1

    notes = dedupe.merge_notes.strip()
    suffix = f"{len(missing)} claim(s) added as singleton chains after LLM omission."
    merge_notes = f"{notes}; {suffix}" if notes and notes.lower() != "none" else suffix
    return DedupeResponse(unique_causal_chains=new_chains, merge_notes=merge_notes)


def renumber_chains(dedupe: DedupeResponse) -> DedupeResponse:
    renumbered = [
        chain.model_copy(update={"chain_id": i})
        for i, chain in enumerate(dedupe.unique_causal_chains, start=1)
    ]
    return DedupeResponse(unique_causal_chains=renumbered, merge_notes=dedupe.merge_notes)


def ensure_full_coverage(
    dedupe: DedupeResponse,
    canonical_df: pd.DataFrame,
) -> DedupeResponse:
    missing, extra = coverage_gaps(dedupe, canonical_df)
    if extra:
        raise ValueError(f"Dedupe returned {len(extra)} unknown claim_uid(s).")
    if missing:
        print(f"  Warning: LLM omitted {len(missing)} claim_uid(s); adding singleton chains.")
        dedupe = append_singleton_chains(dedupe, canonical_df, missing)
    missing_after, _ = coverage_gaps(dedupe, canonical_df)
    if missing_after:
        raise ValueError(f"Dedupe coverage still incomplete: missing={len(missing_after)}")
    return renumber_chains(dedupe)


def chunk_dataframe(df: pd.DataFrame, size: int) -> list[pd.DataFrame]:
    return [df.iloc[i : i + size].copy() for i in range(0, len(df), size)]


def dedupe_panel(
    client: OpenAI,
    model: str,
    canonical_df: pd.DataFrame,
    stage: str,
    outcome: str,
    theory_type: str,
) -> DedupeResponse:
    n_claims = len(canonical_df)
    batches = (
        chunk_dataframe(canonical_df, DEDUPE_BATCH_SIZE)
        if n_claims > DEDUPE_BATCH_SIZE
        else [canonical_df]
    )

    if len(batches) == 1:
        dedupe = dedupe_unique_chains(
            client, model, canonical_df, stage=stage, outcome=outcome, theory_type=theory_type
        )
        return ensure_full_coverage(dedupe, canonical_df)

    partial_models: list[PartialUniqueChain] = []
    for batch_idx, batch_df in enumerate(batches, start=1):
        print(f"  Dedupe batch {batch_idx}/{len(batches)}: {len(batch_df)} claims")
        batch_dedupe = ensure_full_coverage(
            dedupe_unique_chains(
                client,
                model,
                batch_df,
                stage=stage,
                outcome=outcome,
                theory_type=theory_type,
            ),
            batch_df,
        )
        for chain in batch_dedupe.unique_causal_chains:
            partial_models.append(
                PartialUniqueChain(
                    temp_id=f"b{batch_idx}_{chain.chain_id}",
                    direction=chain.direction,
                    antecedent_canonical=chain.antecedent_canonical,
                    mechanism_canonical=chain.mechanism_canonical,
                    outcome_canonical=chain.outcome_canonical,
                    representative_label=chain.representative_label,
                    claim_uids=[str(uid) for uid in chain.claim_uids],
                )
            )

    print(f"  Meta-merge: {len(partial_models)} partial unique chains")
    meta = meta_merge_unique_chains(
        client,
        model,
        partial_models,
        n_claims=n_claims,
        stage=stage,
        outcome=outcome,
        theory_type=theory_type,
    )
    return ensure_full_coverage(meta, canonical_df)


def enrich_unique_chains(
    unique: DedupeResponse,
    canonical_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    uid_to_participant = canonical_df.set_index("claim_uid")["participant_name"].to_dict()
    enriched: list[dict[str, Any]] = []

    for chain in unique.unique_causal_chains:
        participants = {uid_to_participant[uid] for uid in chain.claim_uids if uid in uid_to_participant}
        enriched.append(
            {
                **chain.model_dump(),
                "n_claims": len(chain.claim_uids),
                "n_respondents": len(participants),
            }
        )
    return enriched


def build_claim_assignments(
    unique_chains: list[dict[str, Any]],
    canonical_df: pd.DataFrame,
) -> pd.DataFrame:
    chain_meta = {c["chain_id"]: c for c in unique_chains}
    uid_to_chain: dict[str, int] = {}
    for chain in unique_chains:
        for uid in chain["claim_uids"]:
            uid_to_chain[uid] = chain["chain_id"]

    out = canonical_df.copy()
    out["unique_chain_id"] = out["claim_uid"].map(uid_to_chain)
    out["representative_label"] = out["unique_chain_id"].map(
        lambda cid: chain_meta.get(cid, {}).get("representative_label") if pd.notna(cid) else None
    )
    return out


def all_panels(root: Path = CANONICAL_OUTPUTS_ROOT) -> list[tuple[Path, str, str, str]]:
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
    client: OpenAI,
    model: str = DEFAULT_MODEL,
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
            min_chars=0,
        )
    except ValueError as exc:
        print(f"\n=== {output_dir.name} ===\nSkipped: {exc}")
        return
    print(
        f"\n=== {output_dir.name} ===\n"
        f"Claims for canonicalization: {len(df)} claims, "
        f"{df['participant_name'].nunique()} respondents"
    )

    canonical_df = canonicalize_claims(df, outcome=outcome)

    dedupe = dedupe_panel(
        client,
        model,
        canonical_df,
        stage=stage,
        outcome=outcome,
        theory_type=theory_type,
    )

    unique_chains = enrich_unique_chains(dedupe, canonical_df)
    unique_path = output_dir / "unique_causal_chains.json"
    with unique_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "stage": stage,
                "outcome": outcome,
                "theory_type": theory_type,
                "n_claims_input": len(canonical_df),
                "n_unique_chains": len(unique_chains),
                "model": model,
                "merge_notes": dedupe.merge_notes,
                "unique_causal_chains": unique_chains,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Saved: {unique_path} ({len(unique_chains)} unique chains)")

    canonical_path = output_dir / "claim_canonical_chains.csv"
    canonical_with_assignments = build_claim_assignments(unique_chains, canonical_df)
    canonical_with_assignments.to_csv(canonical_path, index=False)
    print(f"Saved: {canonical_path}")


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")

    input_path = DEFAULT_INPUT_JSON.expanduser().resolve()
    CANONICAL_OUTPUTS_ROOT.mkdir(parents=True, exist_ok=True)
    panels = all_panels()
    client = OpenAI()

    print(f"Input: {input_path}")
    print(f"Panels ({len(panels)}): {[p[0].name for p in panels]}")

    for output_dir, stage, outcome, theory_type in panels:
        run_panel(
            input_path=input_path,
            output_dir=output_dir,
            stage=stage,
            outcome=outcome,
            theory_type=theory_type,
            client=client,
        )


if __name__ == "__main__":
    main()
