#!/usr/bin/env python3
"""
Extract claim-level units from theory explanation responses via OpenAI.

Reads All_Participants_All_Questions.csv (read-only) and writes a nested JSON file:

    participant_name -> stage -> outcome_domain -> theory_type -> [claims]

Participant-level fields:
    group (0/1/2), group_label (student/senior/GenAI)

Post-ML cells use the LLM_refined theory columns (same as embedding pipeline).

Example:
    export OPENAI_API_KEY="..."
    python extract_claims_wide_dataset.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from openai import OpenAI
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_CSV = "/Users/keli/Desktop/iCloudDesktop/PhD/Project_Code/Human-AI/survey_april/All_Participants_All_Questions.csv"
DEFAULT_OUTPUT_JSON = SCRIPT_DIR / "claim_level_dataset.json"

NAME_COLUMN = "What is your full name?"
TYPE_COLUMN = "student_0, senior_1, genAI_2"
DEFAULT_MODEL = "gpt-5.5"

TYPE_MAP = {
    "0": "student",
    "1": "senior",
    "2": "GenAI",
}

OUTCOME_DOMAIN_LABELS = {
    "race": "racial inequality",
    "gender": "gender inequality",
}

THEORY_TYPE_LABELS = {
    "main-effects": "main effects",
    "soi": "second-order interactions",
}

# (stage, task, theory_type_key, csv_column)
EXTRACTION_JOBS: list[tuple[str, str, str, str]] = [
    ("pre-ML", "race", "main-effects", "Q Race.4 pre-ML theory (main effects)"),
    ("pre-ML", "race", "soi", "Q Race.10 pre-ML theory (SOI)"),
    ("post-ML", "race", "main-effects", "Q Race.12 LLM_refined post-ML theory (main effects)"),
    ("post-ML", "race", "soi", "Q Race.15 LLM_refined post-ML theory (SOI)"),
    ("pre-ML", "gender", "main-effects", "Q Gender.4 pre-ML theory (main effects)"),
    ("pre-ML", "gender", "soi", "Q Gender.10 pre-ML theory (SOI)"),
    ("post-ML", "gender", "main-effects", "Q Gender.12 LLM_refined post-ML theory (main effects)"),
    ("post-ML", "gender", "soi", "Q Gender.15 LLM_refined post-ML theory (SOI)"),
]

CLAIM_EXTRACTION_SYSTEM_PROMPT = """\
# Role

You are a research assistant helping analyze participants' theory responses.

Your task is to segment each response into claim-level theoretical units and faithfully extract the structure of each claim.

Your role is extraction only. Do not evaluate, summarize, improve, or extend the participant's theory.

Preserve the participant's intended logic exactly, even if it is vague, incomplete, repetitive, awkwardly phrased, or factually incorrect.

Never invent content that is not explicitly stated or clearly implied.

---

# Definition of a claim

A claim is the smallest self-contained theoretical assertion linking an explanatory factor, mechanism, condition, moderator, or interaction to the likelihood that an academic paper discusses the specified inequality outcome named in the user message.

Claims may express:

* causal relationships
* associational relationships
* conditional relationships
* moderation
* interaction effects
* or other theoretical relationships

A claim does not need to contain a fully specified mechanism to be extracted.

---

# Decision process

Complete the following steps in order for every response.

## Step 1. Identify claims

Identify every distinct claim-level theoretical unit.

Create a new claim when the participant introduces a substantively different:

* antecedent
* mechanism
* moderator
* outcome domain
* causal direction

## Step 2. Locate supporting evidence

For every claim, identify the shortest continuous span from the participant's response that directly supports that claim.

Every extracted claim must have supporting text copied verbatim from the response. If no supporting text can be identified, do not extract the claim.

## Step 3. Extract claim attributes

For each claim:

* determine whether it expresses a causal or explanatory relationship;
* identify the mechanism, if present;
* determine the direction of the relationship relative to the outcome domain in the user message.

If the participant links one antecedent to multiple distinct mechanisms, extract a separate claim for each mechanism.

If no mechanism is provided, preserve the claim and record the mechanism as "unspecified".

## Step 4. Determine review status

Mark a claim for human review only when there is genuine uncertainty.

---

# Observable features

Participants may refer to the following observable characteristics of a paper. Participants may rephrase these features in their own words. Use this list only to interpret participant wording; do not add features they did not mention.

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

hispanic_and_other: The proportion of Hispanic authors and authors from racial groups other than White, Black, or Asian.

authors_race_diversity_score: The racial diversity of the author team. Phrases such as "racially diverse teams," "mixed-race author teams," or "diverse coauthors" may correspond to this feature.

country_race_diversity_score: The racial diversity of the authors' countries of birth. References to diversity across authors' countries of origin or international diversity may correspond to this feature.

## Discourse context

news_inequality_mentions_3_years: The level of media attention to inequality during the three years before publication. Phrases such as "media attention," "news coverage," "public discourse," or "news discussion of inequality" may correspond to this feature.

paper_inequality_mentions_3_years: The level of academic attention to inequality during the three years before publication. Phrases such as "prior research," "recent papers," "scholarly attention," or "academic discussion of inequality" may correspond to this feature.

---

# Output constraints

Return only one JSON object.

Do not include markdown fences, comments, explanations, or any text outside the JSON object.
"""

CLAIM_EXTRACTION_USER_PROMPT = """\
# Context

* Stage: {stage}
  Allowed values: "pre-ML" or "post-ML".
* Outcome domain: {outcome_domain}
  Allowed values: "gender inequality" or "racial inequality".
* Theory type: {theory_type}
  Allowed values: "main effects" or "second-order interactions".

Interpret direction relative to the outcome domain above.

# Response

{response_text}

# Task

Follow the system instructions and return only one JSON object.

## JSON shape example

{{
  "claims": [
    {{
      "claim_id": 1,
      "supporting_text": "...",
      "claim_text": "...",
      "is_causal_claim": true or false,
      "antecedent_text": "...",
      "mechanism_text": "...",
      "direction": "positive" or "negative" or "mixed" or "unclear" or "not_applicable",
      "needs_human_review": true or false,
      "review_reason": "..."
    }}
  ]
}}

If the response contains no usable theoretical content, return:

{{"claims": []}}

## Field definitions

* claim_id: sequential integer starting at 1
* supporting_text: shortest verbatim textual span from the response that supports the claim
* claim_text: concise paraphrase preserving the participant's intended meaning
* is_causal_claim: true if the participant proposes or implies an explanatory relationship; false only if the statement is purely descriptive or too vague to represent a theoretical relationship
* antecedent_text: explanatory factor, predictor, condition, moderator, or interaction in the participant's wording, or "unspecified" if absent
* mechanism_text: mechanism, if present, or "unspecified" if absent
* direction: one of "positive" (the antecedent is positively related to the outcome domain), "negative" (the antecedent is negatively related to the outcome domain), "mixed" (the antecedent is positively and negatively related to the outcome domain), "unclear" (the direction of the relationship is not clear), or "not_applicable" (the antecedent is not related to the outcome domain)
* needs_human_review: true only if there is genuine uncertainty about segmentation, causal interpretation, antecedent extraction, mechanism extraction, or direction
* review_reason: briefly explain the uncertainty if needs_human_review is true; otherwise "none"

"""


class ClaimRecord(BaseModel):
    claim_id: int
    supporting_text: str
    claim_text: str
    is_causal_claim: bool
    antecedent_text: str
    mechanism_text: str
    direction: Literal[
        "positive",
        "negative",
        "mixed",
        "unclear",
        "not_applicable",
    ]
    needs_human_review: bool
    review_reason: str


class ClaimExtractionResponse(BaseModel):
    claims: list[ClaimRecord] = Field(default_factory=list)


def clean_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return " ".join(str(value).replace("\r", " ").replace("\n", " ").split()).strip()


def clean_participant_display_name(name: str) -> str:
    s = str(name).strip()
    if s.endswith("(1)"):
        return s[:-3]
    return s


def parse_group_code(raw: Any) -> int | None:
    text = clean_text(raw)
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def find_column(df: pd.DataFrame, target: str) -> str:
    if target in df.columns:
        return target
    norm_target = " ".join(target.split()).lower()
    for col in df.columns:
        if " ".join(str(col).split()).lower() == norm_target:
            return col
    raise KeyError(f"Could not find column: {target}")


def ensure_participant_node(dataset: dict[str, Any], name: str, group: int | None) -> dict[str, Any]:
    if name not in dataset:
        dataset[name] = {
            "group": group,
            "group_label": TYPE_MAP.get(str(group), "unknown") if group is not None else "unknown",
        }
    return dataset[name]


def ensure_nested_list(
    participant_node: dict[str, Any],
    stage: str,
    outcome_domain: str,
    theory_type: str,
) -> list[dict[str, Any]]:
    stage_node = participant_node.setdefault(stage, {})
    outcome_node = stage_node.setdefault(outcome_domain, {})
    claims = outcome_node.setdefault(theory_type, [])
    if not isinstance(claims, list):
        raise TypeError(
            f"Expected claims list for {stage}/{outcome_domain}/{theory_type}, "
            f"found {type(claims).__name__}"
        )
    return claims


def cell_is_done(
    dataset: dict[str, Any],
    name: str,
    stage: str,
    outcome_domain: str,
    theory_type: str,
) -> bool:
    participant_node = dataset.get(name)
    if not participant_node:
        return False
    try:
        claims = participant_node[stage][outcome_domain][theory_type]
    except KeyError:
        return False
    return isinstance(claims, list)


@retry(wait=wait_exponential(multiplier=2, min=2, max=60), stop=stop_after_attempt(5))
def extract_claims(
    client: OpenAI,
    *,
    model: str,
    stage: str,
    outcome_domain: str,
    theory_type: str,
    response_text: str,
) -> ClaimExtractionResponse:
    user_prompt = CLAIM_EXTRACTION_USER_PROMPT.format(
        stage=stage,
        outcome_domain=outcome_domain,
        theory_type=theory_type,
        response_text=response_text,
    )
    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": CLAIM_EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format=ClaimExtractionResponse,
    )
    parsed = completion.choices[0].message.parsed
    if parsed is None:
        raise RuntimeError("Model returned no parsed claims payload.")
    return parsed


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    tmp_path.replace(path)


def load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")

    input_csv = DEFAULT_INPUT_CSV
    output_json = DEFAULT_OUTPUT_JSON
    model = DEFAULT_MODEL

    print(f"Reading CSV: {input_csv}")
    df = pd.read_csv(input_csv, dtype=str, keep_default_na=False)
    print(f"Participants: {len(df)}")

    name_col = find_column(df, NAME_COLUMN)
    type_col = find_column(df, TYPE_COLUMN)
    column_map = {target: find_column(df, target) for _, _, _, target in EXTRACTION_JOBS}

    dataset: dict[str, Any] = load_json_if_exists(output_json)

    client = OpenAI()

    total_cells = len(df) * len(EXTRACTION_JOBS)
    progress = tqdm(total=total_cells, desc="Claim extraction", unit="cell")

    for row_idx, row in df.iterrows():
        name = clean_participant_display_name(clean_text(row[name_col]))
        if not name:
            progress.update(len(EXTRACTION_JOBS))
            continue

        group = parse_group_code(row[type_col])
        participant_node = ensure_participant_node(dataset, name, group)

        for stage, task, theory_key, target in EXTRACTION_JOBS:
            outcome_domain = OUTCOME_DOMAIN_LABELS[task]
            theory_type = THEORY_TYPE_LABELS[theory_key]
            source_col = column_map[target]

            if cell_is_done(dataset, name, stage, outcome_domain, theory_type):
                progress.update(1)
                continue

            response_text = clean_text(row[source_col])
            claims_list = ensure_nested_list(
                participant_node, stage, outcome_domain, theory_type
            )
            claims_list.clear()

            if not response_text:
                save_json(output_json, dataset)
                progress.update(1)
                continue

            try:
                result = extract_claims(
                    client,
                    model=model,
                    stage=stage,
                    outcome_domain=outcome_domain,
                    theory_type=theory_type,
                    response_text=response_text,
                )
                claims_list.extend(claim.model_dump() for claim in result.claims)
            except Exception as exc:
                save_json(output_json, dataset)

            save_json(output_json, dataset)
            progress.update(1)

    progress.close()
    save_json(output_json, dataset)

    print(f"\nSaved claims dataset: {output_json}")

if __name__ == "__main__":
    main()
