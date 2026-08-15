"""
Shared structure coding for survey diagram responses (Q5 and Q13).

Phase-blind: same coding prompt for pre-ML and post-ML diagrams.
Race vs Gender differ only by outcome wording. Callers pass TARGET_COLUMN
and metric column names.

Empty / no-diagram / ill-defined cases are coded as -1 (not 0).
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Literal, Sequence

import pandas as pd
from openai import OpenAI
from pydantic import BaseModel, Field
from tqdm import tqdm

Outcome = Literal["race", "gender"]

# diagram/ -> textual_analysis/ -> survey_april/
INPUT_CSV = Path(__file__).resolve().parents[2] / "All_Participants_All_Questions.csv"

MODEL = "gpt-5.5"
MISSING_CODE = -1
_CLIENT: OpenAI | None = None

_OUTCOME_SPEC: dict[Outcome, dict[str, str]] = {
    "race": {
        "adjective": "racial",
        "y_synonyms": (
            "Y, racial inequality paper, race inequality research, "
            "probability of race inequality research, "
            "likelihood of racial inequality publication, or similar"
        ),
        "latent_examples": (
            "topic fit, topic popularity, interest, awareness, comfort, "
            "salience, academic trend, perceived legitimacy"
        ),
    },
    "gender": {
        "adjective": "gender",
        "y_synonyms": (
            "Y, gender inequality paper, gender inequality research, "
            "probability of gender inequality research, "
            "likelihood of gender inequality publication, "
            "likelihood of gender inequality discussion, or similar"
        ),
        "latent_examples": (
            "topic fit, topic popularity, interest, awareness, comfort, "
            "salience, academic trend, perceived legitimacy, gender salience, "
            "topic interest"
        ),
    },
}


class DiagramMetrics(BaseModel):
    number_of_paths: int = Field(
        description=(
            "Number of distinct causal paths explicitly present in the diagram. "
            "Use -1 if empty, no diagram, or ill-defined."
        )
    )
    maximum_path_length: int = Field(
        description=(
            "Maximum number of arrows in any single causal path. "
            "Use -1 if empty, no diagram, or ill-defined."
        )
    )
    number_of_latent_variables: int = Field(
        description=(
            "Number of unique variables in the diagram that are not among the 13 "
            "project features and are not the outcome Y. "
            "Use -1 if empty, no diagram, or ill-defined."
        )
    )
    brief_reasoning: str = Field(description="Brief explanation of the coding decision.")


def build_system_prompt(outcome: Outcome) -> str:
    spec = _OUTCOME_SPEC[outcome]
    adj = spec["adjective"]
    return f"""
You are a careful research assistant helping code open-ended survey responses from an AI + social science PhD research project.

PROJECT CONTEXT
The project studies human-AI collaboration in scientific theory building. Respondents forecast whether academic papers discuss inequality-related topics, especially racial inequality and gender inequality. The survey asks respondents to construct theories about predictors of whether a paper discusses {adj} inequality.

DATASET CONTEXT
Each paper has 13 observed features. These are the only official observed features in the project:

1. social_science
   Binary. Equals 1 if the paper is published in a social science journal.

2. natural_science
   Binary. Equals 1 if the paper is published in a natural science journal.

3. engineering_and_technology
   Binary. Equals 1 if the paper is published in an engineering or technology journal.

4. num_authors
   Integer. Total number of authors.

5. female
   Continuous in [0,1]. Estimated share of female authors.

6. asian
   Continuous in [0,1]. Estimated share of Asian authors.

7. black
   Continuous in [0,1]. Estimated share of Black authors.

8. hispanic_and_other
   Continuous in [0,1]. Estimated share of Hispanic and other-race authors.

9. white
   Continuous in [0,1]. Estimated share of White authors.

10. authors_race_diversity_score
    Continuous. Racial diversity within the co-author team, measured using Shannon entropy.

11. country_race_diversity_score
    Continuous. Average racial diversity of the authors' inferred countries of birth, measured using Shannon entropy.

12. news_inequality_mentions_3_years
    Continuous. Average percentage of news articles mentioning inequality during the three years before publication.

13. paper_inequality_mentions_3_years
    Continuous. Average percentage of academic papers mentioning inequality during the three years before publication.

DIAGRAM RESPONSE
Respondents were instructed to provide a diagram with arrows, expressed in text form, to represent their theory, where an arrow indicates a causal relationship between two variables. They may use ‘→’ and indicate effect signs ('+' or '–').

The outcome is whether a paper discusses {adj} inequality. Respondents may refer to the outcome as:
{spec["y_synonyms"]}.

YOUR CODING TASK
For each respondent’s diagram response, code exactly three quantities:

1. number_of_paths
   Count the number of distinct causal paths explicitly written by the respondent.
   A path is usually one causal chain separated by arrows, often appearing as one line or one sentence.
   Example:
   A → B → Y
   C → Y
   counts as 2 paths.

2. maximum_path_length
   Count the maximum number of causal arrows in any one path.
   Example:
   A → B → C → Y has length 3.
   A → Y has length 1.

3. number_of_latent_variables
   Count the number of unique variables/concepts in the diagram that are NOT one of the 13 official observed features and are NOT the outcome Y.
   These include mediators, mechanisms, latent constructs, invented concepts, or renamed theoretical constructs.
   Examples: {spec["latent_examples"]}.
   Do not count:
   - Y or any synonym of the {adj} inequality outcome
   - the 13 official observed features, even if written with minor wording variations
   - signs such as + or -
   - generic labels such as X1, X2, mediator, Med1, if they merely label a substantive variable already named nearby

MISSING / ILL-DEFINED CASES (IMPORTANT)
Set ALL THREE metrics to -1 when ANY of the following hold:
- the response is empty or whitespace only
- the respondent says they have no diagram, refuse to answer, or only write non-diagram commentary
- there is no usable causal structure (no arrows / no clear causal links that can be coded as paths)
- the response is too ambiguous, contradictory, or incomplete to identify distinct paths reliably

Do NOT use 0 for these cases. Use -1.
Only use non-negative integers when a codeable diagram/causal structure is present.
A valid trivial diagram with a single arrow (e.g., A → Y) should be coded as paths=1, max length=1, latents accordingly (not -1).

IMPORTANT NORMALIZATION RULES
Treat common natural-language variants as equivalent to the official features:
- social science, social sciences, social_science, or similar → social_science
- natural science, natural sciences, natural_science, or similar → natural_science
- engineering, technology, computer science, engineering and technology, or similar → engineering_and_technology
- number of authors, team size, or similar → num_authors
- female authors, share of female authors, female_score, majority female author team, or similar → female
- Asian authors, majority Asian author team, or similar → asian
- Black authors, or similar → black
- Hispanic authors, other race authors → hispanic_and_other
- White authors, majority white team, or similar → white
- author race diversity, author racial diversity, racially diverse author team, or similar → authors_race_diversity_score
- country race diversity, racially diverse countries, country diversity, or similar → country_race_diversity_score
- news inequality mentions, media attention, news attention to inequality, societal attention, public attention, or similar → news_inequality_mentions_3_years
- paper inequality mentions, prior publications, academic attention, recent academic attention to inequality, publication trend, or similar → paper_inequality_mentions_3_years

Be conservative but thoughtful. The responses are unstructured and may contain prose, arrows, parentheses, signs, line breaks, and inconsistent names. Use careful reasoning to infer the diagram structure.

Return only valid JSON matching the requested schema.
""".strip()


def build_user_prompt(response_text: str) -> str:
    return f"""
Please analyze the following diagram response.

Respondent response:
\"\"\"
{response_text}
\"\"\"

Return:
1. number_of_paths
2. maximum_path_length
3. number_of_latent_variables
4. brief_reasoning

If empty / no diagram / ill-defined, set the three metrics to -1.
""".strip()


def _client() -> OpenAI:
    global _CLIENT
    if _CLIENT is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        _CLIENT = OpenAI(api_key=api_key)
    return _CLIENT


def _missing_metrics(reason: str) -> DiagramMetrics:
    return DiagramMetrics(
        number_of_paths=MISSING_CODE,
        maximum_path_length=MISSING_CODE,
        number_of_latent_variables=MISSING_CODE,
        brief_reasoning=reason,
    )


def analyze_response(
    response_text: str,
    outcome: Outcome,
    max_retries: int = 3,
) -> DiagramMetrics:
    if not isinstance(response_text, str) or not response_text.strip():
        return _missing_metrics("Empty or missing response; coded as -1.")

    system_prompt = build_system_prompt(outcome)
    user_prompt = build_user_prompt(response_text)

    for attempt in range(max_retries):
        try:
            completion = _client().beta.chat.completions.parse(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=DiagramMetrics,
            )
            message = completion.choices[0].message
            if message.parsed is None:
                raise ValueError(f"Empty parsed response (refusal={message.refusal!r})")
            return message.parsed

        except Exception as e:
            if attempt == max_retries - 1:
                return _missing_metrics(f"API error after retries: {repr(e)}")
            time.sleep(2 ** attempt)

    return _missing_metrics("API error: exhausted retries without result.")


def _is_blank(val: object) -> bool:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return True
    try:
        if pd.isna(val):
            return True
    except (TypeError, ValueError):
        pass
    return str(val).strip() == "" or str(val).strip().lower() in {"nan", "<na>", "none"}


def _row_metrics_done(df: pd.DataFrame, row: int, metric_columns: Sequence[str]) -> bool:
    """True iff all four metric fields are present. -1 counts as coded (missing diagram)."""
    for col in metric_columns:
        if _is_blank(df.at[row, col]):
            return False
    return True


def _write_csv_atomic(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def _ensure_metric_columns_after_target(
    df: pd.DataFrame,
    target_column: str,
    metric_columns: Sequence[str],
) -> pd.DataFrame:
    """Insert / move metric columns immediately after target_column with correct dtypes."""
    if target_column not in df.columns:
        raise ValueError(f"Column not found: {target_column}")

    numeric_columns = list(metric_columns[:3])
    reason_column = metric_columns[3]
    n = len(df)

    for col in numeric_columns:
        if col not in df.columns:
            df[col] = pd.Series([pd.NA] * n, dtype="Int64")
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    if reason_column not in df.columns:
        df[reason_column] = pd.Series([pd.NA] * n, dtype="string")
    else:
        s = df[reason_column]
        if pd.api.types.is_numeric_dtype(s):
            df[reason_column] = pd.Series(
                [pd.NA if (pd.isna(v) or str(v).strip() == "") else str(v) for v in s],
                dtype="string",
            )
        else:
            df[reason_column] = s.astype("string")

    remaining = [c for c in df.columns if c not in metric_columns]
    insert_at = remaining.index(target_column) + 1
    new_order = remaining[:insert_at] + list(metric_columns) + remaining[insert_at:]
    out = df.loc[:, new_order].copy()

    for col in numeric_columns:
        out[col] = out[col].astype("Int64")
    out[reason_column] = out[reason_column].astype("string")
    return out


def _assign_row_metrics(
    df: pd.DataFrame,
    row: int,
    result: DiagramMetrics,
    metric_columns: Sequence[str],
) -> None:
    df.at[row, metric_columns[0]] = int(result.number_of_paths)
    df.at[row, metric_columns[1]] = int(result.maximum_path_length)
    df.at[row, metric_columns[2]] = int(result.number_of_latent_variables)
    df.at[row, metric_columns[3]] = str(result.brief_reasoning)


def run_diagram_coding(
    *,
    target_column: str,
    metric_columns: Sequence[str],
    outcome: Outcome,
    progress_desc: str,
    input_csv: Path | None = None,
) -> None:
    if len(metric_columns) != 4:
        raise ValueError("metric_columns must be exactly 4 names (3 numeric + reasoning).")

    csv_path = input_csv or INPUT_CSV
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    if target_column not in df.columns:
        raise ValueError(
            f"Column not found: {target_column}. Available columns include: {list(df.columns)[:20]}"
        )

    df = _ensure_metric_columns_after_target(df, target_column, metric_columns)
    _write_csv_atomic(df, csv_path)

    incomplete_rows = [
        i for i in range(len(df)) if not _row_metrics_done(df, i, metric_columns)
    ]
    if not incomplete_rows:
        print(
            f"All rows already have metrics for {target_column!r}. "
            f"Metric columns ensured immediately after the target column."
        )
        return

    print(
        f"Scanning {len(df)} rows; {len(incomplete_rows)} row(s) missing metrics "
        f"for {target_column!r}."
    )
    print(f"Metric columns placed immediately after {target_column!r}.")
    print(f"dtypes: { {c: str(df[c].dtype) for c in metric_columns} }")

    with tqdm(range(len(df)), desc=progress_desc, unit="row") as pbar:
        for i in pbar:
            if _row_metrics_done(df, i, metric_columns):
                pbar.set_postfix(row=i + 1, status="skip")
                continue

            pbar.set_postfix(row=i + 1, status="API")
            text = df.at[i, target_column]
            if not isinstance(text, str):
                text = "" if pd.isna(text) else str(text)

            result = analyze_response(text, outcome=outcome)
            _assign_row_metrics(df, i, result, metric_columns)
            _write_csv_atomic(df, csv_path)

    print(f"Done. Updated metrics in-place in: {csv_path}")
