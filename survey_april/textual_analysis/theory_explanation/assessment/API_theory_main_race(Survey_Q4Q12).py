import os
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from openai import BadRequestError, OpenAI
from pydantic import BaseModel

from csv_score_io import ensure_columns_after, locked_csv, locked_csv_read

# =====================
# FILE CONFIG
# =====================
INPUT_CSV = Path(__file__).resolve().parents[3] / "All_Participants_All_Questions.csv"

NAME_COLUMN = "What is your full name?"
Q4_TARGET_COLUMN = "Q Race.4 pre-ML theory (main effects)"
Q12_LLM_REFINED_COLUMN = "Q Race.12 LLM_refined post-ML theory (main effects)"
Q12_TARGET_PREFIX = "Q Race.12 LLM_refined"

# =====================
# OUTPUT COLUMNS
# =====================
Q4_METRIC_COLUMNS = [
    "Q Race.4 Clarity and Coherence",
    "Q Race.4 Causal Reasoning",
    "Q Race.4 Theoretical Depth",
    "Q Race.4 Creativity",
    "Q Race.4 Persuasiveness",
    "Q Race.4 Mechanisms",
    "Q Race.4 Brief Reasoning",
]

Q12_METRIC_COLUMNS = [
    "Q Race.12 Updated Theory Clarity and Coherence",
    "Q Race.12 Updated Theory Causal Reasoning",
    "Q Race.12 Updated Theory Theoretical Depth",
    "Q Race.12 Updated Theory Creativity",
    "Q Race.12 Updated Theory Persuasiveness",
    "Q Race.12 Updated Theory Mechanisms",
    "Q Race.12 Updated Theory Brief Reasoning",
]

WRITE_COLUMNS = Q4_METRIC_COLUMNS + Q12_METRIC_COLUMNS

# =====================
# MODEL
# =====================
MODEL = "gpt-5.5"
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# =====================
# RESPONSE SCHEMA
# =====================
class TheoryQualityMetrics(BaseModel):
    clarity_and_coherence: int
    causal_reasoning: int
    theoretical_depth: int
    creativity: int
    persuasiveness: int
    mechanisms: list[str]
    brief_reasoning: str


# =====================
# SYSTEM PROMPT
# =====================
SYSTEM_PROMPT = """
You are an expert in social science research and theory evaluation.
Use your strongest analytical judgment and highest level of social-scientific reasoning. Be rigorous, discerning, and careful.

PROJECT CONTEXT
This study examines theory building for predicting mentions of racial inequality in academic papers.

The outcome is whether an academic paper discusses racial inequality. 

Respondents first selected a few features they believed were most important, ranked them, and indicated whether each feature was positively or negatively associated with the outcome. They were then asked to provide a theoretical explanation for these choices.

The study includes 13 available features:

1. social_science
   Equals 1 if the paper is published in a journal whose scope lies within the social sciences, and 0 otherwise.

2. natural_science
   Equals 1 if the paper is published in a journal whose scope lies within the natural sciences, and 0 otherwise.

3. engineering_and_technology
   Equals 1 if the paper is published in a journal whose scope lies within engineering and technology, and 0 otherwise.

4. num_authors
   The total number of authors of the paper.

5. female
   The estimated share of female authors on the author team.

6. asian
   The estimated share of Asian authors on the author team.

7. black
   The estimated share of Black authors on the author team.

8. hispanic_and_other
   The estimated share of Hispanic and other-race authors on the author team.

9. white
   The estimated share of White authors on the author team.

10. authors_race_diversity_score
   The racial diversity within the co-author team, measured by Shannon entropy.

11. country_race_diversity_score
   The average racial diversity of the authors’ inferred countries of birth, measured by Shannon entropy.

12. news_inequality_mentions_3_years
   The average percentage of news articles mentioning inequality over the three years preceding the paper’s publication year.

13. paper_inequality_mentions_3_years
   The average percentage of academic papers mentioning inequality over the three years preceding the paper’s publication year.

TASK OVERVIEW
You will be given a theoretical explanation about why certain variables predict whether an academic paper discusses racial inequality.

Your task is to evaluate the QUALITY of this theoretical explanation.

-------------------------------------
EVALUATION DIMENSIONS (1–10 scale)
-------------------------------------

For each dimension, assign a score from 1 (very poor) to 10 (excellent).

1. Clarity and Coherence
Is the explanation clearly written, well-structured, and logically consistent, without ambiguity or internal contradictions?

2. Causal Reasoning
Does the explanation articulate plausible causal mechanisms linking the predictors to the outcome?

3. Theoretical Depth
Does the explanation go beyond surface-level statements and engage with meaningful underlying concepts or mechanisms?

4. Creativity
Does the explanation demonstrate creative or original thinking, such as offering novel perspectives, non-obvious connections, or insightful interpretations?

5. Persuasiveness
Does the explanation provide a convincing theoretical account of why the predictors should be related to the outcome?
-------------------------------------
SCORING GUIDELINES
-------------------------------------

1–2 = poor
3–4 = weak
5–6 = moderate
7–8 = strong
9–10 = excellent

-------------------------------------
MECHANISM IDENTIFICATION
-------------------------------------

Identify the main theoretical mechanisms used in the explanation.

Select all that apply from the list below. Do not select more than 4 unless clearly necessary.

Predefined mechanisms:

- disciplinary fit
- topic relevance
- attention and salience
- academic trend
- identity and lived experience
- demographic representation
- diversity exposure
- motivation or intrinsic interest
- legitimacy or role appropriateness
- data or feasibility constraints

If the explanation does not fit any of the above, you may create ONE new mechanism using a short descriptive phrase.

-------------------------------------
OUTPUT FORMAT
-------------------------------------

Return ONLY JSON:

{
  "clarity_and_coherence": int,
  "causal_reasoning": int,
  "theoretical_depth": int,
  "creativity": int,
  "persuasiveness": int,
  "mechanisms": [list of strings],
  "brief_reasoning": "string"
}

The brief_reasoning should be concise, no more than 5 sentences, and should justify the scores and mechanisms.
"""


# =====================
# ANALYZE FUNCTIONS
# =====================
def analyze_q4_response(q4_text, retries=3):
    """
    Evaluate the initial theoretical explanation.
    """
    if not isinstance(q4_text, str) or not q4_text.strip():
        return TheoryQualityMetrics(
            clarity_and_coherence=0,
            causal_reasoning=0,
            theoretical_depth=0,
            creativity=0,
            persuasiveness=0,
            mechanisms=[],
            brief_reasoning="Empty or missing response.",
        )

    user_prompt = f"""
Evaluate the following theoretical explanation.

\"\"\"
{q4_text}
\"\"\"
"""

    return _call_openai(user_prompt, retries=retries)


def analyze_post_ml_theory(theory_text, retries=3):
    """Evaluate the integrated post-ML theoretical explanation."""
    if not isinstance(theory_text, str) or not theory_text.strip():
        return TheoryQualityMetrics(
            clarity_and_coherence=0,
            causal_reasoning=0,
            theoretical_depth=0,
            creativity=0,
            persuasiveness=0,
            mechanisms=[],
            brief_reasoning="Empty or missing response.",
        )

    user_prompt = f"""
Evaluate the following post-ML theoretical explanation.

\"\"\"
{theory_text}
\"\"\"
"""

    return _call_openai(user_prompt, retries=retries)


def _call_openai(user_prompt, retries=3):
    for attempt in range(retries):
        try:
            completion = client.beta.chat.completions.parse(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=TheoryQualityMetrics,
            )
            return completion.choices[0].message.parsed

        except BadRequestError as e:
            raise RuntimeError(f"OpenAI request rejected: {e}") from e

        except Exception as e:
            if attempt == retries - 1:
                return TheoryQualityMetrics(
                    clarity_and_coherence=-1,
                    causal_reasoning=-1,
                    theoretical_depth=-1,
                    creativity=-1,
                    persuasiveness=-1,
                    mechanisms=[],
                    brief_reasoning=f"API error after retries: {repr(e)}",
                )
            time.sleep(2 ** attempt)


# =====================
# CHECK IF DONE
# =====================
def row_done(df, i, metric_columns):
    val = df.at[i, metric_columns[0]]
    if pd.isna(val):
        return False
    try:
        return float(val) >= 0
    except (TypeError, ValueError):
        return False


def _is_text_metric_column(col: str) -> bool:
    return col.endswith("Mechanisms") or col.endswith("Brief Reasoning")


def prepare_metric_dtypes(df, metric_columns):
    """Coerce only this script's score columns for numeric comparisons."""
    for col in metric_columns:
        if col not in df.columns:
            continue
        if _is_text_metric_column(col):
            df[col] = df[col].astype("object")
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _normalize_header(text):
    return " ".join(str(text).split()).strip().strip('"')


def resolve_column_name(df, exact_name, prefix):
    if exact_name in df.columns:
        return exact_name

    norm_exact = _normalize_header(exact_name)
    norm_prefix = _normalize_header(prefix)

    for col in df.columns:
        if _normalize_header(col) == norm_exact:
            return col

    for col in df.columns:
        if _normalize_header(col).startswith(norm_prefix):
            return col

    raise ValueError(f"Column not found: {exact_name}")


def write_result(df, i, metric_columns, result):
    df.at[i, metric_columns[0]] = result.clarity_and_coherence
    df.at[i, metric_columns[1]] = result.causal_reasoning
    df.at[i, metric_columns[2]] = result.theoretical_depth
    df.at[i, metric_columns[3]] = result.creativity
    df.at[i, metric_columns[4]] = result.persuasiveness
    df.at[i, metric_columns[5]] = ", ".join(result.mechanisms)
    df.at[i, metric_columns[6]] = result.brief_reasoning


# =====================
# MAIN
# =====================
def main():
    with locked_csv(INPUT_CSV) as df:
        if NAME_COLUMN not in df.columns:
            raise ValueError(f"Column not found: {NAME_COLUMN}")
        if Q4_TARGET_COLUMN not in df.columns:
            raise ValueError(f"Column not found: {Q4_TARGET_COLUMN}")
        q12_llm_col = resolve_column_name(df, Q12_LLM_REFINED_COLUMN, Q12_TARGET_PREFIX)
        ensure_columns_after(df, Q4_TARGET_COLUMN, Q4_METRIC_COLUMNS)
        ensure_columns_after(df, q12_llm_col, Q12_METRIC_COLUMNS)
        prepare_metric_dtypes(df, WRITE_COLUMNS)
        row_count = len(df)

    for i in tqdm(
        range(row_count),
        desc="Q Race.4 and Q Race.12 LLM-refined theory quality (API)",
        unit="row",
        mininterval=0.3,
    ):
        with locked_csv_read(INPUT_CSV) as df:
            q12_llm_col = resolve_column_name(df, Q12_LLM_REFINED_COLUMN, Q12_TARGET_PREFIX)
            prepare_metric_dtypes(df, WRITE_COLUMNS)

            q4_done = row_done(df, i, Q4_METRIC_COLUMNS)
            q12_done = row_done(df, i, Q12_METRIC_COLUMNS)
            if q4_done and q12_done:
                continue

            q4_text = None if q4_done else df.at[i, Q4_TARGET_COLUMN]
            q12_text = None if q12_done else df.at[i, q12_llm_col]

        q4_result = analyze_q4_response(q4_text) if q4_text is not None else None
        q12_result = analyze_post_ml_theory(q12_text) if q12_text is not None else None

        with locked_csv(INPUT_CSV) as df:
            prepare_metric_dtypes(df, WRITE_COLUMNS)
            if q4_result is not None:
                write_result(df, i, Q4_METRIC_COLUMNS, q4_result)
            if q12_result is not None:
                write_result(df, i, Q12_METRIC_COLUMNS, q12_result)

    print("Done!")


if __name__ == "__main__":
    main()