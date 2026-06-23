import os
import time
import pandas as pd
from tqdm import tqdm
from openai import BadRequestError, OpenAI
from pydantic import BaseModel

# =====================
# FILE CONFIG
# =====================
INPUT_CSV = "All_Participants_All_Questions.csv"

NAME_COLUMN = "What is your full name?"
Q4_TARGET_COLUMN = "Q Race.4 pre-ML theory (main effects)"
Q12_TARGET_COLUMN = "Q Race.12 post-ML theory (main effects)"

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


def analyze_updated_theory(q4_text, q12_text, retries=3):
    q4_clean = q4_text if isinstance(q4_text, str) and q4_text.strip() else ""
    q12_clean = q12_text if isinstance(q12_text, str) and q12_text.strip() else ""

    if not q4_clean and not q12_clean:
        return TheoryQualityMetrics(
            clarity_and_coherence=0,
            causal_reasoning=0,
            theoretical_depth=0,
            creativity=0,
            persuasiveness=0,
            mechanisms=[],
            brief_reasoning="Empty or missing initial and updated responses.",
        )

    user_prompt = f"""
    Evaluate the respondent's updated theoretical explanation.

    CONTEXT:
    The respondent first provided an initial theory before seeing machine-learning evidence.
    Then, after reviewing machine-learning evidence about the main effects predicting whether a paper discusses racial inequality, the respondent was asked to refine or update their theory.

    Important:
    - If the updated response only modifies part of the initial theory, treat the final theory as the initial theory plus the stated update.
    - If the respondent says they did not update their theory, evaluate the initial theory together with their justification for not updating.
    - Do NOT reward or penalize the respondent merely for agreeing with the machine-learning evidence.
    - Focus on whether the resulting updated theory is clear, causally reasoned, theoretically deep, creative, and persuasive.

    INITIAL THEORETICAL EXPLANATION:
    \"\"\"
    {q4_clean}
    \"\"\"

    UPDATED THEORETICAL EXPLANATION AFTER MACHINE-LEARNING EVIDENCE:
    \"\"\"
    {q12_clean}
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
    """Empty evaluation columns are often inferred as float64; text cols must be object."""
    for col in metric_columns:
        if col not in df.columns:
            continue
        if _is_text_metric_column(col):
            df[col] = df[col].astype("object")
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def ensure_metric_columns_after(df, anchor_column, metric_columns):
    """
    Ensure metric columns exist and are placed immediately after anchor_column.
    Missing columns are inserted in metric_columns order.
    """
    if anchor_column not in df.columns:
        raise ValueError(f"Column not found: {anchor_column}")

    insert_at = df.columns.get_loc(anchor_column) + 1
    for col in metric_columns:
        if col in df.columns:
            continue
        df.insert(insert_at, col, pd.NA)
        insert_at += 1

    return prepare_metric_dtypes(df, metric_columns)


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
    df = pd.read_csv(INPUT_CSV)

    if NAME_COLUMN not in df.columns:
        raise ValueError(f"Column not found: {NAME_COLUMN}")

    if Q4_TARGET_COLUMN not in df.columns:
        raise ValueError(f"Column not found: {Q4_TARGET_COLUMN}")

    if Q12_TARGET_COLUMN not in df.columns:
        raise ValueError(f"Column not found: {Q12_TARGET_COLUMN}")

    # Insert Q4 metrics immediately after Q Race.4.
    df = ensure_metric_columns_after(df, Q4_TARGET_COLUMN, Q4_METRIC_COLUMNS)

    # Insert Q12 metrics immediately after Q Race.12.
    df = ensure_metric_columns_after(df, Q12_TARGET_COLUMN, Q12_METRIC_COLUMNS)

    for i in tqdm(
        range(len(df)),
        desc="Q Race.4 and updated race theory quality (API)",
        unit="row",
        mininterval=0.3,
    ):
        q4_done = row_done(df, i, Q4_METRIC_COLUMNS)
        q12_done = row_done(df, i, Q12_METRIC_COLUMNS)

        if q4_done and q12_done:
            continue

        q4_text = df.at[i, Q4_TARGET_COLUMN]
        q12_text = df.at[i, Q12_TARGET_COLUMN]

        if not q4_done:
            q4_result = analyze_q4_response(q4_text)
            write_result(df, i, Q4_METRIC_COLUMNS, q4_result)
            df.to_csv(INPUT_CSV, index=False)

        if not q12_done:
            q12_result = analyze_updated_theory(q4_text, q12_text)
            write_result(df, i, Q12_METRIC_COLUMNS, q12_result)
            df.to_csv(INPUT_CSV, index=False)

    print("Done!")


if __name__ == "__main__":
    main()