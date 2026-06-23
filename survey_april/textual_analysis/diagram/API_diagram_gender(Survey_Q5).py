import os
import time

import pandas as pd
from openai import OpenAI
from pydantic import BaseModel, Field
from tqdm import tqdm

INPUT_CSV = "All_Participants_All_Questions.csv"
TARGET_COLUMN = "Q Gender.5 pre-ML diagram (main effects)"

METRIC_COLUMNS = [
    "Q Gender.5 Number of paths",
    "Q Gender.5 Maximum path length",
    "Q Gender.5 Number of latent variables",
    "Q Gender.5 Coding reasoning",
]

MODEL = "gpt-5.5"
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


class DiagramMetrics(BaseModel):
    number_of_paths: int = Field(description="Number of distinct causal paths explicitly present in the diagram.")
    maximum_path_length: int = Field(description="Maximum number of arrows in any single causal path.")
    number_of_latent_variables: int = Field(description="Number of unique variables in the diagram that are not among the 13 project features and are not the outcome Y.")
    brief_reasoning: str = Field(description="Brief explanation of the coding decision.")


SYSTEM_PROMPT = """
You are a careful research assistant helping code open-ended survey responses from an AI + social science PhD research project.

PROJECT CONTEXT
The project studies human-AI collaboration in scientific theory building. Respondents forecast whether academic papers discuss inequality-related topics, especially racial inequality and gender inequality. The survey asks respondents to construct theories about predictors of whether a paper discusses gender inequality.

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

SURVEY QUESTION OF INTEREST: Q Gender.5
Respondents were instructed:
“Please provide a diagram with arrows, expressed in text form, to represent your theory, where an arrow indicates a causal relationship between two variables.
You may use the arrow symbol ‘→’ to represent the causal direction.
Please also indicate the sign of the causal effect between the two variables connected by each arrow, e.g. '+' or '–'.”

The outcome is whether a paper discusses gender inequality. Respondents may refer to the outcome as:
Y, gender inequality paper, gender inequality research, probability of gender inequality research, likelihood of gender inequality publication, likelihood of gender inequality discussion, or similar.

YOUR CODING TASK
For each respondent’s Q Gender.5 response, code exactly three quantities:

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
   If no causal arrow or causal relation is present, use 0.

3. number_of_latent_variables
   Count the number of unique variables/concepts in the diagram that are NOT one of the 13 official observed features and are NOT the outcome Y.
   These include mediators, mechanisms, latent constructs, invented concepts, or renamed theoretical constructs.
   Examples: topic fit, topic popularity, interest, awareness, comfort, salience, academic trend, perceived legitimacy, gender salience, topic interest.
   Do not count:
   - Y or any synonym of the gender inequality outcome
   - the 13 official observed features, even if written with minor wording variations
   - signs such as + or -
   - generic labels such as X1, X2, mediator, Med1, if they merely label a substantive variable already named nearby

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
"""


def analyze_response(response_text: str, max_retries: int = 3) -> DiagramMetrics:
    if not isinstance(response_text, str) or not response_text.strip():
        return DiagramMetrics(
            number_of_paths=0,
            maximum_path_length=0,
            number_of_latent_variables=0,
            brief_reasoning="Empty or missing response."
        )

    user_prompt = f"""
Please analyze the following Q Gender.5 diagram response.

Respondent response:
\"\"\"
{response_text}
\"\"\"

Return:
1. number_of_paths
2. maximum_path_length
3. number_of_latent_variables
4. brief_reasoning
"""

    for attempt in range(max_retries):
        try:
            completion = client.beta.chat.completions.parse(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=DiagramMetrics,
            )
            return completion.choices[0].message.parsed

        except Exception as e:
            if attempt == max_retries - 1:
                return DiagramMetrics(
                    number_of_paths=-1,
                    maximum_path_length=-1,
                    number_of_latent_variables=-1,
                    brief_reasoning=f"API error after retries: {repr(e)}"
                )
            time.sleep(2 ** attempt)


def _row_metrics_done(df: pd.DataFrame, row: int) -> bool:
    for col in METRIC_COLUMNS:
        val = df.at[row, col]
        if pd.isna(val) or str(val).strip() == "":
            return False
    return True


def main():
    df = pd.read_csv(INPUT_CSV)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Column not found: {TARGET_COLUMN}. Available columns include: {list(df.columns)[:20]}"
        )

    for col in METRIC_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    # Resume based on the INPUT file itself:
    # find the first row that is missing any metric, then process from there.
    first_missing_idx = next((i for i in range(len(df)) if not _row_metrics_done(df, i)), None)
    if first_missing_idx is None:
        print("All rows already have Q Gender.5 metrics in input file. Nothing to do.")
        return

    # Reset to 0..n-1 so work_df.at[i, ...] with loop index i works.
    work_df = df.iloc[first_missing_idx:].copy().reset_index(drop=True)
    print(
        f"Starting from first incomplete input row index {first_missing_idx} "
        f"(processing {len(work_df)} rows)."
    )

    n = len(work_df)
    with tqdm(range(n), desc="Q Gender.5 diagram coding", unit="row") as pbar:
        for i in pbar:
            source_row = first_missing_idx + i
            if _row_metrics_done(work_df, i):
                pbar.set_postfix(row=source_row + 1, status="skip")
                continue

            pbar.set_postfix(row=source_row + 1, status="API")
            text = work_df.at[i, TARGET_COLUMN]
            result = analyze_response(text)

            work_df.at[i, METRIC_COLUMNS[0]] = result.number_of_paths
            work_df.at[i, METRIC_COLUMNS[1]] = result.maximum_path_length
            work_df.at[i, METRIC_COLUMNS[2]] = result.number_of_latent_variables
            work_df.at[i, METRIC_COLUMNS[3]] = result.brief_reasoning

            # Persist directly into the original input file for true in-place resume.
            for c in METRIC_COLUMNS:
                df.at[source_row, c] = work_df.at[i, c]
            df.to_csv(INPUT_CSV, index=False)

    print(f"Done. Updated metrics in-place in: {INPUT_CSV}")


if __name__ == "__main__":
    main()