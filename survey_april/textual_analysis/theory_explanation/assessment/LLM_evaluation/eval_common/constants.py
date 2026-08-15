"""Shared constants for LLM theory evaluation."""

from __future__ import annotations

from pathlib import Path

# LLM_evaluation/ -> assessment/ -> theory_explanation/ -> textual_analysis/ -> survey_april/
SURVEY_ROOT = Path(__file__).resolve().parents[5]
INPUT_CSV = SURVEY_ROOT / "All_Participants_All_Questions.csv"

NAME_COLUMN = "What is your full name?"
GROUP_COLUMN = "student_0, senior_1, genAI_2"

MODEL = "gpt-5.6-sol"
REASONING_EFFORT = "high"  # none|low|medium|high|xhigh|max
BATCH_SIZE = 25
SHUFFLE_SEED = 20260811
API_RETRIES = 3

# Canonical feature order used throughout the survey
FEATURE_NAMES: tuple[str, ...] = (
    "social_science",
    "natural_science",
    "engineering_and_technology",
    "num_authors",
    "female",
    "asian",
    "black",
    "hispanic_and_other",
    "white",
    "authors_race_diversity_score",
    "country_race_diversity_score",
    "news_inequality_mentions_3_years",
    "paper_inequality_mentions_3_years",
)

FEATURE_DEFINITIONS: dict[str, str] = {
    "social_science": (
        "Equals 1 if the paper is published in a journal whose scope lies "
        "within the social sciences, and 0 otherwise."
    ),
    "natural_science": (
        "Equals 1 if the paper is published in a journal whose scope lies "
        "within the natural sciences, and 0 otherwise."
    ),
    "engineering_and_technology": (
        "Equals 1 if the paper is published in a journal whose scope lies "
        "within engineering and technology, and 0 otherwise."
    ),
    "num_authors": "The total number of authors of the paper.",
    "female": "The estimated share of female authors on the author team.",
    "asian": "The estimated share of Asian authors on the author team.",
    "black": "The estimated share of Black authors on the author team.",
    "hispanic_and_other": (
        "The estimated share of Hispanic and other-race authors on the author team."
    ),
    "white": "The estimated share of White authors on the author team.",
    "authors_race_diversity_score": (
        "The racial diversity within the co-author team, measured by Shannon entropy."
    ),
    "country_race_diversity_score": (
        "The average racial diversity of the authors' inferred countries of birth, "
        "measured by Shannon entropy."
    ),
    "news_inequality_mentions_3_years": (
        "The average percentage of news articles mentioning inequality over the "
        "three years preceding the paper's publication year."
    ),
    "paper_inequality_mentions_3_years": (
        "The average percentage of academic papers mentioning inequality over the "
        "three years preceding the paper's publication year."
    ),
}

SCORE_SUFFIXES: tuple[str, ...] = (
    "Clarity and Coherence",
    "Causal Reasoning",
    "Theoretical Depth",
    "Creativity",
    "Persuasiveness",
)

BRIEF_REASONING_SUFFIX = "Brief Reasoning"
MECHANISMS_SUFFIX = "Mechanisms"
OVERALL_QUALITY_SUFFIX = "Overall Quality Score"

# Appended to every AI-rating column name written by this pipeline.
SCORE_MODEL_TAG = MODEL  # column tag; keep in sync with MODEL


def metric_col(prefix: str, suffix: str, *, model_tag: str = SCORE_MODEL_TAG) -> str:
    """e.g. metric_col('Q Race.4', 'Clarity and Coherence') -> '... (gpt-5.5)'."""
    return f"{prefix} {suffix} ({model_tag})"


SCORE_FIELD_TO_SUFFIX: dict[str, str] = {
    "clarity_and_coherence": "Clarity and Coherence",
    "causal_reasoning": "Causal Reasoning",
    "theoretical_depth": "Theoretical Depth",
    "creativity": "Creativity",
    "persuasiveness": "Persuasiveness",
}
