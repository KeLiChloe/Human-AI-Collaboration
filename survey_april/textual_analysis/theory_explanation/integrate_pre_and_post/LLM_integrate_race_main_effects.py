"""
Reconstruct post-ML main-effects theories for the racial inequality task (Q Race.12).

Updates All_Participants_All_Questions.csv in place by default.

Example:
    python LLM_integrate_race_main_effects.py --model "gpt-5.5"
"""

from llm_common_context import run

QUESTION_CONFIG = {
    "short_name": "Q Race.12",
    "theory_type": "Main effects",
    "outcome": "Whether a paper discusses racial inequality",
    "question_context": (
        "The two text blocks below are the participant's pre-ML theory and "
        "post-ML revision, in that order. "
    ),
    "pre_col": "Q Race.4 pre-ML theory (main effects)",
    "post_col": "Q Race.12 post-ML theory (main effects)",
    "ml_evidence": (
        "The ML results identified the following top main-effect predictors for racial inequality: "
        "social_science (+), female_score (+), country_race_diversity_score (+), asian (+), black (+)."
    ),
}

if __name__ == "__main__":
    run(QUESTION_CONFIG, "Reconstruct Q Race.12 post-ML main-effects theories")
