"""
Reconstruct post-ML main-effects theories for the gender inequality task (Q Gender.12).

Updates All_Participants_All_Questions.csv in place by default.

Example:
    python LLM_integrate_gender_main_effects.py --model "gpt-5.5"
"""

from llm_refine_theory_common import run

QUESTION_CONFIG = {
    "short_name": "Q Gender.12",
    "theory_type": "Main effects",
    "outcome": "Whether a paper discusses gender inequality",
    "question_context": (
        "The three text blocks below are the participant's pre-ML theory, "
        "reaction to ML evidence, and post-ML revision, in that order."
    ),
    "pre_col": "Q Gender.4 pre-ML theory (main effects)",
    "reaction_col": "Q Gender.11 reaction after viewing the ML results (main effects)",
    "post_col": "Q Gender.12 post-ML theory (main effects)",
    "ml_evidence": (
        "The ML results identified the following top main-effect predictors for gender inequality: "
        "social_science (+), female_score (+), natural_science (-), asian (+), "
        "paper_inequality_mentions_3_years (+)."
    ),
}

if __name__ == "__main__":
    run(QUESTION_CONFIG, "Reconstruct Q Gender.12 post-ML main-effects theories")
