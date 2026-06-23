"""
Reconstruct post-ML second-order interaction theories for the gender inequality task (Q Gender.15).

Updates All_Participants_All_Questions.csv in place by default.

Example:
    python LLM_integrate_gender_soi.py --model "gpt-5.5"
"""

from llm_refine_theory_common import run

QUESTION_CONFIG = {
    "short_name": "Q Gender.15",
    "theory_type": "Second-order interactions",
    "outcome": "Whether a paper discusses gender inequality",
    "question_context": (
        "The three text blocks below are the participant's pre-ML interaction theory, "
        "reaction to ML evidence, and post-ML revision, in that order. "
        "Reconstruct interactions only; do not restate main-effects reasoning."
    ),
    "pre_col": "Q Gender.10 pre-ML theory (SOI)",
    "reaction_col": "Q Gender.14 reaction after viewing the ML results (SOI)",
    "post_col": "Q Gender.15 post-ML theory (SOI)",
    "ml_evidence": (
        "The ML results identified the following top second-order interactions for gender inequality: "
        "social_science × paper_inequality_mentions_3_years (+), "
        "social_science × news_inequality_mentions_3_years (+), "
        "female × social_science (-)."
    ),
}

if __name__ == "__main__":
    run(QUESTION_CONFIG, "Reconstruct Q Gender.15 post-ML interaction theories")
