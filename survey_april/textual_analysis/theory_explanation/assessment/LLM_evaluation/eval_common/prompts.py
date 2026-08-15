"""System and user prompts for batch theory evaluation."""

from __future__ import annotations

from .constants import FEATURE_DEFINITIONS, FEATURE_NAMES
from .catalog import TheoryItem


def _feature_block() -> str:
    lines: list[str] = []
    for i, name in enumerate(FEATURE_NAMES, start=1):
        lines.append(f"{i}. {name}")
        lines.append(f"   {FEATURE_DEFINITIONS[name]}")
        lines.append("")
    return "\n".join(lines).rstrip()


def build_system_prompt(*, task: str, effect: str) -> str:
    """
    task: 'Race' | 'Gender'
    effect: 'main' | 'soi'
    """
    if task == "Race":
        outcome = "whether an academic paper discusses racial inequality"
        domain = "racial inequality"
    elif task == "Gender":
        outcome = "whether an academic paper discusses gender inequality"
        domain = "gender inequality"
    else:
        raise ValueError(f"Unknown task: {task}")

    if effect == "main":
        task_blurb = (
            "Each respondent selected 5 features from the list below "
            "(with a predicted positive or negative association with the outcome) "
            "and wrote a theoretical explanation."
        )
    elif effect == "soi":
        task_blurb = (
            "Each respondent selected 3 two-way interactions among the features "
            "below (second-order interactions / SOI, each with a predicted positive "
            "or negative association with the outcome) and wrote a theoretical "
            "explanation."
        )
    else:
        raise ValueError(f"Unknown effect: {effect}")

    return f"""
You are an expert in social science research and theory evaluation.
Use your strongest analytical judgment and highest level of social-scientific reasoning. Be rigorous, discerning, and careful.

PROJECT CONTEXT
This study examines theory building for predicting mentions of {domain} in academic papers.
The outcome is {outcome}.

{task_blurb}

Available features:

{_feature_block()}

TASK
You will receive a batch of theoretical explanations. Each item has an anonymous theory_id and the theory text.
Evaluate each theory independently on its own merits.

-------------------------------------
EVALUATION DIMENSIONS (1–10 scale)
-------------------------------------

For each dimension, assign a score from 1 (very poor) to 10 (excellent).

1. Clarity and Coherence
Is the explanation clearly written, well-structured, and logically consistent, without ambiguity or internal contradictions?

2. Causal Reasoning
Does the explanation articulate plausible causal mechanisms relevant to the outcome?

3. Theoretical Depth
Does the explanation go beyond surface-level statements and engage with meaningful underlying concepts or mechanisms?

4. Creativity
Does the explanation demonstrate creative or original thinking, such as offering novel perspectives, non-obvious connections, or insightful interpretations?

5. Persuasiveness
Does the explanation provide a convincing theoretical account?

-------------------------------------
SCORING GUIDELINES
-------------------------------------

1–2 = poor
3–4 = weak
5–6 = moderate
7–8 = strong
9–10 = excellent

-------------------------------------
OUTPUT
-------------------------------------

Return scores for EVERY theory_id in the batch.
For each theory, provide the five dimension scores and a brief_reasoning (at most 5 sentences) that justifies the scores.
Do not omit any theory_id. Do not invent theory_ids that were not provided.
""".strip()


def format_theory_block(item: TheoryItem) -> str:
    return (
        f"[{item.theory_id}]\n"
        f"Theory:\n\"\"\"\n{item.theory_text}\n\"\"\""
    )


def build_user_prompt(batch: list[TheoryItem]) -> str:
    ids = ", ".join(item.theory_id for item in batch)
    blocks = "\n\n-----\n\n".join(format_theory_block(item) for item in batch)
    return f"""
Evaluate each of the following theoretical explanations independently.

Theory IDs in this batch (in order presented): {ids}

Return one score object per theory_id listed above.

{blocks}
""".strip()
