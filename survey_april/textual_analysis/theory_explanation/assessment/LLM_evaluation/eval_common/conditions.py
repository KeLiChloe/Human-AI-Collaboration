"""Condition configs for the four evaluation runners."""

from __future__ import annotations

from .catalog import ConditionConfig, StageSpec


def race_main() -> ConditionConfig:
    return ConditionConfig(
        key="race_main",
        task="Race",
        effect="main",
        stages=(
            StageSpec(
                stage="pre",
                theory_columns=("Q Race.4 pre-ML theory (main effects)",),
                score_prefix="Q Race.4",
                insert_after="Q Race.4 pre-ML theory (main effects)",
            ),
            StageSpec(
                stage="post",
                theory_columns=(
                    "Q Race.12 LLM_refined post-ML theory (main effects)",
                ),
                # Existing CSV naming for race post ratings
                score_prefix="Q Race.12 Updated Theory",
                insert_after="Q Race.12 LLM_uncertainty_note post-ML theory (main effects)",
            ),
        ),
    )


def race_soi() -> ConditionConfig:
    return ConditionConfig(
        key="race_soi",
        task="Race",
        effect="soi",
        stages=(
            StageSpec(
                stage="pre",
                theory_columns=("Q Race.10 pre-ML theory (SOI)",),
                score_prefix="Q Race.10",
                insert_after="Q Race.10 pre-ML theory (SOI)",
            ),
            StageSpec(
                stage="post",
                theory_columns=(
                    "Q Race.15 LLM_refined post-ML theory (SOI)",
                ),
                score_prefix="Q Race.15",
                insert_after="Q Race.15 LLM_uncertainty_note post-ML theory (SOI)",
            ),
        ),
    )


def gender_main() -> ConditionConfig:
    return ConditionConfig(
        key="gender_main",
        task="Gender",
        effect="main",
        stages=(
            StageSpec(
                stage="pre",
                theory_columns=("Q Gender.4 pre-ML theory (main effects)",),
                score_prefix="Q Gender.4",
                insert_after="Q Gender.4 pre-ML theory (main effects)",
            ),
            StageSpec(
                stage="post",
                theory_columns=(
                    "Q Gender.12 LLM_refined post-ML theory (main effects)",
                ),
                # Existing CSV naming for gender post ratings (no "Updated Theory")
                score_prefix="Q Gender.12",
                insert_after="Q Gender.12 LLM_uncertainty_note post-ML theory (main effects)",
            ),
        ),
    )


def gender_soi() -> ConditionConfig:
    return ConditionConfig(
        key="gender_soi",
        task="Gender",
        effect="soi",
        stages=(
            StageSpec(
                stage="pre",
                theory_columns=("Q Gender.10 pre-ML theory (SOI)",),
                score_prefix="Q Gender.10",
                insert_after="Q Gender.10 pre-ML theory (SOI)",
            ),
            StageSpec(
                stage="post",
                theory_columns=(
                    "Q Gender.15 LLM_refined post-ML theory (SOI)",
                ),
                score_prefix="Q Gender.15",
                insert_after="Q Gender.15 LLM_uncertainty_note post-ML theory (SOI)",
            ),
        ),
    )
