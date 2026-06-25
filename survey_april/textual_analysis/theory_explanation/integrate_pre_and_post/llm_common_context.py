"""Shared helpers for post-ML theory reconstruction scripts."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from openai import OpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from tqdm import tqdm

DEFAULT_INPUT_CSV = Path(__file__).resolve().parents[3] / "All_Participants_All_Questions.csv"

SYSTEM_PROMPT = """
You are a research data preprocessing assistant working on a human–AI collaborative theory-building survey. In this survey, participants were asked to develop theoretical explanations for why academic papers do or do not discuss social inequality. The survey focused on two prediction tasks: whether a paper discusses racial inequality and whether a paper discusses gender inequality.

For each task, participants first provided their own pre-ML theoretical reasoning, including main-effect explanations and second-order interaction explanations. They were then shown machine-learning evidence, including the most predictive features and interactions identified from the data, and were asked to revise their theory after reviewing this evidence. The survey explicitly instructed participants to provide a complete version of their updated theory, rather than only listing the modified parts. However, in practice, many participants only wrote partial revisions, brief reactions, or incremental changes.

Your task is to reconstruct participants’ post-ML theoretical explanations into complete versions by integrating each participant’s pre-ML theory with their post-ML revision, while preserving their original language, reasoning, uncertainty, and conceptual framing as much as possible. Do NOT improve, evaluate, strengthen, or make the theory more persuasive.

You must follow these rules:

1. Preserve the participant’s own reasoning.
   - Do not add new mechanisms, variables, causal claims, or explanations that are not present in the participant’s original responses.
   - Do not make the theory more sophisticated than the participant made it.
   - Do not correct substantive mistakes.
   - Do not impose your own interpretation of the ML evidence.

2. Use the post-ML response as the authoritative update.
   - If the post-ML response modifies the pre-ML theory, integrate those modifications into the full theory.
   - If the post-ML response contradicts the pre-ML theory, the post-ML response takes precedence.

3. Use the pre-ML theory to reconstruct missing context.
   - If the post-ML response only states the modified parts, merge those modifications back into the pre-ML theory to produce a complete post-ML version.
   - If the participant explicitly says they do not want to revise their theory, or indicates no change, return the pre-ML theory verbatim as the complete post-ML theory (status: no_change). Do not add, remove, or rephrase content.
   - If the post-ML response is merely a reaction to the ML evidence, use it only insofar as it reveals what the participant would revise or retain.

4. Preserve style and write as theory, not as a reaction to ML.
   - Preserve the participant’s vocabulary, hedging, and level of specificity.
   - Do not turn the response into bullet points.
   - Do not add citations.
   - Write the output as a standalone theoretical explanation.
   - The ML evidence block is provided only as background for your reconstruction task. Do not introduce ML findings into refined_theory unless the participant explicitly incorporated them into their own theoretical reasoning.

5. Scope.
   - For main-effect questions, reconstruct only the participant’s theory about main effects.
   - For second-order interaction questions, reconstruct only the participant’s theory about second-order interactions.

6. Output JSON only, with exactly these fields:
   - refined_theory: the reconstructed complete post-ML theory.
   - status: one of:
     - already_complete: the post-ML revision field already contains a complete theory; return it with minimal or no merging.
     - merged_pre_post: the post-ML revision contained only partial updates; you merged those updates into the pre-ML theory.
     - no_change: the participant explicitly indicated no revision; return the pre-ML theory verbatim.
     - insufficient_information: pre-ML and post-ML together do not contain enough substance to reconstruct a complete theory.
   - uncertainty_note: any uncertainty about the reconstruction. Use an empty string if none.
""".strip()

REFINED_THEORY_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "refined_theory": {
            "type": "string",
            "description": (
                "The reconstructed complete post-ML theory as one narrative paragraphs. "
            ),
        },
        "status": {
            "type": "string",
            "enum": [
                "already_complete",
                "merged_pre_post",
                "no_change",
                "insufficient_information",
            ],
            "description": (
                "already_complete: post-ML revision is already complete; "
                "merged_pre_post: partial updates merged into pre-ML; "
                "no_change: participant indicated no revision, return pre-ML verbatim; "
                "insufficient_information: not enough substance to reconstruct."
            ),
        },
        "uncertainty_note": {
            "type": "string",
            "description": "Any uncertainty about the reconstruction. Empty string if none.",
        },
    },
    "required": [
        "refined_theory",
        "status",
        "uncertainty_note",
    ],
    "additionalProperties": False,
}


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if pd.isna(value):
        return ""
    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    return " ".join(text.split()).strip()


def find_column(df: pd.DataFrame, exact_col: str) -> str:
    if exact_col in df.columns:
        return exact_col

    normalized_target = " ".join(exact_col.split()).strip()
    for col in df.columns:
        if " ".join(col.split()).strip() == normalized_target:
            return col

    raise KeyError(
        f"Column not found:\n{exact_col}\n\n"
        f"Closest available columns containing the first token:\n"
        + "\n".join([c for c in df.columns if exact_col.split()[0] in c][:20])
    )


def build_user_prompt(
    config: Dict[str, str],
    pre_ml_theory: str,
    post_ml_response: str,
) -> str:
    return f"""
Question context:
{config["question_context"]}

Theory type:
{config["theory_type"]}

Outcome:
{config["outcome"]}

Participant's pre-ML theoretical explanation:
{pre_ml_theory} 

Relevant ML evidence shown to participant:
{config["ml_evidence"]}

Participant's post-ML revision response:
{post_ml_response}

Now reconstruct the participant's complete post-ML theoretical explanation according to the rules above.
Write refined_theory as a standalone theoretical explanation.
If the participant indicated they do not want to revise, return the pre-ML theory verbatim (status: no_change).
""".strip()


class LLMCallError(Exception):
    pass


@retry(
    retry=retry_if_exception_type((LLMCallError, TimeoutError, ConnectionError)),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    stop=stop_after_attempt(5),
)
def call_llm(
    client: OpenAI,
    model: str,
    user_prompt: str,
    max_output_tokens: int = 2400,
) -> Dict[str, str]:
    try:
        response = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "refined_post_ml_theory",
                    "schema": REFINED_THEORY_SCHEMA,
                    "strict": True,
                }
            },
            max_output_tokens=max_output_tokens,
        )
        parsed = json.loads(response.output_text)
        for key in [
            "refined_theory",
            "status",
            "uncertainty_note",
        ]:
            if key not in parsed:
                raise LLMCallError(f"Missing key in model output: {key}")
        return parsed
    except json.JSONDecodeError as exc:
        raise LLMCallError(f"Could not parse JSON from model output: {exc}") from exc
    except Exception as exc:
        raise LLMCallError(f"OpenAI API call failed: {exc}") from exc


LLM_COLUMN_FIELDS = [
    "refined",
    "status",
    "uncertainty_note",
]


def llm_column_label(config: Dict[str, str]) -> str:
    post_col = config["post_col"]
    prefix = config["short_name"]
    if post_col.startswith(prefix):
        return post_col[len(prefix) :].strip()
    return post_col


def llm_column_name(config: Dict[str, str], field: str) -> str:
    return f"{config['short_name']} LLM_{field} {llm_column_label(config)}"


def legacy_llm_column_name(prefix: str, field: str) -> str:
    return f"{prefix} (LLM_{field})"


def llm_column_names(config: Dict[str, str]) -> list[str]:
    return [llm_column_name(config, field) for field in LLM_COLUMN_FIELDS]



def migrate_legacy_llm_columns(df: pd.DataFrame, config: Dict[str, str]) -> pd.DataFrame:
    prefix = config["short_name"]
    for field in LLM_COLUMN_FIELDS:
        old_col = legacy_llm_column_name(prefix, field)
        new_col = llm_column_name(config, field)
        if old_col not in df.columns:
            continue
        if new_col not in df.columns:
            df = df.rename(columns={old_col: new_col})
            continue
        has_old = df[old_col].map(clean_text).astype(bool)
        empty_new = ~df[new_col].map(clean_text).astype(bool)
        df.loc[has_old & empty_new, new_col] = df.loc[has_old & empty_new, old_col]
        df = df.drop(columns=[old_col])
    return df


def initialize_output_columns(df: pd.DataFrame, config: Dict[str, str]) -> pd.DataFrame:
    df = migrate_legacy_llm_columns(df, config)
    llm_cols = llm_column_names(config)

    for col in llm_cols:
        if col not in df.columns:
            df[col] = ""

    post_col = find_column(df, config["post_col"])
    other_cols = [col for col in df.columns if col not in llm_cols]
    insert_at = other_cols.index(post_col) + 1
    return df[other_cols[:insert_at] + llm_cols + other_cols[insert_at:]]


def should_skip_existing(row: pd.Series, config: Dict[str, str], overwrite: bool) -> bool:
    if overwrite:
        return False
    status_col = llm_column_name(config, "status")
    refined_col = llm_column_name(config, "refined")
    return bool(clean_text(row.get(status_col, ""))) and bool(clean_text(row.get(refined_col, "")))


def process_one_cell(
    client: OpenAI,
    model: str,
    row: pd.Series,
    config: Dict[str, str],
    col_lookup: Dict[str, str],
) -> Dict[str, str]:
    pre_ml = clean_text(row.get(col_lookup[config["pre_col"]], ""))
    post_ml = clean_text(row.get(col_lookup[config["post_col"]], ""))

    if not pre_ml and not post_ml:
        return {
            "refined_theory": "",
            "status": "insufficient_information",
            "uncertainty_note": "No theory text was available for reconstruction.",
        }

    return call_llm(
        client=client,
        model=model,
        user_prompt=build_user_prompt(
            config=config,
            pre_ml_theory=pre_ml,
            post_ml_response=post_ml,
        ),
    )


def save_checkpoint(df: pd.DataFrame, output_path: str) -> None:
    df.to_csv(output_path, index=False, encoding="utf-8-sig")


def run(config: Dict[str, str], script_description: str) -> None:
    parser = argparse.ArgumentParser(description=script_description)
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT_CSV),
        help=f"Path to input CSV file (default: {DEFAULT_INPUT_CSV.name}).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to output CSV file. Defaults to --input (in-place update).",
    )
    parser.add_argument("--model", default="gpt-5.5", help="OpenAI model to use.")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional: process only the first N rows for testing.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
        help="Save output after every N successful LLM calls.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Optional sleep in seconds between LLM calls.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, overwrite existing LLM_refined/status columns.",
    )
    parser.add_argument(
        "--resume-from-output",
        action="store_true",
        help=(
            "If set and --output differs from --input, resume from the output file "
            "instead of the input file. In-place mode always resumes from --input."
        ),
    )
    parser.add_argument(
        "--exclude-genai",
        action="store_true",
        help="If set, skip rows where student_0, expert_1, genAI_2 == 2.",
    )
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI()
    input_path = args.input
    output_path = args.output or args.input
    inplace = os.path.abspath(output_path) == os.path.abspath(input_path)

    if args.resume_from_output and not inplace and os.path.exists(output_path):
        input_path = output_path
        print(f"Resuming from existing output file: {output_path}")
    elif inplace:
        print(f"Updating in place: {input_path}")

    print(f"Reading CSV: {input_path}")
    df = pd.read_csv(input_path, dtype=str, keep_default_na=False)
    print(f"Loaded shape: {df.shape}")

    if args.max_rows is not None:
        df = df.head(args.max_rows).copy()
        print(f"Processing first {args.max_rows} rows only. Shape: {df.shape}")

    df = initialize_output_columns(df, config)

    col_lookup = {
        config[key]: find_column(df, config[key])
        for key in ["pre_col", "post_col"]
    }

    participant_type_col = "student_0, expert_1, genAI_2"
    if participant_type_col not in df.columns:
        participant_type_col = None

    prefix = config["short_name"]
    refined_col = llm_column_name(config, "refined")
    status_col = llm_column_name(config, "status")
    uncertainty_col = llm_column_name(config, "uncertainty_note")

    completed_calls = 0
    failed_calls = 0
    attempted_tasks = 0
    skipped_genai = 0
    skipped_done = 0

    print(f"Processing: {prefix}")
    with tqdm(
        df.index,
        total=len(df),
        desc=f"{prefix} LLM refine",
        unit="row",
        mininterval=0.3,
    ) as progress:
        for row_num, idx in enumerate(progress, start=1):
            if args.exclude_genai and participant_type_col is not None:
                if clean_text(df.loc[idx, participant_type_col]) == "2":
                    skipped_genai += 1
                    progress.set_postfix(row=row_num, status="skip genAI", ok=completed_calls)
                    continue

            if should_skip_existing(df.loc[idx], config, overwrite=args.overwrite):
                skipped_done += 1
                progress.set_postfix(row=row_num, status="skip done", ok=completed_calls)
                continue

            attempted_tasks += 1
            progress.set_postfix(row=row_num, status="API", ok=completed_calls)

            try:
                result = process_one_cell(
                    client=client,
                    model=args.model,
                    row=df.loc[idx],
                    config=config,
                    col_lookup=col_lookup,
                )
                df.at[idx, refined_col] = result.get("refined_theory", "")
                df.at[idx, status_col] = result.get("status", "")
                df.at[idx, uncertainty_col] = result.get("uncertainty_note", "")
                completed_calls += 1

                if completed_calls % args.checkpoint_every == 0:
                    save_checkpoint(df, output_path)
                if args.sleep > 0:
                    time.sleep(args.sleep)

                progress.set_postfix(
                    row=row_num,
                    status=result.get("status", "")[:24],
                    ok=completed_calls,
                )
            except Exception as exc:
                failed_calls += 1
                name = clean_text(df.loc[idx, "What is your full name?"]) or f"row {row_num}"
                print(f"ERROR row {row_num} ({name}): {exc}", file=sys.stderr)
                save_checkpoint(df, output_path)
                progress.set_postfix(row=row_num, status="error", ok=completed_calls)

    save_checkpoint(df, output_path)

    print("\nDone.")
    print(f"Rows: {len(df)} | Skipped (genAI): {skipped_genai} | Skipped (done): {skipped_done}")
    print(f"Attempted tasks: {attempted_tasks}")
    print(f"Successful LLM calls: {completed_calls} | Failed: {failed_calls}")
    if inplace:
        print(f"Updated in place: {output_path}")
    else:
        print(f"Output saved to: {output_path}")

    if status_col in df.columns:
        print(f"\n{status_col}")
        print(df[status_col].value_counts(dropna=False).to_string())
