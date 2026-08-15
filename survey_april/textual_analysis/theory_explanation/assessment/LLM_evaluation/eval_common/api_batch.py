"""OpenAI batch scoring calls."""

from __future__ import annotations

import os
import time

from openai import BadRequestError, OpenAI

from .catalog import TheoryItem
from .constants import API_RETRIES, MODEL, REASONING_EFFORT
from .prompts import build_system_prompt, build_user_prompt
from .schemas import BatchScores, TheoryScore


def _client() -> OpenAI:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    return OpenAI(api_key=key)


def score_batch(
    batch: list[TheoryItem],
    *,
    task: str,
    effect: str,
    model: str = MODEL,
    reasoning_effort: str = REASONING_EFFORT,
    retries: int = API_RETRIES,
    client: OpenAI | None = None,
) -> list[TheoryScore]:
    """
    Score one batch in a single API call.
    Validates that every requested theory_id is present exactly once.
    """
    if not batch:
        return []

    api = client or _client()
    system = build_system_prompt(task=task, effect=effect)
    user = build_user_prompt(batch)
    expected = {item.theory_id for item in batch}

    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            # Responses API: reasoning={"effort": ...}
            # (Chat Completions equivalent would be reasoning_effort=...)
            response = api.responses.parse(
                model=model,
                instructions=system,
                input=user,
                text_format=BatchScores,
                reasoning={"effort": reasoning_effort},
            )
            parsed = response.output_parsed
            if parsed is None:
                raise RuntimeError("Model returned empty parsed content")

            got = {s.theory_id for s in parsed.scores}
            missing = expected - got
            extra = got - expected
            if missing or extra:
                raise RuntimeError(
                    f"theory_id mismatch; missing={sorted(missing)} extra={sorted(extra)}"
                )
            if len(parsed.scores) != len(expected):
                raise RuntimeError(
                    f"duplicate theory_ids in response "
                    f"(n={len(parsed.scores)}, expected={len(expected)})"
                )
            # Preserve batch order for logging; map by id for writeback
            by_id = {s.theory_id: s for s in parsed.scores}
            return [by_id[item.theory_id] for item in batch]

        except BadRequestError:
            raise
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if attempt == retries - 1:
                break
            time.sleep(2**attempt)

    raise RuntimeError(f"Batch scoring failed after {retries} retries: {last_err!r}")
