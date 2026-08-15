"""Blind theory rating app: seed controls sample; identifier controls progress."""

from __future__ import annotations

import csv
import io
import json
import os
import random
import re
import secrets
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, field_validator

ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("RATING_DATA_DIR", str(ROOT / "data")))
DATA_PATH = DATA_DIR / "theories.json"
DB_PATH = Path(os.environ.get("RATING_DB_PATH", str(DATA_DIR / "ratings.db")))
SAMPLE_PER_CELL = int(os.environ.get("SAMPLE_PER_CELL", "20"))
SAMPLE_SIZE = SAMPLE_PER_CELL * 4  # race/gender × main/soi
# Relative sampling weights vs PhD students (weight 1.0).
GENAI_SAMPLE_WEIGHT = float(os.environ.get("GENAI_SAMPLE_WEIGHT", "2.0"))
EXPERT_SAMPLE_WEIGHT = float(os.environ.get("EXPERT_SAMPLE_WEIGHT", "3.5"))
DEFAULT_SCORE = 1
# Default public sample. Special raters (EU/KL/PP) get seed 99 with pre/post reweighted.
DEFAULT_PUBLIC_SEED = os.environ.get("DEFAULT_PUBLIC_SEED", "1024").strip() or "1024"
SPECIAL_PUBLIC_SEED = os.environ.get("SPECIAL_PUBLIC_SEED", "99").strip() or "99"
SPECIAL_IDENTIFIERS = {"EU", "KL", "PP"}
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "").strip() or "research-admin-change-me"

GROUP_LABELS = {"0": "PhD Student", "1": "Senior Scientist", "2": "GenAI"}
SOURCE_LABELS = {"human": "Human", "genai": "GenAI", "GenAI": "GenAI"}
PAGES = [
    {
        "key": "race_main",
        "title": "Racial Inequality — Main Effects",
        "task": "race",
        "effect": "main",
        "short": "Race · Main",
    },
    {
        "key": "race_soi",
        "title": "Racial Inequality — Interactions",
        "task": "race",
        "effect": "soi",
        "short": "Race · Interactions",
    },
    {
        "key": "gender_main",
        "title": "Gender Inequality — Main Effects",
        "task": "gender",
        "effect": "main",
        "short": "Gender · Main",
    },
    {
        "key": "gender_soi",
        "title": "Gender Inequality — Interactions",
        "task": "gender",
        "effect": "soi",
        "short": "Gender · Interactions",
    },
]

DIMENSIONS = [
    {
        "key": "clarity_coherence",
        "label": "Clarity and Coherence",
        "description": (
            "Is the explanation clearly written, well-structured, and logically consistent, "
            "without ambiguity or internal contradictions?"
        ),
    },
    {
        "key": "causal_reasoning",
        "label": "Causal Reasoning",
        "description": (
            "Does the explanation articulate plausible causal mechanisms linking the "
            "predictors to the outcome?"
        ),
    },
    {
        "key": "theoretical_depth",
        "label": "Theoretical Depth",
        "description": (
            "Does the explanation go beyond surface-level statements and engage with "
            "meaningful underlying concepts or mechanisms?"
        ),
    },
    {
        "key": "creativity",
        "label": "Creativity",
        "description": (
            "Does the explanation demonstrate creative or original thinking, such as "
            "offering novel perspectives, non-obvious connections, or insightful interpretations?"
        ),
    },
    {
        "key": "persuasiveness",
        "label": "Persuasiveness",
        "description": (
            "Does the explanation provide a convincing theoretical account of why the "
            "predictors should be related to the outcome?"
        ),
    },
]
DIM_KEYS = [d["key"] for d in DIMENSIONS]

app = FastAPI(title="Theory Rating")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
templates = Jinja2Templates(directory=str(ROOT / "templates"))

_POOL: list[dict[str, Any]] | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_pool() -> list[dict[str, Any]]:
    global _POOL
    if _POOL is None:
        _POOL = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    return _POOL


def pool_by_id() -> dict[str, dict[str, Any]]:
    return {t["id"]: t for t in load_pool()}


def normalize_identifier(raw: str) -> str:
    cleaned = re.sub(r"\s+", " ", (raw or "").strip())
    if len(cleaned) < 2:
        raise HTTPException(status_code=400, detail="Identifier must be at least 2 characters.")
    if len(cleaned) > 64:
        raise HTTPException(status_code=400, detail="Identifier must be at most 64 characters.")
    return cleaned


def normalize_seed(raw: str) -> str:
    cleaned = (raw or "").strip()
    if not cleaned:
        raise HTTPException(status_code=400, detail="Seed is required.")
    if len(cleaned) > 64:
        raise HTTPException(status_code=400, detail="Seed must be at most 64 characters.")
    return cleaned


def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with db() as conn:
        # Fresh schema keyed by (identifier, seed). Drop legacy tables if present.
        cols = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(assignments)").fetchall()
        }
        if cols and "seed" not in cols:
            conn.executescript(
                """
                DROP TABLE IF EXISTS ratings;
                DROP TABLE IF EXISTS assignments;
                DROP TABLE IF EXISTS raters;
                DROP TABLE IF EXISTS sessions;
                """
            )
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                identifier TEXT NOT NULL COLLATE NOCASE,
                seed TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                submitted_at TEXT,
                PRIMARY KEY (identifier, seed)
            );

            CREATE TABLE IF NOT EXISTS assignments (
                identifier TEXT NOT NULL COLLATE NOCASE,
                seed TEXT NOT NULL,
                position INTEGER NOT NULL,
                theory_id TEXT NOT NULL,
                PRIMARY KEY (identifier, seed, position),
                FOREIGN KEY (identifier, seed)
                    REFERENCES sessions(identifier, seed) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS ratings (
                identifier TEXT NOT NULL COLLATE NOCASE,
                seed TEXT NOT NULL,
                theory_id TEXT NOT NULL,
                clarity_coherence INTEGER,
                causal_reasoning INTEGER,
                theoretical_depth INTEGER,
                creativity INTEGER,
                persuasiveness INTEGER,
                touched INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (identifier, seed, theory_id),
                FOREIGN KEY (identifier, seed)
                    REFERENCES sessions(identifier, seed) ON DELETE CASCADE
            );
            """
        )
        session_cols = {
            row["name"] for row in conn.execute("PRAGMA table_info(sessions)").fetchall()
        }
        if "submitted_at" not in session_cols:
            conn.execute("ALTER TABLE sessions ADD COLUMN submitted_at TEXT")
        rating_cols = {
            row["name"] for row in conn.execute("PRAGMA table_info(ratings)").fetchall()
        }
        if "touched" not in rating_cols:
            conn.execute(
                "ALTER TABLE ratings ADD COLUMN touched INTEGER NOT NULL DEFAULT 0"
            )


def default_scores() -> dict[str, int]:
    return {k: DEFAULT_SCORE for k in DIM_KEYS}


def resolve_seed_for_identifier(identifier: str) -> str:
    """Map rater id to the canonical public seed (ignores any client-supplied seed)."""
    key = (identifier or "").strip().upper()
    if key in SPECIAL_IDENTIFIERS:
        return SPECIAL_PUBLIC_SEED
    return DEFAULT_PUBLIC_SEED


def _sample_weight(item: dict[str, Any]) -> float:
    group = str(item.get("group", "")).strip()
    if group == "2" or str(item.get("source", "")).lower() == "genai":
        return GENAI_SAMPLE_WEIGHT
    if group == "1":
        return EXPERT_SAMPLE_WEIGHT
    return 1.0


def _weights_for_pool(
    pool: list[dict[str, Any]],
    *,
    balance_pre_post: bool,
) -> list[float]:
    """Group weights; optionally normalize so pre and post have equal total mass."""
    base = [_sample_weight(item) for item in pool]
    if not balance_pre_post:
        return base
    phase_totals: dict[str, float] = {}
    phases: list[str] = []
    for item, weight in zip(pool, base):
        phase = str(item.get("phase") or "").strip().lower()
        if phase not in {"pre", "post"}:
            phase = "pre"
        phases.append(phase)
        phase_totals[phase] = phase_totals.get(phase, 0.0) + weight
    out: list[float] = []
    for weight, phase in zip(base, phases):
        denom = phase_totals.get(phase) or 1.0
        out.append(weight / denom)
    return out


def weighted_sample_without_replacement(
    rng: random.Random,
    items: list[dict[str, Any]],
    k: int,
    *,
    balance_pre_post: bool = False,
) -> list[dict[str, Any]]:
    """Sample k items without replacement using per-group weights."""
    pool = list(items)
    chosen: list[dict[str, Any]] = []
    for _ in range(k):
        weights = _weights_for_pool(pool, balance_pre_post=balance_pre_post)
        pick = rng.choices(pool, weights=weights, k=1)[0]
        chosen.append(pick)
        pool.remove(pick)
    return chosen


def sample_theory_ids(seed: str) -> list[str]:
    """Deterministic stratified sample: SAMPLE_PER_CELL per task×effect cell (4 pages).

    Same seed string always yields the same theory ids (independent of identifier).
    GenAI / Senior Scientist theories are upweighted via GENAI_SAMPLE_WEIGHT / EXPERT_SAMPLE_WEIGHT.
    Seed 99 (special raters) also reweights so pre and post have equal total mass.
    """
    seed_key = str(seed).strip()
    balance_pre_post = seed_key == SPECIAL_PUBLIC_SEED
    pool = load_pool()
    by_cell: dict[tuple[str, str], list[dict[str, Any]]] = {
        (page["task"], page["effect"]): [] for page in PAGES
    }
    for item in pool:
        key = (item.get("task"), item.get("effect"))
        if key in by_cell:
            by_cell[key].append(item)
    rng = random.Random(seed_key)
    chosen: list[str] = []
    for page in PAGES:
        cell = (page["task"], page["effect"])
        items = sorted(by_cell[cell], key=lambda item: item["id"])
        if len(items) < SAMPLE_PER_CELL:
            raise HTTPException(
                status_code=500,
                detail=f"Not enough theories for {page['key']} (have {len(items)}).",
            )
        picked = weighted_sample_without_replacement(
            rng,
            items,
            SAMPLE_PER_CELL,
            balance_pre_post=balance_pre_post,
        )
        chosen.extend(item["id"] for item in picked)
    return chosen


def ensure_assignment(conn: sqlite3.Connection, identifier: str, seed: str) -> list[str]:
    """Bind this rater to the seed's canonical sample (overwrite any stale assignment)."""
    theory_ids = sample_theory_ids(seed)
    now = utc_now()
    conn.execute(
        """
        INSERT INTO sessions (identifier, seed, created_at, updated_at, submitted_at)
        VALUES (?, ?, ?, ?, NULL)
        ON CONFLICT(identifier, seed) DO UPDATE SET updated_at=excluded.updated_at
        """,
        (identifier, seed, now, now),
    )
    conn.execute(
        "DELETE FROM assignments WHERE identifier = ? AND seed = ?",
        (identifier, seed),
    )
    for pos, theory_id in enumerate(theory_ids, start=1):
        conn.execute(
            """
            INSERT INTO assignments (identifier, seed, position, theory_id)
            VALUES (?, ?, ?, ?)
            """,
            (identifier, seed, pos, theory_id),
        )
    # Drop ratings for theories that are no longer in this seed's sample.
    placeholders = ", ".join("?" for _ in theory_ids)
    conn.execute(
        f"""
        DELETE FROM ratings
        WHERE identifier = ? AND seed = ?
          AND theory_id NOT IN ({placeholders})
        """,
        (identifier, seed, *theory_ids),
    )
    return theory_ids


def public_theory(
    item: dict[str, Any],
    ratings: dict[str, Any] | None,
    position: int,
    *,
    started: bool,
    display_label: str,
) -> dict[str, Any]:
    return {
        "id": item["id"],
        "position": position,
        "topic": item["topic"],
        "task": item["task"],
        "effect": item.get("effect"),
        "effect_label": item.get("effect_label"),
        "text": item["text"],
        "selections": item.get("selections") or [],
        "ratings": ratings if ratings else default_scores(),
        "started": started,
        "display_label": display_label,
    }


def get_assignment(conn: sqlite3.Connection, identifier: str, seed: str) -> list[str]:
    rows = conn.execute(
        """
        SELECT theory_id FROM assignments
        WHERE identifier = ? AND seed = ?
        ORDER BY position
        """,
        (identifier, seed),
    ).fetchall()
    return [r["theory_id"] for r in rows]


def get_submitted_at(conn: sqlite3.Connection, identifier: str, seed: str) -> str | None:
    row = conn.execute(
        "SELECT submitted_at FROM sessions WHERE identifier = ? AND seed = ?",
        (identifier, seed),
    ).fetchone()
    return row["submitted_at"] if row else None


def upsert_rating(
    conn: sqlite3.Connection,
    identifier: str,
    seed: str,
    theory_id: str,
    scores: dict[str, int],
    now: str,
    *,
    touched: bool,
) -> None:
    touch_flag = 1 if touched else 0
    conn.execute(
        f"""
        INSERT INTO ratings (
            identifier, seed, theory_id, {', '.join(DIM_KEYS)}, touched, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(identifier, seed, theory_id) DO UPDATE SET
            {', '.join(f'{k}=excluded.{k}' for k in DIM_KEYS)},
            touched=MAX(ratings.touched, excluded.touched),
            updated_at=excluded.updated_at
        """,
        (identifier, seed, theory_id, *(scores[k] for k in DIM_KEYS), touch_flag, now),
    )


def ensure_default_ratings(
    conn: sqlite3.Connection, identifier: str, seed: str, theory_ids: list[str]
) -> None:
    """Fill missing theories with default scores at submit time (does not count as progress)."""
    rated = ratings_map(conn, identifier, seed)
    now = utc_now()
    defaults = default_scores()
    for theory_id in theory_ids:
        if theory_complete(rated.get(theory_id)):
            continue
        upsert_rating(conn, identifier, seed, theory_id, defaults, now, touched=False)


def build_pages(
    theory_ids: list[str],
    by_id: dict[str, dict[str, Any]],
    rated: dict[str, dict[str, int | None]],
    started_ids: set[str],
) -> list[dict[str, Any]]:
    labels = theory_display_labels(theory_ids, by_id)
    theories = []
    for pos, tid in enumerate(theory_ids, start=1):
        item = by_id.get(tid)
        if not item:
            continue
        theories.append(
            public_theory(
                item,
                rated.get(tid),
                pos,
                started=tid in started_ids,
                display_label=labels.get(tid, f"Theory {pos}"),
            )
        )

    pages = []
    for page in PAGES:
        page_theories = [
            t
            for t in theories
            if t.get("task") == page["task"] and t.get("effect") == page["effect"]
        ]
        for t in page_theories:
            # Keep global 1..100 index (assignment position), not per-page 1..25.
            t["page_position"] = t.get("position")
        pages.append(
            {
                "key": page["key"],
                "title": page["title"],
                "task": page["task"],
                "effect": page["effect"],
                "short": page["short"],
                "theories": page_theories,
            }
        )
    return pages


def progress_payload(
    theory_ids: list[str],
    started_ids: set[str],
    submitted_at: str | None,
) -> dict[str, Any]:
    completed = sum(1 for tid in theory_ids if tid in started_ids)
    total = len(theory_ids)
    return {
        "completed": completed,
        "total": total,
        "ready": completed == total and total > 0,
        "submitted": submitted_at is not None,
        "submitted_at": submitted_at,
        "done": submitted_at is not None,
    }


def short_topic(task: str, effect: str | None = None, topic: str | None = None) -> str:
    task_part = "Race" if task == "race" else "Gender" if task == "gender" else (topic or task)
    if effect == "main":
        return f"{task_part} · Main"
    if effect == "soi":
        return f"{task_part} · Interactions"
    return task_part


def theory_display_labels(
    theory_ids: list[str], by_id: dict[str, dict[str, Any]]
) -> dict[str, str]:
    """Map theory_id -> 'Theory N (Race · Main)' with global 1..N numbering."""
    labels: dict[str, str] = {}
    n = 0
    for tid in theory_ids:
        item = by_id.get(tid)
        if not item:
            continue
        n += 1
        labels[tid] = (
            f"Theory {n} "
            f"({short_topic(item.get('task') or '', item.get('effect') or '', item.get('topic'))})"
        )
    return labels


def missing_theory_labels(
    theory_ids: list[str],
    by_id: dict[str, dict[str, Any]],
    started_ids: set[str],
) -> list[str]:
    """Return global theory labels still missing (e.g. 'Theory 3', 'Theory 17')."""
    missing: list[str] = []
    n = 0
    for tid in theory_ids:
        if tid not in by_id:
            continue
        n += 1
        if tid not in started_ids:
            missing.append(f"Theory {n}")
    return missing

def ratings_map(
    conn: sqlite3.Connection, identifier: str, seed: str
) -> dict[str, dict[str, int | None]]:
    rows = conn.execute(
        """
        SELECT theory_id, clarity_coherence, causal_reasoning, theoretical_depth,
               creativity, persuasiveness
        FROM ratings WHERE identifier = ? AND seed = ?
        """,
        (identifier, seed),
    ).fetchall()
    out: dict[str, dict[str, int | None]] = {}
    for row in rows:
        scores = {k: row[k] for k in DIM_KEYS}
        if any(v is not None for v in scores.values()):
            out[row["theory_id"]] = scores
    return out


def started_theory_ids(conn: sqlite3.Connection, identifier: str, seed: str) -> set[str]:
    """Theories the rater has actually dragged (progress counts these only)."""
    rows = conn.execute(
        """
        SELECT theory_id FROM ratings
        WHERE identifier = ? AND seed = ? AND touched = 1
        """,
        (identifier, seed),
    ).fetchall()
    return {r["theory_id"] for r in rows}


def theory_complete(scores: dict[str, int | None] | None) -> bool:
    if not scores:
        return False
    return all(isinstance(scores.get(k), int) for k in DIM_KEYS)


class SessionRequest(BaseModel):
    identifier: str
    seed: str


class RatingPayload(BaseModel):
    identifier: str
    seed: str
    theory_id: str
    scores: dict[str, int] = Field(default_factory=dict)

    @field_validator("scores")
    @classmethod
    def validate_scores(cls, value: dict[str, int]) -> dict[str, int]:
        missing = [k for k in DIM_KEYS if k not in value]
        if missing:
            raise ValueError(f"Missing dimensions: {', '.join(missing)}")
        for key in DIM_KEYS:
            score = value[key]
            if not isinstance(score, int) or score < 1 or score > 10:
                raise ValueError(f"{key} must be an integer from 1 to 10")
        return {k: int(value[k]) for k in DIM_KEYS}


@app.on_event("startup")
def on_startup() -> None:
    init_db()
    load_pool()


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "dimensions": DIMENSIONS,
            "sample_size": SAMPLE_SIZE,
            "sample_per_cell": SAMPLE_PER_CELL,
        },
    )


@app.get("/submitted", response_class=HTMLResponse)
def submitted_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "submitted.html", {})


@app.post("/api/session")
def create_or_resume_session(body: SessionRequest) -> dict[str, Any]:
    identifier = normalize_identifier(body.identifier)
    seed = resolve_seed_for_identifier(identifier)
    by_id = pool_by_id()
    with db() as conn:
        theory_ids = ensure_assignment(conn, identifier, seed)
        rated = ratings_map(conn, identifier, seed)
        started = started_theory_ids(conn, identifier, seed)
        submitted_at = get_submitted_at(conn, identifier, seed)
        conn.commit()

    pages = build_pages(theory_ids, by_id, rated, started)
    return {
        "identifier": identifier,
        "seed": seed,
        "dimensions": DIMENSIONS,
        "pages": pages,
        "sample_per_cell": SAMPLE_PER_CELL,
        "progress": progress_payload(theory_ids, started, submitted_at),
    }


@app.put("/api/ratings")
def save_rating(body: RatingPayload) -> dict[str, Any]:
    identifier = normalize_identifier(body.identifier)
    seed = resolve_seed_for_identifier(identifier)
    by_id = pool_by_id()
    if body.theory_id not in by_id:
        raise HTTPException(status_code=404, detail="Unknown theory.")

    with db() as conn:
        assigned = get_assignment(conn, identifier, seed)
        if body.theory_id not in assigned:
            raise HTTPException(status_code=403, detail="Theory is not in your assignment.")
        now = utc_now()
        conn.execute(
            """
            INSERT OR IGNORE INTO sessions (identifier, seed, created_at, updated_at, submitted_at)
            VALUES (?, ?, ?, ?, NULL)
            """,
            (identifier, seed, now, now),
        )
        upsert_rating(
            conn, identifier, seed, body.theory_id, body.scores, now, touched=True
        )
        prior_submitted_at = get_submitted_at(conn, identifier, seed)
        started = started_theory_ids(conn, identifier, seed)
        # After the first successful submit, every later edit auto-resubmits
        # (keeps submitted_at current). First-time submit still requires /api/submit.
        if prior_submitted_at is not None:
            conn.execute(
                """
                UPDATE sessions
                SET updated_at = ?, submitted_at = ?
                WHERE identifier = ? AND seed = ?
                """,
                (now, now, identifier, seed),
            )
            submitted_at = now
        else:
            conn.execute(
                """
                UPDATE sessions
                SET updated_at = ?
                WHERE identifier = ? AND seed = ?
                """,
                (now, identifier, seed),
            )
            submitted_at = None
        conn.commit()

    return {
        "ok": True,
        "theory_id": body.theory_id,
        "scores": body.scores,
        "progress": progress_payload(assigned, started, submitted_at),
    }


class SubmitRequest(BaseModel):
    identifier: str
    seed: str


@app.post("/api/submit")
def submit_session(body: SubmitRequest) -> dict[str, Any]:
    """Require every theory to be started; re-submit keeps only the latest."""
    identifier = normalize_identifier(body.identifier)
    seed = resolve_seed_for_identifier(identifier)
    by_id = pool_by_id()
    with db() as conn:
        assigned = get_assignment(conn, identifier, seed)
        if not assigned:
            raise HTTPException(status_code=404, detail="No session found.")
        started = started_theory_ids(conn, identifier, seed)
        missing = missing_theory_labels(assigned, by_id, started)
        if missing:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Please rate all theories before submitting.",
                    "missing": missing,
                },
            )
        now = utc_now()
        conn.execute(
            """
            UPDATE sessions
            SET submitted_at = ?, updated_at = ?
            WHERE identifier = ? AND seed = ?
            """,
            (now, now, identifier, seed),
        )
        conn.commit()
        started = started_theory_ids(conn, identifier, seed)
    return {
        "ok": True,
        "progress": progress_payload(assigned, started, now),
    }


@app.get("/api/export/{identifier}/{seed}")
def export_ratings(identifier: str, seed: str, token: str = Query(default="")) -> dict[str, Any]:
    """Server-side export including source labels (researcher only)."""
    require_admin(token)
    ident = normalize_identifier(identifier)
    seed_norm = normalize_seed(seed)
    by_id = pool_by_id()
    with db() as conn:
        theory_ids = get_assignment(conn, ident, seed_norm)
        if not theory_ids:
            raise HTTPException(status_code=404, detail="No session for this identifier + seed.")
        rated = ratings_map(conn, ident, seed_norm)

    rows = []
    for pos, tid in enumerate(theory_ids, start=1):
        item = by_id[tid]
        rows.append(
            {
                "position": pos,
                "theory_id": tid,
                "topic": item["topic"],
                "task": item["task"],
                "effect": item.get("effect"),
                "phase": item.get("phase"),
                "source": item["source"],
                "source_label": SOURCE_LABELS.get(item["source"], item["source"]),
                "group": item["group"],
                "group_label": GROUP_LABELS.get(str(item["group"]), str(item["group"])),
                "participant_name": item["participant_name"],
                "text": item["text"],
                "ratings": rated.get(tid),
            }
        )
    return {"identifier": ident, "seed": seed_norm, "rows": rows}


def require_admin(token: str | None) -> None:
    provided = (token or "").strip()
    expected = ADMIN_TOKEN
    ok = (
        len(provided) == len(expected)
        and secrets.compare_digest(provided, expected)
    )
    if not ok:
        raise HTTPException(status_code=401, detail="Invalid or missing admin token.")


def export_effect_label(effect: str | None) -> str:
    if effect == "soi":
        return "interactions"
    return effect or ""


def build_export_row(
    item: dict[str, Any],
    *,
    rater_identifier: str,
    scores: dict[str, Any],
) -> dict[str, Any]:
    return {
        "participant_name": item["participant_name"],
        "task": item["task"],
        "effect": export_effect_label(item.get("effect")),
        "phase": item.get("phase"),
        "group": item.get("group"),
        "rater_identifier": rater_identifier,
        **{k: scores.get(k) for k in DIM_KEYS},
    }


def enrich_item(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "theory_id": item["id"],
        "topic": item["topic"],
        "task": item["task"],
        "effect": item.get("effect"),
        "effect_label": item.get("effect_label"),
        "phase": item.get("phase"),
        "group_label": GROUP_LABELS.get(str(item["group"]), str(item["group"])),
        "participant_name": item["participant_name"],
        "text": item["text"],
    }


def iter_export_rows(
    conn: sqlite3.Connection,
    seed: str | None = None,
    identifier: str | None = None,
) -> list[dict[str, Any]]:
    by_id = pool_by_id()
    clauses: list[str] = []
    params: list[str] = []
    if identifier:
        clauses.append("identifier = ?")
        params.append(identifier)
    if seed:
        clauses.append("seed = ?")
        params.append(seed)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    sessions = conn.execute(
        f"""
        SELECT identifier, seed, created_at, updated_at
        FROM sessions
        {where}
        ORDER BY updated_at DESC
        """,
        params,
    ).fetchall()

    rows: list[dict[str, Any]] = []
    for sess in sessions:
        ident = sess["identifier"]
        sess_seed = sess["seed"]
        theory_ids = get_assignment(conn, ident, sess_seed)
        rated = ratings_map(conn, ident, sess_seed)
        for tid in theory_ids:
            item = by_id.get(tid)
            if not item:
                continue
            scores = rated.get(tid) or {}
            rows.append(
                build_export_row(
                    item,
                    rater_identifier=ident,
                    scores=scores,
                )
            )
    return rows


@app.get("/admin", response_class=HTMLResponse)
def admin_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "admin.html",
        {"dimensions": DIMENSIONS},
    )


@app.get("/api/admin/sessions")
def admin_sessions(token: str = Query(...)) -> dict[str, Any]:
    require_admin(token)
    with db() as conn:
        sessions = conn.execute(
            """
            SELECT s.identifier, s.seed, s.created_at, s.updated_at, s.submitted_at,
                   COUNT(r.theory_id) AS rated_count
            FROM sessions s
            LEFT JOIN ratings r
              ON r.identifier = s.identifier AND r.seed = s.seed
            GROUP BY s.identifier, s.seed
            ORDER BY s.updated_at DESC
            """
        ).fetchall()
        out = []
        for row in sessions:
            completed = len(started_theory_ids(conn, row["identifier"], row["seed"]))
            assigned = get_assignment(conn, row["identifier"], row["seed"])
            out.append(
                {
                    "identifier": row["identifier"],
                    "seed": row["seed"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                    "submitted_at": row["submitted_at"],
                    "completed": completed,
                    "total": len(assigned),
                    "done": row["submitted_at"] is not None,
                    "ready": len(assigned) > 0,
                }
            )
    return {"sessions": out, "sample_size": SAMPLE_SIZE}


EXPORT_FIELDNAMES = [
    "participant_name",
    "task",
    "effect",
    "phase",
    "group",
    "rater_identifier",
    *DIM_KEYS,
]


def rows_to_csv(rows: list[dict[str, Any]]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=EXPORT_FIELDNAMES, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return buf.getvalue()


@app.get("/api/admin/export.csv")
def admin_export_csv(
    token: str = Query(...),
    seed: str = Query(default=""),
    identifier: str = Query(default=""),
) -> StreamingResponse:
    """Researcher export with source + author name (not shown to raters).

    Optional filters: seed and/or identifier (one submission).
    """
    require_admin(token)
    seed_filter = seed.strip() or None
    ident_filter = identifier.strip() or None
    with db() as conn:
        rows = iter_export_rows(conn, seed=seed_filter, identifier=ident_filter)

    parts = ["theory_ratings"]
    if ident_filter:
        safe = re.sub(r"[^\w.\-]+", "_", ident_filter)[:40]
        parts.append(safe)
    if seed_filter:
        safe_seed = re.sub(r"[^\w.\-]+", "_", seed_filter)[:40]
        parts.append(f"seed{safe_seed}")
    filename = "_".join(parts) + ".csv"

    return StreamingResponse(
        iter([rows_to_csv(rows)]),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/admin/export.json")
def admin_export_json(
    token: str = Query(...),
    seed: str = Query(default=""),
    identifier: str = Query(default=""),
) -> dict[str, Any]:
    require_admin(token)
    seed_filter = seed.strip() or None
    ident_filter = identifier.strip() or None
    with db() as conn:
        rows = iter_export_rows(conn, seed=seed_filter, identifier=ident_filter)
    return {
        "count": len(rows),
        "seed": seed_filter,
        "identifier": ident_filter,
        "rows": rows,
    }


@app.delete("/api/admin/session")
def admin_clear_session(
    token: str = Query(...),
    identifier: str = Query(...),
    seed: str = Query(...),
) -> dict[str, Any]:
    """Delete one rater session (ratings + assignment + session row)."""
    require_admin(token)
    ident = normalize_identifier(identifier)
    seed_norm = normalize_seed(seed)
    with db() as conn:
        exists = conn.execute(
            "SELECT 1 FROM sessions WHERE identifier = ? AND seed = ?",
            (ident, seed_norm),
        ).fetchone()
        if not exists:
            raise HTTPException(status_code=404, detail="Session not found.")
        n_ratings = conn.execute(
            "SELECT COUNT(*) AS n FROM ratings WHERE identifier = ? AND seed = ?",
            (ident, seed_norm),
        ).fetchone()["n"]
        n_assignments = conn.execute(
            "SELECT COUNT(*) AS n FROM assignments WHERE identifier = ? AND seed = ?",
            (ident, seed_norm),
        ).fetchone()["n"]
        conn.execute(
            "DELETE FROM ratings WHERE identifier = ? AND seed = ?",
            (ident, seed_norm),
        )
        conn.execute(
            "DELETE FROM assignments WHERE identifier = ? AND seed = ?",
            (ident, seed_norm),
        )
        conn.execute(
            "DELETE FROM sessions WHERE identifier = ? AND seed = ?",
            (ident, seed_norm),
        )
        conn.commit()
    return {
        "ok": True,
        "identifier": ident,
        "seed": seed_norm,
        "deleted": {
            "sessions": 1,
            "assignments": n_assignments,
            "ratings": n_ratings,
        },
    }


@app.delete("/api/admin/clear")
def admin_clear_all(token: str = Query(...)) -> dict[str, Any]:
    """Wipe all sessions, assignments, and ratings (researcher only)."""
    require_admin(token)
    with db() as conn:
        n_ratings = conn.execute("SELECT COUNT(*) AS n FROM ratings").fetchone()["n"]
        n_assignments = conn.execute("SELECT COUNT(*) AS n FROM assignments").fetchone()["n"]
        n_sessions = conn.execute("SELECT COUNT(*) AS n FROM sessions").fetchone()["n"]
        conn.execute("DELETE FROM ratings")
        conn.execute("DELETE FROM assignments")
        conn.execute("DELETE FROM sessions")
        conn.commit()
    return {
        "ok": True,
        "deleted": {
            "sessions": n_sessions,
            "assignments": n_assignments,
            "ratings": n_ratings,
        },
    }


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8765"))
    uvicorn.run("app:app", host=host, port=port, reload=False)
